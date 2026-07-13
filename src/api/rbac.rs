use crate::api::errors::ApiError;
use crate::api::TenantContext;
use crate::sqlite::SqliteService;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RoleRecord {
    pub id: String,
    pub name: String,
    pub parent_role_id: Option<String>,
    pub description: String,
    pub is_system: i64, // 0 = custom, 1 = built-in (cannot be deleted)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PermissionRecord {
    pub id: i64,
    pub role_id: String,
    pub resource: String,
    pub action: String,
}

/// Numeric privilege level for a known role name.
///
/// Both the original tiers (`readonly`/`user`/`admin`) and the enterprise
/// account roles (`viewer`/`member`/`admin`/`owner`) are recognized and mapped
/// onto a common ladder so a single `require_role` gate works for either.
/// Custom roles return `None` (treated as sufficient — use `RbacService::can`
/// for fine-grained checks on those).
fn role_level(role: &str) -> Option<u32> {
    match role {
        "viewer" | "readonly" => Some(10),
        "member" | "user" => Some(20),
        "admin" => Some(30),
        "owner" => Some(40),
        _ => None,
    }
}

/// Synchronous role-level gate used by routes.
/// Returns 403 if the caller's role is below `min_role`.
/// For custom roles (not in the ladder) this always passes — use
/// `RbacService::can` instead.
pub fn require_role(ctx: &TenantContext, min_role: &str) -> Result<(), ApiError> {
    let caller_level = role_level(&ctx.role).unwrap_or(u32::MAX); // custom role: assume sufficient
    let min_level = role_level(min_role).unwrap_or(0);
    if caller_level >= min_level {
        Ok(())
    } else {
        Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            format!("{min_role} role required"),
        ))
    }
}

#[derive(Clone)]
pub struct RbacService {
    sqlite: Arc<SqliteService>,
}

impl RbacService {
    pub fn new(sqlite: Arc<SqliteService>) -> Self {
        Self { sqlite }
    }

    pub async fn init(&self) -> anyhow::Result<()> {
        self.sqlite
            .execute(
                "CREATE TABLE IF NOT EXISTS sys_roles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    parent_role_id TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    is_system INTEGER NOT NULL DEFAULT 0
                )"
                .to_string(),
                vec![],
            )
            .await?;

        self.sqlite
            .execute(
                "CREATE TABLE IF NOT EXISTS sys_role_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role_id TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    UNIQUE(role_id, resource, action)
                )"
                .to_string(),
                vec![],
            )
            .await?;

        self.seed_defaults().await?;
        Ok(())
    }

    async fn seed_defaults(&self) -> anyhow::Result<()> {
        // Insert system roles (idempotent via INSERT OR IGNORE)
        let roles: &[(&str, Option<&str>, &str)] = &[
            ("readonly", None, "Read-only access to all data endpoints"),
            (
                "user",
                Some("readonly"),
                "Read-write access; inherits readonly",
            ),
            (
                "admin",
                Some("user"),
                "Full administrative access; inherits user",
            ),
            // Enterprise account roles (aliases layered on the same ladder).
            ("viewer", Some("readonly"), "Enterprise: read-only member"),
            ("member", Some("user"), "Enterprise: read-write member"),
            (
                "owner",
                Some("admin"),
                "Enterprise: organization owner; full control",
            ),
        ];
        for (name, parent, desc) in roles {
            self.sqlite.execute(
                "INSERT OR IGNORE INTO sys_roles (id, name, parent_role_id, description, is_system) VALUES (?, ?, ?, ?, 1)".to_string(),
                vec![
                    serde_json::json!(name),
                    serde_json::json!(name),
                    parent.map(|p| serde_json::Value::String(p.to_string())).unwrap_or(serde_json::Value::Null),
                    serde_json::json!(desc),
                ],
            ).await?;
        }

        // readonly: read on all data resources
        for resource in &["vector", "state", "doc", "db", "memory", "events", "search"] {
            self.upsert_perm("readonly", resource, "read").await?;
        }

        // user: write + delete on all data resources (read inherited from readonly)
        for resource in &["vector", "state", "doc", "db", "memory", "events", "search"] {
            self.upsert_perm("user", resource, "write").await?;
            self.upsert_perm("user", resource, "delete").await?;
        }

        // admin: full wildcard (read/write/delete inherited through parent chain)
        self.upsert_perm("admin", "auth", "admin").await?;
        self.upsert_perm("admin", "config", "admin").await?;
        self.upsert_perm("admin", "admin", "admin").await?;
        self.upsert_perm("admin", "*", "*").await?;

        Ok(())
    }

    async fn upsert_perm(&self, role_id: &str, resource: &str, action: &str) -> anyhow::Result<()> {
        self.sqlite
            .execute(
                "INSERT OR IGNORE INTO sys_role_permissions (role_id, resource, action) VALUES (?, ?, ?)".to_string(),
                vec![
                    serde_json::json!(role_id),
                    serde_json::json!(resource),
                    serde_json::json!(action),
                ],
            )
            .await?;
        Ok(())
    }

    /// Check if `role` (or any ancestor) is allowed `action` on `resource`.
    ///
    /// Walks the parent chain and resolves action inheritance:
    /// - `admin` permission implies `write` and `read`
    /// - `write` permission implies `read`
    /// - wildcard resource `*` and wildcard action `*` both match
    pub async fn can(&self, role: &str, resource: &str, action: &str) -> bool {
        let mut current = role.to_string();
        let mut visited = HashSet::new();
        loop {
            if !visited.insert(current.clone()) {
                break; // cycle guard
            }
            if self.role_allows(&current, resource, action).await {
                return true;
            }
            match self.parent_of(&current).await {
                Some(p) => current = p,
                None => break,
            }
        }
        false
    }

    async fn role_allows(&self, role_id: &str, resource: &str, action: &str) -> bool {
        // Build the set of (resource, action) combinations that satisfy the request,
        // including wildcard and action-hierarchy expansions.
        let mut pairs: Vec<(&str, &str)> = vec![
            (resource, action),
            ("*", action),
            (resource, "*"),
            ("*", "*"),
        ];
        // Action hierarchy: higher privilege implies lower
        let implied: &[&str] = match action {
            "read" => &["write", "admin"],
            "write" => &["admin"],
            _ => &[],
        };
        for &higher in implied {
            pairs.push((resource, higher));
            pairs.push(("*", higher));
        }

        for (r, a) in pairs {
            if let Ok(rows) = self
                .sqlite
                .query(
                    "SELECT 1 FROM sys_role_permissions WHERE role_id = ? AND resource = ? AND action = ? LIMIT 1"
                        .to_string(),
                    vec![
                        serde_json::json!(role_id),
                        serde_json::json!(r),
                        serde_json::json!(a),
                    ],
                )
                .await
            {
                if !rows.is_empty() {
                    return true;
                }
            }
        }
        false
    }

    async fn parent_of(&self, role_id: &str) -> Option<String> {
        let rows = self
            .sqlite
            .query(
                "SELECT parent_role_id FROM sys_roles WHERE id = ?".to_string(),
                vec![serde_json::json!(role_id)],
            )
            .await
            .ok()?;
        rows.into_iter()
            .next()?
            .get("parent_role_id")?
            .as_str()
            .map(|s| s.to_string())
    }

    // ---- CRUD ----

    pub async fn list_roles(&self) -> anyhow::Result<Vec<RoleRecord>> {
        let rows = self
            .sqlite
            .query(
                "SELECT id, name, parent_role_id, description, is_system FROM sys_roles ORDER BY name"
                    .to_string(),
                vec![],
            )
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|r| serde_json::from_value(r).ok())
            .collect())
    }

    pub async fn create_role(
        &self,
        name: &str,
        parent_role_id: Option<&str>,
        description: &str,
    ) -> anyhow::Result<String> {
        let id = Uuid::new_v4().to_string();
        self.sqlite
            .execute(
                "INSERT INTO sys_roles (id, name, parent_role_id, description, is_system) VALUES (?, ?, ?, ?, 0)"
                    .to_string(),
                vec![
                    serde_json::json!(id),
                    serde_json::json!(name),
                    parent_role_id
                        .map(|p| serde_json::Value::String(p.to_string()))
                        .unwrap_or(serde_json::Value::Null),
                    serde_json::json!(description),
                ],
            )
            .await?;
        Ok(id)
    }

    pub async fn delete_role(&self, id: &str) -> anyhow::Result<bool> {
        let rows = self
            .sqlite
            .query(
                "SELECT is_system FROM sys_roles WHERE id = ?".to_string(),
                vec![serde_json::json!(id)],
            )
            .await?;
        if let Some(row) = rows.first() {
            let is_sys = row.get("is_system").and_then(|v| v.as_i64()).unwrap_or(0);
            if is_sys == 1 {
                anyhow::bail!("cannot delete a system role");
            }
        } else {
            return Ok(false);
        }
        let n = self
            .sqlite
            .execute(
                "DELETE FROM sys_roles WHERE id = ?".to_string(),
                vec![serde_json::json!(id)],
            )
            .await?;
        let _ = self
            .sqlite
            .execute(
                "DELETE FROM sys_role_permissions WHERE role_id = ?".to_string(),
                vec![serde_json::json!(id)],
            )
            .await;
        Ok(n > 0)
    }

    pub async fn list_permissions(&self, role_id: &str) -> anyhow::Result<Vec<PermissionRecord>> {
        let rows = self
            .sqlite
            .query(
                "SELECT id, role_id, resource, action FROM sys_role_permissions WHERE role_id = ? ORDER BY resource, action"
                    .to_string(),
                vec![serde_json::json!(role_id)],
            )
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|r| serde_json::from_value(r).ok())
            .collect())
    }

    pub async fn add_permission(
        &self,
        role_id: &str,
        resource: &str,
        action: &str,
    ) -> anyhow::Result<()> {
        // Verify role exists
        let rows = self
            .sqlite
            .query(
                "SELECT 1 FROM sys_roles WHERE id = ?".to_string(),
                vec![serde_json::json!(role_id)],
            )
            .await?;
        if rows.is_empty() {
            anyhow::bail!("role '{role_id}' not found");
        }
        self.upsert_perm(role_id, resource, action).await
    }

    pub async fn remove_permission(
        &self,
        role_id: &str,
        resource: &str,
        action: &str,
    ) -> anyhow::Result<bool> {
        let n = self
            .sqlite
            .execute(
                "DELETE FROM sys_role_permissions WHERE role_id = ? AND resource = ? AND action = ?"
                    .to_string(),
                vec![
                    serde_json::json!(role_id),
                    serde_json::json!(resource),
                    serde_json::json!(action),
                ],
            )
            .await?;
        Ok(n > 0)
    }
}
