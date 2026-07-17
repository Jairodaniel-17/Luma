//! Enterprise account layer: organizations, users, password login, opaque
//! session tokens, per-organization data isolation, and a semantic audit trail.
//!
//! This sits alongside the existing API-key [`AuthStore`](crate::api::auth_store)
//! and RBAC service. Passwords are hashed with Argon2id
//! ([`crate::crypto`]); session tokens are random opaque strings whose SHA-256
//! hash is persisted (the plaintext is only ever returned once at login).

use crate::crypto;
use crate::sqlite::SqliteService;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::sync::OnceCell;
use uuid::Uuid;

/// Session lifetime in milliseconds (7 days).
const SESSION_TTL_MS: u64 = 7 * 24 * 60 * 60 * 1000;

/// Roles recognized by the enterprise account layer, most privileged first.
pub const ENTERPRISE_ROLES: &[&str] = &["owner", "admin", "member", "viewer"];

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OrgRecord {
    pub id: String,
    pub name: String,
    pub created_at_ms: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UserRecord {
    pub id: String,
    pub org_id: String,
    pub email: String,
    pub role: String,
    pub status: String,
    pub created_at_ms: i64,
}

/// Self-registration allowlist. An empty policy (no domains and no emails)
/// means registration is open to anyone.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessPolicy {
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub emails: Vec<String>,
}

impl AccessPolicy {
    /// Whether `email` may register. Empty policy = open. Otherwise the address
    /// must exactly match an allowed email or fall under an allowed domain
    /// (case-insensitive).
    pub fn permits(&self, email: &str) -> bool {
        if self.domains.is_empty() && self.emails.is_empty() {
            return true;
        }
        let email = email.trim().to_ascii_lowercase();
        if self.emails.iter().any(|e| e.eq_ignore_ascii_case(&email)) {
            return true;
        }
        if !email.contains('@') {
            return false;
        }
        let domain = email.rsplit('@').next().unwrap_or_default();
        self.domains
            .iter()
            .any(|d| d.trim_start_matches('@').eq_ignore_ascii_case(domain))
    }
}

/// Normalize an allowlist: trim, strip a leading `@`, lowercase, drop empties,
/// dedupe (order-preserving).
fn normalize_access_list(items: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    items
        .iter()
        .map(|s| s.trim().trim_start_matches('@').to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .filter(|s| seen.insert(s.clone()))
        .collect()
}

/// Resolved identity behind a valid session token.
#[derive(Debug, Clone)]
pub struct SessionIdentity {
    pub user_id: String,
    pub org_id: String,
    pub role: String,
}

#[derive(Clone)]
pub struct AccountsService {
    sqlite: Arc<SqliteService>,
    init: Arc<OnceCell<()>>,
    /// Cache of collection-name → owning org_id. Ownership is first-touch and
    /// never changes (nothing deletes sys_collections), so a value read from
    /// SQLite is valid forever — this removes a per-request SQLite query from
    /// the tenant-isolation middleware on the hot path. Only positive results
    /// are cached, and only from an authoritative SQLite read.
    owner_cache: Arc<dashmap::DashMap<String, String>>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn hash_token(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    hex::encode(h.finalize())
}

impl AccountsService {
    pub fn new(sqlite: Arc<SqliteService>) -> Self {
        Self {
            sqlite,
            init: Arc::new(OnceCell::new()),
            owner_cache: Arc::new(dashmap::DashMap::new()),
        }
    }

    /// Create tables on first use (idempotent). Safe to call from every method.
    pub async fn ensure_init(&self) -> anyhow::Result<()> {
        self.init
            .get_or_try_init(|| async {
                self.sqlite
                    .execute(
                        "CREATE TABLE IF NOT EXISTS sys_orgs (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL
                        )"
                        .to_string(),
                        vec![],
                    )
                    .await?;
                self.sqlite
                    .execute(
                        "CREATE TABLE IF NOT EXISTS sys_users (
                            id TEXT PRIMARY KEY,
                            org_id TEXT NOT NULL,
                            email TEXT NOT NULL UNIQUE,
                            password_hash TEXT NOT NULL,
                            role TEXT NOT NULL,
                            status TEXT NOT NULL DEFAULT 'active',
                            created_at_ms INTEGER NOT NULL
                        )"
                        .to_string(),
                        vec![],
                    )
                    .await?;
                self.sqlite
                    .execute(
                        "CREATE TABLE IF NOT EXISTS sys_sessions (
                            token_hash TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            org_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL,
                            expires_at_ms INTEGER NOT NULL
                        )"
                        .to_string(),
                        vec![],
                    )
                    .await?;
                self.sqlite
                    .execute(
                        "CREATE TABLE IF NOT EXISTS sys_collections (
                            name TEXT PRIMARY KEY,
                            org_id TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL
                        )"
                        .to_string(),
                        vec![],
                    )
                    .await?;
                // Many-to-many user↔org membership with a per-membership role.
                // `sys_users.org_id`/`role` remain the user's home org (used by
                // the bootstrap-admin check and the default login target); this
                // table is the source of truth for which orgs a user can access.
                self.sqlite
                    .execute(
                        "CREATE TABLE IF NOT EXISTS sys_memberships (
                            user_id TEXT NOT NULL,
                            org_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            created_at_ms INTEGER NOT NULL,
                            PRIMARY KEY (user_id, org_id)
                        )"
                        .to_string(),
                        vec![],
                    )
                    .await?;
                // Backfill existing single-org users as members of their home org
                // (idempotent) so the M:N table is consistent with the legacy
                // column from the first boot after upgrade.
                self.sqlite
                    .execute(
                        "INSERT OR IGNORE INTO sys_memberships (user_id, org_id, role, created_at_ms)
                         SELECT id, org_id, role, created_at_ms FROM sys_users"
                            .to_string(),
                        vec![],
                    )
                    .await?;
                self.sqlite
                    .execute(
                        "CREATE TABLE IF NOT EXISTS sys_audit_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            ts_ms INTEGER NOT NULL,
                            org_id TEXT,
                            user_id TEXT,
                            action TEXT NOT NULL,
                            resource TEXT,
                            ip TEXT,
                            user_agent TEXT,
                            detail TEXT
                        )"
                        .to_string(),
                        vec![],
                    )
                    .await?;
                self.sqlite
                    .execute(
                        "CREATE INDEX IF NOT EXISTS idx_audit_events_ts ON sys_audit_events(ts_ms)"
                            .to_string(),
                        vec![],
                    )
                    .await?;
                // Access policy: an optional allowlist of email domains / exact
                // addresses that may self-register. Empty = open registration.
                self.sqlite
                    .execute(
                        "CREATE TABLE IF NOT EXISTS sys_access_policy (
                            id INTEGER PRIMARY KEY CHECK (id = 1),
                            domains TEXT NOT NULL DEFAULT '[]',
                            emails TEXT NOT NULL DEFAULT '[]'
                        )"
                        .to_string(),
                        vec![],
                    )
                    .await?;
                self.sqlite
                    .execute(
                        "INSERT OR IGNORE INTO sys_access_policy (id, domains, emails) VALUES (1, '[]', '[]')"
                            .to_string(),
                        vec![],
                    )
                    .await?;
                Ok::<(), anyhow::Error>(())
            })
            .await?;
        Ok(())
    }

    fn gen_session_token() -> String {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut bytes);
        format!("lums_{}", URL_SAFE_NO_PAD.encode(bytes))
    }

    /// Whether a bearer token looks like a session token (vs. an API key).
    pub fn is_session_token(token: &str) -> bool {
        token.starts_with("lums_")
    }

    // ---- Organizations ----

    pub async fn create_org(&self, name: &str) -> anyhow::Result<OrgRecord> {
        self.ensure_init().await?;
        let id = Uuid::new_v4().to_string();
        let ts = now_ms();
        self.sqlite
            .execute(
                "INSERT INTO sys_orgs (id, name, created_at_ms) VALUES (?, ?, ?)".to_string(),
                vec![json!(id), json!(name), json!(ts)],
            )
            .await?;
        Ok(OrgRecord {
            id,
            name: name.to_string(),
            created_at_ms: ts,
        })
    }

    pub async fn list_orgs(&self) -> anyhow::Result<Vec<OrgRecord>> {
        self.ensure_init().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT id, name, created_at_ms FROM sys_orgs ORDER BY created_at_ms DESC"
                    .to_string(),
                vec![],
            )
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|r| serde_json::from_value(r).ok())
            .collect())
    }

    pub async fn delete_org(&self, id: &str) -> anyhow::Result<bool> {
        self.ensure_init().await?;
        let n = self
            .sqlite
            .execute(
                "DELETE FROM sys_orgs WHERE id = ?".to_string(),
                vec![json!(id)],
            )
            .await?;
        let _ = self
            .sqlite
            .execute(
                "DELETE FROM sys_users WHERE org_id = ?".to_string(),
                vec![json!(id)],
            )
            .await;
        let _ = self
            .sqlite
            .execute(
                "DELETE FROM sys_memberships WHERE org_id = ?".to_string(),
                vec![json!(id)],
            )
            .await;
        Ok(n > 0)
    }

    // ---- Users ----

    /// Register a brand-new organization with its first (owner) user.
    pub async fn register(
        &self,
        org_name: &str,
        email: &str,
        password: &str,
    ) -> anyhow::Result<(OrgRecord, UserRecord)> {
        self.ensure_init().await?;
        let org = self.create_org(org_name).await?;
        let user = self.create_user(&org.id, email, password, "owner").await?;
        Ok((org, user))
    }

    // ---- Access policy (self-registration allowlist) ----

    /// Read the current access policy (allowed email domains / exact addresses).
    pub async fn get_access_policy(&self) -> anyhow::Result<AccessPolicy> {
        self.ensure_init().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT domains, emails FROM sys_access_policy WHERE id = 1".to_string(),
                vec![],
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(AccessPolicy::default());
        };
        let parse = |field: &str| {
            row.get(field)
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                .unwrap_or_default()
        };
        Ok(AccessPolicy {
            domains: parse("domains"),
            emails: parse("emails"),
        })
    }

    /// Replace the access policy. Entries are normalized (trimmed, lowercased,
    /// deduped; leading `@` stripped from domains).
    pub async fn set_access_policy(&self, policy: &AccessPolicy) -> anyhow::Result<AccessPolicy> {
        self.ensure_init().await?;
        let normalized = AccessPolicy {
            domains: normalize_access_list(&policy.domains),
            emails: normalize_access_list(&policy.emails),
        };
        self.sqlite
            .execute(
                "UPDATE sys_access_policy SET domains = ?, emails = ? WHERE id = 1".to_string(),
                vec![
                    json!(serde_json::to_string(&normalized.domains)?),
                    json!(serde_json::to_string(&normalized.emails)?),
                ],
            )
            .await?;
        Ok(normalized)
    }

    /// Whether `email` is permitted to self-register under the current policy.
    pub async fn is_email_allowed(&self, email: &str) -> anyhow::Result<bool> {
        Ok(self.get_access_policy().await?.permits(email))
    }

    pub async fn create_user(
        &self,
        org_id: &str,
        email: &str,
        password: &str,
        role: &str,
    ) -> anyhow::Result<UserRecord> {
        self.ensure_init().await?;
        if !ENTERPRISE_ROLES.contains(&role) {
            anyhow::bail!("invalid role '{role}'");
        }
        let id = Uuid::new_v4().to_string();
        let ts = now_ms();
        let phc = crypto::hash_password(password)?;
        self.sqlite
            .execute(
                "INSERT INTO sys_users (id, org_id, email, password_hash, role, status, created_at_ms)
                 VALUES (?, ?, ?, ?, ?, 'active', ?)"
                    .to_string(),
                vec![
                    json!(id),
                    json!(org_id),
                    json!(email),
                    json!(phc),
                    json!(role),
                    json!(ts),
                ],
            )
            .await?;
        // Mirror the home org into the membership table.
        self.sqlite
            .execute(
                "INSERT OR IGNORE INTO sys_memberships (user_id, org_id, role, created_at_ms)
                 VALUES (?, ?, ?, ?)"
                    .to_string(),
                vec![json!(id), json!(org_id), json!(role), json!(ts)],
            )
            .await?;
        Ok(UserRecord {
            id,
            org_id: org_id.to_string(),
            email: email.to_string(),
            role: role.to_string(),
            status: "active".to_string(),
            created_at_ms: ts,
        })
    }

    pub async fn list_users(&self, org_id: Option<&str>) -> anyhow::Result<Vec<UserRecord>> {
        self.ensure_init().await?;
        let (sql, params) = match org_id {
            Some(org) => (
                "SELECT id, org_id, email, role, status, created_at_ms FROM sys_users WHERE org_id = ? ORDER BY created_at_ms DESC".to_string(),
                vec![json!(org)],
            ),
            None => (
                "SELECT id, org_id, email, role, status, created_at_ms FROM sys_users ORDER BY created_at_ms DESC".to_string(),
                vec![],
            ),
        };
        let rows = self.sqlite.query(sql, params).await?;
        Ok(rows
            .into_iter()
            .filter_map(|r| serde_json::from_value(r).ok())
            .collect())
    }

    /// Update a user's role. When `org_scope` is `Some`, the update is confined
    /// to that org (`AND org_id = ?`) so a tenant admin can never mutate a user
    /// in another org; `None` (platform admin) targets any user by id.
    pub async fn update_user_role(
        &self,
        id: &str,
        org_scope: Option<&str>,
        role: &str,
    ) -> anyhow::Result<bool> {
        self.ensure_init().await?;
        if !ENTERPRISE_ROLES.contains(&role) {
            anyhow::bail!("invalid role '{role}'");
        }
        let (sql, params) = match org_scope {
            Some(org) => (
                "UPDATE sys_users SET role = ? WHERE id = ? AND org_id = ?".to_string(),
                vec![json!(role), json!(id), json!(org)],
            ),
            None => (
                "UPDATE sys_users SET role = ? WHERE id = ?".to_string(),
                vec![json!(role), json!(id)],
            ),
        };
        let n = self.sqlite.execute(sql, params).await?;
        Ok(n > 0)
    }

    /// Delete a user. When `org_scope` is `Some`, the delete is confined to that
    /// org (`AND org_id = ?`); `None` (platform admin) targets any user by id.
    pub async fn delete_user(&self, id: &str, org_scope: Option<&str>) -> anyhow::Result<bool> {
        self.ensure_init().await?;
        let (sql, params) = match org_scope {
            Some(org) => (
                "DELETE FROM sys_users WHERE id = ? AND org_id = ?".to_string(),
                vec![json!(id), json!(org)],
            ),
            None => (
                "DELETE FROM sys_users WHERE id = ?".to_string(),
                vec![json!(id)],
            ),
        };
        let n = self.sqlite.execute(sql, params).await?;
        if n > 0 {
            // Drop every membership for a fully-deleted user. When the delete was
            // org-scoped and matched, also drop only that org's membership; a
            // platform-wide delete removes them all.
            let (msql, mparams) = match org_scope {
                Some(org) => (
                    "DELETE FROM sys_memberships WHERE user_id = ? AND org_id = ?".to_string(),
                    vec![json!(id), json!(org)],
                ),
                None => (
                    "DELETE FROM sys_memberships WHERE user_id = ?".to_string(),
                    vec![json!(id)],
                ),
            };
            let _ = self.sqlite.execute(msql, mparams).await;
        }
        Ok(n > 0)
    }

    // ---- Multi-org membership (user ↔ org, many-to-many) ----

    /// Look up a user by email (case-sensitive match, as stored).
    pub async fn user_by_email(&self, email: &str) -> anyhow::Result<Option<UserRecord>> {
        self.ensure_init().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT id, org_id, email, role, status, created_at_ms FROM sys_users WHERE email = ?"
                    .to_string(),
                vec![json!(email)],
            )
            .await?;
        Ok(rows.into_iter().next().and_then(|r| serde_json::from_value(r).ok()))
    }

    /// Add (or update the role of) a user's membership in an org. Verifies both
    /// the user and the org exist first. Returns an error if either is missing.
    pub async fn add_membership(
        &self,
        user_id: &str,
        org_id: &str,
        role: &str,
    ) -> anyhow::Result<()> {
        self.ensure_init().await?;
        if !ENTERPRISE_ROLES.contains(&role) {
            anyhow::bail!("invalid role '{role}'");
        }
        let u = self
            .sqlite
            .query(
                "SELECT 1 FROM sys_users WHERE id = ?".to_string(),
                vec![json!(user_id)],
            )
            .await?;
        if u.is_empty() {
            anyhow::bail!("user not found");
        }
        let o = self
            .sqlite
            .query(
                "SELECT 1 FROM sys_orgs WHERE id = ?".to_string(),
                vec![json!(org_id)],
            )
            .await?;
        if o.is_empty() {
            anyhow::bail!("org not found");
        }
        self.sqlite
            .execute(
                "INSERT INTO sys_memberships (user_id, org_id, role, created_at_ms)
                 VALUES (?, ?, ?, ?)
                 ON CONFLICT(user_id, org_id) DO UPDATE SET role = excluded.role"
                    .to_string(),
                vec![json!(user_id), json!(org_id), json!(role), json!(now_ms())],
            )
            .await?;
        Ok(())
    }

    /// Change the role of an existing membership. Returns false if there is no
    /// such membership.
    pub async fn set_membership_role(
        &self,
        user_id: &str,
        org_id: &str,
        role: &str,
    ) -> anyhow::Result<bool> {
        self.ensure_init().await?;
        if !ENTERPRISE_ROLES.contains(&role) {
            anyhow::bail!("invalid role '{role}'");
        }
        let n = self
            .sqlite
            .execute(
                "UPDATE sys_memberships SET role = ? WHERE user_id = ? AND org_id = ?".to_string(),
                vec![json!(role), json!(user_id), json!(org_id)],
            )
            .await?;
        Ok(n > 0)
    }

    /// Remove a user from an org. Returns false if there was no membership.
    pub async fn remove_membership(&self, user_id: &str, org_id: &str) -> anyhow::Result<bool> {
        self.ensure_init().await?;
        let n = self
            .sqlite
            .execute(
                "DELETE FROM sys_memberships WHERE user_id = ? AND org_id = ?".to_string(),
                vec![json!(user_id), json!(org_id)],
            )
            .await?;
        Ok(n > 0)
    }

    /// The role a user holds in an org, if they are a member. Used to validate
    /// an org switch: a session may only rebind to an org the user belongs to.
    pub async fn membership_role(
        &self,
        user_id: &str,
        org_id: &str,
    ) -> anyhow::Result<Option<String>> {
        self.ensure_init().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT role FROM sys_memberships WHERE user_id = ? AND org_id = ?".to_string(),
                vec![json!(user_id), json!(org_id)],
            )
            .await?;
        Ok(rows
            .into_iter()
            .next()
            .and_then(|r| r.get("role").and_then(|v| v.as_str()).map(String::from)))
    }

    /// Members of an org (joined with user email), most-recent first.
    pub async fn list_org_members(&self, org_id: &str) -> anyhow::Result<Vec<serde_json::Value>> {
        self.ensure_init().await?;
        self.sqlite
            .query(
                "SELECT m.user_id, u.email, m.role, m.created_at_ms
                 FROM sys_memberships m JOIN sys_users u ON u.id = m.user_id
                 WHERE m.org_id = ? ORDER BY m.created_at_ms DESC"
                    .to_string(),
                vec![json!(org_id)],
            )
            .await
    }

    /// Orgs a user belongs to (joined with org name), most-recent first.
    pub async fn list_user_orgs(&self, user_id: &str) -> anyhow::Result<Vec<serde_json::Value>> {
        self.ensure_init().await?;
        self.sqlite
            .query(
                "SELECT m.org_id, o.name, m.role, m.created_at_ms
                 FROM sys_memberships m JOIN sys_orgs o ON o.id = m.org_id
                 WHERE m.user_id = ? ORDER BY m.created_at_ms DESC"
                    .to_string(),
                vec![json!(user_id)],
            )
            .await
    }

    // ---- Login / sessions ----

    /// Verify email+password. Returns the identity on success.
    pub async fn verify_login(
        &self,
        email: &str,
        password: &str,
    ) -> anyhow::Result<Option<SessionIdentity>> {
        self.ensure_init().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT id, org_id, role, status, password_hash FROM sys_users WHERE email = ?"
                    .to_string(),
                vec![json!(email)],
            )
            .await?;
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        let status = row.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if status != "active" {
            return Ok(None);
        }
        let phc = row
            .get("password_hash")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if !crypto::verify_password(password, phc) {
            return Ok(None);
        }
        Ok(Some(SessionIdentity {
            user_id: row
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            org_id: row
                .get("org_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            role: row
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("viewer")
                .to_string(),
        }))
    }

    /// Issue a new session token for an identity. Returns `(token, expires_at_ms)`.
    pub async fn create_session(&self, id: &SessionIdentity) -> anyhow::Result<(String, i64)> {
        self.ensure_init().await?;
        let token = Self::gen_session_token();
        let ts = now_ms();
        let expires = ts + SESSION_TTL_MS as i64;
        self.sqlite
            .execute(
                "INSERT INTO sys_sessions (token_hash, user_id, org_id, role, created_at_ms, expires_at_ms)
                 VALUES (?, ?, ?, ?, ?, ?)"
                    .to_string(),
                vec![
                    json!(hash_token(&token)),
                    json!(id.user_id),
                    json!(id.org_id),
                    json!(id.role),
                    json!(ts),
                    json!(expires),
                ],
            )
            .await?;
        Ok((token, expires))
    }

    /// Validate a session token, returning the identity if live and unexpired.
    /// True if `user_id` is the earliest-registered user. That first account is
    /// the instance operator and is treated as a platform admin (untenanted) —
    /// the bootstrap super-admin pattern (first user to sign up runs the whole
    /// instance). Later sign-ups are ordinary tenant owners.
    pub async fn is_bootstrap_admin(&self, user_id: &str) -> anyhow::Result<bool> {
        self.ensure_init().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT id FROM sys_users ORDER BY created_at_ms ASC, id ASC LIMIT 1".to_string(),
                vec![],
            )
            .await?;
        Ok(rows
            .first()
            .and_then(|r| r.get("id"))
            .and_then(|v| v.as_str())
            == Some(user_id))
    }

    pub async fn validate_session(&self, token: &str) -> anyhow::Result<Option<SessionIdentity>> {
        self.ensure_init().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT user_id, org_id, role, expires_at_ms FROM sys_sessions WHERE token_hash = ?"
                    .to_string(),
                vec![json!(hash_token(token))],
            )
            .await?;
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        let expires = row
            .get("expires_at_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if expires < now_ms() {
            let _ = self.revoke_session(token).await;
            return Ok(None);
        }
        Ok(Some(SessionIdentity {
            user_id: row
                .get("user_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            org_id: row
                .get("org_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            role: row
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("viewer")
                .to_string(),
        }))
    }

    pub async fn revoke_session(&self, token: &str) -> anyhow::Result<bool> {
        self.ensure_init().await?;
        let n = self
            .sqlite
            .execute(
                "DELETE FROM sys_sessions WHERE token_hash = ?".to_string(),
                vec![json!(hash_token(token))],
            )
            .await?;
        Ok(n > 0)
    }

    // ---- Per-org collection ownership (data isolation) ----

    /// Return the org that owns a collection/namespace, if recorded.
    pub async fn collection_owner(&self, name: &str) -> anyhow::Result<Option<String>> {
        if let Some(owner) = self.owner_cache.get(name) {
            return Ok(Some(owner.clone()));
        }
        self.ensure_init().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT org_id FROM sys_collections WHERE name = ?".to_string(),
                vec![json!(name)],
            )
            .await?;
        let owner = rows
            .into_iter()
            .next()
            .and_then(|r| r.get("org_id").and_then(|v| v.as_str()).map(String::from));
        // Cache only a positive, authoritative result. An unowned name must stay
        // uncached because it is about to be claimed via register_collection.
        if let Some(o) = &owner {
            self.owner_cache.insert(name.to_string(), o.clone());
        }
        Ok(owner)
    }

    /// Record ownership of a collection/namespace by an org (first-touch).
    pub async fn register_collection(&self, name: &str, org_id: &str) -> anyhow::Result<()> {
        self.ensure_init().await?;
        self.sqlite
            .execute(
                "INSERT OR IGNORE INTO sys_collections (name, org_id, created_at_ms) VALUES (?, ?, ?)"
                    .to_string(),
                vec![json!(name), json!(org_id), json!(now_ms())],
            )
            .await?;
        Ok(())
    }

    // ---- Semantic audit trail ----

    #[allow(clippy::too_many_arguments)]
    pub async fn record_event(
        &self,
        org_id: Option<&str>,
        user_id: Option<&str>,
        action: &str,
        resource: Option<&str>,
        ip: Option<&str>,
        user_agent: Option<&str>,
        detail: Option<&str>,
    ) {
        if self.ensure_init().await.is_err() {
            return;
        }
        let _ = self
            .sqlite
            .execute(
                "INSERT INTO sys_audit_events (ts_ms, org_id, user_id, action, resource, ip, user_agent, detail)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                    .to_string(),
                vec![
                    json!(now_ms()),
                    json!(org_id),
                    json!(user_id),
                    json!(action),
                    json!(resource),
                    json!(ip),
                    json!(user_agent),
                    json!(detail),
                ],
            )
            .await;
    }

    pub async fn query_events(
        &self,
        org_id: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        self.ensure_init().await?;
        let (sql, params) = match org_id {
            Some(org) => (
                format!(
                    "SELECT id, ts_ms, org_id, user_id, action, resource, ip, user_agent, detail
                     FROM sys_audit_events WHERE org_id = ? ORDER BY ts_ms DESC LIMIT {limit}"
                ),
                vec![json!(org)],
            ),
            None => (
                format!(
                    "SELECT id, ts_ms, org_id, user_id, action, resource, ip, user_agent, detail
                     FROM sys_audit_events ORDER BY ts_ms DESC LIMIT {limit}"
                ),
                vec![],
            ),
        };
        self.sqlite.query(sql, params).await
    }

    // ---- Dashboard stats ----

    async fn count(&self, sql: &str, params: Vec<serde_json::Value>) -> i64 {
        self.sqlite
            .query(sql.to_string(), params)
            .await
            .ok()
            .and_then(|rows| rows.into_iter().next())
            .and_then(|r| r.get("c").and_then(|v| v.as_i64()))
            .unwrap_or(0)
    }

    /// Usage stats. When `org_id` is `Some`, counts are scoped to that org.
    pub async fn stats(&self, org_id: Option<&str>) -> anyhow::Result<serde_json::Value> {
        self.ensure_init().await?;
        let (users, collections) = match org_id {
            Some(org) => (
                self.count(
                    "SELECT COUNT(*) as c FROM sys_users WHERE org_id = ?",
                    vec![json!(org)],
                )
                .await,
                self.count(
                    "SELECT COUNT(*) as c FROM sys_collections WHERE org_id = ?",
                    vec![json!(org)],
                )
                .await,
            ),
            None => (
                self.count("SELECT COUNT(*) as c FROM sys_users", vec![])
                    .await,
                self.count("SELECT COUNT(*) as c FROM sys_collections", vec![])
                    .await,
            ),
        };
        // Scope the org count too: a tenant-bound caller sees only their own org
        // (0/1), never the cluster-wide total.
        let orgs = match org_id {
            Some(org) => {
                self.count(
                    "SELECT COUNT(*) as c FROM sys_orgs WHERE id = ?",
                    vec![json!(org)],
                )
                .await
            }
            None => {
                self.count("SELECT COUNT(*) as c FROM sys_orgs", vec![])
                    .await
            }
        };
        let audit_events = match org_id {
            Some(org) => {
                self.count(
                    "SELECT COUNT(*) as c FROM sys_audit_events WHERE org_id = ?",
                    vec![json!(org)],
                )
                .await
            }
            None => {
                self.count("SELECT COUNT(*) as c FROM sys_audit_events", vec![])
                    .await
            }
        };
        Ok(json!({
            "orgs": orgs,
            "users": users,
            "collections": collections,
            "audit_events": audit_events,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_access_list, AccessPolicy};

    #[test]
    fn empty_policy_allows_everyone() {
        assert!(AccessPolicy::default().permits("anyone@anywhere.com"));
    }

    #[test]
    fn domain_and_email_allowlist() {
        let policy = AccessPolicy {
            domains: vec!["acme.com".into()],
            emails: vec!["ceo@partner.io".into()],
        };
        assert!(policy.permits("jane@acme.com"));
        assert!(policy.permits("JANE@ACME.COM")); // case-insensitive
        assert!(policy.permits("ceo@partner.io")); // exact email
        assert!(!policy.permits("bob@evil.com")); // wrong domain
        assert!(!policy.permits("intern@partner.io")); // email not exactly allowed
        assert!(!policy.permits("notanemail")); // no domain
    }

    #[test]
    fn normalize_dedupes_and_strips() {
        let out = normalize_access_list(&[
            "  @Acme.com ".into(),
            "acme.com".into(),
            "".into(),
            "Foo.IO".into(),
        ]);
        assert_eq!(out, vec!["acme.com".to_string(), "foo.io".to_string()]);
    }

    #[tokio::test]
    async fn multi_org_membership_roundtrip() {
        use crate::sqlite::SqliteService;
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let sqlite =
            SqliteService::new(format!("{}/t.db", dir.path().display())).unwrap();
        let svc = super::AccountsService::new(Arc::new(sqlite));

        // register creates OrgA + owner, and backfills the owner's membership.
        let (org_a, user) = svc.register("OrgA", "u@a.com", "pw").await.unwrap();
        let org_b = svc.create_org("OrgB").await.unwrap();

        // Home org is a member from the start (via create_user mirror).
        assert_eq!(
            svc.membership_role(&user.id, &org_a.id).await.unwrap().as_deref(),
            Some("owner")
        );

        // Add the same user to OrgB as admin → now a member of two orgs.
        svc.add_membership(&user.id, &org_b.id, "admin").await.unwrap();
        let orgs = svc.list_user_orgs(&user.id).await.unwrap();
        assert_eq!(orgs.len(), 2, "user should belong to two orgs");
        assert_eq!(
            svc.membership_role(&user.id, &org_b.id).await.unwrap().as_deref(),
            Some("admin")
        );

        // Adding to a nonexistent org fails; unknown user fails.
        assert!(svc.add_membership(&user.id, "nope", "member").await.is_err());
        assert!(svc.add_membership("ghost", &org_b.id, "member").await.is_err());

        // Role change + membership listing on OrgB.
        assert!(svc.set_membership_role(&user.id, &org_b.id, "viewer").await.unwrap());
        let members = svc.list_org_members(&org_b.id).await.unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].get("email").and_then(|v| v.as_str()), Some("u@a.com"));

        // Remove from OrgB → gone there, still owner of OrgA.
        assert!(svc.remove_membership(&user.id, &org_b.id).await.unwrap());
        assert!(svc.membership_role(&user.id, &org_b.id).await.unwrap().is_none());
        assert_eq!(svc.list_user_orgs(&user.id).await.unwrap().len(), 1);
    }
}
