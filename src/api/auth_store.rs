use crate::sqlite::SqliteService;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiKeyRecord {
    pub id: String,
    pub name: String,
    pub role: String, // "admin" or "user"
    pub permissions: serde_json::Value,
    pub created_at_ms: u64,
}

#[derive(Clone)]
pub struct AuthStore {
    sqlite: Arc<SqliteService>,
}

impl AuthStore {
    pub fn new(sqlite: Arc<SqliteService>) -> Self {
        Self { sqlite }
    }

    pub async fn init(&self) -> anyhow::Result<()> {
        self.sqlite.execute(
            "CREATE TABLE IF NOT EXISTS sys_api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                permissions TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            )".to_string(),
            vec![],
        ).await?;
        
        self.bootstrap().await?;
        Ok(())
    }

    async fn bootstrap(&self) -> anyhow::Result<()> {
        let count_res = self.sqlite.query(
            "SELECT COUNT(*) as count FROM sys_api_keys".to_string(),
            vec![]
        ).await?;
        
        let count = count_res.first()
            .and_then(|row| row.get("count"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        if count == 0 {
            let key = self.generate_api_key();
            tracing::warn!("🔑 INITIAL SETUP: Creating Admin API Key...");
            tracing::warn!("🔑 KEY: {}", key);
            tracing::warn!("⚠️  Save this key! It will not be shown again.");
            
            self.create_key("Admin", "admin", &key, serde_json::json!({"allow": "*"})).await?;
        }
        Ok(())
    }

    pub async fn create_key(&self, name: &str, role: &str, plain_key: &str, permissions: serde_json::Value) -> anyhow::Result<String> {
        let id = Uuid::new_v4().to_string();
        let hash = self.hash_key(plain_key);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;
        
        self.sqlite.execute(
            "INSERT INTO sys_api_keys (id, key_hash, name, role, permissions, created_at_ms) VALUES (?, ?, ?, ?, ?, ?)".to_string(),
            vec![
                serde_json::json!(id),
                serde_json::json!(hash),
                serde_json::json!(name),
                serde_json::json!(role),
                serde_json::Value::String(permissions.to_string()),
                serde_json::json!(now),
            ]
        ).await?;
        
        Ok(id)
    }

    pub async fn validate_key(&self, plain_key: &str) -> anyhow::Result<Option<ApiKeyRecord>> {
        let hash = self.hash_key(plain_key);
        let rows = self.sqlite.query(
            "SELECT * FROM sys_api_keys WHERE key_hash = ?".to_string(),
            vec![serde_json::json!(hash)]
        ).await?;
        
        if let Some(mut row) = rows.into_iter().next() {
            if let Some(obj) = row.as_object_mut() {
                if let Some(perm_str) = obj.get("permissions").and_then(|v| v.as_str()) {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(perm_str) {
                        obj.insert("permissions".to_string(), parsed);
                    }
                }
            }
            let rec: ApiKeyRecord = serde_json::from_value(row)?;
            return Ok(Some(rec));
        }
        Ok(None)
    }

    pub async fn list_keys(&self) -> anyhow::Result<Vec<ApiKeyRecord>> {
        let rows = self.sqlite.query(
            "SELECT * FROM sys_api_keys ORDER BY created_at_ms DESC".to_string(),
            vec![]
        ).await?;
        
        let mut keys = Vec::new();
        for mut row in rows {
            if let Some(obj) = row.as_object_mut() {
                // Redact sensitive hash
                obj.remove("key_hash");
                // Parse permissions
                if let Some(perm_str) = obj.get("permissions").and_then(|v| v.as_str()) {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(perm_str) {
                        obj.insert("permissions".to_string(), parsed);
                    }
                }
            }
            if let Ok(rec) = serde_json::from_value::<ApiKeyRecord>(row) {
                keys.push(rec);
            }
        }
        Ok(keys)
    }

    pub async fn revoke_key(&self, id: &str) -> anyhow::Result<bool> {
        let affected = self.sqlite.execute(
            "DELETE FROM sys_api_keys WHERE id = ?".to_string(),
            vec![serde_json::json!(id)]
        ).await?;
        Ok(affected > 0)
    }

    pub async fn ensure_bootstrap_key(&self, plain_key: &str) -> anyhow::Result<()> {
        let hash = self.hash_key(plain_key);
        let rows = self.sqlite.query(
            "SELECT id FROM sys_api_keys WHERE key_hash = ?".to_string(),
            vec![serde_json::json!(hash)]
        ).await?;

        if rows.is_empty() {
            tracing::info!("🔑 Ensuring bootstrap API Key exists: '{}'", plain_key);
            let id = Uuid::new_v4().to_string();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64;
            
            // Default permissions for admin
            let permissions = serde_json::json!({"allow": "*"});

            self.sqlite.execute(
                "INSERT INTO sys_api_keys (id, key_hash, name, role, permissions, created_at_ms) VALUES (?, ?, ?, ?, ?, ?)".to_string(),
                vec![
                    serde_json::json!(id),
                    serde_json::json!(hash),
                    serde_json::json!("Bootstrap/Dev"),
                    serde_json::json!("admin"),
                    serde_json::Value::String(permissions.to_string()),
                    serde_json::json!(now),
                ]
            ).await?;
        }
        Ok(())
    }

    pub fn generate_api_key(&self) -> String {
        use rand::Rng;
        let random_bytes: [u8; 32] = rand::thread_rng().gen();
        let suffix = URL_SAFE_NO_PAD.encode(random_bytes);
        format!("rk_live_{}", suffix)
    }

    fn hash_key(&self, key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        hex::encode(hasher.finalize())
    }
}
