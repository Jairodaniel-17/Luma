//! S3 access keys, and why they are not the api keys.
//!
//! SigV4 does not send a credential; it sends a signature derived from one. To
//! check it, the server has to *derive the same key*, which means holding the
//! secret in a recoverable form. Api keys are stored as a SHA-256 hash and are
//! therefore unusable here: there is nothing to derive from.
//!
//! So S3 credentials are their own thing, with the secret encrypted at rest with
//! `LUMA_MASTER_KEY` rather than hashed. That is a real reduction in the
//! guarantee — a hash cannot be reversed by anyone, an encrypted secret can be
//! read by whoever holds the master key — and it is inherent to SigV4 rather
//! than a shortcut. Saying so here is the point: an operator should know that
//! these keys have different blast radius from api keys, and rotate them
//! accordingly.
//!
//! Each credential belongs to one organization, so everything the S3 layer does
//! lands in that org's buckets and under that org's quota.

use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use rand::RngCore;
use std::sync::Arc;

use crate::sqlite::SqliteService;

#[derive(Debug, Clone)]
pub struct S3Credential {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub org_id: String,
}

#[derive(Clone)]
pub struct S3Credentials {
    sqlite: Arc<SqliteService>,
}

impl S3Credentials {
    pub fn new(sqlite: Arc<SqliteService>) -> Self {
        Self { sqlite }
    }

    async fn ensure_table(&self) -> Result<()> {
        self.sqlite
            .execute(
                "CREATE TABLE IF NOT EXISTS sys_s3_credentials (
                    access_key_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    secret_enc TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                )"
                .to_string(),
                vec![],
            )
            .await?;
        Ok(())
    }

    /// Mint a credential for an organization.
    ///
    /// The secret is returned **once**, in the clear, and never again: what is
    /// stored is the encrypted form. A caller that loses it mints a new one,
    /// which is the same contract as every other object store.
    pub async fn create(&self, org_id: &str) -> Result<S3Credential> {
        self.ensure_table().await?;

        // 20 characters, uppercase alphanumeric: the shape every S3 client and
        // config file expects. Not a UUID — clients and tools have length
        // assumptions, and a credential that will not paste into a config is not
        // usable however correct it is.
        let access_key_id = random_id(20);
        let secret_access_key = random_secret();
        let sealed = crate::crypto::SecretBox::from_env()
            .encrypt(&secret_access_key)
            .context("encrypting the S3 secret")?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as i64;
        self.sqlite
            .execute(
                "INSERT INTO sys_s3_credentials \
                 (access_key_id, org_id, secret_enc, created_at_ms) VALUES (?, ?, ?, ?)"
                    .to_string(),
                vec![
                    serde_json::json!(access_key_id),
                    serde_json::json!(org_id),
                    serde_json::json!(sealed),
                    serde_json::json!(now),
                ],
            )
            .await?;

        Ok(S3Credential {
            access_key_id,
            secret_access_key,
            org_id: org_id.to_string(),
        })
    }

    /// Look up a credential by access key id, decrypting the secret.
    pub async fn lookup(&self, access_key_id: &str) -> Result<Option<S3Credential>> {
        self.ensure_table().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT org_id, secret_enc FROM sys_s3_credentials WHERE access_key_id = ?"
                    .to_string(),
                vec![serde_json::json!(access_key_id)],
            )
            .await?;
        let Some(row) = rows.into_iter().next() else {
            return Ok(None);
        };
        let org_id = row
            .get("org_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let sealed = row
            .get("secret_enc")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let secret_access_key = crate::crypto::SecretBox::from_env()
            .decrypt(sealed)
            .context("decrypting the S3 secret — has LUMA_MASTER_KEY changed?")?;

        Ok(Some(S3Credential {
            access_key_id: access_key_id.to_string(),
            secret_access_key,
            org_id,
        }))
    }

    /// Every credential an organization holds, without their secrets.
    ///
    /// Deliberately without: a listing endpoint that returns secrets turns one
    /// read-only leak into a full compromise.
    pub async fn list(&self, org_id: &str) -> Result<Vec<(String, i64)>> {
        self.ensure_table().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT access_key_id, created_at_ms FROM sys_s3_credentials \
                 WHERE org_id = ? ORDER BY created_at_ms DESC"
                    .to_string(),
                vec![serde_json::json!(org_id)],
            )
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                Some((
                    row.get("access_key_id")?.as_str()?.to_string(),
                    row.get("created_at_ms")?.as_i64().unwrap_or(0),
                ))
            })
            .collect())
    }

    /// Revoke a credential. Returns whether one was removed.
    pub async fn revoke(&self, access_key_id: &str, org_id: &str) -> Result<bool> {
        self.ensure_table().await?;
        // Scoped by org: without it, knowing an access key id from a log would
        // be enough to revoke another tenant's access.
        let affected = self
            .sqlite
            .execute(
                "DELETE FROM sys_s3_credentials WHERE access_key_id = ? AND org_id = ?".to_string(),
                vec![serde_json::json!(access_key_id), serde_json::json!(org_id)],
            )
            .await?;
        Ok(affected > 0)
    }
}

fn random_id(length: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut bytes = vec![0u8; length];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
        .into_iter()
        .map(|b| ALPHABET[(b as usize) % ALPHABET.len()] as char)
        .collect()
}

fn random_secret() -> String {
    let mut bytes = [0u8; 30];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn store() -> (S3Credentials, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let sqlite = SqliteService::new(dir.path().join("meta.db")).unwrap();
        (S3Credentials::new(Arc::new(sqlite)), dir)
    }

    #[tokio::test]
    async fn a_minted_credential_can_be_looked_up_with_its_secret_intact() {
        // The whole reason this exists: SigV4 needs the secret back, so a hash
        // would be useless. Round-tripping it is the property.
        let (store, _dir) = store().await;
        let minted = store.create("acme").await.unwrap();
        let found = store
            .lookup(&minted.access_key_id)
            .await
            .unwrap()
            .expect("must be found");
        assert_eq!(found.secret_access_key, minted.secret_access_key);
        assert_eq!(found.org_id, "acme");
    }

    #[tokio::test]
    async fn the_secret_is_not_stored_in_the_clear() {
        // If it were, the encryption would be decoration.
        let (store, _dir) = store().await;
        let minted = store.create("acme").await.unwrap();
        let rows = store
            .sqlite
            .query(
                "SELECT secret_enc FROM sys_s3_credentials WHERE access_key_id = ?".to_string(),
                vec![serde_json::json!(minted.access_key_id)],
            )
            .await
            .unwrap();
        let stored = rows[0]["secret_enc"].as_str().unwrap();
        assert_ne!(stored, minted.secret_access_key);
        assert!(
            crate::crypto::SecretBox::is_encrypted(stored),
            "the stored value must carry the encryption marker: {stored}"
        );
    }

    #[tokio::test]
    async fn an_unknown_access_key_is_none_and_not_an_error() {
        // A forged key id is an ordinary "no such credential", not a failure the
        // caller has to distinguish from a database problem.
        let (store, _dir) = store().await;
        assert!(store.lookup("AKIDNOTREAL").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn listing_never_returns_secrets() {
        let (store, _dir) = store().await;
        let minted = store.create("acme").await.unwrap();
        let listed = store.list("acme").await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].0, minted.access_key_id);
    }

    #[tokio::test]
    async fn revocation_is_scoped_to_the_owning_org() {
        // Without the scope, an access key id seen in a log would be enough to
        // revoke another tenant's access.
        let (store, _dir) = store().await;
        let minted = store.create("acme").await.unwrap();
        assert!(!store.revoke(&minted.access_key_id, "globex").await.unwrap());
        assert!(store.lookup(&minted.access_key_id).await.unwrap().is_some());
        assert!(store.revoke(&minted.access_key_id, "acme").await.unwrap());
        assert!(store.lookup(&minted.access_key_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn two_credentials_never_collide() {
        let (store, _dir) = store().await;
        let a = store.create("acme").await.unwrap();
        let b = store.create("acme").await.unwrap();
        assert_ne!(a.access_key_id, b.access_key_id);
        assert_ne!(a.secret_access_key, b.secret_access_key);
    }

    #[test]
    fn the_access_key_id_has_the_shape_clients_expect() {
        // Not cosmetic: tools and config files have length assumptions, and a
        // credential that will not paste into one is unusable however correct.
        let id = random_id(20);
        assert_eq!(id.len(), 20);
        assert!(id
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()));
    }
}
