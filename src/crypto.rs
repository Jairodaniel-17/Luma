//! Encryption-at-rest for sensitive fields (API-key secrets, provider
//! credentials) using an AEAD cipher (ChaCha20-Poly1305).
//!
//! A 256-bit key is derived from a master secret provided via the
//! `LUMA_MASTER_KEY` environment variable (SHA-256 of the secret). Ciphertexts
//! are self-describing (`enc:v1:<base64(nonce||ciphertext)>`), so a field can be
//! transparently detected and decrypted later.
//!
//! Password hashing uses Argon2id (see [`hash_password`] / [`verify_password`]).

use anyhow::{anyhow, Result};
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chacha20poly1305::aead::Aead;
use chacha20poly1305::{ChaCha20Poly1305, Key, KeyInit, Nonce};
use rand::RngCore;
use sha2::{Digest, Sha256};

const CIPHERTEXT_PREFIX: &str = "enc:v1:";

/// AEAD secret-box keyed by a master secret.
#[derive(Clone)]
pub struct SecretBox {
    cipher: ChaCha20Poly1305,
}

impl SecretBox {
    /// Derive a box from a master secret string.
    pub fn from_master(master: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(master.as_bytes());
        let digest = hasher.finalize();
        let key = Key::from_slice(digest.as_slice());
        Self {
            cipher: ChaCha20Poly1305::new(key),
        }
    }

    /// Build the box from the `LUMA_MASTER_KEY` environment variable, falling
    /// back to a well-known insecure development key (with a warning logged by
    /// the caller). Production deployments MUST set `LUMA_MASTER_KEY`.
    pub fn from_env() -> Self {
        let master = std::env::var("LUMA_MASTER_KEY")
            .unwrap_or_else(|_| "luma-insecure-dev-master-key".to_string());
        Self::from_master(&master)
    }

    /// Encrypt `plaintext`, returning a self-describing `enc:v1:...` token.
    pub fn encrypt(&self, plaintext: &str) -> Result<String> {
        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| anyhow!("encryption failed: {e}"))?;
        let mut blob = Vec::with_capacity(12 + ciphertext.len());
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ciphertext);
        Ok(format!("{CIPHERTEXT_PREFIX}{}", STANDARD.encode(blob)))
    }

    /// Decrypt an `enc:v1:...` token produced by [`SecretBox::encrypt`].
    pub fn decrypt(&self, token: &str) -> Result<String> {
        let b64 = token
            .strip_prefix(CIPHERTEXT_PREFIX)
            .ok_or_else(|| anyhow!("not an encrypted value"))?;
        let raw = STANDARD.decode(b64)?;
        if raw.len() < 12 {
            return Err(anyhow!("ciphertext too short"));
        }
        let (nonce_bytes, ciphertext) = raw.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        let plaintext = self
            .cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("decryption failed: {e}"))?;
        Ok(String::from_utf8(plaintext)?)
    }

    /// If `value` looks encrypted, decrypt it; otherwise return it unchanged.
    /// Useful for reading fields that may predate encryption.
    pub fn decrypt_if_needed(&self, value: &str) -> String {
        if Self::is_encrypted(value) {
            self.decrypt(value).unwrap_or_else(|_| value.to_string())
        } else {
            value.to_string()
        }
    }

    /// Whether a string is an `enc:v1:` ciphertext token.
    pub fn is_encrypted(value: &str) -> bool {
        value.starts_with(CIPHERTEXT_PREFIX)
    }
}

/// Hash a password with Argon2id, returning a PHC string.
pub fn hash_password(password: &str) -> Result<String> {
    let mut salt_bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut salt_bytes);
    let salt = SaltString::encode_b64(&salt_bytes).map_err(|e| anyhow!("salt: {e}"))?;
    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| anyhow!("hash: {e}"))?;
    Ok(hash.to_string())
}

/// Verify a plaintext password against a stored Argon2 PHC string.
pub fn verify_password(password: &str, phc: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(phc) else {
        return false;
    };
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aead_roundtrip() {
        let sb = SecretBox::from_master("test-master-key");
        let secret = "sk-provider-credential-1234567890";
        let ct = sb.encrypt(secret).unwrap();
        assert!(SecretBox::is_encrypted(&ct));
        assert_ne!(ct, secret);
        assert_eq!(sb.decrypt(&ct).unwrap(), secret);
    }

    #[test]
    fn aead_wrong_key_fails() {
        let a = SecretBox::from_master("key-a");
        let b = SecretBox::from_master("key-b");
        let ct = a.encrypt("hello").unwrap();
        assert!(b.decrypt(&ct).is_err(), "wrong key must not decrypt");
    }

    #[test]
    fn aead_nonce_is_randomized() {
        let sb = SecretBox::from_master("k");
        assert_ne!(sb.encrypt("same").unwrap(), sb.encrypt("same").unwrap());
    }

    #[test]
    fn password_hash_verify() {
        let phc = hash_password("hunter2").unwrap();
        assert!(phc.starts_with("$argon2"));
        assert!(verify_password("hunter2", &phc));
        assert!(!verify_password("wrong", &phc));
    }
}
