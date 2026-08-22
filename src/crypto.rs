//! Encryption-at-rest for sensitive fields (API-key secrets, provider
//! credentials) using an AEAD cipher (ChaCha20-Poly1305).
//!
//! A 256-bit key is derived from a master secret provided via the
//! `LUMA_MASTER_KEY` environment variable using Argon2id with a fixed,
//! application-specific salt (domain separation). Ciphertexts are
//! self-describing (`enc:v2:<base64(nonce||ciphertext)>`), so a field can be
//! transparently detected and decrypted later. Legacy `enc:v1:` tokens (bare
//! SHA-256 key derivation) are still decryptable for backward compatibility.
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

const CIPHERTEXT_PREFIX_V1: &str = "enc:v1:";
const CIPHERTEXT_PREFIX_V2: &str = "enc:v2:";

/// Fixed, application-specific salt for the Argon2 key derivation. This is not
/// a secret; its purpose is domain separation so the same master key yields a
/// Luma-specific data key (and to make the KDF deterministic, since the derived
/// key must be reproducible across restarts to decrypt stored ciphertexts).
const KDF_SALT: &[u8] = b"luma-secretbox-kdf-v2-salt";

/// AEAD secret-box keyed by a master secret.
#[derive(Clone)]
pub struct SecretBox {
    /// v2 cipher (Argon2id-derived key): used for encryption and v2 decryption.
    cipher: ChaCha20Poly1305,
    /// v1 cipher (legacy bare-SHA-256 key): decrypt-only, for backward compat.
    legacy_cipher: ChaCha20Poly1305,
}

impl SecretBox {
    /// Derive a box from a master secret string.
    ///
    /// v2 keys are derived with Argon2id over a fixed application salt, which is
    /// far more resistant to brute force than the previous bare SHA-256. A
    /// legacy SHA-256 cipher is retained so pre-existing `enc:v1:` ciphertexts
    /// still decrypt.
    pub fn from_master(master: &str) -> Self {
        // v2: Argon2id KDF with a fixed application salt.
        let mut key_bytes = [0u8; 32];
        Argon2::default()
            .hash_password_into(master.as_bytes(), KDF_SALT, &mut key_bytes)
            .expect("argon2 key derivation into 32 bytes cannot fail with valid params");
        let key = Key::from_slice(&key_bytes);

        // v1 (legacy, decrypt-only): bare SHA-256 of the master.
        let mut hasher = Sha256::new();
        hasher.update(master.as_bytes());
        let legacy_digest = hasher.finalize();
        let legacy_key = Key::from_slice(legacy_digest.as_slice());

        Self {
            cipher: ChaCha20Poly1305::new(key),
            legacy_cipher: ChaCha20Poly1305::new(legacy_key),
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

    /// Encrypt raw bytes, returning `nonce || ciphertext`.
    ///
    /// Distinct from [`SecretBox::encrypt`] because that one is for *fields* —
    /// it base64s and prefixes so a ciphertext can sit in a JSON string and be
    /// recognised later. Backup artifacts are whole files, often large, and
    /// wrapping them in base64 would inflate every one by a third for no gain:
    /// the caller already knows the object is encrypted.
    pub fn encrypt_bytes(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| anyhow!("encryption failed: {e}"))?;
        let mut blob = Vec::with_capacity(12 + ciphertext.len());
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ciphertext);
        Ok(blob)
    }

    /// Decrypt bytes produced by [`SecretBox::encrypt_bytes`].
    ///
    /// A failure here means the bytes are not what we wrote — wrong key, or a
    /// tampered or truncated object — and the caller must abort rather than
    /// carry on with whatever came back.
    pub fn decrypt_bytes(&self, sealed: &[u8]) -> Result<Vec<u8>> {
        if sealed.len() < 12 {
            return Err(anyhow!("ciphertext too short to contain a nonce"));
        }
        let (nonce_bytes, ciphertext) = sealed.split_at(12);
        self.cipher
            .decrypt(Nonce::from_slice(nonce_bytes), ciphertext)
            .map_err(|_| anyhow!("decryption failed: wrong key, or the data was modified"))
    }

    /// Encrypt `plaintext`, returning a self-describing `enc:v2:...` token.
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
        Ok(format!("{CIPHERTEXT_PREFIX_V2}{}", STANDARD.encode(blob)))
    }

    /// Decrypt an `enc:v2:...` (or legacy `enc:v1:...`) token produced by
    /// [`SecretBox::encrypt`]. v2 uses the Argon2id-derived key; v1 falls back
    /// to the legacy SHA-256-derived key.
    pub fn decrypt(&self, token: &str) -> Result<String> {
        let (cipher, b64) = if let Some(b64) = token.strip_prefix(CIPHERTEXT_PREFIX_V2) {
            (&self.cipher, b64)
        } else if let Some(b64) = token.strip_prefix(CIPHERTEXT_PREFIX_V1) {
            (&self.legacy_cipher, b64)
        } else {
            return Err(anyhow!("not an encrypted value"));
        };
        let raw = STANDARD.decode(b64)?;
        if raw.len() < 12 {
            return Err(anyhow!("ciphertext too short"));
        }
        let (nonce_bytes, ciphertext) = raw.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        let plaintext = cipher
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

    /// Whether a string is an `enc:v2:` (or legacy `enc:v1:`) ciphertext token.
    pub fn is_encrypted(value: &str) -> bool {
        value.starts_with(CIPHERTEXT_PREFIX_V2) || value.starts_with(CIPHERTEXT_PREFIX_V1)
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
    fn encrypt_uses_v2_prefix() {
        let sb = SecretBox::from_master("k");
        let ct = sb.encrypt("secret").unwrap();
        assert!(ct.starts_with("enc:v2:"), "new ciphertexts must be v2");
    }

    #[test]
    fn decrypts_legacy_v1_ciphertext() {
        // Build a v1 token exactly as the old bare-SHA-256 path would have, then
        // confirm the new SecretBox still decrypts it via the legacy cipher.
        let master = "legacy-master-secret";
        let mut hasher = Sha256::new();
        hasher.update(master.as_bytes());
        let digest = hasher.finalize();
        let legacy_cipher = ChaCha20Poly1305::new(Key::from_slice(digest.as_slice()));

        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let ct = legacy_cipher
            .encrypt(Nonce::from_slice(&nonce_bytes), "legacy-value".as_bytes())
            .unwrap();
        let mut blob = Vec::new();
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ct);
        let token = format!("{CIPHERTEXT_PREFIX_V1}{}", STANDARD.encode(blob));

        let sb = SecretBox::from_master(master);
        assert!(SecretBox::is_encrypted(&token));
        assert_eq!(sb.decrypt(&token).unwrap(), "legacy-value");
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
