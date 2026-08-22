//! Off-host backup destinations.
//!
//! W1.3 of `docs/PLAN-MAESTRO.md`. A backup that lives on the same disk as the
//! data protects against corruption and accidental deletion, and against
//! nothing else: losing the volume loses both copies at once. This uploads a
//! backup directory to object storage — S3, R2, GCS, MinIO — through
//! `object_store`, so one code path covers all of them.
//!
//! ## Encryption
//!
//! Artifacts are encrypted with the instance master key before they leave the
//! host, using the same `ChaCha20-Poly1305` box as encryption at rest. The
//! bucket therefore never holds readable data, which matters because a backup
//! bucket is usually the least-guarded copy of a system's contents: it outlives
//! hosts, gets shared with whoever is doing the restore, and is the first thing
//! a misconfigured policy exposes.
//!
//! ## Testing
//!
//! Every function here takes an `Arc<dyn ObjectStore>` rather than building one
//! internally, so the tests drive the real upload and download logic against
//! `object_store`'s in-memory backend. That covers the part that can be wrong —
//! path construction, ordering, encryption, the manifest — without needing a
//! live bucket. Credentials and endpoint wiring is the part that needs a real
//! MinIO, and is exercised by the runbook rather than by unit tests.

use anyhow::{anyhow, Context, Result};
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Name of the manifest inside a remote backup prefix.
const MANIFEST_OBJECT: &str = "manifest.json";

/// Every file in a local backup directory, as (relative path, absolute path).
///
/// Sorted so an upload is deterministic and a partially-completed one can be
/// reasoned about: the manifest is uploaded **last**, so a prefix carrying a
/// manifest is a prefix whose data finished uploading.
fn collect_files(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)?.flatten() {
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                stack.push(path);
            } else {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|_| anyhow!("file escaped the backup root"))?
                    .to_string_lossy()
                    // Object storage keys use forward slashes on every
                    // platform; a Windows-produced backup must restore on Linux.
                    .replace('\\', "/");
                out.push((relative, path));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Upload a verified local backup directory under `prefix` in `store`.
///
/// Returns the object keys written, in the order they were written.
///
/// The manifest goes last on purpose: an interrupted upload leaves a prefix
/// with no manifest, and [`download`] refuses such a prefix rather than
/// restoring a partial backup that looks complete.
pub async fn upload(
    store: &Arc<dyn ObjectStore>,
    prefix: &str,
    local: &Path,
    encrypt: impl Fn(&[u8]) -> Result<Vec<u8>>,
) -> Result<Vec<String>> {
    let files = collect_files(local)?;
    if files.is_empty() {
        return Err(anyhow!(
            "refusing to upload an empty backup directory: {}",
            local.display()
        ));
    }

    let mut written = Vec::with_capacity(files.len());
    let mut manifest: Option<(String, PathBuf)> = None;

    for (relative, path) in files {
        if relative == MANIFEST_OBJECT {
            manifest = Some((relative, path));
            continue;
        }
        let key = put_one(store, prefix, &relative, &path, &encrypt).await?;
        written.push(key);
    }

    let (relative, path) = manifest.ok_or_else(|| {
        anyhow!("local backup has no manifest.json; run `luma backup` rather than uploading a hand-made directory")
    })?;
    written.push(put_one(store, prefix, &relative, &path, &encrypt).await?);

    tracing::info!(prefix, objects = written.len(), "backup uploaded");
    Ok(written)
}

async fn put_one(
    store: &Arc<dyn ObjectStore>,
    prefix: &str,
    relative: &str,
    path: &Path,
    encrypt: &impl Fn(&[u8]) -> Result<Vec<u8>>,
) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let sealed = encrypt(&bytes).with_context(|| format!("encrypting {relative}"))?;
    let key = format!("{}/{relative}", prefix.trim_end_matches('/'));
    store
        .put(&ObjectPath::from(key.clone()), PutPayload::from(sealed))
        .await
        .with_context(|| format!("uploading {key}"))?;
    Ok(key)
}

/// Download a remote backup prefix into `local`.
///
/// Refuses a prefix with no manifest: that is what an interrupted upload leaves
/// behind, and restoring from it would produce a system that is quietly missing
/// whatever had not been uploaded yet.
pub async fn download(
    store: &Arc<dyn ObjectStore>,
    prefix: &str,
    local: &Path,
    decrypt: impl Fn(&[u8]) -> Result<Vec<u8>>,
) -> Result<usize> {
    let prefix = prefix.trim_end_matches('/').to_string();
    let manifest_key = ObjectPath::from(format!("{prefix}/{MANIFEST_OBJECT}"));
    store.head(&manifest_key).await.map_err(|_| {
        anyhow!(
            "remote backup at `{prefix}` has no {MANIFEST_OBJECT}: it is incomplete \
             (an interrupted upload) and restoring it would silently drop data"
        )
    })?;

    let listing = {
        use futures_util::StreamExt as _;
        let mut stream = store.list(Some(&ObjectPath::from(prefix.clone())));
        let mut keys = Vec::new();
        while let Some(meta) = stream.next().await {
            keys.push(meta?.location);
        }
        keys
    };

    let mut restored = 0;
    for key in listing {
        let relative = key
            .as_ref()
            .strip_prefix(&format!("{prefix}/"))
            .ok_or_else(|| anyhow!("object `{key}` is not under `{prefix}`"))?
            .to_string();

        let sealed = store.get(&key).await?.bytes().await?;
        let plain = decrypt(&sealed).with_context(|| format!("decrypting {relative}"))?;

        let target = local.join(&relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, plain)?;
        restored += 1;
    }

    tracing::info!(prefix, objects = restored, "backup downloaded");
    Ok(restored)
}

/// Delete every remote backup prefix beyond the newest `retention`.
///
/// Prefixes embed a monotonic millisecond stamp, so lexicographic order is
/// chronological — the same property local pruning relies on.
pub async fn prune(
    store: &Arc<dyn ObjectStore>,
    root: &str,
    retention: usize,
) -> Result<Vec<String>> {
    if retention == 0 {
        return Ok(Vec::new());
    }
    let root = root.trim_end_matches('/').to_string();

    let mut prefixes: Vec<String> = {
        use futures_util::StreamExt as _;
        let mut stream = store.list(Some(&ObjectPath::from(root.clone())));
        let mut seen = std::collections::BTreeSet::new();
        while let Some(meta) = stream.next().await {
            let key = meta?.location;
            if let Some(rest) = key.as_ref().strip_prefix(&format!("{root}/")) {
                if let Some((backup, _)) = rest.split_once('/') {
                    seen.insert(backup.to_string());
                }
            }
        }
        seen.into_iter().collect()
    };
    prefixes.sort();

    if prefixes.len() <= retention {
        return Ok(Vec::new());
    }

    let mut removed = Vec::new();
    for stale in &prefixes[..prefixes.len() - retention] {
        let full = format!("{root}/{stale}");
        use futures_util::StreamExt as _;
        let mut stream = store.list(Some(&ObjectPath::from(full.clone())));
        let mut keys = Vec::new();
        while let Some(meta) = stream.next().await {
            keys.push(meta?.location);
        }
        for key in keys {
            store.delete(&key).await?;
        }
        removed.push(full);
    }
    tracing::info!(root, pruned = removed.len(), "remote backups pruned");
    Ok(removed)
}

// ─── configuration and wiring ────────────────────────────────────────────────

/// Build the configured remote store, or `None` when no remote is configured.
///
/// Remote backup is opt-in: with `backup_remote_url` unset this returns `None`
/// and the local backup path is unchanged. Credentials come from the standard
/// AWS environment variables, so an instance running with an instance profile,
/// a `.env`, or a Kubernetes secret all work without Luma-specific plumbing.
pub fn store_from_config(config: &crate::config::Config) -> Result<Option<RemoteTarget>> {
    let url = config.backup_remote_url.trim();
    if url.is_empty() {
        return Ok(None);
    }

    let parsed =
        url::Url::parse(url).with_context(|| format!("backup_remote_url is not a URL: {url}"))?;

    let bucket = parsed
        .host_str()
        .ok_or_else(|| anyhow!("backup_remote_url has no bucket: {url}"))?
        .to_string();
    // The URL path is the prefix inside the bucket, so several instances can
    // share one bucket without colliding.
    let prefix = parsed.path().trim_matches('/').to_string();

    let mut builder = object_store::aws::AmazonS3Builder::from_env().with_bucket_name(&bucket);
    if !config.backup_remote_endpoint.trim().is_empty() {
        // A custom endpoint means MinIO, R2 or another S3-compatible service.
        // Those are commonly reached by path style, and virtual-host style
        // would resolve a bucket-prefixed hostname that does not exist.
        builder = builder
            .with_endpoint(config.backup_remote_endpoint.trim())
            .with_allow_http(config.backup_remote_allow_http)
            .with_virtual_hosted_style_request(false);
    }
    if !config.backup_remote_region.trim().is_empty() {
        builder = builder.with_region(config.backup_remote_region.trim());
    }

    let store: Arc<dyn ObjectStore> = Arc::new(
        builder
            .build()
            .with_context(|| format!("building the S3 client for {url}"))?,
    );
    Ok(Some(RemoteTarget {
        store,
        prefix: if prefix.is_empty() {
            "luma-backups".to_string()
        } else {
            prefix
        },
    }))
}

/// A configured remote destination.
pub struct RemoteTarget {
    pub store: Arc<dyn ObjectStore>,
    /// Prefix inside the bucket that all of this instance's backups live under.
    pub prefix: String,
}

/// Upload a local backup directory to the configured remote and prune old ones.
///
/// The local backup must already be verified: uploading an unverified artifact
/// would replicate a broken backup off-host and consume the retention slot of a
/// good one.
pub async fn ship(
    target: &RemoteTarget,
    local: &Path,
    retention: usize,
    secrets: &crate::crypto::SecretBox,
) -> Result<Vec<String>> {
    let name = local
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("backup directory has no name: {}", local.display()))?;
    let prefix = format!("{}/{name}", target.prefix);

    let keys = upload(&target.store, &prefix, local, |bytes| {
        secrets.encrypt_bytes(bytes)
    })
    .await?;

    // Pruning after a successful upload, never before: losing the newest remote
    // copy to make room for one that then fails to upload is the worst possible
    // ordering.
    prune(&target.store, &target.prefix, retention).await?;
    Ok(keys)
}

/// Fetch a remote backup into `local`, decrypting as it goes.
pub async fn fetch(
    target: &RemoteTarget,
    backup_name: &str,
    local: &Path,
    secrets: &crate::crypto::SecretBox,
) -> Result<usize> {
    let prefix = format!("{}/{backup_name}", target.prefix);
    download(&target.store, &prefix, local, |bytes| {
        secrets.decrypt_bytes(bytes)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    /// Reversible stand-in for the master-key box, so the tests exercise the
    /// encrypt/decrypt plumbing without depending on key material.
    fn seal(bytes: &[u8]) -> Result<Vec<u8>> {
        let mut out = b"SEALED:".to_vec();
        out.extend_from_slice(bytes);
        Ok(out)
    }

    fn unseal(bytes: &[u8]) -> Result<Vec<u8>> {
        bytes
            .strip_prefix(b"SEALED:".as_slice())
            .map(|b| b.to_vec())
            .ok_or_else(|| anyhow!("not sealed"))
    }

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn make_backup(dir: &Path) {
        std::fs::create_dir_all(dir.join("vectors/docs/runs")).unwrap();
        std::fs::write(dir.join("manifest.json"), b"{\"blob_files\":1}").unwrap();
        std::fs::write(dir.join("events-000001.log"), b"wal").unwrap();
        std::fs::write(dir.join("vectors/docs/manifest.json"), b"vec").unwrap();
        std::fs::write(dir.join("vectors/docs/runs/run-1.log"), b"run").unwrap();
    }

    #[tokio::test]
    async fn upload_then_download_roundtrips_the_whole_tree() {
        let src = tempfile::tempdir().unwrap();
        make_backup(src.path());
        let store = store();

        let keys = upload(&store, "backups/backup-1", src.path(), seal)
            .await
            .unwrap();
        assert_eq!(keys.len(), 4);

        let dst = tempfile::tempdir().unwrap();
        let restored = download(&store, "backups/backup-1", dst.path(), unseal)
            .await
            .unwrap();
        assert_eq!(restored, 4);

        // Nested paths survive, which is the part a naive flat upload breaks.
        assert_eq!(
            std::fs::read(dst.path().join("vectors/docs/runs/run-1.log")).unwrap(),
            b"run"
        );
        assert_eq!(
            std::fs::read(dst.path().join("manifest.json")).unwrap(),
            b"{\"blob_files\":1}"
        );
    }

    #[tokio::test]
    async fn objects_are_encrypted_at_rest_in_the_bucket() {
        // A backup bucket is usually the least-guarded copy of a system's
        // contents, so the plaintext must never be what lands there.
        let src = tempfile::tempdir().unwrap();
        make_backup(src.path());
        let store = store();
        upload(&store, "b/backup-1", src.path(), seal)
            .await
            .unwrap();

        let raw = store
            .get(&ObjectPath::from("b/backup-1/events-000001.log"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert!(
            raw.starts_with(b"SEALED:"),
            "object was stored without going through the encryptor"
        );
        assert_ne!(raw.as_ref(), b"wal");
    }

    #[tokio::test]
    async fn the_manifest_is_uploaded_last() {
        // An interrupted upload must not leave a prefix that looks complete.
        // Writing the manifest last makes its presence the completion marker.
        let src = tempfile::tempdir().unwrap();
        make_backup(src.path());
        let store = store();
        let keys = upload(&store, "b/backup-1", src.path(), seal)
            .await
            .unwrap();
        assert!(
            keys.last().unwrap().ends_with("manifest.json"),
            "manifest must be the final object written, got order: {keys:?}"
        );
    }

    #[tokio::test]
    async fn download_refuses_a_prefix_with_no_manifest() {
        let store = store();
        store
            .put(
                &ObjectPath::from("b/backup-1/events-000001.log"),
                PutPayload::from(seal(b"wal").unwrap()),
            )
            .await
            .unwrap();

        let dst = tempfile::tempdir().unwrap();
        let err = download(&store, "b/backup-1", dst.path(), unseal)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no manifest.json"),
            "an incomplete prefix must be refused by name, got: {err}"
        );
    }

    #[tokio::test]
    async fn upload_refuses_a_directory_with_no_manifest() {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("stray.txt"), b"x").unwrap();
        let err = upload(&store(), "b/backup-1", src.path(), seal)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no manifest.json"), "got: {err}");
    }

    #[tokio::test]
    async fn upload_refuses_an_empty_directory() {
        let src = tempfile::tempdir().unwrap();
        let err = upload(&store(), "b/backup-1", src.path(), seal)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty backup directory"), "got: {err}");
    }

    #[tokio::test]
    async fn a_tampered_object_fails_the_restore() {
        // Decryption failing is the signal that the bucket contents are not what
        // we wrote; it must abort rather than write garbage into the data dir.
        let src = tempfile::tempdir().unwrap();
        make_backup(src.path());
        let store = store();
        upload(&store, "b/backup-1", src.path(), seal)
            .await
            .unwrap();

        store
            .put(
                &ObjectPath::from("b/backup-1/events-000001.log"),
                PutPayload::from(b"tampered".to_vec()),
            )
            .await
            .unwrap();

        let dst = tempfile::tempdir().unwrap();
        assert!(download(&store, "b/backup-1", dst.path(), unseal)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn prune_keeps_the_newest_and_deletes_the_rest() {
        let store = store();
        let src = tempfile::tempdir().unwrap();
        make_backup(src.path());
        // Names embed a monotonic stamp, so lexicographic order is chronological.
        for stamp in ["backup-100", "backup-200", "backup-300"] {
            upload(&store, &format!("b/{stamp}"), src.path(), seal)
                .await
                .unwrap();
        }

        let removed = prune(&store, "b", 2).await.unwrap();
        assert_eq!(removed, vec!["b/backup-100".to_string()]);

        // The oldest is gone and the newest two are intact.
        assert!(store
            .head(&ObjectPath::from("b/backup-100/manifest.json"))
            .await
            .is_err());
        assert!(store
            .head(&ObjectPath::from("b/backup-300/manifest.json"))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn prune_with_zero_retention_is_a_no_op() {
        // Retention 0 means "keep everything" everywhere else in the codebase;
        // treating it as "delete everything" here would be catastrophic.
        let store = store();
        let src = tempfile::tempdir().unwrap();
        make_backup(src.path());
        upload(&store, "b/backup-1", src.path(), seal)
            .await
            .unwrap();

        assert!(prune(&store, "b", 0).await.unwrap().is_empty());
        assert!(store
            .head(&ObjectPath::from("b/backup-1/manifest.json"))
            .await
            .is_ok());
    }

    #[test]
    fn keys_use_forward_slashes_on_every_platform() {
        // A backup produced on Windows has to restore on Linux, so the object
        // keys cannot carry backslashes.
        let dir = tempfile::tempdir().unwrap();
        make_backup(dir.path());
        let files = collect_files(dir.path()).unwrap();
        assert!(files.iter().all(|(rel, _)| !rel.contains('\\')));
        assert!(files
            .iter()
            .any(|(rel, _)| rel == "vectors/docs/runs/run-1.log"));
    }
}
