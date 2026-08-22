//! Read-only replica following a shipped WAL.
//!
//! W2.2 of `docs/PLAN-MAESTRO.md`. **Scope is deliberately frozen**: reads only,
//! promotion by hand. The plan names the risk this item carries — that it grows
//! into a consensus project — so the boundary is stated here in code rather than
//! left to judgement. Raft is backlog with an explicit entry criterion (a real
//! multi-writer need), not a natural extension of this.
//!
//! ## How it follows
//!
//! The primary ships snapshot and WAL segments to object storage
//! ([`crate::wal_ship`]). A replica polls that prefix, downloads whatever has
//! grown, and lets the ordinary boot-time replay apply it. There is no separate
//! apply path: **the follow path is the boot path**, which is the same reasoning
//! that made shipping send raw segments — it is already covered by the
//! crash-recovery matrix.
//!
//! ## Why writes are refused rather than forwarded
//!
//! A replica that quietly proxied writes to the primary would be a
//! single-writer cluster with no consensus and no fencing, and the failure mode
//! is split-brain after a network partition. Refusing with a clear error means
//! a misconfigured client finds out immediately instead of at the worst moment.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Marker file that records "this data directory belongs to a replica".
///
/// On disk rather than only in config so a replica cannot be turned into a
/// primary by editing an env var and restarting — promotion is an explicit act
/// (`luma promote`) that leaves a trace.
const REPLICA_MARKER: &str = "REPLICA";

/// Bytes `SecretBox::encrypt_bytes` adds: a 12-byte nonce plus the 16-byte
/// Poly1305 tag. Used only to line a local plaintext size up with a bucket
/// listing after a restart.
const AEAD_OVERHEAD_BYTES: u64 = 28;

/// Runtime role of this instance.
#[derive(Clone)]
pub struct Role {
    read_only: Arc<AtomicBool>,
    /// Bytes of WAL applied since start, for the lag metric.
    followed_bytes: Arc<AtomicU64>,
}

impl Default for Role {
    fn default() -> Self {
        Self::primary()
    }
}

impl Role {
    pub fn primary() -> Self {
        Self {
            read_only: Arc::new(AtomicBool::new(false)),
            followed_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn replica() -> Self {
        Self {
            read_only: Arc::new(AtomicBool::new(true)),
            followed_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Determine the role from the data directory.
    ///
    /// The marker on disk wins over any configuration: a replica that came back
    /// from a restart must not silently become a primary because its
    /// environment changed.
    pub fn from_data_dir(data_dir: &Path) -> Self {
        if data_dir.join(REPLICA_MARKER).exists() {
            Self::replica()
        } else {
            Self::primary()
        }
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only.load(Ordering::Relaxed)
    }

    pub fn followed_bytes(&self) -> u64 {
        self.followed_bytes.load(Ordering::Relaxed)
    }

    pub fn record_followed(&self, bytes: u64) {
        self.followed_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn name(&self) -> &'static str {
        if self.is_read_only() {
            "replica"
        } else {
            "primary"
        }
    }
}

/// Mark a data directory as belonging to a replica.
pub fn mark_replica(data_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(data_dir)?;
    std::fs::write(
        data_dir.join(REPLICA_MARKER),
        b"This data directory is a read-only replica.\n\
          Writes are refused. Run `luma promote` to make it a primary.\n",
    )?;
    Ok(())
}

/// Promote a replica to primary by removing the marker.
///
/// Deliberately manual and deliberately not automatic. Automatic promotion
/// without fencing is how two primaries end up writing to one bucket, and the
/// plan puts fencing in W2.3 — so until that exists, an operator decides.
///
/// Refuses when the directory is already a primary, rather than reporting
/// success: "promoted" on something that was never a replica would tell an
/// operator mid-incident that they had done something they had not.
pub fn promote(data_dir: &Path) -> Result<()> {
    let marker = data_dir.join(REPLICA_MARKER);
    if !marker.exists() {
        return Err(anyhow!(
            "{} is not a replica: nothing to promote",
            data_dir.display()
        ));
    }
    std::fs::remove_file(&marker)?;
    tracing::warn!(
        data_dir = %data_dir.display(),
        "promoted to primary; make sure the old primary is stopped before it writes again"
    );
    Ok(())
}

/// Which shipped objects a replica has already taken, by name and **object**
/// size in the bucket.
///
/// Size rather than a hash because the WAL is append-only: a segment never
/// shrinks or is rewritten, so equal size means equal content.
///
/// It has to be the *ciphertext* size, which is what a listing reports, not the
/// plaintext length after decryption. Comparing a listing's size against a
/// recorded plaintext length never matches — encryption adds a nonce and a tag —
/// so every poll would re-download the entire stream forever, quietly, while
/// looking like it was working.
#[derive(Debug, Default)]
pub struct FollowState {
    seen: std::collections::HashMap<String, u64>,
}

impl FollowState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed from what is already on disk, so a restarted replica does not
    /// re-download the whole stream.
    ///
    /// The local file is plaintext while the bucket holds ciphertext, so the
    /// recorded size is inflated by the AEAD overhead to line the two up. It is
    /// an estimate, and a wrong one costs one extra download of one segment —
    /// far cheaper than the alternative of re-downloading everything.
    pub fn from_data_dir(data_dir: &Path) -> Self {
        let mut state = Self::new();
        if let Ok(entries) = std::fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("events-") && name.ends_with(".log") {
                    let plaintext = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    state
                        .seen
                        .insert(name, plaintext.saturating_add(AEAD_OVERHEAD_BYTES));
                }
            }
        }
        state
    }

    fn is_new(&self, name: &str, len: u64) -> bool {
        match self.seen.get(name) {
            Some(&seen) => len > seen,
            None => true,
        }
    }

    fn record(&mut self, name: &str, len: u64) {
        self.seen.insert(name.to_string(), len);
    }

    pub fn tracked(&self) -> usize {
        self.seen.len()
    }
}

/// One pass of following the shipped stream.
#[derive(Debug, Default, PartialEq)]
pub struct FollowReport {
    /// Objects written to the local data directory this pass.
    pub applied: Vec<String>,
    pub bytes: u64,
    /// Objects that had not changed.
    pub skipped: usize,
}

/// Download whatever the primary has shipped since the last pass.
///
/// Writes into `data_dir` in the layout the server boots from, so the engine's
/// existing replay picks it up. Returns what changed, so a caller can decide
/// whether a reload is needed at all.
pub async fn follow_once(
    store: &Arc<dyn object_store::ObjectStore>,
    prefix: &str,
    data_dir: &Path,
    state: &mut FollowState,
    decrypt: impl Fn(&[u8]) -> Result<Vec<u8>>,
) -> Result<FollowReport> {
    use futures_util::StreamExt as _;
    use object_store::path::Path as ObjectPath;
    use object_store::ObjectStoreExt as _;

    let prefix = prefix.trim_end_matches('/').to_string();
    std::fs::create_dir_all(data_dir)?;

    let mut listing = Vec::new();
    let mut stream = store.list(Some(&ObjectPath::from(prefix.clone())));
    while let Some(meta) = stream.next().await {
        listing.push(meta?);
    }
    if listing.is_empty() {
        return Err(anyhow!("the primary has shipped nothing under `{prefix}`"));
    }
    // Oldest first: a snapshot must land before the segments that follow it, or
    // a replay could start after events it has not seen.
    listing.sort_by(|a, b| a.location.as_ref().cmp(b.location.as_ref()));

    let mut report = FollowReport::default();
    for meta in listing {
        let relative = meta
            .location
            .as_ref()
            .strip_prefix(&format!("{prefix}/"))
            .ok_or_else(|| anyhow!("object `{}` is not under `{prefix}`", meta.location))?
            .to_string();
        // `wal/events-000001.log` lands as `events-000001.log`: the prefix
        // groups objects in the bucket, it is not part of the layout.
        let name = relative
            .strip_prefix("wal/")
            .unwrap_or(&relative)
            .to_string();

        // The snapshot is always taken: it is rewritten in place rather than
        // appended, so size is not a reliable "unchanged" signal for it.
        let is_snapshot = name == "snapshot.json";
        let object_size = meta.size;
        if !is_snapshot && !state.is_new(&name, object_size) {
            report.skipped += 1;
            continue;
        }

        let sealed = store.get(&meta.location).await?.bytes().await?;
        let plain = decrypt(&sealed)?;
        // Durable write: a replica that loses a segment to a crash would have a
        // hole it cannot detect, since its follow state would say it is caught
        // up.
        crate::durability::write_atomic(&data_dir.join(&name), &plain).await?;

        // Record what the listing said, not what decryption produced: the next
        // pass compares against a listing.
        state.record(&name, object_size);
        report.bytes += plain.len() as u64;
        report.applied.push(name);
    }
    Ok(report)
}

/// Spawn the follow loop. No-op unless the instance is configured as a replica.
pub fn spawn(config: crate::config::Config, role: Role) {
    if !role.is_read_only() {
        return;
    }
    let interval = config.replica_poll_interval_secs.max(1);
    let target = match crate::backup_remote::store_from_config(&config) {
        Ok(Some(target)) => target,
        Ok(None) => {
            tracing::error!(
                "this instance is a replica but no remote is configured; it will never catch up"
            );
            return;
        }
        Err(e) => {
            tracing::error!("replica cannot reach the remote: {e}");
            return;
        }
    };
    let Some(data_dir) = config.data_dir.clone() else {
        tracing::error!("a replica needs a data_dir");
        return;
    };

    tracing::info!(
        interval_secs = interval,
        prefix = %target.prefix,
        "following the primary's shipped WAL; this instance is read-only"
    );

    tokio::spawn(async move {
        let secrets = crate::crypto::SecretBox::from_env();
        let data_dir = PathBuf::from(data_dir);
        let mut state = FollowState::from_data_dir(&data_dir);
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        loop {
            ticker.tick().await;
            match follow_once(
                &target.store,
                &format!("{}/wal-stream", target.prefix),
                &data_dir,
                &mut state,
                |bytes| secrets.decrypt_bytes(bytes),
            )
            .await
            {
                Ok(report) if report.applied.is_empty() => {
                    tracing::debug!(skipped = report.skipped, "replica is caught up")
                }
                Ok(report) => {
                    role.record_followed(report.bytes);
                    // Restarting to apply is honest about the current scope: the
                    // engine replays at boot, and turning that into a live
                    // in-process apply is where this item starts becoming a
                    // replication project.
                    tracing::info!(
                        objects = report.applied.len(),
                        bytes = report.bytes,
                        "replica downloaded new WAL; restart to apply it"
                    );
                }
                // Never fatal: the next tick retries. Dying on a transient
                // network error would leave a replica silently frozen at
                // whatever offset it had, still serving reads as if current.
                Err(e) => tracing::error!("replica follow failed, will retry: {e}"),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

    fn seal(bytes: &[u8]) -> Result<Vec<u8>> {
        let mut out = b"S:".to_vec();
        out.extend_from_slice(bytes);
        Ok(out)
    }

    fn unseal(bytes: &[u8]) -> Result<Vec<u8>> {
        bytes
            .strip_prefix(b"S:".as_slice())
            .map(|b| b.to_vec())
            .ok_or_else(|| anyhow!("not sealed"))
    }

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    async fn ship(store: &Arc<dyn ObjectStore>, key: &str, body: &[u8]) {
        store
            .put(
                &object_store::path::Path::from(key.to_string()),
                PutPayload::from(seal(body).unwrap()),
            )
            .await
            .unwrap();
    }

    // ── role ─────────────────────────────────────────────────────────────────

    #[test]
    fn a_fresh_directory_is_a_primary() {
        let dir = tempfile::tempdir().unwrap();
        let role = Role::from_data_dir(dir.path());
        assert!(!role.is_read_only());
        assert_eq!(role.name(), "primary");
    }

    #[test]
    fn the_marker_makes_it_a_replica_across_restarts() {
        // On disk rather than only in config, so a replica cannot become a
        // primary by editing an env var and restarting.
        let dir = tempfile::tempdir().unwrap();
        mark_replica(dir.path()).unwrap();
        assert!(Role::from_data_dir(dir.path()).is_read_only());
        // Re-reading the same directory gives the same answer, which is the
        // point of persisting it.
        assert!(Role::from_data_dir(dir.path()).is_read_only());
    }

    #[test]
    fn promotion_removes_the_marker() {
        let dir = tempfile::tempdir().unwrap();
        mark_replica(dir.path()).unwrap();
        promote(dir.path()).unwrap();
        assert!(!Role::from_data_dir(dir.path()).is_read_only());
    }

    #[test]
    fn promoting_a_primary_is_an_error_not_a_no_op() {
        // Reporting success on something that was never a replica would tell an
        // operator mid-incident that they had done something they had not.
        let dir = tempfile::tempdir().unwrap();
        let err = promote(dir.path()).unwrap_err().to_string();
        assert!(err.contains("not a replica"), "{err}");
    }

    // ── following ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn a_first_pass_downloads_everything() {
        let store = store();
        ship(&store, "inst/wal-stream/snapshot.json", b"{}").await;
        ship(&store, "inst/wal-stream/wal/events-000001.log", b"a\n").await;

        let dir = tempfile::tempdir().unwrap();
        let mut state = FollowState::new();
        let report = follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap();

        assert_eq!(report.applied.len(), 2);
        // The layout is what the server boots from: no `wal/` directory carried
        // over from the bucket.
        assert_eq!(
            std::fs::read(dir.path().join("events-000001.log")).unwrap(),
            b"a\n"
        );
        assert!(!dir.path().join("wal").exists());
    }

    #[tokio::test]
    async fn the_snapshot_lands_before_the_segments() {
        // A replay that started after events it had not seen would silently skip
        // them, so ordering is a correctness property, not tidiness.
        let store = store();
        ship(&store, "inst/wal-stream/wal/events-000002.log", b"b\n").await;
        ship(&store, "inst/wal-stream/snapshot.json", b"{}").await;

        let dir = tempfile::tempdir().unwrap();
        let mut state = FollowState::new();
        let report = follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap();
        assert_eq!(
            report.applied.first().map(|s| s.as_str()),
            Some("snapshot.json"),
            "the snapshot must be applied first, got {:?}",
            report.applied
        );
    }

    #[tokio::test]
    async fn an_unchanged_segment_is_not_re_downloaded() {
        let store = store();
        ship(&store, "inst/wal-stream/wal/events-000001.log", b"a\n").await;
        let dir = tempfile::tempdir().unwrap();
        let mut state = FollowState::new();

        follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap();
        let second = follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap();
        assert!(second.applied.is_empty(), "nothing changed on the primary");
        assert_eq!(second.skipped, 1);
    }

    #[tokio::test]
    async fn a_grown_segment_is_downloaded_again() {
        // The active segment keeps being appended to; without re-reading it the
        // replica would freeze one segment behind forever.
        let store = store();
        ship(&store, "inst/wal-stream/wal/events-000001.log", b"a\n").await;
        let dir = tempfile::tempdir().unwrap();
        let mut state = FollowState::new();
        follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap();

        ship(&store, "inst/wal-stream/wal/events-000001.log", b"a\nb\n").await;
        let report = follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap();
        assert_eq!(report.applied, vec!["events-000001.log"]);
        assert_eq!(
            std::fs::read(dir.path().join("events-000001.log")).unwrap(),
            b"a\nb\n"
        );
    }

    #[tokio::test]
    async fn the_snapshot_is_always_re_read() {
        // It is rewritten in place rather than appended, so an unchanged length
        // does not mean unchanged content.
        let store = store();
        ship(&store, "inst/wal-stream/snapshot.json", b"{\"o\":1}").await;
        let dir = tempfile::tempdir().unwrap();
        let mut state = FollowState::new();
        follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap();

        // Same length, different content — the case a length check would miss.
        ship(&store, "inst/wal-stream/snapshot.json", b"{\"o\":9}").await;
        follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read(dir.path().join("snapshot.json")).unwrap(),
            b"{\"o\":9}"
        );
    }

    #[tokio::test]
    async fn a_restarted_replica_does_not_re_download_the_stream() {
        let store = store();
        ship(&store, "inst/wal-stream/wal/events-000001.log", b"a\n").await;
        let dir = tempfile::tempdir().unwrap();

        let mut first = FollowState::new();
        follow_once(&store, "inst/wal-stream", dir.path(), &mut first, unseal)
            .await
            .unwrap();

        // Restart: the state is rebuilt from what is on disk.
        let mut after_restart = FollowState::from_data_dir(dir.path());
        assert_eq!(after_restart.tracked(), 1);
        let report = follow_once(
            &store,
            "inst/wal-stream",
            dir.path(),
            &mut after_restart,
            unseal,
        )
        .await
        .unwrap();
        assert!(
            report.applied.iter().all(|n| n == "snapshot.json"),
            "a restarted replica must not re-download segments it already has: {:?}",
            report.applied
        );
    }

    #[tokio::test]
    async fn an_empty_prefix_is_an_error_not_silence() {
        // A replica pointed at the wrong bucket must say so, rather than sitting
        // there serving stale reads as if it were current.
        let dir = tempfile::tempdir().unwrap();
        let mut state = FollowState::new();
        let err = follow_once(&store(), "inst/wal-stream", dir.path(), &mut state, unseal)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("shipped nothing"), "{err}");
    }

    #[tokio::test]
    async fn a_tampered_object_aborts_the_pass() {
        let store = store();
        store
            .put(
                &object_store::path::Path::from("inst/wal-stream/wal/events-000001.log"),
                PutPayload::from(b"not sealed".to_vec()),
            )
            .await
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut state = FollowState::new();
        assert!(
            follow_once(&store, "inst/wal-stream", dir.path(), &mut state, unseal)
                .await
                .is_err(),
            "undecryptable bytes must abort rather than land as garbage"
        );
    }

    #[test]
    fn lag_bytes_accumulate() {
        let role = Role::replica();
        assert_eq!(role.followed_bytes(), 0);
        role.record_followed(120);
        role.record_followed(80);
        assert_eq!(role.followed_bytes(), 200);
    }
}
