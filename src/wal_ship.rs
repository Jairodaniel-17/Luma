//! Continuous WAL shipping to object storage.
//!
//! W2.1 of `docs/PLAN-MAESTRO.md`. `luma backup` gives a point-in-time copy
//! every few hours; everything written since the last one is on one disk. WAL
//! shipping closes that window without needing a second server: closed segments
//! are uploaded as they are sealed and the growing one is re-uploaded on an
//! interval, so the recovery point is bounded by that interval rather than by
//! the backup schedule.
//!
//! ## What "shipped" means here
//!
//! A snapshot plus the chain of segments after it reconstructs state up to the
//! last shipped byte. That is the same reconstruction the server already does on
//! startup, which is why this ships raw segments rather than inventing a
//! separate replication format: the restore path is the boot path, already
//! covered by the crash-recovery matrix.
//!
//! ## What it is not
//!
//! Not replication. Nothing follows the stream and applies it live — that is
//! W2.2, the read replica. This is disaster recovery: the bucket holds enough to
//! rebuild the instance elsewhere, with a stated worst-case loss.

use anyhow::{anyhow, Context, Result};
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A WAL segment on disk, with the size it had when we looked.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub name: String,
    pub path: PathBuf,
    pub len: u64,
}

/// List the WAL segments in a data directory, oldest first.
///
/// Sorted by name, which is chronological because segment numbers are
/// zero-padded — the same property the local backup pruning relies on.
pub fn list_segments(data_dir: &Path) -> Result<Vec<Segment>> {
    let mut segments = Vec::new();
    let Ok(entries) = std::fs::read_dir(data_dir) else {
        return Ok(segments);
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !(name.starts_with("events-") && name.ends_with(".log")) {
            continue;
        }
        let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
        segments.push(Segment {
            name,
            path: entry.path(),
            len,
        });
    }
    segments.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(segments)
}

/// Tracks what has already been shipped, so an interval that changed nothing
/// costs a directory listing instead of a re-upload of the whole WAL.
///
/// Keyed by segment name, holding the byte length last uploaded. A segment is
/// re-uploaded when it has grown; a segment that has not changed is skipped.
/// Length is enough because the WAL is append-only — a segment never shrinks or
/// gets rewritten in place, so equal length means equal content.
#[derive(Debug, Default)]
pub struct ShipState {
    shipped: HashMap<String, u64>,
}

impl ShipState {
    pub fn new() -> Self {
        Self::default()
    }

    fn needs_shipping(&self, segment: &Segment) -> bool {
        match self.shipped.get(&segment.name) {
            Some(&len) => segment.len > len,
            None => true,
        }
    }

    fn record(&mut self, segment: &Segment) {
        self.shipped.insert(segment.name.clone(), segment.len);
    }

    /// Segments shipped so far, for metrics and tests.
    pub fn tracked(&self) -> usize {
        self.shipped.len()
    }
}

/// Result of one shipping pass.
#[derive(Debug, Default, PartialEq)]
pub struct ShipReport {
    /// Set when this pass shipped nothing because another node owns the prefix.
    ///
    /// Distinct from "nothing to ship": the caller needs to tell a quiet node
    /// from a fenced one, and an empty report would look identical.
    pub fenced: bool,
    /// Segments uploaded this pass.
    pub uploaded: Vec<String>,
    /// Bytes uploaded this pass.
    pub bytes: u64,
    /// Segments that had not changed.
    pub skipped: usize,
}

/// Ship every new or grown WAL segment, plus the current snapshot.
///
/// The snapshot goes **first**. A restore replays segments on top of a
/// snapshot, so a bucket holding segments newer than its snapshot is
/// recoverable, while one holding a snapshot newer than its segments is not:
/// the replay would start after events that never made it up.
pub async fn ship_once(
    store: &Arc<dyn ObjectStore>,
    prefix: &str,
    data_dir: &Path,
    state: &mut ShipState,
    encrypt: impl Fn(&[u8]) -> Result<Vec<u8>>,
) -> Result<ShipReport> {
    let prefix = prefix.trim_end_matches('/');
    let mut report = ShipReport::default();

    // Before writing anything: if somebody promoted a replica, this node no
    // longer owns the prefix. Two nodes shipping into one prefix interleave
    // their segments, and replay then reads two histories spliced together —
    // not a mess that can be untangled afterwards.
    //
    // Checked here rather than at startup because the promotion can happen at
    // any moment, and the whole point is that a node which *was* the primary
    // finds out.
    match crate::fencing::standing(store, prefix, data_dir).await? {
        crate::fencing::Standing::Current => {}
        crate::fencing::Standing::Fenced { local, remote } => {
            report.fenced = true;
            tracing::error!(
                local_epoch = local,
                remote_epoch = remote,
                "WAL shipping stopped: another node claimed a later epoch for this \
                 prefix. This node must not write again — stop it, or re-seed it \
                 from the new primary."
            );
            return Ok(report);
        }
    }

    let snapshot = data_dir.join("snapshot.json");
    if snapshot.exists() {
        let bytes = std::fs::read(&snapshot)?;
        let sealed = encrypt(&bytes).context("encrypting snapshot")?;
        store
            .put(
                &ObjectPath::from(format!("{prefix}/snapshot.json")),
                PutPayload::from(sealed),
            )
            .await
            .context("uploading snapshot")?;
        report.bytes += bytes.len() as u64;
    }

    for segment in list_segments(data_dir)? {
        if !state.needs_shipping(&segment) {
            report.skipped += 1;
            continue;
        }
        // Read after stat: the segment may have grown in between, which is
        // harmless — we record the length actually read, so the next pass ships
        // the remainder rather than assuming it is already up.
        let bytes = std::fs::read(&segment.path)
            .with_context(|| format!("reading {}", segment.path.display()))?;
        let actual = Segment {
            len: bytes.len() as u64,
            ..segment.clone()
        };
        let sealed = encrypt(&bytes).with_context(|| format!("encrypting {}", segment.name))?;
        store
            .put(
                &ObjectPath::from(format!("{prefix}/wal/{}", segment.name)),
                PutPayload::from(sealed),
            )
            .await
            .with_context(|| format!("uploading {}", segment.name))?;
        state.record(&actual);
        report.bytes += actual.len;
        report.uploaded.push(segment.name);
    }

    Ok(report)
}

/// Reconstruct a data directory from a shipped prefix.
///
/// Writes the snapshot and every segment back, which is exactly the layout the
/// server boots from — so recovery is "download, then start", with no separate
/// apply step to get wrong.
pub async fn restore_shipped(
    store: &Arc<dyn ObjectStore>,
    prefix: &str,
    data_dir: &Path,
    decrypt: impl Fn(&[u8]) -> Result<Vec<u8>>,
) -> Result<usize> {
    let prefix = prefix.trim_end_matches('/').to_string();
    std::fs::create_dir_all(data_dir)?;

    let keys = {
        use futures_util::StreamExt as _;
        let mut stream = store.list(Some(&ObjectPath::from(prefix.clone())));
        let mut keys = Vec::new();
        while let Some(meta) = stream.next().await {
            keys.push(meta?.location);
        }
        keys
    };
    if keys.is_empty() {
        return Err(anyhow!("nothing shipped under `{prefix}`"));
    }

    let mut written = 0;
    for key in keys {
        let relative = key
            .as_ref()
            .strip_prefix(&format!("{prefix}/"))
            .ok_or_else(|| anyhow!("object `{key}` is not under `{prefix}`"))?;
        // `wal/events-000001.log` restores as `events-000001.log`: the prefix
        // only groups objects in the bucket, it is not part of the layout.
        let name = relative.strip_prefix("wal/").unwrap_or(relative);

        let sealed = store.get(&key).await?.bytes().await?;
        let plain = decrypt(&sealed).with_context(|| format!("decrypting {name}"))?;
        std::fs::write(data_dir.join(name), plain)?;
        written += 1;
    }

    tracing::info!(prefix, objects = written, "shipped WAL restored");
    Ok(written)
}

/// Spawn the continuous shipping task. No-op unless a remote is configured and
/// `wal_ship_interval_secs` is non-zero.
///
/// The interval **is** the recovery point objective: a machine lost between
/// ticks loses at most one interval of writes. That is the number to state in a
/// runbook, so it is logged at startup rather than left to be inferred.
pub fn spawn(config: crate::config::Config) {
    let interval = config.wal_ship_interval_secs;
    if interval == 0 {
        return;
    }
    let target = match crate::backup_remote::store_from_config(&config) {
        Ok(Some(target)) => target,
        Ok(None) => return,
        Err(e) => {
            tracing::error!("WAL shipping disabled, remote misconfigured: {e}");
            return;
        }
    };
    let Some(data_dir) = config.data_dir.clone() else {
        tracing::warn!("WAL shipping needs a data_dir; disabled");
        return;
    };

    tracing::info!(
        interval_secs = interval,
        prefix = %target.prefix,
        "WAL shipping enabled; worst-case data loss on host failure is one interval"
    );

    tokio::spawn(async move {
        let secrets = crate::crypto::SecretBox::from_env();
        let mut state = ShipState::new();
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        let data_dir = PathBuf::from(data_dir);
        loop {
            ticker.tick().await;
            let started = std::time::Instant::now();
            match ship_once(
                &target.store,
                &format!("{}/wal-stream", target.prefix),
                &data_dir,
                &mut state,
                |bytes| secrets.encrypt_bytes(bytes),
            )
            .await
            {
                Ok(report) if report.uploaded.is_empty() => {
                    tracing::debug!(skipped = report.skipped, "WAL shipping: nothing new")
                }
                Ok(report) => tracing::info!(
                    segments = report.uploaded.len(),
                    bytes = report.bytes,
                    lag_ms = started.elapsed().as_millis() as u64,
                    "WAL shipped"
                ),
                // Never fatal: the local WAL is intact and the next tick retries.
                // Killing the task on a transient network error would silently
                // stop shipping for the lifetime of the process.
                Err(e) => tracing::error!("WAL shipping failed, will retry: {e}"),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

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

    fn data_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("snapshot.json"), b"{\"offset\":0}").unwrap();
        std::fs::write(dir.path().join("events-000001.log"), b"a\n").unwrap();
        dir
    }

    #[test]
    fn segments_list_oldest_first_and_ignore_other_files() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "events-000003.log",
            "events-000001.log",
            "events-000002.log",
        ] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }
        // Zero padding is what makes lexicographic order chronological.
        std::fs::write(dir.path().join("snapshot.json"), b"{}").unwrap();
        // The KV projection is a *directory* (`state.lsm`), so this decoy also
        // covers the case a plain file would not: a dir entry that is not a file.
        std::fs::create_dir(dir.path().join("state.lsm")).unwrap();

        let names: Vec<_> = list_segments(dir.path())
            .unwrap()
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(
            names,
            vec![
                "events-000001.log",
                "events-000002.log",
                "events-000003.log"
            ]
        );
    }

    #[tokio::test]
    async fn a_first_pass_ships_everything() {
        let dir = data_dir();
        let store = store();
        let mut state = ShipState::new();

        let report = ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();
        assert_eq!(report.uploaded, vec!["events-000001.log"]);
        assert_eq!(report.skipped, 0);
        assert!(store
            .head(&ObjectPath::from("inst/snapshot.json"))
            .await
            .is_ok());
        assert!(store
            .head(&ObjectPath::from("inst/wal/events-000001.log"))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn an_unchanged_segment_is_not_re_uploaded() {
        // The whole point of tracking lengths: a quiet interval must cost a
        // directory listing, not a re-upload of the entire WAL.
        let dir = data_dir();
        let store = store();
        let mut state = ShipState::new();

        ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();
        let second = ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();

        assert!(
            second.uploaded.is_empty(),
            "nothing changed, nothing to ship"
        );
        assert_eq!(second.skipped, 1);
    }

    #[tokio::test]
    async fn a_grown_segment_is_shipped_again() {
        let dir = data_dir();
        let store = store();
        let mut state = ShipState::new();
        ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();

        // The active segment keeps being appended to, so it must be re-shipped
        // for the recovery point to advance at all.
        std::fs::write(dir.path().join("events-000001.log"), b"a\nb\n").unwrap();
        let report = ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();
        assert_eq!(report.uploaded, vec!["events-000001.log"]);

        let stored = store
            .get(&ObjectPath::from("inst/wal/events-000001.log"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(unseal(&stored).unwrap(), b"a\nb\n");
    }

    #[tokio::test]
    async fn a_new_segment_is_picked_up_after_rotation() {
        let dir = data_dir();
        let store = store();
        let mut state = ShipState::new();
        ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();

        std::fs::write(dir.path().join("events-000002.log"), b"c\n").unwrap();
        let report = ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();
        assert_eq!(report.uploaded, vec!["events-000002.log"]);
        assert_eq!(report.skipped, 1, "the sealed one must not be re-uploaded");
        assert_eq!(state.tracked(), 2);
    }

    #[tokio::test]
    async fn shipped_bytes_are_encrypted() {
        let dir = data_dir();
        let store = store();
        let mut state = ShipState::new();
        ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();

        let raw = store
            .get(&ObjectPath::from("inst/wal/events-000001.log"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert!(raw.starts_with(b"S:"), "WAL reached the bucket unencrypted");
    }

    #[tokio::test]
    async fn restore_rebuilds_a_bootable_data_dir() {
        let dir = data_dir();
        std::fs::write(dir.path().join("events-000002.log"), b"c\n").unwrap();
        let store = store();
        let mut state = ShipState::new();
        ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();

        let target = tempfile::tempdir().unwrap();
        let count = restore_shipped(&store, "inst", target.path(), unseal)
            .await
            .unwrap();
        assert_eq!(count, 3);

        // The layout is what the server boots from: snapshot and segments at the
        // top level, no `wal/` directory carried over from the bucket.
        assert_eq!(
            std::fs::read(target.path().join("snapshot.json")).unwrap(),
            b"{\"offset\":0}"
        );
        assert_eq!(
            std::fs::read(target.path().join("events-000002.log")).unwrap(),
            b"c\n"
        );
        assert!(!target.path().join("wal").exists());
    }

    #[tokio::test]
    async fn restore_refuses_an_empty_prefix() {
        let target = tempfile::tempdir().unwrap();
        let err = restore_shipped(&store(), "inst", target.path(), unseal)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("nothing shipped"), "got: {err}");
    }

    #[tokio::test]
    async fn a_data_dir_with_no_wal_ships_nothing_and_does_not_fail() {
        // A brand-new instance has no segments yet; the task must not error on
        // every tick until the first write lands.
        let dir = tempfile::tempdir().unwrap();
        let mut state = ShipState::new();
        let report = ship_once(&store(), "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();
        assert_eq!(report, ShipReport::default());
    }
}
