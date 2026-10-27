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
    /// Hash of the snapshot last uploaded. It is rewritten in place, so its
    /// length says nothing; uploading it on every pass was ~260 000 PUTs a
    /// month at a 10 s interval for a snapshot that changed every few minutes.
    snapshot: Option<u64>,
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
        let hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        };
        // `skipped` counts segments only, so an unchanged snapshot is silent.
        if state.snapshot != Some(hash) {
            let sealed = encrypt(&bytes).context("encrypting snapshot")?;
            store
                .put(
                    &ObjectPath::from(format!("{prefix}/snapshot.json")),
                    PutPayload::from(sealed),
                )
                .await
                .context("uploading snapshot")?;
            report.bytes += bytes.len() as u64;
            state.snapshot = Some(hash);
        }
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

/// Where an instance ships its WAL and files, and where its fencing epoch
/// lives, under the remote root. One function so the shipper, the replica,
/// `promote` and `restore-remote` cannot disagree: `promote` used to claim the
/// epoch at the root while shipping checked it here, so it fenced nobody.
pub fn stream_prefix(root: &str) -> String {
    format!("{}/wal-stream", root.trim_end_matches('/'))
}

/// The data directory name of a shipped WAL object (`snapshot.json`,
/// `wal/events-*.log` → `events-*.log`), or `None` for everything else that
/// shares the prefix: the epoch, the files index and the shipped files.
pub(crate) fn wal_object_name(relative: &str) -> Option<&str> {
    if relative == "snapshot.json" {
        return Some(relative);
    }
    relative
        .strip_prefix("wal/")
        .filter(|name| name.starts_with("events-") && name.ends_with(".log"))
}

/// The data directory trees that live outside the WAL and so are shipped as
/// files: blob objects and queue messages.
const SHIPPED_TREES: &[&str] = &["blobs", "queues"];
/// The list of shipped files, so a restart knows what is already up and a
/// restore knows what exists (and so what was deleted).
const FILES_INDEX: &str = "files-index.json";
/// Objects uploaded, deleted or downloaded at once. One at a time, a real
/// remote (~100-200 ms per request) moves 5-10 files a second: a busy queue
/// outran it and the recovery point slipped from one interval to minutes.
pub const PARALLEL_REQUESTS: usize = 32;

/// What has been shipped of the blob and queue trees: relative path to
/// `(length, modified time in ns)`. Loaded from the remote index on the first
/// pass, so a restart does not re-upload everything.
#[derive(Debug, Default)]
pub struct FileShipState {
    loaded: bool,
    shipped: std::collections::BTreeMap<String, (u64, u128)>,
}

/// Every file in the shipped trees, with its length and modified time.
///
/// Temp files are in-progress writes and are skipped; so are queue leases,
/// which are ephemeral: a restored message without one is simply visible.
fn local_files(data_dir: &Path) -> Result<std::collections::BTreeMap<String, (u64, u128)>> {
    let mut out = std::collections::BTreeMap::new();
    for tree in SHIPPED_TREES {
        let root = data_dir.join(tree);
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                let Ok(meta) = entry.metadata() else {
                    continue; // removed between the listing and the stat
                };
                if meta.is_dir() {
                    stack.push(path);
                    continue;
                }
                if name.starts_with(".tmp-") || name.ends_with(".lease") {
                    continue;
                }
                let relative = path
                    .strip_prefix(data_dir)
                    .map_err(|_| anyhow!("file escaped the data dir"))?
                    .to_string_lossy()
                    .replace('\\', "/");
                let modified = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos());
                out.insert(relative, (meta.len(), modified));
            }
        }
    }
    Ok(out)
}

/// Ship the blob and queue trees: upload what is new or changed, delete what
/// is gone, then the index. Returns how many objects changed.
///
/// Blobs and queue messages are not in the WAL, so shipping only the WAL left
/// them recoverable only as of the last full backup, hours behind. With this
/// the recovery point is the shipping interval for them too.
pub async fn ship_files(
    store: &Arc<dyn ObjectStore>,
    prefix: &str,
    data_dir: &Path,
    state: &mut FileShipState,
    encrypt: impl Fn(&[u8]) -> Result<Vec<u8>>,
    decrypt: impl Fn(&[u8]) -> Result<Vec<u8>>,
) -> Result<usize> {
    let prefix = prefix.trim_end_matches('/');
    let index_key = ObjectPath::from(format!("{prefix}/{FILES_INDEX}"));
    if !state.loaded {
        if let Ok(found) = store.get(&index_key).await {
            let plain = decrypt(&found.bytes().await?)?;
            state.shipped = serde_json::from_slice(&plain)?;
        }
        state.loaded = true;
    }

    use futures_util::StreamExt as _;
    let local = local_files(data_dir)?;
    let mut changed = 0;
    let mut first_error = None;

    let encrypt = &encrypt;
    let pending: Vec<(&String, &(u64, u128))> = local
        .iter()
        .filter(|(relative, stamp)| state.shipped.get(*relative) != Some(*stamp))
        .collect();
    let mut uploads = futures_util::stream::iter(
        pending
            .into_iter()
            .map(|(relative, stamp)| async move {
                // Gone since the listing (acked, deleted): the next pass deletes it.
                let Ok(bytes) = std::fs::read(data_dir.join(relative)) else {
                    return Ok(None);
                };
                let sealed = encrypt(&bytes).with_context(|| format!("encrypting {relative}"))?;
                store
                    .put(
                        &ObjectPath::from(format!("{prefix}/files/{relative}")),
                        PutPayload::from(sealed),
                    )
                    .await
                    .with_context(|| format!("uploading {relative}"))?;
                Ok::<_, anyhow::Error>(Some((relative, stamp)))
            })
            .collect::<Vec<_>>(),
    )
    .buffer_unordered(PARALLEL_REQUESTS);
    // What did go up is recorded even when something else failed, so the
    // retry only sends the rest.
    let mut done_ok = Vec::new();
    while let Some(done) = uploads.next().await {
        match done {
            Ok(Some(file)) => done_ok.push(file),
            Ok(None) => {}
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    drop(uploads);
    for (relative, stamp) in done_ok {
        state.shipped.insert(relative.clone(), *stamp);
        changed += 1;
    }

    let gone: Vec<String> = state
        .shipped
        .keys()
        .filter(|k| !local.contains_key(*k))
        .cloned()
        .collect();
    let mut deletes = futures_util::stream::iter(
        gone.into_iter()
            .map(|relative| async move {
                let key = ObjectPath::from(format!("{prefix}/files/{relative}"));
                match store.delete(&key).await {
                    Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(relative),
                    Err(e) => Err(e).with_context(|| format!("deleting {relative}")),
                }
            })
            .collect::<Vec<_>>(),
    )
    .buffer_unordered(PARALLEL_REQUESTS);
    let mut removed = Vec::new();
    while let Some(done) = deletes.next().await {
        match done {
            Ok(relative) => removed.push(relative),
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    drop(deletes);
    for relative in removed {
        state.shipped.remove(&relative);
        changed += 1;
    }
    if let Some(e) = first_error {
        // No index this pass: it must never list a file that is not up.
        return Err(e);
    }
    if changed > 0 {
        let index = encrypt(&serde_json::to_vec(&state.shipped)?)?;
        store
            .put(&index_key, PutPayload::from(index))
            .await
            .context("uploading the files index")?;
    }
    Ok(changed)
}

/// Bring the blob and queue trees in `data_dir` to what was last shipped:
/// download every indexed file, and remove local files the index does not
/// list (deleted after the backup the directory was restored from).
pub async fn restore_files(
    store: &Arc<dyn ObjectStore>,
    prefix: &str,
    data_dir: &Path,
    decrypt: impl Fn(&[u8]) -> Result<Vec<u8>>,
) -> Result<usize> {
    let prefix = prefix.trim_end_matches('/');
    let index_key = ObjectPath::from(format!("{prefix}/{FILES_INDEX}"));
    let Ok(found) = store.get(&index_key).await else {
        return Ok(0);
    };
    let index: std::collections::BTreeMap<String, (u64, u128)> =
        serde_json::from_slice(&decrypt(&found.bytes().await?)?)?;
    {
        use futures_util::StreamExt as _;
        let decrypt = &decrypt;
        let mut downloads = futures_util::stream::iter(
            index
                .keys()
                .map(|relative| async move {
                    let sealed = store
                        .get(&ObjectPath::from(format!("{prefix}/files/{relative}")))
                        .await
                        .with_context(|| format!("downloading {relative}"))?
                        .bytes()
                        .await?;
                    let target = data_dir.join(relative);
                    if let Some(parent) = target.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(&target, decrypt(&sealed)?)?;
                    Ok::<_, anyhow::Error>(())
                })
                .collect::<Vec<_>>(),
        )
        .buffer_unordered(PARALLEL_REQUESTS);
        while let Some(done) = downloads.next().await {
            done?;
        }
    }
    for relative in local_files(data_dir)?.keys() {
        if !index.contains_key(relative) {
            std::fs::remove_file(data_dir.join(relative))?;
        }
    }
    Ok(index.len())
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
        let Some(name) = wal_object_name(relative) else {
            continue; // epoch and shipped files: see `restore_files`
        };

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

    let stream = stream_prefix(&target.prefix);
    let data_dir = PathBuf::from(data_dir);
    // Blobs and queues ship on their own task: a long pass of files (a burst
    // of uploads, a slow remote) must never hold the WAL back, whose interval
    // is the promise.
    {
        let store = target.store.clone();
        let stream = stream.clone();
        let data_dir = data_dir.clone();
        tokio::spawn(async move {
            let secrets = crate::crypto::SecretBox::from_env();
            let mut files = FileShipState::default();
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                // Skipped while fenced: another node owns the prefix now.
                if !crate::fencing::standing(&store, &stream, &data_dir)
                    .await
                    .is_ok_and(|s| matches!(s, crate::fencing::Standing::Current))
                {
                    continue;
                }
                let started = std::time::Instant::now();
                match ship_files(
                    &store,
                    &stream,
                    &data_dir,
                    &mut files,
                    |bytes| secrets.encrypt_bytes(bytes),
                    |bytes| secrets.decrypt_bytes(bytes),
                )
                .await
                {
                    Ok(0) => {}
                    Ok(n) => tracing::info!(
                        files = n,
                        lag_ms = started.elapsed().as_millis() as u64,
                        "blobs and queues shipped"
                    ),
                    Err(e) => tracing::error!("file shipping failed, will retry: {e}"),
                }
            }
        });
    }

    tokio::spawn(async move {
        let secrets = crate::crypto::SecretBox::from_env();
        let mut state = ShipState::new();
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval));
        loop {
            ticker.tick().await;
            let started = std::time::Instant::now();
            match ship_once(&target.store, &stream, &data_dir, &mut state, |bytes| {
                secrets.encrypt_bytes(bytes)
            })
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

    /// `promote` claims at `stream_prefix(root)`, shipping checks the prefix it
    /// is given: the old primary must find itself fenced there.
    #[tokio::test]
    async fn a_promotion_fences_the_old_primary() {
        let dir = data_dir();
        let store = store();
        let stream = stream_prefix("inst/");
        let mut state = ShipState::new();
        ship_once(&store, &stream, dir.path(), &mut state, seal)
            .await
            .unwrap();
        crate::fencing::claim_next_epoch(&store, &stream)
            .await
            .unwrap();
        std::fs::write(
            dir.path().join("events-000002.log"),
            b"b
",
        )
        .unwrap();
        let report = ship_once(&store, &stream, dir.path(), &mut state, seal)
            .await
            .unwrap();
        assert!(
            report.fenced,
            "the old primary kept shipping after a promotion"
        );
        assert!(report.uploaded.is_empty());
    }

    #[tokio::test]
    async fn files_ship_and_restore_including_deletions() {
        let dir = data_dir();
        let blobs = dir.path().join("blobs").join("b1");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("keep"), b"k").unwrap();
        std::fs::write(blobs.join("gone"), b"g").unwrap();
        std::fs::write(blobs.join(".tmp-x"), b"partial").unwrap();
        let queue = dir.path().join("queues").join("q");
        std::fs::create_dir_all(&queue).unwrap();
        std::fs::write(queue.join("m1.json"), b"{}").unwrap();
        std::fs::write(queue.join("m1.lease"), b"{}").unwrap();

        let store = store();
        let mut state = FileShipState::default();
        let shipped = ship_files(&store, "inst", dir.path(), &mut state, seal, unseal)
            .await
            .unwrap();
        assert_eq!(shipped, 3, "temp files and leases are not shipped");
        std::fs::remove_file(blobs.join("gone")).unwrap();
        // A restart starts from the remote index, not from scratch.
        let mut restarted = FileShipState::default();
        let changed = ship_files(&store, "inst", dir.path(), &mut restarted, seal, unseal)
            .await
            .unwrap();
        assert_eq!(changed, 1, "only the deletion is new");

        // The target holds the old backup's copy of the deleted blob.
        let target = tempfile::tempdir().unwrap();
        let old = target.path().join("blobs").join("b1");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("gone"), b"g").unwrap();
        let restored = restore_files(&store, "inst", target.path(), unseal)
            .await
            .unwrap();
        assert_eq!(restored, 2);
        assert_eq!(std::fs::read(old.join("keep")).unwrap(), b"k");
        assert!(!old.join("gone").exists(), "a deleted blob came back");
        assert!(target.path().join("queues/q/m1.json").exists());
    }

    /// The epoch and the shipped files share the prefix with the WAL; restore
    /// must lay down only snapshot and segments.
    #[tokio::test]
    async fn restore_ignores_the_epoch_and_shipped_files() {
        let dir = data_dir();
        std::fs::create_dir_all(dir.path().join("blobs/b")).unwrap();
        std::fs::write(dir.path().join("blobs/b/o"), b"o").unwrap();
        let store = store();
        ship_once(&store, "inst", dir.path(), &mut ShipState::new(), seal)
            .await
            .unwrap();
        ship_files(
            &store,
            "inst",
            dir.path(),
            &mut FileShipState::default(),
            seal,
            unseal,
        )
        .await
        .unwrap();
        crate::fencing::claim_next_epoch(&store, "inst")
            .await
            .unwrap();

        let target = tempfile::tempdir().unwrap();
        let count = restore_shipped(&store, "inst", target.path(), unseal)
            .await
            .unwrap();
        assert_eq!(count, 2, "snapshot and one segment");
        assert!(!target.path().join("files").exists());
        assert!(!target.path().join(FILES_INDEX).exists());
    }

    /// A pass with nothing new uploads nothing: not even the snapshot, which
    /// used to go up on every pass.
    #[tokio::test]
    async fn an_idle_pass_uploads_nothing() {
        let dir = data_dir();
        let store = store();
        let mut state = ShipState::new();
        let first = ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();
        assert!(first.bytes > 0);
        let idle = ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();
        assert_eq!(idle.bytes, 0, "an idle pass uploaded {} bytes", idle.bytes);
        assert!(idle.uploaded.is_empty());

        std::fs::write(dir.path().join("snapshot.json"), b"{\"offset\":9}").unwrap();
        let changed = ship_once(&store, "inst", dir.path(), &mut state, seal)
            .await
            .unwrap();
        assert!(changed.bytes > 0, "a changed snapshot must go up");
    }
}
