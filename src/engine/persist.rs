use crate::engine::events::EventRecord;
use crate::engine::EventBus;
use crate::vector::VectorStore;
use crc32fast::Hasher;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// PR5: WAL sync mode.
#[derive(Clone, Debug, PartialEq)]
pub enum WalSyncMode {
    /// Fsync every write (safe, lower throughput).
    PerWrite,
    /// Buffer writes; flush on count threshold or explicit flush (higher throughput).
    Group {
        batch_size: usize,
        flush_interval_ms: u64,
    },
}

impl WalSyncMode {
    pub fn from_config(config: &crate::config::Config) -> Self {
        if config.wal_sync_mode.trim().eq_ignore_ascii_case("group") {
            Self::Group {
                batch_size: config.wal_batch_size.max(1),
                flush_interval_ms: config.wal_flush_interval_ms.max(1),
            }
        } else {
            Self::PerWrite
        }
    }
}

#[derive(Clone)]
pub struct Persist(Arc<Inner>);

/// State for the WAL group-commit buffer.
struct GroupBuffer {
    /// Serialized JSON lines buffered for flushing
    pending: Vec<Vec<u8>>,
    last_flush_ms: u64,
}

struct Inner {
    dir: PathBuf,
    wal_lock: Mutex<()>,
    segment_max_bytes: u64,
    retention_segments: usize,
    current_segment: Mutex<u64>,
    /// PR5: group-commit buffer (None when mode is PerWrite)
    group_buffer: Option<Mutex<GroupBuffer>>,
    sync_mode: WalSyncMode,
    /// Highest offset durably applied to all derived stores (redb, vectors).
    /// When gating is enabled, retention only prunes segments lying entirely
    /// at/below this. See `set_durable_floor`.
    durable_floor: AtomicU64,
    /// Whether floor gating is active. Set the first time `set_durable_floor` is
    /// called (i.e. when a derived store uses deferred durability). While gating
    /// is on, a floor of 0 means "prune nothing" (nothing is durable yet) — the
    /// opposite of the legacy count-only pruning used when gating is off.
    floor_gated: AtomicBool,
}

const WAL_RECORD_VERSION: u8 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub last_offset: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ReplayReport {
    pub applied: usize,
    pub duplicates_skipped: usize,
    pub corrupted_records: usize,
    pub gap_detected: bool,
    pub highest_offset_seen: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WalEnvelope {
    version: u8,
    offset: u64,
    crc32: u32,
    event: EventRecord,
}

impl Persist {
    pub fn new(
        dir: impl AsRef<Path>,
        segment_max_bytes: u64,
        retention_segments: usize,
    ) -> std::io::Result<Self> {
        Self::new_with_mode(
            dir,
            segment_max_bytes,
            retention_segments,
            WalSyncMode::PerWrite,
        )
    }

    pub fn new_with_mode(
        dir: impl AsRef<Path>,
        segment_max_bytes: u64,
        retention_segments: usize,
        sync_mode: WalSyncMode,
    ) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let current_segment = find_latest_segment_id(&dir).unwrap_or(1);
        let group_buffer = if matches!(sync_mode, WalSyncMode::Group { .. }) {
            Some(Mutex::new(GroupBuffer {
                pending: Vec::new(),
                last_flush_ms: now_ms(),
            }))
        } else {
            None
        };
        Ok(Self(Arc::new(Inner {
            dir,
            wal_lock: Mutex::new(()),
            segment_max_bytes: segment_max_bytes.max(1024 * 1024),
            retention_segments: retention_segments.max(1),
            current_segment: Mutex::new(current_segment),
            group_buffer,
            sync_mode,
            durable_floor: AtomicU64::new(0),
            floor_gated: AtomicBool::new(false),
        })))
    }

    /// Append a single event to the WAL.
    /// In PerWrite mode: fsync immediately.
    /// In Group mode: buffer until batch_size or flush_interval_ms.
    pub fn append_event(&self, event: &EventRecord) -> std::io::Result<()> {
        let line = encode_wal_record(event)?;

        if let Some(buf) = &self.0.group_buffer {
            let should_flush = {
                let mut guard = buf.lock();
                guard.pending.push(line);
                let (batch_size, flush_interval_ms) = if let WalSyncMode::Group {
                    batch_size,
                    flush_interval_ms,
                } = self.0.sync_mode
                {
                    (batch_size, flush_interval_ms)
                } else {
                    (64, 10)
                };
                guard.pending.len() >= batch_size
                    || now_ms().saturating_sub(guard.last_flush_ms) >= flush_interval_ms
            };
            if should_flush {
                self.flush_group_buffer()?;
            }
            return Ok(());
        }

        // PerWrite mode: fsync each event
        self.write_lines_to_wal(std::slice::from_ref(&line), true)
    }

    /// Flush any buffered events to disk (group commit mode only).
    /// Always called before snapshot rotation to ensure durability.
    pub fn flush_buffer(&self) -> std::io::Result<()> {
        if self.0.group_buffer.is_some() {
            self.flush_group_buffer()?;
        }
        Ok(())
    }

    pub fn group_flush_interval(&self) -> Option<std::time::Duration> {
        match self.0.sync_mode {
            WalSyncMode::Group {
                flush_interval_ms, ..
            } => Some(std::time::Duration::from_millis(flush_interval_ms)),
            WalSyncMode::PerWrite => None,
        }
    }

    fn flush_group_buffer(&self) -> std::io::Result<()> {
        let Some(buf) = &self.0.group_buffer else {
            return Ok(());
        };
        let lines: Vec<Vec<u8>> = {
            let mut guard = buf.lock();
            if guard.pending.is_empty() {
                return Ok(());
            }
            let drained = std::mem::take(&mut guard.pending);
            guard.last_flush_ms = now_ms();
            drained
        };
        self.write_lines_to_wal(&lines, true)
    }

    /// Write a batch of serialized JSON lines to the current WAL segment.
    fn write_lines_to_wal(&self, lines: &[Vec<u8>], sync: bool) -> std::io::Result<()> {
        let _g = self.0.wal_lock.lock();

        let mut seg = *self.0.current_segment.lock();
        let mut path = self.segment_path(seg);
        ensure_file_exists(&path)?;

        // Estimate total bytes to write
        let total_bytes: u64 = lines.iter().map(|l| l.len() as u64 + 1).sum();
        let current_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if current_size.saturating_add(total_bytes) > self.0.segment_max_bytes {
            seg = seg.saturating_add(1);
            *self.0.current_segment.lock() = seg;
            path = self.segment_path(seg);
            ensure_file_exists(&path)?;
            // New segment file created on rotation: make its directory entry durable.
            fsync_dir(&self.0.dir)?;
        }

        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        for line in lines {
            file.write_all(line)?;
            file.write_all(b"\n")?;
        }
        file.flush()?;
        if sync {
            file.sync_data()?;
        }

        self.enforce_retention_locked(seg)?;
        Ok(())
    }

    pub fn load_snapshot(&self) -> std::io::Result<Option<Snapshot>> {
        let path = self.snapshot_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(path)?;
        let snap = serde_json::from_slice(&bytes)?;
        Ok(Some(snap))
    }

    pub fn write_snapshot_and_rotate(&self, snapshot: &Snapshot) -> std::io::Result<()> {
        // PR5: flush any buffered events before rotating
        self.flush_buffer()?;

        let _g = self.0.wal_lock.lock();

        let tmp = self.0.dir.join("snapshot.json.tmp");
        let mut f = File::create(&tmp)?;
        serde_json::to_writer_pretty(&mut f, snapshot)?;
        f.flush()?;
        f.sync_data()?;
        drop(f);
        std::fs::rename(tmp, self.snapshot_path())?;
        // Durably record the snapshot rename in the directory entry.
        fsync_dir(&self.0.dir)?;

        let seg = {
            let mut current = self.0.current_segment.lock();
            *current = current.saturating_add(1);
            *current
        };
        let path = self.segment_path(seg);
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        f.flush()?;
        f.sync_data()?;
        // Durably record the new segment file in the directory entry.
        fsync_dir(&self.0.dir)?;

        self.enforce_retention_locked(seg)?;
        Ok(())
    }

    pub fn replay_wal_since(
        &self,
        since_offset: u64,
        state: &crate::engine::state::StateStore,
        vectors: &VectorStore,
        events: &EventBus,
    ) -> std::io::Result<ReplayReport> {
        self.for_each_decoded_event_since(since_offset, |ev, _| {
            apply_event(state, vectors, &ev);
            events.set_next_offset(ev.offset.saturating_add(1));
            Ok(true)
        })
    }

    pub fn list_segments(&self) -> Vec<PathBuf> {
        list_segments_sorted(&self.0.dir)
    }

    pub fn for_each_event_since<F>(&self, since_offset: u64, mut f: F) -> std::io::Result<()>
    where
        F: FnMut(EventRecord) -> bool,
    {
        let _ = self.for_each_decoded_event_since(since_offset, |ev, _| Ok(f(ev)))?;
        Ok(())
    }

    pub fn try_for_each_event_since<F>(
        &self,
        since_offset: u64,
        mut f: F,
    ) -> std::io::Result<ReplayReport>
    where
        F: FnMut(EventRecord) -> std::io::Result<bool>,
    {
        self.for_each_decoded_event_since(since_offset, |ev, _| f(ev))
    }

    pub fn replay_report(&self, since_offset: u64) -> std::io::Result<ReplayReport> {
        self.for_each_decoded_event_since(since_offset, |_ev, _| Ok(true))
    }

    pub fn earliest_persisted_offset(&self) -> std::io::Result<Option<u64>> {
        let mut earliest = None;
        let _ = self.for_each_decoded_event_since(0, |ev, _| {
            earliest = Some(ev.offset);
            Ok(false)
        })?;
        Ok(earliest)
    }

    fn segment_path(&self, seg: u64) -> PathBuf {
        self.0.dir.join(format!("events-{seg:06}.log"))
    }

    fn snapshot_path(&self) -> PathBuf {
        self.0.dir.join("snapshot.json")
    }

    /// Records the highest offset durably applied to every derived store (redb,
    /// vectors). WAL segments are pruned only when they lie entirely at/below
    /// this floor, so retention can never delete a record a derived store hasn't
    /// persisted yet — required now that redb uses Eventual durability. A floor
    /// of 0 keeps the legacy count-only behavior.
    pub fn set_durable_floor(&self, offset: u64) {
        self.0.durable_floor.store(offset, Ordering::Relaxed);
        self.0.floor_gated.store(true, Ordering::Relaxed);
    }

    fn enforce_retention_locked(&self, current_seg: u64) -> std::io::Result<()> {
        let keep = self.0.retention_segments;
        let start_keep = current_seg.saturating_sub(keep as u64).saturating_add(1);
        let gated = self.0.floor_gated.load(Ordering::Relaxed);
        let floor = self.0.durable_floor.load(Ordering::Relaxed);
        for path in list_segments_sorted(&self.0.dir) {
            if let Some(seg) = parse_segment_id(&path) {
                if seg < start_keep {
                    if !gated {
                        // Legacy (no deferred-durability store): prune by count.
                        let _ = std::fs::remove_file(path);
                    } else if segment_max_offset(&path).is_some_and(|max| max <= floor) {
                        // Gated: every record in this segment is durably applied.
                        // (floor == 0 => nothing durable yet => keep all.)
                        let _ = std::fs::remove_file(path);
                    }
                    // Otherwise keep: not durable yet (or unreadable) — never drop
                    // records a derived store still needs to rebuild from.
                }
            }
        }
        Ok(())
    }

    fn for_each_decoded_event_since<F>(
        &self,
        since_offset: u64,
        mut f: F,
    ) -> std::io::Result<ReplayReport>
    where
        F: FnMut(EventRecord, &mut ReplayReport) -> std::io::Result<bool>,
    {
        let mut report = ReplayReport::default();
        let mut last_seen_offset = since_offset;

        let segments = list_segments_sorted(&self.0.dir);
        let last_segment_idx = segments.len().saturating_sub(1);
        for (seg_idx, path) in segments.iter().enumerate() {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            // Peekable so we can tell whether a bad record is the very last line
            // (a torn/partial tail from an interrupted append, expected and safe
            // to drop) versus mid-segment corruption (real data loss to surface).
            let mut lines = reader.lines().peekable();
            let mut skipped_in_segment = 0usize;
            while let Some(line) = lines.next() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let Some(ev) = decode_wal_record(&line) else {
                    report.corrupted_records = report.corrupted_records.saturating_add(1);
                    skipped_in_segment = skipped_in_segment.saturating_add(1);
                    // Torn tail: last non-empty line of the last (active) segment.
                    let is_torn_tail = seg_idx == last_segment_idx && lines.peek().is_none();
                    if is_torn_tail {
                        tracing::debug!(
                            segment = %path.display(),
                            "dropping torn/partial WAL tail record"
                        );
                    } else {
                        // Mid-segment corruption: skip this one record and keep
                        // replaying the rest of the segment rather than silently
                        // abandoning every valid record after it. The resulting
                        // offset discontinuity is flagged via gap_detected below.
                        tracing::warn!(
                            segment = %path.display(),
                            "skipping corrupt mid-segment WAL record; continuing replay"
                        );
                    }
                    continue;
                };
                report.highest_offset_seen = report.highest_offset_seen.max(ev.offset);
                if ev.offset <= since_offset {
                    report.duplicates_skipped = report.duplicates_skipped.saturating_add(1);
                    continue;
                }
                if report.applied > 0 && ev.offset != last_seen_offset.saturating_add(1) {
                    report.gap_detected = true;
                }
                last_seen_offset = ev.offset;
                // Count every delivered record here so `applied` and gap detection
                // are consistent across all callers (not just closures that happen
                // to increment it themselves).
                report.applied = report.applied.saturating_add(1);
                if !f(ev, &mut report)? {
                    return Ok(report);
                }
            }
            if skipped_in_segment > 0 {
                tracing::warn!(
                    segment = %path.display(),
                    skipped = skipped_in_segment,
                    "WAL segment had corrupt/undecodable records"
                );
            }
        }

        Ok(report)
    }
}

fn apply_event(state: &crate::engine::state::StateStore, _vectors: &VectorStore, ev: &EventRecord) {
    match ev.event_type.as_str() {
        "state_updated" => {
            if let Some(key) = ev.data.get("key").and_then(|v| v.as_str()) {
                let value = ev
                    .data
                    .get("value")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let expires_at_ms = ev.data.get("expires_at_ms").and_then(|v| v.as_u64());
                let revision = ev
                    .data
                    .get("revision")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1);
                state.apply_wal_set(key.to_string(), value, revision, expires_at_ms);
            }
        }
        "state_deleted" => {
            if let Some(key) = ev.data.get("key").and_then(|v| v.as_str()) {
                let _ = state.delete(key);
            }
        }
        "vector_collection_created"
        | "vector_added"
        | "vector_upserted"
        | "vector_updated"
        | "vector_deleted" => {
            let _ = _vectors.apply_event(ev);
        }
        _ => {}
    }
}

fn ensure_file_exists(path: &Path) -> std::io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    let _ = OpenOptions::new().create(true).append(true).open(path)?;
    Ok(())
}

/// Fsync the directory so a preceding file create/rename is durable (the entry
/// itself, not just the file contents, survives a crash). Best-effort on
/// platforms that reject opening a directory for fsync.
fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    match File::open(dir) {
        Ok(f) => match f.sync_all() {
            Ok(()) => Ok(()),
            // ponytail: not all filesystems support fsync on a directory handle.
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => Ok(()),
            Err(e) => Err(e),
        },
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => Ok(()),
        Err(e) => Err(e),
    }
}

/// Highest record offset in a WAL segment (offsets are monotonic within a
/// segment, so this is effectively the last decodable record). None if the
/// segment is empty or can't be read — callers treat that as "not prunable".
fn segment_max_offset(path: &Path) -> Option<u64> {
    let file = File::open(path).ok()?;
    let mut max = None;
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { return max };
        if line.trim().is_empty() {
            continue;
        }
        if let Some(ev) = decode_wal_record(&line) {
            max = Some(ev.offset.max(max.unwrap_or(0)));
        }
    }
    max
}

fn list_segments_sorted(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<(u64, PathBuf)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if let Some(seg) = parse_segment_id(&path) {
                v.push((seg, path));
            }
        }
    }
    v.sort_by_key(|(seg, _)| *seg);
    v.into_iter().map(|(_, p)| p).collect()
}

fn find_latest_segment_id(dir: &Path) -> Option<u64> {
    list_segments_sorted(dir)
        .into_iter()
        .filter_map(|p| parse_segment_id(&p))
        .max()
}

fn parse_segment_id(path: &Path) -> Option<u64> {
    let name = path.file_name()?.to_str()?;
    let name = name.strip_prefix("events-")?;
    let name = name.strip_suffix(".log")?;
    name.parse::<u64>().ok()
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn encode_wal_record(event: &EventRecord) -> std::io::Result<Vec<u8>> {
    let payload = serde_json::to_vec(event)?;
    let mut hasher = Hasher::new();
    hasher.update(&payload);
    let envelope = WalEnvelope {
        version: WAL_RECORD_VERSION,
        offset: event.offset,
        crc32: hasher.finalize(),
        event: event.clone(),
    };
    serde_json::to_vec(&envelope).map_err(std::io::Error::other)
}

fn decode_wal_record(line: &str) -> Option<EventRecord> {
    if let Ok(envelope) = serde_json::from_str::<WalEnvelope>(line) {
        if envelope.version != WAL_RECORD_VERSION || envelope.offset != envelope.event.offset {
            return None;
        }
        let payload = serde_json::to_vec(&envelope.event).ok()?;
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        if hasher.finalize() != envelope.crc32 {
            return None;
        }
        return Some(envelope.event);
    }
    serde_json::from_str::<EventRecord>(line).ok()
}

#[cfg(test)]
mod tests {
    use super::{
        decode_wal_record, encode_wal_record, segment_max_offset, Persist, Snapshot, WalSyncMode,
    };
    use crate::engine::events::EventRecord;
    use tempfile::tempdir;

    fn sample_event(offset: u64) -> EventRecord {
        EventRecord {
            offset,
            ts_ms: 1,
            event_type: "state_updated".to_string(),
            data: serde_json::json!({
                "key": format!("k{offset}"),
                "value": offset,
                "revision": 1
            }),
        }
    }

    #[test]
    fn segment_max_offset_reads_highest_record() {
        use std::io::Write as _;
        let dir = tempdir().unwrap();
        let path = dir.path().join("events-000001.log");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            for off in [3u64, 7, 42] {
                f.write_all(&encode_wal_record(&sample_event(off)).unwrap())
                    .unwrap();
                f.write_all(b"\n").unwrap();
            }
        }
        // Highest offset drives the retention floor gate.
        assert_eq!(segment_max_offset(&path), Some(42));
        // Empty and missing segments are treated as "not prunable".
        let empty = dir.path().join("events-000002.log");
        std::fs::File::create(&empty).unwrap();
        assert_eq!(segment_max_offset(&empty), None);
        assert_eq!(segment_max_offset(&dir.path().join("nope.log")), None);
    }

    #[test]
    fn group_mode_flush_buffer_persists_pending_events() {
        let dir = tempdir().unwrap();
        let persist = Persist::new_with_mode(
            dir.path(),
            1024 * 1024,
            4,
            WalSyncMode::Group {
                batch_size: 64,
                flush_interval_ms: 10_000,
            },
        )
        .unwrap();

        persist.append_event(&sample_event(1)).unwrap();
        let wal_path = dir.path().join("events-000001.log");
        assert!(!wal_path.exists() || std::fs::read_to_string(&wal_path).unwrap().is_empty());

        persist.flush_buffer().unwrap();

        let wal_after = std::fs::read_to_string(wal_path).unwrap();
        assert!(wal_after.contains("\"offset\":1"));
    }

    #[test]
    fn snapshot_rotation_flushes_group_buffer_first() {
        let dir = tempdir().unwrap();
        let persist = Persist::new_with_mode(
            dir.path(),
            1024 * 1024,
            4,
            WalSyncMode::Group {
                batch_size: 64,
                flush_interval_ms: 10_000,
            },
        )
        .unwrap();

        persist.append_event(&sample_event(7)).unwrap();
        persist
            .write_snapshot_and_rotate(&Snapshot { last_offset: 7 })
            .unwrap();

        let first_segment = std::fs::read_to_string(dir.path().join("events-000001.log")).unwrap();
        assert!(first_segment.contains("\"offset\":7"));
        assert!(dir.path().join("snapshot.json").exists());
        assert!(dir.path().join("events-000002.log").exists());
    }

    #[test]
    fn mid_segment_corruption_skips_one_record_and_continues() {
        use std::io::Write as _;

        let dir = tempdir().unwrap();
        let persist = Persist::new(dir.path(), 1024 * 1024, 4).unwrap();

        // events-000001.log: valid(1), valid(2), <garbage>, valid(4).
        // The corrupt line stands in for what would have been offset 3.
        let seg_path = dir.path().join("events-000001.log");
        {
            let mut f = std::fs::File::create(&seg_path).unwrap();
            for offset in [1u64, 2] {
                f.write_all(&encode_wal_record(&sample_event(offset)).unwrap())
                    .unwrap();
                f.write_all(b"\n").unwrap();
            }
            f.write_all(b"{ this is not a valid wal record\n").unwrap();
            f.write_all(&encode_wal_record(&sample_event(4)).unwrap())
                .unwrap();
            f.write_all(b"\n").unwrap();
        }

        let mut seen = Vec::new();
        let report = persist
            .try_for_each_event_since(0, |ev| {
                seen.push(ev.offset);
                Ok(true)
            })
            .unwrap();

        // The valid record AFTER the corruption is still replayed (not abandoned).
        assert_eq!(seen, vec![1, 2, 4]);
        assert_eq!(report.applied, 3);
        assert_eq!(report.corrupted_records, 1);
        // The skipped record leaves a visible offset discontinuity.
        assert!(report.gap_detected);
    }

    #[test]
    fn wal_record_checksum_detects_tampering() {
        let encoded = encode_wal_record(&sample_event(9)).unwrap();
        let mut line = String::from_utf8(encoded).unwrap();
        line = line.replace("\"offset\":9", "\"offset\":99");
        assert!(decode_wal_record(&line).is_none());
    }
}
