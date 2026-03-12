use crate::engine::events::EventRecord;
use crate::engine::EventBus;
use crate::vector::VectorStore;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub last_offset: u64,
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
        })))
    }

    /// Append a single event to the WAL.
    /// In PerWrite mode: fsync immediately.
    /// In Group mode: buffer until batch_size or flush_interval_ms.
    pub fn append_event(&self, event: &EventRecord) -> std::io::Result<()> {
        let line = serde_json::to_vec(event)?;

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

        self.enforce_retention_locked(seg)?;
        Ok(())
    }

    pub fn replay_wal_since(
        &self,
        since_offset: u64,
        state: &crate::engine::state::StateStore,
        vectors: &VectorStore,
        events: &EventBus,
    ) -> std::io::Result<usize> {
        let mut applied = 0usize;

        for path in list_segments_sorted(&self.0.dir) {
            let f = File::open(path)?;
            let reader = BufReader::new(f);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let ev: EventRecord = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if ev.offset <= since_offset {
                    continue;
                }
                apply_event(state, vectors, &ev);
                events.set_next_offset(ev.offset.saturating_add(1));
                applied += 1;
            }
        }

        Ok(applied)
    }

    pub fn list_segments(&self) -> Vec<PathBuf> {
        list_segments_sorted(&self.0.dir)
    }

    pub fn for_each_event_since<F>(&self, since_offset: u64, mut f: F) -> std::io::Result<()>
    where
        F: FnMut(EventRecord) -> bool,
    {
        for path in list_segments_sorted(&self.0.dir) {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let ev: EventRecord = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if ev.offset <= since_offset {
                    continue;
                }
                if !f(ev) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn segment_path(&self, seg: u64) -> PathBuf {
        self.0.dir.join(format!("events-{seg:06}.log"))
    }

    fn snapshot_path(&self) -> PathBuf {
        self.0.dir.join("snapshot.json")
    }

    fn enforce_retention_locked(&self, current_seg: u64) -> std::io::Result<()> {
        let keep = self.0.retention_segments;
        let start_keep = current_seg.saturating_sub(keep as u64).saturating_add(1);
        for path in list_segments_sorted(&self.0.dir) {
            if let Some(seg) = parse_segment_id(&path) {
                if seg < start_keep {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::{Persist, Snapshot, WalSyncMode};
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
}
