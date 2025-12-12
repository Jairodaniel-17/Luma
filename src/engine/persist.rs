use crate::engine::events::EventRecord;
use crate::engine::EventBus;
use crate::vector::VectorStore;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct Persist(Arc<Inner>);

struct Inner {
    dir: PathBuf,
    wal_lock: Mutex<()>,
    segment_max_bytes: u64,
    retention_segments: usize,
    current_segment: Mutex<u64>,
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
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let current_segment = find_latest_segment_id(&dir).unwrap_or(1);
        Ok(Self(Arc::new(Inner {
            dir,
            wal_lock: Mutex::new(()),
            segment_max_bytes: segment_max_bytes.max(1024 * 1024),
            retention_segments: retention_segments.max(1),
            current_segment: Mutex::new(current_segment),
        })))
    }

    pub fn append_event(&self, event: &EventRecord) -> std::io::Result<()> {
        let _g = self.0.wal_lock.lock();

        let mut seg = *self.0.current_segment.lock();
        let mut path = self.segment_path(seg);
        ensure_file_exists(&path)?;

        let line = serde_json::to_vec(event)?;
        let estimated = line.len() as u64 + 1;
        let current_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if current_size.saturating_add(estimated) > self.0.segment_max_bytes {
            seg = seg.saturating_add(1);
            *self.0.current_segment.lock() = seg;
            path = self.segment_path(seg);
            ensure_file_exists(&path)?;
        }

        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        file.write_all(&line)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_data()?;

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
