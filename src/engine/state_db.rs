use crate::engine::events::EventRecord;
use crate::engine::state::{StateError, StateItem};
use anyhow::Context;
use redb::{Database, Durability, ReadableTable, TableDefinition};
use std::path::Path;
use std::sync::Arc;

const STATE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("state");
const EXPIRES: TableDefinition<&[u8], u8> = TableDefinition::new("expires");
const META: TableDefinition<&[u8], &[u8]> = TableDefinition::new("meta");

const META_APPLIED_OFFSET: &[u8] = b"applied_offset";

#[derive(Clone)]
pub struct StateDb {
    db: Arc<Database>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredValue {
    value: serde_json::Value,
    revision: u64,
    expires_at_ms: Option<u64>,
}

impl StateDb {
    pub fn open(data_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = data_dir.as_ref().join("state.redb");
        let db = Database::create(&path).context("create/open redb")?;
        let this = Self { db: Arc::new(db) };
        this.init_tables().context("init tables")?;
        Ok(this)
    }

    fn init_tables(&self) -> anyhow::Result<()> {
        let wtx = self.db.begin_write()?;
        let _ = wtx.open_table(STATE)?;
        let _ = wtx.open_table(EXPIRES)?;
        let _ = wtx.open_table(META)?;
        wtx.commit()?;
        Ok(())
    }

    pub fn get_state(&self, key: &str) -> anyhow::Result<Option<StateItem>> {
        let tx = self.db.begin_read()?;
        let table = match tx.open_table(STATE) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        let now = now_ms();
        let Some(raw) = table.get(key.as_bytes())? else {
            return Ok(None);
        };
        let stored: StoredValue =
            serde_json::from_slice(raw.value()).context("decode stored value")?;
        if stored.expires_at_ms.is_some_and(|e| e <= now) {
            return Ok(None);
        }
        Ok(Some(StateItem {
            key: key.to_string(),
            value: stored.value,
            revision: stored.revision,
            expires_at_ms: stored.expires_at_ms,
        }))
    }

    pub fn exists_live(&self, key: &str) -> anyhow::Result<bool> {
        Ok(self.get_state(key)?.is_some())
    }

    pub fn exists_any(&self, key: &str) -> anyhow::Result<bool> {
        let tx = self.db.begin_read()?;
        let table = match tx.open_table(STATE) {
            Ok(t) => t,
            Err(_) => return Ok(false),
        };
        Ok(table.get(key.as_bytes())?.is_some())
    }

    pub fn list(&self, prefix: Option<&str>, limit: usize) -> anyhow::Result<Vec<StateItem>> {
        let end = prefix.and_then(next_prefix_boundary);
        self.list_range(prefix, end.as_deref(), limit)
    }

    pub fn list_range(
        &self,
        start: Option<&str>,
        end: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<StateItem>> {
        let tx = self.db.begin_read()?;
        let table = match tx.open_table(STATE) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };
        let now = now_ms();
        let mut out = Vec::new();

        let iter = match start {
            Some(start) => table.range(start.as_bytes()..)?,
            None => table.iter()?,
        };
        for kv in iter {
            let (k, v) = kv?;
            let key = std::str::from_utf8(k.value()).unwrap_or_default();
            if let Some(end) = end {
                if key >= end {
                    break;
                }
            }
            let stored: StoredValue = serde_json::from_slice(v.value())?;
            if stored.expires_at_ms.is_some_and(|e| e <= now) {
                continue;
            }
            out.push(StateItem {
                key: key.to_string(),
                value: stored.value,
                revision: stored.revision,
                expires_at_ms: stored.expires_at_ms,
            });
            if out.len() >= limit {
                break;
            }
        }

        Ok(out)
    }

    pub fn prepare_put_revision(
        &self,
        key: &str,
        if_revision: Option<u64>,
    ) -> Result<u64, StateError> {
        let tx = self
            .db
            .begin_read()
            .map_err(|_| StateError::RevisionMismatch)?;
        let table = match tx.open_table(STATE) {
            Ok(t) => t,
            Err(_) => {
                if if_revision.is_some() {
                    return Err(StateError::RevisionMismatch);
                }
                return Ok(1);
            }
        };
        let now = now_ms();

        let current = table
            .get(key.as_bytes())
            .ok()
            .flatten()
            .and_then(|raw| serde_json::from_slice::<StoredValue>(raw.value()).ok())
            .filter(|v| v.expires_at_ms.is_none_or(|e| e > now));

        match current {
            Some(v) => {
                if let Some(expected) = if_revision {
                    if v.revision != expected {
                        return Err(StateError::RevisionMismatch);
                    }
                }
                Ok(v.revision.saturating_add(1))
            }
            None => {
                if if_revision.is_some() {
                    return Err(StateError::RevisionMismatch);
                }
                Ok(1)
            }
        }
    }

    pub fn apply_state_updated(&self, ev: &EventRecord) -> anyhow::Result<()> {
        if ev.offset <= self.applied_offset()? {
            return Ok(());
        }
        let key = ev
            .data
            .get("key")
            .and_then(|v| v.as_str())
            .context("missing key")?;
        let revision = ev
            .data
            .get("revision")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        let expires_at_ms = ev.data.get("expires_at_ms").and_then(|v| v.as_u64());
        let value = ev
            .data
            .get("value")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let mut wtx = self.db.begin_write()?;
        {
            let mut state = wtx.open_table(STATE)?;
            let mut expires = wtx.open_table(EXPIRES)?;

            let prev = if let Some(prev_raw) = state.get(key.as_bytes())? {
                let bytes = prev_raw.value().to_vec();
                serde_json::from_slice::<StoredValue>(&bytes).ok()
            } else {
                None
            };
            if let Some(prev) = prev {
                if let Some(exp) = prev.expires_at_ms {
                    let idx = expires_key(exp, key.as_bytes());
                    let _ = expires.remove(idx.as_slice())?;
                }
            }

            let stored = StoredValue {
                value,
                revision,
                expires_at_ms,
            };
            let bytes = serde_json::to_vec(&stored)?;
            state.insert(key.as_bytes(), bytes.as_slice())?;

            if let Some(exp) = expires_at_ms {
                let idx = expires_key(exp, key.as_bytes());
                expires.insert(idx.as_slice(), 0u8)?;
            }
        }
        set_applied_offset(&mut wtx, ev.offset)?;
        // Eventual durability: don't fsync on every write. The WAL is the durable
        // source of truth; this store is a projection rebuilt from it on replay.
        // `flush()` (an Immediate commit) is called at each snapshot/checkpoint to
        // make it durable and let WAL retention advance. On crash, redb rolls back
        // to the last Immediate commit and replay re-applies the rest.
        wtx.set_durability(Durability::Eventual);
        wtx.commit()?;
        Ok(())
    }

    /// Force all pending Eventual commits durable (a checkpoint) and return the
    /// offset now guaranteed persistent. Called before a snapshot records its
    /// offset and WAL segments at/below the returned offset are pruned.
    ///
    /// The applied_offset is read inside this exclusive write transaction, so no
    /// concurrent Eventual apply can advance it between the read and the fsync —
    /// the returned value is exactly what this Immediate commit makes durable.
    pub fn flush(&self) -> anyhow::Result<u64> {
        let wtx = self.db.begin_write()?;
        let meta = wtx.open_table(META)?;
        let offset = meta
            .get(META_APPLIED_OFFSET)?
            .map(|v| u64::from_le_bytes(v.value().try_into().unwrap_or([0; 8])))
            .unwrap_or(0);
        drop(meta); // release the table borrow before consuming wtx in commit()
                    // Default durability is Immediate → fsync, persisting every prior Eventual
                    // commit up to `offset`.
        wtx.commit()?;
        Ok(offset)
    }

    pub fn apply_state_deleted(&self, ev: &EventRecord) -> anyhow::Result<()> {
        if ev.offset <= self.applied_offset()? {
            return Ok(());
        }
        let key = ev
            .data
            .get("key")
            .and_then(|v| v.as_str())
            .context("missing key")?;

        let mut wtx = self.db.begin_write()?;
        {
            let mut state = wtx.open_table(STATE)?;
            let mut expires = wtx.open_table(EXPIRES)?;
            let prev = if let Some(prev_raw) = state.remove(key.as_bytes())? {
                let bytes = prev_raw.value().to_vec();
                serde_json::from_slice::<StoredValue>(&bytes).ok()
            } else {
                None
            };
            if let Some(prev) = prev {
                if let Some(exp) = prev.expires_at_ms {
                    let idx = expires_key(exp, key.as_bytes());
                    let _ = expires.remove(idx.as_slice())?;
                }
            };
        }
        set_applied_offset(&mut wtx, ev.offset)?;
        wtx.set_durability(Durability::Eventual);
        wtx.commit()?;
        Ok(())
    }

    pub fn applied_offset(&self) -> anyhow::Result<u64> {
        let tx = self.db.begin_read()?;
        let meta = match tx.open_table(META) {
            Ok(t) => t,
            Err(_) => return Ok(0),
        };
        let Some(v) = meta.get(META_APPLIED_OFFSET)? else {
            return Ok(0);
        };
        Ok(u64::from_le_bytes(v.value().try_into().unwrap_or([0; 8])))
    }

    pub fn expired_keys_due(&self, now_ms: u64, limit: usize) -> anyhow::Result<Vec<String>> {
        let tx = self.db.begin_read()?;
        let expires = match tx.open_table(EXPIRES) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        let mut out = Vec::new();
        let start = expires_key(0, &[]);
        let end = expires_key(now_ms, &[0xFF; 1]);
        for kv in expires.range(start.as_slice()..=end.as_slice())? {
            let (k, _) = kv?;
            if let Some(key) = parse_expires_key(k.value()) {
                out.push(key);
                if out.len() >= limit {
                    break;
                }
            }
        }
        Ok(out)
    }
}

fn set_applied_offset(wtx: &mut redb::WriteTransaction, offset: u64) -> anyhow::Result<()> {
    let mut meta = wtx.open_table(META)?;
    meta.insert(META_APPLIED_OFFSET, offset.to_le_bytes().as_slice())?;
    Ok(())
}

fn expires_key(expires_at_ms: u64, key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + key.len());
    out.extend_from_slice(&expires_at_ms.to_be_bytes());
    out.extend_from_slice(key);
    out
}

fn parse_expires_key(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 8 {
        return None;
    }
    std::str::from_utf8(&bytes[8..]).ok().map(|s| s.to_string())
}

fn now_ms() -> u64 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}

fn next_prefix_boundary(prefix: &str) -> Option<String> {
    let mut bytes = prefix.as_bytes().to_vec();
    for idx in (0..bytes.len()).rev() {
        if bytes[idx] != u8::MAX {
            bytes[idx] = bytes[idx].saturating_add(1);
            bytes.truncate(idx + 1);
            return String::from_utf8(bytes).ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::StateDb;
    use crate::engine::events::EventRecord;

    fn ev(offset: u64, key: &str, val: u64) -> EventRecord {
        EventRecord {
            offset,
            ts_ms: 1,
            event_type: "state_updated".to_string(),
            data: serde_json::json!({ "key": key, "value": val, "revision": 1 }),
        }
    }

    #[test]
    fn eventual_apply_flush_and_reopen_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = StateDb::open(dir.path()).unwrap();
            for i in 1..=3u64 {
                db.apply_state_updated(&ev(i, &format!("k{i}"), i)).unwrap();
            }
            // Checkpoint: flush reports the offset it makes durable.
            assert_eq!(db.flush().unwrap(), 3);
            // Eventual writes after the checkpoint.
            for i in 4..=5u64 {
                db.apply_state_updated(&ev(i, &format!("k{i}"), i)).unwrap();
            }
            assert_eq!(db.applied_offset().unwrap(), 5);
        }
        // Reopen: applied_offset and all values persist across the store's lifetime.
        let db2 = StateDb::open(dir.path()).unwrap();
        assert_eq!(db2.applied_offset().unwrap(), 5);
        for i in 1..=5u64 {
            let item = db2.get_state(&format!("k{i}")).unwrap().unwrap();
            assert_eq!(item.value, serde_json::json!(i));
        }
    }

    #[test]
    fn apply_is_idempotent_below_applied_offset() {
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        db.apply_state_updated(&ev(5, "k", 5)).unwrap();
        // Replaying an older offset (as happens when replay starts below
        // applied_offset) must be a no-op and must not regress state.
        db.apply_state_updated(&ev(3, "k", 999)).unwrap();
        assert_eq!(db.applied_offset().unwrap(), 5);
        assert_eq!(
            db.get_state("k").unwrap().unwrap().value,
            serde_json::json!(5)
        );
    }
}
