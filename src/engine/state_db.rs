use crate::engine::events::EventRecord;
use crate::engine::state::{StateError, StateItem};
use anyhow::Context;
use redb::{Database, ReadableTable, TableDefinition};
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

    pub fn list(&self, prefix: Option<&str>, limit: usize) -> anyhow::Result<Vec<StateItem>> {
        let tx = self.db.begin_read()?;
        let table = match tx.open_table(STATE) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };
        let now = now_ms();
        let mut out = Vec::new();

        if let Some(prefix) = prefix {
            let start = prefix.as_bytes().to_vec();
            for kv in table.range(start.as_slice()..)? {
                let (k, v) = kv?;
                let key = std::str::from_utf8(k.value()).unwrap_or_default();
                if !key.starts_with(prefix) {
                    break;
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
        } else {
            for kv in table.iter()? {
                let (k, v) = kv?;
                let key = std::str::from_utf8(k.value()).unwrap_or_default();
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
        wtx.commit()?;
        Ok(())
    }

    pub fn apply_state_deleted(&self, ev: &EventRecord) -> anyhow::Result<()> {
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
