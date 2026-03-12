use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// PR8: Number of shards for the KV store. Must be a power of 2.
const NUM_SHARDS: usize = 16;

/// FNV-1a hash for shard routing — fast, avoids external dependency.
fn shard_index(key: &str) -> usize {
    const FNV_PRIME: u64 = 1_099_511_628_211;
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut hash = FNV_OFFSET;
    for byte in key.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    (hash as usize) & (NUM_SHARDS - 1)
}

/// PR8: ShardedStateStore — 16 independent RwLock<HashMap> shards.
/// Replaces the single global RwLock with per-key routing, reducing contention
/// under concurrent write workloads.
#[derive(Clone)]
pub struct StateStore(Arc<Inner>);

struct Inner {
    shards: [RwLock<HashMap<String, Entry>>; NUM_SHARDS],
}

#[derive(Clone, Debug)]
struct Entry {
    value: serde_json::Value,
    revision: u64,
    expires_at_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateItem {
    pub key: String,
    pub value: serde_json::Value,
    pub revision: u64,
    pub expires_at_ms: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("revision mismatch")]
    RevisionMismatch,
}

impl StateStore {
    pub fn new() -> Self {
        Self(Arc::new(Inner {
            shards: std::array::from_fn(|_| RwLock::new(HashMap::new())),
        }))
    }

    pub fn get(&self, key: &str) -> Option<StateItem> {
        let now = now_ms();
        let shard = &self.0.shards[shard_index(key)];
        let map = shard.read();
        match map.get(key) {
            Some(e) if !is_expired(e, now) => Some(StateItem {
                key: key.to_string(),
                value: e.value.clone(),
                revision: e.revision,
                expires_at_ms: e.expires_at_ms,
            }),
            _ => None,
        }
    }

    pub fn list(&self, prefix: Option<&str>, limit: usize) -> Vec<StateItem> {
        let now = now_ms();
        let mut out = Vec::new();
        for shard in &self.0.shards {
            let map = shard.read();
            for (k, v) in map.iter() {
                if let Some(p) = prefix {
                    if !k.starts_with(p) {
                        continue;
                    }
                }
                if is_expired(v, now) {
                    continue;
                }
                out.push(StateItem {
                    key: k.clone(),
                    value: v.value.clone(),
                    revision: v.revision,
                    expires_at_ms: v.expires_at_ms,
                });
                if out.len() >= limit {
                    return out;
                }
            }
        }
        out
    }

    pub fn put(
        &self,
        key: String,
        value: serde_json::Value,
        ttl_ms: Option<u64>,
        if_revision: Option<u64>,
    ) -> Result<StateItem, StateError> {
        let now = now_ms();
        let expires_at_ms = ttl_ms.map(|ttl| now.saturating_add(ttl));
        let shard = &self.0.shards[shard_index(&key)];
        let mut map = shard.write();

        let entry = map.entry(key.clone());
        let (revision, value_out) = match entry {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if let Some(expected) = if_revision {
                    if e.get().revision != expected {
                        return Err(StateError::RevisionMismatch);
                    }
                }
                let next_rev = e.get().revision.saturating_add(1);
                e.insert(Entry {
                    value: value.clone(),
                    revision: next_rev,
                    expires_at_ms,
                });
                (next_rev, value)
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                if if_revision.is_some() {
                    return Err(StateError::RevisionMismatch);
                }
                e.insert(Entry {
                    value: value.clone(),
                    revision: 1,
                    expires_at_ms,
                });
                (1, value)
            }
        };

        Ok(StateItem {
            key,
            value: value_out,
            revision,
            expires_at_ms,
        })
    }

    pub fn delete(&self, key: &str) -> bool {
        let shard = &self.0.shards[shard_index(key)];
        let mut map = shard.write();
        map.remove(key).is_some()
    }

    pub fn peek_meta(&self, key: &str) -> Option<(u64, Option<u64>)> {
        let shard = &self.0.shards[shard_index(key)];
        let map = shard.read();
        map.get(key).map(|e| (e.revision, e.expires_at_ms))
    }

    pub fn snapshot(&self) -> Vec<(String, PersistStateEntry)> {
        let now = now_ms();
        let mut out = Vec::new();
        for shard in &self.0.shards {
            let map = shard.read();
            for (k, v) in map.iter() {
                if is_expired(v, now) {
                    continue;
                }
                out.push((
                    k.clone(),
                    PersistStateEntry {
                        value: v.value.clone(),
                        revision: v.revision,
                        expires_at_ms: v.expires_at_ms,
                    },
                ));
            }
        }
        out
    }

    pub fn load_snapshot(&self, entries: Vec<(String, PersistStateEntry)>) -> anyhow::Result<()> {
        // Clear all shards first
        for shard in &self.0.shards {
            shard.write().clear();
        }
        for (k, e) in entries {
            let shard = &self.0.shards[shard_index(&k)];
            shard.write().insert(
                k,
                Entry {
                    value: e.value,
                    revision: e.revision,
                    expires_at_ms: e.expires_at_ms,
                },
            );
        }
        Ok(())
    }

    pub fn apply_wal_set(
        &self,
        key: String,
        value: serde_json::Value,
        revision: u64,
        expires_at_ms: Option<u64>,
    ) {
        let shard = &self.0.shards[shard_index(&key)];
        let mut map = shard.write();
        map.insert(
            key,
            Entry {
                value,
                revision,
                expires_at_ms,
            },
        );
    }

    pub fn prepare_put_revision(
        &self,
        key: &str,
        if_revision: Option<u64>,
    ) -> Result<u64, StateError> {
        let now = now_ms();
        let shard = &self.0.shards[shard_index(key)];
        let map = shard.read();
        let current = map.get(key).filter(|e| !is_expired(e, now));
        match current {
            Some(e) => {
                if let Some(expected) = if_revision {
                    if e.revision != expected {
                        return Err(StateError::RevisionMismatch);
                    }
                }
                Ok(e.revision.saturating_add(1))
            }
            None => {
                if if_revision.is_some() {
                    return Err(StateError::RevisionMismatch);
                }
                Ok(1)
            }
        }
    }

    pub fn apply_put_with_revision(
        &self,
        key: String,
        value: serde_json::Value,
        revision: u64,
        expires_at_ms: Option<u64>,
    ) -> StateItem {
        let shard = &self.0.shards[shard_index(&key)];
        let mut map = shard.write();
        map.insert(
            key.clone(),
            Entry {
                value: value.clone(),
                revision,
                expires_at_ms,
            },
        );
        StateItem {
            key,
            value,
            revision,
            expires_at_ms,
        }
    }

    pub fn exists_live(&self, key: &str) -> bool {
        let shard = &self.0.shards[shard_index(key)];
        let map = shard.read();
        map.contains_key(key)
    }

    pub fn expired_keys(&self, now_ms: u64, limit: usize) -> Vec<String> {
        let mut out = Vec::new();
        for shard in &self.0.shards {
            let map = shard.read();
            for (k, v) in map.iter() {
                if let Some(exp) = v.expires_at_ms {
                    if exp <= now_ms {
                        out.push(k.clone());
                        if out.len() >= limit {
                            return out;
                        }
                    }
                }
            }
        }
        out
    }
}

impl Default for StateStore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistStateEntry {
    pub value: serde_json::Value,
    pub revision: u64,
    pub expires_at_ms: Option<u64>,
}

fn now_ms() -> u64 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}

fn is_expired(e: &Entry, now_ms: u64) -> bool {
    e.expires_at_ms.is_some_and(|exp| exp <= now_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_delete_revision() {
        let s = StateStore::new();
        let item1 = s
            .put("k".to_string(), serde_json::json!({"a": 1}), None, None)
            .unwrap();
        assert_eq!(item1.revision, 1);

        let item2 = s
            .put("k".to_string(), serde_json::json!({"a": 2}), None, Some(1))
            .unwrap();
        assert_eq!(item2.revision, 2);

        assert!(matches!(
            s.put("k".to_string(), serde_json::json!({"a": 3}), None, Some(1),),
            Err(StateError::RevisionMismatch)
        ));

        let got = s.get("k").unwrap();
        assert_eq!(got.revision, 2);

        assert!(s.delete("k"));
        assert!(s.get("k").is_none());
    }

    #[test]
    fn sharding_distributes_keys() {
        let s = StateStore::new();
        for i in 0..100 {
            s.put(format!("key-{i}"), serde_json::json!(i), None, None)
                .unwrap();
        }
        let all = s.list(None, 200);
        assert_eq!(all.len(), 100);
    }
}
