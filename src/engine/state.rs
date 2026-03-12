use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
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
    expiry_heap: RwLock<BinaryHeap<Reverse<(u64, String, u64)>>>,
    secondary_fields: RwLock<HashSet<String>>,
    secondary_index: RwLock<HashMap<String, HashMap<String, HashSet<String>>>>,
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
            expiry_heap: RwLock::new(BinaryHeap::new()),
            secondary_fields: RwLock::new(HashSet::new()),
            secondary_index: RwLock::new(HashMap::new()),
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
        let end = prefix.and_then(next_prefix_boundary);
        self.list_range(prefix, end.as_deref(), limit)
    }

    pub fn list_range(
        &self,
        start: Option<&str>,
        end: Option<&str>,
        limit: usize,
    ) -> Vec<StateItem> {
        let now = now_ms();
        let guards: Vec<_> = self.0.shards.iter().map(|shard| shard.read()).collect();
        let mut out = Vec::new();
        for map in &guards {
            for (k, v) in map.iter() {
                if let Some(start) = start {
                    if k.as_str() < start {
                        continue;
                    }
                }
                if let Some(end) = end {
                    if k.as_str() >= end {
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
            }
        }
        out.sort_by(|a, b| a.key.cmp(&b.key));
        out.truncate(limit);
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
        let (revision, value_out, old_value) = match entry {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if let Some(expected) = if_revision {
                    if e.get().revision != expected {
                        return Err(StateError::RevisionMismatch);
                    }
                }
                let old_value = e.get().value.clone();
                let next_rev = e.get().revision.saturating_add(1);
                e.insert(Entry {
                    value: value.clone(),
                    revision: next_rev,
                    expires_at_ms,
                });
                (next_rev, value, Some(old_value))
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
                (1, value, None)
            }
        };

        let item = StateItem {
            key,
            value: value_out,
            revision,
            expires_at_ms,
        };
        self.track_expiry(&item.key, item.revision, item.expires_at_ms);
        if let Some(old_value) = old_value.as_ref() {
            self.reindex_key(&item.key, Some(old_value), None);
        }
        self.reindex_key(&item.key, None, Some(&item.value));
        Ok(item)
    }

    pub fn delete(&self, key: &str) -> bool {
        let shard = &self.0.shards[shard_index(key)];
        let mut map = shard.write();
        let removed = map.remove(key);
        if let Some(entry) = removed {
            self.reindex_key(key, Some(&entry.value), None);
            true
        } else {
            false
        }
    }

    pub fn peek_meta(&self, key: &str) -> Option<(u64, Option<u64>)> {
        let shard = &self.0.shards[shard_index(key)];
        let map = shard.read();
        map.get(key).map(|e| (e.revision, e.expires_at_ms))
    }

    pub fn snapshot(&self) -> Vec<(String, PersistStateEntry)> {
        let now = now_ms();
        let guards: Vec<_> = self.0.shards.iter().map(|shard| shard.read()).collect();
        let mut out = Vec::new();
        for map in &guards {
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
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    pub fn load_snapshot(&self, entries: Vec<(String, PersistStateEntry)>) -> anyhow::Result<()> {
        let mut guards: Vec<_> = self.0.shards.iter().map(|shard| shard.write()).collect();
        for shard in &mut guards {
            shard.clear();
        }
        for (k, e) in entries {
            let shard = &mut guards[shard_index(&k)];
            shard.insert(
                k,
                Entry {
                    value: e.value,
                    revision: e.revision,
                    expires_at_ms: e.expires_at_ms,
                },
            );
        }
        drop(guards);
        self.rebuild_secondary_and_expiry_indexes();
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
        if map
            .get(&key)
            .is_some_and(|existing| existing.revision >= revision)
        {
            return;
        }
        let old = map.get(&key).map(|entry| entry.value.clone());
        map.insert(
            key.clone(),
            Entry {
                value,
                revision,
                expires_at_ms,
            },
        );
        drop(map);
        if let Some(old) = old.as_ref() {
            self.reindex_key(&key, Some(old), None);
        }
        let shard = &self.0.shards[shard_index(&key)];
        let map = shard.read();
        if let Some(entry) = map.get(&key) {
            self.reindex_key(&key, None, Some(&entry.value));
        }
        self.track_expiry(&key, revision, expires_at_ms);
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
        let old = map.get(&key).map(|entry| entry.value.clone());
        map.insert(
            key.clone(),
            Entry {
                value: value.clone(),
                revision,
                expires_at_ms,
            },
        );
        drop(map);
        if let Some(old) = old.as_ref() {
            self.reindex_key(&key, Some(old), None);
        }
        self.reindex_key(&key, None, Some(&value));
        self.track_expiry(&key, revision, expires_at_ms);
        StateItem {
            key,
            value,
            revision,
            expires_at_ms,
        }
    }

    pub fn exists_live(&self, key: &str) -> bool {
        let now = now_ms();
        let shard = &self.0.shards[shard_index(key)];
        let map = shard.read();
        map.get(key).is_some_and(|entry| !is_expired(entry, now))
    }

    pub fn exists_any(&self, key: &str) -> bool {
        let shard = &self.0.shards[shard_index(key)];
        let map = shard.read();
        map.contains_key(key)
    }

    pub fn expired_keys(&self, now_ms: u64, limit: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut heap = self.0.expiry_heap.write();
        while let Some(Reverse((expires_at_ms, key, revision))) = heap.peek().cloned() {
            if expires_at_ms > now_ms || out.len() >= limit {
                break;
            }
            heap.pop();
            let shard = &self.0.shards[shard_index(&key)];
            let map = shard.read();
            if let Some(entry) = map.get(&key) {
                if entry.revision == revision
                    && entry.expires_at_ms.is_some_and(|exp| exp <= now_ms)
                {
                    out.push(key);
                }
            }
        }
        out
    }

    pub fn create_secondary_index(&self, field: &str) {
        self.0.secondary_fields.write().insert(field.to_string());
        self.rebuild_secondary_and_expiry_indexes();
    }

    pub fn query_secondary_index(&self, field: &str, value: &str, limit: usize) -> Vec<StateItem> {
        let keys = self
            .0
            .secondary_index
            .read()
            .get(field)
            .and_then(|values| values.get(value))
            .cloned()
            .unwrap_or_default();
        let mut items = keys
            .into_iter()
            .filter_map(|key| self.get(&key))
            .filter(|item| {
                item.value
                    .get(field)
                    .and_then(|field_value| indexable_json_value(field_value, ""))
                    .or_else(|| indexable_json_value(&item.value, field))
                    .is_some_and(|actual| actual == value)
            })
            .collect::<Vec<_>>();
        items.sort_by(|a, b| a.key.cmp(&b.key));
        items.truncate(limit);
        items
    }

    fn rebuild_secondary_and_expiry_indexes(&self) {
        let fields = self.0.secondary_fields.read().clone();
        let guards: Vec<_> = self.0.shards.iter().map(|shard| shard.read()).collect();
        let now = now_ms();
        let mut secondary = HashMap::<String, HashMap<String, HashSet<String>>>::new();
        let mut heap = BinaryHeap::new();
        for map in &guards {
            for (key, entry) in map.iter() {
                if let Some(expires_at_ms) = entry.expires_at_ms {
                    heap.push(Reverse((expires_at_ms, key.clone(), entry.revision)));
                }
                if is_expired(entry, now) {
                    continue;
                }
                for field in &fields {
                    if let Some(value) = indexable_json_value(&entry.value, field) {
                        secondary
                            .entry(field.clone())
                            .or_default()
                            .entry(value)
                            .or_default()
                            .insert(key.clone());
                    }
                }
            }
        }
        *self.0.secondary_index.write() = secondary;
        *self.0.expiry_heap.write() = heap;
    }

    fn reindex_key(
        &self,
        key: &str,
        old_value: Option<&serde_json::Value>,
        new_value: Option<&serde_json::Value>,
    ) {
        let fields = self.0.secondary_fields.read().clone();
        if fields.is_empty() {
            return;
        }
        let mut index = self.0.secondary_index.write();
        for field in fields {
            if let Some(old_value) = old_value.and_then(|value| indexable_json_value(value, &field))
            {
                if let Some(values) = index.get_mut(&field) {
                    if let Some(keys) = values.get_mut(&old_value) {
                        keys.remove(key);
                        if keys.is_empty() {
                            values.remove(&old_value);
                        }
                    }
                }
            }
            if let Some(new_value) = new_value.and_then(|value| indexable_json_value(value, &field))
            {
                index
                    .entry(field.clone())
                    .or_default()
                    .entry(new_value)
                    .or_default()
                    .insert(key.to_string());
            }
        }
    }

    fn track_expiry(&self, key: &str, revision: u64, expires_at_ms: Option<u64>) {
        let Some(expires_at_ms) = expires_at_ms else {
            return;
        };
        self.0
            .expiry_heap
            .write()
            .push(Reverse((expires_at_ms, key.to_string(), revision)));
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

fn indexable_json_value(value: &serde_json::Value, field: &str) -> Option<String> {
    match value.get(field) {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Number(n)) => Some(n.to_string()),
        Some(serde_json::Value::Bool(b)) => Some(b.to_string()),
        _ => None,
    }
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

    #[test]
    fn global_operations_are_sorted_and_expiry_aware() {
        let s = StateStore::new();
        let now = now_ms();
        s.apply_wal_set("z-key".to_string(), serde_json::json!(1), 1, None);
        s.apply_wal_set("a-key".to_string(), serde_json::json!(2), 1, None);
        s.apply_wal_set(
            "ttl-expired".to_string(),
            serde_json::json!(3),
            1,
            Some(now.saturating_sub(1)),
        );

        let listed = s.list(None, 10);
        let listed_keys: Vec<_> = listed.iter().map(|item| item.key.as_str()).collect();
        assert_eq!(listed_keys, vec!["a-key", "z-key"]);
        assert!(!s.exists_live("ttl-expired"));

        let snapshot_keys: Vec<_> = s.snapshot().into_iter().map(|(key, _)| key).collect();
        assert_eq!(
            snapshot_keys,
            vec!["a-key".to_string(), "z-key".to_string()]
        );

        let expired = s.expired_keys(now, 10);
        assert_eq!(expired, vec!["ttl-expired".to_string()]);
    }

    #[test]
    fn list_range_is_sorted_and_end_exclusive() {
        let s = StateStore::new();
        for key in ["a:1", "a:2", "b:1", "c:1"] {
            s.apply_wal_set(key.to_string(), serde_json::json!(key), 1, None);
        }
        let listed = s.list_range(Some("a:"), Some("c:"), 10);
        let keys: Vec<_> = listed.into_iter().map(|item| item.key).collect();
        assert_eq!(keys, vec!["a:1", "a:2", "b:1"]);
    }

    #[test]
    fn wal_set_is_idempotent_for_older_revisions() {
        let s = StateStore::new();
        s.apply_wal_set("k".to_string(), serde_json::json!(1), 4, None);
        s.apply_wal_set("k".to_string(), serde_json::json!(2), 3, None);
        assert_eq!(s.get("k").unwrap().value, serde_json::json!(1));
    }

    #[test]
    fn secondary_index_tracks_updates_and_deletes() {
        let s = StateStore::new();
        s.create_secondary_index("tenant");
        s.put(
            "doc:1".to_string(),
            serde_json::json!({"tenant":"acme","value":1}),
            None,
            None,
        )
        .unwrap();
        s.put(
            "doc:2".to_string(),
            serde_json::json!({"tenant":"globex","value":2}),
            None,
            None,
        )
        .unwrap();
        assert_eq!(s.query_secondary_index("tenant", "acme", 10).len(), 1);

        s.put(
            "doc:1".to_string(),
            serde_json::json!({"tenant":"globex","value":3}),
            None,
            Some(1),
        )
        .unwrap();
        assert!(s.query_secondary_index("tenant", "acme", 10).is_empty());
        assert_eq!(s.query_secondary_index("tenant", "globex", 10).len(), 2);

        s.delete("doc:2");
        assert_eq!(s.query_secondary_index("tenant", "globex", 10).len(), 1);
    }

    #[test]
    fn ttl_heap_returns_due_keys_without_full_scan_ordering() {
        let s = StateStore::new();
        let now = now_ms();
        s.apply_wal_set("a".to_string(), serde_json::json!(1), 1, Some(now + 20));
        s.apply_wal_set("b".to_string(), serde_json::json!(2), 1, Some(now + 10));
        s.apply_wal_set("c".to_string(), serde_json::json!(3), 1, Some(now + 30));

        let due = s.expired_keys(now + 15, 10);
        assert_eq!(due, vec!["b".to_string()]);
    }

    #[test]
    fn load_snapshot_replaces_all_shards_consistently() {
        let s = StateStore::new();
        s.apply_wal_set("old-a".to_string(), serde_json::json!(1), 1, None);
        s.apply_wal_set("old-b".to_string(), serde_json::json!(2), 1, None);

        s.load_snapshot(vec![
            (
                "new-z".to_string(),
                PersistStateEntry {
                    value: serde_json::json!(10),
                    revision: 3,
                    expires_at_ms: None,
                },
            ),
            (
                "new-a".to_string(),
                PersistStateEntry {
                    value: serde_json::json!(11),
                    revision: 4,
                    expires_at_ms: None,
                },
            ),
        ])
        .unwrap();

        assert!(s.get("old-a").is_none());
        assert!(s.get("old-b").is_none());
        let keys: Vec<_> = s.list(None, 10).into_iter().map(|item| item.key).collect();
        assert_eq!(keys, vec!["new-a".to_string(), "new-z".to_string()]);
    }
}
