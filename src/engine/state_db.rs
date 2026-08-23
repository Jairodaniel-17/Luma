//! The KV projection: an LSM store rebuilt from the WAL.
//!
//! ## Why an LSM and not a B-tree
//!
//! This used to be redb, a copy-on-write B-tree, and that was the wrong shape
//! for a write-heavy projection. A single-key insert in a COW B-tree rewrites
//! the page path from leaf to root — about 16 KB written for a 30-byte value,
//! roughly 500x amplification. An LSM writes into an in-memory table and flushes
//! it to a sorted file, and a background job merges those files, so a write never
//! rewrites a tree.
//!
//! Measured, same machine, same shape (`tests/redb_ceiling.rs`,
//! `tests/lsm_ceiling.rs`):
//!
//! ```text
//! redb, Eventual, one txn per write      1 308/s
//! redb, None, one txn per write         11 401/s
//! redb, None, batched 32 per txn       235 679/s
//! LSM, projection role                 432 764/s
//! LSM, 128 concurrent writers          208 944/s
//! ```
//!
//! Three things made the swap cheap rather than risky:
//!
//! - **There is nothing to migrate.** This store holds no data of its own; it is
//!   a projection of the WAL, rebuilt from `applied_offset`. Changing engines is
//!   deleting a directory and replaying.
//! - **Compaction comes written.** An index over the WAL — the other candidate —
//!   would have meant writing and testing a compaction subsystem, which is the
//!   part of that design that puts data at risk. An LSM already has one.
//! - **Ordered iteration is native.** `KEYS`, `SCAN` and `list_range` walk keys
//!   in order. An LSM is sorted, so they work directly; a hash index over the WAL
//!   could not do it at any price.
//!
//! ## Durability, and why so little of it
//!
//! Nothing here fsyncs on the write path, and that is correct rather than
//! reckless for the same reason it was correct for redb: **this is not the source
//! of truth.** The WAL is. On a crash this store rolls back to its last persisted
//! point, `applied_offset` rolls back with it because it is written in the same
//! atomic batch, and replay re-applies the difference. WAL retention will not
//! prune past that point — `set_durable_floor` holds the line using the offset
//! `flush` returns. So a crash costs a rebuild, never data.
//!
//! The cost is memory: unpersisted writes live in the memtable. Bounded by how
//! often `flush` runs and by the memtable size, not by the size of the dataset —
//! 50 000 inserts measured at 18.3 MiB.

use crate::engine::events::EventRecord;
use crate::engine::state::{StateError, StateItem};
use anyhow::Context;
use fjall::{Config, Keyspace, PartitionCreateOptions, PartitionHandle, PersistMode};
use std::path::Path;

/// Values, by key.
const STATE: &str = "state";
/// `(expires_at_ms, key)` → nothing, so TTL expiry is a range scan and not a
/// full pass over the keyspace.
const EXPIRES: &str = "expires";
/// Bookkeeping, which is only ever `applied_offset`.
const META: &str = "meta";

const META_APPLIED_OFFSET: &[u8] = b"applied_offset";

#[derive(Clone)]
pub struct StateDb {
    keyspace: Keyspace,
    state: PartitionHandle,
    expires: PartitionHandle,
    meta: PartitionHandle,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredValue {
    /// `StoredVal` deserializes a bare JSON value back to `Json`, so every
    /// record written before the raw variant existed loads unchanged.
    value: crate::engine::stored::StoredVal,
    revision: u64,
    expires_at_ms: Option<u64>,
}

impl StateDb {
    pub fn open(data_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        // A directory, not a file: an LSM keeps its journal and its sorted runs
        // separately. The name says which engine wrote it, so a downgrade finds
        // no directory rather than a file it would misread.
        let path = data_dir.as_ref().join("state.lsm");
        let keyspace = Config::new(&path).open().context("open the LSM keyspace")?;
        let options = PartitionCreateOptions::default();
        Ok(Self {
            state: keyspace.open_partition(STATE, options.clone())?,
            expires: keyspace.open_partition(EXPIRES, options.clone())?,
            meta: keyspace.open_partition(META, options)?,
            keyspace,
        })
    }

    fn read(&self, key: &str) -> anyhow::Result<Option<StoredValue>> {
        let Some(raw) = self.state.get(key.as_bytes())? else {
            return Ok(None);
        };
        Ok(Some(
            serde_json::from_slice(&raw).context("decode stored value")?,
        ))
    }

    pub fn get_state(&self, key: &str) -> anyhow::Result<Option<StateItem>> {
        let Some(stored) = self.read(key)? else {
            return Ok(None);
        };
        // An expired key reads as absent even before the TTL sweep removes it,
        // so a reader never sees a value the clock has already retired.
        if stored.expires_at_ms.is_some_and(|e| e <= now_ms()) {
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
        Ok(self.state.get(key.as_bytes())?.is_some())
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
        let now = now_ms();
        let mut out = Vec::new();

        // The range is left-inclusive and right-exclusive, matching what
        // `next_prefix_boundary` produces and what the KV API documents.
        let iter: Box<dyn Iterator<Item = fjall::Result<(fjall::Slice, fjall::Slice)>>> =
            match start {
                Some(start) => Box::new(self.state.range(start.as_bytes().to_vec()..)),
                None => Box::new(self.state.iter()),
            };

        for kv in iter {
            let (k, v) = kv?;
            let key = std::str::from_utf8(&k).unwrap_or_default().to_string();
            if let Some(end) = end {
                if key.as_str() >= end {
                    break;
                }
            }
            let stored: StoredValue = serde_json::from_slice(&v)?;
            if stored.expires_at_ms.is_some_and(|e| e <= now) {
                continue;
            }
            out.push(StateItem {
                key,
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
        let now = now_ms();
        // A read failure is treated as absent rather than propagated: the caller
        // only accepts a revision or a mismatch, and inventing a revision from a
        // failed read is worse than refusing the compare-and-swap.
        let current = self
            .read(key)
            .ok()
            .flatten()
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
        self.apply_events(&[ev]).map(|_| ())
    }

    pub fn apply_state_deleted(&self, ev: &EventRecord) -> anyhow::Result<()> {
        self.apply_events(&[ev]).map(|_| ())
    }

    /// Apply a `state_batch` record: every op in one atomic batch.
    ///
    /// One batch because a crash must not leave the projection holding half a
    /// move with `applied_offset` already past it — replay would not repair
    /// that, since it skips anything at or below the offset.
    pub fn apply_state_batch(&self, ev: &EventRecord) -> anyhow::Result<()> {
        self.apply_events(&[ev]).map(|_| ())
    }

    /// Apply many events in **one** atomic batch.
    ///
    /// The single write path: the three `apply_*` methods above all route here,
    /// so there is one place where an event becomes a mutation. When they were
    /// separate bodies they drifted — the expiry-index removal is exactly the
    /// step that gets forgotten in a copy, and forgetting it leaks index entries
    /// that outlive their values.
    ///
    /// Batching is also where the throughput is: 32 records sharing one commit
    /// measured 235 679/s against 11 401/s one at a time, back when this was a
    /// B-tree. `applied_offset` goes into the same batch as the data, so the two
    /// cannot come back from a crash disagreeing.
    ///
    /// Non-state events are skipped rather than rejected: a batch arrives mixed
    /// and the vector store applies its own.
    pub fn apply_events(&self, events: &[&EventRecord]) -> anyhow::Result<u64> {
        let already = self.applied_offset()?;
        let mut batch = self.keyspace.batch();
        let mut highest = 0u64;

        for ev in events {
            // The idempotence both callers rely on: replay re-offers records the
            // projection already holds, and they must be no-ops rather than
            // double applications.
            if ev.offset <= already {
                continue;
            }
            let handled = match ev.event_type.as_str() {
                "state_updated" => {
                    self.stage_put(&mut batch, &ev.data)?;
                    true
                }
                "state_deleted" => {
                    self.stage_delete(&mut batch, &ev.data)?;
                    true
                }
                "state_batch" => {
                    let ops = ev
                        .data
                        .get("ops")
                        .and_then(|v| v.as_array())
                        .context("state_batch without ops")?;
                    for op in ops {
                        if op.get("op").and_then(|v| v.as_str()) == Some("delete") {
                            self.stage_delete(&mut batch, op)?;
                        } else {
                            self.stage_put(&mut batch, op)?;
                        }
                    }
                    true
                }
                _ => false,
            };
            if handled {
                highest = highest.max(ev.offset);
            }
        }

        if highest > already {
            batch.insert(&self.meta, META_APPLIED_OFFSET, highest.to_le_bytes());
            batch.commit()?;
        }
        Ok(highest)
    }

    /// Stage one key's new value, and retire the expiry index entry it replaces.
    fn stage_put(
        &self,
        batch: &mut fjall::Batch,
        source: &serde_json::Value,
    ) -> anyhow::Result<()> {
        let key = source
            .get("key")
            .and_then(|v| v.as_str())
            .context("missing key")?;
        let revision = source.get("revision").and_then(|v| v.as_u64()).unwrap_or(1);
        let expires_at_ms = source.get("expires_at_ms").and_then(|v| v.as_u64());
        // Decoded through `StoredVal` so a raw payload lands as bytes rather than
        // as its marker object.
        let value = source
            .get("value")
            .cloned()
            .map(serde_json::from_value::<crate::engine::stored::StoredVal>)
            .transpose()
            .unwrap_or(None)
            .unwrap_or_default();

        // The previous index entry points at a value about to be replaced.
        // Leaving it behind would fire a TTL sweep against a key whose expiry
        // has moved.
        if let Some(previous) = self.read(key)? {
            if let Some(at) = previous.expires_at_ms {
                batch.remove(&self.expires, expires_key(at, key.as_bytes()));
            }
        }

        let stored = StoredValue {
            value,
            revision,
            expires_at_ms,
        };
        batch.insert(&self.state, key.as_bytes(), serde_json::to_vec(&stored)?);
        if let Some(at) = expires_at_ms {
            batch.insert(&self.expires, expires_key(at, key.as_bytes()), [0u8]);
        }
        Ok(())
    }

    /// Stage one key's removal, index entry included.
    fn stage_delete(
        &self,
        batch: &mut fjall::Batch,
        source: &serde_json::Value,
    ) -> anyhow::Result<()> {
        let key = source
            .get("key")
            .and_then(|v| v.as_str())
            .context("missing key")?;
        if let Some(previous) = self.read(key)? {
            if let Some(at) = previous.expires_at_ms {
                batch.remove(&self.expires, expires_key(at, key.as_bytes()));
            }
        }
        batch.remove(&self.state, key.as_bytes());
        Ok(())
    }

    /// Make everything applied so far durable, and report the offset that is now
    /// safe for WAL retention to prune below.
    ///
    /// The checkpoint. Called before a snapshot records its offset, and it is
    /// what keeps the WAL from discarding records this projection has not
    /// persisted — which is the only thing standing between a crash and real
    /// data loss, given nothing on the write path fsyncs.
    pub fn flush(&self) -> anyhow::Result<u64> {
        self.keyspace
            .persist(PersistMode::SyncAll)
            .context("persist the LSM journal")?;
        self.applied_offset()
    }

    pub fn applied_offset(&self) -> anyhow::Result<u64> {
        let Some(raw) = self.meta.get(META_APPLIED_OFFSET)? else {
            return Ok(0);
        };
        Ok(u64::from_le_bytes(
            raw.as_ref().try_into().unwrap_or([0; 8]),
        ))
    }

    pub fn expired_keys_due(&self, now_ms: u64, limit: usize) -> anyhow::Result<Vec<String>> {
        let mut out = Vec::new();
        let start = expires_key(0, &[]);
        let end = expires_key(now_ms, &[0xFF; 1]);
        for kv in self.expires.range(start..=end) {
            let (k, _) = kv?;
            if let Some(key) = parse_expires_key(&k) {
                out.push(key);
                if out.len() >= limit {
                    break;
                }
            }
        }
        Ok(out)
    }
}

/// `(expires_at_ms, key)`, big-endian so byte order is time order and the sweep
/// is a range scan rather than a full pass.
fn expires_key(expires_at_ms: u64, key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + key.len());
    out.extend_from_slice(&expires_at_ms.to_be_bytes());
    out.extend_from_slice(key);
    out
}

fn parse_expires_key(raw: &[u8]) -> Option<String> {
    if raw.len() <= 8 {
        return None;
    }
    std::str::from_utf8(&raw[8..]).ok().map(str::to_string)
}

/// The exclusive upper bound of a prefix range.
///
/// `None` when the prefix is all `0xFF`, which has no successor — the caller
/// then scans to the end, which is the correct answer rather than an empty one.
fn next_prefix_boundary(prefix: &str) -> Option<String> {
    let mut bytes = prefix.as_bytes().to_vec();
    while let Some(last) = bytes.pop() {
        if last < 0xFF {
            bytes.push(last + 1);
            return String::from_utf8(bytes).ok();
        }
    }
    None
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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

    fn with_ttl(offset: u64, key: &str, expires_at_ms: u64) -> EventRecord {
        EventRecord {
            offset,
            ts_ms: 1,
            event_type: "state_updated".to_string(),
            data: serde_json::json!({
                "key": key, "value": offset, "revision": 1, "expires_at_ms": expires_at_ms
            }),
        }
    }

    #[test]
    fn apply_flush_and_reopen_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        {
            let db = StateDb::open(dir.path()).unwrap();
            for i in 1..=3u64 {
                db.apply_state_updated(&ev(i, &format!("k{i}"), i)).unwrap();
            }
            // The checkpoint reports the offset it makes durable, which is what
            // WAL retention is then allowed to prune below.
            assert_eq!(db.flush().unwrap(), 3);
            for i in 4..=5u64 {
                db.apply_state_updated(&ev(i, &format!("k{i}"), i)).unwrap();
            }
            assert_eq!(db.applied_offset().unwrap(), 5);
        }
        let reopened = StateDb::open(dir.path()).unwrap();
        assert_eq!(reopened.applied_offset().unwrap(), 5);
        for i in 1..=5u64 {
            let item = reopened.get_state(&format!("k{i}")).unwrap().unwrap();
            assert_eq!(item.value.as_json(), Some(&serde_json::json!(i)));
        }
    }

    #[test]
    fn apply_is_idempotent_below_applied_offset() {
        // Replay starts below `applied_offset` and re-offers records the
        // projection already holds. They must be no-ops, not regressions.
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        db.apply_state_updated(&ev(5, "k", 5)).unwrap();
        db.apply_state_updated(&ev(3, "k", 999)).unwrap();
        assert_eq!(db.applied_offset().unwrap(), 5);
        assert_eq!(
            db.get_state("k").unwrap().unwrap().value.as_json(),
            Some(&serde_json::json!(5))
        );
    }

    #[test]
    fn a_batch_lands_whole_and_advances_the_offset_once() {
        // The property that makes one atomic batch non-negotiable: a crash must
        // not leave half a move applied with the offset already past it, because
        // replay skips anything at or below the offset and would never repair it.
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        db.apply_state_updated(&ev(1, "from", 42)).unwrap();

        let batch = EventRecord {
            offset: 7,
            ts_ms: 1,
            event_type: "state_batch".to_string(),
            data: serde_json::json!({ "ops": [
                { "op": "put", "key": "to", "value": 1, "revision": 1 },
                { "op": "delete", "key": "from" }
            ]}),
        };
        db.apply_state_batch(&batch).unwrap();

        assert_eq!(db.applied_offset().unwrap(), 7);
        assert!(db.get_state("from").unwrap().is_none());
        assert!(db.get_state("to").unwrap().is_some());
    }

    #[test]
    fn a_mixed_batch_applies_only_the_state_events() {
        // The commit pipeline hands over whatever was queued, vector events
        // included. Those own their own storage; taking them here would apply
        // them twice.
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        let vector = EventRecord {
            offset: 2,
            ts_ms: 1,
            event_type: "vector_upserted".to_string(),
            data: serde_json::json!({ "collection": "c", "id": "v" }),
        };
        let state = ev(3, "k", 1);
        let highest = db.apply_events(&[&vector, &state]).unwrap();
        assert_eq!(highest, 3, "the state event is the highest applied here");
        assert!(db.get_state("k").unwrap().is_some());
    }

    #[test]
    fn an_expired_key_reads_as_absent_before_the_sweep_removes_it() {
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        db.apply_state_updated(&with_ttl(1, "gone", 1)).unwrap();

        assert!(
            db.get_state("gone").unwrap().is_none(),
            "a reader must never see a value the clock has retired"
        );
        assert!(
            db.exists_any("gone").unwrap(),
            "the record is still on disk until the sweep takes it"
        );
        assert_eq!(
            db.expired_keys_due(super::now_ms(), 10).unwrap(),
            vec!["gone"]
        );
    }

    #[test]
    fn replacing_a_value_retires_its_old_expiry_entry() {
        // Left behind, the stale index entry fires a TTL sweep against a key
        // whose expiry has moved — deleting a value that should still be live.
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        db.apply_state_updated(&with_ttl(1, "k", 1_000)).unwrap();
        db.apply_state_updated(&with_ttl(2, "k", u64::MAX)).unwrap();

        assert!(
            db.expired_keys_due(super::now_ms(), 10).unwrap().is_empty(),
            "the entry at 1000 must be gone, so a sweep now finds nothing"
        );
        assert!(db.get_state("k").unwrap().is_some());
    }

    #[test]
    fn a_delete_removes_the_key_and_its_index_entry() {
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        db.apply_state_updated(&with_ttl(1, "k", u64::MAX)).unwrap();
        db.apply_state_deleted(&EventRecord {
            offset: 2,
            ts_ms: 1,
            event_type: "state_deleted".to_string(),
            data: serde_json::json!({ "key": "k" }),
        })
        .unwrap();

        assert!(!db.exists_any("k").unwrap());
        assert!(
            db.expired_keys_due(u64::MAX, 10).unwrap().is_empty(),
            "a sweep must not report a key that no longer exists"
        );
    }

    #[test]
    fn a_prefix_listing_stops_at_the_prefix_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        for (offset, key) in [(1u64, "a:1"), (2, "a:2"), (3, "b:1")] {
            db.apply_state_updated(&ev(offset, key, offset)).unwrap();
        }
        let keys: Vec<String> = db
            .list(Some("a:"), 100)
            .unwrap()
            .into_iter()
            .map(|item| item.key)
            .collect();
        assert_eq!(keys, vec!["a:1", "a:2"], "b:1 is past the boundary");
    }

    #[test]
    fn a_revision_guard_refuses_a_stale_expectation() {
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::open(dir.path()).unwrap();
        assert_eq!(db.prepare_put_revision("new", None).unwrap(), 1);
        assert!(
            db.prepare_put_revision("new", Some(1)).is_err(),
            "a compare-and-swap against a key that does not exist must fail"
        );

        db.apply_state_updated(&ev(1, "k", 1)).unwrap();
        assert_eq!(db.prepare_put_revision("k", Some(1)).unwrap(), 2);
        assert!(db.prepare_put_revision("k", Some(99)).is_err());
    }
}
