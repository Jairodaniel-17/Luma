pub mod adapters;
pub mod chunking;
pub mod embeddings;
mod events;
pub mod hub;
pub mod meta;
pub mod metrics;
pub mod parser;
mod persist;
mod state;
mod state_db;
pub mod traits;

use crate::config::Config;
use crate::engine::metrics::VectorMetricsSnapshot;
use crate::vector::index::{DiskAnnBuildParams, DiskIndexStatus};
use crate::vector::{
    Metric, SearchHit, SearchRequest, VectorCollectionInfo, VectorError, VectorItem,
    VectorSettings, VectorStore,
};
use anyhow::Context;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub struct Engine(Arc<Inner>);

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("persistence error: {0}")]
    Persistence(#[from] std::io::Error),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
    #[error(transparent)]
    State(#[from] state::StateError),
    #[error(transparent)]
    Vector(#[from] VectorError),
}

struct Inner {
    config: Config,
    state: state::StateStore,
    state_db: Option<state_db::StateDb>,
    vectors: VectorStore,
    events: events::EventBus,
    metrics: Arc<metrics::Metrics>,
    persist: Option<persist::Persist>,
    memory_rss_bytes: AtomicU64,

    shutdown: CancellationToken,
}

/// Upper bound on the in-RAM event replay buffer when persistence is enabled.
/// The durable log handles replay for lagging subscribers, so this only needs to
/// cover incidental in-process use. Keeps vector-payload events from piling up in
/// RAM (was effectively `event_buffer_size`, default 10k × ~KBs each).
const PERSISTENT_EVENT_BUFFER_CAP: usize = 256;

const VECTOR_MANIFEST_PREFIX: &str = "vector:";
const VECTOR_MANIFEST_SUFFIX: &str = ":manifest";
const VECTOR_MANIFEST_SCAN_LIMIT: usize = 4096;

impl Engine {
    pub fn new(config: Config, shutdown: CancellationToken) -> anyhow::Result<Self> {
        // The in-RAM replay buffer retains full event payloads (a vector upsert
        // event carries the whole vector as a serde_json::Value, ~KBs each). When
        // persistence is on, the durable event log is the replay source for
        // subscribers that fall behind (see routes_events), so the in-RAM buffer
        // is never read — cap it hard to avoid holding tens/hundreds of MB of
        // vector payloads. In memory-only mode the buffer *is* the replay source,
        // so keep the configured size.
        let event_buffer_size = if config.data_dir.is_some() {
            config.event_buffer_size.min(PERSISTENT_EVENT_BUFFER_CAP)
        } else {
            config.event_buffer_size
        };
        let events = events::EventBus::new(event_buffer_size, config.live_broadcast_capacity);
        let metrics = Arc::new(metrics::Metrics::default());

        let persist = match &config.data_dir {
            Some(dir) => Some(
                persist::Persist::new_with_mode(
                    dir,
                    config.wal_segment_max_bytes,
                    config.wal_retention_segments,
                    persist::WalSyncMode::from_config(&config),
                )
                .context("init persistence")?,
            ),
            None => None,
        };

        let state_db = match &config.data_dir {
            Some(dir) => Some(state_db::StateDb::open(dir).context("open state db")?),
            None => None,
        };
        let state = state::StateStore::new();
        let vector_settings = VectorSettings::from_config(&config);
        let vectors = match &config.data_dir {
            Some(dir) => VectorStore::open_with_settings(dir, vector_settings.clone())
                .context("open vector store")?,
            None => VectorStore::with_settings(vector_settings.clone()),
        };

        let engine = Self(Arc::new(Inner {
            config: config.clone(),
            state,
            state_db,
            vectors,
            events,
            metrics,
            persist,
            memory_rss_bytes: AtomicU64::new(read_process_memory_rss()),

            shutdown,
        }));

        if engine.0.persist.is_some() {
            engine.load_from_disk().context("load from disk")?;
            engine.start_snapshot_task_if_runtime();
        }
        if let Err(err) = engine.expire_due_keys(10_000) {
            tracing::warn!(error = %err, "startup ttl expire failed");
        }
        engine.start_ttl_task_if_runtime();
        engine.start_wal_flush_task_if_runtime();
        engine.start_hnsw_compaction_task_if_runtime();
        engine.start_process_metrics_task_if_runtime();

        Ok(engine)
    }

    pub fn shutdown(&self) {
        if let Some(persist) = &self.0.persist {
            if let Err(err) = persist.flush_buffer() {
                tracing::warn!(error = %err, "wal flush during shutdown failed");
            }
        }
        self.0.shutdown.cancel();
    }

    pub fn metrics_text(&self) -> String {
        let cached = self.0.metrics.cached_render();
        if !cached.is_empty() {
            return cached;
        }
        self.refresh_metrics_snapshot()
    }

    fn refresh_metrics_snapshot(&self) -> String {
        let collections = self.0.vectors.list_collections();
        let active_collections = collections.len() as u64;
        let total_vectors: u64 = collections.iter().map(|c| c.live_count as u64).sum();
        let vector_metrics = collections
            .iter()
            .map(|c| VectorMetricsSnapshot {
                collection: c.collection.clone(),
                live_count: c.live_count as u64,
                segments: c.segments.unwrap_or_default() as u64,
                deleted_count: c.deleted_count.unwrap_or_default(),
                fragmentation_score: c.fragmentation_score.unwrap_or_default(),
            })
            .collect::<Vec<_>>();
        let memory_rss_bytes = self.0.memory_rss_bytes.load(Ordering::Relaxed);
        self.0.metrics.render(
            active_collections,
            total_vectors,
            memory_rss_bytes,
            &vector_metrics,
        )
    }

    pub fn health(&self) -> &'static str {
        "ok"
    }

    pub fn events(&self) -> &events::EventBus {
        &self.0.events
    }

    pub fn metrics(&self) -> Arc<metrics::Metrics> {
        self.0.metrics.clone()
    }

    pub fn persist(&self) -> Option<persist::Persist> {
        self.0.persist.clone()
    }

    fn load_from_disk(&self) -> anyhow::Result<()> {
        let Some(persist) = &self.0.persist else {
            return Ok(());
        };

        let mut since_offset = 0u64;
        if let Some(snapshot) = persist.load_snapshot().context("read snapshot")? {
            self.0.events.set_next_offset(snapshot.last_offset + 1);
            since_offset = snapshot.last_offset;
        }
        if let Some(db) = &self.0.state_db {
            // redb uses Eventual durability, so after a crash it may sit BELOW the
            // snapshot offset (rolled back to the last checkpoint). Replay from
            // redb's own durable applied offset — never above it — so no state
            // event it lost is skipped. WAL retention keeps everything above this
            // floor, so those records are still on disk. Vectors fsync per write
            // (Immediate) and are idempotent, so replaying from this possibly-lower
            // offset only re-applies work they already have. Enabling the floor
            // here also gates retention for the whole run (see set_durable_floor).
            let applied = db.applied_offset().unwrap_or(0);
            since_offset = applied;
            persist.set_durable_floor(applied);
        }

        if let Some(db) = &self.0.state_db {
            let vectors = self.0.vectors.clone();
            let events = self.0.events.clone();
            let replay = persist
                .try_for_each_event_since(since_offset, |ev| {
                    match ev.event_type.as_str() {
                        "state_updated" => {
                            db.apply_state_updated(&ev)
                                .map_err(|err| std::io::Error::other(err.to_string()))?;
                        }
                        "state_deleted" => {
                            db.apply_state_deleted(&ev)
                                .map_err(|err| std::io::Error::other(err.to_string()))?;
                        }
                        "vector_collection_created"
                        | "vector_added"
                        | "vector_upserted"
                        | "vector_updated"
                        | "vector_deleted" => {
                            vectors
                                .apply_event(&ev)
                                .map_err(|err| std::io::Error::other(err.to_string()))?;
                        }
                        _ => {}
                    }
                    events.set_next_offset(ev.offset.saturating_add(1));
                    Ok(true)
                })
                .context("replay wal (db)")?;
            self.0.metrics.observe_replay(
                replay.applied as u64,
                replay.duplicates_skipped as u64,
                replay.corrupted_records as u64,
                replay.gap_detected,
            );
            tracing::info!(
                applied = replay.applied,
                duplicates = replay.duplicates_skipped,
                corrupted = replay.corrupted_records,
                gap_detected = replay.gap_detected,
                "replayed wal events"
            );
        } else {
            let replay = persist
                .replay_wal_since(since_offset, &self.0.state, &self.0.vectors, &self.0.events)
                .context("replay wal")?;
            self.0.metrics.observe_replay(
                replay.applied as u64,
                replay.duplicates_skipped as u64,
                replay.corrupted_records as u64,
                replay.gap_detected,
            );
            tracing::info!(
                applied = replay.applied,
                duplicates = replay.duplicates_skipped,
                corrupted = replay.corrupted_records,
                gap_detected = replay.gap_detected,
                "replayed wal events"
            );
        }

        Ok(())
    }

    fn start_snapshot_task_if_runtime(&self) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let interval_secs = self.0.config.snapshot_interval_secs;
        let weak = Arc::downgrade(&self.0);
        let shutdown = self.0.shutdown.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let Some(inner) = weak.upgrade() else { break };
                        let engine = Engine(inner);
                        let snapshot_engine = engine.clone();
                        let res = tokio::task::spawn_blocking(move || snapshot_engine.snapshot_once()).await;
                        match res {
                            Ok(Ok(())) => {
                                engine.0.metrics.inc_snapshot_ok();
                                tracing::info!("snapshot ok");
                            }
                            Ok(Err(err)) => {
                                engine.0.metrics.inc_snapshot_failed();
                                tracing::warn!(error = %err, "snapshot failed");
                            }
                            Err(err) => {
                                engine.0.metrics.inc_snapshot_failed();
                                tracing::warn!(error = %err, "snapshot task join failed");
                            }
                        };
                    }
                    _ = shutdown.cancelled() => {
                        tracing::info!("snapshot task stopping");
                        break;
                    }
                }
            }
        });
    }

    fn snapshot_once(&self) -> std::io::Result<()> {
        let Some(persist) = &self.0.persist else {
            return Ok(());
        };
        loop {
            match self.expire_due_keys_locked(now_ms(), 10_000) {
                Ok(0) => break,
                Ok(_) => continue,
                Err(err) => {
                    tracing::warn!(error = %err, "ttl expire during snapshot failed");
                    break;
                }
            }
        }
        // Flush the WAL buffer first so the durable WAL is a superset of whatever
        // redb is about to checkpoint (redb can never end up ahead of the WAL).
        persist.flush_buffer()?;
        // Checkpoint the derived store before recording the snapshot and pruning
        // WAL segments. redb now uses Eventual durability (no per-write fsync), so
        // flush it to a durable point and set that as the WAL retention floor:
        // retention may only drop segments whose records are all durably applied.
        // Vector segments already fsync per write, so redb is the only lossy store.
        if let Some(db) = &self.0.state_db {
            match db.flush() {
                Ok(durable_offset) => persist.set_durable_floor(durable_offset),
                Err(err) => {
                    tracing::warn!(error = %err, "state_db checkpoint flush failed; skipping snapshot");
                    return Ok(());
                }
            }
        }
        let snapshot = persist::Snapshot {
            last_offset: self.0.events.last_published_offset(),
        };
        persist.write_snapshot_and_rotate(&snapshot)?;
        self.0.metrics.inc_wal_rotation();
        Ok(())
    }

    pub fn force_snapshot(&self) -> Result<(), EngineError> {
        self.snapshot_once()?;
        Ok(())
    }

    fn start_ttl_task_if_runtime(&self) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let weak = Arc::downgrade(&self.0);
        let shutdown = self.0.shutdown.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let Some(inner) = weak.upgrade() else { break };
                        let engine = Engine(inner);
                        let res = tokio::task::spawn_blocking(move || engine.expire_due_keys(1000)).await;
                        match res {
                            Ok(Ok(expired)) if expired > 0 => tracing::info!(expired, "ttl expired"),
                            Ok(Ok(_)) => {}
                            Ok(Err(err)) => tracing::warn!(error = %err, "ttl task failed"),
                            Err(err) => tracing::warn!(error = %err, "ttl task join failed"),
                        }
                    }
                    _ = shutdown.cancelled() => {
                        tracing::info!("ttl task stopping");
                        break;
                    }
                }
            }
        });
    }

    fn start_wal_flush_task_if_runtime(&self) {
        let Some(persist) = self.persist() else {
            return;
        };
        let Some(flush_interval) = persist.group_flush_interval() else {
            return;
        };
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }

        let shutdown = self.0.shutdown.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(flush_interval);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let persist = persist.clone();
                        let res = tokio::task::spawn_blocking(move || persist.flush_buffer()).await;
                        match res {
                            Ok(Ok(())) => {}
                            Ok(Err(err)) => tracing::warn!(error = %err, "wal flush task failed"),
                            Err(err) => tracing::warn!(error = %err, "wal flush task join failed"),
                        }
                    }
                    _ = shutdown.cancelled() => {
                        let persist = persist.clone();
                        let _ = tokio::task::spawn_blocking(move || persist.flush_buffer()).await;
                        tracing::info!("wal flush task stopping");
                        break;
                    }
                }
            }
        });
    }

    /// PR6: Background task that periodically compacts HNSW segments with high tombstone ratios.
    fn start_hnsw_compaction_task_if_runtime(&self) {
        if !self.0.config.hnsw_segment_compaction_enabled {
            return;
        }
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let weak = Arc::downgrade(&self.0);
        let shutdown = self.0.shutdown.clone();
        let threshold = self
            .0
            .config
            .hnsw_segment_compaction_threshold
            .clamp(0.0, 1.0);
        let interval_secs = self.0.config.hnsw_segment_compaction_interval_secs.max(5);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let Some(inner) = weak.upgrade() else { break };
                        let engine = Engine(inner.clone());
                        let res = tokio::task::spawn_blocking(move || {
                            engine.0.vectors.compact_hnsw_segments(threshold)
                        })
                        .await;
                        match res {
                            Ok(compacted) if !compacted.is_empty() => {
                                inner.metrics.inc_hnsw_compaction();
                                tracing::info!(collections = ?compacted, "HNSW segment compaction completed");
                            }
                            Ok(_) => {}
                            Err(err) => {
                                tracing::warn!(error = %err, "HNSW compaction task failed");
                            }
                        }
                    }
                    _ = shutdown.cancelled() => {
                        tracing::info!("HNSW compaction task stopping");
                        break;
                    }
                }
            }
        });
    }

    fn start_process_metrics_task_if_runtime(&self) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let weak = Arc::downgrade(&self.0);
        let shutdown = self.0.shutdown.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let Some(inner) = weak.upgrade() else { break };
                        inner
                            .memory_rss_bytes
                            .store(read_process_memory_rss(), Ordering::Relaxed);
                        let engine = Engine(inner);
                        let _ = engine.refresh_metrics_snapshot();
                    }
                    _ = shutdown.cancelled() => break,
                }
            }
        });
    }

    pub fn list_state(&self, prefix: Option<&str>, limit: usize) -> Vec<state::StateItem> {
        if let Some(db) = &self.0.state_db {
            return db.list(prefix, limit).unwrap_or_default();
        }
        self.0.state.list(prefix, limit)
    }

    pub fn list_state_range(
        &self,
        start: Option<&str>,
        end: Option<&str>,
        limit: usize,
    ) -> Vec<state::StateItem> {
        if let Some(db) = &self.0.state_db {
            return db.list_range(start, end, limit).unwrap_or_default();
        }
        self.0.state.list_range(start, end, limit)
    }

    pub fn get_state(&self, key: &str) -> Option<state::StateItem> {
        if let Some(db) = &self.0.state_db {
            return db.get_state(key).ok().flatten();
        }
        self.0.state.get(key)
    }

    pub fn get_consumer_offset(&self, group: &str) -> Option<u64> {
        self.get_state(&format!("consumer_offset:{group}"))
            .and_then(|item| item.value.get("offset").and_then(|value| value.as_u64()))
    }

    pub fn commit_consumer_offset(&self, group: &str, offset: u64) -> Result<(), EngineError> {
        let key = format!("consumer_offset:{group}");
        let revision = self.get_state(&key).map(|item| item.revision);
        let _ = self.put_state(
            key,
            serde_json::json!({
                "group": group,
                "offset": offset,
                "updated_at_ms": now_ms(),
            }),
            None,
            revision,
        )?;
        Ok(())
    }

    pub fn create_state_secondary_index(&self, field: &str) {
        self.0.state.create_secondary_index(field);
    }

    pub fn query_state_secondary_index(
        &self,
        field: &str,
        value: &str,
        limit: usize,
    ) -> Vec<state::StateItem> {
        self.0.state.query_secondary_index(field, value, limit)
    }

    pub fn put_state(
        &self,
        key: String,
        value: serde_json::Value,
        ttl_ms: Option<u64>,
        if_revision: Option<u64>,
    ) -> Result<state::StateItem, EngineError> {
        let now = now_ms();
        let expires_at_ms = ttl_ms.map(|ttl| now.saturating_add(ttl));
        let revision = if let Some(db) = &self.0.state_db {
            db.prepare_put_revision(&key, if_revision)?
        } else {
            self.0.state.prepare_put_revision(&key, if_revision)?
        };

        let event_data = serde_json::json!({
            "key": key,
            "revision": revision,
            "value": value,
            "expires_at_ms": expires_at_ms,
        });
        let value = event_data["value"].clone();
        let key = event_data["key"].as_str().unwrap_or_default().to_string();
        // Hold the append guard across offset allocation, WAL append and publish
        // so offset order == WAL file order == publish order (no false gaps).
        let append = self.0.events.append_guard();
        let event = self.0.events.next_record("state_updated", event_data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        if let Some(db) = &self.0.state_db {
            db.apply_state_updated(&event)?;
        }
        self.0.events.publish_record(event.clone());
        drop(append);
        self.metrics().inc_events();

        self.metrics().inc_state_put();
        // state_db path: we just applied this exact (key,value,revision,expires)
        // above, so build the returned item directly instead of paying a second
        // redb read transaction to read back what we already know.
        let item = if self.0.state_db.is_some() {
            state::StateItem {
                key,
                value,
                revision,
                expires_at_ms,
            }
        } else {
            self.0
                .state
                .apply_put_with_revision(key, value, revision, expires_at_ms)
        };
        Ok(item)
    }

    pub fn delete_state(&self, key: &str) -> Result<bool, EngineError> {
        self.delete_state_with_reason(key, "explicit")
    }

    pub fn delete_state_with_reason(
        &self,
        key: &str,
        reason: &'static str,
    ) -> Result<bool, EngineError> {
        let exists = if let Some(db) = &self.0.state_db {
            db.exists_live(key)?
        } else {
            self.0.state.exists_live(key)
        };
        if !exists {
            return Ok(false);
        }

        let data = serde_json::json!({
            "key": key,
            "reason": reason,
        });
        let append = self.0.events.append_guard();
        let event = self.0.events.next_record("state_deleted", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        if let Some(db) = &self.0.state_db {
            db.apply_state_deleted(&event)?;
        }
        self.0.events.publish_record(event);
        drop(append);
        self.metrics().inc_events();

        let deleted = if self.0.state_db.is_some() {
            true
        } else {
            self.0.state.delete(key)
        };
        if deleted {
            self.metrics().inc_state_delete();
        }
        Ok(deleted)
    }

    pub fn vectors(&self) -> &VectorStore {
        &self.0.vectors
    }

    pub fn vector_collection_info(&self, collection: &str) -> Option<VectorCollectionInfo> {
        self.0.vectors.get_collection_info(collection)
    }

    pub fn vector_manifest_value(&self, collection: &str) -> Option<serde_json::Value> {
        self.get_state(&vector_manifest_key(collection))
            .map(|item| item.value)
    }

    pub fn list_vector_collections(&self) -> Vec<VectorCollectionInfo> {
        let mut collections: HashMap<String, VectorCollectionInfo> = self
            .0
            .vectors
            .list_collections()
            .into_iter()
            .map(|info| (info.collection.clone(), info))
            .collect();

        let manifest_items =
            self.list_state(Some(VECTOR_MANIFEST_PREFIX), VECTOR_MANIFEST_SCAN_LIMIT);
        for item in manifest_items {
            let Some(collection) = collection_from_manifest_key(&item.key) else {
                continue;
            };
            let Some(meta) = parse_vector_manifest_metadata(&collection, &item.value) else {
                continue;
            };
            if let Some(existing) = collections.get_mut(&collection) {
                existing.created_at_ms = meta.created_at_ms;
                existing.updated_at_ms = meta.updated_at_ms;
                if let Some(dim) = meta.dim {
                    existing.dim = dim;
                }
                if let Some(metric) = meta.metric {
                    existing.metric = metric;
                }
                continue;
            }
            let (Some(dim), Some(metric)) = (meta.dim, meta.metric) else {
                continue;
            };
            collections.insert(
                collection.clone(),
                VectorCollectionInfo {
                    collection,
                    dim,
                    metric,
                    live_count: 0,
                    total_records: 0,
                    upsert_count: 0,
                    file_len: 0,
                    applied_offset: 0,
                    created_at_ms: meta.created_at_ms,
                    updated_at_ms: meta.updated_at_ms,
                    segments: None,
                    deleted_count: None,
                    fragmentation_score: None,
                },
            );
        }

        let mut out: Vec<_> = collections.into_values().collect();
        out.sort_by(|a, b| a.collection.cmp(&b.collection));
        out
    }

    fn persist_vector_manifest_state(
        &self,
        collection: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<(), EngineError> {
        let key = vector_manifest_key(collection);
        let existing = self.get_state(&key);
        let created_at_ms = existing
            .as_ref()
            .and_then(|item| item.value.get("created_at_ms"))
            .and_then(|v| v.as_u64())
            .unwrap_or_else(now_ms);
        let updated_at_ms = now_ms();
        let value = serde_json::json!({
            "collection": collection,
            "dim": dim,
            "metric": metric,
            "created_at_ms": created_at_ms,
            "updated_at_ms": updated_at_ms,
        });
        let revision = existing.as_ref().map(|item| item.revision);
        let _ = self.put_state(key, value, None, revision)?;
        Ok(())
    }

    pub fn create_vector_collection(
        &self,
        collection: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<(), EngineError> {
        if self.0.vectors.get_collection(collection).is_some() {
            return Err(VectorError::CollectionExists.into());
        }
        let data = serde_json::json!({
            "collection": collection,
            "dim": dim,
            "metric": metric,
        });
        let append = self.0.events.append_guard();
        let event = self.0.events.next_record("vector_collection_created", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.create_collection(collection, dim, metric)?;
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
        drop(append);
        self.metrics().inc_events();
        self.metrics().inc_vector_op();

        if let Err(err) = self.persist_vector_manifest_state(collection, dim, metric) {
            tracing::warn!(
                error = %err,
                collection,
                "failed to persist vector manifest metadata"
            );
        }
        Ok(())
    }

    pub fn vector_add(
        &self,
        collection: &str,
        id: &str,
        item: VectorItem,
    ) -> Result<(), EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        if self.0.vectors.get(collection, id)?.is_some() {
            return Err(VectorError::IdExists.into());
        }
        let data = serde_json::json!({
            "collection": collection,
            "id": id,
            "vector": item.vector.clone(),
            "meta": item.meta.clone(),
        });
        let append = self.0.events.append_guard();
        let event = self.0.events.next_record("vector_added", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
        drop(append);
        self.metrics().inc_events();
        self.metrics().inc_vector_op();
        Ok(())
    }

    pub fn vector_upsert(
        &self,
        collection: &str,
        id: &str,
        item: VectorItem,
    ) -> Result<(), EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let data = serde_json::json!({
            "collection": collection,
            "id": id,
            "vector": item.vector.clone(),
            "meta": item.meta.clone(),
        });
        let append = self.0.events.append_guard();
        let event = self.0.events.next_record("vector_upserted", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
        drop(append);
        self.metrics().inc_events();
        self.metrics().inc_vector_op();
        Ok(())
    }

    pub fn vector_update(
        &self,
        collection: &str,
        id: &str,
        vector: Option<Vec<f32>>,
        meta: Option<serde_json::Value>,
    ) -> Result<(), EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let current = self
            .0
            .vectors
            .get(collection, id)?
            .ok_or(VectorError::IdNotFound)?;
        let new_vec = vector.unwrap_or(current.vector);
        let new_meta = meta.unwrap_or(current.meta);
        let data = serde_json::json!({
            "collection": collection,
            "id": id,
            "vector": new_vec.clone(),
            "meta": new_meta.clone(),
        });
        let append = self.0.events.append_guard();
        let event = self.0.events.next_record("vector_updated", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
        drop(append);
        self.metrics().inc_events();
        self.metrics().inc_vector_op();
        Ok(())
    }

    pub fn vector_delete(&self, collection: &str, id: &str) -> Result<(), EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        if self.0.vectors.get(collection, id)?.is_none() {
            return Err(VectorError::IdNotFound.into());
        }
        let data = serde_json::json!({
            "collection": collection,
            "id": id,
        });
        let append = self.0.events.append_guard();
        let event = self.0.events.next_record("vector_deleted", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
        drop(append);
        self.metrics().inc_events();
        self.metrics().inc_vector_op();
        Ok(())
    }

    pub fn vector_compact_collection(&self, collection: &str) -> Result<bool, EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        Ok(self
            .0
            .vectors
            .compact_collection_with_options(collection, false)?)
    }

    pub fn vector_force_compact_collection(&self, collection: &str) -> Result<bool, EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        Ok(self
            .0
            .vectors
            .compact_collection_with_options(collection, true)?)
    }

    pub fn vector_retrain_ivf(&self, collection: &str, force: bool) -> Result<bool, EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        Ok(self.0.vectors.retrain_ivf(collection, force)?)
    }

    pub fn vector_build_disk_index(
        &self,
        collection: &str,
        params: DiskAnnBuildParams,
    ) -> Result<(), EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        Ok(self.0.vectors.build_disk_index(collection, params)?)
    }

    pub fn vector_drop_disk_index(&self, collection: &str) -> Result<(), EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        Ok(self.0.vectors.drop_disk_index(collection)?)
    }

    pub fn vector_disk_index_status(
        &self,
        collection: &str,
    ) -> Result<DiskIndexStatus, EngineError> {
        Ok(self.0.vectors.disk_index_status(collection)?)
    }

    pub fn vector_update_disk_index_params(
        &self,
        collection: &str,
        params: DiskAnnBuildParams,
    ) -> Result<DiskAnnBuildParams, EngineError> {
        let _ = self
            .0
            .vectors
            .get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        Ok(self
            .0
            .vectors
            .update_disk_index_params(collection, params)?)
    }

    pub fn vector_get(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<VectorItem>, VectorError> {
        self.0.vectors.get(collection, id)
    }

    pub fn vector_search(
        &self,
        collection: &str,
        req: SearchRequest,
    ) -> Result<Vec<SearchHit>, VectorError> {
        self.metrics().inc_vector_op();
        self.0.vectors.search(collection, req)
    }

    pub fn vector_search_batch(
        &self,
        collection: &str,
        requests: Vec<SearchRequest>,
    ) -> Vec<Result<Vec<SearchHit>, VectorError>> {
        use rayon::prelude::*;
        self.metrics().inc_vector_op();
        requests
            .into_par_iter()
            .map(|req| self.0.vectors.search(collection, req))
            .collect()
    }

    pub fn vector_scroll(
        &self,
        collection: &str,
        cursor: Option<&str>,
        limit: usize,
        include_vectors: bool,
    ) -> Result<(Vec<crate::vector::ScrollItem>, Option<String>), VectorError> {
        self.0
            .vectors
            .scroll(collection, cursor, limit, include_vectors)
    }

    pub fn vector_aggregate(
        &self,
        collection: &str,
        req: crate::vector::AggregateRequest,
    ) -> Result<Vec<crate::vector::AggregationBucket>, VectorError> {
        self.0.vectors.aggregate(collection, req)
    }

    fn expire_due_keys(&self, limit: usize) -> Result<usize, EngineError> {
        self.expire_due_keys_locked(now_ms(), limit)
    }

    fn expire_due_keys_locked(&self, now: u64, limit: usize) -> Result<usize, EngineError> {
        let keys = if let Some(db) = &self.0.state_db {
            db.expired_keys_due(now, limit).unwrap_or_default()
        } else {
            self.0.state.expired_keys(now, limit)
        };
        let mut expired = 0usize;
        for key in keys {
            let present = if let Some(db) = &self.0.state_db {
                db.exists_any(&key).unwrap_or(false)
            } else {
                self.0.state.exists_any(&key)
            };
            if !present {
                continue;
            }
            let data = serde_json::json!({
                "key": key,
                "reason": "ttl",
            });
            let append = self.0.events.append_guard();
            let event = self.0.events.next_record("state_deleted", data);
            if let Some(persist) = &self.0.persist {
                persist.append_event(&event)?;
            }
            if let Some(db) = &self.0.state_db {
                db.apply_state_deleted(&event)?;
            } else {
                let _ = self
                    .0
                    .state
                    .delete(event.data["key"].as_str().unwrap_or_default());
            }
            self.0.events.publish_record(event);
            drop(append);
            self.metrics().inc_events();
            self.metrics().inc_state_delete();
            self.metrics().add_ttl_expired(1);
            expired += 1;
        }
        Ok(expired)
    }
}

pub use events::{EventBus, EventRecord};
pub use metrics::Metrics;
pub use state::{StateError, StateItem};

fn now_ms() -> u64 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}

fn read_process_memory_rss() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_processes();
    let pid = sysinfo::get_current_pid().ok();
    pid.and_then(|p| sys.process(p))
        .map(|p| p.memory())
        .unwrap_or(0)
}

fn vector_manifest_key(collection: &str) -> String {
    format!("{VECTOR_MANIFEST_PREFIX}{collection}{VECTOR_MANIFEST_SUFFIX}")
}

fn collection_from_manifest_key(key: &str) -> Option<String> {
    if !key.starts_with(VECTOR_MANIFEST_PREFIX) || !key.ends_with(VECTOR_MANIFEST_SUFFIX) {
        return None;
    }
    let inner =
        &key[VECTOR_MANIFEST_PREFIX.len()..key.len().saturating_sub(VECTOR_MANIFEST_SUFFIX.len())];
    if inner.is_empty() {
        return None;
    }
    Some(inner.to_string())
}

#[derive(Default)]
struct VectorManifestMeta {
    dim: Option<usize>,
    metric: Option<Metric>,
    created_at_ms: Option<u64>,
    updated_at_ms: Option<u64>,
}

fn parse_vector_manifest_metadata(
    collection: &str,
    value: &serde_json::Value,
) -> Option<VectorManifestMeta> {
    if let Some(other) = value
        .get("collection")
        .and_then(|v| v.as_str())
        .filter(|name| !name.is_empty())
    {
        if other != collection {
            return None;
        }
    }
    let dim = value
        .get("dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let metric = value
        .get("metric")
        .cloned()
        .and_then(|v| serde_json::from_value::<Metric>(v).ok());
    let created_at_ms = value.get("created_at_ms").and_then(|v| v.as_u64());
    let updated_at_ms = value.get("updated_at_ms").and_then(|v| v.as_u64());
    if dim.is_none() && metric.is_none() && created_at_ms.is_none() && updated_at_ms.is_none() {
        return None;
    }
    Some(VectorManifestMeta {
        dim,
        metric,
        created_at_ms,
        updated_at_ms,
    })
}
