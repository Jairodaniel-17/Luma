mod events;
mod metrics;
mod persist;
mod state;
mod state_db;

use crate::config::Config;
use crate::vector::index::{DiskAnnBuildParams, DiskIndexStatus};
use crate::vector::{
    Metric, SearchHit, SearchRequest, VectorCollectionInfo, VectorError, VectorItem,
    VectorSettings, VectorStore,
};
use anyhow::Context;
use parking_lot::Mutex;
use std::collections::HashMap;
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
    commit_lock: Mutex<()>,
    shutdown: CancellationToken,
}

const VECTOR_MANIFEST_PREFIX: &str = "vector:";
const VECTOR_MANIFEST_SUFFIX: &str = ":manifest";
const VECTOR_MANIFEST_SCAN_LIMIT: usize = 4096;

impl Engine {
    pub fn new(config: Config, shutdown: CancellationToken) -> anyhow::Result<Self> {
        let events =
            events::EventBus::new(config.event_buffer_size, config.live_broadcast_capacity);
        let metrics = Arc::new(metrics::Metrics::default());

        let persist = match &config.data_dir {
            Some(dir) => Some(
                persist::Persist::new(
                    dir,
                    config.wal_segment_max_bytes,
                    config.wal_retention_segments,
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
            commit_lock: Mutex::new(()),
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

        Ok(engine)
    }

    pub fn shutdown(&self) {
        self.0.shutdown.cancel();
    }

    pub fn metrics_text(&self) -> String {
        self.0.metrics.render()
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
        if let Some(db) = &self.0.state_db {
            since_offset = since_offset.max(db.applied_offset().unwrap_or(0));
        }
        if let Some(snapshot) = persist.load_snapshot().context("read snapshot")? {
            self.0.events.set_next_offset(snapshot.last_offset + 1);
            since_offset = snapshot.last_offset;
        }

        let mut applied = 0usize;
        if let Some(db) = &self.0.state_db {
            let vectors = self.0.vectors.clone();
            let events = self.0.events.clone();
            persist
                .for_each_event_since(since_offset, |ev| {
                    match ev.event_type.as_str() {
                        "state_updated" => {
                            let _ = db.apply_state_updated(&ev);
                        }
                        "state_deleted" => {
                            let _ = db.apply_state_deleted(&ev);
                        }
                        "vector_collection_created"
                        | "vector_added"
                        | "vector_upserted"
                        | "vector_updated"
                        | "vector_deleted" => {
                            let _ = vectors.apply_event(&ev);
                        }
                        _ => {}
                    }
                    events.set_next_offset(ev.offset.saturating_add(1));
                    applied += 1;
                    true
                })
                .context("replay wal (db)")?;
        } else {
            applied = persist
                .replay_wal_since(since_offset, &self.0.state, &self.0.vectors, &self.0.events)
                .context("replay wal")?;
        }
        tracing::info!(applied, "replayed wal events");

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
                        let res = tokio::task::spawn_blocking(move || engine.snapshot_once()).await;
                        match res {
                            Ok(Ok(())) => tracing::info!("snapshot ok"),
                            Ok(Err(err)) => tracing::warn!(error = %err, "snapshot failed"),
                            Err(err) => tracing::warn!(error = %err, "snapshot task join failed"),
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
        let _g = self.0.commit_lock.lock();
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
        let snapshot = persist::Snapshot {
            last_offset: self.0.events.last_published_offset(),
        };
        persist.write_snapshot_and_rotate(&snapshot)
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

    pub fn list_state(&self, prefix: Option<&str>, limit: usize) -> Vec<state::StateItem> {
        if let Some(db) = &self.0.state_db {
            return db.list(prefix, limit).unwrap_or_default();
        }
        self.0.state.list(prefix, limit)
    }

    pub fn get_state(&self, key: &str) -> Option<state::StateItem> {
        if let Some(db) = &self.0.state_db {
            return db.get_state(key).ok().flatten();
        }
        self.0.state.get(key)
    }

    pub fn put_state(
        &self,
        key: String,
        value: serde_json::Value,
        ttl_ms: Option<u64>,
        if_revision: Option<u64>,
    ) -> Result<state::StateItem, EngineError> {
        let _g = self.0.commit_lock.lock();

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
        let event = self.0.events.next_record("state_updated", event_data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        if let Some(db) = &self.0.state_db {
            db.apply_state_updated(&event)?;
        }
        self.0.events.publish_record(event.clone());
        self.metrics().inc_events();

        self.metrics().inc_state_put();
        let item = if let Some(db) = &self.0.state_db {
            db.get_state(&key)?
                .ok_or_else(|| anyhow::anyhow!("state missing after put"))?
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
        let _g = self.0.commit_lock.lock();

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
        let event = self.0.events.next_record("state_deleted", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        if let Some(db) = &self.0.state_db {
            db.apply_state_deleted(&event)?;
        }
        self.0.events.publish_record(event);
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
        let _g = self.0.commit_lock.lock();
        if self.0.vectors.get_collection(collection).is_some() {
            return Err(VectorError::CollectionExists.into());
        }
        let data = serde_json::json!({
            "collection": collection,
            "dim": dim,
            "metric": metric,
        });
        let event = self.0.events.next_record("vector_collection_created", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.create_collection(collection, dim, metric)?;
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
        self.metrics().inc_events();
        self.metrics().inc_vector_op();
        drop(_g);
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
        let _g = self.0.commit_lock.lock();
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
        let event = self.0.events.next_record("vector_added", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
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
        let _g = self.0.commit_lock.lock();
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
        let event = self.0.events.next_record("vector_upserted", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
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
        let _g = self.0.commit_lock.lock();
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
        let event = self.0.events.next_record("vector_updated", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
        self.metrics().inc_events();
        self.metrics().inc_vector_op();
        Ok(())
    }

    pub fn vector_delete(&self, collection: &str, id: &str) -> Result<(), EngineError> {
        let _g = self.0.commit_lock.lock();
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
        let event = self.0.events.next_record("vector_deleted", data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&event)?;
        }
        self.0.vectors.apply_event(&event)?;
        self.0.events.publish_record(event);
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

    fn expire_due_keys(&self, limit: usize) -> Result<usize, EngineError> {
        let _g = self.0.commit_lock.lock();
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
            let live = if let Some(db) = &self.0.state_db {
                db.exists_live(&key).unwrap_or(false)
            } else {
                self.0.state.exists_live(&key)
            };
            if !live {
                continue;
            }
            let data = serde_json::json!({
                "key": key,
                "reason": "ttl",
            });
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
            self.metrics().inc_events();
            self.metrics().inc_state_delete();
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
