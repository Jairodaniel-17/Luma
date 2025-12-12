mod events;
mod metrics;
mod persist;
mod state;

use crate::config::Config;
use crate::vector::{Metric, SearchHit, SearchRequest, VectorError, VectorItem, VectorStore};
use anyhow::Context;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone)]
pub struct Engine(Arc<Inner>);

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("persistence error: {0}")]
    Persistence(#[from] std::io::Error),
    #[error(transparent)]
    State(#[from] state::StateError),
    #[error(transparent)]
    Vector(#[from] VectorError),
}

struct Inner {
    config: Config,
    state: state::StateStore,
    vectors: VectorStore,
    events: events::EventBus,
    metrics: Arc<metrics::Metrics>,
    persist: Option<persist::Persist>,
    commit_lock: Mutex<()>,
}

impl Engine {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let events = events::EventBus::new(config.event_buffer_size, config.live_broadcast_capacity);
        let metrics = Arc::new(metrics::Metrics::default());

        let persist = match &config.data_dir {
            Some(dir) => Some(
                persist::Persist::new(dir, config.wal_segment_max_bytes, config.wal_retention_segments)
                    .context("init persistence")?,
            ),
            None => None,
        };

        let state = state::StateStore::new();
        let vectors = VectorStore::new();

        let engine = Self(Arc::new(Inner {
            config: config.clone(),
            state,
            vectors,
            events,
            metrics,
            persist,
            commit_lock: Mutex::new(()),
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

    pub fn shutdown(&self) {}

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
        if let Some(snapshot) = persist.load_snapshot().context("read snapshot")? {
            self.0.state.load_snapshot(snapshot.state)?;
            self.0.vectors.load_snapshot(snapshot.vectors)?;
            self.0.events.set_next_offset(snapshot.last_offset + 1);
            since_offset = snapshot.last_offset;
        }

        let applied = persist
            .replay_wal_since(since_offset, &self.0.state, &self.0.vectors, &self.0.events)
            .context("replay wal")?;
        tracing::info!(applied, "replayed wal events");

        Ok(())
    }

    fn start_snapshot_task_if_runtime(&self) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let interval_secs = self.0.config.snapshot_interval_secs;
        let engine = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            loop {
                interval.tick().await;
                let engine = engine.clone();
                let res = tokio::task::spawn_blocking(move || engine.snapshot_once()).await;
                match res {
                    Ok(Ok(())) => tracing::info!("snapshot ok"),
                    Ok(Err(err)) => tracing::warn!(error = %err, "snapshot failed"),
                    Err(err) => tracing::warn!(error = %err, "snapshot task join failed"),
                };
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
            state: self.0.state.snapshot(),
            vectors: self.0.vectors.snapshot(),
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
        let engine = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                interval.tick().await;
                let engine = engine.clone();
                let res = tokio::task::spawn_blocking(move || engine.expire_due_keys(1000)).await;
                match res {
                    Ok(Ok(expired)) if expired > 0 => tracing::info!(expired, "ttl expired"),
                    Ok(Ok(_)) => {}
                    Ok(Err(err)) => tracing::warn!(error = %err, "ttl task failed"),
                    Err(err) => tracing::warn!(error = %err, "ttl task join failed"),
                }
            }
        });
    }

    pub fn list_state(&self, prefix: Option<&str>, limit: usize) -> Vec<state::StateItem> {
        self.0.state.list(prefix, limit)
    }

    pub fn get_state(&self, key: &str) -> Option<state::StateItem> {
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
        if let Some((_rev, Some(exp))) = self.0.state.peek_meta(&key) {
            if exp <= now {
                let _ = self.append_event_durable(
                    "state_deleted",
                    serde_json::json!({
                        "key": key.clone(),
                        "reason": "ttl",
                    }),
                )?;
                let _ = self.0.state.delete(&key);
                self.metrics().inc_state_delete();
            }
        }

        let revision = self.0.state.prepare_put_revision(&key, if_revision)?;
        let expires_at_ms = ttl_ms.map(|ttl| now.saturating_add(ttl));

        let event_data = serde_json::json!({
            "key": key,
            "revision": revision,
            "value": value,
            "expires_at_ms": expires_at_ms,
        });
        let value = event_data["value"].clone();
        let key = event_data["key"].as_str().unwrap_or_default().to_string();
        let _event = self.append_event_durable("state_updated", event_data)?;

        self.metrics().inc_state_put();
        let item = self
            .0
            .state
            .apply_put_with_revision(key, value, revision, expires_at_ms);
        Ok(item)
    }

    pub fn delete_state(&self, key: &str) -> Result<bool, EngineError> {
        self.delete_state_with_reason(key, "explicit")
    }

    pub fn delete_state_with_reason(&self, key: &str, reason: &'static str) -> Result<bool, EngineError> {
        let _g = self.0.commit_lock.lock();

        if !self.0.state.exists_live(key) {
            return Ok(false);
        }

        let data = serde_json::json!({
            "key": key,
            "reason": reason,
        });
        let _event = self.append_event_durable("state_deleted", data)?;

        let deleted = self.0.state.delete(key);
        if deleted {
            self.metrics().inc_state_delete();
        }
        Ok(deleted)
    }

    pub fn vectors(&self) -> &VectorStore {
        &self.0.vectors
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
        let _event = self.append_event_durable("vector_collection_created", data)?;
        self.0.vectors.create_collection(collection, dim, metric)?;
        self.metrics().inc_vector_op();
        Ok(())
    }

    pub fn vector_add(&self, collection: &str, id: &str, item: VectorItem) -> Result<(), EngineError> {
        let _g = self.0.commit_lock.lock();
        let _ = self.0.vectors.get_collection(collection).ok_or(VectorError::CollectionNotFound)?;
        if self.0.vectors.get(collection, id)?.is_some() {
            return Err(VectorError::IdExists.into());
        }
        let event_data = serde_json::json!({
            "collection": collection,
            "id": id,
            "vector": item.vector.clone(),
            "meta": item.meta.clone(),
        });
        let _event = self.append_event_durable("vector_added", event_data)?;
        self.0.vectors.add(collection, id, item)?;
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
        let _ = self.0.vectors.get_collection(collection).ok_or(VectorError::CollectionNotFound)?;
        let event_data = serde_json::json!({
            "collection": collection,
            "id": id,
            "vector": item.vector.clone(),
            "meta": item.meta.clone(),
        });
        let _event = self.append_event_durable("vector_upserted", event_data)?;
        self.0.vectors.upsert(collection, id, item)?;
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
        let _ = self.0.vectors.get_collection(collection).ok_or(VectorError::CollectionNotFound)?;
        let current = self
            .0
            .vectors
            .get(collection, id)?
            .ok_or(VectorError::IdNotFound)?;
        let new_vec = vector.unwrap_or(current.vector);
        let new_meta = meta.unwrap_or(current.meta);
        let event_data = serde_json::json!({
            "collection": collection,
            "id": id,
            "vector": new_vec.clone(),
            "meta": new_meta.clone(),
        });
        let _event = self.append_event_durable("vector_updated", event_data)?;
        self.0.vectors.upsert(
            collection,
            id,
            VectorItem {
                vector: new_vec,
                meta: new_meta,
            },
        )?;
        self.metrics().inc_vector_op();
        Ok(())
    }

    pub fn vector_delete(&self, collection: &str, id: &str) -> Result<(), EngineError> {
        let _g = self.0.commit_lock.lock();
        let _ = self.0.vectors.get_collection(collection).ok_or(VectorError::CollectionNotFound)?;
        if self.0.vectors.get(collection, id)?.is_none() {
            return Err(VectorError::IdNotFound.into());
        }
        let data = serde_json::json!({
            "collection": collection,
            "id": id,
        });
        let _event = self.append_event_durable("vector_deleted", data)?;
        self.0.vectors.delete(collection, id)?;
        self.metrics().inc_vector_op();
        Ok(())
    }

    pub fn vector_get(&self, collection: &str, id: &str) -> Result<Option<VectorItem>, VectorError> {
        self.0.vectors.get(collection, id)
    }

    pub fn vector_search(&self, collection: &str, req: SearchRequest) -> Result<Vec<SearchHit>, VectorError> {
        self.metrics().inc_vector_op();
        self.0.vectors.search(collection, req)
    }

    fn expire_due_keys(&self, limit: usize) -> Result<usize, EngineError> {
        let _g = self.0.commit_lock.lock();
        self.expire_due_keys_locked(now_ms(), limit)
    }

    fn expire_due_keys_locked(&self, now: u64, limit: usize) -> Result<usize, EngineError> {
        let keys = self.0.state.expired_keys(now, limit);
        let mut expired = 0usize;
        for key in keys {
            if let Some((_rev, Some(exp))) = self.0.state.peek_meta(&key) {
                if exp > now {
                    continue;
                }
                let _ = self.append_event_durable(
                    "state_deleted",
                    serde_json::json!({
                        "key": key,
                        "reason": "ttl",
                    }),
                )?;
                if self.0.state.delete(&key) {
                    self.metrics().inc_state_delete();
                    expired += 1;
                }
            }
        }
        Ok(expired)
    }

    fn append_event_durable(
        &self,
        event_type: &'static str,
        data: serde_json::Value,
    ) -> Result<events::EventRecord, EngineError> {
        let record = self.0.events.next_record(event_type, data);
        if let Some(persist) = &self.0.persist {
            persist.append_event(&record)?;
        }
        self.0.events.publish_record(record.clone());
        self.metrics().inc_events();
        Ok(record)
    }
}

pub use state::{StateError, StateItem};
pub use events::{EventBus, EventRecord};
pub use metrics::Metrics;

fn now_ms() -> u64 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}
