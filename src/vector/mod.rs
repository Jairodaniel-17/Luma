mod persist;

use crate::vector::persist::{CollectionLayout, Manifest, Record, RecordOp};
use anyhow::Context;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct VectorStore(Arc<Inner>);

struct Inner {
    data_dir: Option<PathBuf>,
    collections: RwLock<HashMap<String, Collection>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    pub dim: usize,
    pub metric: Metric,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorItem {
    pub vector: Vec<f32>,
    pub meta: serde_json::Value,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    Cosine,
    Dot,
}

#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("collection not found")]
    CollectionNotFound,
    #[error("collection already exists")]
    CollectionExists,
    #[error("id not found")]
    IdNotFound,
    #[error("id already exists")]
    IdExists,
    #[error("vector dim mismatch")]
    DimMismatch,
    #[error("invalid collection manifest")]
    InvalidManifest,
    #[error("persistence error")]
    Persistence,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    pub filters: Option<serde_json::Value>,
    pub include_meta: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub meta: Option<serde_json::Value>,
}

struct Collection {
    dim: usize,
    metric: Metric,
    layout: Option<CollectionLayout>,
    manifest: Manifest,
    items: HashMap<String, VectorItem>,
    applied_offset: u64,
    hnsw: HnswIndex,
    data_ids: HashMap<String, usize>,
    id_by_data_id: Vec<String>,
    deleted_data_id: Vec<bool>,
    hnsw_capacity: usize,
}

enum HnswIndex {
    Cosine(Hnsw<'static, f32, anndists::dist::distances::DistCosine>),
    Dot(Hnsw<'static, f32, anndists::dist::distances::DistDot>),
}

impl VectorStore {
    pub fn new() -> Self {
        Self(Arc::new(Inner {
            data_dir: None,
            collections: RwLock::new(HashMap::new()),
        }))
    }

    pub fn open(data_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let vectors_dir = data_dir.join("vectors");
        std::fs::create_dir_all(&vectors_dir)?;

        let mut collections = HashMap::new();
        for entry in std::fs::read_dir(&vectors_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let layout = CollectionLayout::new(&vectors_dir, &name);
            let (manifest, items, applied_offset) = persist::load_collection(&layout)
                .with_context(|| format!("load vector collection {name}"))?;
            let mut c = Collection::new(Some(layout), manifest, items, applied_offset)?;
            c.rebuild_index();
            collections.insert(name, c);
        }

        Ok(Self(Arc::new(Inner {
            data_dir: Some(data_dir),
            collections: RwLock::new(collections),
        })))
    }

    pub fn applied_offset(&self) -> u64 {
        let cols = self.0.collections.read();
        cols.values().map(|c| c.applied_offset).max().unwrap_or(0)
    }

    pub fn get_collection(&self, name: &str) -> Option<(usize, Metric)> {
        let cols = self.0.collections.read();
        cols.get(name).map(|c| (c.dim, c.metric))
    }

    pub fn create_collection(
        &self,
        name: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        if cols.contains_key(name) {
            return Err(VectorError::CollectionExists);
        }
        let layout = self.layout_for(name);
        let (manifest, items, applied_offset) = if let Some(layout) = &layout {
            persist::init_collection(layout, dim, metric).map_err(|_| VectorError::Persistence)?;
            persist::load_collection(layout).map_err(|_| VectorError::Persistence)?
        } else {
            (Manifest::new(dim, metric), HashMap::new(), 0)
        };
        let mut c = Collection::new(layout.clone(), manifest, items, applied_offset)?;
        c.rebuild_index();
        cols.insert(name.to_string(), c);
        Ok(())
    }

    pub fn get(&self, collection: &str, id: &str) -> Result<Option<VectorItem>, VectorError> {
        let cols = self.0.collections.read();
        let c = cols
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        Ok(c.items.get(id).cloned())
    }

    pub fn apply_event(&self, ev: &crate::engine::EventRecord) -> Result<(), VectorError> {
        match ev.event_type.as_str() {
            "vector_collection_created" => {
                let name = ev
                    .data
                    .get("collection")
                    .and_then(|v| v.as_str())
                    .ok_or(VectorError::InvalidManifest)?;
                let dim = ev.data.get("dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let metric: Metric = serde_json::from_value(
                    ev.data
                        .get("metric")
                        .cloned()
                        .unwrap_or(serde_json::Value::String("cosine".into())),
                )
                .map_err(|_| VectorError::InvalidManifest)?;

                let mut cols = self.0.collections.write();
                if let Some(existing) = cols.get_mut(name) {
                    if existing.dim != dim || existing.metric != metric {
                        return Err(VectorError::InvalidManifest);
                    }
                    existing.mark_applied_offset(ev.offset)?;
                    return Ok(());
                }

                let layout = self.layout_for(name);
                let (manifest, items, applied_offset) = if let Some(layout) = &layout {
                    persist::init_collection(layout, dim, metric)
                        .map_err(|_| VectorError::Persistence)?;
                    persist::load_collection(layout).map_err(|_| VectorError::Persistence)?
                } else {
                    (Manifest::new(dim, metric), HashMap::new(), 0)
                };
                let mut c = Collection::new(layout.clone(), manifest, items, applied_offset)?;
                c.mark_applied_offset(ev.offset)?;
                c.rebuild_index();
                cols.insert(name.to_string(), c);
                Ok(())
            }
            "vector_added" | "vector_upserted" | "vector_updated" | "vector_deleted" => {
                let collection = ev
                    .data
                    .get("collection")
                    .and_then(|v| v.as_str())
                    .ok_or(VectorError::InvalidManifest)?;
                let id = ev
                    .data
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or(VectorError::InvalidManifest)?;

                let mut cols = self.0.collections.write();
                let c = cols
                    .get_mut(collection)
                    .ok_or(VectorError::CollectionNotFound)?;
                if ev.offset <= c.applied_offset {
                    return Ok(());
                }

                match ev.event_type.as_str() {
                    "vector_deleted" => {
                        let record = Record {
                            offset: ev.offset,
                            op: RecordOp::Delete,
                            id: id.to_string(),
                            vector: None,
                            meta: None,
                        };
                        c.apply_record(record, None)?;
                    }
                    _ => {
                        let vector: Vec<f32> = serde_json::from_value(
                            ev.data
                                .get("vector")
                                .cloned()
                                .unwrap_or(serde_json::Value::Array(vec![])),
                        )
                        .map_err(|_| VectorError::InvalidManifest)?;
                        let meta = ev
                            .data
                            .get("meta")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        let record = Record {
                            offset: ev.offset,
                            op: RecordOp::Upsert,
                            id: id.to_string(),
                            vector: Some(vector),
                            meta: Some(meta),
                        };
                        c.apply_record(record, None)?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn add(&self, collection: &str, id: &str, item: VectorItem) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols
            .get_mut(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        if c.items.contains_key(id) {
            return Err(VectorError::IdExists);
        }
        if item.vector.len() != c.dim {
            return Err(VectorError::DimMismatch);
        }
        let record = Record {
            offset: 0,
            op: RecordOp::Upsert,
            id: id.to_string(),
            vector: Some(item.vector),
            meta: Some(item.meta),
        };
        c.apply_record(record, Some(ApplyMode::InMemoryOnly))?;
        Ok(())
    }

    pub fn upsert(&self, collection: &str, id: &str, item: VectorItem) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols
            .get_mut(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        if item.vector.len() != c.dim {
            return Err(VectorError::DimMismatch);
        }
        let record = Record {
            offset: 0,
            op: RecordOp::Upsert,
            id: id.to_string(),
            vector: Some(item.vector),
            meta: Some(item.meta),
        };
        c.apply_record(record, Some(ApplyMode::InMemoryOnly))?;
        Ok(())
    }

    pub fn update(
        &self,
        collection: &str,
        id: &str,
        vector: Option<Vec<f32>>,
        meta: Option<serde_json::Value>,
    ) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols
            .get_mut(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let current = c.items.get(id).cloned().ok_or(VectorError::IdNotFound)?;
        let new_vec = vector.unwrap_or(current.vector);
        if new_vec.len() != c.dim {
            return Err(VectorError::DimMismatch);
        }
        let new_meta = meta.unwrap_or(current.meta);
        let record = Record {
            offset: 0,
            op: RecordOp::Upsert,
            id: id.to_string(),
            vector: Some(new_vec),
            meta: Some(new_meta),
        };
        c.apply_record(record, Some(ApplyMode::InMemoryOnly))?;
        Ok(())
    }

    pub fn delete(&self, collection: &str, id: &str) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols
            .get_mut(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        if !c.items.contains_key(id) {
            return Err(VectorError::IdNotFound);
        }
        let record = Record {
            offset: 0,
            op: RecordOp::Delete,
            id: id.to_string(),
            vector: None,
            meta: None,
        };
        c.apply_record(record, Some(ApplyMode::InMemoryOnly))?;
        Ok(())
    }

    pub fn search(
        &self,
        collection: &str,
        req: SearchRequest,
    ) -> Result<Vec<SearchHit>, VectorError> {
        let cols = self.0.collections.read();
        let c = cols
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        c.search(req)
    }

    fn layout_for(&self, collection: &str) -> Option<CollectionLayout> {
        let base = self.0.data_dir.as_ref()?.join("vectors");
        Some(CollectionLayout::new(&base, collection))
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
enum ApplyMode {
    InMemoryOnly,
}

impl Collection {
    fn new(
        layout: Option<CollectionLayout>,
        manifest: Manifest,
        items: HashMap<String, VectorItem>,
        applied_offset: u64,
    ) -> Result<Self, VectorError> {
        let dim = manifest.dim;
        let metric = manifest.metric;
        let mut c = Self {
            dim,
            metric,
            layout,
            manifest,
            items,
            applied_offset,
            hnsw: make_hnsw(metric, 16, 1024, 16, 200),
            data_ids: HashMap::new(),
            id_by_data_id: Vec::new(),
            deleted_data_id: Vec::new(),
            hnsw_capacity: 1024,
        };
        c.rebuild_index();
        Ok(c)
    }

    fn rebuild_index(&mut self) {
        let baseline = (self.manifest.upsert_count as usize).max(self.items.len());
        let capacity = (baseline.max(1) * 2).max(1024);
        self.hnsw_capacity = capacity;
        self.data_ids.clear();
        self.id_by_data_id.clear();
        self.deleted_data_id.clear();
        self.hnsw = make_hnsw(self.metric, 16, capacity, 16, 200);

        for (data_id, (id, item)) in self.items.iter().enumerate() {
            let vec = normalize_if_needed(self.metric, item.vector.clone());
            self.data_ids.insert(id.clone(), data_id);
            self.id_by_data_id.push(id.clone());
            self.deleted_data_id.push(false);
            insert_into_hnsw(&mut self.hnsw, vec, data_id);
        }
    }

    fn apply_record(
        &mut self,
        mut record: Record,
        mode: Option<ApplyMode>,
    ) -> Result<(), VectorError> {
        let normalized_vec = if record.op == RecordOp::Upsert {
            let Some(vec) = record.vector.take() else {
                return Err(VectorError::InvalidManifest);
            };
            if vec.len() != self.dim {
                return Err(VectorError::DimMismatch);
            }
            let normalized = normalize_if_needed(self.metric, vec);
            record.vector = Some(normalized.clone());
            Some(normalized)
        } else {
            None
        };

        if let Some(layout) = &self.layout {
            if mode.is_none() {
                let appended = persist::append_record(layout, &record)
                    .map_err(|_| VectorError::Persistence)?;
                self.manifest.file_len = self.manifest.file_len.saturating_add(appended);
            }
        }

        if record.offset > 0 {
            self.manifest.applied_offset = self.manifest.applied_offset.max(record.offset);
            self.applied_offset = self.applied_offset.max(record.offset);
        }
        self.manifest.total_records = self.manifest.total_records.saturating_add(1);

        match record.op {
            RecordOp::Delete => {
                if self.items.remove(&record.id).is_some() {
                    self.manifest.live_count = self.manifest.live_count.saturating_sub(1);
                }
                if let Some(old) = self.data_ids.remove(&record.id) {
                    if old < self.deleted_data_id.len() {
                        self.deleted_data_id[old] = true;
                    }
                }
            }
            RecordOp::Upsert => {
                self.manifest.upsert_count = self.manifest.upsert_count.saturating_add(1);
                let vec = normalized_vec.clone().ok_or(VectorError::InvalidManifest)?;
                let meta = record.meta.take().unwrap_or(serde_json::Value::Null);
                let new_item = VectorItem {
                    vector: vec.clone(),
                    meta,
                };
                if let Some(old) = self.data_ids.get(&record.id).cloned() {
                    if old < self.deleted_data_id.len() {
                        self.deleted_data_id[old] = true;
                    }
                }
                let existed = self.items.insert(record.id.clone(), new_item).is_some();
                if !existed {
                    self.manifest.live_count += 1;
                }

                if self.id_by_data_id.len() + 1 > self.hnsw_capacity {
                    self.rebuild_index();
                } else {
                    let data_id = self.id_by_data_id.len();
                    self.id_by_data_id.push(record.id.clone());
                    self.deleted_data_id.push(false);
                    self.data_ids.insert(record.id, data_id);
                    insert_into_hnsw(&mut self.hnsw, vec, data_id);
                }
            }
        }

        self.manifest.live_count = self.items.len();

        if self.layout.is_some() && mode.is_none() {
            self.persist_manifest()
                .map_err(|_| VectorError::Persistence)?;
        }

        Ok(())
    }

    fn persist_manifest(&self) -> std::io::Result<()> {
        if let Some(layout) = &self.layout {
            persist::store_manifest(layout, &self.manifest)?;
        }
        Ok(())
    }

    fn mark_applied_offset(&mut self, offset: u64) -> Result<(), VectorError> {
        if offset <= self.applied_offset {
            return Ok(());
        }
        self.applied_offset = offset;
        self.manifest.applied_offset = offset;
        if self.layout.is_some() {
            self.persist_manifest()
                .map_err(|_| VectorError::Persistence)?;
        }
        Ok(())
    }

    fn search(&self, req: SearchRequest) -> Result<Vec<SearchHit>, VectorError> {
        if req.vector.len() != self.dim {
            return Err(VectorError::DimMismatch);
        }
        let include_meta = req.include_meta.unwrap_or(false);
        let k = req.k.max(1);
        let query = normalize_if_needed(self.metric, req.vector);
        if self.items.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_k = (k * 20).min(self.items.len()).max(k);
        let ef = (candidate_k * 2).clamp(50, 10_000);

        let neighbours = match &self.hnsw {
            HnswIndex::Cosine(h) => h.search(query.as_slice(), candidate_k, ef),
            HnswIndex::Dot(h) => h.search(query.as_slice(), candidate_k, ef),
        };

        let mut hits = Vec::new();
        for n in neighbours {
            let data_id = n.d_id;
            if data_id >= self.id_by_data_id.len() {
                continue;
            }
            if self.deleted_data_id.get(data_id).copied().unwrap_or(true) {
                continue;
            }
            let id = &self.id_by_data_id[data_id];
            let Some(item) = self.items.get(id) else {
                continue;
            };
            if !matches_filters(&item.meta, req.filters.as_ref()) {
                continue;
            }
            let score = 1.0 - n.distance;
            hits.push(SearchHit {
                id: id.clone(),
                score,
                meta: include_meta.then(|| item.meta.clone()),
            });
            if hits.len() >= k {
                break;
            }
        }

        Ok(hits)
    }
}

fn matches_filters(meta: &serde_json::Value, filters: Option<&serde_json::Value>) -> bool {
    let Some(filters) = filters else { return true };
    let serde_json::Value::Object(f) = filters else {
        return false;
    };
    let serde_json::Value::Object(m) = meta else {
        return false;
    };

    for (k, v) in f.iter() {
        match m.get(k) {
            Some(mv) if mv == v => {}
            _ => return false,
        }
    }
    true
}

fn normalize_if_needed(metric: Metric, mut v: Vec<f32>) -> Vec<f32> {
    if metric == Metric::Dot {
        anndists::dist::distances::l2_normalize(v.as_mut_slice());
    }
    v
}

fn make_hnsw(
    metric: Metric,
    max_nb_conn: usize,
    max_elem: usize,
    nb_layer: usize,
    ef_c: usize,
) -> HnswIndex {
    match metric {
        Metric::Cosine => {
            HnswIndex::Cosine(Hnsw::<f32, anndists::dist::distances::DistCosine>::new(
                max_nb_conn,
                max_elem,
                nb_layer,
                ef_c,
                anndists::dist::distances::DistCosine {},
            ))
        }
        Metric::Dot => HnswIndex::Dot(Hnsw::<f32, anndists::dist::distances::DistDot>::new(
            max_nb_conn,
            max_elem,
            nb_layer,
            ef_c,
            anndists::dist::distances::DistDot {},
        )),
    }
}

fn insert_into_hnsw(hnsw: &mut HnswIndex, v: Vec<f32>, data_id: usize) {
    match hnsw {
        HnswIndex::Cosine(h) => h.insert((&v, data_id)),
        HnswIndex::Dot(h) => h.insert((&v, data_id)),
    }
}
