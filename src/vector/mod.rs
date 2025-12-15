mod persist;

use crate::vector::persist::{CollectionLayout, Manifest, Record, RecordOp};
use anyhow::Context;
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct VectorStore(Arc<Inner>);

struct Inner {
    data_dir: Option<PathBuf>,
    collections: RwLock<HashMap<String, Collection>>,
}

const DEFAULT_SEGMENT_MAX: usize = 8_192;

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
    segments: Vec<SegmentIndex>,
    item_segments: HashMap<String, usize>,
    segment_max_items: usize,
    keyword_index: HashMap<String, HashMap<String, HashSet<String>>>,
}

enum HnswIndex {
    Cosine(Hnsw<'static, f32, anndists::dist::distances::DistCosine>),
    Dot(Hnsw<'static, f32, anndists::dist::distances::DistDot>),
}

struct SegmentIndex {
    hnsw: HnswIndex,
    data_ids: HashMap<String, usize>,
    id_by_data_id: Vec<String>,
    deleted: Vec<bool>,
    live: usize,
    capacity: usize,
}

impl SegmentIndex {
    fn new(metric: Metric, capacity: usize) -> Self {
        Self {
            hnsw: make_hnsw(metric, 16, capacity.max(1024), 16, 200),
            data_ids: HashMap::new(),
            id_by_data_id: Vec::new(),
            deleted: Vec::new(),
            live: 0,
            capacity: capacity.max(1024),
        }
    }

    fn insert(&mut self, id: String, vector: Vec<f32>) {
        let data_id = self.id_by_data_id.len();
        self.id_by_data_id.push(id.clone());
        self.deleted.push(false);
        self.data_ids.insert(id, data_id);
        insert_into_hnsw(&mut self.hnsw, vector, data_id);
        self.live = self.live.saturating_add(1);
    }

    fn mark_deleted(&mut self, id: &str) {
        if let Some(idx) = self.data_ids.remove(id) {
            if idx < self.deleted.len() && !self.deleted[idx] {
                self.deleted[idx] = true;
                self.live = self.live.saturating_sub(1);
            }
        }
    }

    fn search_candidates(&self, query: &[f32], candidate_k: usize) -> Vec<(String, f32)> {
        if self.live == 0 {
            return Vec::new();
        }
        let neighbours = match &self.hnsw {
            HnswIndex::Cosine(h) => h.search(
                query,
                candidate_k,
                candidate_k.saturating_mul(2).clamp(50, 10_000),
            ),
            HnswIndex::Dot(h) => h.search(
                query,
                candidate_k,
                candidate_k.saturating_mul(2).clamp(50, 10_000),
            ),
        };
        let mut hits = Vec::new();
        for n in neighbours {
            let data_id = n.d_id;
            if data_id >= self.id_by_data_id.len() {
                continue;
            }
            if self.deleted.get(data_id).copied().unwrap_or(true) {
                continue;
            }
            let id = self.id_by_data_id[data_id].clone();
            let score = 1.0 - n.distance;
            hits.push((id, score));
            if hits.len() >= candidate_k {
                break;
            }
        }
        hits
    }
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

    pub fn vacuum_collection(&self, collection: &str) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols
            .get_mut(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let layout = c.layout.clone().ok_or(VectorError::Persistence)?;
        let updated = persist::rewrite_collection(&layout, &c.manifest, &c.items)
            .map_err(|_| VectorError::Persistence)?;
        c.manifest = updated;
        c.rebuild_index();
        Ok(())
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
            segments: Vec::new(),
            item_segments: HashMap::new(),
            segment_max_items: DEFAULT_SEGMENT_MAX,
            keyword_index: HashMap::new(),
        };
        c.rebuild_index();
        Ok(c)
    }

    fn rebuild_index(&mut self) {
        self.keyword_index.clear();
        let metas: Vec<(String, serde_json::Value)> = self
            .items
            .iter()
            .map(|(id, item)| (id.clone(), item.meta.clone()))
            .collect();
        for (id, meta) in metas {
            self.add_meta_to_index(&id, &meta);
        }
        self.rebuild_segments();
    }

    fn rebuild_segments(&mut self) {
        self.item_segments.clear();
        self.segments.clear();
        if self.items.is_empty() {
            self.segments
                .push(SegmentIndex::new(self.metric, self.segment_max_items));
            return;
        }
        let mut current = SegmentIndex::new(self.metric, self.segment_max_items);
        for (id, item) in self.items.iter() {
            if current.live >= current.capacity {
                self.segments.push(current);
                current = SegmentIndex::new(self.metric, self.segment_max_items);
            }
            current.insert(id.clone(), item.vector.clone());
            let idx = self.segments.len();
            self.item_segments.insert(id.clone(), idx);
        }
        self.segments.push(current);
        if self.segments.is_empty() {
            self.segments
                .push(SegmentIndex::new(self.metric, self.segment_max_items));
        }
    }

    fn ensure_active_segment(&mut self) -> usize {
        if self.segments.is_empty() {
            self.segments
                .push(SegmentIndex::new(self.metric, self.segment_max_items));
        }
        let last_idx = self.segments.len() - 1;
        if self.segments[last_idx].live >= self.segments[last_idx].capacity {
            self.segments
                .push(SegmentIndex::new(self.metric, self.segment_max_items));
            return self.segments.len() - 1;
        }
        last_idx
    }

    fn insert_into_segments(&mut self, id: &str, vector: Vec<f32>) {
        if let Some(seg_idx) = self.item_segments.remove(id) {
            if let Some(seg) = self.segments.get_mut(seg_idx) {
                seg.mark_deleted(id);
            }
        }
        let idx = self.ensure_active_segment();
        if let Some(seg) = self.segments.get_mut(idx) {
            seg.insert(id.to_string(), vector);
            self.item_segments.insert(id.to_string(), idx);
        }
    }

    fn remove_from_segments(&mut self, id: &str) {
        if let Some(seg_idx) = self.item_segments.remove(id) {
            if let Some(seg) = self.segments.get_mut(seg_idx) {
                seg.mark_deleted(id);
            }
        }
    }

    fn add_meta_to_index(&mut self, id: &str, meta: &serde_json::Value) {
        let Some(obj) = meta.as_object() else {
            return;
        };
        for (k, v) in obj {
            let Some(value) = v.as_str() else {
                continue;
            };
            self.keyword_index
                .entry(k.clone())
                .or_default()
                .entry(value.to_string())
                .or_default()
                .insert(id.to_string());
        }
    }

    fn remove_meta_from_index(&mut self, id: &str, meta: Option<&serde_json::Value>) {
        let Some(meta) = meta else { return };
        let Some(obj) = meta.as_object() else { return };
        for (k, v) in obj {
            let Some(value) = v.as_str() else {
                continue;
            };
            if let Some(by_value) = self.keyword_index.get_mut(k) {
                if let Some(set) = by_value.get_mut(value) {
                    set.remove(id);
                    if set.is_empty() {
                        by_value.remove(value);
                    }
                }
                if by_value.is_empty() {
                    self.keyword_index.remove(k);
                }
            }
        }
    }

    fn keyword_candidates(&self, filters: &serde_json::Value) -> Option<HashSet<String>> {
        let obj = filters.as_object()?;
        let mut current: Option<HashSet<String>> = None;
        for (k, v) in obj {
            let Some(value) = v.as_str() else {
                return None;
            };
            let Some(by_value) = self.keyword_index.get(k) else {
                return Some(HashSet::new());
            };
            let Some(ids) = by_value.get(value) else {
                return Some(HashSet::new());
            };
            let ids_cloned: HashSet<String> = ids.iter().cloned().collect();
            current = match current {
                None => Some(ids_cloned),
                Some(mut acc) => {
                    acc.retain(|id| ids.contains(id));
                    Some(acc)
                }
            };
        }
        current
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
                let removed = self.items.remove(&record.id);
                if let Some(old) = removed.as_ref() {
                    self.remove_meta_from_index(&record.id, Some(&old.meta));
                }
                self.remove_from_segments(&record.id);
                if removed.is_some() {
                    self.manifest.live_count = self.manifest.live_count.saturating_sub(1);
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
                let previous = self.items.insert(record.id.clone(), new_item.clone());
                if let Some(prev) = previous.as_ref() {
                    self.remove_meta_from_index(&record.id, Some(&prev.meta));
                }
                self.add_meta_to_index(&record.id, &new_item.meta);
                self.insert_into_segments(&record.id, new_item.vector.clone());
                if previous.is_none() {
                    self.manifest.live_count += 1;
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

        let filter_candidates = req
            .filters
            .as_ref()
            .and_then(|f| self.keyword_candidates(f));
        if let Some(ref set) = filter_candidates {
            if set.is_empty() {
                return Ok(Vec::new());
            }
            if set.len() <= 512 {
                return Ok(self.search_subset_bruteforce(
                    query.as_slice(),
                    include_meta,
                    set,
                    req.filters.as_ref(),
                    k,
                ));
            }
        }

        let candidate_k = (k * 10).min(self.items.len()).max(k);
        let mut combined = Vec::new();
        for segment in &self.segments {
            combined.extend(segment.search_candidates(query.as_slice(), candidate_k));
        }
        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut hits = Vec::new();
        let mut seen = HashSet::new();
        for (id, score) in combined {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Some(ref set) = filter_candidates {
                if !set.contains(&id) {
                    continue;
                }
            }
            let Some(item) = self.items.get(&id) else {
                continue;
            };
            if !matches_filters(&item.meta, req.filters.as_ref()) {
                continue;
            }
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

    fn search_subset_bruteforce(
        &self,
        query: &[f32],
        include_meta: bool,
        candidates: &HashSet<String>,
        filters: Option<&serde_json::Value>,
        k: usize,
    ) -> Vec<SearchHit> {
        let mut scored = Vec::new();
        for id in candidates {
            let Some(item) = self.items.get(id) else {
                continue;
            };
            if !matches_filters(&item.meta, filters) {
                continue;
            }
            let score = exact_score(self.metric, &item.vector, query);
            scored.push((id.clone(), score));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut hits = Vec::new();
        for (id, score) in scored.into_iter().take(k) {
            if let Some(item) = self.items.get(&id) {
                hits.push(SearchHit {
                    id,
                    score,
                    meta: include_meta.then(|| item.meta.clone()),
                });
            }
        }
        hits
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

fn exact_score(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::Cosine => {
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                dot += x * y;
                norm_a += x * x;
                norm_b += y * y;
            }
            if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot / (norm_a.sqrt() * norm_b.sqrt())
            }
        }
        Metric::Dot => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
    }
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
