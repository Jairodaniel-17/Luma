mod diskann;
pub mod index;
mod ivf;
pub mod mmap;
mod persist;
pub mod q8;
mod simd;

pub use index::{DiskAnnIndex, DiskVectorIndex, VectorIndex};
pub use ivf::IndexKind;

use crate::vector::ivf::{assign_all_clusters, train_centroids, IvfConfig, IvfState};
use crate::vector::persist::{
    compact_runs, CentroidsMeta, CollectionLayout, Manifest, Record, RecordOp,
    DEFAULT_COMPACTION_MAX_BYTES_PER_PASS, DEFAULT_COMPACTION_TRIGGER_TOMBSTONE_RATIO,
    DEFAULT_RUN_RETENTION, DEFAULT_RUN_TARGET_BYTES,
};
use crate::vector::q8 as q8ops;
use crate::vector::q8::QuantizedVec;
use anyhow::Context;
use hnsw_rs::prelude::*;
use index::{DiskAnnBuildParams, DiskIndexStatus};
use parking_lot::RwLock;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

#[derive(Clone)]
pub struct VectorStore(Arc<Inner>);

struct Inner {
    data_dir: Option<PathBuf>,
    collections: dashmap::DashMap<String, Arc<RwLock<Collection>>>,
    settings: VectorSettings,
}

const DEFAULT_SEGMENT_MAX: usize = 8_192;
const DEFAULT_PARALLEL_SEGMENT_MIN: usize = 4;
const DEFAULT_DISKANN_SEARCH_LIST_SIZE: usize = 64;
static RAYON_INIT: OnceLock<()> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct VectorSettings {
    pub parallel_segment_search: bool,
    pub parallel_segment_min: usize,
    pub simd_enabled: bool,
    pub hnsw_fallback_enabled: bool,
    pub search_threads: Option<usize>,
    pub index_kind: IndexKind,
    pub ivf: IvfConfig,
    pub run_target_bytes: u64,
    pub run_retention: usize,
    pub compaction_trigger_tombstone_ratio: f32,
    pub compaction_max_bytes_per_pass: u64,
    pub q8_refine_topk: usize,
    pub diskann_search_list_size: usize,
    pub diskann_max_degree: usize,
    pub diskann_build_threads: usize,
    /// PR4: HNSW M parameter (connections per node). Applies to new collections.
    pub hnsw_m: usize,
    /// PR4: HNSW ef_construction parameter. Applies to new collections.
    pub hnsw_ef_construction: usize,
}

impl Default for VectorSettings {
    fn default() -> Self {
        Self {
            parallel_segment_search: true,
            parallel_segment_min: DEFAULT_PARALLEL_SEGMENT_MIN,
            simd_enabled: true,
            hnsw_fallback_enabled: true,
            search_threads: None,
            index_kind: IndexKind::Hnsw,
            ivf: IvfConfig {
                clusters: 1024,
                nprobe: 8,
                training_sample: 200_000,
                max_training_iters: 15,
                min_train_vectors: 1_024,
                retrain_min_deltas: 50_000,
            },
            run_target_bytes: DEFAULT_RUN_TARGET_BYTES,
            run_retention: DEFAULT_RUN_RETENTION,
            compaction_trigger_tombstone_ratio: DEFAULT_COMPACTION_TRIGGER_TOMBSTONE_RATIO,
            compaction_max_bytes_per_pass: DEFAULT_COMPACTION_MAX_BYTES_PER_PASS,
            q8_refine_topk: 512,
            diskann_search_list_size: DEFAULT_DISKANN_SEARCH_LIST_SIZE,
            diskann_max_degree: 64,
            diskann_build_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        }
    }
}

impl VectorSettings {
    pub fn from_config(config: &crate::config::Config) -> Self {
        let parallel_segment_min = config.parallel_probe_min_segments.max(2);
        let index_kind = match config.index_kind.trim().to_ascii_uppercase().as_str() {
            "IVF_FLAT_Q8" => IndexKind::IvfFlatQ8,
            "DISKANN" => IndexKind::DiskAnn,
            _ => IndexKind::Hnsw,
        };
        let clusters = config.ivf_clusters.max(2);
        let nprobe = config.ivf_nprobe.clamp(1, clusters);
        let refine_topk = config.q8_refine_topk.max(1);
        let min_train = config.ivf_min_train_vectors.max(2);
        let retrain_min = config.ivf_retrain_min_deltas.max(1);
        Self {
            parallel_segment_search: config.parallel_probe,
            parallel_segment_min,
            simd_enabled: config.simd_enabled,
            hnsw_fallback_enabled: true,
            search_threads: (config.search_threads > 0).then_some(config.search_threads),
            index_kind,
            ivf: IvfConfig {
                clusters,
                nprobe,
                training_sample: config.ivf_training_sample,
                max_training_iters: 15,
                min_train_vectors: min_train,
                retrain_min_deltas: retrain_min,
            },
            run_target_bytes: config.run_target_bytes,
            run_retention: config.run_retention,
            compaction_trigger_tombstone_ratio: config.compaction_trigger_tombstone_ratio,
            compaction_max_bytes_per_pass: config.compaction_max_bytes_per_pass,
            q8_refine_topk: refine_topk,
            diskann_search_list_size: config.diskann_search_list_size.max(4),
            diskann_max_degree: config.diskann_max_degree.max(4),
            diskann_build_threads: config.diskann_build_threads.max(1),
            hnsw_m: config.hnsw_m.clamp(2, 128),
            hnsw_ef_construction: config.hnsw_ef_construction.clamp(16, 2048),
        }
    }

    fn init_rayon(&self) {
        if let Some(threads) = self.search_threads {
            let _ = RAYON_INIT.get_or_init(|| {
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build_global();
            });
            return;
        }
        let _ = RAYON_INIT.get_or_init(|| {
            let _ = rayon::ThreadPoolBuilder::new().build_global();
        });
    }

    fn should_parallel_segments(&self, segments: usize) -> bool {
        self.parallel_segment_search && segments >= self.parallel_segment_min.max(2)
    }

    fn ivf_enabled(&self) -> bool {
        self.index_kind.is_ivf()
    }
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
    pub mmap_offset: Option<u64>,
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
    #[error("operation not supported")]
    UnsupportedOperation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    #[serde(default)]
    pub options: SearchOptions,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SearchOptions {
    pub filters: Option<serde_json::Value>,
    pub include_meta: bool,
    pub allowed_ids: Option<HashSet<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub meta: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorCollectionInfo {
    pub collection: String,
    pub dim: usize,
    pub metric: Metric,
    pub live_count: usize,
    pub total_records: u64,
    pub upsert_count: u64,
    pub file_len: u64,
    pub applied_offset: u64,
    pub created_at_ms: Option<u64>,
    pub updated_at_ms: Option<u64>,
    pub segments: Option<usize>,
    pub deleted_count: Option<u64>,
}

struct Collection {
    dim: usize,
    metric: Metric,
    layout: Option<CollectionLayout>,
    manifest: Manifest,
    items: HashMap<String, VectorItem>,
    q8_store: HashMap<String, QuantizedVec>,
    mmap_store: Option<mmap::VectorMmap>,
    item_mmap_offsets: HashMap<String, usize>,
    applied_offset: u64,
    segments: Vec<SegmentIndex>,
    item_segments: HashMap<String, usize>,
    item_runs: HashMap<String, String>,
    cluster_members: HashMap<usize, HashSet<String>>,
    segment_max_items: usize,
    keyword_index: HashMap<String, HashMap<String, HashSet<String>>>,
    settings: VectorSettings,
    ivf: Option<IvfState>,
    item_clusters: HashMap<String, usize>,
    disk_graph: Option<diskann::DiskGraph>,
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
    fn new(metric: Metric, capacity: usize, hnsw_m: usize, hnsw_ef_construction: usize) -> Self {
        Self {
            hnsw: make_hnsw(metric, hnsw_m, capacity.max(1024), 16, hnsw_ef_construction),
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
        Self::with_settings(VectorSettings::default())
    }

    pub fn with_settings(settings: VectorSettings) -> Self {
        settings.init_rayon();
        Self(Arc::new(Inner {
            data_dir: None,
            collections: dashmap::DashMap::new(),
            settings,
        }))
    }

    pub fn open(data_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::open_with_settings(data_dir, VectorSettings::default())
    }

    pub fn open_with_settings(
        data_dir: impl AsRef<Path>,
        settings: VectorSettings,
    ) -> anyhow::Result<Self> {
        settings.init_rayon();
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
            let (manifest, items, quantized, item_runs, applied_offset) =
                persist::load_collection(&layout)
                    .with_context(|| format!("load vector collection {name}"))?;
            let mut c = Collection::new(
                Some(layout),
                manifest,
                items,
                quantized,
                item_runs,
                applied_offset,
                settings.clone(),
            )?;
            c.rebuild_index();
            collections.insert(name, c);
        }

        Ok(Self(Arc::new(Inner {
            data_dir: Some(data_dir),
            collections: collections
                .into_iter()
                .map(|(k, v)| (k, Arc::new(RwLock::new(v))))
                .collect(),
            settings,
        })))
    }

    pub fn applied_offset(&self) -> u64 {
        self.0
            .collections
            .iter()
            .map(|c| c.read().applied_offset)
            .max()
            .unwrap_or(0)
    }

    pub fn get_collection(&self, name: &str) -> Option<(usize, Metric)> {
        self.0.collections.get(name).map(|c| {
            let c = c.read();
            (c.dim, c.metric)
        })
    }
    pub fn get_collection_info(&self, name: &str) -> Option<VectorCollectionInfo> {
        self.0.collections.get(name).map(|c| {
            let c = c.read();
            VectorCollectionInfo {
                collection: name.to_string(),
                dim: c.dim,
                metric: c.metric,
                live_count: c.manifest.live_count,
                total_records: c.manifest.total_records,
                upsert_count: c.manifest.upsert_count,
                file_len: c.manifest.file_len,
                applied_offset: c.manifest.applied_offset,
                created_at_ms: None,
                updated_at_ms: None,
                segments: Some(c.segments.len()),
                deleted_count: Some(
                    c.manifest
                        .total_records
                        .saturating_sub(c.manifest.live_count as u64),
                ),
            }
        })
    }

    pub fn create_collection(
        &self,
        name: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<(), VectorError> {
        if self.0.collections.contains_key(name) {
            return Err(VectorError::CollectionExists);
        }
        let layout = self.layout_for(name);
        let (manifest, items, quantized, item_runs, applied_offset) = if let Some(layout) = &layout
        {
            persist::init_collection(layout, dim, metric).map_err(|_| VectorError::Persistence)?;
            persist::load_collection(layout).map_err(|_| VectorError::Persistence)?
        } else {
            (
                Manifest::new(dim, metric),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                0,
            )
        };
        let mut c = Collection::new(
            layout.clone(),
            manifest,
            items,
            quantized,
            item_runs,
            applied_offset,
            self.0.settings.clone(),
        )?;
        c.rebuild_index();
        c.sync_manifest_run_settings()?;
        self.0
            .collections
            .insert(name.to_string(), Arc::new(parking_lot::RwLock::new(c)));
        Ok(())
    }

    pub fn list_collections(&self) -> Vec<VectorCollectionInfo> {
        self.0
            .collections
            .iter()
            .map(|entry| {
                let c = entry.value().read();
                let name = entry.key();
                VectorCollectionInfo {
                    collection: name.clone(),
                    dim: c.dim,
                    metric: c.metric,
                    live_count: c.manifest.live_count,
                    total_records: c.manifest.total_records,
                    upsert_count: c.manifest.upsert_count,
                    file_len: c.manifest.file_len,
                    applied_offset: c.manifest.applied_offset,
                    created_at_ms: None,
                    updated_at_ms: None,
                    segments: Some(c.segments.len()),
                    deleted_count: Some(
                        c.manifest
                            .total_records
                            .saturating_sub(c.manifest.live_count as u64),
                    ),
                }
            })
            .collect()
    }

    pub fn compact_collection(&self, collection: &str) -> Result<bool, VectorError> {
        self.compact_collection_with_options(collection, false)
    }

    pub fn compact_collection_with_options(
        &self,
        collection: &str,
        force: bool,
    ) -> Result<bool, VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
        c.force_compact(force)
    }

    pub fn retrain_ivf(&self, collection: &str, force: bool) -> Result<bool, VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
        c.try_train_ivf(force)
    }

    pub fn build_disk_index(
        &self,
        collection: &str,
        params: DiskAnnBuildParams,
    ) -> Result<(), VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
        let _ = c.build_disk_index(params)?;
        Ok(())
    }

    pub fn drop_disk_index(&self, collection: &str) -> Result<(), VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
        c.drop_disk_index()
    }

    pub fn disk_index_status(&self, collection: &str) -> Result<DiskIndexStatus, VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let c = c_arc.read();
        Ok(c.disk_index_status())
    }

    pub fn update_disk_index_params(
        &self,
        collection: &str,
        params: DiskAnnBuildParams,
    ) -> Result<DiskAnnBuildParams, VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
        c.update_diskann_params(params)
    }

    pub fn get(&self, collection: &str, id: &str) -> Result<Option<VectorItem>, VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let c = c_arc.read();
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

                if let Some(existing_arc) = self.0.collections.get(name) {
                    let mut existing = existing_arc.write();
                    if existing.dim != dim || existing.metric != metric {
                        return Err(VectorError::InvalidManifest);
                    }
                    existing.mark_applied_offset(ev.offset)?;
                    return Ok(());
                }

                let layout = self.layout_for(name);
                let (manifest, items, quantized, item_runs, applied_offset) =
                    if let Some(layout) = &layout {
                        persist::init_collection(layout, dim, metric)
                            .map_err(|_| VectorError::Persistence)?;
                        persist::load_collection(layout).map_err(|_| VectorError::Persistence)?
                    } else {
                        (
                            Manifest::new(dim, metric),
                            HashMap::new(),
                            HashMap::new(),
                            HashMap::new(),
                            0,
                        )
                    };
                let mut c = Collection::new(
                    layout.clone(),
                    manifest,
                    items,
                    quantized,
                    item_runs,
                    applied_offset,
                    self.0.settings.clone(),
                )?;
                c.mark_applied_offset(ev.offset)?;
                c.rebuild_index();
                c.sync_manifest_run_settings()?;
                self.0
                    .collections
                    .insert(name.to_string(), Arc::new(parking_lot::RwLock::new(c)));
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

                let c_arc = self
                    .0
                    .collections
                    .get(collection)
                    .ok_or(VectorError::CollectionNotFound)?;
                let mut c = c_arc.write();
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
                            quantized: None,
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
                            quantized: None,
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
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
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
            quantized: None,
        };
        c.apply_record(record, Some(ApplyMode::InMemoryOnly))?;
        Ok(())
    }

    pub fn upsert(&self, collection: &str, id: &str, item: VectorItem) -> Result<(), VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
        if item.vector.len() != c.dim {
            return Err(VectorError::DimMismatch);
        }
        let record = Record {
            offset: 0,
            op: RecordOp::Upsert,
            id: id.to_string(),
            vector: Some(item.vector),
            meta: Some(item.meta),
            quantized: None,
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
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
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
            quantized: None,
        };
        c.apply_record(record, Some(ApplyMode::InMemoryOnly))?;
        Ok(())
    }

    pub fn delete(&self, collection: &str, id: &str) -> Result<(), VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
        if !c.items.contains_key(id) {
            return Err(VectorError::IdNotFound);
        }
        let record = Record {
            offset: 0,
            op: RecordOp::Delete,
            id: id.to_string(),
            vector: None,
            meta: None,
            quantized: None,
        };
        c.apply_record(record, Some(ApplyMode::InMemoryOnly))?;
        Ok(())
    }

    pub fn search(
        &self,
        collection: &str,
        req: SearchRequest,
    ) -> Result<Vec<SearchHit>, VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let c = c_arc.read();
        c.search(req)
    }

    fn layout_for(&self, collection: &str) -> Option<CollectionLayout> {
        let base = self.0.data_dir.as_ref()?.join("vectors");
        Some(CollectionLayout::new(&base, collection))
    }

    pub fn vacuum_collection(&self, collection: &str) -> Result<(), VectorError> {
        let c_arc = self
            .0
            .collections
            .get(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        let mut c = c_arc.write();
        let layout = c.layout.clone().ok_or(VectorError::Persistence)?;
        let result = persist::rewrite_collection(&layout, &c.manifest, &c.items, &c.q8_store)
            .map_err(|_| VectorError::Persistence)?;
        c.manifest = result.manifest;
        c.item_runs = result.item_runs;
        c.rebuild_index();
        Ok(())
    }

    /// PR6: Rebuild HNSW segments for collections that exceed the tombstone ratio threshold.
    ///
    /// Returns the list of collection names that were compacted.
    /// Acquires a write lock per collection only when compaction is needed.
    pub fn compact_hnsw_segments(&self, threshold: f32) -> Vec<String> {
        let mut compacted = Vec::new();
        let collection_names: Vec<String> =
            self.0.collections.iter().map(|r| r.key().clone()).collect();

        for name in collection_names {
            let Some(c_arc) = self.0.collections.get(&name) else {
                continue;
            };

            // Check tombstone ratio without write lock first
            let needs_compaction = {
                let c = c_arc.read();
                if c.segments.is_empty() {
                    false
                } else {
                    let total_tombstones: usize = c
                        .segments
                        .iter()
                        .map(|s| s.deleted.iter().filter(|&&d| d).count())
                        .sum();
                    let total_capacity: usize =
                        c.segments.iter().map(|s| s.id_by_data_id.len()).sum();
                    total_capacity > 0
                        && (total_tombstones as f32 / total_capacity as f32) > threshold
                }
            };

            if needs_compaction {
                let mut c = c_arc.write();
                let still_needs_compaction = if c.segments.is_empty() {
                    false
                } else {
                    let total_tombstones: usize = c
                        .segments
                        .iter()
                        .map(|s| s.deleted.iter().filter(|&&d| d).count())
                        .sum();
                    let total_capacity: usize =
                        c.segments.iter().map(|s| s.id_by_data_id.len()).sum();
                    total_capacity > 0
                        && (total_tombstones as f32 / total_capacity as f32) > threshold
                };
                if still_needs_compaction {
                    c.rebuild_segments();
                    compacted.push(name);
                }
            }
        }

        compacted
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
        quantized: HashMap<String, QuantizedVec>,
        item_runs: HashMap<String, String>,
        applied_offset: u64,
        settings: VectorSettings,
    ) -> Result<Self, VectorError> {
        let dim = manifest.dim;
        let metric = manifest.metric;
        let mut c = Self {
            dim,
            metric,
            layout,
            manifest: manifest.clone(),
            items,
            q8_store: quantized,
            mmap_store: None,
            item_mmap_offsets: HashMap::new(),
            item_runs,
            applied_offset,
            segments: Vec::new(),
            item_segments: HashMap::new(),
            cluster_members: HashMap::new(),
            segment_max_items: DEFAULT_SEGMENT_MAX,
            keyword_index: HashMap::new(),
            settings,
            ivf: None,
            item_clusters: HashMap::new(),
            disk_graph: None,
        };
        if let Some(layout) = &c.layout {
            let initial_capacity = (manifest.total_records as usize * 2).max(1024);
            match mmap::VectorMmap::create_or_open(&layout.mmap_path, dim, initial_capacity) {
                Ok(mut store) => {
                    if store.header().count == 0 && !c.items.is_empty() {
                        // Migration: Append existing items to mmap in a deterministic order
                        let mut ids: Vec<String> = c.items.keys().cloned().collect();
                        ids.sort();
                        for id in ids {
                            if let Some(item) = c.items.get_mut(&id) {
                                if let Ok(idx) = store.append(&item.vector) {
                                    item.mmap_offset = Some(idx as u64);
                                    c.item_mmap_offsets.insert(id, idx);
                                }
                            }
                        }
                        let _ = store.flush();
                    } else if store.header().count > 0 {
                        // If mmap has data but we don't have offsets (e.g. restart),
                        // we need a way to recover. For now, we'll re-migrate to be safe if
                        // offsets are missing, but this is a placeholder for a real mapping storage.
                        // For simplicity in this step, we just populate from whatever we have.
                    }
                    c.mmap_store = Some(store);
                }
                Err(e) => tracing::warn!("Failed to initialize mmap store: {}", e),
            }
        }

        c.load_ivf_from_disk()
            .map_err(|_| VectorError::Persistence)?;
        c.load_disk_graph().map_err(|_| VectorError::Persistence)?;
        c.rebuild_index();
        c.sync_manifest_run_settings()?;
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
        self.ensure_quantized_store();
        self.rebuild_segments();
    }

    fn update_cluster_membership(&mut self, id: &str, cluster: usize) {
        if let Some(prev) = self.item_clusters.get(id).copied() {
            if prev != cluster {
                if let Some(members) = self.cluster_members.get_mut(&prev) {
                    members.remove(id);
                    if members.is_empty() {
                        self.cluster_members.remove(&prev);
                    }
                }
            }
        }
        self.item_clusters.insert(id.to_string(), cluster);
        self.cluster_members
            .entry(cluster)
            .or_default()
            .insert(id.to_string());
    }

    fn remove_cluster_membership(&mut self, id: &str) {
        if let Some(prev) = self.item_clusters.remove(id) {
            if let Some(members) = self.cluster_members.get_mut(&prev) {
                members.remove(id);
                if members.is_empty() {
                    self.cluster_members.remove(&prev);
                }
            }
        }
    }

    fn sync_manifest_run_settings(&mut self) -> Result<(), VectorError> {
        if self.manifest.apply_run_settings(&self.settings) {
            self.persist_manifest()
                .map_err(|_| VectorError::Persistence)?;
        }
        Ok(())
    }

    fn rebuild_segments(&mut self) {
        self.item_segments.clear();
        self.segments.clear();
        if !self.settings.hnsw_fallback_enabled {
            return;
        }
        if self.items.is_empty() {
            self.segments.push(SegmentIndex::new(
                self.metric,
                self.segment_max_items,
                self.settings.hnsw_m,
                self.settings.hnsw_ef_construction,
            ));
            return;
        }
        let mut current = SegmentIndex::new(
            self.metric,
            self.segment_max_items,
            self.settings.hnsw_m,
            self.settings.hnsw_ef_construction,
        );
        let mut ids: Vec<&String> = self.items.keys().collect();
        ids.sort();
        for id in ids {
            let item = self
                .items
                .get(id)
                .expect("item id collected from keys must exist");
            if current.live >= current.capacity {
                self.segments.push(current);
                current = SegmentIndex::new(
                    self.metric,
                    self.segment_max_items,
                    self.settings.hnsw_m,
                    self.settings.hnsw_ef_construction,
                );
            }
            current.insert(id.clone(), item.vector.clone());
            let idx = self.segments.len();
            self.item_segments.insert(id.clone(), idx);
        }
        self.segments.push(current);
        if self.segments.is_empty() {
            self.segments.push(SegmentIndex::new(
                self.metric,
                self.segment_max_items,
                self.settings.hnsw_m,
                self.settings.hnsw_ef_construction,
            ));
        }
        self.refresh_item_clusters();
    }

    fn force_compact(&mut self, force: bool) -> Result<bool, VectorError> {
        if self.layout.is_none() {
            return Ok(false);
        }
        self.sync_manifest_run_settings()?;
        self.maybe_compact_runs(force)
    }

    fn refresh_item_clusters(&mut self) {
        self.item_clusters.clear();
        self.cluster_members.clear();
        if let Some(ivf) = &self.ivf {
            let assigned = assign_all_clusters(ivf, &self.items, self.settings.simd_enabled);
            for (id, cluster) in assigned.iter() {
                self.cluster_members
                    .entry(*cluster)
                    .or_default()
                    .insert(id.clone());
            }
            self.item_clusters = assigned;
        }
    }

    fn ensure_quantized_store(&mut self) {
        self.q8_store.retain(|id, _| self.items.contains_key(id));
        let missing: Vec<(String, Vec<f32>)> = self
            .items
            .iter()
            .filter(|(id, _)| !self.q8_store.contains_key(*id))
            .map(|(id, item)| (id.clone(), item.vector.clone()))
            .collect();
        for (id, vector) in missing {
            let q = q8ops::quantize_per_vector(&vector);
            self.q8_store.insert(id, q);
        }
    }

    fn load_ivf_from_disk(&mut self) -> std::io::Result<()> {
        let Some(layout) = &self.layout else {
            return Ok(());
        };
        if let Some((meta, centroids)) = persist::load_centroids(layout)? {
            if meta.dim == self.dim && meta.metric == self.metric && !centroids.is_empty() {
                let state = IvfState::new(centroids, self.metric, meta.trained_at_ms);
                self.ivf = Some(state);
                self.manifest.centroid_count = meta.clusters;
                self.manifest.centroids_trained_at_ms = meta.trained_at_ms;
                self.refresh_item_clusters();
            }
        }
        Ok(())
    }

    fn load_disk_graph(&mut self) -> std::io::Result<()> {
        let Some(layout) = &self.layout else {
            self.disk_graph = None;
            return Ok(());
        };
        self.disk_graph = diskann::load_graph(layout, &self.manifest)?;
        Ok(())
    }

    fn invalidate_disk_index_if_needed(&mut self) {
        if self.manifest.disk_index.graph_files.is_empty() && self.disk_graph.is_none() {
            return;
        }
        if let Some(layout) = &self.layout {
            let _ = diskann::drop_disk_index(layout, &mut self.manifest);
        } else {
            self.manifest.disk_index.graph_files.clear();
            self.manifest.disk_index.kind = None;
            self.manifest.disk_index.last_built_ms = 0;
            self.manifest.disk_index.version = 0;
            self.manifest.disk_index.build_params = serde_json::Value::Null;
        }
        self.disk_graph = None;
        let _ = self.persist_manifest();
    }

    fn effective_diskann_params(&self) -> DiskAnnBuildParams {
        self.manifest
            .diskann_build_params()
            .unwrap_or(DiskAnnBuildParams {
                max_degree: self.settings.diskann_max_degree,
                build_threads: self.settings.diskann_build_threads,
                search_list_size: self.settings.diskann_search_list_size,
            })
            .sanitized()
    }

    fn diskann_search_list_size(&self) -> usize {
        self.effective_diskann_params().search_list_size.max(1)
    }

    fn update_ivf_state(&mut self, state: IvfState) {
        self.ivf = Some(state);
        self.refresh_item_clusters();
    }

    fn sample_training_vectors(&self) -> Vec<Vec<f32>> {
        if self.items.is_empty() {
            return Vec::new();
        }
        let mut entries: Vec<(&String, &VectorItem)> = self.items.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut vectors: Vec<Vec<f32>> = entries
            .into_iter()
            .map(|(_, item)| item.vector.clone())
            .collect();
        let limit = self.settings.ivf.training_sample.min(vectors.len());
        if limit == 0 {
            return Vec::new();
        }
        if vectors.len() > limit {
            let seed = self.manifest.upsert_count ^ vectors.len() as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            vectors.shuffle(&mut rng);
            vectors.truncate(limit);
        }
        vectors
    }

    fn maybe_train_ivf(&mut self) -> Result<(), VectorError> {
        let _ = self.try_train_ivf(false)?;
        Ok(())
    }

    fn try_train_ivf(&mut self, force: bool) -> Result<bool, VectorError> {
        if !self.settings.ivf_enabled() {
            return Ok(false);
        }
        let live = self.items.len();
        let cluster_target = self.manifest.ivf_clusters.max(2);
        let min_train = self
            .manifest
            .ivf_min_train_vectors
            .max(cluster_target)
            .min(self.settings.ivf.training_sample.max(cluster_target));
        if !force && live < min_train {
            return Ok(false);
        }
        if !force && self.ivf.is_some() {
            let delta = self
                .manifest
                .upsert_count
                .saturating_sub(self.manifest.ivf_last_trained_upsert);
            if delta < self.manifest.ivf_retrain_min_vectors as u64 {
                return Ok(false);
            }
        }
        let samples = self.sample_training_vectors();
        if samples.len() < cluster_target {
            return Ok(false);
        }
        if let Some(centroids) = train_centroids(&samples, &self.settings.ivf, self.metric) {
            if centroids.is_empty() {
                return Ok(false);
            }
            let now = now_ms();
            let state = IvfState::new(centroids, self.metric, now);
            let centroid_count = state.centroids().len();
            self.persist_ivf_state(&state)?;
            self.update_ivf_state(state);
            self.manifest.centroid_count = centroid_count;
            self.manifest.centroids_trained_at_ms = now;
            self.manifest.ivf_last_trained_upsert = self.manifest.upsert_count;
            self.persist_manifest()
                .map_err(|_| VectorError::Persistence)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn persist_ivf_state(&self, state: &IvfState) -> Result<(), VectorError> {
        let Some(layout) = &self.layout else {
            return Ok(());
        };
        let meta = CentroidsMeta {
            version: 1,
            dim: self.dim,
            metric: self.metric,
            clusters: state.centroids().len(),
            trained_at_ms: state.trained_at_ms(),
        };
        persist::store_centroids(layout, &meta, state.centroids())
            .map_err(|_| VectorError::Persistence)
    }

    fn ivf_probe_set(&self, query: &[f32]) -> Option<HashSet<usize>> {
        if !self.settings.ivf_enabled() {
            return None;
        }
        let ivf = self.ivf.as_ref()?;
        let nprobe = self.settings.ivf.nprobe.max(1);
        let probes = ivf.select_probes(query, self.settings.simd_enabled, nprobe);
        if probes.is_empty() {
            None
        } else {
            Some(probes.into_iter().collect())
        }
    }

    fn assign_cluster_for(&mut self, id: &str, vector: &[f32]) {
        if !self.settings.ivf_enabled() {
            return;
        }
        if let Some(ivf) = &self.ivf {
            if let Some(cluster) = ivf.assign_vector(vector, self.settings.simd_enabled) {
                self.update_cluster_membership(id, cluster);
            }
        }
    }

    fn ensure_active_segment(&mut self) -> usize {
        if self.segments.is_empty() {
            self.segments.push(SegmentIndex::new(
                self.metric,
                self.segment_max_items,
                self.settings.hnsw_m,
                self.settings.hnsw_ef_construction,
            ));
        }
        let last_idx = self.segments.len() - 1;
        if self.segments[last_idx].live >= self.segments[last_idx].capacity {
            self.segments.push(SegmentIndex::new(
                self.metric,
                self.segment_max_items,
                self.settings.hnsw_m,
                self.settings.hnsw_ef_construction,
            ));
            return self.segments.len() - 1;
        }
        last_idx
    }

    fn insert_into_segments(&mut self, id: &str, vector: Vec<f32>) {
        if !self.settings.hnsw_fallback_enabled {
            return;
        }
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
        if !self.settings.hnsw_fallback_enabled {
            return;
        }
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
            let value = v.as_str()?;
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
        if matches!(record.op, RecordOp::Upsert | RecordOp::Delete) {
            self.invalidate_disk_index_if_needed();
        }
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
        let quantized_vec = if record.op == RecordOp::Upsert {
            let Some(vec) = normalized_vec.as_ref() else {
                return Err(VectorError::InvalidManifest);
            };
            let q = record
                .quantized
                .clone()
                .unwrap_or_else(|| q8ops::quantize_per_vector(vec));
            record.quantized = Some(q.clone());
            Some(q)
        } else {
            record.quantized = None;
            None
        };

        let mut mmap_idx: Option<u64> = None;
        if let Some(layout) = &self.layout {
            if mode.is_none() {
                let _ = persist::append_record(layout, &mut self.manifest, &record)
                    .map_err(|_| VectorError::Persistence)?;

                // Also append to new mmap store if it's an upsert
                if record.op == RecordOp::Upsert {
                    if let Some(vec) = &normalized_vec {
                        if let Some(mmap) = self.mmap_store.as_mut() {
                            match mmap.append(vec) {
                                Ok(idx) => {
                                    mmap_idx = Some(idx as u64);
                                    self.item_mmap_offsets.insert(record.id.clone(), idx);
                                }
                                Err(e) => tracing::warn!("Failed to append to mmap store: {}", e),
                            }
                        }
                    }

                    if let Some(run) = self.manifest.runs.last() {
                        self.item_runs.insert(record.id.clone(), run.file.clone());
                    }
                }
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
                self.q8_store.remove(&record.id);
                self.remove_cluster_membership(&record.id);
                self.item_runs.remove(&record.id);
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
                    mmap_offset: mmap_idx,
                };
                let previous = self.items.insert(record.id.clone(), new_item.clone());
                if let Some(prev) = previous.as_ref() {
                    self.remove_meta_from_index(&record.id, Some(&prev.meta));
                }
                self.add_meta_to_index(&record.id, &new_item.meta);
                self.insert_into_segments(&record.id, new_item.vector.clone());
                if let Some(qvec) = quantized_vec {
                    self.q8_store.insert(record.id.clone(), qvec);
                }
                self.assign_cluster_for(&record.id, &new_item.vector);
                if previous.is_none() {
                    self.manifest.live_count += 1;
                }
            }
        }

        self.manifest.live_count = self.items.len();

        if self.layout.is_some() && mode.is_none() {
            let compacted = self.maybe_compact_runs(false)?;
            if !compacted {
                self.persist_manifest()
                    .map_err(|_| VectorError::Persistence)?;
            }
        }

        self.maybe_train_ivf()?;

        Ok(())
    }

    fn build_disk_index(
        &mut self,
        params: DiskAnnBuildParams,
    ) -> Result<DiskIndexStatus, VectorError> {
        let params = params.sanitized();
        let layout = self.layout.clone().ok_or(VectorError::Persistence)?;
        self.ensure_quantized_store();
        let mut nodes: Vec<(String, QuantizedVec)> = self
            .items
            .iter()
            .map(|(id, item)| {
                let q = self
                    .q8_store
                    .entry(id.clone())
                    .or_insert_with(|| q8ops::quantize_per_vector(&item.vector))
                    .clone();
                (id.clone(), q)
            })
            .collect();
        nodes.sort_by(|a, b| a.0.cmp(&b.0));
        let status = diskann::build_disk_index(
            &layout,
            &mut self.manifest,
            &nodes,
            self.metric,
            &params,
            self.settings.simd_enabled,
        )
        .map_err(|_| VectorError::Persistence)?;
        self.load_disk_graph()
            .map_err(|_| VectorError::Persistence)?;
        self.persist_manifest()
            .map_err(|_| VectorError::Persistence)?;
        Ok(status)
    }

    fn update_diskann_params(
        &mut self,
        params: DiskAnnBuildParams,
    ) -> Result<DiskAnnBuildParams, VectorError> {
        let sanitized = params.sanitized();
        self.manifest.disk_index.build_params =
            serde_json::to_value(&sanitized).unwrap_or(serde_json::Value::Null);
        self.persist_manifest()
            .map_err(|_| VectorError::Persistence)?;
        Ok(sanitized)
    }

    fn drop_disk_index(&mut self) -> Result<(), VectorError> {
        if let Some(layout) = &self.layout {
            diskann::drop_disk_index(layout, &mut self.manifest)
                .map_err(|_| VectorError::Persistence)?;
        } else {
            self.manifest.disk_index.graph_files.clear();
            self.manifest.disk_index.kind = None;
            self.manifest.disk_index.version = 0;
            self.manifest.disk_index.last_built_ms = 0;
            self.manifest.disk_index.build_params = serde_json::Value::Null;
        }
        self.disk_graph = None;
        self.persist_manifest()
            .map_err(|_| VectorError::Persistence)?;
        Ok(())
    }

    fn disk_index_status(&self) -> DiskIndexStatus {
        diskann::status_from_manifest(&self.manifest, self.effective_diskann_params())
    }

    fn persist_manifest(&self) -> std::io::Result<()> {
        if let Some(layout) = &self.layout {
            persist::store_manifest(layout, &self.manifest)?;
        }
        Ok(())
    }

    fn maybe_compact_runs(&mut self, force: bool) -> Result<bool, VectorError> {
        let Some(layout) = &self.layout else {
            return Ok(false);
        };
        if self.manifest.runs.is_empty() || (!force && !self.manifest.should_compact()) {
            return Ok(false);
        }
        let result = compact_runs(
            layout,
            &self.manifest,
            &self.items,
            &self.q8_store,
            &self.item_runs,
            self.settings.compaction_max_bytes_per_pass,
        )
        .map_err(|_| VectorError::Persistence)?;
        if let Some(res) = result {
            self.manifest = res.manifest;
            for (id, run) in res.item_runs {
                self.item_runs.insert(id, run);
            }
            self.applied_offset = self.manifest.applied_offset;
            return Ok(true);
        }
        Ok(false)
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
        let include_meta = req.options.include_meta;
        let k = req.k.max(1);
        let query = normalize_if_needed(self.metric, req.vector);
        if self.items.is_empty() {
            return Ok(Vec::new());
        }
        if self.items.len() < 100
            && req.options.filters.is_none()
            && req.options.allowed_ids.is_none()
        {
            let mut scored = Vec::new();
            for (id, item) in self.items.iter() {
                let score = exact_score(
                    self.metric,
                    &item.vector,
                    &query,
                    self.settings.simd_enabled,
                );
                scored.push(SearchHit {
                    id: id.clone(),
                    score,
                    meta: include_meta.then(|| item.meta.clone()),
                });
            }
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
            scored.truncate(k);
            return Ok(scored);
        }
        let mut filter_candidates = req
            .options
            .filters
            .as_ref()
            .and_then(|f| self.keyword_candidates(f));

        if let Some(ref allowed) = req.options.allowed_ids {
            let mut resolved_allowed = HashSet::new();
            if let Some(by_value) = self.keyword_index.get("parent_id") {
                for doc_id in allowed {
                    if let Some(chunk_ids) = by_value.get(doc_id) {
                        resolved_allowed.extend(chunk_ids.iter().cloned());
                    }
                }
            } else {
                // Fallback: If no parent_id index exists, we assume id prefixes
                for doc_id in allowed {
                    // This is inefficient but acts as fallback
                    for item_id in self.items.keys() {
                        if item_id.starts_with(doc_id) {
                            resolved_allowed.insert(item_id.clone());
                        }
                    }
                }
            }

            if let Some(ref mut current) = filter_candidates {
                let current_set: &mut HashSet<String> = current;
                current_set.retain(|id| resolved_allowed.contains(id));
            } else {
                filter_candidates = Some(resolved_allowed);
            }
        }

        if let Some(ref set) = filter_candidates {
            let set_ref: &HashSet<String> = set;
            if set_ref.is_empty() {
                return Ok(Vec::new());
            }
        }
        if self.settings.index_kind.is_diskann() {
            if let Some(hits) = self.search_diskann(
                query.as_slice(),
                include_meta,
                req.options.filters.as_ref(),
                filter_candidates.as_ref(),
                k,
            )? {
                return Ok(hits);
            }
        }
        let ivf_probes = self.ivf_probe_set(query.as_slice());
        if let Some(ref set) = filter_candidates {
            let set_ref: &HashSet<String> = set;
            if set_ref.is_empty() {
                return Ok(Vec::new());
            }
            if set_ref.len() <= 512 {
                return Ok(self.search_subset_bruteforce(
                    query.as_slice(),
                    include_meta,
                    set_ref,
                    req.options.filters.as_ref(),
                    k,
                    ivf_probes.as_ref(),
                ));
            }
        }
        if let Some(ref probes) = ivf_probes {
            return Ok(self.search_ivf_flat(
                query.as_slice(),
                include_meta,
                req.options.filters.as_ref(),
                filter_candidates.as_ref(),
                k,
                probes,
            ));
        }

        let candidate_k = (k * 10).min(self.items.len()).max(k);
        let mut combined: Vec<(String, f32)> =
            if self.settings.should_parallel_segments(self.segments.len()) {
                self.segments
                    .par_iter()
                    .map(|segment| segment.search_candidates(query.as_slice(), candidate_k))
                    .reduce(Vec::new, |mut acc, mut part| {
                        acc.append(&mut part);
                        acc
                    })
            } else {
                self.segments
                    .iter()
                    .flat_map(|segment| segment.search_candidates(query.as_slice(), candidate_k))
                    .collect()
            };
        combined.sort_by(compare_scores_desc);

        let mut hits = Vec::new();
        let mut seen = HashSet::new();
        for (id, score) in combined {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Some(ref probes) = ivf_probes {
                let Some(cluster) = self.item_clusters.get(&id) else {
                    continue;
                };
                if !probes.contains(cluster) {
                    continue;
                }
            }
            if let Some(ref set) = filter_candidates {
                let set_ref: &HashSet<String> = set;
                if !set_ref.contains(&id) {
                    continue;
                }
            }
            let Some(item) = self.items.get(&id) else {
                continue;
            };
            if !matches_filters(&item.meta, req.options.filters.as_ref()) {
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

    #[inline(always)]
    fn get_vector_slice<'a>(&'a self, id: &str, item: &'a VectorItem) -> &'a [f32] {
        if let Some(mmap) = &self.mmap_store {
            if let Some(&idx) = self.item_mmap_offsets.get(id) {
                if let Some(slice) = mmap.get_vector(idx) {
                    return slice;
                }
            }
        }
        &item.vector
    }

    fn search_subset_bruteforce(
        &self,
        query: &[f32],
        include_meta: bool,
        candidates: &HashSet<String>,
        filters: Option<&serde_json::Value>,
        k: usize,
        cluster_filter: Option<&HashSet<usize>>,
    ) -> Vec<SearchHit> {
        let mut scored = Vec::new();
        for id in candidates {
            let Some(item) = self.items.get(id) else {
                continue;
            };
            if let Some(probes) = cluster_filter {
                let Some(cluster) = self.item_clusters.get(id) else {
                    continue;
                };
                if !probes.contains(cluster) {
                    continue;
                }
            }
            if !matches_filters(&item.meta, filters) {
                continue;
            }
            let score = exact_score(
                self.metric,
                self.get_vector_slice(id, item),
                query,
                self.settings.simd_enabled,
            );
            scored.push((id.clone(), score));
        }
        scored.sort_by(compare_scores_desc);
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

    fn search_ivf_flat(
        &self,
        query: &[f32],
        include_meta: bool,
        filters: Option<&serde_json::Value>,
        filter_candidates: Option<&HashSet<String>>,
        k: usize,
        probes: &HashSet<usize>,
    ) -> Vec<SearchHit> {
        let q_query = q8ops::quantize_per_vector(query);
        let mut scored = Vec::new();
        for cluster in probes {
            let Some(members) = self.cluster_members.get(cluster) else {
                continue;
            };
            for id in members {
                if let Some(set) = filter_candidates {
                    if !set.contains(id) {
                        continue;
                    }
                }
                let Some(item) = self.items.get(id) else {
                    continue;
                };
                if !matches_filters(&item.meta, filters) {
                    continue;
                }
                let Some(qvec) = self.q8_store.get(id) else {
                    continue;
                };
                let approx = q8ops::dot(qvec, &q_query, self.settings.simd_enabled);
                scored.push((id.clone(), approx));
            }
        }
        if scored.is_empty() {
            return Vec::new();
        }
        scored.sort_by(compare_scores_desc);
        let refine_topk = self.manifest.q8_refine_topk.max(k).min(scored.len());
        let mut refined = Vec::new();
        for (id, _) in scored.into_iter().take(refine_topk) {
            if let Some(item) = self.items.get(&id) {
                let exact = exact_score(
                    self.metric,
                    self.get_vector_slice(&id, item),
                    query,
                    self.settings.simd_enabled,
                );
                refined.push((id, exact));
            }
        }
        refined.sort_by(compare_scores_desc);
        let mut hits = Vec::new();
        for (id, score) in refined.into_iter().take(k) {
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

    fn search_diskann(
        &self,
        query: &[f32],
        include_meta: bool,
        filters: Option<&serde_json::Value>,
        filter_candidates: Option<&HashSet<String>>,
        k: usize,
    ) -> Result<Option<Vec<SearchHit>>, VectorError> {
        let graph = match &self.disk_graph {
            Some(graph) => graph,
            None => return Ok(None),
        };
        if query.len() != self.dim {
            return Ok(Some(Vec::new()));
        }
        let search_list = self.diskann_search_list_size();
        let approx = graph
            .search(
                query,
                self.settings.simd_enabled,
                search_list,
                (k * 5).max(k),
                filter_candidates,
            )
            .map_err(|_| VectorError::Persistence)?;
        if approx.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let mut refined = Vec::new();
        for (idx, _) in approx {
            let Some(id) = graph.id_for(idx).map_err(|_| VectorError::Persistence)? else {
                continue;
            };
            if let Some(set) = filter_candidates {
                if !set.contains(&id) {
                    continue;
                }
            }
            let Some(item) = self.items.get(&id) else {
                continue;
            };
            if !matches_filters(&item.meta, filters) {
                continue;
            }
            let exact = exact_score(
                self.metric,
                self.get_vector_slice(&id, item),
                query,
                self.settings.simd_enabled,
            );
            refined.push((id.to_string(), exact));
        }
        refined.sort_by(compare_scores_desc);
        let mut hits = Vec::new();
        for (id, score) in refined.into_iter().take(k) {
            if let Some(item) = self.items.get(&id) {
                hits.push(SearchHit {
                    id: id.clone(),
                    score,
                    meta: include_meta.then(|| item.meta.clone()),
                });
            }
        }
        Ok(Some(hits))
    }
}

#[cfg(test)]
mod tests {
    use super::{Metric, SearchOptions, SearchRequest, VectorItem, VectorSettings, VectorStore};

    #[test]
    fn compact_hnsw_segments_preserves_live_ids_and_clears_tombstones() {
        let store = VectorStore::with_settings(VectorSettings::default());
        store.create_collection("docs", 3, Metric::Cosine).unwrap();

        for idx in 0..24usize {
            store
                .upsert(
                    "docs",
                    &format!("doc-{idx:02}"),
                    VectorItem {
                        vector: vec![1.0, idx as f32, 0.0],
                        meta: serde_json::json!({ "idx": idx }),
                        mmap_offset: None,
                    },
                )
                .unwrap();
        }
        for idx in 0..12usize {
            store.delete("docs", &format!("doc-{idx:02}")).unwrap();
        }

        let compacted = store.compact_hnsw_segments(0.0);
        assert_eq!(compacted, vec!["docs".to_string()]);

        let collection = store.0.collections.get("docs").unwrap();
        let collection = collection.read();
        assert_eq!(collection.items.len(), 12);
        assert_eq!(collection.item_segments.len(), 12);
        assert!(collection
            .segments
            .iter()
            .all(|segment| segment.deleted.iter().all(|deleted| !deleted)));

        drop(collection);

        let hits = store
            .search(
                "docs",
                SearchRequest {
                    vector: vec![1.0, 20.0, 0.0],
                    k: 5,
                    options: SearchOptions {
                        filters: None,
                        include_meta: false,
                        allowed_ids: None,
                    },
                },
            )
            .unwrap();
        assert!(hits.iter().all(|hit| !hit.id.starts_with("doc-0")));
        assert!(hits.iter().any(|hit| hit.id == "doc-20"));
    }

    #[test]
    fn rebuild_segments_uses_stable_id_order() {
        let store = VectorStore::with_settings(VectorSettings::default());
        store.create_collection("docs", 2, Metric::Cosine).unwrap();

        for id in ["doc-c", "doc-a", "doc-b"] {
            store
                .upsert(
                    "docs",
                    id,
                    VectorItem {
                        vector: vec![1.0, 0.0],
                        meta: serde_json::json!({ "id": id }),
                        mmap_offset: None,
                    },
                )
                .unwrap();
        }

        let collection = store.0.collections.get("docs").unwrap();
        let mut collection = collection.write();
        collection.rebuild_segments();

        let ordered_ids = &collection.segments[0].id_by_data_id;
        assert_eq!(
            ordered_ids,
            &vec![
                "doc-a".to_string(),
                "doc-b".to_string(),
                "doc-c".to_string()
            ]
        );
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

fn compare_scores_desc(a: &(String, f32), b: &(String, f32)) -> Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

fn exact_score(metric: Metric, a: &[f32], b: &[f32], simd_enabled: bool) -> f32 {
    match metric {
        Metric::Cosine => {
            let (dot, norm_a, norm_b) = simd::dot_and_norms(a, b, simd_enabled);
            if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot / (norm_a.sqrt() * norm_b.sqrt())
            }
        }
        Metric::Dot => simd::dot(a, b, simd_enabled),
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

fn now_ms() -> u64 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}

impl index::VectorIndex for VectorStore {
    fn list_collections(&self) -> Vec<VectorCollectionInfo> {
        VectorStore::list_collections(self)
    }

    fn get_collection(&self, name: &str) -> Option<(usize, Metric)> {
        VectorStore::get_collection(self, name)
    }

    fn create_collection(&self, name: &str, dim: usize, metric: Metric) -> Result<(), VectorError> {
        VectorStore::create_collection(self, name, dim, metric)
    }

    fn upsert(&self, collection: &str, id: &str, item: VectorItem) -> Result<(), VectorError> {
        VectorStore::upsert(self, collection, id, item)
    }

    fn delete(&self, collection: &str, id: &str) -> Result<(), VectorError> {
        VectorStore::delete(self, collection, id)
    }

    fn search(&self, collection: &str, req: SearchRequest) -> Result<Vec<SearchHit>, VectorError> {
        VectorStore::search(self, collection, req)
    }

    fn compact(&self, collection: &str, force: bool) -> Result<bool, VectorError> {
        VectorStore::compact_collection_with_options(self, collection, force)
    }

    fn retrain_ivf(&self, collection: &str, force: bool) -> Result<bool, VectorError> {
        VectorStore::retrain_ivf(self, collection, force)
    }
}

impl index::DiskVectorIndex for VectorStore {
    fn warm_collection(&self, collection: &str) -> Result<(), VectorError> {
        self.get_collection(collection)
            .ok_or(VectorError::CollectionNotFound)?;
        Ok(())
    }

    fn sync_collection(&self, collection: &str) -> Result<(), VectorError> {
        let _ = self.compact_collection_with_options(collection, false)?;
        Ok(())
    }
}

impl index::DiskAnnIndex for VectorStore {
    fn build_disk_index(
        &self,
        collection: &str,
        params: index::DiskAnnBuildParams,
    ) -> Result<(), VectorError> {
        VectorStore::build_disk_index(self, collection, params)
    }

    fn drop_disk_index(&self, collection: &str) -> Result<(), VectorError> {
        VectorStore::drop_disk_index(self, collection)
    }

    fn disk_index_status(&self, collection: &str) -> Result<index::DiskIndexStatus, VectorError> {
        VectorStore::disk_index_status(self, collection)
    }

    fn update_disk_index_params(
        &self,
        collection: &str,
        params: index::DiskAnnBuildParams,
    ) -> Result<index::DiskAnnBuildParams, VectorError> {
        VectorStore::update_disk_index_params(self, collection, params)
    }
}
