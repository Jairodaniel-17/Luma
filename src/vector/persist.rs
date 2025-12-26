use super::index::DiskAnnBuildParams;
use super::q8::{quantize_per_vector, QuantizedVec};
use super::VectorSettings;
use crate::vector::{Metric, VectorError, VectorItem};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub(super) const DEFAULT_RUN_TARGET_BYTES: u64 = 134_217_728;
pub(super) const DEFAULT_RUN_RETENTION: usize = 8;
pub(super) const DEFAULT_COMPACTION_TRIGGER_TOMBSTONE_RATIO: f32 = 0.2;
pub(super) const DEFAULT_COMPACTION_MAX_BYTES_PER_PASS: u64 = 1_073_741_824;
const DEFAULT_IVF_CLUSTERS: usize = 1_024;
const DEFAULT_IVF_NPROBE: usize = 8;
const DEFAULT_Q8_REFINE_TOPK: usize = 512;
const DEFAULT_IVF_MIN_TRAIN_VECTORS: usize = 1_024;
const DEFAULT_IVF_RETRAIN_MIN_VECTORS: usize = 50_000;

fn default_run_target_bytes() -> u64 {
    DEFAULT_RUN_TARGET_BYTES
}

fn default_run_retention() -> usize {
    DEFAULT_RUN_RETENTION
}

fn default_compaction_trigger_tombstone_ratio() -> f32 {
    DEFAULT_COMPACTION_TRIGGER_TOMBSTONE_RATIO
}

fn default_compaction_max_bytes_per_pass() -> u64 {
    DEFAULT_COMPACTION_MAX_BYTES_PER_PASS
}

fn default_ivf_clusters() -> usize {
    DEFAULT_IVF_CLUSTERS
}

fn default_ivf_nprobe() -> usize {
    DEFAULT_IVF_NPROBE
}

fn default_q8_refine_topk() -> usize {
    DEFAULT_Q8_REFINE_TOPK
}

fn default_ivf_min_train_vectors() -> usize {
    DEFAULT_IVF_MIN_TRAIN_VECTORS
}

fn default_ivf_retrain_min_vectors() -> usize {
    DEFAULT_IVF_RETRAIN_MIN_VECTORS
}

#[derive(Clone)]
pub struct CollectionLayout {
    pub dir: PathBuf,
    pub manifest_path: PathBuf,
    pub bin_path: PathBuf,
    pub centroids_meta_path: PathBuf,
    pub centroids_bin_path: PathBuf,
    pub runs_dir: PathBuf,
}

impl CollectionLayout {
    pub fn new(base: &Path, collection: &str) -> Self {
        let dir = base.join(collection);
        Self {
            manifest_path: dir.join("manifest.json"),
            bin_path: dir.join("vectors.bin"),
            centroids_meta_path: dir.join("centroids.json"),
            centroids_bin_path: dir.join("centroids.bin"),
            runs_dir: dir.join("runs"),
            dir,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub dim: usize,
    pub metric: Metric,
    pub applied_offset: u64,
    #[serde(default)]
    pub total_records: u64,
    #[serde(default)]
    pub live_count: usize,
    #[serde(default)]
    pub upsert_count: u64,
    #[serde(default)]
    pub file_len: u64,
    #[serde(default)]
    pub runs: Vec<RunInfo>,
    #[serde(default)]
    pub next_run_id: u64,
    #[serde(default = "default_run_target_bytes")]
    pub run_target_bytes: u64,
    #[serde(default = "default_run_retention")]
    pub run_retention: usize,
    #[serde(default = "default_compaction_trigger_tombstone_ratio")]
    pub compaction_trigger_tombstone_ratio: f32,
    #[serde(default = "default_compaction_max_bytes_per_pass")]
    pub compaction_max_bytes_per_pass: u64,
    #[serde(default)]
    pub centroid_count: usize,
    #[serde(default)]
    pub centroids_trained_at_ms: u64,
    #[serde(default = "default_ivf_clusters")]
    pub ivf_clusters: usize,
    #[serde(default = "default_ivf_nprobe")]
    pub ivf_nprobe: usize,
    #[serde(default = "default_q8_refine_topk")]
    pub q8_refine_topk: usize,
    #[serde(default = "default_ivf_min_train_vectors")]
    pub ivf_min_train_vectors: usize,
    #[serde(default = "default_ivf_retrain_min_vectors")]
    pub ivf_retrain_min_vectors: usize,
    #[serde(default)]
    pub ivf_last_trained_upsert: u64,
    #[serde(default)]
    pub disk_index: DiskIndexManifest,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DiskIndexManifest {
    pub kind: Option<String>,
    pub version: u32,
    pub last_built_ms: u64,
    #[serde(default)]
    pub graph_files: Vec<String>,
    #[serde(default)]
    pub build_params: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct RewriteResult {
    pub manifest: Manifest,
    pub item_runs: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunInfo {
    pub file: String,
    pub bytes: u64,
    pub records: u64,
    pub tombstones: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecordOp {
    Upsert,
    Delete,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub offset: u64,
    pub op: RecordOp,
    pub id: String,
    pub vector: Option<Vec<f32>>,
    pub meta: Option<serde_json::Value>,
    pub quantized: Option<QuantizedVec>,
}

#[derive(Serialize, Deserialize)]
struct DiskRecord {
    offset: u64,
    op: RecordOp,
    id: String,
    vector: Option<Vec<f32>>,
    meta: Option<Vec<u8>>,
    #[serde(default)]
    quantized: Option<QuantizedVec>,
}

struct CollectionRecords {
    items: HashMap<String, VectorItem>,
    quantized: HashMap<String, QuantizedVec>,
    applied_offset: u64,
    total_records: u64,
    file_len: u64,
    upserts: u64,
    item_runs: HashMap<String, String>,
}

impl CollectionRecords {
    fn new(applied_offset: u64) -> Self {
        Self {
            items: HashMap::new(),
            quantized: HashMap::new(),
            applied_offset,
            total_records: 0,
            file_len: 0,
            upserts: 0,
            item_runs: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CentroidsMeta {
    pub version: u32,
    pub dim: usize,
    pub metric: Metric,
    pub clusters: usize,
    pub trained_at_ms: u64,
}

impl Manifest {
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            version: 1,
            dim,
            metric,
            applied_offset: 0,
            total_records: 0,
            live_count: 0,
            upsert_count: 0,
            file_len: 0,
            runs: Vec::new(),
            next_run_id: 1,
            run_target_bytes: default_run_target_bytes(),
            run_retention: default_run_retention(),
            compaction_trigger_tombstone_ratio: default_compaction_trigger_tombstone_ratio(),
            compaction_max_bytes_per_pass: default_compaction_max_bytes_per_pass(),
            centroid_count: 0,
            centroids_trained_at_ms: 0,
            ivf_clusters: default_ivf_clusters(),
            ivf_nprobe: default_ivf_nprobe(),
            q8_refine_topk: default_q8_refine_topk(),
            ivf_min_train_vectors: default_ivf_min_train_vectors(),
            ivf_retrain_min_vectors: default_ivf_retrain_min_vectors(),
            ivf_last_trained_upsert: 0,
            disk_index: DiskIndexManifest::default(),
        }
    }

    pub fn diskann_build_params(&self) -> Option<DiskAnnBuildParams> {
        serde_json::from_value(self.disk_index.build_params.clone()).ok()
    }

    pub fn apply_run_settings(&mut self, settings: &VectorSettings) -> bool {
        let mut changed = false;
        if self.run_target_bytes != settings.run_target_bytes {
            self.run_target_bytes = settings.run_target_bytes.max(1);
            changed = true;
        }
        if self.run_retention != settings.run_retention {
            self.run_retention = settings.run_retention;
            changed = true;
        }
        if (self.compaction_trigger_tombstone_ratio - settings.compaction_trigger_tombstone_ratio)
            .abs()
            > f32::EPSILON
        {
            self.compaction_trigger_tombstone_ratio =
                settings.compaction_trigger_tombstone_ratio.max(0.0);
            changed = true;
        }
        if self.compaction_max_bytes_per_pass != settings.compaction_max_bytes_per_pass {
            self.compaction_max_bytes_per_pass = settings.compaction_max_bytes_per_pass;
            changed = true;
        }
        if self.ivf_clusters != settings.ivf.clusters {
            self.ivf_clusters = settings.ivf.clusters;
            changed = true;
        }
        if self.ivf_nprobe != settings.ivf.nprobe {
            self.ivf_nprobe = settings.ivf.nprobe;
            changed = true;
        }
        if self.q8_refine_topk != settings.q8_refine_topk {
            self.q8_refine_topk = settings.q8_refine_topk;
            changed = true;
        }
        if self.ivf_min_train_vectors != settings.ivf.min_train_vectors {
            self.ivf_min_train_vectors = settings.ivf.min_train_vectors;
            changed = true;
        }
        if self.ivf_retrain_min_vectors != settings.ivf.retrain_min_deltas {
            self.ivf_retrain_min_vectors = settings.ivf.retrain_min_deltas;
            changed = true;
        }
        changed
    }

    pub fn should_compact(&self) -> bool {
        if self.run_retention > 0 && self.runs.len() > self.run_retention {
            return true;
        }
        if self.compaction_trigger_tombstone_ratio <= 0.0 {
            return false;
        }
        let ratio = self.tombstone_ratio();
        ratio >= self.compaction_trigger_tombstone_ratio && ratio > 0.0
    }

    pub fn tombstone_ratio(&self) -> f32 {
        let mut total_records = 0u64;
        let mut total_tombstones = 0u64;
        for run in &self.runs {
            total_records = total_records.saturating_add(run.records);
            total_tombstones = total_tombstones.saturating_add(run.tombstones);
        }
        if total_records == 0 {
            return 0.0;
        }
        total_tombstones as f32 / total_records as f32
    }
}

pub fn init_collection(
    layout: &CollectionLayout,
    dim: usize,
    metric: Metric,
) -> std::io::Result<()> {
    std::fs::create_dir_all(&layout.dir)?;
    std::fs::create_dir_all(&layout.runs_dir)?;
    if !layout.manifest_path.exists() {
        let manifest = Manifest::new(dim, metric);
        store_manifest(layout, &manifest)?;
    }
    if !layout.bin_path.exists() {
        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&layout.bin_path)?;
    }
    Ok(())
}

pub fn load_collection(
    layout: &CollectionLayout,
) -> anyhow::Result<(
    Manifest,
    HashMap<String, VectorItem>,
    HashMap<String, QuantizedVec>,
    HashMap<String, String>,
    u64,
)> {
    let manifest = read_manifest(layout).map_err(|_| VectorError::Persistence)?;
    let data = read_records(layout, &manifest)?;
    let mut manifest2 = manifest.clone();
    manifest2.applied_offset = data.applied_offset;
    manifest2.total_records = data.total_records;
    manifest2.live_count = data.items.len();
    manifest2.file_len = data.file_len;
    manifest2.upsert_count = data.upserts;
    let _ = store_manifest(layout, &manifest2);
    Ok((
        manifest2,
        data.items,
        data.quantized,
        data.item_runs,
        data.applied_offset,
    ))
}

pub fn append_record(
    layout: &CollectionLayout,
    manifest: &mut Manifest,
    record: &Record,
) -> std::io::Result<u64> {
    std::fs::create_dir_all(&layout.runs_dir)?;
    ensure_active_run(layout, manifest)?;
    let run = manifest
        .runs
        .last_mut()
        .expect("active run must exist after ensure");
    let path = layout.runs_dir.join(&run.file);
    let disk_record = disk_record_from(record)?;
    let payload = bincode::serialize(&disk_record)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bincode serialize"))?;
    let mut hasher = Hasher::new();
    hasher.update(&payload);
    let header = RunHeader::new(&record.op, payload.len(), hasher.finalize());
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    file.write_all(&header.encode())?;
    file.write_all(&payload)?;
    file.flush()?;
    file.sync_data()?;
    let appended = (RUN_HEADER_BYTES + payload.len()) as u64;
    run.bytes = run.bytes.saturating_add(appended);
    run.records = run.records.saturating_add(1);
    if record.op == RecordOp::Delete {
        run.tombstones = run.tombstones.saturating_add(1);
    }
    manifest.file_len = manifest.file_len.saturating_add(appended);
    Ok(appended)
}

pub fn store_manifest(layout: &CollectionLayout, manifest: &Manifest) -> std::io::Result<()> {
    write_manifest(layout, manifest)
}

pub fn rewrite_collection(
    layout: &CollectionLayout,
    manifest: &Manifest,
    items: &HashMap<String, VectorItem>,
    quantized: &HashMap<String, QuantizedVec>,
) -> std::io::Result<RewriteResult> {
    std::fs::create_dir_all(&layout.dir)?;
    std::fs::create_dir_all(&layout.runs_dir)?;
    let mut new_manifest = manifest.clone();
    let old_run_paths: Vec<PathBuf> = manifest
        .runs
        .iter()
        .map(|run| layout.runs_dir.join(&run.file))
        .collect();
    new_manifest.runs.clear();
    new_manifest.file_len = 0;
    let original_target = new_manifest.run_target_bytes;
    new_manifest.run_target_bytes = u64::MAX.saturating_sub(1);
    let mut entries: Vec<_> = items.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut new_item_runs = HashMap::new();
    for (id, item) in entries {
        let record = Record {
            offset: 0,
            op: RecordOp::Upsert,
            id: id.clone(),
            vector: Some(item.vector.clone()),
            meta: Some(item.meta.clone()),
            quantized: quantized.get(id).cloned(),
        };
        let _ = append_record(layout, &mut new_manifest, &record)?;
        if let Some(run) = new_manifest.runs.last() {
            new_item_runs.insert(id.clone(), run.file.clone());
        }
    }
    new_manifest.run_target_bytes = original_target;
    new_manifest.total_records = new_manifest.runs.iter().map(|r| r.records).sum();
    new_manifest.upsert_count = new_manifest.total_records;
    new_manifest.live_count = items.len();
    let _ = std::fs::remove_file(&layout.bin_path);
    store_manifest(layout, &new_manifest)?;
    for path in old_run_paths {
        let _ = std::fs::remove_file(path);
    }
    Ok(RewriteResult {
        manifest: new_manifest,
        item_runs: new_item_runs,
    })
}

pub fn compact_runs(
    layout: &CollectionLayout,
    manifest: &Manifest,
    items: &HashMap<String, VectorItem>,
    quantized: &HashMap<String, QuantizedVec>,
    item_runs: &HashMap<String, String>,
    max_bytes_per_pass: u64,
) -> std::io::Result<Option<RewriteResult>> {
    let selected = select_runs_for_compaction(manifest, max_bytes_per_pass);
    if selected.is_empty() {
        return Ok(None);
    }
    let selected_set: HashSet<String> = selected.into_iter().collect();
    let mut entries: Vec<_> = items
        .iter()
        .filter(|(id, _)| {
            item_runs
                .get(*id)
                .map(|run| selected_set.contains(run))
                .unwrap_or(false)
        })
        .collect();
    let mut new_manifest = manifest.clone();
    let mut removed_paths = Vec::new();
    new_manifest.runs.retain(|run| {
        if selected_set.contains(&run.file) {
            removed_paths.push(layout.runs_dir.join(&run.file));
            false
        } else {
            true
        }
    });
    if entries.is_empty() && removed_paths.is_empty() {
        return Ok(None);
    }
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut updated_item_runs = HashMap::new();
    for (id, item) in entries {
        let record = Record {
            offset: 0,
            op: RecordOp::Upsert,
            id: id.clone(),
            vector: Some(item.vector.clone()),
            meta: Some(item.meta.clone()),
            quantized: quantized.get(id).cloned(),
        };
        let _ = append_record(layout, &mut new_manifest, &record)?;
        if let Some(run) = new_manifest.runs.last() {
            updated_item_runs.insert(id.clone(), run.file.clone());
        }
    }
    new_manifest.file_len = new_manifest.runs.iter().map(|r| r.bytes).sum();
    new_manifest.total_records = new_manifest.runs.iter().map(|r| r.records).sum();
    new_manifest.upsert_count = new_manifest.total_records;
    new_manifest.live_count = items.len();
    store_manifest(layout, &new_manifest)?;
    for path in removed_paths {
        let _ = std::fs::remove_file(path);
    }
    Ok(Some(RewriteResult {
        manifest: new_manifest,
        item_runs: updated_item_runs,
    }))
}

pub fn store_centroids(
    layout: &CollectionLayout,
    meta: &CentroidsMeta,
    centroids: &[Vec<f32>],
) -> std::io::Result<()> {
    std::fs::create_dir_all(&layout.dir)?;
    let tmp_bin = layout.dir.join("centroids.bin.tmp");
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_bin)?,
    );
    bincode::serialize_into(&mut writer, centroids)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "serialize centroids"))?;
    writer.flush()?;
    writer.get_ref().sync_data()?;
    std::fs::rename(&tmp_bin, &layout.centroids_bin_path)?;

    let tmp_meta = layout.dir.join("centroids.json.tmp");
    let mut meta_writer = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_meta)?;
    serde_json::to_writer_pretty(&mut meta_writer, meta)?;
    meta_writer.flush()?;
    meta_writer.sync_data()?;
    std::fs::rename(&tmp_meta, &layout.centroids_meta_path)?;
    Ok(())
}

pub fn load_centroids(
    layout: &CollectionLayout,
) -> std::io::Result<Option<(CentroidsMeta, Vec<Vec<f32>>)>> {
    if !layout.centroids_bin_path.exists() || !layout.centroids_meta_path.exists() {
        return Ok(None);
    }
    let meta_bytes = std::fs::read(&layout.centroids_meta_path)?;
    let meta: CentroidsMeta = serde_json::from_slice(&meta_bytes).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "centroids meta deserialize",
        )
    })?;
    let file = File::open(&layout.centroids_bin_path)?;
    let mut reader = BufReader::new(file);
    let centroids: Vec<Vec<f32>> = bincode::deserialize_from(&mut reader).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "centroids deserialize")
    })?;
    Ok(Some((meta, centroids)))
}

fn read_records(
    layout: &CollectionLayout,
    manifest: &Manifest,
) -> anyhow::Result<CollectionRecords> {
    let mut state = CollectionRecords::new(manifest.applied_offset);
    if layout.bin_path.exists() {
        read_legacy_file(layout, manifest.dim, &mut state)?;
    }
    if !manifest.runs.is_empty() {
        read_run_files(layout, manifest.dim, &manifest.runs, &mut state)?;
    }
    Ok(state)
}

fn read_legacy_file(
    layout: &CollectionLayout,
    dim: usize,
    state: &mut CollectionRecords,
) -> anyhow::Result<()> {
    let file_len = fs::metadata(&layout.bin_path)?.len();
    let file = File::open(&layout.bin_path)?;
    let mut reader = BufReader::new(file);
    loop {
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; len];
        if let Err(err) = reader.read_exact(&mut payload) {
            if err.kind() == io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(err.into());
        }
        let record: DiskRecord = match bincode::deserialize(&payload) {
            Ok(r) => r,
            Err(_) => break,
        };
        apply_disk_record(record, dim, state, Some("legacy"));
    }
    state.file_len = state.file_len.saturating_add(file_len);
    Ok(())
}

fn read_run_files(
    layout: &CollectionLayout,
    dim: usize,
    runs: &[RunInfo],
    state: &mut CollectionRecords,
) -> anyhow::Result<()> {
    for run in runs {
        let path = layout.runs_dir.join(&run.file);
        if !path.exists() {
            continue;
        }
        let mut file = BufReader::new(File::open(&path)?);
        loop {
            let mut header_buf = [0u8; RUN_HEADER_BYTES];
            match file.read_exact(&mut header_buf) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err.into()),
            }
            let Some(header) = RunHeader::decode(&header_buf) else {
                break;
            };
            let header_op = header.op();
            let mut payload = vec![0u8; header.len as usize];
            if let Err(err) = file.read_exact(&mut payload) {
                if err.kind() == io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(err.into());
            }
            let mut hasher = Hasher::new();
            hasher.update(&payload);
            if hasher.finalize() != header.crc32 {
                break;
            }
            let record: DiskRecord = match bincode::deserialize(&payload) {
                Ok(rec) => rec,
                Err(_) => break,
            };
            if record.op != header_op {
                continue;
            }
            apply_disk_record(record, dim, state, Some(&run.file));
        }
        state.file_len = state.file_len.saturating_add(fs::metadata(&path)?.len());
    }
    Ok(())
}

fn apply_disk_record(
    record: DiskRecord,
    dim: usize,
    state: &mut CollectionRecords,
    run_file: Option<&str>,
) {
    state.total_records = state.total_records.saturating_add(1);
    state.applied_offset = state.applied_offset.max(record.offset);
    match record.op {
        RecordOp::Delete => {
            state.items.remove(&record.id);
            state.quantized.remove(&record.id);
            state.item_runs.remove(&record.id);
        }
        RecordOp::Upsert => {
            state.upserts = state.upserts.saturating_add(1);
            let v = record.vector.unwrap_or_default();
            if v.len() != dim {
                return;
            }
            let meta = record
                .meta
                .as_deref()
                .and_then(|bytes| serde_json::from_slice(bytes).ok())
                .unwrap_or(serde_json::Value::Null);
            let q = record
                .quantized
                .clone()
                .unwrap_or_else(|| quantize_per_vector(&v));
            state
                .items
                .insert(record.id.clone(), VectorItem { vector: v, meta });
            state.quantized.insert(record.id.clone(), q);
            if let Some(run) = run_file {
                state.item_runs.insert(record.id, run.to_string());
            }
        }
    }
}

fn ensure_active_run(layout: &CollectionLayout, manifest: &mut Manifest) -> std::io::Result<()> {
    let mut rotate = false;
    if manifest.runs.is_empty() {
        rotate = true;
    } else if let Some(run) = manifest.runs.last() {
        let target = manifest.run_target_bytes.max(1);
        if run.bytes >= target {
            rotate = true;
        }
    }
    if rotate {
        create_new_run(layout, manifest)?;
    }
    Ok(())
}

fn create_new_run(layout: &CollectionLayout, manifest: &mut Manifest) -> std::io::Result<()> {
    std::fs::create_dir_all(&layout.runs_dir)?;
    let run_id = manifest.next_run_id;
    manifest.next_run_id = manifest.next_run_id.saturating_add(1);
    let file = format!("run-{run_id:06}.log");
    let path = layout.runs_dir.join(&file);
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)?;
    manifest.runs.push(RunInfo {
        file,
        ..RunInfo::default()
    });
    Ok(())
}

fn write_manifest(layout: &CollectionLayout, manifest: &Manifest) -> std::io::Result<()> {
    let tmp = layout.dir.join("manifest.json.tmp");
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp)?;
    serde_json::to_writer_pretty(&mut f, manifest)?;
    f.flush()?;
    f.sync_data()?;
    std::fs::rename(tmp, &layout.manifest_path)?;
    Ok(())
}

fn select_runs_for_compaction(manifest: &Manifest, max_bytes: u64) -> Vec<String> {
    if manifest.runs.is_empty() {
        return Vec::new();
    }
    let budget = max_bytes.max(1);
    let mut selected = Vec::new();
    let mut total = 0u64;
    for run in &manifest.runs {
        selected.push(run.file.clone());
        total = total.saturating_add(run.bytes.max(1));
        if total >= budget {
            break;
        }
    }
    if selected.is_empty() {
        if let Some(last) = manifest.runs.first() {
            selected.push(last.file.clone());
        }
    }
    selected
}

fn read_manifest(layout: &CollectionLayout) -> std::io::Result<Manifest> {
    let bytes = std::fs::read(&layout.manifest_path)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    Ok(manifest)
}

fn disk_record_from(record: &Record) -> std::io::Result<DiskRecord> {
    let meta_bytes =
        match &record.meta {
            Some(meta) => Some(serde_json::to_vec(meta).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "meta serialize")
            })?),
            None => None,
        };
    Ok(DiskRecord {
        offset: record.offset,
        op: record.op.clone(),
        id: record.id.clone(),
        vector: record.vector.clone(),
        meta: meta_bytes,
        quantized: record.quantized.clone(),
    })
}
const RUN_MAGIC: u32 = 0x524B5631;
const RUN_VERSION: u16 = 1;
const RUN_HEADER_BYTES: usize = 16;

#[derive(Clone, Copy)]
struct RunHeader {
    magic: u32,
    version: u16,
    flags: u16,
    len: u32,
    crc32: u32,
}

impl RunHeader {
    fn new(op: &RecordOp, len: usize, crc32: u32) -> Self {
        let flags = match op {
            RecordOp::Upsert => 0,
            RecordOp::Delete => 1,
        };
        Self {
            magic: RUN_MAGIC,
            version: RUN_VERSION,
            flags,
            len: len as u32,
            crc32,
        }
    }

    fn encode(&self) -> [u8; RUN_HEADER_BYTES] {
        let mut buf = [0u8; RUN_HEADER_BYTES];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..12].copy_from_slice(&self.len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.crc32.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < RUN_HEADER_BYTES {
            return None;
        }
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != RUN_MAGIC {
            return None;
        }
        let version = u16::from_le_bytes([buf[4], buf[5]]);
        if version != RUN_VERSION {
            return None;
        }
        let flags = u16::from_le_bytes([buf[6], buf[7]]);
        let len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let crc32 = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        Some(Self {
            magic,
            version,
            flags,
            len,
            crc32,
        })
    }

    fn op(&self) -> RecordOp {
        if self.flags & 1 == 1 {
            RecordOp::Delete
        } else {
            RecordOp::Upsert
        }
    }
}
