use crate::vector::{Metric, VectorError, VectorItem};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct CollectionLayout {
    pub dir: PathBuf,
    pub manifest_path: PathBuf,
    pub bin_path: PathBuf,
}

impl CollectionLayout {
    pub fn new(base: &Path, collection: &str) -> Self {
        let dir = base.join(collection);
        Self {
            manifest_path: dir.join("manifest.json"),
            bin_path: dir.join("vectors.bin"),
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
}

#[derive(Serialize, Deserialize)]
struct DiskRecord {
    offset: u64,
    op: RecordOp,
    id: String,
    vector: Option<Vec<f32>>,
    meta: Option<Vec<u8>>,
}

struct CollectionRecords {
    items: HashMap<String, VectorItem>,
    applied_offset: u64,
    total_records: u64,
    file_len: u64,
    upserts: u64,
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
        }
    }
}

pub fn init_collection(
    layout: &CollectionLayout,
    dim: usize,
    metric: Metric,
) -> std::io::Result<()> {
    std::fs::create_dir_all(&layout.dir)?;
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
) -> anyhow::Result<(Manifest, HashMap<String, VectorItem>, u64)> {
    let manifest = read_manifest(layout).map_err(|_| VectorError::Persistence)?;
    let data = read_records(layout, &manifest)?;
    let mut manifest2 = manifest.clone();
    manifest2.applied_offset = data.applied_offset;
    manifest2.total_records = data.total_records;
    manifest2.live_count = data.items.len();
    manifest2.file_len = data.file_len;
    manifest2.upsert_count = data.upserts;
    let _ = store_manifest(layout, &manifest2);
    Ok((manifest2, data.items, data.applied_offset))
}

pub fn append_record(layout: &CollectionLayout, record: &Record) -> std::io::Result<u64> {
    let disk_record = disk_record_from(record)?;
    let payload = bincode::serialize(&disk_record)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "bincode serialize"))?;
    let len = payload.len() as u32;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&layout.bin_path)?;
    file.write_all(&len.to_le_bytes())?;
    file.write_all(&payload)?;
    file.flush()?;
    file.sync_data()?;
    Ok((4 + payload.len()) as u64)
}

pub fn store_manifest(layout: &CollectionLayout, manifest: &Manifest) -> std::io::Result<()> {
    write_manifest(layout, manifest)
}

pub fn rewrite_collection(
    layout: &CollectionLayout,
    manifest: &Manifest,
    items: &HashMap<String, VectorItem>,
) -> std::io::Result<Manifest> {
    std::fs::create_dir_all(&layout.dir)?;
    let tmp = layout.dir.join("vectors.bin.compacting");
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp)?;
    let mut total_bytes = 0u64;
    let mut total_records = 0u64;
    let mut upserts = 0u64;

    let mut entries: Vec<_> = items.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    for (id, item) in entries {
        let record = Record {
            offset: 0,
            op: RecordOp::Upsert,
            id: id.clone(),
            vector: Some(item.vector.clone()),
            meta: Some(item.meta.clone()),
        };
        let disk = disk_record_from(&record)?;
        let payload = bincode::serialize(&disk).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "bincode serialize")
        })?;
        let len = payload.len() as u32;
        file.write_all(&len.to_le_bytes())?;
        file.write_all(&payload)?;
        total_bytes = total_bytes.saturating_add(4 + payload.len() as u64);
        total_records = total_records.saturating_add(1);
        if record.op == RecordOp::Upsert {
            upserts = upserts.saturating_add(1);
        }
    }
    file.flush()?;
    file.sync_data()?;
    std::fs::rename(&tmp, &layout.bin_path)?;

    let mut new_manifest = manifest.clone();
    new_manifest.total_records = total_records;
    new_manifest.live_count = items.len();
    new_manifest.upsert_count = upserts;
    new_manifest.file_len = total_bytes;
    store_manifest(layout, &new_manifest)?;
    Ok(new_manifest)
}

fn read_records(
    layout: &CollectionLayout,
    manifest: &Manifest,
) -> anyhow::Result<CollectionRecords> {
    if !layout.bin_path.exists() {
        return Ok(CollectionRecords {
            items: HashMap::new(),
            applied_offset: manifest.applied_offset,
            total_records: 0,
            file_len: 0,
            upserts: 0,
        });
    }

    let file_len = fs::metadata(&layout.bin_path)?.len();
    let file = File::open(&layout.bin_path)?;
    let mut reader = BufReader::new(file);
    let mut items: HashMap<String, VectorItem> = HashMap::new();
    let mut applied = manifest.applied_offset;
    let mut total = 0u64;
    let mut upserts = 0u64;

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
        total += 1;
        applied = applied.max(record.offset);
        match record.op {
            RecordOp::Delete => {
                items.remove(&record.id);
            }
            RecordOp::Upsert => {
                upserts += 1;
                let v = record.vector.unwrap_or_default();
                if v.len() != manifest.dim {
                    continue;
                }
                let meta = record
                    .meta
                    .as_deref()
                    .and_then(|bytes| serde_json::from_slice(bytes).ok())
                    .unwrap_or(serde_json::Value::Null);
                items.insert(record.id, VectorItem { vector: v, meta });
            }
        }
    }

    Ok(CollectionRecords {
        items,
        applied_offset: applied,
        total_records: total,
        file_len,
        upserts,
    })
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
    })
}
