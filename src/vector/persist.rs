use crate::vector::{Metric, VectorError, VectorItem};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Write};
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
    pub total_records: u64,
    pub live_count: usize,
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

pub fn init_collection(layout: &CollectionLayout, dim: usize, metric: Metric) -> std::io::Result<()> {
    std::fs::create_dir_all(&layout.dir)?;
    if !layout.manifest_path.exists() {
        let manifest = Manifest {
            version: 1,
            dim,
            metric,
            applied_offset: 0,
            total_records: 0,
            live_count: 0,
        };
        write_manifest(layout, &manifest)?;
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
    let (items, applied, total_records) = read_records(layout, &manifest)?;
    let mut manifest2 = manifest.clone();
    manifest2.applied_offset = applied;
    manifest2.total_records = total_records;
    manifest2.live_count = items.len();
    let _ = write_manifest(layout, &manifest2);
    Ok((manifest2, items, applied))
}

pub fn append_record(layout: &CollectionLayout, record: &Record) -> std::io::Result<()> {
    let payload = bincode::serialize(record)
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
    Ok(())
}

pub fn update_applied_offset(layout: &CollectionLayout, offset: u64) -> std::io::Result<()> {
    let mut manifest = read_manifest(layout)?;
    if offset > manifest.applied_offset {
        manifest.applied_offset = offset;
    }
    write_manifest(layout, &manifest)
}

fn read_records(
    layout: &CollectionLayout,
    manifest: &Manifest,
) -> anyhow::Result<(HashMap<String, VectorItem>, u64, u64)> {
    if !layout.bin_path.exists() {
        return Ok((HashMap::new(), manifest.applied_offset, 0));
    }

    let mut f = OpenOptions::new().read(true).open(&layout.bin_path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let mut pos = 0usize;
    let mut items: HashMap<String, VectorItem> = HashMap::new();
    let mut applied = manifest.applied_offset;
    let mut total = 0u64;

    while pos + 4 <= buf.len() {
        let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + len > buf.len() {
            break;
        }
        let payload = &buf[pos..pos + len];
        pos += len;
        let record: Record = match bincode::deserialize(payload) {
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
                let v = record.vector.unwrap_or_default();
                if v.len() != manifest.dim {
                    continue;
                }
                let meta = record.meta.unwrap_or(serde_json::Value::Null);
                items.insert(record.id, VectorItem { vector: v, meta });
            }
        }
    }

    Ok((items, applied, total))
}

fn write_manifest(layout: &CollectionLayout, manifest: &Manifest) -> std::io::Result<()> {
    let tmp = layout.dir.join("manifest.json.tmp");
    let mut f = OpenOptions::new().create(true).write(true).truncate(true).open(&tmp)?;
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
