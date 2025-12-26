use super::graph::{encode_graph_file, GraphFile};
use crate::vector::persist::CollectionLayout;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn ensure_diskann_dir(layout: &CollectionLayout) -> io::Result<PathBuf> {
    let dir = layout.dir.join("diskann");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn write_graph_file(layout: &CollectionLayout, graph: &GraphFile) -> io::Result<String> {
    let dir = ensure_diskann_dir(layout)?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let filename = format!("graph-{ts}.bin");
    let path = dir.join(&filename);
    let bytes = encode_graph_file(graph)?;
    atomic_write_bytes(&path, &bytes)?;
    Ok(format!("diskann/{filename}"))
}

pub fn remove_graph_files(layout: &CollectionLayout, graph_files: &[String]) -> io::Result<()> {
    for rel in graph_files {
        let path = layout.dir.join(rel);
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
    }
    Ok(())
}

pub fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_data()?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

pub fn graph_path(layout: &CollectionLayout, relative: &str) -> PathBuf {
    layout.dir.join(relative)
}
