mod builder;
mod graph;
mod io;

use crate::vector::index::{DiskAnnBuildParams, DiskIndexStatus};
use crate::vector::persist::{CollectionLayout, Manifest};
use crate::vector::q8::QuantizedVec;
use crate::vector::Metric;
pub use graph::DiskGraph;

const DISKANN_KIND: &str = "diskann_vamana";

pub fn build_disk_index(
    layout: &CollectionLayout,
    manifest: &mut Manifest,
    nodes: &[(String, QuantizedVec)],
    metric: Metric,
    params: &DiskAnnBuildParams,
    simd_enabled: bool,
) -> std::io::Result<DiskIndexStatus> {
    let sanitized = params.clone().sanitized();
    let graph_file = builder::build_graph_file(metric, nodes, &sanitized, simd_enabled);
    let rel_path = io::write_graph_file(layout, &graph_file)?;
    manifest.disk_index.kind = Some(DISKANN_KIND.to_string());
    manifest.disk_index.version = manifest.disk_index.version.max(1);
    manifest.disk_index.last_built_ms = now_ms();
    manifest.disk_index.graph_files = vec![rel_path.clone()];
    manifest.disk_index.build_params =
        serde_json::to_value(&sanitized).unwrap_or(serde_json::Value::Null);
    Ok(DiskIndexStatus {
        available: true,
        last_built_ms: manifest.disk_index.last_built_ms,
        graph_files: vec![rel_path],
        params: sanitized,
    })
}

pub fn drop_disk_index(layout: &CollectionLayout, manifest: &mut Manifest) -> std::io::Result<()> {
    io::remove_graph_files(layout, &manifest.disk_index.graph_files)?;
    manifest.disk_index.graph_files.clear();
    manifest.disk_index.kind = None;
    manifest.disk_index.last_built_ms = 0;
    manifest.disk_index.version = 0;
    manifest.disk_index.build_params = serde_json::Value::Null;
    Ok(())
}

pub fn load_graph(
    layout: &CollectionLayout,
    manifest: &Manifest,
) -> std::io::Result<Option<DiskGraph>> {
    let Some(rel_path) = manifest.disk_index.graph_files.first() else {
        return Ok(None);
    };
    let path = io::graph_path(layout, rel_path);
    if !path.exists() {
        return Ok(None);
    }
    DiskGraph::load_from_path(&path).map(Some)
}

pub fn status_from_manifest(manifest: &Manifest, fallback: DiskAnnBuildParams) -> DiskIndexStatus {
    DiskIndexStatus {
        available: !manifest.disk_index.graph_files.is_empty(),
        last_built_ms: manifest.disk_index.last_built_ms,
        graph_files: manifest.disk_index.graph_files.clone(),
        params: manifest
            .diskann_build_params()
            .unwrap_or(fallback)
            .sanitized(),
    }
}

fn now_ms() -> u64 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}
