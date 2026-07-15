mod builder;
mod graph;
mod io;

use crate::vector::index::{DiskAnnBuildParams, DiskIndexStatus};
use crate::vector::persist::{CollectionLayout, Manifest};
use crate::vector::q8::QuantizedVec;
use crate::vector::Metric;
pub use graph::DiskGraph;

const DISKANN_KIND: &str = "diskann_vamana";

/// A graph file that has been written to disk but not yet wired into a
/// collection's manifest. Produced by [`build_graph_to_file`] (the CPU-heavy,
/// lock-free half of a build) and consumed by [`apply_built_graph`] (the cheap
/// swap half). Splitting the build this way lets the caller run the expensive
/// graph construction with no collection lock held and then flip the manifest
/// over to the new file under a brief lock.
pub struct BuiltGraph {
    pub rel_path: String,
    pub params: DiskAnnBuildParams,
    pub last_built_ms: u64,
}

/// Pure, `Collection`-free half of a DiskANN build: construct the Vamana graph
/// from `nodes` and write it atomically to disk. Touches no `Collection` and no
/// `Manifest`, so it is safe to call with NO collection lock held (e.g. inside
/// `spawn_blocking`). Call [`apply_built_graph`] afterwards, under a brief lock,
/// to point the manifest at the result.
pub fn build_graph_to_file(
    layout: &CollectionLayout,
    nodes: &[(String, QuantizedVec)],
    metric: Metric,
    params: &DiskAnnBuildParams,
    simd_enabled: bool,
) -> std::io::Result<BuiltGraph> {
    let sanitized = params.clone().sanitized();
    let graph_file = builder::build_graph_file(metric, nodes, &sanitized, simd_enabled);
    let rel_path = io::write_graph_file(layout, &graph_file)?;
    Ok(BuiltGraph {
        rel_path,
        params: sanitized,
        last_built_ms: now_ms(),
    })
}

/// Swap half of a DiskANN build: point `manifest.disk_index` at a graph file
/// produced by [`build_graph_to_file`]. Cheap — mutates the manifest only. The
/// caller is responsible for reloading the in-RAM `DiskGraph` and persisting the
/// manifest.
pub fn apply_built_graph(manifest: &mut Manifest, built: &BuiltGraph) -> DiskIndexStatus {
    manifest.disk_index.kind = Some(DISKANN_KIND.to_string());
    manifest.disk_index.version = manifest.disk_index.version.max(1);
    manifest.disk_index.last_built_ms = built.last_built_ms;
    manifest.disk_index.graph_files = vec![built.rel_path.clone()];
    manifest.disk_index.build_params =
        serde_json::to_value(&built.params).unwrap_or(serde_json::Value::Null);
    DiskIndexStatus {
        available: true,
        last_built_ms: built.last_built_ms,
        graph_files: vec![built.rel_path.clone()],
        params: built.params.clone(),
    }
}

pub fn build_disk_index(
    layout: &CollectionLayout,
    manifest: &mut Manifest,
    nodes: &[(String, QuantizedVec)],
    metric: Metric,
    params: &DiskAnnBuildParams,
    simd_enabled: bool,
) -> std::io::Result<DiskIndexStatus> {
    let built = build_graph_to_file(layout, nodes, metric, params, simd_enabled)?;
    Ok(apply_built_graph(manifest, &built))
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
