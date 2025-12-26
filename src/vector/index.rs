use super::{Metric, SearchHit, SearchRequest, VectorCollectionInfo, VectorError, VectorItem};
use serde::{Deserialize, Serialize};

/// Parameters used while (re)building a DiskANN/Vamana index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiskAnnBuildParams {
    pub max_degree: usize,
    pub build_threads: usize,
    pub search_list_size: usize,
}

impl Default for DiskAnnBuildParams {
    fn default() -> Self {
        Self {
            max_degree: 64,
            build_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            search_list_size: 128,
        }
    }
}

impl DiskAnnBuildParams {
    pub fn sanitized(mut self) -> Self {
        self.max_degree = self.max_degree.clamp(4, 1024);
        self.build_threads = self.build_threads.max(1);
        self.search_list_size = self.search_list_size.clamp(8, 65535);
        self
    }
}

/// Status information returned by DiskANN/Vamana indexes.
#[derive(Clone, Debug, Default, Serialize)]
pub struct DiskIndexStatus {
    pub available: bool,
    pub last_built_ms: u64,
    pub graph_files: Vec<String>,
    pub params: DiskAnnBuildParams,
}

/// `VectorIndex` describes the capabilities that any in-memory vector index must expose
/// so the engine can remain agnostic of the concrete implementation (IVF/Q8, HNSW, etc.).
/// Keeping this trait small and focused lets us plug future implementations (e.g. IVF_HNSW)
/// without rewriting the engine surface.
pub trait VectorIndex: Send + Sync {
    fn list_collections(&self) -> Vec<VectorCollectionInfo>;
    fn get_collection(&self, name: &str) -> Option<(usize, Metric)>;
    fn create_collection(&self, name: &str, dim: usize, metric: Metric) -> Result<(), VectorError>;
    fn upsert(&self, collection: &str, id: &str, item: VectorItem) -> Result<(), VectorError>;
    fn delete(&self, collection: &str, id: &str) -> Result<(), VectorError>;
    fn search(&self, collection: &str, req: SearchRequest) -> Result<Vec<SearchHit>, VectorError>;
    fn compact(&self, collection: &str, force: bool) -> Result<bool, VectorError>;
    fn retrain_ivf(&self, collection: &str, force: bool) -> Result<bool, VectorError>;
}

/// `DiskVectorIndex` is the contract future DiskANN/Vamana style indexes must satisfy.
/// It intentionally mirrors `VectorIndex` and adds hooks for loading/offloading state
/// so that on-disk indexes can manage their own caches and crash safety.
pub trait DiskVectorIndex: VectorIndex {
    /// Prepare a collection for heavy queries (e.g. open mmap files, prefetch manifests).
    fn warm_collection(&self, collection: &str) -> Result<(), VectorError>;
    /// Flush pending writes and ensure manifests/run metadata are durable.
    fn sync_collection(&self, collection: &str) -> Result<(), VectorError>;
}

/// Additional operations required for DiskANN/Vamana style indexes.
pub trait DiskAnnIndex: DiskVectorIndex {
    /// Build or rebuild the on-disk graph representation.
    fn build_disk_index(
        &self,
        collection: &str,
        params: DiskAnnBuildParams,
    ) -> Result<(), VectorError>;
    /// Drop any on-disk assets for the collection (graph files, manifests, caches).
    fn drop_disk_index(&self, collection: &str) -> Result<(), VectorError>;
    /// Report whether an on-disk graph is available and basic metadata.
    fn disk_index_status(&self, collection: &str) -> Result<DiskIndexStatus, VectorError>;
    /// Update tuning knobs stored in the manifest without rebuilding immediately.
    fn update_disk_index_params(
        &self,
        collection: &str,
        params: DiskAnnBuildParams,
    ) -> Result<DiskAnnBuildParams, VectorError>;
}
