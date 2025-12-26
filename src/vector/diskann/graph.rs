use crate::vector::q8::{self, QuantizedVec};
use crate::vector::Metric;
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::convert::TryFrom;
use std::io::{self, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

const GRAPH_MAGIC_V2: u32 = 0x444B414E; // "DKAN"
const GRAPH_VERSION_V2: u16 = 2;
const GRAPH_HEADER_BYTES: usize = 32;
const ENTRY_REFINEMENT_STEPS: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphFile {
    pub dim: usize,
    pub metric: Metric,
    #[serde(default)]
    pub entry_idx: usize,
    pub nodes: Vec<GraphFileNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphFileNode {
    pub id: String,
    pub vector: QuantizedVec,
    pub neighbors: Vec<usize>,
}

#[derive(Clone)]
pub struct DiskGraph {
    dim: usize,
    entry_idx: usize,
    backing: GraphBacking,
}

#[derive(Clone)]
enum GraphBacking {
    InMemory(Vec<Arc<CachedNode>>),
    Paged(Arc<PagedGraph>),
}

#[derive(Clone)]
struct CachedNode {
    id: String,
    vector: QuantizedVec,
    neighbors: Vec<usize>,
}

struct PagedGraph {
    file: Mutex<std::fs::File>,
    offsets: Vec<u64>,
    cache: Mutex<LruCache<usize, Arc<CachedNode>>>,
    cache_capacity: usize,
}

impl DiskGraph {
    pub fn from_graph_file(file: GraphFile) -> Self {
        let entry = file.entry_idx.min(file.nodes.len().saturating_sub(1));
        let nodes: Vec<Arc<CachedNode>> = file
            .nodes
            .into_iter()
            .map(|node| {
                Arc::new(CachedNode {
                    id: node.id,
                    vector: node.vector,
                    neighbors: node.neighbors,
                })
            })
            .collect();
        Self {
            dim: file.dim,
            entry_idx: entry,
            backing: GraphBacking::InMemory(nodes),
        }
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        match read_header(path) {
            Ok(Some(header)) => {
                let offsets = read_offsets(path, header.node_count)?;
                let paged = PagedGraph::new(path, offsets)?;
                Ok(Self {
                    dim: header.dim as usize,
                    entry_idx: header.entry_idx as usize,
                    backing: GraphBacking::Paged(Arc::new(paged)),
                })
            }
            _ => {
                let bytes = std::fs::read(path)?;
                let graph: GraphFile = bincode::deserialize(&bytes).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "diskann graph deserialize")
                })?;
                Ok(Self::from_graph_file(graph))
            }
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        simd_enabled: bool,
        search_list: usize,
        max_candidates: usize,
    ) -> io::Result<Vec<(usize, f32)>> {
        if self.len() == 0 || search_list == 0 || query.len() != self.dim {
            return Ok(Vec::new());
        }
        let q_query = q8::quantize_per_vector(query);
        let entry = self.entry_point(&q_query, simd_enabled)?;
        let entry_node = self.load_node(entry)?;
        let entry_score = q8::dot(&entry_node.vector, &q_query, simd_enabled);
        let mut visited = HashSet::new();
        let mut to_visit = BinaryHeap::new();
        to_visit.push(VisitState {
            idx: entry,
            score: entry_score,
        });
        let mut approx_hits = Vec::new();
        while let Some(state) = to_visit.pop() {
            if !visited.insert(state.idx) {
                continue;
            }
            approx_hits.push((state.idx, state.score));
            if visited.len() >= search_list {
                break;
            }
            let node = self.load_node(state.idx)?;
            for &neighbor in &node.neighbors {
                if visited.contains(&neighbor) {
                    continue;
                }
                let neighbor_node = self.load_node(neighbor)?;
                let score = q8::dot(&neighbor_node.vector, &q_query, simd_enabled);
                to_visit.push(VisitState {
                    idx: neighbor,
                    score,
                });
            }
        }
        approx_hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        approx_hits.truncate(max_candidates.max(1));
        Ok(approx_hits)
    }

    pub fn id_for(&self, idx: usize) -> io::Result<Option<String>> {
        if idx >= self.len() {
            return Ok(None);
        }
        let node = self.load_node(idx)?;
        Ok(Some(node.id.clone()))
    }

    fn len(&self) -> usize {
        match &self.backing {
            GraphBacking::InMemory(nodes) => nodes.len(),
            GraphBacking::Paged(paged) => paged.offsets.len(),
        }
    }

    fn load_node(&self, idx: usize) -> io::Result<Arc<CachedNode>> {
        match &self.backing {
            GraphBacking::InMemory(nodes) => nodes.get(idx).cloned().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "node idx out of bounds")
            }),
            GraphBacking::Paged(paged) => paged.node(idx),
        }
    }

    fn entry_point(&self, query: &QuantizedVec, simd_enabled: bool) -> io::Result<usize> {
        if self.len() <= 1 {
            return Ok(0);
        }
        let mut current = self.entry_idx.min(self.len() - 1);
        let mut best = current;
        let mut best_score = self.score_node(current, query, simd_enabled)?;
        for _ in 0..ENTRY_REFINEMENT_STEPS {
            let node = self.load_node(best)?;
            let mut improved = false;
            for &neighbor in &node.neighbors {
                let neighbor_score = self.score_node(neighbor, query, simd_enabled)?;
                if neighbor_score > best_score {
                    best_score = neighbor_score;
                    best = neighbor;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
            current = best;
        }
        Ok(current)
    }

    fn score_node(&self, idx: usize, query: &QuantizedVec, simd_enabled: bool) -> io::Result<f32> {
        let node = self.load_node(idx)?;
        Ok(q8::dot(&node.vector, query, simd_enabled))
    }
}

impl PagedGraph {
    fn new(path: &Path, offsets: Vec<u64>) -> io::Result<Self> {
        let cache_capacity = default_cache_capacity(offsets.len());
        let file = std::fs::File::open(path)?;
        let cache = LruCache::new(
            NonZeroUsize::new(cache_capacity).unwrap_or_else(|| NonZeroUsize::new(64).unwrap()),
        );
        Ok(Self {
            file: Mutex::new(file),
            offsets,
            cache: Mutex::new(cache),
            cache_capacity,
        })
    }

    fn node(&self, idx: usize) -> io::Result<Arc<CachedNode>> {
        if idx >= self.offsets.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "disk graph node idx",
            ));
        }
        if let Some(node) = self.cache.lock().get(&idx) {
            return Ok(node.clone());
        }
        let node = Arc::new(self.read_node(idx)?);
        let mut cache = self.cache.lock();
        if cache.len() >= self.cache_capacity {
            cache.pop_lru();
        }
        cache.put(idx, node.clone());
        Ok(node)
    }

    fn read_node(&self, idx: usize) -> io::Result<CachedNode> {
        let offset = self.offsets[idx];
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(offset))?;
        let id_len = read_u16(&mut *file)? as usize;
        let mut id_bytes = vec![0u8; id_len];
        file.read_exact(&mut id_bytes)?;
        let id = String::from_utf8(id_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "diskann id utf8"))?;
        let scale = read_f32(&mut *file)?;
        let dims = read_u32(&mut *file)? as usize;
        let mut data_bytes = vec![0u8; dims];
        file.read_exact(&mut data_bytes)?;
        let data = data_bytes.into_iter().map(|b| b as i8).collect();
        let neighbor_count = read_u16(&mut *file)? as usize;
        let mut neighbors = Vec::with_capacity(neighbor_count);
        for _ in 0..neighbor_count {
            neighbors.push(read_u32(&mut *file)? as usize);
        }
        Ok(CachedNode {
            id,
            vector: QuantizedVec { scale, data },
            neighbors,
        })
    }
}

pub(crate) fn encode_graph_file(graph: &GraphFile) -> io::Result<Vec<u8>> {
    let node_count = graph.nodes.len();
    if node_count >= u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "diskann graph too large",
        ));
    }
    let mut buffer = vec![0u8; GRAPH_HEADER_BYTES + node_count * 8];
    let mut offsets = Vec::with_capacity(node_count);
    for node in &graph.nodes {
        let offset = buffer.len() as u64;
        offsets.push(offset);
        encode_node(&mut buffer, node)?;
    }
    let header = GraphHeader {
        dim: graph.dim as u32,
        metric: graph.metric,
        node_count: node_count as u32,
        entry_idx: graph.entry_idx.min(node_count.saturating_sub(1)) as u32,
    };
    header.write(&mut buffer[..GRAPH_HEADER_BYTES]);
    for (idx, offset) in offsets.iter().enumerate() {
        let start = GRAPH_HEADER_BYTES + idx * 8;
        buffer[start..start + 8].copy_from_slice(&offset.to_le_bytes());
    }
    Ok(buffer)
}

fn encode_node(buffer: &mut Vec<u8>, node: &GraphFileNode) -> io::Result<()> {
    let id_bytes = node.id.as_bytes();
    if id_bytes.len() > u16::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "diskann id too long",
        ));
    }
    if node.vector.data.len() > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "diskann vec too long",
        ));
    }
    buffer.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
    buffer.extend_from_slice(id_bytes);
    buffer.extend_from_slice(&node.vector.scale.to_le_bytes());
    buffer.extend_from_slice(&(node.vector.data.len() as u32).to_le_bytes());
    buffer.extend(node.vector.data.iter().map(|v| *v as u8));
    let neighbor_count = node.neighbors.len().min(u16::MAX as usize);
    buffer.extend_from_slice(&(neighbor_count as u16).to_le_bytes());
    for &neighbor in node.neighbors.iter().take(neighbor_count) {
        let idx = u32::try_from(neighbor).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "diskann neighbor idx overflow")
        })?;
        buffer.extend_from_slice(&idx.to_le_bytes());
    }
    Ok(())
}

fn read_header(path: &Path) -> io::Result<Option<GraphHeader>> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; GRAPH_HEADER_BYTES];
    match file.read_exact(&mut buf) {
        Ok(_) => Ok(GraphHeader::decode(&buf)),
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(err) => Err(err),
    }
}

fn read_offsets(path: &Path, node_count: u32) -> io::Result<Vec<u64>> {
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(GRAPH_HEADER_BYTES as u64))?;
    let mut offsets = vec![0u8; node_count as usize * 8];
    file.read_exact(&mut offsets)?;
    let mut result = Vec::with_capacity(node_count as usize);
    for chunk in offsets.chunks_exact(8) {
        result.push(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(result)
}

struct GraphHeader {
    dim: u32,
    metric: Metric,
    node_count: u32,
    entry_idx: u32,
}

impl GraphHeader {
    fn decode(buf: &[u8; GRAPH_HEADER_BYTES]) -> Option<Self> {
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != GRAPH_MAGIC_V2 {
            return None;
        }
        let version = u16::from_le_bytes([buf[4], buf[5]]);
        if version != GRAPH_VERSION_V2 {
            return None;
        }
        let metric = Metric::try_from(buf[6]).ok()?;
        let dim = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let node_count = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        let entry_idx = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        Some(Self {
            dim,
            metric,
            node_count,
            entry_idx,
        })
    }

    fn write(&self, dest: &mut [u8]) {
        dest[0..4].copy_from_slice(&GRAPH_MAGIC_V2.to_le_bytes());
        dest[4..6].copy_from_slice(&GRAPH_VERSION_V2.to_le_bytes());
        dest[6] = self.metric.into();
        dest[7] = 0;
        dest[8..12].copy_from_slice(&self.dim.to_le_bytes());
        dest[12..16].copy_from_slice(&self.node_count.to_le_bytes());
        dest[16..20].copy_from_slice(&self.entry_idx.to_le_bytes());
        dest[20..GRAPH_HEADER_BYTES].fill(0);
    }
}

impl TryFrom<u8> for Metric {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Metric::Cosine),
            1 => Ok(Metric::Dot),
            _ => Err(()),
        }
    }
}

impl From<Metric> for u8 {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::Cosine => 0,
            Metric::Dot => 1,
        }
    }
}

#[derive(Clone)]
struct VisitState {
    idx: usize,
    score: f32,
}

impl PartialEq for VisitState {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}

impl Eq for VisitState {}

impl PartialOrd for VisitState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for VisitState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn default_cache_capacity(node_count: usize) -> usize {
    match node_count {
        0..=512 => node_count.max(64),
        _ => ((node_count as f64).sqrt() as usize * 16).clamp(256, 8192),
    }
}

fn read_u16(reader: &mut dyn Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(reader: &mut dyn Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32(reader: &mut dyn Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::q8;

    #[test]
    fn paged_graph_roundtrip_search() {
        let qa = q8::quantize_per_vector(&[1.0, 0.0]);
        let qb = q8::quantize_per_vector(&[0.0, 1.0]);
        let nodes = vec![
            GraphFileNode {
                id: "a".into(),
                vector: qa.clone(),
                neighbors: vec![1],
            },
            GraphFileNode {
                id: "b".into(),
                vector: qb.clone(),
                neighbors: vec![0],
            },
        ];
        let graph_file = GraphFile {
            dim: 2,
            metric: Metric::Cosine,
            entry_idx: 0,
            nodes,
        };
        let bytes = encode_graph_file(&graph_file).expect("encode");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.bin");
        std::fs::write(&path, bytes).unwrap();
        let graph = DiskGraph::load_from_path(&path).expect("load graph");
        let hits = graph.search(&[1.0, 0.0], true, 4, 2).expect("search hits");
        assert!(!hits.is_empty());
        let id = graph.id_for(hits[0].0).unwrap().unwrap();
        assert_eq!(id, "a");
    }
}
