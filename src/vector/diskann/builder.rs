use super::graph::{GraphFile, GraphFileNode};
use crate::vector::index::DiskAnnBuildParams;
use crate::vector::q8::{self, QuantizedVec};
use crate::vector::Metric;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

const BUILDER_SEED: u64 = 0xD15C_AAAE;
const MIN_ENTRY_EXPANSIONS: usize = 4;

pub fn build_graph_file(
    metric: Metric,
    vectors: &[(String, QuantizedVec)],
    params: &DiskAnnBuildParams,
    simd_enabled: bool,
) -> GraphFile {
    let dim = vectors.first().map(|(_, q)| q.dims()).unwrap_or(0);
    let builder = GraphBuilder::new(vectors, params, simd_enabled);
    let (adjacency, entry_point) = builder.build();
    let nodes = vectors
        .iter()
        .enumerate()
        .map(|(idx, (id, vector))| GraphFileNode {
            id: id.clone(),
            vector: vector.clone(),
            neighbors: adjacency[idx].clone(),
        })
        .collect();
    GraphFile {
        dim,
        metric,
        entry_idx: entry_point,
        nodes,
    }
}

struct GraphBuilder<'a> {
    vectors: &'a [(String, QuantizedVec)],
    params: &'a DiskAnnBuildParams,
    simd_enabled: bool,
    rng: StdRng,
    max_degree: usize,
}

impl<'a> GraphBuilder<'a> {
    fn new(
        vectors: &'a [(String, QuantizedVec)],
        params: &'a DiskAnnBuildParams,
        simd_enabled: bool,
    ) -> Self {
        Self {
            vectors,
            params,
            simd_enabled,
            rng: StdRng::seed_from_u64(BUILDER_SEED),
            max_degree: params.max_degree.max(1),
        }
    }

    fn build(mut self) -> (Vec<Vec<usize>>, usize) {
        let n = self.vectors.len();
        let mut adjacency = vec![Vec::new(); n];
        if n <= 1 {
            return (adjacency, 0);
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut self.rng);
        let mut processed = vec![false; n];
        let mut entry_idx = order[0];
        processed[entry_idx] = true;

        for (pos, &idx) in order.iter().enumerate() {
            if pos == 0 {
                continue;
            }
            let neighbors = self.attach_node(idx, entry_idx, &mut adjacency, &processed);
            processed[idx] = true;
            if neighbors.len() >= MIN_ENTRY_EXPANSIONS && self.rng.gen_bool(0.1) {
                entry_idx = idx;
            }
        }
        self.ensure_bidirectional(&mut adjacency);
        self.ensure_min_degree(&mut adjacency);
        (adjacency, entry_idx)
    }

    fn attach_node(
        &mut self,
        idx: usize,
        entry_idx: usize,
        adjacency: &mut [Vec<usize>],
        processed: &[bool],
    ) -> Vec<usize> {
        let mut candidates = self.search_candidates(idx, entry_idx, adjacency, processed);
        if candidates.is_empty() {
            candidates = self
                .sample_candidates(idx, processed, self.max_degree * 2)
                .into_iter()
                .map(|cand| (cand, self.score(idx, cand)))
                .collect();
        }
        let neighbors = self.prune_neighbors(idx, candidates);
        adjacency[idx] = neighbors.clone();
        for &neighbor in &neighbors {
            if adjacency[neighbor].contains(&idx) {
                continue;
            }
            adjacency[neighbor].push(idx);
            if adjacency[neighbor].len() > self.max_degree {
                self.truncate_neighbors(neighbor, adjacency);
            }
        }
        neighbors
    }

    fn search_candidates(
        &mut self,
        idx: usize,
        entry_idx: usize,
        adjacency: &[Vec<usize>],
        processed: &[bool],
    ) -> Vec<(usize, f32)> {
        let processed_count = processed.iter().filter(|v| **v).count();
        if processed_count == 0 {
            return Vec::new();
        }
        let entry = if processed[entry_idx] {
            entry_idx
        } else {
            processed
                .iter()
                .enumerate()
                .find(|(_, seen)| **seen)
                .map(|(i, _)| i)
                .unwrap_or(entry_idx)
        };
        let budget = self
            .params
            .search_list_size
            .max(self.max_degree * 2)
            .min(processed_count.max(1));
        let mut visited = HashSet::new();
        let mut to_visit = BinaryHeap::new();
        let entry_score = self.score(idx, entry);
        to_visit.push(VisitState {
            idx: entry,
            score: entry_score,
        });
        let mut found = Vec::new();
        while let Some(state) = to_visit.pop() {
            if state.idx == idx || !visited.insert(state.idx) {
                continue;
            }
            found.push((state.idx, state.score));
            if visited.len() >= budget {
                break;
            }
            for &neighbor in &adjacency[state.idx] {
                if !processed[neighbor] || neighbor == idx {
                    continue;
                }
                if visited.contains(&neighbor) {
                    continue;
                }
                let score = self.score(idx, neighbor);
                to_visit.push(VisitState {
                    idx: neighbor,
                    score,
                });
            }
        }
        if found.len() < self.max_degree {
            let supplement =
                self.sample_candidates(idx, processed, self.max_degree.saturating_mul(2));
            found.extend(supplement.into_iter().map(|cand| {
                let score = self.score(idx, cand);
                (cand, score)
            }));
            found.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            found.dedup_by(|a, b| a.0 == b.0);
        }
        found
    }

    fn sample_candidates(&mut self, idx: usize, processed: &[bool], target: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = processed
            .iter()
            .enumerate()
            .filter_map(|(i, seen)| (*seen && i != idx).then_some(i))
            .collect();
        if pool.is_empty() {
            return Vec::new();
        }
        pool.shuffle(&mut self.rng);
        pool.truncate(target.min(pool.len()));
        pool
    }

    fn prune_neighbors(&self, idx: usize, mut candidates: Vec<(usize, f32)>) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        candidates.dedup_by(|a, b| a.0 == b.0);
        let min_degree = self.max_degree.max(2) / 2;
        let dynamic_target = self.dynamic_degree(candidates.len());
        let mut selected = Vec::new();
        for (cand_idx, score) in &candidates {
            if *cand_idx == idx {
                continue;
            }
            if selected.len() >= dynamic_target {
                break;
            }
            let redundant = selected.iter().any(|existing| {
                let diversity = self.score(*existing, *cand_idx);
                diversity >= *score * 0.98
            });
            if redundant {
                continue;
            }
            selected.push(*cand_idx);
        }
        if selected.len() < min_degree {
            for (cand_idx, _) in candidates {
                if selected.len() >= min_degree || selected.contains(&cand_idx) || cand_idx == idx {
                    continue;
                }
                selected.push(cand_idx);
            }
        }
        if selected.len() > self.max_degree {
            selected.truncate(self.max_degree);
        }
        selected
    }

    fn dynamic_degree(&self, candidate_count: usize) -> usize {
        if self.max_degree <= 2 {
            return self.max_degree;
        }
        let base = (self.max_degree as f32 * 0.6).ceil() as usize;
        let density = (candidate_count as f32 / self.max_degree.max(1) as f32).clamp(0.5, 2.0);
        let desired = (self.max_degree as f32 * density).round() as usize;
        desired.clamp(base, self.max_degree)
    }

    fn ensure_bidirectional(&self, adjacency: &mut [Vec<usize>]) {
        for idx in 0..adjacency.len() {
            let neighbors = adjacency[idx].clone();
            for &neighbor in &neighbors {
                if adjacency[neighbor].contains(&idx) {
                    continue;
                }
                adjacency[neighbor].push(idx);
                if adjacency[neighbor].len() > self.max_degree {
                    self.truncate_neighbors(neighbor, adjacency);
                }
            }
            adjacency[idx].sort_unstable();
            adjacency[idx].dedup();
        }
    }

    fn ensure_min_degree(&self, adjacency: &mut [Vec<usize>]) {
        let min_degree = self.max_degree.max(2) / 2;
        if min_degree == 0 {
            return;
        }
        for idx in 0..adjacency.len() {
            if adjacency[idx].len() >= min_degree {
                continue;
            }
            let mut scored: Vec<(usize, f32)> = (0..self.vectors.len())
                .filter(|&other| other != idx)
                .map(|other| (other, self.score(idx, other)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            for (other, _) in scored {
                if adjacency[idx].contains(&other) || idx == other {
                    continue;
                }
                adjacency[idx].push(other);
                adjacency[other].push(idx);
                self.truncate_neighbors(other, adjacency);
                if adjacency[idx].len() >= min_degree {
                    break;
                }
            }
            self.truncate_neighbors(idx, adjacency);
        }
    }

    fn truncate_neighbors(&self, idx: usize, adjacency: &mut [Vec<usize>]) {
        if adjacency[idx].len() <= self.max_degree {
            return;
        }
        let mut scored: Vec<(usize, f32)> = adjacency[idx]
            .iter()
            .map(|neighbor| (*neighbor, self.score(idx, *neighbor)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(self.max_degree);
        adjacency[idx] = scored.into_iter().map(|(n, _)| n).collect();
    }

    fn score(&self, a: usize, b: usize) -> f32 {
        q8::dot(&self.vectors[a].1, &self.vectors[b].1, self.simd_enabled)
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
