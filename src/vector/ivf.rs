use crate::vector::{simd, Metric, VectorItem};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IndexKind {
    Hnsw,
    IvfFlatQ8,
    DiskAnn,
}

impl IndexKind {
    pub fn is_ivf(self) -> bool {
        matches!(self, IndexKind::IvfFlatQ8)
    }

    pub fn is_diskann(self) -> bool {
        matches!(self, IndexKind::DiskAnn)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IvfConfig {
    pub clusters: usize,
    pub nprobe: usize,
    pub training_sample: usize,
    pub max_training_iters: usize,
    pub min_train_vectors: usize,
    pub retrain_min_deltas: usize,
}

#[derive(Clone, Debug)]
pub struct IvfState {
    centroids: Vec<Vec<f32>>,
    metric: Metric,
    trained_at_ms: u64,
}

impl IvfState {
    pub fn new(centroids: Vec<Vec<f32>>, metric: Metric, trained_at_ms: u64) -> Self {
        Self {
            centroids,
            metric,
            trained_at_ms,
        }
    }

    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub fn trained_at_ms(&self) -> u64 {
        self.trained_at_ms
    }

    pub fn assign_vector(&self, vector: &[f32], simd_enabled: bool) -> Option<usize> {
        if self.centroids.is_empty() {
            return None;
        }
        let mut best_idx = 0usize;
        let mut best_score = f32::MIN;
        for (idx, centroid) in self.centroids.iter().enumerate() {
            let score = centroid_score(self.metric, centroid, vector, simd_enabled);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        Some(best_idx)
    }

    pub fn select_probes(&self, query: &[f32], simd_enabled: bool, nprobe: usize) -> Vec<usize> {
        if self.centroids.is_empty() || nprobe == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| {
                (
                    idx,
                    centroid_score(self.metric, centroid, query, simd_enabled),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(nprobe.min(self.centroids.len()))
            .map(|(idx, _)| idx)
            .collect()
    }
}

pub fn assign_all_clusters(
    ivf: &IvfState,
    items: &HashMap<String, VectorItem>,
    simd_enabled: bool,
) -> HashMap<String, usize> {
    let mut out = HashMap::with_capacity(items.len());
    for (id, item) in items.iter() {
        if let Some(cluster) = ivf.assign_vector(&item.vector, simd_enabled) {
            out.insert(id.clone(), cluster);
        }
    }
    out
}

pub fn train_centroids(
    vectors: &[Vec<f32>],
    config: &IvfConfig,
    metric: Metric,
) -> Option<Vec<Vec<f32>>> {
    if vectors.is_empty() || config.clusters < 2 {
        return None;
    }
    let k = config.clusters.min(vectors.len());
    let mut centroids = init_kmeans_pp(vectors, k, metric);
    for _ in 0..config.max_training_iters.max(1) {
        let mut buckets: Vec<Vec<&[f32]>> = vec![Vec::new(); centroids.len()];
        for vec in vectors.iter() {
            let mut best_idx = 0usize;
            let mut best_score = f32::MIN;
            for (idx, centroid) in centroids.iter().enumerate() {
                let score = centroid_score(metric, centroid, vec, true);
                if score > best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }
            buckets[best_idx].push(vec);
        }
        for (idx, bucket) in buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let mut new_centroid = vec![0.0f32; centroids[idx].len()];
            for vec in bucket {
                for (dst, &src) in new_centroid.iter_mut().zip(vec.iter()) {
                    *dst += src;
                }
            }
            let inv = 1.0f32 / bucket.len() as f32;
            for value in new_centroid.iter_mut() {
                *value *= inv;
            }
            centroids[idx] = new_centroid;
        }
    }
    Some(centroids)
}

fn init_kmeans_pp(vectors: &[Vec<f32>], k: usize, metric: Metric) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let mut centroids = Vec::with_capacity(k);
    if let Some(first) = vectors.choose(&mut rng) {
        centroids.push(first.clone());
    }
    while centroids.len() < k {
        let mut distances = Vec::with_capacity(vectors.len());
        let mut total = 0.0f32;
        for vec in vectors {
            let mut best = f32::MIN;
            for centroid in &centroids {
                let score = centroid_score(metric, centroid, vec, true);
                best = best.max(score);
            }
            // convert similarity to distance weight
            let dist = (1.0 - best).max(0.0);
            let weight = dist * dist;
            distances.push(weight);
            total += weight;
        }
        if total <= f32::EPSILON {
            break;
        }
        let mut target = rng.gen::<f32>() * total;
        let mut chosen_idx = 0usize;
        for (idx, weight) in distances.iter().enumerate() {
            target -= *weight;
            if target <= 0.0 {
                chosen_idx = idx;
                break;
            }
        }
        centroids.push(vectors[chosen_idx].clone());
    }
    centroids
}

fn centroid_score(_metric: Metric, centroid: &[f32], vector: &[f32], simd_enabled: bool) -> f32 {
    simd::dot(centroid, vector, simd_enabled)
}
