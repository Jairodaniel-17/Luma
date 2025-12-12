use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct VectorStore(Arc<Inner>);

struct Inner {
    collections: RwLock<HashMap<String, Collection>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    pub dim: usize,
    pub metric: Metric,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorItem {
    pub vector: Vec<f32>,
    pub meta: serde_json::Value,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    Cosine,
    Dot,
}

#[derive(Clone, Debug)]
struct Collection {
    dim: usize,
    metric: Metric,
    items: HashMap<String, VectorItem>,
}

#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("collection not found")]
    CollectionNotFound,
    #[error("id not found")]
    IdNotFound,
    #[error("collection already exists")]
    CollectionExists,
    #[error("vector dim mismatch")]
    DimMismatch,
    #[error("id already exists")]
    IdExists,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    pub filters: Option<serde_json::Value>,
    pub include_meta: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub meta: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistVectorSnapshot {
    pub collections: HashMap<String, PersistCollection>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistCollection {
    pub dim: usize,
    pub metric: Metric,
    pub items: HashMap<String, VectorItem>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self(Arc::new(Inner {
            collections: RwLock::new(HashMap::new()),
        }))
    }

    pub fn create_collection(
        &self,
        name: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        if cols.contains_key(name) {
            return Err(VectorError::CollectionExists);
        }
        cols.insert(
            name.to_string(),
            Collection {
                dim,
                metric,
                items: HashMap::new(),
            },
        );
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Option<(usize, Metric)> {
        let cols = self.0.collections.read();
        cols.get(name).map(|c| (c.dim, c.metric))
    }

    pub fn add(&self, collection: &str, id: &str, item: VectorItem) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols.get_mut(collection).ok_or(VectorError::CollectionNotFound)?;
        if item.vector.len() != c.dim {
            return Err(VectorError::DimMismatch);
        }
        if c.items.contains_key(id) {
            return Err(VectorError::IdExists);
        }
        c.items.insert(id.to_string(), item);
        Ok(())
    }

    pub fn upsert(
        &self,
        collection: &str,
        id: &str,
        item: VectorItem,
    ) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols.get_mut(collection).ok_or(VectorError::CollectionNotFound)?;
        if item.vector.len() != c.dim {
            return Err(VectorError::DimMismatch);
        }
        c.items.insert(id.to_string(), item);
        Ok(())
    }

    pub fn update(
        &self,
        collection: &str,
        id: &str,
        vector: Option<Vec<f32>>,
        meta: Option<serde_json::Value>,
    ) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols.get_mut(collection).ok_or(VectorError::CollectionNotFound)?;
        let existing = c.items.get_mut(id).ok_or(VectorError::IdNotFound)?;

        if let Some(v) = vector {
            if v.len() != c.dim {
                return Err(VectorError::DimMismatch);
            }
            existing.vector = v;
        }
        if let Some(m) = meta {
            existing.meta = m;
        }
        Ok(())
    }

    pub fn delete(&self, collection: &str, id: &str) -> Result<(), VectorError> {
        let mut cols = self.0.collections.write();
        let c = cols.get_mut(collection).ok_or(VectorError::CollectionNotFound)?;
        if c.items.remove(id).is_some() {
            Ok(())
        } else {
            Err(VectorError::IdNotFound)
        }
    }

    pub fn get(&self, collection: &str, id: &str) -> Result<Option<VectorItem>, VectorError> {
        let cols = self.0.collections.read();
        let c = cols.get(collection).ok_or(VectorError::CollectionNotFound)?;
        Ok(c.items.get(id).cloned())
    }

    pub fn search(&self, collection: &str, req: SearchRequest) -> Result<Vec<SearchHit>, VectorError> {
        let cols = self.0.collections.read();
        let c = cols.get(collection).ok_or(VectorError::CollectionNotFound)?;
        if req.vector.len() != c.dim {
            return Err(VectorError::DimMismatch);
        }
        let include_meta = req.include_meta.unwrap_or(false);
        let k = req.k.max(1).min(10_000);

        let query_norm = if matches!(c.metric, Metric::Cosine) {
            norm(&req.vector).max(1e-9)
        } else {
            1.0
        };

        let mut hits = Vec::new();
        for (id, item) in c.items.iter() {
            if !matches_filters(&item.meta, req.filters.as_ref()) {
                continue;
            }
            let score = match c.metric {
                Metric::Dot => dot(&req.vector, &item.vector),
                Metric::Cosine => {
                    let denom = query_norm * norm(&item.vector).max(1e-9);
                    dot(&req.vector, &item.vector) / denom
                }
            };
            hits.push(SearchHit {
                id: id.clone(),
                score,
                meta: include_meta.then(|| item.meta.clone()),
            });
        }

        if hits.len() > k {
            hits.select_nth_unstable_by(k - 1, |a, b| b.score.total_cmp(&a.score));
            hits.truncate(k);
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(hits)
    }

    pub fn snapshot(&self) -> PersistVectorSnapshot {
        let cols = self.0.collections.read();
        PersistVectorSnapshot {
            collections: cols
                .iter()
                .map(|(name, c)| {
                    (
                        name.clone(),
                        PersistCollection {
                            dim: c.dim,
                            metric: c.metric,
                            items: c.items.clone(),
                        },
                    )
                })
                .collect(),
        }
    }

    pub fn load_snapshot(&self, snapshot: PersistVectorSnapshot) -> anyhow::Result<()> {
        let mut cols = self.0.collections.write();
        cols.clear();
        for (name, c) in snapshot.collections {
            cols.insert(
                name,
                Collection {
                    dim: c.dim,
                    metric: c.metric,
                    items: c.items,
                },
            );
        }
        Ok(())
    }

    pub fn apply_wal_create(&self, data: &serde_json::Value) -> anyhow::Result<()> {
        let collection = data
            .get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing collection"))?;
        let dim = data.get("dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let metric: Metric = serde_json::from_value(data.get("metric").cloned().unwrap_or(serde_json::Value::String("cosine".into())))?;
        let _ = self.create_collection(collection, dim, metric);
        Ok(())
    }

    pub fn apply_wal_item(&self, event_type: &str, data: &serde_json::Value) -> anyhow::Result<()> {
        let collection = data
            .get("collection")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing collection"))?;
        let id = data
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing id"))?;
        match event_type {
            "vector_deleted" => {
                let _ = self.delete(collection, id);
            }
            "vector_added" | "vector_upserted" | "vector_updated" => {
                let vector: Vec<f32> = serde_json::from_value(
                    data.get("vector")
                        .cloned()
                        .unwrap_or(serde_json::Value::Array(vec![])),
                )?;
                let meta = data.get("meta").cloned().unwrap_or(serde_json::Value::Null);
                let item = VectorItem { vector, meta };
                match event_type {
                    "vector_added" => {
                        let _ = self.add(collection, id, item);
                    }
                    "vector_upserted" => {
                        let _ = self.upsert(collection, id, item);
                    }
                    "vector_updated" => {
                        let _ = self.upsert(collection, id, item);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn matches_filters(meta: &serde_json::Value, filters: Option<&serde_json::Value>) -> bool {
    let Some(filters) = filters else { return true };
    let serde_json::Value::Object(f) = filters else { return false };
    let serde_json::Value::Object(m) = meta else { return false };

    for (k, v) in f.iter() {
        match m.get(k) {
            Some(mv) if mv == v => {}
            _ => return false,
        }
    }
    true
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_cosine_prefers_closer() {
        let store = VectorStore::new();
        store.create_collection("docs", 2, Metric::Cosine).unwrap();
        store
            .upsert(
                "docs",
                "a",
                VectorItem {
                    vector: vec![1.0, 0.0],
                    meta: serde_json::json!({"tag":"x"}),
                },
            )
            .unwrap();
        store
            .upsert(
                "docs",
                "b",
                VectorItem {
                    vector: vec![0.0, 1.0],
                    meta: serde_json::json!({"tag":"y"}),
                },
            )
            .unwrap();

        let hits = store
            .search(
                "docs",
                SearchRequest {
                    vector: vec![0.9, 0.1],
                    k: 1,
                    filters: None,
                    include_meta: None,
                },
            )
            .unwrap();
        assert_eq!(hits[0].id, "a");
    }
}
