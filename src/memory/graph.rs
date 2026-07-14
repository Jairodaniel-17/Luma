use crate::memory::ingest::now_ms;
use crate::memory::types::{BeliefVersion, EdgeType, MemoryEdge, MemoryRecord, SemanticWalkConfig};
use crate::sqlite::SqliteService;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

// ─── Scored node for the walk priority queue ──────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredNode {
    pub score: f32,
    pub hop: usize,
    pub id: String,
}

impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── GraphService ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct GraphService {
    sqlite: Arc<SqliteService>,
}

impl GraphService {
    pub fn new(sqlite: Arc<SqliteService>) -> Self {
        Self { sqlite }
    }

    // ── Edge CRUD ──────────────────────────────────────────────────────────

    pub async fn upsert_edge(&self, edge: &MemoryEdge) -> anyhow::Result<()> {
        let (sql, params) = edge_upsert_stmt(edge);
        self.sqlite.execute(sql, params).await.map(|_| ())
    }

    pub async fn delete_edge(&self, id: &str, namespace: &str) -> anyhow::Result<()> {
        self.sqlite
            .execute(
                "DELETE FROM memory_edges WHERE id = ? AND namespace = ?".to_string(),
                vec![
                    serde_json::Value::String(id.to_string()),
                    serde_json::Value::String(namespace.to_string()),
                ],
            )
            .await
            .map(|_| ())
    }

    pub async fn get_edges_from(
        &self,
        namespace: &str,
        source_id: &str,
    ) -> anyhow::Result<Vec<MemoryEdge>> {
        let rows = self
            .sqlite
            .query(
                "SELECT id, namespace, source_id, target_id, edge_type, weight, metadata, created_at_ms \
                 FROM memory_edges WHERE namespace = ? AND source_id = ?"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(source_id.to_string()),
                ],
            )
            .await?;
        rows.into_iter().map(row_to_edge).collect()
    }

    pub async fn get_edges_to(
        &self,
        namespace: &str,
        target_id: &str,
    ) -> anyhow::Result<Vec<MemoryEdge>> {
        let rows = self
            .sqlite
            .query(
                "SELECT id, namespace, source_id, target_id, edge_type, weight, metadata, created_at_ms \
                 FROM memory_edges WHERE namespace = ? AND target_id = ?"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(target_id.to_string()),
                ],
            )
            .await?;
        rows.into_iter().map(row_to_edge).collect()
    }

    // Returns all edges for a node (both directions), deduplicated.
    pub async fn get_node_edges(
        &self,
        namespace: &str,
        node_id: &str,
    ) -> anyhow::Result<(Vec<MemoryEdge>, Vec<MemoryEdge>)> {
        let (outgoing, incoming) = tokio::try_join!(
            self.get_edges_from(namespace, node_id),
            self.get_edges_to(namespace, node_id),
        )?;
        Ok((outgoing, incoming))
    }

    // ── Semantic Walk (BFS guided by cosine similarity) ────────────────────

    /// Expands from `seeds` following typed edges, scoring each node as
    /// `cosine_similarity * edge_factor * (1 + centrality)`.
    /// Returns scored nodes sorted descending by score.
    #[allow(clippy::too_many_arguments)]
    pub async fn semantic_walk(
        &self,
        namespace: &str,
        seeds: Vec<String>,
        seed_scores: &HashMap<String, f32>,
        config: &SemanticWalkConfig,
        query_vec: &[f32],
        get_vector: &(dyn Fn(&str) -> Option<Vec<f32>> + Send + Sync),
        centrality: &HashMap<String, f32>,
    ) -> anyhow::Result<Vec<ScoredNode>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut heap: BinaryHeap<ScoredNode> = BinaryHeap::new();
        let mut results: Vec<ScoredNode> = Vec::new();

        for id in &seeds {
            let base_score = seed_scores.get(id).copied().unwrap_or(0.0);
            let c = centrality.get(id).copied().unwrap_or(0.0);
            let score = base_score * (1.0 + c);
            if score >= config.min_similarity {
                heap.push(ScoredNode {
                    score,
                    hop: 0,
                    id: id.clone(),
                });
                visited.insert(id.clone());
            }
        }

        while let Some(node) = heap.pop() {
            if node.score < config.min_similarity {
                break;
            }
            results.push(node.clone());
            if results.len() >= config.max_nodes {
                break;
            }
            if node.hop >= config.max_hops {
                continue;
            }

            let (outgoing, incoming) = self.get_node_edges(namespace, &node.id).await?;
            let all_edges: Vec<_> = outgoing.iter().chain(incoming.iter()).collect();

            for edge in all_edges {
                let neighbor_id = if edge.source_id == node.id {
                    edge.target_id.clone()
                } else {
                    edge.source_id.clone()
                };
                if visited.contains(&neighbor_id) {
                    continue;
                }
                let factor = edge_factor(edge.edge_type, edge.weight);
                if factor <= 0.0 {
                    continue;
                }
                let neighbor_vec = match get_vector(&neighbor_id) {
                    Some(v) => v,
                    None => continue,
                };
                let sim = cosine_similarity(query_vec, &neighbor_vec);
                if sim < config.min_similarity {
                    continue;
                }
                let c = centrality.get(&neighbor_id).copied().unwrap_or(0.0);
                let combined = sim * factor * (1.0 + c);
                visited.insert(neighbor_id.clone());
                heap.push(ScoredNode {
                    score: combined,
                    hop: node.hop + 1,
                    id: neighbor_id,
                });
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    // ── PageRank (damping 0.85, mass-conserving, convergence-checked) ──────

    /// Computes PageRank for all memory nodes in the namespace using positive
    /// edge types only (supports, triggered_by, related_to).
    ///
    /// Edges are treated as UNDIRECTED here, mirroring `semantic_walk` (which
    /// follows edges in both directions). Centrality means "well-connected",
    /// not "pointed-to", so directionality is intentionally dropped for
    /// consistency between the two traversals.
    ///
    /// Returns mass-conserved scores that sum to ~1.0 (no max-normalization),
    /// so a dangling-node leak cannot silently shrink the totals.
    pub async fn compute_centrality(
        &self,
        namespace: &str,
    ) -> anyhow::Result<HashMap<String, f32>> {
        let rows = self
            .sqlite
            .query(
                "SELECT source_id, target_id, weight FROM memory_edges \
                 WHERE namespace = ? AND edge_type IN ('supports','triggered_by','related_to')"
                    .to_string(),
                vec![serde_json::Value::String(namespace.to_string())],
            )
            .await?;

        if rows.is_empty() {
            return Ok(HashMap::new());
        }

        let mut adjacency: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        let mut all_nodes: HashSet<String> = HashSet::new();

        for row in &rows {
            let src = row
                .get("source_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let tgt = row
                .get("target_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let w = row.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            if src.is_empty() || tgt.is_empty() {
                continue;
            }
            // Mirror the edge → undirected adjacency.
            adjacency
                .entry(src.clone())
                .or_default()
                .push((tgt.clone(), w));
            adjacency
                .entry(tgt.clone())
                .or_default()
                .push((src.clone(), w));
            all_nodes.insert(src);
            all_nodes.insert(tgt);
        }

        let nodes: Vec<String> = all_nodes.into_iter().collect();
        Ok(pagerank(&nodes, &adjacency))
    }

    /// Recomputes centrality scores and persists them to `memory_records`.
    pub async fn update_centrality_scores(&self, namespace: &str) -> anyhow::Result<usize> {
        let scores = self.compute_centrality(namespace).await?;
        let count = scores.len();
        for (id, score) in &scores {
            let _ = self
                .sqlite
                .execute(
                    "UPDATE memory_records SET centrality_score = ? WHERE id = ? AND namespace = ?"
                        .to_string(),
                    vec![
                        serde_json::json!(score),
                        serde_json::Value::String(id.clone()),
                        serde_json::Value::String(namespace.to_string()),
                    ],
                )
                .await;
        }
        Ok(count)
    }

    // ── Belief versioning ─────────────────────────────────────────────────

    /// Snapshots a MemoryRecord into `memory_history` before it is superseded.
    pub async fn append_belief_history(
        &self,
        record: &MemoryRecord,
        valid_until: u64,
    ) -> anyhow::Result<String> {
        let history_id = Uuid::new_v4().to_string();
        let (sql, params) = history_insert_stmt(&history_id, record, valid_until);
        self.sqlite.execute(sql, params).await?;
        Ok(history_id)
    }

    /// Atomically supersedes `existing`: snapshots it into `memory_history`,
    /// records the belief edge, and archives the old row — all in ONE
    /// transaction so a mid-sequence failure can't leave history without its
    /// edge or an un-archived stale record. Returns the new history id.
    ///
    /// The `EdgeType` (Contradicts vs Supersedes) is chosen by the caller; the
    /// edge id embeds the freshly generated history id, so it must be built here.
    pub async fn supersede_with_history(
        &self,
        namespace: &str,
        existing: &MemoryRecord,
        new_id: &str,
        edge_type: EdgeType,
        is_contradiction: bool,
        now_ms: u64,
    ) -> anyhow::Result<String> {
        let history_id = Uuid::new_v4().to_string();
        let (history_sql, history_params) = history_insert_stmt(&history_id, existing, now_ms);

        let edge = MemoryEdge {
            id: format!("{edge_type}::{new_id}::{history_id}"),
            namespace: namespace.to_string(),
            source_id: new_id.to_string(),
            target_id: existing.id.clone(),
            edge_type,
            weight: 1.0,
            metadata: serde_json::json!({
                "reason": "upsert_fact_overwrite",
                "contradiction": is_contradiction,
            }),
            created_at_ms: now_ms,
        };
        let (edge_sql, edge_params) = edge_upsert_stmt(&edge);

        let archive = (
            "UPDATE memory_records SET status = 'archived' WHERE namespace = ? AND id = ?"
                .to_string(),
            vec![
                serde_json::Value::String(namespace.to_string()),
                serde_json::Value::String(existing.id.clone()),
            ],
        );

        self.sqlite
            .execute_tx(vec![
                (history_sql, history_params),
                (edge_sql, edge_params),
                archive,
            ])
            .await?;
        Ok(history_id)
    }

    pub async fn get_belief_history(
        &self,
        namespace: &str,
        fact_key: &str,
    ) -> anyhow::Result<Vec<BeliefVersion>> {
        let rows = self
            .sqlite
            .query(
                "SELECT id, fact_key, namespace, entity_id, content, confidence, status, \
                  superseded_by, valid_from, valid_until, created_at_ms \
                 FROM memory_history WHERE namespace = ? AND fact_key = ? \
                 ORDER BY valid_from DESC"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(fact_key.to_string()),
                ],
            )
            .await?;
        rows.into_iter().map(row_to_belief_version).collect()
    }

    // ── Cached centrality lookup ───────────────────────────────────────────

    pub async fn load_centrality_scores(
        &self,
        namespace: &str,
    ) -> anyhow::Result<HashMap<String, f32>> {
        let rows = self
            .sqlite
            .query(
                "SELECT id, centrality_score FROM memory_records WHERE namespace = ? AND centrality_score > 0"
                    .to_string(),
                vec![serde_json::Value::String(namespace.to_string())],
            )
            .await?;
        let map = rows
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?.to_string();
                let score = row.get("centrality_score")?.as_f64()? as f32;
                Some((id, score))
            })
            .collect();
        Ok(map)
    }
}

// ── SQL statement builders (shared by single-write and transactional paths) ─

fn edge_upsert_stmt(edge: &MemoryEdge) -> (String, Vec<serde_json::Value>) {
    (
        "INSERT OR REPLACE INTO memory_edges \
         (id, namespace, source_id, target_id, edge_type, weight, metadata, created_at_ms) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            .to_string(),
        vec![
            serde_json::Value::String(edge.id.clone()),
            serde_json::Value::String(edge.namespace.clone()),
            serde_json::Value::String(edge.source_id.clone()),
            serde_json::Value::String(edge.target_id.clone()),
            serde_json::Value::String(edge.edge_type.to_string()),
            serde_json::json!(edge.weight),
            serde_json::Value::String(edge.metadata.to_string()),
            serde_json::json!(edge.created_at_ms),
        ],
    )
}

fn history_insert_stmt(
    history_id: &str,
    record: &MemoryRecord,
    valid_until: u64,
) -> (String, Vec<serde_json::Value>) {
    let status_str = match record.status {
        crate::memory::types::MemoryStatus::Draft => "draft",
        crate::memory::types::MemoryStatus::Active => "active",
        crate::memory::types::MemoryStatus::Archived => "archived",
    };
    let fact_key = record
        .metadata
        .get("fact_key")
        .and_then(|v| v.as_str())
        .unwrap_or(&record.id)
        .to_string();
    (
        "INSERT INTO memory_history \
         (id, fact_key, namespace, entity_id, content, confidence, status, \
          superseded_by, valid_from, valid_until, created_at_ms) \
         VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)"
            .to_string(),
        vec![
            serde_json::Value::String(history_id.to_string()),
            serde_json::Value::String(fact_key),
            serde_json::Value::String(record.namespace.clone()),
            record
                .entity_id
                .clone()
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null),
            serde_json::Value::String(record.content.clone()),
            serde_json::json!(record.confidence),
            serde_json::Value::String(status_str.to_string()),
            serde_json::json!(record.created_at_ms),
            serde_json::json!(valid_until),
            serde_json::json!(now_ms()),
        ],
    )
}

// ── Row deserialization helpers ────────────────────────────────────────────

fn row_to_edge(row: serde_json::Value) -> anyhow::Result<MemoryEdge> {
    let edge_type_str = row
        .get("edge_type")
        .and_then(|v| v.as_str())
        .unwrap_or("related_to");
    let edge_type: EdgeType = edge_type_str.parse().unwrap_or(EdgeType::RelatedTo);
    Ok(MemoryEdge {
        id: row_str(&row, "id")?,
        namespace: row_str(&row, "namespace")?,
        source_id: row_str(&row, "source_id")?,
        target_id: row_str(&row, "target_id")?,
        edge_type,
        weight: row.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        metadata: row
            .get("metadata")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::Value::Null),
        created_at_ms: row
            .get("created_at_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or_default(),
    })
}

fn row_to_belief_version(row: serde_json::Value) -> anyhow::Result<BeliefVersion> {
    Ok(BeliefVersion {
        id: row_str(&row, "id")?,
        fact_key: row_str(&row, "fact_key")?,
        namespace: row_str(&row, "namespace")?,
        entity_id: row
            .get("entity_id")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        content: row_str(&row, "content")?,
        confidence: row
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32,
        status: row_str(&row, "status")?,
        superseded_by: row
            .get("superseded_by")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        valid_from: row
            .get("valid_from")
            .and_then(|v| v.as_u64())
            .unwrap_or_default(),
        valid_until: row.get("valid_until").and_then(|v| v.as_u64()),
        created_at_ms: row
            .get("created_at_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or_default(),
    })
}

fn row_str(row: &serde_json::Value, key: &str) -> anyhow::Result<String> {
    row.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("missing field `{key}` in row"))
}

// ── Math helpers ──────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

fn edge_factor(edge_type: EdgeType, weight: f32) -> f32 {
    match edge_type {
        EdgeType::Supports => 1.0 * weight,
        EdgeType::TriggeredBy => 0.8 * weight,
        EdgeType::RelatedTo => 0.7 * weight,
        EdgeType::Contradicts => -0.5,
        EdgeType::Supersedes => 0.0,
    }
}

/// Standard PageRank over a (pre-built) weighted adjacency map.
///
/// Conserves total mass: each iteration the rank sitting on dangling nodes
/// (no outgoing weight) is collected and redistributed uniformly, so scores
/// sum to ~1.0 instead of leaking away. Iterates until the L1 delta drops
/// below `EPSILON` or `MAX_ITERS` is hit. O(nodes + edges) per iteration.
fn pagerank(
    nodes: &[String],
    adjacency: &HashMap<String, Vec<(String, f32)>>,
) -> HashMap<String, f32> {
    let n = nodes.len();
    if n == 0 {
        return HashMap::new();
    }
    const DAMPING: f32 = 0.85;
    const EPSILON: f32 = 1e-6;
    const MAX_ITERS: usize = 100;
    let n_f = n as f32;

    let init = 1.0 / n_f;
    let mut scores: HashMap<String, f32> = nodes.iter().map(|id| (id.clone(), init)).collect();

    // Precompute out-weight totals once; adjacency is immutable across iterations.
    let out_totals: HashMap<String, f32> = adjacency
        .iter()
        .map(|(src, targets)| (src.clone(), targets.iter().map(|(_, w)| *w).sum()))
        .collect();

    for _ in 0..MAX_ITERS {
        // Rank held by dangling nodes (no positive outgoing weight) is pooled
        // and spread uniformly so no mass is lost.
        let dangling: f32 = nodes
            .iter()
            .filter(|id| out_totals.get(*id).copied().unwrap_or(0.0) <= 0.0)
            .map(|id| scores.get(id).copied().unwrap_or(0.0))
            .sum();

        let base = (1.0 - DAMPING) / n_f + DAMPING * dangling / n_f;
        let mut new_scores: HashMap<String, f32> =
            nodes.iter().map(|id| (id.clone(), base)).collect();

        for (src, targets) in adjacency {
            let out_total = out_totals.get(src).copied().unwrap_or(0.0);
            if out_total <= 0.0 {
                continue;
            }
            let src_score = scores.get(src).copied().unwrap_or(0.0);
            for (tgt, w) in targets {
                *new_scores.entry(tgt.clone()).or_insert(0.0) +=
                    DAMPING * src_score * (w / out_total);
            }
        }

        let delta: f32 = nodes
            .iter()
            .map(|id| {
                (new_scores.get(id).copied().unwrap_or(0.0)
                    - scores.get(id).copied().unwrap_or(0.0))
                .abs()
            })
            .sum();
        scores = new_scores;
        if delta < EPSILON {
            break;
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::pagerank;
    use std::collections::HashMap;

    #[test]
    fn pagerank_conserves_mass_and_ranks_hub_above_leaf() {
        // Undirected star: "hub" connects to a, b, c; "leaf" hangs off a only.
        //   hub—a, hub—b, hub—c, a—leaf
        let nodes: Vec<String> = ["hub", "a", "b", "c", "leaf"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut adj: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        let mut edge = |x: &str, y: &str| {
            adj.entry(x.to_string())
                .or_default()
                .push((y.to_string(), 1.0));
            adj.entry(y.to_string())
                .or_default()
                .push((x.to_string(), 1.0));
        };
        edge("hub", "a");
        edge("hub", "b");
        edge("hub", "c");
        edge("a", "leaf");

        let scores = pagerank(&nodes, &adj);

        let sum: f32 = scores.values().sum();
        assert!((sum - 1.0).abs() < 1e-3, "mass not conserved: sum = {sum}");
        assert!(
            scores["hub"] > scores["leaf"],
            "hub ({}) should outrank leaf ({})",
            scores["hub"],
            scores["leaf"]
        );
    }
}
