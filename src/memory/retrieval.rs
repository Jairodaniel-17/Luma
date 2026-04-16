use crate::memory::ingest::{parse_memory_kind, parse_memory_status};
use crate::memory::service::MemoryService;
use crate::memory::types::{
    MemoryEvidence, MemoryKind, MemoryQueryMode, MemoryQueryRequest, MemoryQueryResponse,
    MemoryRecord, MemoryResult, SemanticWalkConfig, TimelineResponse,
};
use crate::vector::{SearchOptions, SearchRequest};
use anyhow::Context;
use std::collections::{HashMap, HashSet};

impl MemoryService {
    pub async fn query(
        &self,
        namespace: &str,
        request: MemoryQueryRequest,
    ) -> anyhow::Result<MemoryQueryResponse> {
        self.ensure_schema().await?;
        let mode = self.resolve_query_mode(&request.query, request.mode);
        match mode {
            MemoryQueryMode::Timeline => {
                let entity_id = request
                    .entity_id
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("entity_id is required for timeline mode"))?;
                let timeline = self.timeline(namespace, &entity_id).await?;
                let event_count = timeline.events.len();
                Ok(MemoryQueryResponse {
                    mode,
                    results: timeline
                        .events
                        .into_iter()
                        .map(|record| MemoryResult { record, score: None })
                        .collect(),
                    evidence: None,
                    next_step: None,
                    plan: request.include_plan.unwrap_or(false).then(|| {
                        serde_json::json!({
                            "strategy": "sqlite_timeline",
                            "namespace": namespace,
                        })
                    }),
                    diagnostics: request.include_diagnostics.unwrap_or(false).then(|| {
                        serde_json::json!({
                            "result_count": event_count,
                        })
                    }),
                })
            }
            MemoryQueryMode::NextStep => {
                let procedure_id = request
                    .procedure_id
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("procedure_id is required for next_step mode"))?;
                let next_step = self
                    .next_step(
                        namespace,
                        crate::memory::types::NextStepRequest {
                            procedure_id,
                            current_node_id: request.current_node_id.clone(),
                            context: request.context.clone(),
                        },
                    )
                    .await?;
                Ok(MemoryQueryResponse {
                    mode,
                    results: Vec::new(),
                    evidence: None,
                    next_step: Some(next_step),
                    plan: request.include_plan.unwrap_or(false).then(|| {
                        serde_json::json!({
                            "strategy": "procedural_dag",
                            "namespace": namespace,
                        })
                    }),
                    diagnostics: None,
                })
            }
            _ => self.recall(namespace, request, mode).await,
        }
    }

    pub async fn timeline(
        &self,
        namespace: &str,
        entity_id: &str,
    ) -> anyhow::Result<TimelineResponse> {
        self.ensure_schema().await?;
        let Some(sqlite) = &self.sqlite else {
            anyhow::bail!("sqlite module is required for memory APIs");
        };
        let rows = sqlite
            .query(
                "SELECT id, namespace, entity_id, kind, status, content, metadata, confidence, source,
                    created_at_ms, updated_at_ms, expires_at_ms, embedding_ref
                 FROM memory_records
                 WHERE namespace = ? AND entity_id = ? AND kind = 'episodic'
                 ORDER BY created_at_ms DESC"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(entity_id.to_string()),
                ],
            )
            .await?;
        let events = rows
            .into_iter()
            .map(row_to_memory_record)
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(TimelineResponse {
            entity_id: entity_id.to_string(),
            events,
        })
    }

    async fn recall(
        &self,
        namespace: &str,
        request: MemoryQueryRequest,
        mode: MemoryQueryMode,
    ) -> anyhow::Result<MemoryQueryResponse> {
        let query_vector = self.embeddings.embed(&request.query).await?;
        let limit = self.default_limit(request.limit);

        // ── Step 1: Initial K-NN seeds ────────────────────────────────────
        let seed_k = (limit * 3).min(self.config.memory_walk_max_nodes);
        let allowed_ids = self
            .fetch_allowed_memory_ids(namespace, request.entity_id.as_deref())
            .await?;

        let mut seed_scores: HashMap<String, f32> = HashMap::new();
        let mut seed_ids: Vec<String> = Vec::new();
        let mut evidence_by_id: HashMap<String, MemoryEvidence> = HashMap::new();

        for kind in [MemoryKind::Semantic, MemoryKind::Episodic] {
            let collection = self.memory_collection(namespace, kind);
            let exists = self
                .engine
                .list_vector_collections()
                .iter()
                .any(|c| c.collection == collection);
            if !exists {
                continue;
            }
            let hits = self
                .engine
                .vector_search(
                    &collection,
                    SearchRequest {
                        vector: query_vector.clone(),
                        k: seed_k,
                        options: SearchOptions {
                            filters: None,
                            include_meta: true,
                            allowed_ids: allowed_ids.clone(),
                        },
                    },
                )
                .context("memory vector search")?;

            for hit in hits {
                if seed_scores.contains_key(&hit.id) {
                    continue;
                }
                seed_scores.insert(hit.id.clone(), hit.score);
                seed_ids.push(hit.id.clone());
                if request.include_evidence.unwrap_or(true) {
                    let snippet = hit
                        .meta
                        .as_ref()
                        .and_then(|m| m.get("snippet"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    evidence_by_id.insert(
                        hit.id.clone(),
                        MemoryEvidence {
                            memory_id: hit.id.clone(),
                            kind,
                            score: hit.score,
                            source: String::new(),
                            snippet,
                        },
                    );
                }
            }
        }

        // ── Step 2: Semantic Walk (BFS expansion) ─────────────────────────
        let walk_results = if let Some(graph) = &self.graph {
            let centrality = if self.config.memory_centrality_enabled {
                graph.load_centrality_scores(namespace).await.unwrap_or_default()
            } else {
                HashMap::new()
            };

            // Closure to fetch a vector from the engine by memory_id
            let engine = &self.engine;
            let sem_col = self.memory_collection(namespace, MemoryKind::Semantic);
            let epi_col = self.memory_collection(namespace, MemoryKind::Episodic);
            let get_vector = |id: &str| -> Option<Vec<f32>> {
                engine
                    .vector_get(&sem_col, id)
                    .ok()
                    .flatten()
                    .or_else(|| engine.vector_get(&epi_col, id).ok().flatten())
                    .map(|item| item.vector)
            };

            let walk_config = SemanticWalkConfig {
                max_hops: self.config.memory_walk_max_hops,
                min_similarity: self.config.memory_walk_min_similarity,
                max_nodes: self.config.memory_walk_max_nodes,
            };

            graph
                .semantic_walk(
                    namespace,
                    seed_ids,
                    &seed_scores,
                    &walk_config,
                    &query_vector,
                    &get_vector,
                    &centrality,
                )
                .await
                .unwrap_or_default()
        } else {
            // No graph service: fall back to seed scores as flat list
            seed_ids
                .iter()
                .map(|id| crate::memory::graph::ScoredNode {
                    score: seed_scores.get(id).copied().unwrap_or(0.0),
                    hop: 0,
                    id: id.clone(),
                })
                .collect()
        };

        // ── Step 3: Fetch records, filter, deduplicate ────────────────────
        let mut results: Vec<MemoryResult> = Vec::new();
        let mut evidence: Vec<MemoryEvidence> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for node in walk_results {
            if results.len() >= limit {
                break;
            }
            if seen.contains(&node.id) {
                continue;
            }
            seen.insert(node.id.clone());

            if let Some(record) = self.get_memory_record(namespace, &node.id).await? {
                if !matches!(record.status, crate::memory::types::MemoryStatus::Active) {
                    continue;
                }
                if request.include_evidence.unwrap_or(true) {
                    let mut ev = evidence_by_id.remove(&node.id).unwrap_or_else(|| {
                        MemoryEvidence {
                            memory_id: record.id.clone(),
                            kind: record.kind,
                            score: node.score,
                            source: record.source.clone(),
                            snippet: record.content.chars().take(160).collect(),
                        }
                    });
                    ev.score = node.score;
                    ev.source = record.source.clone();
                    evidence.push(ev);
                }
                results.push(MemoryResult { record, score: Some(node.score) });
            }
        }

        evidence.truncate(self.config.memory_max_evidence.max(1));

        Ok(MemoryQueryResponse {
            mode,
            results,
            evidence: request.include_evidence.unwrap_or(true).then_some(evidence),
            next_step: None,
            plan: request.include_plan.unwrap_or(false).then(|| {
                serde_json::json!({
                    "strategy": "semantic_walk",
                    "namespace": namespace,
                    "seeds": seed_scores.len(),
                    "max_hops": self.config.memory_walk_max_hops,
                    "min_similarity": self.config.memory_walk_min_similarity,
                    "centrality_enabled": self.config.memory_centrality_enabled,
                })
            }),
            diagnostics: request.include_diagnostics.unwrap_or(false).then(|| {
                serde_json::json!({
                    "limit": limit,
                    "entity_filter": request.entity_id,
                    "seed_count": seed_scores.len(),
                })
            }),
        })
    }

    async fn fetch_allowed_memory_ids(
        &self,
        namespace: &str,
        entity_id: Option<&str>,
    ) -> anyhow::Result<Option<HashSet<String>>> {
        let Some(entity_id) = entity_id else {
            return Ok(None);
        };
        let Some(sqlite) = &self.sqlite else {
            return Ok(None);
        };
        let rows = sqlite
            .query(
                "SELECT id FROM memory_records
                 WHERE namespace = ? AND entity_id = ? AND status = 'active'"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(entity_id.to_string()),
                ],
            )
            .await?;
        let ids = rows
            .into_iter()
            .filter_map(|row| row.get("id").and_then(|value| value.as_str()).map(str::to_string))
            .collect::<HashSet<_>>();
        if ids.is_empty() {
            return Ok(Some(HashSet::new()));
        }
        Ok(Some(ids))
    }

    pub(crate) async fn get_memory_record(
        &self,
        namespace: &str,
        memory_id: &str,
    ) -> anyhow::Result<Option<MemoryRecord>> {
        let Some(sqlite) = &self.sqlite else {
            return Ok(None);
        };
        let rows = sqlite
            .query(
                "SELECT id, namespace, entity_id, kind, status, content, metadata, confidence, source,
                    created_at_ms, updated_at_ms, expires_at_ms, embedding_ref
                 FROM memory_records WHERE namespace = ? AND id = ? LIMIT 1"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(memory_id.to_string()),
                ],
            )
            .await?;
        rows.into_iter()
            .next()
            .map(row_to_memory_record)
            .transpose()
    }
}

pub(crate) fn row_to_memory_record(row: serde_json::Value) -> anyhow::Result<MemoryRecord> {
    Ok(MemoryRecord {
        id: row_string(&row, "id")?,
        namespace: row_string(&row, "namespace")?,
        entity_id: row.get("entity_id").and_then(|value| value.as_str()).map(str::to_string),
        kind: parse_memory_kind(row.get("kind").and_then(|value| value.as_str()).unwrap_or("episodic")),
        status: parse_memory_status(row.get("status").and_then(|value| value.as_str()).unwrap_or("active")),
        content: row_string(&row, "content")?,
        metadata: parse_json_string(row.get("metadata")),
        confidence: row.get("confidence").and_then(|value| value.as_f64()).unwrap_or(1.0) as f32,
        source: row_string(&row, "source")?,
        created_at_ms: row.get("created_at_ms").and_then(|value| value.as_u64()).unwrap_or_default(),
        updated_at_ms: row.get("updated_at_ms").and_then(|value| value.as_u64()).unwrap_or_default(),
        expires_at_ms: row.get("expires_at_ms").and_then(|value| value.as_u64()),
        embedding_ref: row.get("embedding_ref").and_then(|value| value.as_str()).map(str::to_string),
    })
}

pub(crate) fn parse_json_string(value: Option<&serde_json::Value>) -> serde_json::Value {
    value
        .and_then(|raw| raw.as_str())
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or(serde_json::Value::Null)
}

fn row_string(row: &serde_json::Value, key: &str) -> anyhow::Result<String> {
    row.get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("missing string field `{key}` in sqlite row"))
}
