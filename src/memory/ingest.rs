use crate::memory::service::MemoryService;
use crate::memory::types::{
    EdgeType, IngestEventRequest, MemoryKind, MemoryRecord, MemoryStatus, UpsertFactRequest,
};
use crate::vector::{Metric, VectorItem};
use anyhow::Context;
use uuid::Uuid;

impl MemoryService {
    pub async fn ingest_event(
        &self,
        namespace: &str,
        request: IngestEventRequest,
    ) -> anyhow::Result<MemoryRecord> {
        self.ensure_schema().await?;
        let memory_id = request.id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let now_ms = now_ms();
        let record = MemoryRecord {
            id: memory_id.clone(),
            namespace: namespace.to_string(),
            entity_id: request.entity_id.clone(),
            kind: MemoryKind::Episodic,
            status: MemoryStatus::Active,
            content: request.text,
            metadata: request.metadata,
            confidence: 1.0,
            source: request.source.unwrap_or_else(|| "api".to_string()),
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            expires_at_ms: request.expires_at_ms,
            embedding_ref: Some(memory_id.clone()),
            decay_score: 1.0,
        };

        self.persist_memory_record(&record).await?;
        self.index_memory_record(&record).await?;
        self.persist_working_memory(namespace, request.session_id.as_deref(), &record)?;
        self.consolidator.process(self, &record).await?;
        Ok(record)
    }

    pub async fn upsert_fact(
        &self,
        namespace: &str,
        request: UpsertFactRequest,
    ) -> anyhow::Result<MemoryRecord> {
        self.ensure_schema().await?;
        let memory_id = request.id.unwrap_or_else(|| {
            let suffix = request
                .fact_key
                .clone()
                .unwrap_or_else(|| Uuid::new_v4().to_string());
            format!("fact::{namespace}::{suffix}")
        });
        let now_ms = now_ms();
        let mut metadata = request.metadata;
        if let Some(key) = &request.fact_key {
            metadata["fact_key"] = serde_json::Value::String(key.clone());
        }

        // ── Belief versioning: snapshot old record before overwriting ──────
        // A contradiction is when the new fact is about the SAME subject as the
        // old one (high embedding cosine, i.e. we're overwriting the same
        // fact_key) AND the stored value actually changed. Low cosine means the
        // new content is unrelated — that is NOT a contradiction. We label the
        // former `Contradicts` and everything else `Supersedes`.
        if let Some(graph) = &self.graph {
            if let Ok(Some(existing)) = self.get_memory_record(namespace, &memory_id).await {
                let is_contradiction = 'check: {
                    let content_changed = request.content.trim() != existing.content.trim();
                    if !content_changed {
                        break 'check false;
                    }
                    let Ok(new_vec) = self.embeddings.embed(&request.content).await else {
                        break 'check false;
                    };
                    let Ok(old_vec) = self.embeddings.embed(&existing.content).await else {
                        break 'check false;
                    };
                    is_semantic_contradiction(
                        cosine_similarity(&new_vec, &old_vec),
                        content_changed,
                    )
                };

                // Overwriting a fact_key is always a belief *supersession* (the new
                // value replaces the old, temporally). Whether the values actually
                // conflict is an orthogonal property recorded as the `contradiction`
                // metadata flag on that same edge — a low-cosine unrelated update is
                // NOT a contradiction; a same-subject value change is.
                // ponytail: representing contradiction as a flag on the supersedes
                // edge keeps versioning intact; a distinct Contradicts edge_type
                // isn't consumed anywhere, so it would only complicate the graph.
                let edge_type = EdgeType::Supersedes;

                // Snapshot history + belief edge + archive in one transaction so
                // a mid-sequence failure can't corrupt the belief chain.
                match graph
                    .supersede_with_history(
                        namespace,
                        &existing,
                        &memory_id,
                        edge_type,
                        is_contradiction,
                        now_ms,
                    )
                    .await
                {
                    Ok(_) if is_contradiction => tracing::info!(
                        namespace = %namespace,
                        fact_id = %memory_id,
                        "detected belief contradiction (same subject, value changed)"
                    ),
                    Ok(_) => {}
                    Err(e) => tracing::warn!("Failed to version belief: {}", e),
                }
            }
        }

        let record = MemoryRecord {
            id: memory_id.clone(),
            namespace: namespace.to_string(),
            entity_id: request.entity_id,
            kind: MemoryKind::Semantic,
            status: request.status.unwrap_or(MemoryStatus::Active),
            content: request.content,
            metadata,
            confidence: request.confidence.unwrap_or(0.95).clamp(0.0, 1.0),
            source: request.source.unwrap_or_else(|| "api".to_string()),
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            expires_at_ms: None,
            embedding_ref: Some(memory_id.clone()),
            decay_score: 1.0,
        };
        self.persist_memory_record(&record).await?;
        self.index_memory_record(&record).await?;
        Ok(record)
    }

    pub(crate) async fn persist_memory_record(&self, record: &MemoryRecord) -> anyhow::Result<()> {
        let Some(sqlite) = &self.sqlite else {
            anyhow::bail!("sqlite module is required for memory APIs");
        };
        sqlite
            .execute(
                "INSERT OR REPLACE INTO memory_records (
                    id, namespace, entity_id, kind, status, content, metadata, confidence, source,
                    created_at_ms, updated_at_ms, expires_at_ms, embedding_ref, decay_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    .to_string(),
                vec![
                    serde_json::Value::String(record.id.clone()),
                    serde_json::Value::String(record.namespace.clone()),
                    opt_string(record.entity_id.clone()),
                    serde_json::Value::String(memory_kind_str(record.kind).to_string()),
                    serde_json::Value::String(memory_status_str(record.status).to_string()),
                    serde_json::Value::String(record.content.clone()),
                    serde_json::Value::String(record.metadata.to_string()),
                    serde_json::json!(record.confidence),
                    serde_json::Value::String(record.source.clone()),
                    serde_json::json!(record.created_at_ms),
                    serde_json::json!(record.updated_at_ms),
                    opt_u64(record.expires_at_ms),
                    opt_string(record.embedding_ref.clone()),
                    serde_json::json!(record.decay_score),
                ],
            )
            .await
            .map(|_| ())
            .context("persist memory record")
    }

    pub(crate) async fn index_memory_record(&self, record: &MemoryRecord) -> anyhow::Result<()> {
        if !matches!(record.kind, MemoryKind::Episodic | MemoryKind::Semantic) {
            return Ok(());
        }

        let vector = self.embeddings.embed(&record.content).await?;
        self.ensure_memory_collection(&record.namespace, record.kind, vector.len())?;
        let collection = self.memory_collection(&record.namespace, record.kind);
        let item = VectorItem {
            vector,
            meta: serde_json::json!({
                "memory_id": record.id,
                "kind": memory_kind_str(record.kind),
                "entity_id": record.entity_id,
                "source": record.source,
                "snippet": truncate_snippet(&record.content),
            }),
            mmap_offset: None,
        };
        self.engine
            .vector_upsert(&collection, &record.id, item)
            .map(|_| ())
            .context("index memory record")
    }

    fn ensure_memory_collection(
        &self,
        namespace: &str,
        kind: MemoryKind,
        detected_dim: usize,
    ) -> anyhow::Result<()> {
        let collection = self.memory_collection(namespace, kind);
        let exists = self
            .engine
            .list_vector_collections()
            .iter()
            .any(|c| c.collection == collection);
        if !exists {
            self.engine
                .create_vector_collection(&collection, detected_dim, Metric::Cosine)?;
        }
        Ok(())
    }

    fn persist_working_memory(
        &self,
        namespace: &str,
        session_id: Option<&str>,
        record: &MemoryRecord,
    ) -> anyhow::Result<()> {
        let Some(session_id) = session_id else {
            return Ok(());
        };
        let key = format!(
            "mem:working:{}:{}:{}",
            namespace,
            session_id,
            record.entity_id.as_deref().unwrap_or("anon")
        );
        let ttl_ms = self.working_ttl_ms();
        self.engine.put_state(
            key,
            serde_json::json!({
                "memory_id": record.id,
                "content": record.content,
                "entity_id": record.entity_id,
                "kind": memory_kind_str(record.kind),
                "metadata": record.metadata,
            }),
            Some(ttl_ms),
            None,
        )?;
        Ok(())
    }

    pub(crate) fn memory_collection(&self, namespace: &str, kind: MemoryKind) -> String {
        format!("mem__{}__{}", memory_kind_str(kind), namespace)
    }
}

pub(crate) fn memory_kind_str(kind: MemoryKind) -> &'static str {
    match kind {
        MemoryKind::Episodic => "episodic",
        MemoryKind::Semantic => "semantic",
        MemoryKind::Procedural => "procedural",
        MemoryKind::Working => "working",
    }
}

pub(crate) fn memory_status_str(status: MemoryStatus) -> &'static str {
    match status {
        MemoryStatus::Draft => "draft",
        MemoryStatus::Active => "active",
        MemoryStatus::Archived => "archived",
    }
}

pub(crate) fn parse_memory_kind(value: &str) -> MemoryKind {
    match value {
        "semantic" => MemoryKind::Semantic,
        "procedural" => MemoryKind::Procedural,
        "working" => MemoryKind::Working,
        _ => MemoryKind::Episodic,
    }
}

pub(crate) fn parse_memory_status(value: &str) -> MemoryStatus {
    match value {
        "draft" => MemoryStatus::Draft,
        "archived" => MemoryStatus::Archived,
        _ => MemoryStatus::Active,
    }
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn truncate_snippet(content: &str) -> String {
    content.chars().take(160).collect()
}

fn opt_string(value: Option<String>) -> serde_json::Value {
    value
        .map(serde_json::Value::String)
        .unwrap_or(serde_json::Value::Null)
}

fn opt_u64(value: Option<u64>) -> serde_json::Value {
    value
        .map(serde_json::Value::from)
        .unwrap_or(serde_json::Value::Null)
}

// ── Contradiction heuristic ─────────────────────────────────────────────────

/// Cosine at/above this means the two texts are about the SAME subject, so an
/// overwrite of the same `fact_key` is replacing the value of a known fact.
const SAME_SUBJECT_COSINE: f32 = 0.75;

/// A contradiction is a *same-subject* overwrite whose value changed.
///
// ponytail: embedding cosine is only a coarse proxy for "same subject" — it
// can't tell "the sky is blue" from "the sky is not blue", both high-cosine.
// A proper NLI / entailment check would be the real fix, but that's out of
// scope here; high cosine + a changed value is a defensible cheap signal.
fn is_semantic_contradiction(cosine: f32, content_changed: bool) -> bool {
    content_changed && cosine >= SAME_SUBJECT_COSINE
}

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

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, is_semantic_contradiction};

    #[test]
    fn same_subject_changed_value_is_contradiction() {
        // Two facts about the same subject sit close in embedding space.
        let old_vec = [1.0_f32, 0.2, 0.1];
        let new_vec = [0.98_f32, 0.25, 0.12];
        let cosine = cosine_similarity(&new_vec, &old_vec);
        assert!(cosine >= 0.75, "expected high cosine, got {cosine}");
        assert!(is_semantic_contradiction(cosine, true));
    }

    #[test]
    fn unrelated_topic_is_not_contradiction() {
        // Orthogonal vectors => cosine ~0 => unrelated, not a contradiction.
        let old_vec = [1.0_f32, 0.0, 0.0];
        let new_vec = [0.0_f32, 1.0, 0.0];
        let cosine = cosine_similarity(&new_vec, &old_vec);
        assert!(cosine < 0.75, "expected low cosine, got {cosine}");
        assert!(!is_semantic_contradiction(cosine, true));
    }

    #[test]
    fn unchanged_value_is_not_contradiction() {
        // Identical content (cosine ~1.0) but nothing changed => just a rewrite.
        assert!(!is_semantic_contradiction(1.0, false));
    }
}
