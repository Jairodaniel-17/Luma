use crate::memory::service::MemoryService;
use crate::memory::types::{
    EdgeType, MemoryEdge, MemoryKind, MemoryRecord, MemoryStatus, UpsertFactRequest,
};
use crate::vector::{SearchOptions, SearchRequest};
use uuid::Uuid;

const DEDUP_SIMILARITY_THRESHOLD: f32 = 0.95;

#[derive(Clone, Default)]
pub struct Consolidator;

impl Consolidator {
    /// Returns the ID of an existing semantic fact that is semantically equivalent
    /// (cosine > DEDUP_SIMILARITY_THRESHOLD) to `content` in the given namespace.
    async fn find_duplicate_fact(
        &self,
        service: &MemoryService,
        namespace: &str,
        content: &str,
    ) -> Option<String> {
        let vector = service.embeddings.current().embed(content).await.ok()?;
        let collection = service.memory_collection(namespace, MemoryKind::Semantic);
        let exists = service
            .engine
            .list_vector_collections()
            .iter()
            .any(|c| c.collection == collection);
        if !exists {
            return None;
        }
        let hits = service
            .engine
            .vector_search(
                &collection,
                SearchRequest {
                    vector,
                    k: 1,
                    options: SearchOptions {
                        filters: None,
                        filter: None,
                        min_score: Some(DEDUP_SIMILARITY_THRESHOLD),
                        include_meta: true,
                        allowed_ids: None,
                    },
                },
            )
            .ok()?;
        hits.into_iter().next().map(|h| {
            h.meta
                .as_ref()
                .and_then(|m| m.get("memory_id"))
                .and_then(|v| v.as_str())
                .unwrap_or(&h.id)
                .to_string()
        })
    }

    pub async fn process(
        &self,
        service: &MemoryService,
        record: &MemoryRecord,
    ) -> anyhow::Result<()> {
        if !service.config.memory_consolidation_enabled {
            return Ok(());
        }
        if record.kind != MemoryKind::Episodic {
            return Ok(());
        }
        let Some(entity_id) = record.entity_id.clone() else {
            return Ok(());
        };

        let candidates = service
            .llm
            .extract_facts(&record.content, &record.metadata)
            .await?;

        for candidate in candidates {
            // Skip duplicate facts — if a semantically equivalent fact already
            // exists (cosine > 0.95), avoid creating a redundant entry.
            if let Some(dup_id) = self
                .find_duplicate_fact(service, &record.namespace, &candidate.content)
                .await
            {
                let fact_id = format!(
                    "fact::{}::{}::{}",
                    record.namespace, entity_id, candidate.fact_key
                );
                // Only skip if the duplicate is not the same fact we would create
                if dup_id != fact_id {
                    tracing::debug!(
                        namespace = %record.namespace,
                        duplicate_id = %dup_id,
                        "consolidator: skipping duplicate fact (cosine >= {DEDUP_SIMILARITY_THRESHOLD})"
                    );
                    continue;
                }
            }

            let confidence = candidate.confidence; // capture before move
            let status = if confidence >= service.config.memory_fact_promotion_threshold {
                MemoryStatus::Active
            } else {
                MemoryStatus::Draft
            };
            let mut metadata = candidate.metadata;
            metadata["derived_from_memory_id"] = serde_json::Value::String(record.id.clone());
            metadata["derived_from_kind"] = serde_json::Value::String("episodic".to_string());

            let fact_id = format!(
                "fact::{}::{}::{}",
                record.namespace, entity_id, candidate.fact_key
            );
            service
                .upsert_fact(
                    &record.namespace,
                    UpsertFactRequest {
                        id: Some(fact_id.clone()),
                        entity_id: Some(entity_id.clone()),
                        fact_key: Some(candidate.fact_key),
                        content: candidate.content,
                        metadata,
                        source: Some(format!("consolidator:{}", record.id)),
                        confidence: Some(confidence),
                        status: Some(status),
                    },
                )
                .await?;

            // Auto-edge: episodic event → derived semantic fact
            if let Some(graph) = &service.graph {
                let edge = MemoryEdge {
                    id: format!("triggered::{}::{}", record.id, fact_id),
                    namespace: record.namespace.clone(),
                    source_id: record.id.clone(),
                    target_id: fact_id.clone(),
                    edge_type: EdgeType::TriggeredBy,
                    weight: confidence,
                    metadata: serde_json::json!({"auto": true}),
                    created_at_ms: crate::memory::ingest::now_ms(),
                };
                if let Err(e) = graph.upsert_edge(&edge).await {
                    tracing::warn!(
                        "Failed to create TriggeredBy edge {} → {}: {}",
                        record.id,
                        fact_id,
                        e
                    );
                }
            }
        }

        service.emit_consolidation_event(record, &entity_id).await?;
        Ok(())
    }
}

impl MemoryService {
    pub(crate) async fn emit_consolidation_event(
        &self,
        record: &MemoryRecord,
        entity_id: &str,
    ) -> anyhow::Result<()> {
        let key = format!(
            "mem:consolidation:{}:{}:{}",
            record.namespace,
            entity_id,
            Uuid::new_v4()
        );
        self.engine.put_state(
            key,
            serde_json::json!({
                "source_memory_id": record.id,
                "namespace": record.namespace,
                "entity_id": entity_id,
                "event": "episodic_promoted_to_semantic",
                "created_at_ms": crate::memory::ingest::now_ms(),
            }),
            Some(60_000),
            None,
        )?;
        Ok(())
    }
}
