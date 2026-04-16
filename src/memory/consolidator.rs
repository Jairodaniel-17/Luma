use crate::memory::service::MemoryService;
use crate::memory::types::{
    EdgeType, MemoryEdge, MemoryKind, MemoryRecord, MemoryStatus, UpsertFactRequest,
};
use uuid::Uuid;

#[derive(Clone, Default)]
pub struct Consolidator;

impl Consolidator {
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
                    target_id: fact_id,
                    edge_type: EdgeType::TriggeredBy,
                    weight: confidence,
                    metadata: serde_json::json!({"auto": true}),
                    created_at_ms: crate::memory::ingest::now_ms(),
                };
                let _ = graph.upsert_edge(&edge).await;
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
