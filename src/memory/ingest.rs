use crate::memory::service::MemoryService;
use crate::memory::types::{
    EdgeType, IngestEventRequest, MemoryEdge, MemoryKind, MemoryRecord, MemoryStatus,
    UpsertFactRequest,
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
        if let Some(graph) = &self.graph {
            if let Ok(Some(existing)) = self.get_memory_record(namespace, &memory_id).await {
                // Append old version to history
                if let Ok(history_id) = graph.append_belief_history(&existing, now_ms).await {
                    // Create supersedes edge: new → old (target is the history snapshot id)
                    let edge = MemoryEdge {
                        id: format!("supersedes::{memory_id}::{history_id}"),
                        namespace: namespace.to_string(),
                        source_id: memory_id.clone(),
                        target_id: existing.id.clone(),
                        edge_type: EdgeType::Supersedes,
                        weight: 1.0,
                        metadata: serde_json::json!({"reason": "upsert_fact_overwrite"}),
                        created_at_ms: now_ms,
                    };
                    let _ = graph.upsert_edge(&edge).await;
                }
                // Archive the old version in memory_records
                let _ = self.archive_memory_record(namespace, &existing.id).await;
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
        };
        self.persist_memory_record(&record).await?;
        self.index_memory_record(&record).await?;
        Ok(record)
    }

    pub(crate) async fn archive_memory_record(
        &self,
        namespace: &str,
        id: &str,
    ) -> anyhow::Result<()> {
        let Some(sqlite) = &self.sqlite else {
            return Ok(());
        };
        sqlite
            .execute(
                "UPDATE memory_records SET status = 'archived' WHERE namespace = ? AND id = ?"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(id.to_string()),
                ],
            )
            .await
            .map(|_| ())
    }

    pub(crate) async fn persist_memory_record(&self, record: &MemoryRecord) -> anyhow::Result<()> {
        let Some(sqlite) = &self.sqlite else {
            anyhow::bail!("sqlite module is required for memory APIs");
        };
        sqlite
            .execute(
                "INSERT OR REPLACE INTO memory_records (
                    id, namespace, entity_id, kind, status, content, metadata, confidence, source,
                    created_at_ms, updated_at_ms, expires_at_ms, embedding_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
    value.map(serde_json::Value::String).unwrap_or(serde_json::Value::Null)
}

fn opt_u64(value: Option<u64>) -> serde_json::Value {
    value.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null)
}
