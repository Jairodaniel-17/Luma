use crate::memory::service::MemoryService;
use crate::memory::types::{
    BeliefHistoryResponse, MemoryEdge, NodeEdgesResponse, UpsertEdgeRequest,
};
use uuid::Uuid;

impl MemoryService {
    pub async fn create_edge(
        &self,
        namespace: &str,
        request: UpsertEdgeRequest,
    ) -> anyhow::Result<MemoryEdge> {
        self.ensure_schema().await?;
        let Some(graph) = &self.graph else {
            anyhow::bail!("graph service requires sqlite to be enabled");
        };
        let now_ms = crate::memory::ingest::now_ms();
        let edge = MemoryEdge {
            id: request.id.unwrap_or_else(|| {
                format!(
                    "edge::{}::{}::{}",
                    request.source_id,
                    request.target_id,
                    Uuid::new_v4()
                )
            }),
            namespace: namespace.to_string(),
            source_id: request.source_id,
            target_id: request.target_id,
            edge_type: request.edge_type,
            weight: request.weight.unwrap_or(1.0).clamp(0.0, 1.0),
            metadata: request.metadata,
            created_at_ms: now_ms,
        };
        graph.upsert_edge(&edge).await?;
        Ok(edge)
    }

    pub async fn node_edges(
        &self,
        namespace: &str,
        memory_id: &str,
    ) -> anyhow::Result<NodeEdgesResponse> {
        self.ensure_schema().await?;
        let Some(graph) = &self.graph else {
            anyhow::bail!("graph service requires sqlite to be enabled");
        };
        let (outgoing, incoming) = graph.get_node_edges(namespace, memory_id).await?;
        Ok(NodeEdgesResponse {
            memory_id: memory_id.to_string(),
            outgoing,
            incoming,
        })
    }

    pub async fn remove_edge(&self, namespace: &str, edge_id: &str) -> anyhow::Result<()> {
        self.ensure_schema().await?;
        let Some(graph) = &self.graph else {
            anyhow::bail!("graph service requires sqlite to be enabled");
        };
        graph.delete_edge(edge_id, namespace).await
    }

    pub async fn get_belief_history(
        &self,
        namespace: &str,
        fact_key: &str,
    ) -> anyhow::Result<BeliefHistoryResponse> {
        self.ensure_schema().await?;
        let Some(graph) = &self.graph else {
            anyhow::bail!("graph service requires sqlite to be enabled");
        };
        let versions = graph.get_belief_history(namespace, fact_key).await?;
        Ok(BeliefHistoryResponse {
            fact_key: fact_key.to_string(),
            versions,
        })
    }

    pub async fn refresh_centrality(&self, namespace: &str) -> anyhow::Result<usize> {
        self.ensure_schema().await?;
        let Some(graph) = &self.graph else {
            anyhow::bail!("graph service requires sqlite to be enabled");
        };
        graph.update_centrality_scores(namespace).await
    }
}
