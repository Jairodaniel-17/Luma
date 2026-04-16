use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Episodic,
    Semantic,
    Procedural,
    Working,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Draft,
    #[default]
    Active,
    Archived,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryQueryMode {
    #[default]
    Auto,
    Recall,
    Timeline,
    NextStep,
    ConstraintCheck,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureNodeKind {
    Start,
    Action,
    Decision,
    Goal,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureStatus {
    Draft,
    #[default]
    Active,
    Archived,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintOperator {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    Contains,
    In,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleCondition {
    pub field: String,
    pub op: ConstraintOperator,
    pub value: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub namespace: String,
    pub entity_id: Option<String>,
    pub kind: MemoryKind,
    pub status: MemoryStatus,
    pub content: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub confidence: f32,
    pub source: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub expires_at_ms: Option<u64>,
    pub embedding_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryEvidence {
    pub memory_id: String,
    pub kind: MemoryKind,
    pub score: f32,
    pub source: String,
    pub snippet: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryResult {
    pub record: MemoryRecord,
    pub score: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcedureNode {
    pub node_id: String,
    pub kind: ProcedureNodeKind,
    pub label: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcedureEdge {
    pub from_node_id: String,
    pub to_node_id: String,
    #[serde(default)]
    pub priority: i32,
    pub condition: Option<RuleCondition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcedureConstraint {
    pub constraint_id: String,
    pub target_node_id: Option<String>,
    pub condition: RuleCondition,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcedureDefinition {
    pub procedure_id: String,
    pub namespace: String,
    pub name: String,
    pub version: i64,
    pub status: ProcedureStatus,
    pub description: Option<String>,
    pub confidence: f32,
    pub source: String,
    pub nodes: Vec<ProcedureNode>,
    pub edges: Vec<ProcedureEdge>,
    pub constraints: Vec<ProcedureConstraint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IngestEventRequest {
    pub id: Option<String>,
    pub entity_id: Option<String>,
    pub text: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub source: Option<String>,
    pub session_id: Option<String>,
    pub expires_at_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpsertFactRequest {
    pub id: Option<String>,
    pub entity_id: Option<String>,
    pub fact_key: Option<String>,
    pub content: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub source: Option<String>,
    pub confidence: Option<f32>,
    pub status: Option<MemoryStatus>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpsertProcedureRequest {
    pub procedure_id: String,
    pub name: String,
    pub version: Option<i64>,
    pub status: Option<ProcedureStatus>,
    pub description: Option<String>,
    pub confidence: Option<f32>,
    pub source: Option<String>,
    pub nodes: Vec<ProcedureNode>,
    pub edges: Vec<ProcedureEdge>,
    #[serde(default)]
    pub constraints: Vec<ProcedureConstraint>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MemoryQueryRequest {
    pub query: String,
    pub entity_id: Option<String>,
    pub session_id: Option<String>,
    pub procedure_id: Option<String>,
    pub current_node_id: Option<String>,
    #[serde(default)]
    pub context: serde_json::Value,
    pub mode: Option<MemoryQueryMode>,
    pub limit: Option<usize>,
    pub include_evidence: Option<bool>,
    pub include_plan: Option<bool>,
    pub include_diagnostics: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimelineResponse {
    pub entity_id: String,
    pub events: Vec<MemoryRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NextStepRequest {
    pub procedure_id: String,
    pub current_node_id: Option<String>,
    #[serde(default)]
    pub context: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NextStepResponse {
    pub procedure_id: String,
    pub current_node_id: Option<String>,
    pub next_node: Option<ProcedureNode>,
    pub edge: Option<ProcedureEdge>,
    pub blocked_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryQueryResponse {
    pub mode: MemoryQueryMode,
    pub results: Vec<MemoryResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Vec<MemoryEvidence>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_step: Option<NextStepResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<serde_json::Value>,
}

// ─── Graph Layer ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    Supports,
    Contradicts,
    Supersedes,
    TriggeredBy,
    RelatedTo,
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EdgeType::Supports => "supports",
            EdgeType::Contradicts => "contradicts",
            EdgeType::Supersedes => "supersedes",
            EdgeType::TriggeredBy => "triggered_by",
            EdgeType::RelatedTo => "related_to",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for EdgeType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "supports" => Ok(EdgeType::Supports),
            "contradicts" => Ok(EdgeType::Contradicts),
            "supersedes" => Ok(EdgeType::Supersedes),
            "triggered_by" => Ok(EdgeType::TriggeredBy),
            "related_to" => Ok(EdgeType::RelatedTo),
            other => anyhow::bail!("unknown edge type: {other}"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: String,
    pub namespace: String,
    pub source_id: String,
    pub target_id: String,
    pub edge_type: EdgeType,
    pub weight: f32,
    #[serde(default)]
    pub metadata: serde_json::Value,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BeliefVersion {
    pub id: String,
    pub fact_key: String,
    pub namespace: String,
    pub entity_id: Option<String>,
    pub content: String,
    pub confidence: f32,
    pub status: String,
    pub superseded_by: Option<String>,
    pub valid_from: u64,
    pub valid_until: Option<u64>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticWalkConfig {
    pub max_hops: usize,
    pub min_similarity: f32,
    pub max_nodes: usize,
}

impl Default for SemanticWalkConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            min_similarity: 0.65,
            max_nodes: 40,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpsertEdgeRequest {
    pub id: Option<String>,
    pub source_id: String,
    pub target_id: String,
    pub edge_type: EdgeType,
    pub weight: Option<f32>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeEdgesResponse {
    pub memory_id: String,
    pub outgoing: Vec<MemoryEdge>,
    pub incoming: Vec<MemoryEdge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BeliefHistoryResponse {
    pub fact_key: String,
    pub versions: Vec<BeliefVersion>,
}
