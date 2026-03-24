use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Episodic,
    Semantic,
    Procedural,
    Working,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Draft,
    Active,
    Archived,
}

impl Default for MemoryStatus {
    fn default() -> Self {
        Self::Active
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryQueryMode {
    Auto,
    Recall,
    Timeline,
    NextStep,
    ConstraintCheck,
}

impl Default for MemoryQueryMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureNodeKind {
    Start,
    Action,
    Decision,
    Goal,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureStatus {
    Draft,
    Active,
    Archived,
}

impl Default for ProcedureStatus {
    fn default() -> Self {
        Self::Active
    }
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
