use crate::memory::ingest::now_ms;
use crate::memory::retrieval::parse_json_string;
use crate::memory::rules::evaluate_condition;
use crate::memory::service::MemoryService;
use crate::memory::types::{
    NextStepRequest, NextStepResponse, ProcedureConstraint, ProcedureDefinition, ProcedureEdge,
    ProcedureNode, ProcedureNodeKind, ProcedureStatus, UpsertProcedureRequest,
};
use anyhow::Context;

impl MemoryService {
    pub async fn upsert_procedure(
        &self,
        namespace: &str,
        request: UpsertProcedureRequest,
    ) -> anyhow::Result<ProcedureDefinition> {
        self.ensure_schema().await?;
        let Some(sqlite) = &self.sqlite else {
            anyhow::bail!("sqlite module is required for memory APIs");
        };
        if request.nodes.len() > self.config.memory_procedural_max_nodes {
            anyhow::bail!("procedure exceeds memory_procedural_max_nodes");
        }
        let now_ms = now_ms();
        let version = request.version.unwrap_or(now_ms as i64);
        let status = request.status.unwrap_or(ProcedureStatus::Active);

        sqlite
            .execute(
                "DELETE FROM procedures WHERE namespace = ? AND procedure_id = ?".to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(request.procedure_id.clone()),
                ],
            )
            .await?;
        sqlite
            .execute(
                "DELETE FROM procedure_nodes WHERE namespace = ? AND procedure_id = ?".to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(request.procedure_id.clone()),
                ],
            )
            .await?;
        sqlite
            .execute(
                "DELETE FROM procedure_edges WHERE namespace = ? AND procedure_id = ?".to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(request.procedure_id.clone()),
                ],
            )
            .await?;
        sqlite
            .execute(
                "DELETE FROM procedure_constraints WHERE namespace = ? AND procedure_id = ?"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(request.procedure_id.clone()),
                ],
            )
            .await?;

        sqlite
            .execute(
                "INSERT INTO procedures (
                    procedure_id, namespace, name, version, status, description, confidence, source, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    .to_string(),
                vec![
                    serde_json::Value::String(request.procedure_id.clone()),
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(request.name.clone()),
                    serde_json::json!(version),
                    serde_json::Value::String(procedure_status_str(status).to_string()),
                    request
                        .description
                        .clone()
                        .map(serde_json::Value::String)
                        .unwrap_or(serde_json::Value::Null),
                    serde_json::json!(request.confidence.unwrap_or(1.0)),
                    serde_json::Value::String(request.source.clone().unwrap_or_else(|| "api".to_string())),
                    serde_json::json!(now_ms),
                    serde_json::json!(now_ms),
                ],
            )
            .await?;

        for node in &request.nodes {
            sqlite
                .execute(
                    "INSERT INTO procedure_nodes (procedure_id, namespace, version, node_id, kind, label, payload)
                     VALUES (?, ?, ?, ?, ?, ?, ?)"
                        .to_string(),
                    vec![
                        serde_json::Value::String(request.procedure_id.clone()),
                        serde_json::Value::String(namespace.to_string()),
                        serde_json::json!(version),
                        serde_json::Value::String(node.node_id.clone()),
                        serde_json::Value::String(node_kind_str(node.kind).to_string()),
                        serde_json::Value::String(node.label.clone()),
                        serde_json::Value::String(node.payload.to_string()),
                    ],
                )
                .await?;
        }

        for edge in &request.edges {
            sqlite
                .execute(
                    "INSERT INTO procedure_edges (procedure_id, namespace, version, from_node_id, to_node_id, priority, condition_json)
                     VALUES (?, ?, ?, ?, ?, ?, ?)"
                        .to_string(),
                    vec![
                        serde_json::Value::String(request.procedure_id.clone()),
                        serde_json::Value::String(namespace.to_string()),
                        serde_json::json!(version),
                        serde_json::Value::String(edge.from_node_id.clone()),
                        serde_json::Value::String(edge.to_node_id.clone()),
                        serde_json::json!(edge.priority),
                        edge.condition
                            .as_ref()
                            .map(|value| serde_json::Value::String(serde_json::to_string(value).unwrap_or_default()))
                            .unwrap_or(serde_json::Value::Null),
                    ],
                )
                .await?;
        }

        for constraint in &request.constraints {
            sqlite
                .execute(
                    "INSERT INTO procedure_constraints (
                        constraint_id, procedure_id, namespace, version, target_node_id, condition_json, message
                     ) VALUES (?, ?, ?, ?, ?, ?, ?)"
                        .to_string(),
                    vec![
                        serde_json::Value::String(constraint.constraint_id.clone()),
                        serde_json::Value::String(request.procedure_id.clone()),
                        serde_json::Value::String(namespace.to_string()),
                        serde_json::json!(version),
                        constraint
                            .target_node_id
                            .clone()
                            .map(serde_json::Value::String)
                            .unwrap_or(serde_json::Value::Null),
                        serde_json::Value::String(serde_json::to_string(&constraint.condition).unwrap_or_default()),
                        constraint
                            .message
                            .clone()
                            .map(serde_json::Value::String)
                            .unwrap_or(serde_json::Value::Null),
                    ],
                )
                .await?;
        }

        Ok(ProcedureDefinition {
            procedure_id: request.procedure_id,
            namespace: namespace.to_string(),
            name: request.name,
            version,
            status,
            description: request.description,
            confidence: request.confidence.unwrap_or(1.0),
            source: request.source.unwrap_or_else(|| "api".to_string()),
            nodes: request.nodes,
            edges: request.edges,
            constraints: request.constraints,
        })
    }

    pub async fn next_step(
        &self,
        namespace: &str,
        request: NextStepRequest,
    ) -> anyhow::Result<NextStepResponse> {
        self.ensure_schema().await?;
        let definition = self
            .load_active_procedure(namespace, &request.procedure_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("active procedure not found"))?;

        let current_node_id = request.current_node_id.clone().or_else(|| {
            definition
                .nodes
                .iter()
                .find(|node| node.kind == ProcedureNodeKind::Start)
                .map(|node| node.node_id.clone())
        });

        let Some(current_id) = current_node_id.clone() else {
            return Ok(NextStepResponse {
                procedure_id: request.procedure_id,
                current_node_id: None,
                next_node: None,
                edge: None,
                blocked_reason: Some("procedure has no start node".to_string()),
            });
        };

        let mut edges = definition
            .edges
            .iter()
            .filter(|edge| edge.from_node_id == current_id)
            .cloned()
            .collect::<Vec<_>>();
        edges.sort_by_key(|e| std::cmp::Reverse(e.priority));

        for edge in edges {
            if let Some(condition) = &edge.condition {
                if !evaluate_condition(condition, &request.context) {
                    continue;
                }
            }
            let Some(next_node) = definition
                .nodes
                .iter()
                .find(|node| node.node_id == edge.to_node_id)
                .cloned()
            else {
                continue;
            };

            let failed = definition.constraints.iter().find(|constraint| {
                constraint
                    .target_node_id
                    .as_deref()
                    .is_none_or(|target| target == next_node.node_id)
                    && !evaluate_condition(&constraint.condition, &request.context)
            });
            if let Some(constraint) = failed {
                return Ok(NextStepResponse {
                    procedure_id: request.procedure_id,
                    current_node_id: current_node_id.clone(),
                    next_node: None,
                    edge: None,
                    blocked_reason: constraint
                        .message
                        .clone()
                        .or_else(|| Some("constraints blocked next step".to_string())),
                });
            }

            return Ok(NextStepResponse {
                procedure_id: request.procedure_id,
                current_node_id: current_node_id.clone(),
                next_node: Some(next_node),
                edge: Some(edge),
                blocked_reason: None,
            });
        }

        Ok(NextStepResponse {
            procedure_id: request.procedure_id,
            current_node_id,
            next_node: None,
            edge: None,
            blocked_reason: Some("no valid outgoing transition".to_string()),
        })
    }

    async fn load_active_procedure(
        &self,
        namespace: &str,
        procedure_id: &str,
    ) -> anyhow::Result<Option<ProcedureDefinition>> {
        let Some(sqlite) = &self.sqlite else {
            return Ok(None);
        };
        let procedure_rows = sqlite
            .query(
                "SELECT procedure_id, namespace, name, version, status, description, confidence, source
                 FROM procedures
                 WHERE namespace = ? AND procedure_id = ? AND status = 'active'
                 ORDER BY version DESC LIMIT 1"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(procedure_id.to_string()),
                ],
            )
            .await?;
        let Some(procedure_row) = procedure_rows.into_iter().next() else {
            return Ok(None);
        };
        let version = procedure_row
            .get("version")
            .and_then(|value| value.as_i64())
            .unwrap_or_default();

        let nodes = sqlite
            .query(
                "SELECT node_id, kind, label, payload FROM procedure_nodes
                 WHERE namespace = ? AND procedure_id = ? AND version = ?"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(procedure_id.to_string()),
                    serde_json::json!(version),
                ],
            )
            .await?
            .into_iter()
            .map(|row| ProcedureNode {
                node_id: row
                    .get("node_id")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
                kind: parse_node_kind(
                    row.get("kind")
                        .and_then(|value| value.as_str())
                        .unwrap_or("action"),
                ),
                label: row
                    .get("label")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
                payload: parse_json_string(row.get("payload")),
            })
            .collect::<Vec<_>>();

        let edges = sqlite
            .query(
                "SELECT from_node_id, to_node_id, priority, condition_json
                 FROM procedure_edges
                 WHERE namespace = ? AND procedure_id = ? AND version = ?
                 ORDER BY priority DESC"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(procedure_id.to_string()),
                    serde_json::json!(version),
                ],
            )
            .await?
            .into_iter()
            .map(|row| {
                Ok(ProcedureEdge {
                    from_node_id: row
                        .get("from_node_id")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    to_node_id: row
                        .get("to_node_id")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    priority: row
                        .get("priority")
                        .and_then(|value| value.as_i64())
                        .unwrap_or_default() as i32,
                    condition: row
                        .get("condition_json")
                        .and_then(|value| value.as_str())
                        .filter(|value| !value.is_empty())
                        .map(serde_json::from_str)
                        .transpose()
                        .context("parse procedure edge condition")?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let constraints = sqlite
            .query(
                "SELECT constraint_id, target_node_id, condition_json, message
                 FROM procedure_constraints
                 WHERE namespace = ? AND procedure_id = ? AND version = ?"
                    .to_string(),
                vec![
                    serde_json::Value::String(namespace.to_string()),
                    serde_json::Value::String(procedure_id.to_string()),
                    serde_json::json!(version),
                ],
            )
            .await?
            .into_iter()
            .map(|row| {
                Ok(ProcedureConstraint {
                    constraint_id: row
                        .get("constraint_id")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    target_node_id: row
                        .get("target_node_id")
                        .and_then(|value| value.as_str())
                        .map(str::to_string),
                    condition: serde_json::from_str(
                        row.get("condition_json")
                            .and_then(|value| value.as_str())
                            .unwrap_or("{}"),
                    )
                    .context("parse procedure constraint")?,
                    message: row
                        .get("message")
                        .and_then(|value| value.as_str())
                        .map(str::to_string),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Some(ProcedureDefinition {
            procedure_id: procedure_row
                .get("procedure_id")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
            namespace: procedure_row
                .get("namespace")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
            name: procedure_row
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
            version,
            status: ProcedureStatus::Active,
            description: procedure_row
                .get("description")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            confidence: procedure_row
                .get("confidence")
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0) as f32,
            source: procedure_row
                .get("source")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
            nodes,
            edges,
            constraints,
        }))
    }
}

fn node_kind_str(kind: ProcedureNodeKind) -> &'static str {
    match kind {
        ProcedureNodeKind::Start => "start",
        ProcedureNodeKind::Action => "action",
        ProcedureNodeKind::Decision => "decision",
        ProcedureNodeKind::Goal => "goal",
    }
}

fn parse_node_kind(value: &str) -> ProcedureNodeKind {
    match value {
        "start" => ProcedureNodeKind::Start,
        "decision" => ProcedureNodeKind::Decision,
        "goal" => ProcedureNodeKind::Goal,
        _ => ProcedureNodeKind::Action,
    }
}

fn procedure_status_str(status: ProcedureStatus) -> &'static str {
    match status {
        ProcedureStatus::Draft => "draft",
        ProcedureStatus::Active => "active",
        ProcedureStatus::Archived => "archived",
    }
}
