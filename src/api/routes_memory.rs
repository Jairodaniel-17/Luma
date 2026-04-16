use crate::api::errors::ApiError;
use crate::api::{AppState, TenantContext};
use crate::memory::types::{
    IngestEventRequest, MemoryQueryRequest, NextStepRequest, UpsertFactRequest,
    UpsertProcedureRequest,
};
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::Json;

pub async fn ingest_event(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace): Path<String>,
    Json(payload): Json<IngestEventRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let record = state
        .memory
        .ingest_event(&namespace, payload)
        .await
        .map_err(internal_error("memory_ingest_error"))?;
    Ok(Json(serde_json::to_value(record).unwrap_or(serde_json::Value::Null)))
}

pub async fn upsert_fact(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace): Path<String>,
    Json(payload): Json<UpsertFactRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let record = state
        .memory
        .upsert_fact(&namespace, payload)
        .await
        .map_err(internal_error("memory_fact_error"))?;
    Ok(Json(serde_json::to_value(record).unwrap_or(serde_json::Value::Null)))
}

pub async fn upsert_procedure(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace): Path<String>,
    Json(payload): Json<UpsertProcedureRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let procedure = state
        .memory
        .upsert_procedure(&namespace, payload)
        .await
        .map_err(internal_error("memory_procedure_error"))?;
    Ok(Json(
        serde_json::to_value(procedure).unwrap_or(serde_json::Value::Null),
    ))
}

pub async fn query(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace): Path<String>,
    Json(payload): Json<MemoryQueryRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let response = state
        .memory
        .query(&namespace, payload)
        .await
        .map_err(internal_error("memory_query_error"))?;
    Ok(Json(
        serde_json::to_value(response).unwrap_or(serde_json::Value::Null),
    ))
}

pub async fn next_step(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace): Path<String>,
    Json(payload): Json<NextStepRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let response = state
        .memory
        .next_step(&namespace, payload)
        .await
        .map_err(internal_error("memory_next_step_error"))?;
    Ok(Json(
        serde_json::to_value(response).unwrap_or(serde_json::Value::Null),
    ))
}

pub async fn timeline(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace, entity_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let response = state
        .memory
        .timeline(&namespace, &entity_id)
        .await
        .map_err(internal_error("memory_timeline_error"))?;
    Ok(Json(
        serde_json::to_value(response).unwrap_or(serde_json::Value::Null),
    ))
}

pub async fn upsert_edge(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace): Path<String>,
    Json(payload): Json<crate::memory::types::UpsertEdgeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let response = state
        .memory
        .create_edge(&namespace, payload)
        .await
        .map_err(internal_error("memory_edge_error"))?;
    Ok(Json(serde_json::to_value(response).unwrap_or(serde_json::Value::Null)))
}

pub async fn get_node_edges(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace, memory_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let response = state
        .memory
        .node_edges(&namespace, &memory_id)
        .await
        .map_err(internal_error("memory_edges_error"))?;
    Ok(Json(serde_json::to_value(response).unwrap_or(serde_json::Value::Null)))
}

pub async fn delete_edge(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace, edge_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    state
        .memory
        .remove_edge(&namespace, &edge_id)
        .await
        .map_err(internal_error("memory_edge_delete_error"))?;
    Ok(Json(serde_json::json!({"deleted": true, "edge_id": edge_id})))
}

pub async fn belief_history(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path((namespace, fact_key)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let response = state
        .memory
        .get_belief_history(&namespace, &fact_key)
        .await
        .map_err(internal_error("memory_history_error"))?;
    Ok(Json(serde_json::to_value(response).unwrap_or(serde_json::Value::Null)))
}

pub async fn recompute_centrality(
    State(state): State<AppState>,
    Extension(tenant): Extension<TenantContext>,
    Path(namespace): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let namespace = scope_namespace(&tenant, &namespace);
    let updated = state
        .memory
        .refresh_centrality(&namespace)
        .await
        .map_err(internal_error("memory_centrality_error"))?;
    Ok(Json(serde_json::json!({"updated_nodes": updated, "namespace": namespace})))
}

fn scope_namespace(tenant: &TenantContext, namespace: &str) -> String {
    match tenant.tenant_id.as_deref() {
        Some(tenant_id) => format!("tenant__{}__{}", tenant_id, namespace),
        None => namespace.to_string(),
    }
}

fn internal_error(code: &'static str) -> impl Fn(anyhow::Error) -> ApiError {
    move |error| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, code, error.to_string())
}
