use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use uuid::Uuid;

pub async fn ingest(
    State(state): State<AppState>,
    Path(namespace): Path<String>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let text = payload
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::new(axum::http::StatusCode::BAD_REQUEST, "invalid_request", "Missing 'text' field"))?;

    let doc_id_val = payload.get("id").and_then(|v| v.as_str());
    let doc_id = doc_id_val.unwrap_or_else(|| "");
    let generated_id = if doc_id.is_empty() {
        Uuid::new_v4().to_string()
    } else {
        doc_id.to_string()
    };

    let metadata = payload.get("metadata").cloned();

    state.hub.ingest_document(&namespace, &generated_id, text, payload.clone(), metadata)
        .await
        .map_err(|e| {
            if e.to_string().contains("I/O") || e.to_string().contains("disk") {
                ApiError::new(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "io_failure_during_transaction",
                    format!("Transaction aborted and rolled back. CRITICAL: Check disk health, available space, and I/O permissions. Internal detail: {}", e)
                )
            } else if e.to_string().contains("dimension mismatch") {
                ApiError::new(axum::http::StatusCode::BAD_REQUEST, "dimension_mismatch", e.to_string())
            } else {
                ApiError::new(axum::http::StatusCode::INTERNAL_SERVER_ERROR, "ingest_error", e.to_string())
            }
        })?;

    Ok(Json(serde_json::json!({
        "status": "success",
        "doc_id": generated_id,
        "namespace": namespace
    })))
}

pub async fn search(
    State(state): State<AppState>,
    Path(namespace): Path<String>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let query = payload
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::new(axum::http::StatusCode::BAD_REQUEST, "invalid_request", "Missing 'query' field"))?;

    let sql_filter = payload.get("sql_filter").and_then(|v| v.as_str());
    let limit = payload.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let results = state.hub
        .search(&namespace, query, sql_filter, limit)
        .await
        .map_err(|e| ApiError::new(axum::http::StatusCode::INTERNAL_SERVER_ERROR, "search_error", e.to_string()))?;

    Ok(Json(serde_json::json!({
        "results": results
    })))
}
