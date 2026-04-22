use crate::api::errors::ApiError;
use crate::api::{AppState, TenantContext};
use axum::extract::{Extension, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct AuditQuery {
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    pub key: Option<String>,
    pub limit: Option<usize>,
}

pub async fn backup(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    if ctx.role != "admin" {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "admin role required",
        ));
    }
    state
        .engine
        .force_snapshot()
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;
    let offset = state.engine.events().last_published_offset();
    Ok(Json(serde_json::json!({ "ok": true, "offset": offset })))
}

pub async fn get_audit_log(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Query(params): Query<AuditQuery>,
) -> Result<impl IntoResponse, ApiError> {
    if ctx.role != "admin" {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "admin role required",
        ));
    }
    let Some(audit_log) = &state.audit_log else {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "audit log not available (sqlite disabled)",
        ));
    };
    let limit = params.limit.unwrap_or(100).min(1000);
    let entries = audit_log
        .query(params.from_ms, params.to_ms, params.key.as_deref(), limit)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;
    let count = entries.len();
    Ok(Json(
        serde_json::json!({ "entries": entries, "count": count }),
    ))
}
