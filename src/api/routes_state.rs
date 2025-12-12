use crate::api::errors::ApiError;
use crate::api::AppState;
use crate::engine::{EngineError, StateError};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    state.engine.health()
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    (StatusCode::OK, state.engine.metrics_text())
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub prefix: Option<String>,
    pub limit: Option<usize>,
}

pub async fn list(
    State(state): State<AppState>,
    Query(q): Query<ListQuery>,
) -> Result<impl IntoResponse, ApiError> {
    if let Some(prefix) = &q.prefix {
        if prefix.len() > state.config.max_key_len {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "prefix too long",
            ));
        }
    }
    let limit = q.limit.unwrap_or(100).min(1000);
    let items = state.engine.list_state(q.prefix.as_deref(), limit);
    Ok(axum::Json(items))
}

pub async fn get(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    if key.len() > state.config.max_key_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key too long",
        ));
    }
    let Some(item) = state.engine.get_state(&key) else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "key not found",
        ));
    };
    Ok(axum::Json(item))
}

#[derive(Debug, Deserialize)]
pub struct PutBody {
    pub value: serde_json::Value,
    pub ttl_ms: Option<u64>,
    pub if_revision: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct PutResponse {
    pub key: String,
    pub revision: u64,
    pub expires_at_ms: Option<u64>,
}

pub async fn put(
    State(state): State<AppState>,
    Path(key): Path<String>,
    axum::Json(body): axum::Json<PutBody>,
) -> Result<impl IntoResponse, ApiError> {
    if key.len() > state.config.max_key_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key too long",
        ));
    }
    let estimated = serde_json::to_vec(&body.value).map(|v| v.len()).unwrap_or(0);
    if estimated > state.config.max_json_bytes {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "value too large",
        ));
    }
    match state
        .engine
        .put_state(key.clone(), body.value, body.ttl_ms, body.if_revision)
    {
        Ok(item) => Ok(axum::Json(PutResponse {
            key,
            revision: item.revision,
            expires_at_ms: item.expires_at_ms,
        })),
        Err(EngineError::State(StateError::RevisionMismatch)) => Err(ApiError::new(
            StatusCode::CONFLICT,
            "revision_mismatch",
            "if_revision mismatch",
        )),
        Err(EngineError::Persistence(_)) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "persistence_error",
            "failed to persist event",
        )),
        Err(_) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "internal error",
        )),
    }
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

pub async fn delete(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    if key.len() > state.config.max_key_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key too long",
        ));
    }
    let deleted = state.engine.delete_state(&key).map_err(|err| match err {
        EngineError::Persistence(_) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "persistence_error",
            "failed to persist event",
        ),
        _ => ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", "internal error"),
    })?;
    Ok(axum::Json(DeleteResponse { deleted }))
}
