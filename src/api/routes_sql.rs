use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct SqlBody {
    pub sql: String,
    pub params: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct SqlQueryResponse {
    pub rows: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct SqlExecResponse {
    pub rows_affected: u64,
}

pub async fn query(
    State(state): State<AppState>,
    axum::Json(body): axum::Json<SqlBody>,
) -> Result<impl IntoResponse, ApiError> {
    let Some(service) = state.sqlite.as_ref() else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_enabled",
            "sqlite module is disabled",
        ));
    };
    if !body.sql.trim_start().to_lowercase().starts_with("select") {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "sql query endpoint only accepts SELECT statements",
        ));
    }
    let rows = service
        .query(body.sql, body.params.unwrap_or_default())
        .await
        .map_err(|err| ApiError::new(StatusCode::BAD_REQUEST, "sqlite_error", err.to_string()))?;
    Ok(axum::Json(SqlQueryResponse { rows }))
}

pub async fn exec(
    State(state): State<AppState>,
    axum::Json(body): axum::Json<SqlBody>,
) -> Result<impl IntoResponse, ApiError> {
    let Some(service) = state.sqlite.as_ref() else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_enabled",
            "sqlite module is disabled",
        ));
    };
    let affected = service
        .execute(body.sql, body.params.unwrap_or_default())
        .await
        .map_err(|err| ApiError::new(StatusCode::BAD_REQUEST, "sqlite_error", err.to_string()))?;
    Ok(axum::Json(SqlExecResponse {
        rows_affected: affected,
    }))
}
