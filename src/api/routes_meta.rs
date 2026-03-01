use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::extract::{Path, State};
use axum::Json;
use serde_json::Value;
use std::sync::Arc;

pub async fn execute(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    Json(query): Json<Value>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let meta_engine = crate::engine::meta::MetaEngine::new(
        Arc::new(state.engine.clone()),
        state.sqlite.clone().map(Arc::new),
        state.search_engine.clone(),
    );
    let results = meta_engine.execute(&collection, query).await.map_err(|e| {
        ApiError::new(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            e.to_string(),
        )
    })?;
    Ok(Json(results))
}
