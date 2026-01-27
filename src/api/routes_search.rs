use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use crate::api::AppState;
use crate::search::types::{SearchRequest, IngestRequest};

pub async fn search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> impl IntoResponse {
    match state.search_engine.search(payload) {
        Ok(res) => (StatusCode::OK, Json(res)).into_response(),
        Err(err) => {
            tracing::error!(%err, "search failed");
            (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response()
        }
    }
}

pub async fn ingest(
    State(state): State<AppState>,
    Json(payload): Json<IngestRequest>,
) -> impl IntoResponse {
    match state.search_engine.ingest(payload.document) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(err) => {
            tracing::error!(%err, "ingest failed");
            (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response()
        }
    }
}
