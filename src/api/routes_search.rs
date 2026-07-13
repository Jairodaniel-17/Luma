use crate::api::errors::ApiError;
use crate::api::AppState;
use crate::search::types::{IngestRequest, SearchRequest};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};

pub async fn search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if payload.query.len() > 1024 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "query too long",
        ));
    }
    if payload.top_k == 0 || payload.top_k > state.config.max_k {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "top_k invalid",
        ));
    }
    // Embed the query with the configured provider (or honor an explicit
    // TEST_VEC: vector). Previously this used a random placeholder vector, which
    // made semantic search return meaningless results.
    let query_vector = match crate::search::engine::parse_test_vec(&payload.query) {
        Some(v) => v,
        None => state.embeddings.embed(&payload.query).await.map_err(|err| {
            tracing::error!(%err, "query embedding failed");
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "query embedding failed",
            )
        })?,
    };
    match state
        .search_engine
        .search_with_query_vector(payload, query_vector)
    {
        Ok(res) => Ok((StatusCode::OK, Json(res))),
        Err(err) => {
            tracing::error!(%err, "search failed");
            Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "search failed",
            ))
        }
    }
}

pub async fn ingest(
    State(state): State<AppState>,
    Json(payload): Json<IngestRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if payload.document.content.len() > state.config.max_body_bytes {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "document content too large",
        ));
    }
    match state.search_engine.ingest(payload.document) {
        Ok(_) => Ok(StatusCode::OK),
        Err(err) => {
            tracing::error!(%err, "ingest failed");
            Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "ingest failed",
            ))
        }
    }
}
