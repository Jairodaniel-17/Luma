use crate::api::errors::ApiError;
use crate::api::AppState;
use crate::docstore::{self, DocRecord};
use crate::engine::EngineError;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct DocResponse {
    pub id: String,
    pub revision: u64,
    pub doc: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct FindBody {
    pub filter: Option<serde_json::Value>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct FindResponse {
    pub documents: Vec<DocRecord>,
}

pub async fn put(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, String)>,
    axum::Json(body): axum::Json<serde_json::Value>,
) -> Result<impl IntoResponse, ApiError> {
    validate_collection_and_id(&state, &collection, &id)?;
    enforce_doc_size(&state, &body)?;
    let record =
        docstore::put_doc(&state.engine, &collection, &id, body).map_err(map_engine_error)?;
    Ok(axum::Json(DocResponse {
        id: record.id,
        revision: record.revision,
        doc: record.doc,
    }))
}

pub async fn get(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    validate_collection_and_id(&state, &collection, &id)?;
    let Some(record) =
        docstore::get_doc(&state.engine, &collection, &id).map_err(map_engine_error)?
    else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "document not found",
        ));
    };
    Ok(axum::Json(DocResponse {
        id: record.id,
        revision: record.revision,
        doc: record.doc,
    }))
}

#[derive(Debug, Serialize)]
pub struct DocDeleteResponse {
    pub deleted: bool,
}

pub async fn delete(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    validate_collection_and_id(&state, &collection, &id)?;
    let deleted =
        docstore::delete_doc(&state.engine, &collection, &id).map_err(map_engine_error)?;
    Ok(axum::Json(DocDeleteResponse { deleted }))
}

pub async fn find(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<FindBody>,
) -> Result<impl IntoResponse, ApiError> {
    validate_collection(&state, &collection)?;
    let limit = body.limit.unwrap_or(20).min(state.config.max_doc_find);
    let documents = docstore::find_docs(&state.engine, &collection, body.filter.as_ref(), limit)
        .map_err(map_engine_error)?;
    Ok(axum::Json(FindResponse { documents }))
}

fn validate_collection_and_id(
    state: &AppState,
    collection: &str,
    id: &str,
) -> Result<(), ApiError> {
    validate_collection(state, collection)?;
    if id.len() > state.config.max_id_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "id too long",
        ));
    }
    Ok(())
}

fn validate_collection(state: &AppState, collection: &str) -> Result<(), ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    Ok(())
}

fn enforce_doc_size(state: &AppState, doc: &serde_json::Value) -> Result<(), ApiError> {
    let estimated = serde_json::to_vec(doc).map(|v| v.len()).unwrap_or(0);
    if estimated > state.config.max_json_bytes {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "doc too large",
        ));
    }
    Ok(())
}

fn map_engine_error(err: EngineError) -> ApiError {
    match err {
        EngineError::Persistence(_) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "persistence_error",
            "failed to persist document",
        ),
        _ => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "internal error",
        ),
    }
}
