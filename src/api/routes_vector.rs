use crate::api::errors::ApiError;
use crate::api::AppState;
use crate::engine::EngineError;
use crate::vector::{Metric, SearchRequest, VectorError, VectorItem};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct CreateCollectionBody {
    pub dim: usize,
    pub metric: Metric,
}

#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub collection: String,
    pub dim: usize,
    pub metric: Metric,
}

pub async fn create_collection(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<CreateCollectionBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.dim == 0 || body.dim > state.config.max_vector_dim {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "invalid dim",
        ));
    }
    state
        .engine
        .create_vector_collection(&collection, body.dim, body.metric)
        .map_err(map_engine_error)?;
    Ok(axum::Json(CreateCollectionResponse {
        collection,
        dim: body.dim,
        metric: body.metric,
    }))
}

#[derive(Debug, Deserialize)]
pub struct AddBody {
    pub id: String,
    pub vector: Vec<f32>,
    pub meta: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct OkResponse {
    pub ok: bool,
}

pub async fn add(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<AddBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.id.len() > state.config.max_id_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "id too long",
        ));
    }
    if body.vector.len() > state.config.max_vector_dim {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "vector too large",
        ));
    }
    if let Some(meta) = &body.meta {
        let estimated = serde_json::to_vec(meta).map(|v| v.len()).unwrap_or(0);
        if estimated > state.config.max_json_bytes {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "meta too large",
            ));
        }
    }
    state
        .engine
        .vector_add(
            &collection,
            &body.id,
            VectorItem {
                vector: body.vector,
                meta: body.meta.unwrap_or(serde_json::Value::Null),
            },
        )
        .map_err(map_engine_error)?;
    Ok(axum::Json(OkResponse { ok: true }))
}

pub async fn upsert(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<AddBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.id.len() > state.config.max_id_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "id too long",
        ));
    }
    if body.vector.len() > state.config.max_vector_dim {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "vector too large",
        ));
    }
    if let Some(meta) = &body.meta {
        let estimated = serde_json::to_vec(meta).map(|v| v.len()).unwrap_or(0);
        if estimated > state.config.max_json_bytes {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "meta too large",
            ));
        }
    }
    state
        .engine
        .vector_upsert(
            &collection,
            &body.id,
            VectorItem {
                vector: body.vector,
                meta: body.meta.unwrap_or(serde_json::Value::Null),
            },
        )
        .map_err(map_engine_error)?;
    Ok(axum::Json(OkResponse { ok: true }))
}

#[derive(Debug, Deserialize)]
pub struct UpdateBody {
    pub id: String,
    pub vector: Option<Vec<f32>>,
    pub meta: Option<serde_json::Value>,
}

pub async fn update(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<UpdateBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.id.len() > state.config.max_id_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "id too long",
        ));
    }
    if let Some(v) = &body.vector {
        if v.len() > state.config.max_vector_dim {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "vector too large",
            ));
        }
    }
    if let Some(meta) = &body.meta {
        let estimated = serde_json::to_vec(meta).map(|v| v.len()).unwrap_or(0);
        if estimated > state.config.max_json_bytes {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "meta too large",
            ));
        }
    }
    state
        .engine
        .vector_update(&collection, &body.id, body.vector, body.meta)
        .map_err(map_engine_error)?;
    Ok(axum::Json(OkResponse { ok: true }))
}

#[derive(Debug, Deserialize)]
pub struct DeleteBody {
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

pub async fn delete(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<DeleteBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.id.len() > state.config.max_id_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "id too long",
        ));
    }
    state
        .engine
        .vector_delete(&collection, &body.id)
        .map_err(map_engine_error)?;
    Ok(axum::Json(DeleteResponse { deleted: true }))
}

#[derive(Debug, Deserialize)]
pub struct GetQuery {
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct GetResponse {
    pub id: String,
    pub vector: Vec<f32>,
    pub meta: serde_json::Value,
}

pub async fn get(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    Query(q): Query<GetQuery>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if q.id.len() > state.config.max_id_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "id too long",
        ));
    }
    let item = state
        .engine
        .vector_get(&collection, &q.id)
        .map_err(map_vector_error)?;
    let Some(item) = item else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "vector id not found",
        ));
    };
    Ok(axum::Json(GetResponse {
        id: q.id,
        vector: item.vector,
        meta: item.meta,
    }))
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub hits: Vec<crate::vector::SearchHit>,
}

pub async fn search(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<SearchRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.k == 0 || body.k > state.config.max_k {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "invalid k",
        ));
    }
    if body.vector.len() > state.config.max_vector_dim {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "vector too large",
        ));
    }
    if let Some(filters) = &body.filters {
        let estimated = serde_json::to_vec(filters).map(|v| v.len()).unwrap_or(0);
        if estimated > state.config.max_json_bytes {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "filters too large",
            ));
        }
    }
    let hits = state
        .engine
        .vector_search(&collection, body)
        .map_err(map_vector_error)?;
    Ok(axum::Json(SearchResponse { hits }))
}

fn map_vector_error(err: VectorError) -> ApiError {
    match err {
        VectorError::CollectionNotFound => ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "collection or id not found",
        ),
        VectorError::IdNotFound => {
            ApiError::new(StatusCode::NOT_FOUND, "not_found", "id not found")
        }
        VectorError::CollectionExists => ApiError::new(
            StatusCode::CONFLICT,
            "already_exists",
            "collection already exists",
        ),
        VectorError::DimMismatch => ApiError::new(
            StatusCode::BAD_REQUEST,
            "dim_mismatch",
            "vector dimension mismatch",
        ),
        VectorError::IdExists => {
            ApiError::new(StatusCode::CONFLICT, "already_exists", "id already exists")
        }
        VectorError::InvalidManifest | VectorError::Persistence => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "persistence_error",
            "vector persistence error",
        ),
    }
}

fn map_engine_error(err: EngineError) -> ApiError {
    match err {
        EngineError::Persistence(_) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "persistence_error",
            "failed to persist event",
        ),
        EngineError::Vector(v) => map_vector_error(v),
        EngineError::State(_) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "internal error",
        ),
        EngineError::Internal(_) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "internal error",
        ),
    }
}
