use crate::api::errors::{ApiError, ErrorBody};
use crate::api::AppState;
use crate::engine::EngineError;
use crate::vector::index::{DiskAnnBuildParams, DiskIndexStatus};
use crate::vector::{Metric, SearchRequest, VectorCollectionInfo, VectorError, VectorItem, SearchHit};
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

#[derive(Debug, Serialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<VectorCollectionInfo>,
}

#[derive(Debug, Serialize)]
pub struct VectorCollectionDetailResponse {
    pub collection: String,
    pub dim: Option<usize>,
    pub metric: Option<Metric>,
    pub count: Option<usize>,
    pub created_at_ms: Option<u64>,
    pub updated_at_ms: Option<u64>,
    pub manifest: Option<serde_json::Value>,
    pub notes: Option<String>,
    pub segments: Option<usize>,
    pub deleted: Option<u64>,
}

pub async fn list_collections(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let mut collections = state.engine.list_vector_collections();
    for col in &mut collections {
        if let Some(val) = state.engine.vector_manifest_value(&col.collection) {
            if let Some(ts) = val.get("created_at_ms").and_then(|v| v.as_u64()) {
                col.created_at_ms = Some(ts);
            }
            if let Some(ts) = val.get("updated_at_ms").and_then(|v| v.as_u64()) {
                col.updated_at_ms = Some(ts);
            }
        }
    }
    Ok(axum::Json(ListCollectionsResponse { collections }))
}

pub async fn get_collection_detail(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    let stats = state.engine.vector_collection_info(&collection);
    let manifest = state.engine.vector_manifest_value(&collection);
    if stats.is_none() && manifest.is_none() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "collection not found",
        ));
    }
    let manifest_dim = manifest
        .as_ref()
        .and_then(|v| v.get("dim"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let manifest_metric = manifest
        .as_ref()
        .and_then(|v| v.get("metric"))
        .cloned()
        .and_then(|val| serde_json::from_value::<Metric>(val).ok());
    let created_at_ms = manifest
        .as_ref()
        .and_then(|v| v.get("created_at_ms"))
        .and_then(|v| v.as_u64());
    let updated_at_ms = manifest
        .as_ref()
        .and_then(|v| v.get("updated_at_ms"))
        .and_then(|v| v.as_u64());

    let (dim, metric, count, segments, deleted) = if let Some(info) = stats.as_ref() {
        (
            Some(info.dim),
            Some(info.metric),
            Some(info.live_count),
            info.segments,
            info.deleted_count,
        )
    } else {
        (None, None, None, None, None)
    };

    let mut notes = None;
    if stats.is_none() && manifest.is_some() {
        notes = Some("using manifest fallback".to_string());
    }

    let response = VectorCollectionDetailResponse {
        collection,
        dim: dim.or(manifest_dim),
        metric: metric.or(manifest_metric),
        count: count,
        created_at_ms,
        updated_at_ms,
        manifest,
        notes,
        segments,
        deleted,
    };
    Ok(axum::Json(response))
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

#[derive(Debug, Clone, Deserialize)]
pub struct AddBody {
    pub id: String,
    pub vector: Vec<f32>,
    pub meta: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct OkResponse {
    pub ok: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpsertBatchBody {
    pub items: Vec<AddBody>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteBatchBody {
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct VectorBatchResponse {
    pub results: Vec<VectorBatchResult>,
}

#[derive(Debug, Deserialize)]
pub struct DiskAnnBuildRequest {
    pub max_degree: Option<usize>,
    pub build_threads: Option<usize>,
    pub search_list_size: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct DiskAnnStatusResponse {
    pub available: bool,
    pub last_built_ms: u64,
    pub graph_files: Vec<String>,
    pub params: DiskAnnBuildParams,
}

#[derive(Debug, Serialize)]
pub struct DiskAnnMutationResponse {
    pub ok: bool,
    pub params: DiskAnnBuildParams,
    pub status: DiskAnnStatusResponse,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum VectorBatchResult {
    Upserted { id: String },
    Deleted { id: String },
    Error { id: String, error: ErrorBody },
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

pub async fn upsert_batch(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<UpsertBatchBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.items.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "items required",
        ));
    }
    if body.items.len() > state.config.max_vector_batch {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "too many items",
        ));
    }
    let mut results = Vec::with_capacity(body.items.len());
    for op in body.items {
        let AddBody { id, vector, meta } = op;
        if id.len() > state.config.max_id_len {
            results.push(VectorBatchResult::Error {
                id,
                error: ErrorBody {
                    error: "invalid_argument",
                    message: "id too long".into(),
                },
            });
            continue;
        }
        if vector.len() > state.config.max_vector_dim {
            results.push(VectorBatchResult::Error {
                id,
                error: ErrorBody {
                    error: "invalid_argument",
                    message: "vector too large".into(),
                },
            });
            continue;
        }
        if let Some(meta) = &meta {
            let estimated = serde_json::to_vec(meta).map(|v| v.len()).unwrap_or(0);
            if estimated > state.config.max_json_bytes {
                results.push(VectorBatchResult::Error {
                    id,
                    error: ErrorBody {
                        error: "payload_too_large",
                        message: "meta too large".into(),
                    },
                });
                continue;
            }
        }
        match state.engine.vector_upsert(
            &collection,
            &id,
            VectorItem {
                vector,
                meta: meta.unwrap_or(serde_json::Value::Null),
            },
        ) {
            Ok(_) => results.push(VectorBatchResult::Upserted { id }),
            Err(EngineError::Vector(VectorError::DimMismatch)) => {
                results.push(VectorBatchResult::Error {
                    id,
                    error: ErrorBody {
                        error: "dim_mismatch",
                        message: "vector dimension mismatch".into(),
                    },
                });
            }
            Err(EngineError::Vector(VectorError::CollectionNotFound)) => {
                return Err(map_vector_error(VectorError::CollectionNotFound));
            }
            Err(EngineError::Vector(VectorError::InvalidManifest)) => {
                return Err(map_vector_error(VectorError::InvalidManifest));
            }
            Err(EngineError::Vector(VectorError::Persistence)) => {
                return Err(map_vector_error(VectorError::Persistence));
            }
            Err(EngineError::Vector(VectorError::IdExists)) => {
                results.push(VectorBatchResult::Error {
                    id,
                    error: ErrorBody {
                        error: "already_exists",
                        message: "id already exists".into(),
                    },
                });
            }
            Err(EngineError::Persistence(_)) => {
                return Err(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "persistence_error",
                    "failed to persist vector",
                ));
            }
            Err(EngineError::Internal(_)) | Err(EngineError::State(_)) => {
                return Err(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal",
                    "internal error",
                ));
            }
            Err(EngineError::Vector(other)) => {
                return Err(map_vector_error(other));
            }
        }
    }
    Ok(axum::Json(VectorBatchResponse { results }))
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

#[derive(Debug, Clone, Deserialize)]
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

pub async fn delete_batch(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<DeleteBatchBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.ids.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "ids required",
        ));
    }
    if body.ids.len() > state.config.max_vector_batch {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "too many ids",
        ));
    }
    let mut results = Vec::with_capacity(body.ids.len());
    for id in body.ids {
        if id.len() > state.config.max_id_len {
            results.push(VectorBatchResult::Error {
                id,
                error: ErrorBody {
                    error: "invalid_argument",
                    message: "id too long".into(),
                },
            });
            continue;
        }
        match state.engine.vector_delete(&collection, &id) {
            Ok(_) => results.push(VectorBatchResult::Deleted { id }),
            Err(EngineError::Vector(VectorError::IdNotFound)) => {
                results.push(VectorBatchResult::Error {
                    id,
                    error: ErrorBody {
                        error: "not_found",
                        message: "id not found".into(),
                    },
                });
            }
            Err(EngineError::Vector(VectorError::CollectionNotFound)) => {
                return Err(map_vector_error(VectorError::CollectionNotFound));
            }
            Err(EngineError::Vector(VectorError::InvalidManifest)) => {
                return Err(map_vector_error(VectorError::InvalidManifest));
            }
            Err(EngineError::Vector(VectorError::Persistence)) => {
                return Err(map_vector_error(VectorError::Persistence));
            }
            Err(EngineError::Persistence(_)) => {
                return Err(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "persistence_error",
                    "failed to persist vector",
                ));
            }
            Err(_) => {
                return Err(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal",
                    "internal error",
                ));
            }
        }
    }
    Ok(axum::Json(VectorBatchResponse { results }))
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
    pub hits: Vec<SearchHit>,
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
        VectorError::UnsupportedOperation => ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_supported",
            "vector operation not supported",
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
pub async fn diskann_build(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<DiskAnnBuildRequest>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_collection_len(&collection, &state)?;
    let params = diskann_params_from_request(&state, &body);
    state
        .engine
        .vector_build_disk_index(&collection, params.clone())
        .map_err(map_engine_error)?;
    let status = state
        .engine
        .vector_disk_index_status(&collection)
        .map_err(map_engine_error)?;
    Ok(axum::Json(DiskAnnMutationResponse {
        ok: true,
        params,
        status: status.into(),
    }))
}

pub async fn diskann_tune(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<DiskAnnBuildRequest>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_collection_len(&collection, &state)?;
    let params = diskann_params_from_request(&state, &body);
    let applied = state
        .engine
        .vector_update_disk_index_params(&collection, params.clone())
        .map_err(map_engine_error)?;
    let status = state
        .engine
        .vector_disk_index_status(&collection)
        .map_err(map_engine_error)?;
    Ok(axum::Json(DiskAnnMutationResponse {
        ok: true,
        params: applied,
        status: status.into(),
    }))
}

pub async fn diskann_status(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_collection_len(&collection, &state)?;
    let status = state
        .engine
        .vector_disk_index_status(&collection)
        .map_err(map_engine_error)?;
    Ok(axum::Json(DiskAnnStatusResponse::from(status)))
}

fn ensure_collection_len(collection: &str, state: &AppState) -> Result<(), ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    Ok(())
}

fn diskann_params_from_request(state: &AppState, body: &DiskAnnBuildRequest) -> DiskAnnBuildParams {
    DiskAnnBuildParams {
        max_degree: body
            .max_degree
            .unwrap_or(state.config.diskann_max_degree)
            .max(4),
        build_threads: body
            .build_threads
            .unwrap_or(state.config.diskann_build_threads)
            .max(1),
        search_list_size: body
            .search_list_size
            .unwrap_or(state.config.diskann_search_list_size)
            .max(8),
    }
    .sanitized()
}

impl From<DiskIndexStatus> for DiskAnnStatusResponse {
    fn from(value: DiskIndexStatus) -> Self {
        Self {
            available: value.available,
            last_built_ms: value.last_built_ms,
            graph_files: value.graph_files,
            params: value.params,
        }
    }
}
