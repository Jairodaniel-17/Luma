use crate::api::errors::{ApiError, ErrorBody};
use crate::api::AppState;
use crate::engine::EngineError;
use crate::vector::index::{DiskAnnBuildParams, DiskIndexStatus};
use crate::vector::{
    AggregateRequest, AggregationBucket, Metric, ScrollItem, SearchHit, SearchRequest,
    VectorCollectionInfo, VectorError, VectorItem,
};
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
        count,
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

/// Drop a whole collection (index in memory, on-disk data, and ownership row).
/// Tenant isolation (path-based) already guarantees the caller owns it; a
/// platform admin may drop any. Idempotent-ish: 404 if it doesn't exist.
pub async fn delete_collection(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let dropped = state
        .engine
        .drop_vector_collection(&collection)
        .map_err(map_engine_error)?;
    if !dropped {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "collection not found",
        ));
    }
    if let Some(accounts) = &state.accounts {
        let _ = accounts.unregister_collection(&collection).await;
    }
    Ok(axum::Json(
        serde_json::json!({ "dropped": true, "collection": collection }),
    ))
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
                mmap_offset: None,
            },
        )
        .map_err(map_engine_error)?;
    Ok(axum::Json(OkResponse { ok: true }))
}

pub async fn upsert(
    State(state): State<AppState>,
    axum::extract::Extension(ctx): axum::extract::Extension<crate::api::TenantContext>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<AddBody>,
) -> Result<impl IntoResponse, ApiError> {
    // One vector at a time here; the batch route charges for its whole batch.
    crate::api::quotas::guard_vector_write(&state.engine, state.accounts.as_deref(), &ctx, 1)
        .await?;
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
                mmap_offset: None,
            },
        )
        .map_err(map_engine_error)?;
    Ok(axum::Json(OkResponse { ok: true }))
}

pub async fn upsert_batch(
    State(state): State<AppState>,
    axum::extract::Extension(ctx): axum::extract::Extension<crate::api::TenantContext>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<UpsertBatchBody>,
) -> Result<impl IntoResponse, ApiError> {
    // The whole batch is charged before any of it is applied: half-applying a
    // batch leaves the caller unable to tell what landed.
    crate::api::quotas::guard_vector_write(
        &state.engine,
        state.accounts.as_deref(),
        &ctx,
        body.items.len() as u64,
    )
    .await?;
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
    // Per-item validation first (preserving order via slots), then a single
    // batched engine call for everything that passed. Batching amortizes the
    // append_guard, collection lock, run-WAL fsync and compaction/training pass
    // over the whole batch instead of paying them per item.
    let mut results: Vec<Option<VectorBatchResult>> = (0..body.items.len()).map(|_| None).collect();
    let mut batch: Vec<(usize, String)> = Vec::new();
    let mut batch_items: Vec<(String, Vec<f32>, serde_json::Value)> = Vec::new();
    for (idx, op) in body.items.into_iter().enumerate() {
        let AddBody { id, vector, meta } = op;
        if id.len() > state.config.max_id_len {
            results[idx] = Some(VectorBatchResult::Error {
                id,
                error: ErrorBody {
                    error: "invalid_argument",
                    message: "id too long".into(),
                },
            });
            continue;
        }
        if vector.len() > state.config.max_vector_dim {
            results[idx] = Some(VectorBatchResult::Error {
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
                results[idx] = Some(VectorBatchResult::Error {
                    id,
                    error: ErrorBody {
                        error: "payload_too_large",
                        message: "meta too large".into(),
                    },
                });
                continue;
            }
        }
        batch.push((idx, id.clone()));
        batch_items.push((id, vector, meta.unwrap_or(serde_json::Value::Null)));
    }

    if !batch_items.is_empty() {
        match state.engine.vector_upsert_batch(&collection, batch_items) {
            Ok(outcomes) => {
                for ((idx, id), outcome) in batch.into_iter().zip(outcomes) {
                    results[idx] = Some(match outcome {
                        Ok(()) => VectorBatchResult::Upserted { id },
                        Err(VectorError::DimMismatch) => VectorBatchResult::Error {
                            id,
                            error: ErrorBody {
                                error: "dim_mismatch",
                                message: "vector dimension mismatch".into(),
                            },
                        },
                        Err(_) => VectorBatchResult::Error {
                            id,
                            error: ErrorBody {
                                error: "internal",
                                message: "upsert failed".into(),
                            },
                        },
                    });
                }
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

    let results: Vec<VectorBatchResult> = results.into_iter().flatten().collect();
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
    if let Some(filters) = &body.options.filters {
        let estimated = serde_json::to_vec(filters).map(|v| v.len()).unwrap_or(0);
        if estimated > state.config.max_json_bytes {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "filters too large",
            ));
        }
    }
    if let Some(f) = &body.options.filter {
        let estimated = serde_json::to_vec(f).map(|v| v.len()).unwrap_or(0);
        if estimated > state.config.max_json_bytes {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "filter too large",
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
        // The reason string is built by the manifest check and names the
        // recorded vs active model, so it is forwarded verbatim: a caller that
        // hits this needs to know which model to switch back to.
        VectorError::EmbeddingMismatch(reason) => {
            ApiError::new(StatusCode::BAD_REQUEST, "embedding_mismatch", reason)
        }
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
        VectorError::StorageQuotaExceeded => ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "storage_quota_exceeded",
            "collection has reached its maximum vector limit",
        ),
        VectorError::InvalidFilterField => ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "metadata filter field contains invalid characters",
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

// ── Batch search ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SearchBatchBody {
    pub queries: Vec<SearchRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchQueryResult {
    pub hits: Vec<SearchHit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchBatchResponse {
    pub results: Vec<BatchQueryResult>,
}

pub async fn search_batch(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<SearchBatchBody>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_collection_len(&collection, &state)?;
    if body.queries.is_empty() {
        return Ok(axum::Json(SearchBatchResponse { results: vec![] }));
    }
    if body.queries.len() > 100 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "too many queries (max 100)",
        ));
    }
    for q in &body.queries {
        if q.k == 0 || q.k > state.config.max_k {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "invalid k",
            ));
        }
        if q.vector.len() > state.config.max_vector_dim {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "vector too large",
            ));
        }
    }
    let engine = state.engine.clone();
    let coll = collection.clone();
    let raw = tokio::task::spawn_blocking(move || engine.vector_search_batch(&coll, body.queries))
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;

    let results = raw
        .into_iter()
        .map(|r| match r {
            Ok(hits) => BatchQueryResult { hits, error: None },
            Err(e) => BatchQueryResult {
                hits: vec![],
                error: Some(e.to_string()),
            },
        })
        .collect();
    Ok(axum::Json(SearchBatchResponse { results }))
}

// ── Scroll / cursor ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ScrollParams {
    pub cursor: Option<String>,
    pub limit: Option<usize>,
    pub include_vectors: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ScrollResponse {
    pub items: Vec<ScrollItem>,
    pub next_cursor: Option<String>,
    pub count: usize,
}

pub async fn scroll(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    Query(params): Query<ScrollParams>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_collection_len(&collection, &state)?;
    let limit = params.limit.unwrap_or(100).clamp(1, 1000);
    let include_vectors = params.include_vectors.unwrap_or(false);
    let engine = state.engine.clone();
    let coll = collection.clone();
    let cursor = params.cursor.clone();
    let (items, next_cursor) = tokio::task::spawn_blocking(move || {
        engine.vector_scroll(&coll, cursor.as_deref(), limit, include_vectors)
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?
    .map_err(map_vector_error)?;

    let count = items.len();
    Ok(axum::Json(ScrollResponse {
        items,
        next_cursor,
        count,
    }))
}

// ── Rerank ───────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RerankBody {
    /// Pre-computed query vector (optional if `query` is provided).
    pub query_vector: Option<Vec<f32>>,
    /// Text query to embed (used when `query_vector` is not provided).
    pub query: Option<String>,
    /// IDs to rerank. Must exist in the collection.
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    pub id: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
}

pub async fn rerank(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<RerankBody>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_collection_len(&collection, &state)?;
    if body.ids.is_empty() {
        return Ok(axum::Json(RerankResponse { results: vec![] }));
    }

    // Resolve query vector
    let query_vector = if let Some(v) = body.query_vector {
        v
    } else if let Some(text) = body.query {
        state.embeddings.current().embed(&text).await.map_err(|e| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "embedding_error",
                e.to_string(),
            )
        })?
    } else {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "either query or query_vector is required",
        ));
    };

    if query_vector.len() > state.config.max_vector_dim {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "vector too large",
        ));
    }

    // Fetch stored vectors and score them
    let engine = state.engine.clone();
    let coll = collection.clone();
    let ids = body.ids.clone();
    let mut results = tokio::task::spawn_blocking(move || {
        ids.into_iter()
            .filter_map(|id| {
                engine
                    .vector_get(&coll, &id)
                    .ok()
                    .flatten()
                    .map(|item| (id, item.vector))
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;

    // Compute cosine similarity
    let qnorm: f32 = query_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut ranked: Vec<RerankResult> = results
        .iter_mut()
        .map(|(id, vec)| {
            let dot: f32 = query_vector
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| a * b)
                .sum();
            let vnorm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let score = if qnorm > 0.0 && vnorm > 0.0 {
                dot / (qnorm * vnorm)
            } else {
                0.0
            };
            RerankResult {
                id: id.clone(),
                score,
            }
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(axum::Json(RerankResponse { results: ranked }))
}

#[derive(Debug, Deserialize)]
pub struct AggregateBody {
    pub group_by: String,
    pub filter: Option<crate::vector::filter::MetadataFilter>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct AggregateResponse {
    pub buckets: Vec<AggregationBucket>,
}

pub async fn aggregate(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<AggregateBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if body.group_by.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "group_by must not be empty",
        ));
    }
    let req = AggregateRequest {
        group_by: body.group_by,
        filter: body.filter,
        limit: body.limit,
    };
    let buckets = state
        .engine
        .vector_aggregate(&collection, req)
        .map_err(|e| match e {
            VectorError::CollectionNotFound => {
                ApiError::new(StatusCode::NOT_FOUND, "not_found", "collection not found")
            }
            _ => ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
        })?;
    Ok(axum::Json(AggregateResponse { buckets }))
}

#[derive(Debug, Deserialize)]
pub struct ReindexBody {
    /// Collection to write the re-embedded vectors into. Defaults to
    /// `{source}__reindex`.
    pub target: Option<String>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ReindexStartResponse {
    pub job_id: String,
    pub source: String,
    pub target: String,
    /// KV key the job publishes progress to; also readable via
    /// `GET /v1/vector/{collection}/reindex/{job_id}`.
    pub progress_key: String,
}

/// Kick off a re-embedding of `collection` under the currently configured
/// model.
///
/// Returns immediately with a job id: re-embedding is bounded by the provider's
/// throughput, so holding the request open would tie the outcome to one HTTP
/// timeout. Progress is polled via the status route.
///
/// The result lands in a **new** collection. A collection's dimension is fixed
/// in its manifest and a new model usually has a different one, so rewriting in
/// place would mean dropping first — and a provider failure halfway through
/// would then leave nothing to fall back to. The caller verifies the new
/// collection and swaps.
pub async fn reindex(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    axum::Json(body): axum::Json<ReindexBody>,
) -> Result<impl IntoResponse, ApiError> {
    if collection.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection too long",
        ));
    }
    if state.engine.vector_collection_info(&collection).is_none() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "collection not found",
        ));
    }

    let target = body
        .target
        .unwrap_or_else(|| format!("{collection}__reindex"));
    if target == collection {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "target must differ from source: rewriting in place would drop the only copy",
        ));
    }
    if target.len() > state.config.max_collection_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "target collection name too long",
        ));
    }

    let job_id = format!("{}-{}", collection, now_ms());
    let progress_key = crate::engine::hub::reindex_progress_key(&job_id);

    let hub = state.hub.clone();
    let source = collection.clone();
    let job = job_id.clone();
    let target_for_task = target.clone();
    let batch_size = body.batch_size.unwrap_or(64);
    tokio::spawn(async move {
        if let Err(err) = hub
            .reindex_collection(&source, &target_for_task, batch_size, &job)
            .await
        {
            tracing::error!(%err, job = %job, "reindex failed");
            // Publish the failure so a poller sees `failed` instead of a job
            // that stops updating and is indistinguishable from a hang.
            hub.publish_reindex_failure(
                crate::engine::hub::ReindexProgress {
                    job_id: job.clone(),
                    source,
                    target: target_for_task,
                    status: "failed".to_string(),
                    processed: 0,
                    reembedded: 0,
                    skipped_no_text: 0,
                    target_dim: None,
                    error: None,
                    started_at_ms: now_ms(),
                    updated_at_ms: now_ms(),
                },
                err.to_string(),
            );
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        axum::Json(ReindexStartResponse {
            job_id,
            source: collection,
            target,
            progress_key,
        }),
    ))
}

/// Read a reindex job's progress.
pub async fn reindex_status(
    State(state): State<AppState>,
    Path((_collection, job_id)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let key = crate::engine::hub::reindex_progress_key(&job_id);
    match state.engine.get_state(&key) {
        Some(item) => Ok(axum::Json(item.value)),
        None => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "no reindex job with that id (progress is not retained forever)",
        )),
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
