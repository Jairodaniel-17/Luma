use crate::api::errors::{ApiError, ErrorBody};
use crate::api::{AppState, TenantContext};
use crate::engine::{EngineError, StateError};
use axum::extract::{Extension, Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

/// Per-tenant keyspace isolation for the global state store.
///
/// A tenant-bound caller's keys are transparently prefixed with `"{tenant}:"`
/// so no two orgs can ever read or overwrite each other's keys. A platform
/// admin (no `tenant_id`) operates on the raw keyspace unmodified — a
/// superuser view that also spans the tenant-prefixed partitions.
fn scope_key(ctx: &TenantContext, key: &str) -> String {
    match &ctx.tenant_id {
        Some(t) => format!("{t}:{key}"),
        None => key.to_string(),
    }
}

/// The `"{tenant}:"` prefix for the caller, or `None` for a platform admin.
fn tenant_prefix(ctx: &TenantContext) -> Option<String> {
    ctx.tenant_id.as_ref().map(|t| format!("{t}:"))
}

/// Strip the tenant prefix from a returned key so callers only ever see their
/// own unprefixed keys.
fn unscope_key(prefix: &Option<String>, key: &mut String) {
    if let Some(p) = prefix {
        if let Some(stripped) = key.strip_prefix(p.as_str()) {
            *key = stripped.to_string();
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_secs: u64,
}

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    axum::Json(HealthResponse {
        status: state.engine.health().to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: 0,
    })
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    // Prometheus only parses a body served as `text/plain; version=0.0.4`;
    // without the header axum falls back to a content type scrapers reject, so
    // the exposition format was already correct but unscrapeable.
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.engine.metrics_text(),
    )
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub prefix: Option<String>,
    pub start: Option<String>,
    pub end: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CreateIndexBody {
    pub field: String,
}

pub async fn list(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
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
    if let Some(start) = &q.start {
        if start.len() > state.config.max_key_len {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "start too long",
            ));
        }
    }
    if let Some(end) = &q.end {
        if end.len() > state.config.max_key_len {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "end too long",
            ));
        }
    }
    if q.prefix.is_some() && (q.start.is_some() || q.end.is_some()) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "prefix cannot be combined with start/end",
        ));
    }
    let limit = q.limit.unwrap_or(100).min(1000);
    let tp = tenant_prefix(&ctx);
    let mut items = match &tp {
        // Platform admin: raw keyspace, unchanged behavior.
        None => {
            if q.prefix.is_some() {
                state.engine.list_state(q.prefix.as_deref(), limit)
            } else {
                state
                    .engine
                    .list_state_range(q.start.as_deref(), q.end.as_deref(), limit)
            }
        }
        // Tenant: confine every scan to the caller's `"{tenant}:"` partition.
        Some(tp) => {
            if q.start.is_some() || q.end.is_some() {
                let start = format!("{tp}{}", q.start.as_deref().unwrap_or(""));
                // Missing upper bound => end of the partition. ':' is 0x3A, so
                // replacing it with ';' (0x3B) is the first key past the prefix.
                let end = q
                    .end
                    .as_deref()
                    .map(|e| format!("{tp}{e}"))
                    .unwrap_or_else(|| format!("{};", &tp[..tp.len() - 1]));
                state
                    .engine
                    .list_state_range(Some(&start), Some(&end), limit)
            } else {
                let prefix = format!("{tp}{}", q.prefix.as_deref().unwrap_or(""));
                state.engine.list_state(Some(&prefix), limit)
            }
        }
    };
    for item in items.iter_mut() {
        unscope_key(&tp, &mut item.key);
    }
    Ok(axum::Json(items))
}

pub async fn get(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(key): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    if key.len() > state.config.max_key_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key too long",
        ));
    }
    let Some(mut item) = state.engine.get_state(&scope_key(&ctx, &key)) else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "key not found",
        ));
    };
    // Return the caller's original (unprefixed) key.
    item.key = key;
    Ok(axum::Json(item))
}

#[derive(Debug, Deserialize)]
pub struct PutBody {
    /// A JSON document, or a raw byte payload written as
    /// `{"__luma_raw": "<base64>", "content_type": "..."}`.
    ///
    /// `StoredVal` decides which by looking for the marker key, so an ordinary
    /// JSON value is unchanged and bytes round-trip symmetrically: a GET
    /// returns the same marker form. `__luma_raw` is reserved at the top level
    /// for exactly this reason.
    pub value: crate::engine::stored::StoredVal,
    pub ttl_ms: Option<u64>,
    pub if_revision: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct PutResponse {
    pub key: String,
    pub revision: u64,
    pub expires_at_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct BatchPutBody {
    pub operations: Vec<PutBodyWithKey>,
}

#[derive(Debug, Deserialize)]
pub struct PutBodyWithKey {
    pub key: String,
    /// Same encoding as [`PutBody::value`].
    pub value: crate::engine::stored::StoredVal,
    pub ttl_ms: Option<u64>,
    pub if_revision: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct BatchPutResponse {
    pub results: Vec<BatchPutResult>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BatchPutResult {
    Ok {
        key: String,
        revision: u64,
        expires_at_ms: Option<u64>,
    },
    Error {
        key: String,
        error: ErrorBody,
    },
}

pub async fn put(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
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
    let estimated = serde_json::to_vec(&body.value)
        .map(|v| v.len())
        .unwrap_or(0);
    if estimated > state.config.max_json_bytes {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "value too large",
        ));
    }
    match state.engine.put_state(
        scope_key(&ctx, &key),
        body.value,
        body.ttl_ms,
        body.if_revision,
    ) {
        Ok(item) => Ok(axum::Json(PutResponse {
            // Echo back the caller's original (unprefixed) key.
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

pub async fn batch_put(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    axum::Json(body): axum::Json<BatchPutBody>,
) -> Result<impl IntoResponse, ApiError> {
    if body.operations.is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "operations required",
        ));
    }
    if body.operations.len() > state.config.max_state_batch {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "too many operations",
        ));
    }

    let mut results = Vec::with_capacity(body.operations.len());
    for op in body.operations {
        if op.key.len() > state.config.max_key_len {
            results.push(BatchPutResult::Error {
                key: op.key,
                error: ErrorBody {
                    error: "invalid_argument",
                    message: "key too long".into(),
                },
            });
            continue;
        }
        let estimated = serde_json::to_vec(&op.value).map(|v| v.len()).unwrap_or(0);
        if estimated > state.config.max_json_bytes {
            results.push(BatchPutResult::Error {
                key: op.key,
                error: ErrorBody {
                    error: "payload_too_large",
                    message: "value too large".into(),
                },
            });
            continue;
        }
        match state.engine.put_state(
            scope_key(&ctx, &op.key),
            op.value,
            op.ttl_ms,
            op.if_revision,
        ) {
            Ok(item) => results.push(BatchPutResult::Ok {
                key: op.key,
                revision: item.revision,
                expires_at_ms: item.expires_at_ms,
            }),
            Err(EngineError::State(StateError::RevisionMismatch)) => {
                results.push(BatchPutResult::Error {
                    key: op.key,
                    error: ErrorBody {
                        error: "revision_mismatch",
                        message: "if_revision mismatch".into(),
                    },
                })
            }
            Err(EngineError::Persistence(_)) => {
                return Err(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "persistence_error",
                    "failed to persist event",
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

    Ok(axum::Json(BatchPutResponse { results }))
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

pub async fn delete(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(key): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    if key.len() > state.config.max_key_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key too long",
        ));
    }
    let deleted = state
        .engine
        .delete_state(&scope_key(&ctx, &key))
        .map_err(|err| match err {
            EngineError::Persistence(_) => ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "persistence_error",
                "failed to persist event",
            ),
            _ => ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "internal error",
            ),
        })?;
    Ok(axum::Json(DeleteResponse { deleted }))
}

pub async fn create_index(
    State(state): State<AppState>,
    axum::Json(body): axum::Json<CreateIndexBody>,
) -> Result<impl IntoResponse, ApiError> {
    if body.field.is_empty() || body.field.len() > state.config.max_key_len {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "invalid field",
        ));
    }
    state.engine.create_state_secondary_index(&body.field);
    Ok(axum::Json(serde_json::json!({
        "status": "ok",
        "field": body.field
    })))
}

pub async fn query_index(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path((field, value)): Path<(String, String)>,
    Query(q): Query<ListQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let limit = q.limit.unwrap_or(100).min(1000);
    let mut items = state
        .engine
        .query_state_secondary_index(&field, &value, limit);
    // Secondary-index hits span the whole keyspace; a tenant may only see hits
    // inside their own partition. ponytail: filtering after the limit can yield
    // fewer than `limit` rows for a tenant — acceptable for a secondary index.
    let tp = tenant_prefix(&ctx);
    if let Some(p) = &tp {
        items.retain(|item| item.key.starts_with(p.as_str()));
        for item in items.iter_mut() {
            unscope_key(&tp, &mut item.key);
        }
    }
    Ok(axum::Json(items))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(tenant: Option<&str>) -> TenantContext {
        TenantContext {
            tenant_id: tenant.map(|s| s.to_string()),
            user_id: None,
            role: "user".to_string(),
            platform_admin: false,
            permissions: serde_json::json!({}),
            quotas: serde_json::json!({}),
        }
    }

    #[test]
    fn key_scoping_is_reversible_and_tenant_bound() {
        let t = ctx(Some("orgA"));
        assert_eq!(scope_key(&t, "foo"), "orgA:foo");
        let p = tenant_prefix(&t);
        let mut k = scope_key(&t, "foo");
        unscope_key(&p, &mut k);
        assert_eq!(k, "foo");

        // Platform admin: no prefixing, no stripping.
        let admin = ctx(None);
        assert_eq!(scope_key(&admin, "foo"), "foo");
        assert!(tenant_prefix(&admin).is_none());

        // A tenant's prefix never matches another tenant's keys.
        let other = tenant_prefix(&ctx(Some("orgB")));
        let mut leaked = "orgA:secret".to_string();
        unscope_key(&other, &mut leaked);
        assert_eq!(
            leaked, "orgA:secret",
            "orgB prefix must not strip orgA keys"
        );
    }
}
