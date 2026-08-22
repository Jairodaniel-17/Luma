use crate::api::errors::ApiError;
use crate::api::rbac::require_role;
use crate::api::{AppState, TenantContext};
use axum::extract::{Extension, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct AuditQuery {
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    pub key: Option<String>,
    pub limit: Option<usize>,
}

pub async fn backup(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    state
        .engine
        .force_snapshot()
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;
    let offset = state.engine.events().last_published_offset();
    Ok(Json(serde_json::json!({ "ok": true, "offset": offset })))
}

pub async fn get_audit_log(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Query(params): Query<AuditQuery>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let Some(audit_log) = &state.audit_log else {
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "audit log not available (sqlite disabled)",
        ));
    };
    let limit = params.limit.unwrap_or(100).min(1000);
    let entries = audit_log
        .query(params.from_ms, params.to_ms, params.key.as_deref(), limit)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;
    let count = entries.len();
    Ok(Json(
        serde_json::json!({ "entries": entries, "count": count }),
    ))
}

/// RESP activity, broken down by organization.
///
/// F4.2 of `docs/PLAN-MAESTRO.md`: the panel shows connections per org and
/// commands per second.
///
/// Cumulative counters plus the server's own clock, deliberately — **not** a
/// rate computed here. A rate needs a window, and a window chosen by the
/// server is a number whose meaning the caller cannot see: smoothed over what?
/// Since when? The panel polls this twice and divides, so the rate it shows is
/// exactly \"what happened between these two samples\", which is a claim the
/// reader can check.
///
/// The consequence is honest and worth stating: a freshly opened panel shows
/// no rate until its second poll.
pub async fn resp_activity(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let Some(metrics) = &state.resp_metrics else {
        // The listener is off. A 200 with zeroes would read as \"nobody is
        // using it\", which is a different fact.
        return Err(ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "the RESP listener is not running (resp_port is 0)",
        ));
    };

    use std::sync::atomic::Ordering;
    let orgs: Vec<serde_json::Value> = metrics
        .per_org_snapshot()
        .into_iter()
        .map(|(org, counters)| {
            serde_json::json!({
                "org": org,
                "connections_open": counters.connections_open,
                "connections_total": counters.connections_total,
                "commands_total": counters.commands_total,
                "errors_total": counters.errors_total,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        // The server's clock, so the panel divides by a real interval rather
        // than by however long its own timer thought it slept.
        "sampled_at_ms": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        "totals": {
            "connections_open": metrics.connections_open.load(Ordering::Relaxed),
            "connections_total": metrics.connections_total.load(Ordering::Relaxed),
            "commands_total": metrics.commands_total.load(Ordering::Relaxed),
            "errors_total": metrics.errors_total.load(Ordering::Relaxed),
            "auth_failures_total": metrics.auth_failures_total.load(Ordering::Relaxed),
            "rejected_at_limit_total": metrics
                .rejected_at_limit_total
                .load(Ordering::Relaxed),
        },
        "orgs": orgs,
        // Said out loud, because otherwise the first operator to add them up
        // will assume the difference is a bug. A connection is counted
        // globally when it is accepted but has no organization until it
        // authenticates.
        "note": "per-org connections do not sum to the total: a connection is \
                  counted globally on accept and attributed to an org only \
                  once it authenticates",
    })))
}

/// Everything the S3 credential endpoints need.
fn s3_credentials(state: &AppState) -> Result<crate::s3::credentials::S3Credentials, ApiError> {
    let sqlite = state.sqlite.clone().ok_or_else(|| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "S3 credentials need SQLite",
        )
    })?;
    Ok(crate::s3::credentials::S3Credentials::new(
        std::sync::Arc::new(sqlite),
    ))
}

/// The organization these credentials belong to.
///
/// An org-scoped caller gets their own; a platform caller has none of their own
/// and must name one. Never defaulted: a credential with no owner would produce
/// objects that belong to nobody and therefore count against nobody's quota,
/// which is the one thing this whole layer exists to prevent.
fn owning_org(ctx: &TenantContext, requested: Option<&str>) -> Result<String, ApiError> {
    if let Some(org) = ctx.tenant_id.clone() {
        // An org-scoped caller cannot mint for somebody else, whatever they ask
        // for. Ignoring the field silently would be worse than refusing.
        if let Some(requested) = requested {
            if requested != org {
                return Err(ApiError::new(
                    StatusCode::FORBIDDEN,
                    "forbidden",
                    "cannot mint credentials for another organization",
                ));
            }
        }
        return Ok(org);
    }
    requested.map(str::to_string).ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "this caller is not scoped to an organization: pass {\"org_id\": \"…\"}",
        )
    })
}

#[derive(Deserialize, Default)]
pub struct S3CredentialRequest {
    /// Required for a platform caller, optional (and must match) otherwise.
    pub org_id: Option<String>,
}

/// `POST /v1/admin/s3-credentials` — mint an access key for this organization.
///
/// The secret is in the response and **nowhere else**: what is stored is its
/// encrypted form, and there is no endpoint that returns it again. Same contract
/// as every other object store, and the reason is the same — a secret that can be
/// re-read is a secret that leaks twice.
pub async fn create_s3_credential(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    body: Option<Json<S3CredentialRequest>>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let requested = body.and_then(|Json(b)| b.org_id);
    let org = owning_org(&ctx, requested.as_deref())?;
    let minted = s3_credentials(&state)?
        .create(&org)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "access_key_id": minted.access_key_id,
            "secret_access_key": minted.secret_access_key,
            "org_id": minted.org_id,
            "note": "the secret is shown once and is not recoverable; store it now",
        })),
    ))
}

/// `GET /v1/admin/s3-credentials` — list this organization's access keys.
///
/// Ids and dates, never secrets: a listing endpoint that returned them would
/// turn one read-only leak into a full compromise.
pub async fn list_s3_credentials(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Query(query): Query<S3CredentialRequest>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let org = owning_org(&ctx, query.org_id.as_deref())?;
    let listed = s3_credentials(&state)?
        .list(&org)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;

    Ok(Json(serde_json::json!({
        "credentials": listed
            .into_iter()
            .map(|(id, created)| serde_json::json!({
                "access_key_id": id,
                "created_at_ms": created,
            }))
            .collect::<Vec<_>>(),
    })))
}

/// `DELETE /v1/admin/s3-credentials/{access_key_id}` — revoke one.
pub async fn revoke_s3_credential(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    axum::extract::Path(access_key_id): axum::extract::Path<String>,
    Query(query): Query<S3CredentialRequest>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let org = owning_org(&ctx, query.org_id.as_deref())?;
    let removed = s3_credentials(&state)?
        .revoke(&access_key_id, &org)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?;

    if !removed {
        // 404 rather than 204: revoking something that was never there is a
        // fact the caller wants, not a success to be reported.
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "no such S3 credential for this organization",
        ));
    }
    Ok(Json(serde_json::json!({ "revoked": access_key_id })))
}
