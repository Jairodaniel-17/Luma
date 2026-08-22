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
