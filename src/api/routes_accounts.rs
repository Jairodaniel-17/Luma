//! HTTP handlers for the enterprise account layer: registration, password
//! login, session lifecycle, and admin management of users/orgs plus dashboard
//! stats. See [`crate::api::accounts`].

use crate::api::accounts::{AccountsService, SessionIdentity};
use crate::api::errors::ApiError;
use crate::api::rbac::{require_platform_admin, require_role, role_strictly_below};
use crate::api::{AppState, TenantContext};
use axum::extract::{Extension, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::json;

fn accounts(state: &AppState) -> Result<&std::sync::Arc<AccountsService>, ApiError> {
    state.accounts.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "account layer unavailable (sqlite disabled)",
        )
    })
}

fn client_ip(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-forwarded-for")
        .or_else(|| headers.get("x-real-ip"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn user_agent(headers: &HeaderMap) -> Option<String> {
    headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn internal(e: impl std::fmt::Display) -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string())
}

// ---- Public: register / login ----

#[derive(Deserialize)]
pub struct RegisterBody {
    pub org_name: String,
    pub email: String,
    pub password: String,
}

pub async fn register(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<RegisterBody>,
) -> Result<impl IntoResponse, ApiError> {
    let svc = accounts(&state)?;
    if body.password.len() < 8 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "weak_password",
            "password must be at least 8 characters",
        ));
    }
    // Generic 409 on any conflict (e.g. duplicate email) — do not echo the raw
    // DB error, which would leak that an account for this email already exists.
    let (org, user) = svc
        .register(&body.org_name, &body.email, &body.password)
        .await
        .map_err(|e| {
            tracing::warn!("register failed: {e}");
            ApiError::new(
                StatusCode::CONFLICT,
                "conflict",
                "registration could not be completed",
            )
        })?;
    svc.record_event(
        Some(&org.id),
        Some(&user.id),
        "org.register",
        Some("org"),
        client_ip(&headers).as_deref(),
        user_agent(&headers).as_deref(),
        Some(&org.name),
    )
    .await;
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "org_id": org.id,
            "user_id": user.id,
            "email": user.email,
            "role": user.role,
        })),
    ))
}

#[derive(Deserialize)]
pub struct LoginBody {
    pub email: String,
    pub password: String,
}

pub async fn login(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LoginBody>,
) -> Result<impl IntoResponse, ApiError> {
    let svc = accounts(&state)?;
    let identity = svc
        .verify_login(&body.email, &body.password)
        .await
        .map_err(internal)?;
    let Some(identity) = identity else {
        svc.record_event(
            None,
            None,
            "auth.login_failed",
            Some("session"),
            client_ip(&headers).as_deref(),
            user_agent(&headers).as_deref(),
            Some(&body.email),
        )
        .await;
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "invalid email or password",
        ));
    };
    let (token, expires) = svc.create_session(&identity).await.map_err(internal)?;
    svc.record_event(
        Some(&identity.org_id),
        Some(&identity.user_id),
        "auth.login",
        Some("session"),
        client_ip(&headers).as_deref(),
        user_agent(&headers).as_deref(),
        None,
    )
    .await;
    Ok(Json(json!({
        "token": token,
        "expires_at_ms": expires,
        "role": identity.role,
        "org_id": identity.org_id,
        "user_id": identity.user_id,
    })))
}

// ---- Authenticated: logout / refresh ----

fn bearer(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(|s| s.trim().to_string())
}

pub async fn logout(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    let svc = accounts(&state)?;
    if let Some(token) = bearer(&headers) {
        let _ = svc.revoke_session(&token).await;
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn refresh(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    let svc = accounts(&state)?;
    let (Some(org_id), Some(user_id)) = (ctx.tenant_id.clone(), ctx.user_id.clone()) else {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "not_a_session",
            "refresh requires a session token",
        ));
    };
    // Rotate: revoke old, issue new.
    if let Some(old) = bearer(&headers) {
        let _ = svc.revoke_session(&old).await;
    }
    let identity = SessionIdentity {
        user_id,
        org_id,
        role: ctx.role.clone(),
    };
    let (token, expires) = svc.create_session(&identity).await.map_err(internal)?;
    Ok(Json(json!({ "token": token, "expires_at_ms": expires })))
}

// ---- Admin: orgs ----

pub async fn list_orgs(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = accounts(&state)?;
    let orgs = svc.list_orgs().await.map_err(internal)?;
    // A tenant-bound admin/owner may only see their own org; platform admins
    // (no tenant) see every org.
    let orgs = match &ctx.tenant_id {
        None => orgs,
        Some(tid) => orgs.into_iter().filter(|o| &o.id == tid).collect(),
    };
    Ok(Json(json!({ "orgs": orgs })))
}

#[derive(Deserialize)]
pub struct CreateOrgBody {
    pub name: String,
}

pub async fn create_org(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    headers: HeaderMap,
    Json(body): Json<CreateOrgBody>,
) -> Result<impl IntoResponse, ApiError> {
    // Creating a brand-new org is a platform-wide operation.
    require_platform_admin(&ctx)?;
    let svc = accounts(&state)?;
    let org = svc.create_org(&body.name).await.map_err(internal)?;
    svc.record_event(
        Some(&org.id),
        ctx.user_id.as_deref(),
        "org.create",
        Some("org"),
        client_ip(&headers).as_deref(),
        user_agent(&headers).as_deref(),
        Some(&org.name),
    )
    .await;
    Ok((StatusCode::CREATED, Json(org)))
}

pub async fn delete_org(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    // No cross-tenant IDOR: a tenant-bound admin may only delete their own org.
    // Return 404 (not 403) so the existence of other orgs is not revealed.
    if let Some(tid) = &ctx.tenant_id {
        if tid != &id {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found",
                "org not found",
            ));
        }
    }
    let svc = accounts(&state)?;
    let ok = svc.delete_org(&id).await.map_err(internal)?;
    if !ok {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "org not found",
        ));
    }
    svc.record_event(
        Some(&id),
        ctx.user_id.as_deref(),
        "org.delete",
        Some("org"),
        client_ip(&headers).as_deref(),
        user_agent(&headers).as_deref(),
        None,
    )
    .await;
    Ok(StatusCode::NO_CONTENT)
}

// ---- Admin: users ----

pub async fn list_users(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = accounts(&state)?;
    // Platform admins (no tenant) see everyone; org admins see their org only.
    let users = svc
        .list_users(ctx.tenant_id.as_deref())
        .await
        .map_err(internal)?;
    Ok(Json(json!({ "users": users })))
}

#[derive(Deserialize)]
pub struct CreateUserBody {
    pub email: String,
    pub password: String,
    pub role: String,
    /// Only honored for platform admins; org admins always create in their org.
    pub org_id: Option<String>,
}

pub async fn create_user(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    headers: HeaderMap,
    Json(body): Json<CreateUserBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = accounts(&state)?;
    if body.password.len() < 8 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "weak_password",
            "password must be at least 8 characters",
        ));
    }
    let is_platform_admin =
        ctx.tenant_id.is_none() && matches!(ctx.role.as_str(), "admin" | "owner");
    // No self/peer escalation: a tenant-bound admin/owner may only create users
    // strictly below their own role. Platform admins may seed any role.
    if !is_platform_admin && !role_strictly_below(&ctx.role, &body.role) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "cannot create a user with a role at or above your own",
        ));
    }
    let org_id = match &ctx.tenant_id {
        Some(org) => org.clone(),
        None => body.org_id.clone().ok_or_else(|| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "missing_org",
                "platform admin must provide org_id",
            )
        })?,
    };
    let user = svc
        .create_user(&org_id, &body.email, &body.password, &body.role)
        .await
        .map_err(|e| ApiError::new(StatusCode::CONFLICT, "conflict", e.to_string()))?;
    svc.record_event(
        Some(&org_id),
        ctx.user_id.as_deref(),
        "user.create",
        Some("user"),
        client_ip(&headers).as_deref(),
        user_agent(&headers).as_deref(),
        Some(&user.email),
    )
    .await;
    Ok((
        StatusCode::CREATED,
        Json(
            json!({ "id": user.id, "email": user.email, "role": user.role, "org_id": user.org_id }),
        ),
    ))
}

#[derive(Deserialize)]
pub struct UpdateUserRoleBody {
    pub role: String,
}

pub async fn update_user_role(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<UpdateUserRoleBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let is_platform_admin =
        ctx.tenant_id.is_none() && matches!(ctx.role.as_str(), "admin" | "owner");
    // No promoting a user to a role at or above the caller's own.
    if !is_platform_admin && !role_strictly_below(&ctx.role, &body.role) {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "cannot assign a role at or above your own",
        ));
    }
    let svc = accounts(&state)?;
    // Scope the update to the caller's own tenant (no cross-tenant IDOR).
    // Platform admins (no tenant) may target any user.
    let ok = svc
        .update_user_role(&id, ctx.tenant_id.as_deref(), &body.role)
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "bad_request", e.to_string()))?;
    if !ok {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "user not found",
        ));
    }
    svc.record_event(
        ctx.tenant_id.as_deref(),
        ctx.user_id.as_deref(),
        "user.update_role",
        Some("user"),
        client_ip(&headers).as_deref(),
        user_agent(&headers).as_deref(),
        Some(&format!("{id} -> {}", body.role)),
    )
    .await;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn delete_user(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = accounts(&state)?;
    // Scope the delete to the caller's own tenant (no cross-tenant IDOR).
    // Platform admins (no tenant) may target any user.
    let ok = svc
        .delete_user(&id, ctx.tenant_id.as_deref())
        .await
        .map_err(internal)?;
    if !ok {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "user not found",
        ));
    }
    svc.record_event(
        ctx.tenant_id.as_deref(),
        ctx.user_id.as_deref(),
        "user.delete",
        Some("user"),
        client_ip(&headers).as_deref(),
        user_agent(&headers).as_deref(),
        Some(&id),
    )
    .await;
    Ok(StatusCode::NO_CONTENT)
}

// ---- Admin: stats + semantic audit ----

pub async fn stats(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = accounts(&state)?;
    let mut stats = svc
        .stats(ctx.tenant_id.as_deref())
        .await
        .map_err(internal)?;
    // The live DB size covers every tenant, so only expose it to platform
    // admins — a tenant-bound caller must not learn whole-cluster storage.
    if ctx.tenant_id.is_none() {
        if let Some(db) = crate::backup::sqlite_db_path(&state.config) {
            if let Ok(meta) = std::fs::metadata(&db) {
                if let Some(obj) = stats.as_object_mut() {
                    obj.insert("storage_bytes".to_string(), json!(meta.len()));
                }
            }
        }
    }
    Ok(Json(stats))
}

#[derive(Deserialize)]
pub struct AuditEventsQuery {
    pub limit: Option<usize>,
}

pub async fn audit_events(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    axum::extract::Query(params): axum::extract::Query<AuditEventsQuery>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = accounts(&state)?;
    let limit = params.limit.unwrap_or(100).min(1000);
    let events = svc
        .query_events(ctx.tenant_id.as_deref(), limit)
        .await
        .map_err(internal)?;
    let count = events.len();
    Ok(Json(json!({ "events": events, "count": count })))
}
