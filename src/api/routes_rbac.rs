use crate::api::errors::ApiError;
use crate::api::rbac::require_role;
use crate::api::{AppState, TenantContext};
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

// ---- Roles ----

pub async fn list_roles(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = rbac_svc(&state)?;
    let roles = svc.list_roles().await.map_err(internal)?;
    Ok(Json(serde_json::json!({ "roles": roles })))
}

#[derive(Deserialize)]
pub struct CreateRoleBody {
    pub name: String,
    pub parent_role_id: Option<String>,
    pub description: Option<String>,
}

pub async fn create_role(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Json(body): Json<CreateRoleBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = rbac_svc(&state)?;
    let desc = body.description.unwrap_or_default();
    let id = svc
        .create_role(&body.name, body.parent_role_id.as_deref(), &desc)
        .await
        .map_err(internal)?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "id": id, "name": body.name })),
    ))
}

pub async fn delete_role(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = rbac_svc(&state)?;
    match svc.delete_role(&id).await {
        Ok(true) => Ok(StatusCode::NO_CONTENT),
        Ok(false) => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "role not found",
        )),
        Err(e) if e.to_string().contains("system role") => Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "system_role",
            e.to_string(),
        )),
        Err(e) => Err(internal(e)),
    }
}

// ---- Permissions ----

pub async fn list_permissions(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(role_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = rbac_svc(&state)?;
    let perms = svc.list_permissions(&role_id).await.map_err(internal)?;
    Ok(Json(serde_json::json!({ "role_id": role_id, "permissions": perms })))
}

#[derive(Deserialize)]
pub struct PermBody {
    pub resource: String,
    pub action: String,
}

pub async fn add_permission(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(role_id): Path<String>,
    Json(body): Json<PermBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = rbac_svc(&state)?;
    svc.add_permission(&role_id, &body.resource, &body.action)
        .await
        .map_err(internal)?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn remove_permission(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(role_id): Path<String>,
    Json(body): Json<PermBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = rbac_svc(&state)?;
    let removed = svc
        .remove_permission(&role_id, &body.resource, &body.action)
        .await
        .map_err(internal)?;
    if removed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "permission not found",
        ))
    }
}

// ---- Check (for testing / debugging) ----

#[derive(Deserialize)]
pub struct CheckQuery {
    pub role: String,
    pub resource: String,
    pub action: String,
}

pub async fn check_permission(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    axum::extract::Query(q): axum::extract::Query<CheckQuery>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let svc = rbac_svc(&state)?;
    let allowed = svc.can(&q.role, &q.resource, &q.action).await;
    Ok(Json(
        serde_json::json!({ "role": q.role, "resource": q.resource, "action": q.action, "allowed": allowed }),
    ))
}

// ---- Helpers ----

fn rbac_svc(state: &AppState) -> Result<&crate::api::rbac::RbacService, ApiError> {
    state.rbac.as_deref().ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "RBAC service not available (sqlite disabled)",
        )
    })
}

fn internal(e: anyhow::Error) -> ApiError {
    ApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal",
        e.to_string(),
    )
}
