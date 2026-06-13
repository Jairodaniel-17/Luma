use crate::api::errors::ApiError;
use crate::api::rbac::require_role;
use crate::api::{AppState, TenantContext};
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct CreateKeyResponse {
    pub id: String,
    pub key: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateKeyBody {
    pub name: String,
    pub tenant_id: Option<String>,
    pub role: Option<String>,
    pub permissions: Option<serde_json::Value>,
    pub quotas: Option<serde_json::Value>,
}

pub async fn list_keys(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };
    let keys = store.list_keys().await.map_err(|err| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            err.to_string(),
        )
    })?;
    Ok(axum::Json(keys))
}

pub async fn create_key(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    axum::Json(body): axum::Json<CreateKeyBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };

    let plain_key = store.generate_api_key();
    let role = body.role.unwrap_or_else(|| "user".to_string());
    let permissions = body.permissions.unwrap_or(serde_json::json!({}));
    let quotas = body.quotas.unwrap_or(serde_json::json!({
        "storage_bytes": 1_073_741_824u64,
        "qps": 100u64,
        "max_collections": 32u64
    }));

    let id = store
        .create_key(
            &body.name,
            body.tenant_id.as_deref(),
            &role,
            &plain_key,
            permissions,
            quotas,
        )
        .await
        .map_err(|err| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                err.to_string(),
            )
        })?;

    Ok(axum::Json(CreateKeyResponse { id, key: plain_key }))
}

pub async fn revoke_key(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };
    let revoked = store.revoke_key(&id).await.map_err(|err| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            err.to_string(),
        )
    })?;
    if !revoked {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "key id not found",
        ));
    }
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
pub struct UpdateKeyRoleBody {
    pub role: String,
    pub permissions: Option<serde_json::Value>,
}

pub async fn update_key_role(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(id): Path<String>,
    axum::Json(body): axum::Json<UpdateKeyRoleBody>,
) -> Result<impl IntoResponse, ApiError> {
    require_role(&ctx, "admin")?;
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };
    store
        .update_key_role(&id, &body.role, body.permissions)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?
        .then_some(())
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "not_found", "key id not found"))?;
    Ok(StatusCode::NO_CONTENT)
}
