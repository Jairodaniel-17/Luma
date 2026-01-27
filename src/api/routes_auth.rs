use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::extract::{Path, State};
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
    pub role: Option<String>,
    pub permissions: Option<serde_json::Value>,
}

pub async fn list_keys(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
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
    axum::Json(body): axum::Json<CreateKeyBody>,
) -> Result<impl IntoResponse, ApiError> {
    let Some(store) = &state.auth_store else {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_enabled",
            "auth store not enabled",
        ));
    };
    
    // TODO: Verify current user is admin (requires pulling context from request extension)
    // For now, assuming only admins have access or we trust the bearer token holder if they have admin role?
    // We haven't implemented context injection in auth middleware yet.
    // Let's assume for this MVP that any valid key can manage keys (or we lock it down later).
    
    let plain_key = store.generate_api_key();
    let role = body.role.unwrap_or_else(|| "user".to_string());
    let permissions = body.permissions.unwrap_or(serde_json::json!({}));
    
    let id = store.create_key(&body.name, &role, &plain_key, permissions).await.map_err(|err| {
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
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
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
        return Err(ApiError::new(StatusCode::NOT_FOUND, "not_found", "key id not found"));
    }
    Ok(StatusCode::NO_CONTENT)
}
