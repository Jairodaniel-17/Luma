use crate::api::errors::ApiError;
use crate::api::rbac::require_role;
use crate::api::{AppState, TenantContext};
use crate::config::Config;
use axum::extract::State;
use axum::{Extension, Json};

pub async fn get_config(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<Json<Config>, ApiError> {
    // Instance configuration is admin-only (it exposes provider URLs, tunables,
    // etc.). Secrets are never serialized, so api keys are omitted regardless.
    require_role(&ctx, "admin")?;
    Ok(Json(state.config.clone()))
}

pub async fn update_config(
    State(_state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Json(payload): Json<Config>,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_role(&ctx, "admin")?;
    payload.save().map_err(|e| {
        ApiError::new(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "config_save_error",
            format!("Failed to save configuration to luma.toml: {}", e),
        )
    })?;

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": "Configuration saved to luma.toml. A server restart is required for changes to take effect."
    })))
}
