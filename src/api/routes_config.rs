use crate::api::errors::ApiError;
use crate::api::AppState;
use crate::config::Config;
use axum::extract::State;
use axum::Json;

pub async fn get_config(State(state): State<AppState>) -> Result<Json<Config>, ApiError> {
    // Return the currently running configuration
    Ok(Json(state.config.clone()))
}

pub async fn update_config(
    State(_state): State<AppState>,
    Json(payload): Json<Config>,
) -> Result<Json<serde_json::Value>, ApiError> {
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
