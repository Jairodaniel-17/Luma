use crate::api::errors::ApiError;
use crate::api::rbac::require_role;
use crate::api::{AppState, TenantContext};
use crate::config::Config;
use crate::engine::embeddings::{EmbeddingClient, EmbeddingProvider};
use axum::extract::State;
use axum::{Extension, Json};
use serde::Deserialize;

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
    State(state): State<AppState>,
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

    // Embedding settings apply immediately: the client is rebuilt and swapped
    // into the shared handle, so the hub, NS-Mem and the vector text-search
    // route all pick it up on their next call. Everything else in the config
    // still needs a restart, and the response says which is which rather than
    // claiming a blanket "restart required" that is no longer true.
    let previous = state.embeddings.current();
    let rebuilt = EmbeddingClient::from_config(&payload, Some(state.engine.metrics()));
    let embedding_changed = previous.provider_name() != rebuilt.provider_name()
        || previous.model_name() != rebuilt.model_name();
    state.embeddings.replace(rebuilt);
    if embedding_changed {
        tracing::info!(
            provider = state.embeddings.current().provider_name(),
            model = state.embeddings.current().model_name(),
            "embedding client reloaded from config without restart"
        );
    }

    Ok(Json(serde_json::json!({
        "status": "success",
        "message": "Configuration saved to luma.toml. Embedding settings are already live; other settings require a server restart.",
        "embedding_reloaded": true,
        "embedding_changed": embedding_changed,
        "embedding_provider": state.embeddings.current().provider_name(),
        "embedding_model": state.embeddings.current().model_name(),
    })))
}

/// A candidate embedding configuration to test before saving. Mirrors the
/// preset the UI offers; `google` and any self-hosted server are OpenAI-wire
/// compatible, so they map onto the OpenAI provider with their own URL.
#[derive(Deserialize)]
pub struct EmbeddingProbeBody {
    pub provider: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub model: String,
    // Azure-only extras (optional).
    #[serde(default)]
    pub azure_api_base: String,
    #[serde(default)]
    pub azure_deployment: String,
    #[serde(default)]
    pub azure_api_version: String,
}

fn provider_from_probe(b: &EmbeddingProbeBody) -> EmbeddingProvider {
    match b.provider.to_lowercase().as_str() {
        // OpenAI and every OpenAI-wire-compatible backend (Google Gemini's
        // OpenAI endpoint, vLLM, LM Studio, a self-hosted server, …).
        "openai" | "google" | "gemini" | "custom" | "openai_compatible" => {
            EmbeddingProvider::OpenAI {
                api_url: b.url.clone(),
                api_key: b.api_key.clone(),
                model: b.model.clone(),
            }
        }
        "ollama" => EmbeddingProvider::Ollama {
            api_url: b.url.clone(),
            model: b.model.clone(),
        },
        "azure" | "azure_openai" | "azure-openai" => EmbeddingProvider::AzureOpenAI {
            api_base: b.azure_api_base.clone(),
            deployment: b.azure_deployment.clone(),
            api_key: b.api_key.clone(),
            api_version: b.azure_api_version.clone(),
        },
        "cohere" => EmbeddingProvider::Cohere {
            api_url: if b.url.is_empty() {
                "https://api.cohere.ai".to_string()
            } else {
                b.url.clone()
            },
            api_key: b.api_key.clone(),
            model: b.model.clone(),
            input_type: "search_document".to_string(),
        },
        "huggingface" | "hf" => EmbeddingProvider::HuggingFace {
            api_url: if b.url.is_empty() {
                "https://api-inference.huggingface.co".to_string()
            } else {
                b.url.clone()
            },
            api_key: b.api_key.clone(),
            model: b.model.clone(),
        },
        _ => EmbeddingProvider::None,
    }
}

/// Test a candidate embedding config by embedding a short probe string, and
/// report the *actual* output dimension. This removes the #1 configuration
/// footgun — hand-typing the wrong dim — by measuring it instead. Always
/// returns 200; `ok:false` carries the provider's error message so the UI can
/// show it inline rather than as an opaque HTTP failure.
pub async fn probe_embedding(
    State(_state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Json(body): Json<EmbeddingProbeBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    require_role(&ctx, "admin")?;
    let provider = provider_from_probe(&body);
    if matches!(provider, EmbeddingProvider::None) {
        return Ok(Json(serde_json::json!({
            "ok": false,
            "error": format!("unknown or unset provider '{}'", body.provider),
        })));
    }
    let client = EmbeddingClient::new(provider);
    match client.embed("Luma dimension probe.").await {
        Ok(v) => Ok(Json(serde_json::json!({
            "ok": true,
            "dim": v.len(),
            "provider": body.provider,
            "model": body.model,
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "ok": false,
            "error": e.to_string(),
        }))),
    }
}
