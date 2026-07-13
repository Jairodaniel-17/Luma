use crate::api::audit::AuditKeyId;
use crate::api::errors::ApiError;
use crate::api::{AppState, TenantContext};
use axum::extract::State;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;
use subtle::ConstantTimeEq;

pub async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, ApiError> {
    // Allow public assets
    let path = req.uri().path();
    // Public endpoints: health + password register/login.
    // Static assets and the embedded SPA are served for any non-API GET so
    // client-side routing and deep links work without a token.
    let is_static_get = req.method() == axum::http::Method::GET && !path.starts_with("/v1");
    if is_static_get
        || path == "/v1/health"
        || path == "/v1/auth/register"
        || path == "/v1/auth/login"
        || path.ends_with("openapi.yaml")
    {
        return Ok(next.run(req).await);
    }

    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    let token: String = if let Some(header) = auth_header {
        if let Some(bearer) = header.strip_prefix("Bearer ") {
            bearer.trim().to_string()
        } else {
            return Err(ApiError::new(
                axum::http::StatusCode::UNAUTHORIZED,
                "unauthorized",
                "invalid authorization header format",
            ));
        }
    } else if let Some(h) = req.headers().get("x-api-key").and_then(|h| h.to_str().ok()) {
        h.to_string()
    } else {
        return Err(ApiError::new(
            axum::http::StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing authorization header",
        ));
    };

    // 1. Check AuthStore (DB) — API keys.
    if let Some(store) = &state.auth_store {
        match store.validate_key(&token).await {
            Ok(Some(record)) => {
                let mut req = req;
                req.extensions_mut().insert(AuditKeyId(record.id.clone()));
                req.extensions_mut().insert(TenantContext {
                    tenant_id: record.tenant_id.clone(),
                    user_id: None,
                    role: record.role,
                    permissions: record.permissions,
                    quotas: record.quotas,
                });
                return Ok(next.run(req).await);
            }
            Ok(None) => {
                // Fallthrough
            }
            Err(e) => {
                tracing::error!("Auth DB error: {}", e);
                return Err(ApiError::new(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "internal",
                    "auth error",
                ));
            }
        }
    }

    // 2. Check user session tokens (email+password login).
    if crate::api::accounts::AccountsService::is_session_token(&token) {
        if let Some(accounts) = &state.accounts {
            match accounts.validate_session(&token).await {
                Ok(Some(identity)) => {
                    let mut req = req;
                    req.extensions_mut()
                        .insert(AuditKeyId(identity.user_id.clone()));
                    req.extensions_mut().insert(TenantContext {
                        tenant_id: Some(identity.org_id.clone()),
                        user_id: Some(identity.user_id.clone()),
                        role: identity.role.clone(),
                        permissions: serde_json::json!({}),
                        quotas: serde_json::json!({}),
                    });
                    return Ok(next.run(req).await);
                }
                Ok(None) => {
                    return Err(ApiError::new(
                        axum::http::StatusCode::UNAUTHORIZED,
                        "unauthorized",
                        "invalid or expired session",
                    ));
                }
                Err(e) => {
                    tracing::error!("Session validation error: {}", e);
                    return Err(ApiError::new(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "internal",
                        "auth error",
                    ));
                }
            }
        }
    }

    // 3. Check Static Config
    if state.config.api_key.is_empty() {
        return Err(ApiError::new(
            axum::http::StatusCode::UNAUTHORIZED,
            "unauthorized",
            "server requires an api key but none is configured",
        ));
    }

    if token
        .as_bytes()
        .ct_eq(state.config.api_key.as_bytes())
        .unwrap_u8()
        == 1
    {
        let mut req = req;
        req.extensions_mut()
            .insert(AuditKeyId("static".to_string()));
        req.extensions_mut().insert(TenantContext {
            tenant_id: None,
            user_id: None,
            role: "admin".to_string(),
            permissions: serde_json::json!({"allow":"*"}),
            quotas: serde_json::json!({"storage_bytes":"unlimited","qps":"unlimited"}),
        });
        return Ok(next.run(req).await);
    }

    Err(ApiError::new(
        axum::http::StatusCode::UNAUTHORIZED,
        "unauthorized",
        "invalid api key",
    ))
}
