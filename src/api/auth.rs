use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::extract::State;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;

pub async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, ApiError> {
    // Allow public assets
    let path = req.uri().path();
    if path == "/" || path == "/index.html" || path.starts_with("/assets/") || path == "/v1/health" || path.starts_with("/docs") || path.ends_with("openapi.yaml") {
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
        // Query param fallback
        if let Some(q) = req.uri().query() {
            let params: Vec<(String, String)> = serde_urlencoded::from_str(q).unwrap_or_default();
            if let Some((_, v)) = params.iter().find(|(k, _)| k == "api_key") {
                v.clone()
            } else {
                 return Err(ApiError::new(
                    axum::http::StatusCode::UNAUTHORIZED,
                    "unauthorized",
                    "missing authorization header",
                ));
            }
        } else {
             return Err(ApiError::new(
                axum::http::StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing authorization header",
            ));
        }
    };

    // 1. Check AuthStore (DB)
    if let Some(store) = &state.auth_store {
        match store.validate_key(&token).await {
            Ok(Some(_record)) => {
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

    // 2. Check Static Config
    if token == state.config.api_key {
        return Ok(next.run(req).await);
    }

    Err(ApiError::new(
        axum::http::StatusCode::UNAUTHORIZED,
        "unauthorized",
        "invalid api key",
    ))
}
