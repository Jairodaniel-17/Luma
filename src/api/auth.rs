use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use subtle::ConstantTimeEq;

pub async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, ApiError> {
    let path = req.uri().path();
    if path == "/v1/health" || path == "/v1/metrics" {
        return Ok(next.run(req).await);
    }

    let Some(auth) = req.headers().get(header::AUTHORIZATION) else {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing Authorization header",
        ));
    };
    let Ok(auth) = auth.to_str() else {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "invalid Authorization header",
        ));
    };

    let Some(token) = auth.strip_prefix("Bearer ") else {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "invalid Authorization scheme",
        ));
    };

    let a = token.as_bytes();
    let b = state.config.api_key.as_bytes();
    let ok = a.len() == b.len() && a.ct_eq(b).into();
    if !ok {
        return Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "invalid API key",
        ));
    }

    Ok(next.run(req).await)
}
