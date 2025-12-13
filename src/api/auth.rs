use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::extract::State;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;

pub async fn auth_middleware(
    State(_state): State<AppState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, ApiError> {
    Ok(next.run(req).await)
}
