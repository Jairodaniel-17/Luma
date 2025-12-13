mod auth;
mod errors;
mod routes_docs;
mod routes_events;
mod routes_state;
mod routes_vector;

use crate::config::Config;
use crate::engine::Engine;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::routing::{delete, get, post, put};
use axum::Router;
use std::time::Duration;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
#[derive(Clone)]
pub struct AppState {
    pub engine: Engine,
    pub config: Config,
}

pub fn router(engine: Engine, config: Config) -> Router {
    let state = AppState { engine, config };
    let cors = match &state.config.cors_allowed_origins {
        None => CorsLayer::new()
            .allow_origin(Any)
            .allow_headers(Any)
            .allow_methods(Any),
        Some(list) => {
            let origins: Vec<axum::http::HeaderValue> = list
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok())
                .collect();
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(origins))
                .allow_headers(Any)
                .allow_methods(Any)
        }
    };
    Router::<AppState>::new()
        .merge(routes_docs::routes_docs())
        .route("/v1/health", get(routes_state::health))
        .route("/v1/metrics", get(routes_state::metrics))
        .route("/v1/state", get(routes_state::list))
        .route("/v1/state/:key", get(routes_state::get))
        .route("/v1/state/:key", put(routes_state::put))
        .route("/v1/state/:key", delete(routes_state::delete))
        .route("/v1/events", get(routes_events::events))
        .route("/v1/stream", get(routes_events::stream))
        .route(
            "/v1/vector/:collection",
            post(routes_vector::create_collection),
        )
        .route("/v1/vector/:collection/add", post(routes_vector::add))
        .route("/v1/vector/:collection/upsert", post(routes_vector::upsert))
        .route("/v1/vector/:collection/update", post(routes_vector::update))
        .route("/v1/vector/:collection/delete", post(routes_vector::delete))
        .route("/v1/vector/:collection/get", get(routes_vector::get))
        .route("/v1/vector/:collection/search", post(routes_vector::search))
        .layer(DefaultBodyLimit::max(state.config.max_body_bytes))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(state.config.request_timeout_secs),
        ))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .with_state(state)
}
