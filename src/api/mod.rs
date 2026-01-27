pub mod auth;
pub mod errors;
pub mod routes_doc;
pub mod routes_docs;
pub mod routes_events;
pub mod routes_search;
pub mod routes_sql;
pub mod routes_state;
pub mod routes_ui;
pub mod routes_vector;

use crate::config::Config;
use crate::engine::Engine;
use crate::search::engine::SearchEngine;
use crate::sqlite::SqliteService;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::routing::{delete, get, post, put};
use axum::Router;
use std::sync::Arc;
use std::time::Duration;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
#[derive(Clone)]
pub struct AppState {
    pub engine: Engine,
    pub config: Config,
    pub sqlite: Option<SqliteService>,
    pub search_engine: Arc<SearchEngine>,
}

pub fn router(
    engine: Engine,
    config: Config,
    sqlite: Option<SqliteService>,
    search_engine: Arc<SearchEngine>,
) -> Router {
    let state = AppState {
        engine,
        config,
        sqlite,
        search_engine,
    };
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
        .route("/", get(routes_ui::handler))
        .route("/index.html", get(routes_ui::handler))
        .merge(routes_docs::routes_docs())
        .route("/v1/health", get(routes_state::health))
        .route("/v1/metrics", get(routes_state::metrics))
        .route("/v1/state", get(routes_state::list))
        .route("/v1/state/batch_put", post(routes_state::batch_put))
        .route("/v1/state/:key", get(routes_state::get))
        .route("/v1/state/:key", put(routes_state::put))
        .route("/v1/state/:key", delete(routes_state::delete))
        .route("/v1/doc/:collection/:id", put(routes_doc::put))
        .route("/v1/doc/:collection/:id", get(routes_doc::get))
        .route("/v1/doc/:collection/:id", delete(routes_doc::delete))
        .route("/v1/doc/:collection/find", post(routes_doc::find))
        .route("/v1/events", get(routes_events::events))
        .route("/v1/stream", get(routes_events::stream))
        .route("/v1/vector", get(routes_vector::list_collections))
        .route(
            "/v1/vector/:collection",
            get(routes_vector::get_collection_detail).post(routes_vector::create_collection),
        )
        .route("/v1/vector/:collection/add", post(routes_vector::add))
        .route("/v1/vector/:collection/upsert", post(routes_vector::upsert))
        .route(
            "/v1/vector/:collection/upsert_batch",
            post(routes_vector::upsert_batch),
        )
        .route("/v1/vector/:collection/update", post(routes_vector::update))
        .route("/v1/vector/:collection/delete", post(routes_vector::delete))
        .route(
            "/v1/vector/:collection/delete_batch",
            post(routes_vector::delete_batch),
        )
        .route("/v1/vector/:collection/get", get(routes_vector::get))
        .route("/v1/vector/:collection/search", post(routes_vector::search))
        .route(
            "/v1/vector/:collection/diskann/build",
            post(routes_vector::diskann_build),
        )
        .route(
            "/v1/vector/:collection/diskann/tune",
            post(routes_vector::diskann_tune),
        )
        .route(
            "/v1/vector/:collection/diskann/status",
            get(routes_vector::diskann_status),
        )
        .route("/v1/sql/query", post(routes_sql::query))
        .route("/v1/sql/exec", post(routes_sql::exec))
        .route("/search", post(routes_search::search))
        .route("/search/ingest", post(routes_search::ingest))
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
