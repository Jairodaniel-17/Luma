pub mod accounts;
pub mod audit;
pub mod auth;
pub mod auth_store;
pub mod errors;
pub mod rbac;
pub mod routes_accounts;
pub mod routes_admin;
pub mod routes_auth;
pub mod routes_blob;
pub mod routes_config;
pub mod routes_doc;
pub mod routes_docs;
pub mod routes_events;
pub mod routes_hub;
pub mod routes_image;
pub mod routes_memory;
pub mod routes_meta;
pub mod routes_queue;
pub mod routes_rbac;
pub mod routes_search;
pub mod routes_state;
pub mod routes_ui;
pub mod routes_vector;

use crate::config::Config;
use crate::engine::Engine;
use crate::search::engine::SearchEngine;
use crate::sqlite::SqliteService;
use auth_store::AuthStore;
use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderValue, StatusCode};
use axum::routing::{delete, get, post, put};
use axum::Router;
use std::sync::Arc;
use std::time::Duration;
use tower_governor::governor::GovernorConfigBuilder;
use tower_governor::GovernorLayer;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
#[derive(Clone)]
pub struct AppState {
    pub engine: Engine,
    pub config: Config,
    pub sqlite: Option<SqliteService>,
    pub search_engine: Arc<SearchEngine>,
    pub auth_store: Option<Arc<AuthStore>>,
    pub embeddings: Arc<crate::engine::embeddings::EmbeddingClient>,
    pub hub: Arc<crate::engine::hub::LumaDatabase>,
    pub memory: Arc<crate::memory::MemoryService>,
    pub audit_log: Option<Arc<audit::AuditLog>>,
    pub rbac: Option<Arc<rbac::RbacService>>,
    pub accounts: Option<Arc<accounts::AccountsService>>,
}

#[derive(Clone, Debug)]
pub struct TenantContext {
    pub tenant_id: Option<String>,
    /// Set when the caller authenticated with a user session token.
    pub user_id: Option<String>,
    pub role: String,
    /// Passes platform-admin gates (instance-wide settings, cross-tenant admin)
    /// even while `tenant_id` stays set. The first-registered user (instance
    /// operator) gets this so instance settings work without making their data
    /// untenanted, which would break tenant isolation for their collections.
    pub platform_admin: bool,
    pub permissions: serde_json::Value,
    pub quotas: serde_json::Value,
}

/// Content-Security-Policy applied to every response.
///
/// `script-src 'self'` (no `'unsafe-inline'`) blocks reflected/stored inline
/// script injection — the embedded admin SPA loads its JS from `/assets/*`.
/// The Scalar API docs page loads its bundle from jsdelivr, so that origin is
/// allow-listed for scripts/styles/fonts. `object-src 'none'` blocks plugins.
/// `frame-ancestors 'self'` lets the admin SPA embed same-origin pages (e.g. the
/// docs in an iframe) while still blocking cross-origin clickjacking.
const CSP: &str = "default-src 'self'; \
script-src 'self' https://cdn.jsdelivr.net; \
style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; \
img-src 'self' data: https:; \
font-src 'self' data: https://cdn.jsdelivr.net; \
connect-src 'self'; \
object-src 'none'; \
frame-ancestors 'self'; \
base-uri 'self'";

async fn security_headers(mut response: axum::response::Response) -> axum::response::Response {
    let headers = response.headers_mut();
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("x-frame-options", HeaderValue::from_static("SAMEORIGIN"));
    headers.insert(
        "referrer-policy",
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    headers.insert("x-xss-protection", HeaderValue::from_static("0"));
    headers.insert(
        "permissions-policy",
        HeaderValue::from_static("geolocation=(), microphone=(), camera=()"),
    );
    headers.insert("content-security-policy", HeaderValue::from_static(CSP));
    // HSTS is honored by browsers only over HTTPS; harmless over plain HTTP.
    headers.insert(
        "strict-transport-security",
        HeaderValue::from_static("max-age=63072000; includeSubDomains"),
    );
    response
}

/// Extract the collection/bucket segment from a raw-store path, if any.
///
/// Only the raw stores that key purely by name are covered here: the vector
/// store, the JSON document store, and the blob store. The hub (`/v1/db`) and
/// NS-Mem (`/v1/memory`) intentionally share namespaces across tenants and
/// isolate *internally* by the token's tenant id, so exclusive name ownership
/// must not be imposed on them.
fn scoped_resource_name(path: &str) -> Option<&str> {
    // `/v1/image/` transforms read the same `blobs/{bucket}/{key}` files as
    // `/v1/blob/`, so it must inherit the identical per-bucket ownership check.
    for prefix in ["/v1/vector/", "/v1/doc/", "/v1/blob/", "/v1/image/"] {
        if let Some(rest) = path.strip_prefix(prefix) {
            let seg = rest.split('/').next().unwrap_or("");
            if !seg.is_empty() {
                return Some(seg);
            }
        }
    }
    None
}

/// Per-organization data isolation. Runs after authentication (so the
/// [`TenantContext`] is present) and enforces that a collection/namespace is
/// only ever accessed by the org that first created it. Platform admins (no
/// `tenant_id`) and requests without a scoped resource are passed through.
///
/// First-touch by an org records ownership; any other org touching the same
/// name gets a `404` (existence is hidden across tenants).
async fn tenant_isolation_middleware(
    axum::extract::State(state): axum::extract::State<AppState>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, errors::ApiError> {
    let Some(ctx) = req.extensions().get::<TenantContext>().cloned() else {
        return Ok(next.run(req).await);
    };
    let Some(org) = ctx.tenant_id.clone() else {
        // Platform admin / static key: not scoped to a single org.
        return Ok(next.run(req).await);
    };
    let Some(accounts) = state.accounts.clone() else {
        return Ok(next.run(req).await);
    };
    let Some(name) = scoped_resource_name(req.uri().path()).map(str::to_string) else {
        return Ok(next.run(req).await);
    };

    match accounts.collection_owner(&name).await {
        Ok(Some(owner)) if owner != org => {
            return Err(errors::ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found",
                "resource not found",
            ));
        }
        Ok(Some(_)) => {} // owned by the caller's org
        Ok(None) => {
            // First touch: this org now owns the resource.
            let _ = accounts.register_collection(&name, &org).await;
        }
        Err(e) => {
            tracing::warn!("tenant isolation registry error for '{name}': {e}");
        }
    }

    Ok(next.run(req).await)
}

pub struct RouterDeps {
    pub engine: Engine,
    pub config: Config,
    pub sqlite: Option<SqliteService>,
    pub search_engine: Arc<SearchEngine>,
    pub auth_store: Option<Arc<AuthStore>>,
    pub embeddings: Arc<crate::engine::embeddings::EmbeddingClient>,
    pub audit_log: Option<Arc<audit::AuditLog>>,
    pub rbac: Option<Arc<rbac::RbacService>>,
}

pub fn router(deps: RouterDeps) -> Router<()> {
    let RouterDeps {
        engine,
        config,
        sqlite,
        search_engine,
        auth_store,
        embeddings,
        audit_log,
        rbac,
    } = deps;

    let memory = Arc::new(crate::memory::MemoryService::new(
        Arc::new(engine.clone()),
        sqlite.clone().map(Arc::new),
        (*embeddings).clone(),
        config.clone(),
    ));
    let hub = Arc::new(crate::engine::hub::LumaDatabase::new(
        Arc::new(engine.clone()),
        sqlite.clone().map(Arc::new),
        (*embeddings).clone(),
        crate::engine::chunking::ChunkingEngine::default(),
        config.clone(),
    ));

    // Enterprise account layer (orgs/users/sessions/isolation) — available
    // whenever SQLite is enabled. Tables are created lazily on first use.
    let accounts = sqlite
        .clone()
        .map(|svc| Arc::new(accounts::AccountsService::new(Arc::new(svc))));

    let state = AppState {
        engine,
        config,
        sqlite,
        search_engine,
        auth_store,
        embeddings,
        hub,
        memory,
        audit_log,
        rbac,
        accounts,
    };
    let cors = match state.config.cors_allowed_origins.as_deref().map(str::trim) {
        // No config: same-origin only. An empty CorsLayer emits no
        // `Access-Control-Allow-*` headers, so browsers block cross-origin
        // requests. Wide-open `Any/Any/Any` is never the implicit default.
        None => CorsLayer::new(),
        // Explicit opt-in to fully permissive CORS.
        Some("*") => CorsLayer::new()
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
    let app = Router::<AppState>::new()
        .route("/", get(routes_ui::handler))
        .route("/index.html", get(routes_ui::handler))
        .merge(routes_docs::routes_docs())
        .route("/v1/health", get(routes_state::health))
        .route("/v1/metrics", get(routes_state::metrics))
        // ---- Enterprise accounts: register / login are public; the rest need a token ----
        .route("/v1/auth/register", post(routes_accounts::register))
        .route("/v1/auth/login", post(routes_accounts::login))
        .route("/v1/auth/logout", post(routes_accounts::logout))
        .route("/v1/auth/refresh", post(routes_accounts::refresh))
        .route(
            "/v1/auth/access-policy",
            get(routes_accounts::get_access_policy).put(routes_accounts::set_access_policy),
        )
        .route(
            "/v1/auth/domain-orgs",
            get(routes_accounts::list_domain_orgs).put(routes_accounts::set_domain_org),
        )
        .route(
            "/v1/auth/domain-orgs/:domain",
            delete(routes_accounts::delete_domain_org),
        )
        .route(
            "/v1/admin/orgs",
            get(routes_accounts::list_orgs).post(routes_accounts::create_org),
        )
        .route("/v1/admin/orgs/:id", delete(routes_accounts::delete_org))
        .route(
            "/v1/admin/orgs/:id/members",
            get(routes_accounts::list_org_members).post(routes_accounts::add_org_member),
        )
        .route(
            "/v1/admin/orgs/:id/invite",
            post(routes_accounts::invite_member),
        )
        .route(
            "/v1/admin/orgs/:id/members/:user_id",
            put(routes_accounts::update_org_member_role).delete(routes_accounts::remove_org_member),
        )
        .route(
            "/v1/admin/users",
            get(routes_accounts::list_users).post(routes_accounts::create_user),
        )
        .route("/v1/admin/users/:id", delete(routes_accounts::delete_user))
        .route(
            "/v1/admin/users/:id/role",
            put(routes_accounts::update_user_role),
        )
        .route(
            "/v1/admin/users/:id/orgs",
            get(routes_accounts::list_user_orgs),
        )
        .route("/v1/auth/my-orgs", get(routes_accounts::my_orgs))
        .route("/v1/auth/switch-org", post(routes_accounts::switch_org))
        .route("/v1/auth/sessions", get(routes_accounts::list_sessions))
        .route(
            "/v1/auth/sessions/revoke-all",
            post(routes_accounts::revoke_all_sessions),
        )
        .route("/v1/admin/stats", get(routes_accounts::stats))
        .route("/v1/admin/audit-events", get(routes_accounts::audit_events))
        .route(
            "/v1/auth/keys",
            get(routes_auth::list_keys).post(routes_auth::create_key),
        )
        .route("/v1/auth/keys/:id", delete(routes_auth::revoke_key))
        .route("/v1/auth/keys/:id/role", put(routes_auth::update_key_role))
        .route(
            "/v1/auth/roles",
            get(routes_rbac::list_roles).post(routes_rbac::create_role),
        )
        .route("/v1/auth/roles/:id", delete(routes_rbac::delete_role))
        .route(
            "/v1/auth/roles/:id/permissions",
            get(routes_rbac::list_permissions)
                .post(routes_rbac::add_permission)
                .delete(routes_rbac::remove_permission),
        )
        .route("/v1/auth/roles/check", get(routes_rbac::check_permission))
        .route("/v1/state", get(routes_state::list))
        .route("/v1/state/indexes", post(routes_state::create_index))
        .route(
            "/v1/state/index/:field/:value",
            get(routes_state::query_index),
        )
        .route("/v1/state/batch_put", post(routes_state::batch_put))
        .route("/v1/state/:key", get(routes_state::get))
        .route("/v1/state/:key", put(routes_state::put))
        .route("/v1/state/:key", delete(routes_state::delete))
        .route("/v1/doc/:collection/:id", put(routes_doc::put))
        .route("/v1/doc/:collection/:id", get(routes_doc::get))
        .route("/v1/doc/:collection/:id", delete(routes_doc::delete))
        .route("/v1/doc/:collection/find", post(routes_doc::find))
        // Motor de consultas sobre una colección (MetaEngine: SQL/meta). El
        // handler existía pero no estaba montado — caía al SPA fallback (HTML).
        .route("/v1/meta/:collection/execute", post(routes_meta::execute))
        .route("/v1/blob/:bucket", get(routes_blob::list))
        .route("/v1/blob/:bucket/:key", put(routes_blob::put))
        .route("/v1/blob/:bucket/:key", get(routes_blob::get))
        .route("/v1/blob/:bucket/:key", delete(routes_blob::delete))
        .route("/v1/queue/:queue", post(routes_queue::enqueue))
        .route("/v1/queue/:queue", get(routes_queue::stats))
        .route("/v1/queue/:queue/receive", post(routes_queue::receive))
        .route("/v1/queue/:queue/:id", delete(routes_queue::ack))
        .route("/v1/image/:bucket/:key", get(routes_image::transform))
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
            "/v1/vector/:collection/search_batch",
            post(routes_vector::search_batch),
        )
        .route("/v1/vector/:collection/scroll", get(routes_vector::scroll))
        .route("/v1/vector/:collection/rerank", post(routes_vector::rerank))
        .route(
            "/v1/vector/:collection/aggregate",
            post(routes_vector::aggregate),
        )
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
        .route("/v1/db/:namespace/ingest", post(routes_hub::ingest))
        .route("/v1/db/:namespace/search", post(routes_hub::search))
        .route(
            "/v1/memory/:namespace/ingest_event",
            post(routes_memory::ingest_event),
        )
        .route(
            "/v1/memory/:namespace/upsert_fact",
            post(routes_memory::upsert_fact),
        )
        .route(
            "/v1/memory/:namespace/upsert_procedure",
            post(routes_memory::upsert_procedure),
        )
        .route("/v1/memory/:namespace/query", post(routes_memory::query))
        .route(
            "/v1/memory/:namespace/next_step",
            post(routes_memory::next_step),
        )
        .route(
            "/v1/memory/:namespace/timeline/:entity_id",
            get(routes_memory::timeline),
        )
        .route(
            "/v1/memory/:namespace/edges",
            post(routes_memory::upsert_edge),
        )
        .route(
            "/v1/memory/:namespace/edges/:memory_id",
            get(routes_memory::get_node_edges),
        )
        .route(
            "/v1/memory/:namespace/edges/:edge_id/delete",
            post(routes_memory::delete_edge),
        )
        .route(
            "/v1/memory/:namespace/beliefs/:fact_key/history",
            get(routes_memory::belief_history),
        )
        .route(
            "/v1/memory/:namespace/graph/centrality",
            post(routes_memory::recompute_centrality),
        )
        .route("/v1/config", get(routes_config::get_config))
        .route("/v1/config", put(routes_config::update_config))
        .route(
            "/v1/config/embedding/probe",
            post(routes_config::probe_embedding),
        )
        .route("/v1/admin/backup", post(routes_admin::backup))
        .route("/v1/admin/audit", get(routes_admin::get_audit_log))
        .route("/search", post(routes_search::search))
        .route("/search/ingest", post(routes_search::ingest))
        // SPA fallback: serves embedded admin panel assets + index.html for any
        // unmatched route. API routes above are matched first, so no collision.
        .fallback(routes_ui::spa_fallback)
        .layer(DefaultBodyLimit::max(state.config.max_body_bytes))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(state.config.request_timeout_secs),
        ))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            audit::audit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            tenant_isolation_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(axum::middleware::map_response(security_headers))
        .with_state(state.clone());

    if state.config.rate_limit_rps > 0 {
        let burst = if state.config.rate_limit_burst > 0 {
            state.config.rate_limit_burst
        } else {
            state.config.rate_limit_rps * 10
        };
        let governor_conf = Arc::new(
            GovernorConfigBuilder::default()
                .per_second(state.config.rate_limit_rps as u64)
                .burst_size(burst)
                .finish()
                .expect("invalid rate limit configuration"),
        );
        app.layer(GovernorLayer {
            config: governor_conf,
        })
        // ponytail: PeerIpKeyExtractor returns 500 (UnableToExtractKey) when no
        // ConnectInfo<SocketAddr> extension is present. Guarantee one so rate
        // limiting never crashes requests even if the server is served without
        // `into_make_service_with_connect_info` (library embedders, tests). Real
        // per-IP limiting still applies once ConnectInfo is wired — see server.rs.
        .layer(axum::middleware::from_fn(ensure_connect_info))
    } else {
        app
    }
}

/// Ensure a `ConnectInfo<SocketAddr>` extension exists on every request so the
/// rate limiter's peer-IP key extractor never fails. Real connection info (set
/// by the serving layer) is left untouched; only missing info gets a fallback.
async fn ensure_connect_info(
    mut req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::extract::ConnectInfo;
    use std::net::{Ipv4Addr, SocketAddr};
    if req.extensions().get::<ConnectInfo<SocketAddr>>().is_none() {
        let fallback = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0));
        req.extensions_mut().insert(ConnectInfo(fallback));
    }
    next.run(req).await
}
