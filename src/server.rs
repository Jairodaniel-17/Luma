use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use luma::sqlite::SqliteService;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub async fn run(config: Config) -> anyhow::Result<()> {
    // Router role: forward requests to backend nodes by namespace sharding
    // instead of running the engine locally. See `luma::router`.
    if config.role.eq_ignore_ascii_case("router") {
        return run_router(config).await;
    }

    // ---------------------------------------------------------
    // 1. Config Validation & Logging (Anti-DoS Order)
    // ---------------------------------------------------------
    if config.max_json_bytes > config.max_body_bytes {
        let msg = format!(
            "CRITICAL: MAX_JSON_MB ({:.4}) cannot be greater than MAX_BODY_MB ({:.4})",
            config.max_json_bytes as f64 / 1_048_576.0,
            config.max_body_bytes as f64 / 1_048_576.0
        );
        tracing::error!("{}", msg);
        anyhow::bail!(msg);
    }

    // Fail-fast on insecure secrets unless the operator explicitly opts in via
    // LUMA_ALLOW_INSECURE (intended for local/dev use only). No such flag set
    // means production posture: refuse to start.
    check_secure_startup(&config)?;

    tracing::info!(
        "[config] max_body_mb = {:.4}",
        config.max_body_bytes as f64 / 1_048_576.0
    );
    tracing::info!(
        "[config] max_json_mb = {:.4}",
        config.max_json_bytes as f64 / 1_048_576.0
    );
    tracing::info!("[config] max_vector_dim = {}", config.max_vector_dim);
    tracing::info!("[config] max_k = {}", config.max_k);
    tracing::info!(
        "[config] request_timeout_secs = {}",
        config.request_timeout_secs
    );
    tracing::info!(
        "[config] wal_retention_segments = {}",
        config.wal_retention_segments
    );

    // ---------------------------------------------------------

    if let Some(ref dir) = config.data_dir {
        ensure_data_dir(dir)?;
        let abs_path = fs::canonicalize(dir)?;
        tracing::info!("Data Directory: {}", abs_path.display());
    }

    let sqlite = if config.sqlite_enabled {
        if let Some(ref url) = config.libsql_url {
            tracing::info!("SQLite backend: libSQL/Turso remote — {url}");
            Some(luma::sqlite::SqliteService::new_remote(
                url.clone(),
                config.libsql_auth_token.clone(),
            ))
        } else {
            tracing::info!("SQLite backend: local rusqlite (WAL)");
            Some(init_sqlite(&config)?)
        }
    } else {
        None
    };

    let auth_store = if let Some(svc) = &sqlite {
        let store = Arc::new(luma::api::auth_store::AuthStore::new(Arc::new(svc.clone())));
        store.init().await?;
        store.ensure_bootstrap_key(&config.api_key).await?;
        Some(store)
    } else {
        None
    };

    // The RESP listener needs the same key store, so `AUTH <api-key>` binds a
    // connection to the org that owns the key instead of only accepting the
    // instance-wide password.
    let auth_store_for_resp = auth_store.clone();

    let rbac = if let Some(svc) = &sqlite {
        let r = Arc::new(luma::api::rbac::RbacService::new(Arc::new(svc.clone())));
        r.init().await?;
        Some(r)
    } else {
        None
    };

    let audit_log = if let Some(svc) = &sqlite {
        let log = Arc::new(luma::api::audit::AuditLog::new(Arc::new(svc.clone())));
        log.init().await?;
        Some(log)
    } else {
        None
    };

    let shutdown_token = CancellationToken::new();
    let engine = Engine::new(config.clone(), shutdown_token.clone())?;

    let data_dir = config
        .data_dir
        .clone()
        .map(PathBuf::from)
        .unwrap_or(PathBuf::from("data"));
    let search_engine = Arc::new(SearchEngine::new(data_dir)?);

    let embeddings = init_embeddings(&config, engine.metrics());
    // Allocated before the router so `/v1/metrics` can render the RESP counters
    // from the same instance the listener increments.
    let resp_metrics = std::sync::Arc::new(luma::resp::listener::RespMetrics::default());
    let app = luma::api::router(luma::api::RouterDeps {
        engine: engine.clone(),
        config: config.clone(),
        sqlite,
        search_engine,
        auth_store: auth_store.clone(),
        embeddings,
        resp_metrics: config
            .resp_port
            .gt(&0)
            .then(|| std::sync::Arc::clone(&resp_metrics)),
        audit_log,
        rbac,
    });
    let addr = SocketAddr::new(config.bind_addr, config.port);

    // Opt-in periodic backups (SQLite + snapshot + WAL).
    luma::backup::spawn_backup_task(config.clone());
    // Continuous WAL shipping: bounds the recovery point to one interval
    // instead of to the gap between full backups.
    luma::wal_ship::spawn(config.clone());

    // Redis-protocol listener. Off unless `resp_port` is set: an engine that
    // starts listening on 6379 on upgrade is a surprise, and on a shared host a
    // conflict with the real Redis.
    match luma::resp::listener::spawn(
        &config,
        engine.clone(),
        std::sync::Arc::clone(&resp_metrics),
        auth_store_for_resp,
        shutdown_token.clone(),
    )
    .await
    {
        Ok(Some(port)) => tracing::info!(port, "RESP listener bound"),
        Ok(None) => {}
        // A failed bind must not take the HTTP server down with it: the rest of
        // the instance is perfectly serviceable without RESP.
        Err(e) => tracing::error!("RESP listener failed to bind, continuing without it: {e}"),
    }

    tracing::info!(%addr, "listening");
    tracing::info!("Process ID: {}", std::process::id());

    match (&config.tls_cert_path, &config.tls_key_path) {
        (Some(cert), Some(key)) => {
            tracing::info!("TLS enabled — cert: {cert}");
            serve_tls(app, addr, cert, key, shutdown_token).await?;
        }
        (Some(_), None) | (None, Some(_)) => {
            tracing::warn!(
                "TLS partially configured: both tls_cert_path and tls_key_path must be set. \
                 Falling back to plain HTTP."
            );
            serve_plain(app, addr, shutdown_token).await?;
        }
        _ => {
            tracing::info!("TLS disabled — plain HTTP");
            serve_plain(app, addr, shutdown_token).await?;
        }
    }

    tracing::info!("Server stopped.");
    Ok(())
}

/// Run in router role: forward requests to backend nodes by namespace sharding.
async fn run_router(config: Config) -> anyhow::Result<()> {
    if config.router_nodes.is_empty() {
        anyhow::bail!(
            "role=router requires ROUTER_NODES (comma-separated backend base URLs, \
             e.g. http://node-a:1234,http://node-b:1234)"
        );
    }
    let addr = SocketAddr::new(config.bind_addr, config.port);
    tracing::info!(
        nodes = ?config.router_nodes,
        %addr,
        "starting Luma in ROUTER mode (namespace sharding)"
    );
    let state = luma::router::RouterState::new(
        config.router_nodes.clone(),
        config.max_body_bytes,
        config.request_timeout_secs,
    );
    let app = luma::router::build_app(state);
    let shutdown_token = CancellationToken::new();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal(shutdown_token))
        .await?;
    tracing::info!("Router stopped.");
    Ok(())
}

async fn serve_plain(
    app: axum::Router,
    addr: SocketAddr,
    shutdown_token: CancellationToken,
) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    // into_make_service_with_connect_info supplies the ConnectInfo<SocketAddr>
    // the rate limiter's peer-IP key extractor needs (otherwise every request 500s).
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal(shutdown_token))
    .await?;
    Ok(())
}

/// TLS listener using tokio-rustls (no axum-server dependency).
///
/// Loads a PEM certificate chain + PKCS#8 private key, builds a rustls
/// ServerConfig, and serves each accepted connection through hyper directly.
async fn serve_tls(
    app: axum::Router,
    addr: SocketAddr,
    cert_path: &str,
    key_path: &str,
    shutdown_token: CancellationToken,
) -> anyhow::Result<()> {
    use hyper::body::Incoming;
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder;
    use rustls::ServerConfig;
    use rustls_pemfile::{certs, pkcs8_private_keys};
    use std::io::BufReader;
    use tokio::net::TcpListener;
    use tokio_rustls::TlsAcceptor;

    // Same reason as the RESP listener: an ambiguous provider is a panic at the
    // first handshake, not an error at startup.
    luma::install_crypto_provider();

    // Load certificate chain (PEM)
    let cert_file = fs::File::open(cert_path)
        .map_err(|e| anyhow::anyhow!("cannot open TLS cert '{cert_path}': {e}"))?;
    let certs: Vec<_> = certs(&mut BufReader::new(cert_file))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("invalid TLS cert '{cert_path}': {e}"))?;

    // Load private key (PKCS#8 PEM)
    let key_file = fs::File::open(key_path)
        .map_err(|e| anyhow::anyhow!("cannot open TLS key '{key_path}': {e}"))?;
    let mut keys: Vec<_> = pkcs8_private_keys(&mut BufReader::new(key_file))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("invalid TLS key '{key_path}': {e}"))?;
    if keys.is_empty() {
        anyhow::bail!(
            "No PKCS#8 private key found in '{key_path}'. \
             Use `openssl pkcs8 -topk8 -nocrypt` to convert RSA keys."
        );
    }

    let tls_cfg = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(
            certs,
            rustls::pki_types::PrivateKeyDer::Pkcs8(keys.remove(0)),
        )?;
    let acceptor = TlsAcceptor::from(Arc::new(tls_cfg));
    let listener = TcpListener::bind(addr).await?;

    loop {
        let (tcp_stream, peer_addr) = tokio::select! {
            biased;
            () = shutdown_token.cancelled() => break,
            result = listener.accept() => match result {
                Ok(conn) => conn,
                Err(e) => {
                    tracing::warn!("TCP accept error: {e}");
                    continue;
                }
            }
        };

        let acceptor = acceptor.clone();
        // Clone the Router for this connection; Router is cheap to clone (Arc inside).
        let app = app.clone();

        tokio::spawn(async move {
            let tls_stream = match acceptor.accept(tcp_stream).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::debug!("TLS handshake failed: {e}");
                    return;
                }
            };
            let io = TokioIo::new(tls_stream);
            // Convert hyper::Request<Incoming> → axum::Request<Body> in the service fn.
            let hyper_service = hyper::service::service_fn(move |req: hyper::Request<Incoming>| {
                let app = app.clone();
                async move {
                    use tower::ServiceExt as _;
                    // Supply ConnectInfo<SocketAddr> so the rate limiter's peer-IP
                    // key extractor can identify the client on TLS connections too.
                    let mut req = req.map(axum::body::Body::new);
                    req.extensions_mut()
                        .insert(axum::extract::ConnectInfo(peer_addr));
                    app.oneshot(req).await
                }
            });
            if let Err(e) = Builder::new(TokioExecutor::new())
                .serve_connection_with_upgrades(io, hyper_service)
                .await
            {
                tracing::debug!("connection error: {e}");
            }
        });
    }

    Ok(())
}

/// Whether the operator explicitly opted in to running with insecure secrets.
fn allow_insecure() -> bool {
    std::env::var("LUMA_ALLOW_INSECURE")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
        .unwrap_or(false)
}

/// Collect insecure-secret problems for the given config + environment.
fn insecure_startup_problems(api_key: &str, master_key_set: bool) -> Vec<&'static str> {
    let mut problems = Vec::new();
    if api_key == "dev" || api_key.is_empty() || api_key.len() < 16 {
        problems
            .push("api_key is weak or default ('dev'/empty/<16 chars); set LUMA_API_KEY to a strong secret");
    }
    if !master_key_set {
        problems.push(
            "LUMA_MASTER_KEY is not set; encryption-at-rest would use a well-known development key",
        );
    }
    problems
}

/// Refuse to start with insecure secrets unless `LUMA_ALLOW_INSECURE` is set.
/// When the opt-out is set, insecure secrets only produce warnings.
fn check_secure_startup(config: &Config) -> anyhow::Result<()> {
    let problems =
        insecure_startup_problems(&config.api_key, std::env::var("LUMA_MASTER_KEY").is_ok());
    if problems.is_empty() {
        return Ok(());
    }

    if allow_insecure() {
        for p in &problems {
            tracing::warn!("INSECURE (allowed via LUMA_ALLOW_INSECURE): {p}");
        }
        Ok(())
    } else {
        for p in &problems {
            tracing::error!("INSECURE: {p}");
        }
        anyhow::bail!(
            "refusing to start with {} insecure secret setting(s); fix the issues logged above, \
             or set LUMA_ALLOW_INSECURE=1 to override (local/dev only)",
            problems.len()
        )
    }
}

fn ensure_data_dir(path: &str) -> anyhow::Result<()> {
    let p = Path::new(path);
    if !p.exists() {
        fs::create_dir_all(p)?;
    } else if !p.is_dir() {
        anyhow::bail!("DATA_DIR exists but is not a directory: {}", p.display());
    }
    Ok(())
}

fn init_sqlite(config: &Config) -> anyhow::Result<SqliteService> {
    let path = config
        .sqlite_path
        .clone()
        .or_else(|| {
            config
                .data_dir
                .as_ref()
                .map(|d| format!("{d}/sqlite/rustkiss.db"))
        })
        .ok_or_else(|| anyhow::anyhow!("SQLITE_ENABLED requiere DATA_DIR o SQLITE_DB_PATH"))?;
    SqliteService::new(path)
}

/// Builds the swappable embedding handle every subsystem shares.
///
/// The provider mapping itself lives in `EmbeddingProvider::from_config` so the
/// hot-reload path in `PUT /v1/config` applies exactly the same rules as
/// startup.
fn init_embeddings(
    config: &Config,
    metrics: std::sync::Arc<luma::engine::metrics::Metrics>,
) -> luma::engine::embeddings::EmbeddingHandle {
    use luma::engine::embeddings::{EmbeddingClient, EmbeddingHandle};

    EmbeddingHandle::new(EmbeddingClient::from_config(config, Some(metrics)))
}

async fn shutdown_signal(token: CancellationToken) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sig = signal(SignalKind::terminate()).expect("failed to install signal handler");
        sig.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            tracing::info!("Received terminate signal, shutting down...");
        },
    }

    token.cancel();

    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        tracing::warn!("Graceful shutdown timed out. Forcing exit.");
        std::process::exit(0);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_config_has_no_problems() {
        let problems = insecure_startup_problems("a-strong-api-key-32-chars-long!!", true);
        assert!(problems.is_empty());
    }

    #[test]
    fn weak_api_key_is_flagged() {
        assert_eq!(insecure_startup_problems("dev", true).len(), 1);
        assert_eq!(insecure_startup_problems("", true).len(), 1);
        assert_eq!(insecure_startup_problems("short", true).len(), 1);
    }

    #[test]
    fn missing_master_key_is_flagged() {
        assert_eq!(
            insecure_startup_problems("a-strong-api-key-32-chars-long!!", false).len(),
            1
        );
    }

    #[test]
    fn both_insecure_reports_two_problems() {
        assert_eq!(insecure_startup_problems("dev", false).len(), 2);
    }
}
