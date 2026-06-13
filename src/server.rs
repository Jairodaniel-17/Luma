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

    if config.api_key == "dev" || config.api_key.len() < 16 {
        tracing::warn!(
            "INSECURE: api_key is weak or default ('dev'). \
             Set LUMA_API_KEY to a strong secret before production use."
        );
    }

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
    let app = luma::api::router(luma::api::RouterDeps {
        engine: engine.clone(),
        config: config.clone(),
        sqlite,
        search_engine,
        auth_store,
        embeddings,
        audit_log,
        rbac,
    });
    let addr = SocketAddr::new(config.bind_addr, config.port);

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

async fn serve_plain(
    app: axum::Router,
    addr: SocketAddr,
    shutdown_token: CancellationToken,
) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
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
        .with_single_cert(certs, rustls::pki_types::PrivateKeyDer::Pkcs8(keys.remove(0)))?;
    let acceptor = TlsAcceptor::from(Arc::new(tls_cfg));
    let listener = TcpListener::bind(addr).await?;

    loop {
        let (tcp_stream, _peer_addr) = tokio::select! {
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
                    app.oneshot(req.map(axum::body::Body::new)).await
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

fn init_embeddings(
    config: &Config,
    metrics: std::sync::Arc<luma::engine::metrics::Metrics>,
) -> Arc<luma::engine::embeddings::EmbeddingClient> {
    use luma::engine::embeddings::{EmbeddingClient, EmbeddingProvider};

    let provider = match config.embedding_provider.to_lowercase().as_str() {
        "ollama" => EmbeddingProvider::Ollama {
            api_url: config.embedding_url.clone(),
            model: config.embedding_model.clone(),
        },
        "openai" => EmbeddingProvider::OpenAI {
            api_url: config.embedding_url.clone(),
            api_key: config.embedding_api_key.clone(),
            model: config.embedding_model.clone(),
        },
        "azure" | "azure_openai" | "azure-openai" => EmbeddingProvider::AzureOpenAI {
            api_base: config.embedding_azure_api_base.clone(),
            deployment: config.embedding_azure_deployment.clone(),
            api_key: config.embedding_api_key.clone(),
            api_version: config.embedding_azure_api_version.clone(),
        },
        "cohere" => EmbeddingProvider::Cohere {
            api_url: if config.embedding_url.is_empty() {
                "https://api.cohere.ai".to_string()
            } else {
                config.embedding_url.clone()
            },
            api_key: config.embedding_api_key.clone(),
            model: config.embedding_model.clone(),
            input_type: config.embedding_cohere_input_type.clone(),
        },
        "huggingface" | "hf" => EmbeddingProvider::HuggingFace {
            api_url: if config.embedding_url.is_empty() {
                "https://api-inference.huggingface.co".to_string()
            } else {
                config.embedding_url.clone()
            },
            api_key: config.embedding_api_key.clone(),
            model: config.embedding_model.clone(),
        },
        "mock" => EmbeddingProvider::Mock {
            dim: config.embedding_dim,
        },
        _ => EmbeddingProvider::None,
    };

    Arc::new(
        EmbeddingClient::with_limits_and_dim(
            provider,
            config.embedding_cache_size,
            config.embedding_max_inflight_requests,
            Some(metrics),
            config.embedding_dim,
        )
        .with_retry(
            config.embedding_retry_attempts,
            config.embedding_retry_initial_ms,
        ),
    )
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
