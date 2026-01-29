use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use luma::sqlite::SqliteService;
use std::net::SocketAddr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub async fn run(config: Config) -> anyhow::Result<()> {
    if let Some(ref dir) = config.data_dir {
        ensure_data_dir(dir)?;
        let abs_path = fs::canonicalize(dir)?;
        tracing::info!("💽 Data Directory: {}", abs_path.display());
    }

    let sqlite = if config.sqlite_enabled {
        Some(init_sqlite(&config)?)
    } else {
        None
    };

    let auth_store = if let Some(svc) = &sqlite {
        let store = Arc::new(luma::api::auth_store::AuthStore::new(Arc::new(svc.clone())));
        store.init().await?;
        // Ensure the key configured in env/args (default "dev") exists
        store.ensure_bootstrap_key(&config.api_key).await?;
        Some(store)
    } else {
        None
    };

    let shutdown_token = CancellationToken::new();
    let engine = Engine::new(config.clone(), shutdown_token.clone())?;

    let data_dir = config.data_dir.clone().map(PathBuf::from).unwrap_or(PathBuf::from("data"));
    let search_engine = Arc::new(SearchEngine::new(data_dir)?);

    let app = luma::api::router(engine.clone(), config.clone(), sqlite, search_engine, auth_store);
    let addr = SocketAddr::new(config.bind_addr, config.port);

    tracing::info!(%addr, "listening");
    tracing::info!("Process ID: {}", std::process::id());

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_token))
        .await?;

    tracing::info!("Server stopped.");
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
    let path = config.sqlite_path.clone()
        .or_else(|| {
            config.data_dir
                .as_ref()
                .map(|d| format!("{d}/sqlite/rustkiss.db"))
        })
        .ok_or_else(|| anyhow::anyhow!("SQLITE_ENABLED requiere DATA_DIR o SQLITE_DB_PATH"))?;

    SqliteService::new(path)
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

    // Force exit if graceful shutdown takes too long (e.g. open streams)
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        tracing::warn!("Graceful shutdown timed out. Forcing exit.");
        tracing::warn!("If stuck, use: kill {}", std::process::id());
        std::process::exit(0);
    });
}
