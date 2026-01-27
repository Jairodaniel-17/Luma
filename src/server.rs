use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::search::engine::SearchEngine;
use rust_kiss_vdb::sqlite::SqliteService;
use std::net::SocketAddr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub async fn run(config: Config) -> anyhow::Result<()> {
    if let Some(ref dir) = config.data_dir {
        ensure_data_dir(dir)?;
    }

    let sqlite = if config.sqlite_enabled {
        Some(init_sqlite(&config)?)
    } else {
        None
    };

    let engine = Engine::new(config.clone())?;

    let data_dir = config.data_dir.clone().map(PathBuf::from).unwrap_or(PathBuf::from("data"));
    let search_engine = Arc::new(SearchEngine::new(data_dir)?);

    let app = rust_kiss_vdb::api::router(engine.clone(), config.clone(), sqlite, search_engine);
    let addr = SocketAddr::new(config.bind_addr, config.port);

    tracing::info!(%addr, "listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    engine.shutdown();
    Ok(())
}

fn ensure_data_dir(path: &str) -> anyhow::Result<()> {
    let p = Path::new(path);

    if !p.exists() {
        fs::create_dir_all(p)?;
        tracing::info!("data directory created at {}", p.display());
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

async fn shutdown_signal() {
    let ctrl_c = async { let _ = tokio::signal::ctrl_c().await; };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sig = signal(SignalKind::terminate()).unwrap();
        sig.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
