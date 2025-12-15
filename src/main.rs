use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::sqlite::SqliteService;
use rust_kiss_vdb::vector::VectorStore;
use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    match parse_command()? {
        Command::Vacuum { collection } => {
            run_vacuum(collection)?;
            return Ok(());
        }
        Command::Serve => {}
    }

    let filter = log_filter_from_args();
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let config = Config::from_env()?;
    let sqlite_service = if config.sqlite_enabled {
        Some(init_sqlite(&config)?)
    } else {
        None
    };
    let engine = Engine::new(config.clone())?;

    let app = rust_kiss_vdb::api::router(engine.clone(), config.clone(), sqlite_service.clone());
    let addr = SocketAddr::new(config.bind_addr, config.port);

    tracing::info!(%addr, "listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    engine.shutdown();
    Ok(())
}

fn run_vacuum(collection: String) -> anyhow::Result<()> {
    let config = Config::from_env()?;
    let data_dir = config
        .data_dir
        .ok_or_else(|| anyhow::anyhow!("DATA_DIR requerido para vacuum"))?;
    let store = VectorStore::open(&data_dir)?;
    store
        .vacuum_collection(&collection)
        .map_err(|err| anyhow::anyhow!("vacuum failed: {err}"))?;
    println!("Colección `{collection}` compactada correctamente.");
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
                .map(|dir| format!("{dir}/sqlite/rustkiss.db"))
        })
        .ok_or_else(|| anyhow::anyhow!("SQLITE_ENABLED requiere DATA_DIR o SQLITE_DB_PATH"))?;
    SqliteService::new(path)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        term.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn log_filter_from_args() -> EnvFilter {
    let override_level = parse_log_arg();
    if let Some(level) = override_level {
        return EnvFilter::new(level);
    }
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
}

fn parse_log_arg() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--logs" {
            let Some(raw) = args.next() else {
                eprintln!(
                    "`--logs` requiere un valor (info|warning|error|critical). Usando `info`."
                );
                return Some("info".to_string());
            };
            let mapped = map_log_level(&raw);
            if let Some(level) = mapped {
                return Some(level.to_string());
            }
            eprintln!(
                "Nivel de logs desconocido `{}`. Usa uno de: info, warning, error, critical. Usando `info`.",
                raw
            );
            return Some("info".to_string());
        }
    }
    None
}

enum Command {
    Serve,
    Vacuum { collection: String },
}

fn parse_command() -> anyhow::Result<Command> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 {
        match args[1].as_str() {
            "serve" => return Ok(Command::Serve),
            "vacuum" => {
                let mut collection: Option<String> = None;
                let mut iter = args.iter().skip(2);
                while let Some(arg) = iter.next() {
                    if arg == "--collection" && collection.is_none() {
                        if let Some(value) = iter.next() {
                            collection = Some(value.to_string());
                        } else {
                            anyhow::bail!("`vacuum` requiere `--collection <name>`");
                        }
                    }
                }
                let collection =
                    collection.ok_or_else(|| anyhow::anyhow!("vacuum requiere --collection"))?;
                return Ok(Command::Vacuum { collection });
            }
            _ => {}
        }
    }
    Ok(Command::Serve)
}

fn map_log_level(raw: &str) -> Option<&'static str> {
    match raw.to_ascii_lowercase().as_str() {
        "info" => Some("info"),
        "warning" | "warn" => Some("warn"),
        "error" => Some("error"),
        "critical" => Some("error"),
        _ => None,
    }
}
