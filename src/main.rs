use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let filter = log_filter_from_args();
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let config = Config::from_env()?;
    let engine = Engine::new(config.clone())?;

    let app = rust_kiss_vdb::api::router(engine.clone(), config.clone());
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));

    tracing::info!(%addr, "listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    engine.shutdown();
    Ok(())
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

fn map_log_level(raw: &str) -> Option<&'static str> {
    match raw.to_ascii_lowercase().as_str() {
        "info" => Some("info"),
        "warning" | "warn" => Some("warn"),
        "error" => Some("error"),
        "critical" => Some("error"),
        _ => None,
    }
}
