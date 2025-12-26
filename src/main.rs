use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::sqlite::SqliteService;
use rust_kiss_vdb::vector::index::DiskAnnBuildParams;
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
        Command::DiskAnnBuild(opts) => {
            run_diskann_build(opts)?;
            return Ok(());
        }
        Command::DiskAnnTune(opts) => {
            run_diskann_tune(opts)?;
            return Ok(());
        }
        Command::DiskAnnStatus { collection } => {
            run_diskann_status(collection)?;
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

fn run_diskann_build(opts: DiskAnnCliCommand) -> anyhow::Result<()> {
    let config = Config::from_env()?;
    let params = diskann_params_from_cli(&opts, &config);
    let engine = Engine::new(config.clone())?;
    engine
        .vector_build_disk_index(&opts.collection, params.clone())
        .map_err(|err| anyhow::anyhow!("diskann build falló: {err}"))?;
    let status = engine
        .vector_disk_index_status(&opts.collection)
        .map_err(|err| anyhow::anyhow!("diskann status error: {err}"))?;
    println!(
        "DiskANN listo para `{}`: available={} last_built_ms={} degree={} search_list={} files={:?}",
        opts.collection, status.available, status.last_built_ms, status.params.max_degree, status.params.search_list_size, status.graph_files
    );
    Ok(())
}

fn run_diskann_tune(opts: DiskAnnCliCommand) -> anyhow::Result<()> {
    let config = Config::from_env()?;
    let params = diskann_params_from_cli(&opts, &config);
    let engine = Engine::new(config.clone())?;
    let applied = engine
        .vector_update_disk_index_params(&opts.collection, params.clone())
        .map_err(|err| anyhow::anyhow!("diskann tune falló: {err}"))?;
    println!(
        "DiskANN tuning aplicado a `{}`: degree={} search_list={} build_threads={}",
        opts.collection, applied.max_degree, applied.search_list_size, applied.build_threads
    );
    Ok(())
}

fn run_diskann_status(collection: String) -> anyhow::Result<()> {
    let config = Config::from_env()?;
    let engine = Engine::new(config)?;
    let status = engine
        .vector_disk_index_status(&collection)
        .map_err(|err| anyhow::anyhow!("diskann status falló: {err}"))?;
    println!(
        "DiskANN status `{}` => available={} last_built_ms={} degree={} search_list={} files={:?}",
        collection,
        status.available,
        status.last_built_ms,
        status.params.max_degree,
        status.params.search_list_size,
        status.graph_files
    );
    Ok(())
}

fn diskann_params_from_cli(opts: &DiskAnnCliCommand, config: &Config) -> DiskAnnBuildParams {
    DiskAnnBuildParams {
        max_degree: opts.max_degree.unwrap_or(config.diskann_max_degree).max(4),
        build_threads: opts
            .build_threads
            .unwrap_or(config.diskann_build_threads)
            .max(1),
        search_list_size: opts
            .search_list_size
            .unwrap_or(config.diskann_search_list_size)
            .max(8),
    }
    .sanitized()
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
    DiskAnnBuild(DiskAnnCliCommand),
    DiskAnnTune(DiskAnnCliCommand),
    DiskAnnStatus { collection: String },
}

#[derive(Clone, Debug)]
struct DiskAnnCliCommand {
    collection: String,
    max_degree: Option<usize>,
    build_threads: Option<usize>,
    search_list_size: Option<usize>,
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
            "diskann" => {
                return parse_diskann_command(&args[2..]);
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

fn parse_diskann_command(args: &[String]) -> anyhow::Result<Command> {
    if args.is_empty() {
        anyhow::bail!("diskann requiere subcomando (build|status|tune)");
    }
    match args[0].as_str() {
        "build" => Ok(Command::DiskAnnBuild(parse_diskann_cli(&args[1..])?)),
        "tune" => Ok(Command::DiskAnnTune(parse_diskann_cli(&args[1..])?)),
        "status" => {
            let collection = parse_diskann_collection(&args[1..])?;
            Ok(Command::DiskAnnStatus { collection })
        }
        other => anyhow::bail!("subcomando diskann desconocido `{other}` (usa build|status|tune)"),
    }
}

fn parse_diskann_cli(args: &[String]) -> anyhow::Result<DiskAnnCliCommand> {
    let mut collection = None;
    let mut max_degree = None;
    let mut build_threads = None;
    let mut search_list_size = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--collection" if collection.is_none() => {
                let value = iter
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--collection requiere un valor"))?;
                collection = Some(value.to_string());
            }
            "--max-degree" => {
                let raw = iter
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--max-degree requiere un valor"))?;
                max_degree = Some(raw.parse()?);
            }
            "--build-threads" => {
                let raw = iter
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--build-threads requiere un valor"))?;
                build_threads = Some(raw.parse()?);
            }
            "--search-list" => {
                let raw = iter
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--search-list requiere un valor"))?;
                search_list_size = Some(raw.parse()?);
            }
            other => {
                anyhow::bail!("bandera diskann desconocida `{other}`");
            }
        }
    }
    let collection =
        collection.ok_or_else(|| anyhow::anyhow!("diskann requiere --collection <nombre>"))?;
    Ok(DiskAnnCliCommand {
        collection,
        max_degree,
        build_threads,
        search_list_size,
    })
}

fn parse_diskann_collection(args: &[String]) -> anyhow::Result<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--collection" {
            let value = iter
                .next()
                .ok_or_else(|| anyhow::anyhow!("--collection requiere un valor"))?;
            return Ok(value.to_string());
        } else {
            anyhow::bail!("bandera desconocida `{arg}`; usa --collection <nombre>");
        }
    }
    anyhow::bail!("diskann status requiere --collection <nombre>");
}
