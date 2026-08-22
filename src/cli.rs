use luma::config::Config;
use luma::vector::VectorStore;

#[derive(Debug)]
pub enum Command {
    Serve,
    Vacuum {
        collection: String,
    },
    DiskAnnBuild(crate::diskann::DiskAnnCli),
    DiskAnnTune(crate::diskann::DiskAnnCli),
    DiskAnnStatus {
        collection: String,
    },
    Backup {
        verify: bool,
    },
    Restore {
        path: String,
    },
    /// Turn a replica back into a primary.
    ///
    /// The plan promised promotion through `luma promote` and the function
    /// existed in `replica.rs`, but the CLI never grew the command — so the
    /// replica could be built and could not be promoted, which is the only
    /// thing it is for.
    Promote,
    /// Turn a primary into a replica.
    ///
    /// The pair of `promote`, and it was missing too: `mark_replica` existed in
    /// `replica.rs` and only the tests ever called it, so a replica could not be
    /// created from outside the crate. A replica nobody can create is not a
    /// feature.
    Demote,
    /// Print which role a data directory holds. Read-only.
    Role,
    /// Follow a Postgres table into a Luma collection.
    ///
    /// `luma connect postgres <config.toml>`. The source system is part of the
    /// command rather than inferred from the config, so a second connector —
    /// MySQL, say — is a new word here and not a silently different meaning for
    /// the same one.
    Connect {
        source: String,
        config_path: String,
        /// Run one bounded pass and report, instead of following forever.
        /// What a cron job and a test both want.
        once: bool,
        /// Skip the initial COPY even if the configuration asks for it. For
        /// resuming a connector whose backfill already finished.
        no_backfill: bool,
    },
    Help,
}

pub fn parse_command() -> anyhow::Result<Command> {
    parse_args(&std::env::args().collect::<Vec<String>>())
}

/// The parser proper, taking its arguments rather than reading the process.
///
/// Split out so it can be tested at all: reading `std::env::args()` inside the
/// only entry point meant the dispatch table was unreachable from a test, which
/// is how it came to silently start a server on a mistyped subcommand.
pub fn parse_args(args: &[String]) -> anyhow::Result<Command> {
    if args.len() < 2 {
        return Ok(Command::Serve);
    }

    let first = args[1].as_str();
    match first {
        "help" | "--help" | "-h" => Ok(Command::Help),
        "serve" => Ok(Command::Serve),
        "vacuum" => parse_vacuum(&args[2..]),
        "diskann" => crate::diskann::parse_diskann(&args[2..]),
        "backup" => Ok(Command::Backup {
            // A backup nobody has restored is a hypothesis, so `--verify`
            // reads it back and checks it against its own manifest.
            verify: args.iter().any(|a| a == "--verify"),
        }),
        "restore" => {
            let path = args.get(2).cloned().ok_or_else(|| {
                anyhow::anyhow!("restore requiere <path> al directorio de backup")
            })?;
            Ok(Command::Restore { path })
        }
        "connect" => parse_connect(&args[2..]),
        "promote" => Ok(Command::Promote),
        "demote" => Ok(Command::Demote),
        "role" => Ok(Command::Role),
        // A leading flag means "serve, configured by flags", which is how the
        // server is normally started.
        _ if first.starts_with('-') => Ok(Command::Serve),
        // Anything else is a mistyped subcommand, and it used to *start a
        // server*. `luma backupp` bringing up a listener on the production port
        // is not a typo an operator finds out about gently.
        other => anyhow::bail!(
            "subcomando desconocido `{other}`. Ejecuta `luma help` para ver los disponibles."
        ),
    }
}

/// `connect <source> <config.toml> [--once] [--no-backfill]`
///
/// The source is required and checked here rather than accepted and validated
/// later: `luma connect config.toml` would otherwise read the config path as a
/// source name and complain about something the operator did not type.
fn parse_connect(args: &[String]) -> anyhow::Result<Command> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    let source = positional.first().map(|s| s.to_string()).ok_or_else(|| {
        anyhow::anyhow!("connect requiere un origen: `luma connect postgres <config.toml>`")
    })?;
    if source != "postgres" {
        anyhow::bail!("origen `{source}` no soportado; el único disponible es `postgres`");
    }
    let config_path = positional
        .get(1)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("connect requiere la ruta del fichero de conector: `luma connect postgres <config.toml>`"))?;
    Ok(Command::Connect {
        source,
        config_path,
        once: args.iter().any(|a| a == "--once"),
        no_backfill: args.iter().any(|a| a == "--no-backfill"),
    })
}

/// Usage text.
///
/// Written out rather than derived from a parser library: the parser here is
/// twenty lines and adding a dependency to print one screen is not a trade
/// worth making. The cost is that this text and the `match` above can drift,
/// which is what `cli::tests` checks.
pub fn help_text() -> String {
    [
        "luma — motor de datos convergente",
        "",
        "USO:",
        "    luma [SUBCOMANDO] [OPCIONES]",
        "",
        "SUBCOMANDOS:",
        "    serve                    Arranca el servidor (por defecto si no se indica ninguno)",
        "    backup [--verify]        Copia el estado persistente; --verify lo restaura y lo comprueba",
        "    restore <path>           Restaura desde un directorio de backup",
        "    promote                  Convierte una réplica en primario (quita el marcador REPLICA)",
        "    demote                   Convierte un primario en réplica de solo lectura",
        "    role                     Dice si el data_dir es `primary` o `replica`",
        "    connect postgres <cfg>   Sigue tablas de Postgres hacia colecciones de Luma",
        "    vacuum --collection <c>  Compacta una colección vectorial",
        "    diskann <...>            Construye/ajusta/consulta el índice DiskANN",
        "    help                     Esto",
        "",
        "OPCIONES DE CONNECT:",
        "    --once                   Una pasada acotada y un informe, en vez de seguir siempre",
        "    --no-backfill            Omite el COPY inicial (para reanudar uno ya terminado)",
        "",
        "OPCIONES DE SERVE (también configurables en luma.toml):",
        "    --port <n>               Puerto HTTP",
        "    --bind <addr>            Dirección de escucha",
        "    --unsafe-bind            Atajo de --bind 0.0.0.0; lo dice en el nombre",
        "    --data <path>            Directorio de datos",
        "",
        "La configuración se resuelve: flags > entorno > luma.toml > valores por defecto.",
    ]
    .join("\n")
}

pub async fn run_promote(config: &Config) -> anyhow::Result<()> {
    let data_dir = config
        .data_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no hay data_dir configurado"))?;
    let path = std::path::Path::new(&data_dir);
    luma::replica::promote(path)?;

    // Claiming the next epoch is what actually stops the old primary. The
    // marker only changes this node; the epoch changes what the *prefix* says,
    // and the old primary reads it on its next shipping pass.
    //
    // Done after removing the marker, so a failure here leaves a node that is a
    // primary locally and has fenced nobody — visible, and fixed by running the
    // command again. The reverse order would fence the old primary while this
    // one stayed read-only, which is an outage with no writer at all.
    match luma::backup_remote::store_from_config(config) {
        Ok(Some(target)) => {
            let epoch = luma::fencing::claim_next_epoch(&target.store, &target.prefix).await?;
            luma::fencing::set_local_epoch(path, epoch)?;
            println!(
                "Promovido a primario: {data_dir}\n\
                 Epoch {epoch} reclamada en {}: el primario anterior dejará de \
                 enviar en su próxima pasada.",
                target.prefix
            );
        }
        Ok(None) => {
            println!(
                "Promovido a primario: {data_dir}\n\
                 Sin destino remoto configurado, así que no hay epoch que reclamar: \
                 asegúrate tú de que el primario anterior está detenido."
            );
        }
        Err(e) => {
            // The local promotion already happened, so this is a warning rather
            // than an error: returning `Err` would suggest nothing had changed.
            println!(
                "Promovido a primario: {data_dir}\n\
                 ATENCIÓN: no se pudo reclamar la epoch remota ({e}). El primario \
                 anterior NO está cercado; detenlo a mano."
            );
        }
    }
    Ok(())
}

pub fn run_demote(config: &Config) -> anyhow::Result<()> {
    let data_dir = config
        .data_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no hay data_dir configurado"))?;
    let path = std::path::Path::new(&data_dir);
    // Refuse rather than report success on something already a replica, for the
    // same reason `promote` does: mid-incident, "done" on a no-op is worse than
    // an error.
    if luma::replica::Role::from_data_dir(path).is_read_only() {
        anyhow::bail!("{data_dir} ya es una réplica");
    }
    luma::replica::mark_replica(path)?;
    println!(
        "Marcado como réplica de solo lectura: {data_dir}
         Las escrituras se rechazan. `luma promote` lo revierte."
    );
    Ok(())
}

pub fn run_role(config: &Config) -> anyhow::Result<()> {
    let data_dir = config
        .data_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no hay data_dir configurado"))?;
    let role = luma::replica::Role::from_data_dir(std::path::Path::new(&data_dir));
    println!("{}", role.name());
    Ok(())
}

/// `luma connect postgres <config.toml>`
///
/// Runs against the same data directory the server uses, which is the point:
/// the connector writes into the collections `/v1/db` and `/v1/memory` read.
/// Running it as a separate process rather than a task inside `serve` is
/// deliberate — a connector that falls over should not take the API with it,
/// and one that needs restarting should not need the server restarted.
pub async fn run_connect(
    config: &Config,
    config_path: &str,
    once: bool,
    no_backfill: bool,
) -> anyhow::Result<()> {
    use anyhow::Context;
    use luma::pgcdc::{Connector, ConnectorConfig};
    use std::sync::Arc;

    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("no se pudo leer {config_path}"))?;
    let mut connector_config = ConnectorConfig::from_toml(&text)?;
    if no_backfill {
        connector_config.backfill = false;
    }

    let shutdown = tokio_util::sync::CancellationToken::new();
    let engine = Arc::new(luma::engine::Engine::new(config.clone(), shutdown.clone())?);
    let sqlite = config
        .data_dir
        .as_deref()
        .map(|dir| -> anyhow::Result<Arc<luma::sqlite::SqliteService>> {
            let path = std::path::Path::new(dir).join("sqlite").join("rustkiss.db");
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Ok(Arc::new(luma::sqlite::SqliteService::new(path)?))
        })
        .transpose()?;
    let embeddings = luma::engine::embeddings::EmbeddingHandle::new(
        luma::engine::embeddings::EmbeddingClient::from_config(config, Some(engine.metrics())),
    );
    let hub = Arc::new(luma::engine::hub::LumaDatabase::new(
        engine.clone(),
        sqlite,
        embeddings,
        luma::engine::chunking::ChunkingEngine::default(),
        config.clone(),
    ));

    let name = connector_config.name.clone();
    let wants_backfill = connector_config.backfill;
    let connector = Connector::new(connector_config, hub)?;

    // The slot before the copy, always. A backfill taken first would miss every
    // change made between the copy and the slot's creation, and those rows are
    // gone with nothing to say they were expected.
    let fresh_slot = connector.prepare().await?;
    match fresh_slot {
        Some(lsn) => println!(
            "slot creado; punto consistente {}",
            luma::pgcdc::pgoutput::format_lsn(lsn)
        ),
        None => println!("slot existente; se reanuda desde donde quedó"),
    }

    if wants_backfill && fresh_slot.is_some() {
        let rows = connector.backfill().await?;
        println!("backfill: {rows} filas");
    } else if wants_backfill {
        println!("backfill omitido: el slot ya existía, así que el stream ya cubre estas filas");
    }

    if once {
        let report = connector
            .stream_once(std::time::Duration::from_secs(30), u64::MAX)
            .await?;
        println!(
            "{name}: +{} ~{} -{} (omitidas {}, ya aplicadas {}) en {}",
            report.inserted,
            report.updated,
            report.deleted,
            report.skipped,
            report.already_applied,
            luma::pgcdc::pgoutput::format_lsn(report.last_lsn)
        );
        if !report.truncated_tables.is_empty() {
            println!(
                "ATENCIÓN: llegó un TRUNCATE de {:?}. La colección derivada quedó obsoleta y                  Luma no la vacía sola.",
                report.truncated_tables
            );
        }
        return Ok(());
    }

    println!("{name}: siguiendo el stream (Ctrl-C para parar)");
    loop {
        // Each pass reconnects. A connector that reconnects on a schedule
        // recovers from a network blip, a primary failover, and a Postgres
        // restart with the same code path, instead of three.
        match connector
            .stream_once(std::time::Duration::from_secs(60), u64::MAX)
            .await
        {
            Ok(report) if report.applied() > 0 => tracing::info!(
                connector = %name,
                inserted = report.inserted,
                updated = report.updated,
                deleted = report.deleted,
                lsn = %luma::pgcdc::pgoutput::format_lsn(report.last_lsn),
                "applied"
            ),
            Ok(_) => {}
            Err(e) => {
                tracing::error!(connector = %name, error = %e, "the connector pass failed; retrying");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
    }
}

pub fn run_backup(config: &Config, verify: bool) -> anyhow::Result<()> {
    let dest = luma::backup::run_backup(config)?;
    println!("Backup creado en {}", dest.display());
    if verify {
        let manifest = luma::backup::verify(&dest)?;
        println!(
            "Verificado: sqlite={} snapshot={} wal={} colecciones={} blobs={} mensajes={}",
            manifest.sqlite,
            manifest.snapshot,
            manifest.wal_segments,
            manifest.vector_collections,
            manifest.blob_files,
            manifest.queue_files
        );
    }
    Ok(())
}

pub fn run_restore(config: &Config, path: String) -> anyhow::Result<()> {
    luma::backup::restore(config, &path)?;
    println!("Restauración desde `{path}` completada.");
    Ok(())
}

fn parse_vacuum(args: &[String]) -> anyhow::Result<Command> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--collection" {
            let value = iter
                .next()
                .ok_or_else(|| anyhow::anyhow!("--collection requerido"))?;
            return Ok(Command::Vacuum {
                collection: value.to_string(),
            });
        }
    }
    anyhow::bail!("vacuum requiere --collection")
}

pub fn run_vacuum(config: &Config, collection: String) -> anyhow::Result<()> {
    let dir = config
        .data_dir
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DATA_DIR requerido para vacuum"))?;

    let store = VectorStore::open(dir)?;
    store.vacuum_collection(&collection)?;

    println!("Colección `{collection}` compactada.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every subcommand the parser accepts must appear in the help.
    ///
    /// The help text is written by hand, so it can drift from the `match`
    /// that dispatches. A help screen that omits a command is a command
    /// nobody finds, which for `promote` means an operator mid-incident
    /// reading that promotion is manual and having no idea how.
    ///
    /// The list is duplicated here on purpose: it is the third copy, and the
    /// test exists precisely to make the three agree. Deriving it from the
    /// parser would test nothing.
    #[test]
    fn the_help_mentions_every_subcommand() {
        const SUBCOMMANDS: &[&str] = &[
            "serve", "backup", "restore", "promote", "demote", "role", "vacuum", "diskann",
            "connect", "help",
        ];
        let help = help_text();
        let missing: Vec<&&str> = SUBCOMMANDS
            .iter()
            .filter(|name| !help.contains(**name))
            .collect();
        assert!(
            missing.is_empty(),
            "these subcommands are not in the help text: {missing:?}"
        );
    }

    /// And the reverse: the parser must accept everything the help promises.
    #[test]
    fn the_parser_accepts_every_subcommand_the_help_lists() {
        // `restore` needs an argument and `vacuum`/`diskann` need flags, so
        // they are exercised with the minimum that should parse.
        for argv in [
            vec!["luma", "serve"],
            vec!["luma", "backup"],
            vec!["luma", "backup", "--verify"],
            vec!["luma", "restore", "/tmp/backup"],
            vec!["luma", "promote"],
            vec!["luma", "demote"],
            vec!["luma", "role"],
            vec!["luma", "help"],
            vec!["luma", "--help"],
            vec!["luma", "-h"],
            vec!["luma", "vacuum", "--collection", "c"],
            vec!["luma", "connect", "postgres", "erp.toml"],
            vec![
                "luma",
                "connect",
                "postgres",
                "erp.toml",
                "--once",
                "--no-backfill",
            ],
        ] {
            let owned: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
            assert!(parse_args(&owned).is_ok(), "{argv:?} should parse");
        }
    }

    #[test]
    fn connect_requires_a_source_and_a_config_and_says_which_is_missing() {
        // `luma connect erp.toml` would otherwise read the config path as a
        // source name and complain about `erp.toml` not being supported, which
        // sends the operator looking for a spelling mistake they did not make.
        let parse =
            |args: &[&str]| parse_args(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert!(parse(&["luma", "connect"]).is_err());

        let err = parse(&["luma", "connect", "postgres"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("fichero de conector"), "{err}");

        let err = parse(&["luma", "connect", "mysql", "cfg.toml"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("mysql"), "{err}");
        assert!(err.contains("postgres"), "{err}");
    }

    #[test]
    fn connect_reads_its_flags_in_any_position() {
        let parse = |args: &[&str]| {
            parse_args(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
        };
        let Command::Connect {
            source,
            config_path,
            once,
            no_backfill,
        } = parse(&["luma", "connect", "--once", "postgres", "erp.toml"])
        else {
            panic!("expected a connect command");
        };
        assert_eq!(source, "postgres");
        // The flag before the positionals must not be taken for one of them.
        assert_eq!(config_path, "erp.toml");
        assert!(once);
        assert!(!no_backfill);
    }

    #[test]
    fn a_mistyped_subcommand_is_an_error_and_not_a_running_server() {
        // It used to fall through to `serve`. `luma backupp` would bring up a
        // listener on the production port, which is not how an operator wants
        // to find out about a typo.
        for argv in ["backupp", "promot", "restor", "nonsense"] {
            let owned = vec!["luma".to_string(), argv.to_string()];
            let outcome = parse_args(&owned);
            assert!(outcome.is_err(), "`{argv}` must be refused, not served");
            let message = outcome.unwrap_err().to_string();
            assert!(
                message.contains(argv) && message.contains("luma help"),
                "the error must name the typo and point somewhere useful: {message}"
            );
        }
    }

    #[test]
    fn a_bare_invocation_and_a_leading_flag_both_serve() {
        // `luma` on its own is the documented way to start it, and
        // `luma --port 9000` is how it is started with flags. Neither is a
        // subcommand, and neither should be refused.
        assert!(matches!(
            parse_args(&["luma".to_string()]).unwrap(),
            Command::Serve
        ));
        assert!(matches!(
            parse_args(&["luma".to_string(), "--port".to_string(), "9000".to_string()]).unwrap(),
            Command::Serve
        ));
    }

    #[test]
    fn restore_without_a_path_is_refused() {
        // Restoring from an unspecified directory is not something to guess at.
        let outcome = parse_args(&["luma".to_string(), "restore".to_string()]);
        assert!(outcome.is_err());
    }
}
