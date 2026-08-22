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
        "    vacuum --collection <c>  Compacta una colección vectorial",
        "    diskann <...>            Construye/ajusta/consulta el índice DiskANN",
        "    help                     Esto",
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
            "serve", "backup", "restore", "promote", "demote", "role", "vacuum", "diskann", "help",
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
        ] {
            let owned: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
            assert!(parse_args(&owned).is_ok(), "{argv:?} should parse");
        }
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
