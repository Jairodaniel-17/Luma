use luma::config::Config;
use luma::vector::VectorStore;

#[derive(Debug)]
pub enum Command {
    Serve,
    Vacuum { collection: String },
    DiskAnnBuild(crate::diskann::DiskAnnCli),
    DiskAnnTune(crate::diskann::DiskAnnCli),
    DiskAnnStatus { collection: String },
    Backup,
    Restore { path: String },
}

pub fn parse_command() -> anyhow::Result<Command> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Ok(Command::Serve);
    }

    match args[1].as_str() {
        "serve" => Ok(Command::Serve),
        "vacuum" => parse_vacuum(&args[2..]),
        "diskann" => crate::diskann::parse_diskann(&args[2..]),
        "backup" => Ok(Command::Backup),
        "restore" => {
            let path = args.get(2).cloned().ok_or_else(|| {
                anyhow::anyhow!("restore requiere <path> al directorio de backup")
            })?;
            Ok(Command::Restore { path })
        }
        _ => Ok(Command::Serve),
    }
}

pub fn run_backup(config: &Config) -> anyhow::Result<()> {
    let dest = luma::backup::run_backup(config)?;
    println!("Backup creado en {}", dest.display());
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
