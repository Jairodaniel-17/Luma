use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::vector::VectorStore;

#[derive(Debug)]
pub enum Command {
    Serve,
    Vacuum { collection: String },
    DiskAnnBuild(crate::diskann::DiskAnnCli),
    DiskAnnTune(crate::diskann::DiskAnnCli),
    DiskAnnStatus { collection: String },
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
        _ => Ok(Command::Serve),
    }
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
