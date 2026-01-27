use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::vector::index::DiskAnnBuildParams;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
pub struct DiskAnnCli {
    pub collection: String,
    pub max_degree: Option<usize>,
    pub build_threads: Option<usize>,
    pub search_list_size: Option<usize>,
}

pub fn parse_diskann(args: &[String]) -> anyhow::Result<crate::cli::Command> {
    if args.is_empty() {
        anyhow::bail!("diskann requiere subcomando");
    }

    match args[0].as_str() {
        "build" => Ok(crate::cli::Command::DiskAnnBuild(parse_opts(&args[1..])?)),
        "tune" => Ok(crate::cli::Command::DiskAnnTune(parse_opts(&args[1..])?)),
        "status" => {
            let collection = parse_collection(&args[1..])?;
            Ok(crate::cli::Command::DiskAnnStatus { collection })
        }
        _ => anyhow::bail!("subcomando diskann inválido"),
    }
}

fn parse_opts(args: &[String]) -> anyhow::Result<DiskAnnCli> {
    let mut iter = args.iter();
    let mut opts = DiskAnnCli {
        collection: String::new(),
        max_degree: None,
        build_threads: None,
        search_list_size: None,
    };

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--collection" => opts.collection = iter.next().unwrap().to_string(),
            "--max-degree" => opts.max_degree = Some(iter.next().unwrap().parse()?),
            "--build-threads" => opts.build_threads = Some(iter.next().unwrap().parse()?),
            "--search-list" => opts.search_list_size = Some(iter.next().unwrap().parse()?),
            other => anyhow::bail!("flag desconocida `{other}`"),
        }
    }

    if opts.collection.is_empty() {
        anyhow::bail!("diskann requiere --collection");
    }

    Ok(opts)
}

fn parse_collection(args: &[String]) -> anyhow::Result<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--collection" {
            return Ok(iter.next().unwrap().to_string());
        }
    }
    anyhow::bail!("status requiere --collection")
}

pub fn run_build(config: &Config, opts: DiskAnnCli) -> anyhow::Result<()> {
    let engine = Engine::new(config.clone(), CancellationToken::new())?;
    let params = params_from_cli(&opts, config);

    engine.vector_build_disk_index(&opts.collection, params)?;
    Ok(())
}

pub fn run_tune(config: &Config, opts: DiskAnnCli) -> anyhow::Result<()> {
    let engine = Engine::new(config.clone(), CancellationToken::new())?;
    let params = params_from_cli(&opts, config);

    engine.vector_update_disk_index_params(&opts.collection, params)?;
    Ok(())
}

pub fn run_status(config: &Config, collection: String) -> anyhow::Result<()> {
    let engine = Engine::new(config.clone(), CancellationToken::new())?;
    let status = engine.vector_disk_index_status(&collection)?;

    println!("{status:#?}");
    Ok(())
}

fn params_from_cli(opts: &DiskAnnCli, config: &Config) -> DiskAnnBuildParams {
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
