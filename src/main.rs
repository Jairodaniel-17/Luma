use rust_kiss_vdb::config::Config;
use tracing_subscriber::EnvFilter;

mod cli;
mod server;
mod diskann;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();

    let command = cli::parse_command()?;
    let config = Config::from_env()?;

    match command {
        cli::Command::Serve => {
            server::run(config).await?;
        }
        cli::Command::Vacuum { collection } => {
            cli::run_vacuum(&config, collection)?;
        }
        cli::Command::DiskAnnBuild(opts) => {
            diskann::run_build(&config, opts)?;
        }
        cli::Command::DiskAnnTune(opts) => {
            diskann::run_tune(&config, opts)?;
        }
        cli::Command::DiskAnnStatus { collection } => {
            diskann::run_status(&config, collection)?;
        }
    }

    Ok(())
}

fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();
}
