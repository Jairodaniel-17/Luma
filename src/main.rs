use luma::config::Config;
use tracing::info;

mod cli;
mod server;
mod diskann;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting Luma (powered by RustKissVDB) v{}", env!("CARGO_PKG_VERSION"));


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
