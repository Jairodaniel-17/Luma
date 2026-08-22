use luma::config::Config;
use tracing::info;

mod cli;
mod diskann;
mod server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // The command and config are read before tracing is installed: the OTLP
    // endpoint lives in the config, and installing a subscriber twice is a
    // panic. The cost is that a config error is reported without the pretty
    // formatting, which is a fair trade for not guessing the endpoint.
    let command = cli::parse_command()?;
    let config = Config::load()?;
    let mut telemetry = luma::telemetry::init(config.otel_endpoint.as_deref(), "luma");

    info!(
        "Starting Luma (powered by RustKissVDB) v{}",
        env!("CARGO_PKG_VERSION")
    );

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
        cli::Command::Backup { verify } => {
            cli::run_backup(&config, verify)?;
        }
        cli::Command::Restore { path } => {
            cli::run_restore(&config, path)?;
        }
        cli::Command::Promote => {
            cli::run_promote(&config)?;
        }
        cli::Command::Demote => {
            cli::run_demote(&config)?;
        }
        cli::Command::Role => {
            cli::run_role(&config)?;
        }
        cli::Command::Help => {
            println!("{}", cli::help_text());
        }
    }

    // Flush pending spans before the process ends: the last spans before a

    // shutdown are the interesting ones if it is shutting down badly.

    telemetry.shutdown();

    Ok(())
}
