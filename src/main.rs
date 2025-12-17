use anyhow::Result;
use scalper::{config::Config, ui::App};
use std::fs::OpenOptions;
use std::sync::Mutex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to file (not terminal since we use TUI)
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("scalper.log")
        .expect("Failed to open log file");

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(Mutex::new(log_file))
                .with_ansi(false),
        )
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!("Starting Kraken Scalper Bot");

    // Load configuration
    let config = Config::load()?;
    tracing::info!(
        "Loaded config for {} crypto pairs, {} stock pairs",
        config.trading.crypto_pairs.len(),
        config.trading.stock_pairs.len()
    );

    // Run the application
    let mut app = App::new(config).await?;
    app.run().await?;

    Ok(())
}
