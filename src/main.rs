use anyhow::Result;
use scalper::{config::Config, ui::App};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (to file, not terminal since we use TUI)
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tracing::info!("Starting Kraken Scalper Bot");

    // Load configuration
    let config = Config::load()?;
    tracing::info!("Loaded config for pairs: {:?}", config.trading.pairs);

    // Run the application
    let mut app = App::new(config).await?;
    app.run().await?;

    Ok(())
}
