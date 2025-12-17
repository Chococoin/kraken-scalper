//! Manual HuggingFace sync utility
//!
//! Upload existing data files to HuggingFace.
//! Usage: HF_TOKEN=xxx cargo run --bin hf_sync

use anyhow::Result;
use scalper::config::Config;
use scalper::storage::HfUploader;
use std::path::Path;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    info!("HuggingFace Manual Sync");

    // Load configuration
    let config = Config::load()?;

    if !config.huggingface.enabled {
        println!("HuggingFace upload is disabled in config.");
        println!("Set [huggingface] enabled = true in config/default.toml");
        return Ok(());
    }

    // Get token
    let token = match std::env::var("HF_TOKEN") {
        Ok(t) => t,
        Err(_) => {
            eprintln!("Error: HF_TOKEN environment variable required");
            eprintln!("Usage: HF_TOKEN=xxx cargo run --bin hf_sync");
            std::process::exit(1);
        }
    };

    // Check Python dependencies
    info!("Checking Python dependencies...");
    HfUploader::check_dependencies()?;

    info!("Repo: {}", config.huggingface.repo_id);
    info!("Data dir: {}", config.recording.data_dir);

    // Create uploader
    let mut uploader = HfUploader::new(
        &config.huggingface.repo_id,
        Path::new(&config.recording.data_dir),
        &token,
        0,
    )?;

    // Sync
    info!("Starting sync...");
    let _uploaded = uploader.sync().await?;

    // Final stats
    let (files, bytes) = uploader.stats();
    println!();
    println!("========================================");
    println!("Sync complete!");
    println!("========================================");
    println!("Files uploaded: {}", files);
    println!("Total size: {:.2} MB", bytes as f64 / 1_000_000.0);
    println!();
    println!(
        "Dataset URL: https://huggingface.co/datasets/{}",
        config.huggingface.repo_id
    );

    Ok(())
}
