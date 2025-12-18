//! Data recorder daemon
//!
//! Captures market data from Kraken WebSocket and saves to Parquet files.
//! Run with: cargo run --bin recorder
//!
//! Stops gracefully on SIGINT (Ctrl+C) or SIGTERM.

use anyhow::Result;
use chrono::{Datelike, Timelike, Utc, Weekday};
use scalper::api::{KrakenWebSocket, MarketEvent};
use scalper::config::Config;
use scalper::storage::{DataRecorder, HfUploader};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Check if US stock market (NYSE) is currently open
/// NYSE hours: 9:30 AM - 4:00 PM Eastern Time, Monday-Friday
fn is_us_market_open() -> bool {
    // Get current time in Eastern Time (US)
    // Note: This doesn't handle DST transitions perfectly, but is close enough
    // Eastern Time is UTC-5 (EST) or UTC-4 (EDT)
    let now_utc = Utc::now();

    // Approximate Eastern Time (using UTC-5 for simplicity)
    // In summer it's UTC-4, in winter UTC-5
    // We'll use a conservative approach: check both possible ET hours
    let month = now_utc.month();
    let offset_hours = if month >= 3 && month <= 11 { 4 } else { 5 }; // Rough DST approximation

    let eastern_hour = if now_utc.hour() >= offset_hours {
        now_utc.hour() - offset_hours
    } else {
        24 + now_utc.hour() - offset_hours
    };
    let eastern_minute = now_utc.minute();

    // Check if it's a weekday
    let weekday = now_utc.weekday();
    let is_weekday = !matches!(weekday, Weekday::Sat | Weekday::Sun);

    // Market hours: 9:30 AM - 4:00 PM ET
    // Convert to minutes since midnight for easier comparison
    let current_minutes = eastern_hour * 60 + eastern_minute;
    let market_open = 9 * 60 + 30;  // 9:30 AM = 570 minutes
    let market_close = 16 * 60;      // 4:00 PM = 960 minutes

    let is_market_hours = current_minutes >= market_open && current_minutes < market_close;

    is_weekday && is_market_hours
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("scalper=info".parse().unwrap())
                .add_directive("recorder=info".parse().unwrap()),
        )
        .init();

    info!("Starting Kraken Data Recorder Daemon");

    // Load configuration
    let config = Config::load()?;
    info!(
        "Recording {} crypto pairs ({}s interval), {} stock pairs ({}s interval)",
        config.trading.crypto_pairs.len(),
        config.recording.crypto_sample_interval_secs,
        config.trading.stock_pairs.len(),
        config.recording.stock_sample_interval_secs,
    );

    if !config.recording.enabled {
        warn!("Recording is disabled in config. Enable it to capture data.");
        return Ok(());
    }

    // Create recorder
    let mut recorder = DataRecorder::new(&config);

    // Create HuggingFace uploader if enabled
    let mut hf_uploader = if config.huggingface.enabled {
        match std::env::var("HF_TOKEN") {
            Ok(token) => {
                info!(
                    "HuggingFace upload enabled for repo: {}",
                    config.huggingface.repo_id
                );
                match HfUploader::new(
                    &config.huggingface.repo_id,
                    Path::new(&config.recording.data_dir),
                    &token,
                    config.huggingface.upload_delay_hours,
                ) {
                    Ok(uploader) => Some(uploader),
                    Err(e) => {
                        error!("Failed to create HuggingFace uploader: {}", e);
                        None
                    }
                }
            }
            Err(_) => {
                error!("HuggingFace upload enabled but HF_TOKEN not set!");
                None
            }
        }
    } else {
        None
    };
    let mut last_hf_sync = Instant::now();
    let hf_upload_interval = Duration::from_secs(config.huggingface.upload_interval_secs);

    // Create event channel
    let (event_tx, mut event_rx) = mpsc::channel::<MarketEvent>(1000);

    // Create WebSocket
    let ws = KrakenWebSocket::new(
        config.kraken.clone(),
        config.trading.crypto_pairs.clone(),
        config.trading.stock_pairs.clone(),
        config.recording.crypto_book_depth,
        config.recording.stock_book_depth,
        config.ui.chart_candles,
        config.ui.chart_interval(),
    );

    // Get references to WebSocket stores
    let ws_tickers = Arc::clone(&ws.tickers);
    let ws_orderbooks = Arc::clone(&ws.orderbooks);
    let ws_candles = Arc::clone(&ws.candles);

    // Spawn WebSocket connection task
    let ws_event_tx = event_tx.clone();
    tokio::spawn(async move {
        loop {
            info!("Connecting to Kraken WebSocket...");
            if let Err(e) = ws.connect(ws_event_tx.clone()).await {
                error!("WebSocket error: {}", e);
            }
            warn!("WebSocket disconnected, reconnecting in 5s...");
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });

    // Setup shutdown signal
    let mut shutdown = false;

    info!("Recorder daemon running. Press Ctrl+C to stop.");

    // Main loop
    loop {
        // Check for shutdown signal
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Shutdown signal received");
                shutdown = true;
            }
            // Process events with timeout
            result = tokio::time::timeout(Duration::from_secs(1), event_rx.recv()) => {
                if let Ok(Some(event)) = result {
                    match event {
                        MarketEvent::Connected => {
                            info!("Connected to Kraken WebSocket");
                        }
                        MarketEvent::Disconnected => {
                            warn!("Disconnected from Kraken WebSocket");
                        }
                        MarketEvent::TradeUpdate(trade) => {
                            recorder.record_trade(
                                &trade.pair,
                                &trade.side,
                                trade.price,
                                trade.qty,
                                trade.trade_id,
                            );
                        }
                        MarketEvent::TickerUpdate(_) |
                        MarketEvent::OrderBookUpdate(_) |
                        MarketEvent::CandleUpdate(_) => {
                            // Data is stored in WebSocket internal stores
                        }
                        MarketEvent::Error(err) => {
                            error!("Market error: {}", err);
                        }
                        // AI events not relevant for recorder
                        MarketEvent::AiSignalUpdate(_) |
                        MarketEvent::AiStatsUpdate(_) |
                        MarketEvent::AiTradesUpdate(_) => {}
                    }
                }
            }
        }

        if shutdown {
            break;
        }

        // Sample crypto data
        if recorder.should_sample_crypto() {
            let mut sampled = 0;
            if let Ok(tickers) = ws_tickers.try_lock() {
                if let Ok(orderbooks) = ws_orderbooks.try_lock() {
                    if let Ok(candles) = ws_candles.try_lock() {
                        for pair in &config.trading.crypto_pairs {
                            if let Some(ticker) = tickers.get(pair) {
                                recorder.record_ticker(ticker);
                                sampled += 1;
                            }
                            if let Some(orderbook) = orderbooks.get(pair) {
                                recorder.record_orderbook(pair, orderbook);
                            }
                            if let Some(pair_candles) = candles.get(pair) {
                                if let Some(candle) = pair_candles.back() {
                                    recorder.record_ohlc(pair, candle);
                                }
                            }
                        }
                    }
                }
            }
            recorder.mark_crypto_sampled();
            if sampled > 0 {
                info!("Sampled crypto data: {} pairs", sampled);
            }
        }

        // Sample stock data (only during US market hours)
        if recorder.should_sample_stock() {
            recorder.mark_stock_sampled();

            // Only record stock data when US market is open (NYSE: 9:30 AM - 4:00 PM ET, Mon-Fri)
            if is_us_market_open() {
                let mut sampled = 0;
                let ticker_count = ws_tickers.try_lock().map(|t| t.all().count()).unwrap_or(0);
                if let Ok(tickers) = ws_tickers.try_lock() {
                    if let Ok(orderbooks) = ws_orderbooks.try_lock() {
                        if let Ok(candles) = ws_candles.try_lock() {
                            for pair in &config.trading.stock_pairs {
                                if let Some(ticker) = tickers.get(pair) {
                                    recorder.record_ticker(ticker);
                                    sampled += 1;
                                }
                                if let Some(orderbook) = orderbooks.get(pair) {
                                    recorder.record_orderbook(pair, orderbook);
                                }
                                if let Some(pair_candles) = candles.get(pair) {
                                    if let Some(candle) = pair_candles.back() {
                                        recorder.record_ohlc(pair, candle);
                                    }
                                }
                            }
                        }
                    }
                }
                if sampled > 0 {
                    info!("Sampled stock data: {} pairs (market open)", sampled);

                    // Flush immediately after stock sampling to ensure stock data is saved
                    let stats = recorder.buffer_stats();
                    info!(
                        "Flushing - crypto: {}t/{}b/{}o/{}tr, stocks: {}t/{}b/{}o/{}tr",
                        stats.0, stats.1, stats.2, stats.3,
                        stats.4, stats.5, stats.6, stats.7
                    );
                    if let Err(e) = recorder.flush() {
                        error!("Failed to flush: {}", e);
                    }
                } else if ticker_count > 0 {
                    // Only warn if we have some data but no stocks matched
                    warn!("Stock sample: 0/{} stock pairs (ticker map has {} entries, data may still be loading)",
                        config.trading.stock_pairs.len(), ticker_count);
                }
            }
        }

        // Sync to HuggingFace periodically
        if let Some(ref mut uploader) = hf_uploader {
            if last_hf_sync.elapsed() >= hf_upload_interval {
                info!("Syncing data to HuggingFace...");
                match uploader.sync().await {
                    Ok(count) => {
                        if count > 0 {
                            info!("Uploaded {} files to HuggingFace", count);
                        }
                        let (files, bytes) = uploader.stats();
                        info!(
                            "HuggingFace stats: {} files uploaded ({:.2} MB total)",
                            files,
                            bytes as f64 / 1_000_000.0
                        );
                    }
                    Err(e) => error!("HuggingFace sync failed: {}", e),
                }
                last_hf_sync = Instant::now();
            }
        }
    }

    // Final flush
    info!("Flushing remaining data...");
    let stats = recorder.buffer_stats();
    info!(
        "Final buffer stats - crypto: {}t/{}b/{}o/{}tr, stocks: {}t/{}b/{}o/{}tr",
        stats.0, stats.1, stats.2, stats.3,
        stats.4, stats.5, stats.6, stats.7
    );
    if let Err(e) = recorder.flush() {
        error!("Failed to flush on shutdown: {}", e);
    } else {
        info!("Data flushed successfully");
    }

    // Final HuggingFace sync
    if let Some(ref mut uploader) = hf_uploader {
        info!("Final sync to HuggingFace...");
        match uploader.sync().await {
            Ok(count) => {
                if count > 0 {
                    info!("Uploaded {} files in final sync", count);
                }
            }
            Err(e) => error!("Final HuggingFace sync failed: {}", e),
        }
    }

    info!("Recorder daemon stopped");
    Ok(())
}
