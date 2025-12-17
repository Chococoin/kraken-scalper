//! Data recorder daemon
//!
//! Captures market data from Kraken WebSocket and saves to Parquet files.
//! Run with: cargo run --bin recorder
//!
//! Stops gracefully on SIGINT (Ctrl+C) or SIGTERM.

use anyhow::Result;
use scalper::api::{KrakenWebSocket, MarketEvent};
use scalper::config::Config;
use scalper::storage::DataRecorder;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
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

        // Sample stock data
        if recorder.should_sample_stock() {
            let mut sampled = 0;
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
            recorder.mark_stock_sampled();
            if sampled > 0 {
                info!("Sampled stock data: {} pairs", sampled);
            }
        }

        // Flush to disk periodically
        if recorder.should_flush() {
            let stats = recorder.buffer_stats();
            info!(
                "Flushing - crypto: {}t/{}b/{}o/{}tr, stocks: {}t/{}b/{}o/{}tr",
                stats.0, stats.1, stats.2, stats.3,
                stats.4, stats.5, stats.6, stats.7
            );
            if let Err(e) = recorder.flush() {
                error!("Failed to flush: {}", e);
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

    info!("Recorder daemon stopped");
    Ok(())
}
