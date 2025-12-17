//! Data recorder daemon
//!
//! Captures market data from Kraken WebSocket and saves to Parquet files.
//! Run with: cargo run --bin recorder
//!
//! Stops gracefully on SIGINT (Ctrl+C) or SIGTERM.

use anyhow::Result;
use scalper::api::{KrakenWebSocket, MarketEvent};
use scalper::config::Config;
use scalper::data::{CandleStore, OrderBookStore, TickerStore};
use scalper::storage::DataRecorder;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("scalper=info".parse().unwrap()),
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
    let recorder = Arc::new(Mutex::new(DataRecorder::new(&config)));

    // Create shared data stores
    let tickers = Arc::new(Mutex::new(TickerStore::new()));
    let orderbooks = Arc::new(Mutex::new(OrderBookStore::new()));
    let candles = Arc::new(Mutex::new(CandleStore::new(config.ui.chart_candles)));

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

    // Clone references for WebSocket task
    let ws_tickers = Arc::clone(&ws.tickers);
    let ws_orderbooks = Arc::clone(&ws.orderbooks);
    let ws_candles = Arc::clone(&ws.candles);

    // Spawn WebSocket connection task
    let ws_event_tx = event_tx.clone();
    let ws_handle = tokio::spawn(async move {
        loop {
            info!("Connecting to Kraken WebSocket...");
            if let Err(e) = ws.connect(ws_event_tx.clone()).await {
                error!("WebSocket error: {}", e);
            }
            warn!("WebSocket disconnected, reconnecting in 5s...");
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });

    // Clone for sampling task
    let sampling_recorder = Arc::clone(&recorder);
    let sampling_tickers = Arc::clone(&tickers);
    let sampling_orderbooks = Arc::clone(&orderbooks);
    let sampling_candles = Arc::clone(&candles);
    let crypto_pairs = config.trading.crypto_pairs.clone();
    let stock_pairs = config.trading.stock_pairs.clone();

    // Spawn sampling task
    let sampling_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));

        loop {
            interval.tick().await;

            let mut rec = sampling_recorder.lock().await;

            // Sample crypto data
            if rec.should_sample_crypto() {
                let tickers_guard = sampling_tickers.lock().await;
                let orderbooks_guard = sampling_orderbooks.lock().await;
                let candles_guard = sampling_candles.lock().await;

                for pair in &crypto_pairs {
                    if let Some(ticker) = tickers_guard.get(pair) {
                        rec.record_ticker(ticker);
                    }
                    if let Some(orderbook) = orderbooks_guard.get(pair) {
                        rec.record_orderbook(pair, orderbook);
                    }
                    if let Some(pair_candles) = candles_guard.get(pair) {
                        if let Some(candle) = pair_candles.back() {
                            rec.record_ohlc(pair, candle);
                        }
                    }
                }
                rec.mark_crypto_sampled();
                info!("Sampled crypto data for {} pairs", crypto_pairs.len());
            }

            // Sample stock data
            if rec.should_sample_stock() {
                let tickers_guard = sampling_tickers.lock().await;
                let orderbooks_guard = sampling_orderbooks.lock().await;
                let candles_guard = sampling_candles.lock().await;

                for pair in &stock_pairs {
                    if let Some(ticker) = tickers_guard.get(pair) {
                        rec.record_ticker(ticker);
                    }
                    if let Some(orderbook) = orderbooks_guard.get(pair) {
                        rec.record_orderbook(pair, orderbook);
                    }
                    if let Some(pair_candles) = candles_guard.get(pair) {
                        if let Some(candle) = pair_candles.back() {
                            rec.record_ohlc(pair, candle);
                        }
                    }
                }
                rec.mark_stock_sampled();
                info!("Sampled stock data for {} pairs", stock_pairs.len());
            }

            // Flush to disk
            if rec.should_flush() {
                if let Err(e) = rec.flush() {
                    error!("Failed to flush data: {}", e);
                }
            }
        }
    });

    // Clone for event processing
    let event_recorder = Arc::clone(&recorder);
    let event_tickers = Arc::clone(&tickers);

    // Clone for sync task (before moving into event_handle)
    let sync_tickers = Arc::clone(&tickers);
    let sync_orderbooks = Arc::clone(&orderbooks);
    let sync_candles = Arc::clone(&candles);

    // Spawn event processing task
    let event_handle = tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                MarketEvent::Connected => {
                    info!("Connected to Kraken WebSocket");
                }
                MarketEvent::Disconnected => {
                    warn!("Disconnected from Kraken WebSocket");
                }
                MarketEvent::TickerUpdate(ticker) => {
                    event_tickers.lock().await.update(ticker);
                }
                MarketEvent::OrderBookUpdate(pair) => {
                    // Order book is updated in WebSocket handler
                    let _ = pair;
                }
                MarketEvent::CandleUpdate(pair) => {
                    // Candles are updated in WebSocket handler
                    let _ = pair;
                }
                MarketEvent::TradeUpdate(trade) => {
                    // Record trades in real-time
                    let mut rec = event_recorder.lock().await;
                    rec.record_trade(
                        &trade.pair,
                        &trade.side,
                        trade.price,
                        trade.qty,
                        trade.trade_id,
                    );
                }
                MarketEvent::Error(err) => {
                    error!("Market error: {}", err);
                }
            }
        }
    });
    let all_pairs = config.all_pairs();

    let sync_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(500));

        loop {
            interval.tick().await;

            // Sync tickers
            if let Ok(ws_tickers_guard) = ws_tickers.try_lock() {
                let mut local = sync_tickers.lock().await;
                for ticker in ws_tickers_guard.all() {
                    local.update(ticker.clone());
                }
            }

            // Sync orderbooks
            if let Ok(ws_orderbooks_guard) = ws_orderbooks.try_lock() {
                let mut local = sync_orderbooks.lock().await;
                for pair in &all_pairs {
                    if let Some(ob) = ws_orderbooks_guard.get(pair) {
                        let local_ob = local.get_or_create(pair);
                        *local_ob = ob.clone();
                    }
                }
            }

            // Sync candles
            if let Ok(ws_candles_guard) = ws_candles.try_lock() {
                let mut local = sync_candles.lock().await;
                for pair in &all_pairs {
                    if let Some(pair_candles) = ws_candles_guard.get(pair) {
                        for candle in pair_candles.iter() {
                            local.update_last(pair, candle.clone());
                        }
                    }
                }
            }
        }
    });

    // Wait for shutdown signal
    info!("Recorder daemon running. Press Ctrl+C to stop.");

    tokio::signal::ctrl_c().await?;
    info!("Shutdown signal received, flushing data...");

    // Flush remaining data
    {
        let mut rec = recorder.lock().await;
        if let Err(e) = rec.flush() {
            error!("Failed to flush on shutdown: {}", e);
        } else {
            info!("Data flushed successfully");
        }
    }

    // Abort tasks
    ws_handle.abort();
    sampling_handle.abort();
    event_handle.abort();
    sync_handle.abort();

    info!("Recorder daemon stopped");
    Ok(())
}
