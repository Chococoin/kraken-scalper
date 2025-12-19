//! Data recorder daemon
//!
//! Captures market data from Kraken WebSocket and saves to Parquet files.
//! Run with: cargo run --bin recorder
//!
//! Stops gracefully on SIGINT (Ctrl+C) or SIGTERM.
//!
//! HTTP stats endpoint available at http://localhost:8080/

use anyhow::Result;
use axum::{extract::State, response::Html, routing::get, Json, Router};
use chrono::{DateTime, Datelike, Timelike, Utc, Weekday};
use scalper::api::{KrakenWebSocket, MarketEvent};
use scalper::config::Config;
use scalper::storage::{DataRecorder, HfUploader, KrakenOhlcFetcher};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{error, info, warn};

/// Shared stats for HTTP endpoint
#[derive(Debug, Clone, Serialize)]
struct RecorderStats {
    uptime_secs: u64,
    started_at: String,
    connected: bool,
    crypto_pairs: usize,
    stock_pairs: usize,
    trades_recorded: u64,
    tickers_recorded: u64,
    orderbooks_recorded: u64,
    ohlc_recorded: u64,
    last_sample: String,
    us_market_open: bool,
}

/// Atomic counters for stats
struct StatsCounters {
    trades: AtomicU64,
    tickers: AtomicU64,
    orderbooks: AtomicU64,
    ohlc: AtomicU64,
    connected: AtomicBool,
    last_sample: RwLock<DateTime<Utc>>,
}

impl StatsCounters {
    fn new() -> Self {
        Self {
            trades: AtomicU64::new(0),
            tickers: AtomicU64::new(0),
            orderbooks: AtomicU64::new(0),
            ohlc: AtomicU64::new(0),
            connected: AtomicBool::new(false),
            last_sample: RwLock::new(Utc::now()),
        }
    }
}

/// HTTP state
struct AppState {
    start_time: Instant,
    started_at: DateTime<Utc>,
    crypto_pairs: usize,
    stock_pairs: usize,
    counters: Arc<StatsCounters>,
}

async fn stats_json(State(state): State<Arc<AppState>>) -> Json<RecorderStats> {
    let last_sample = state.counters.last_sample.read().await;
    Json(RecorderStats {
        uptime_secs: state.start_time.elapsed().as_secs(),
        started_at: state.started_at.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        connected: state.counters.connected.load(Ordering::Relaxed),
        crypto_pairs: state.crypto_pairs,
        stock_pairs: state.stock_pairs,
        trades_recorded: state.counters.trades.load(Ordering::Relaxed),
        tickers_recorded: state.counters.tickers.load(Ordering::Relaxed),
        orderbooks_recorded: state.counters.orderbooks.load(Ordering::Relaxed),
        ohlc_recorded: state.counters.ohlc.load(Ordering::Relaxed),
        last_sample: last_sample.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        us_market_open: is_us_market_open(),
    })
}

async fn stats_html(State(state): State<Arc<AppState>>) -> Html<String> {
    let last_sample = state.counters.last_sample.read().await;
    let uptime = state.start_time.elapsed().as_secs();
    let hours = uptime / 3600;
    let mins = (uptime % 3600) / 60;
    let connected = state.counters.connected.load(Ordering::Relaxed);

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Scalper Recorder Stats</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body {{ font-family: monospace; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #00ff88; }}
        .stat {{ margin: 10px 0; }}
        .label {{ color: #888; }}
        .value {{ color: #00ff88; font-weight: bold; }}
        .connected {{ color: #00ff88; }}
        .disconnected {{ color: #ff4444; }}
        .market-open {{ color: #00ff88; }}
        .market-closed {{ color: #ff8800; }}
    </style>
</head>
<body>
    <h1>📊 Scalper Recorder</h1>
    <div class="stat"><span class="label">Status:</span> <span class="{}">{}</span></div>
    <div class="stat"><span class="label">Uptime:</span> <span class="value">{}h {}m</span></div>
    <div class="stat"><span class="label">Started:</span> <span class="value">{}</span></div>
    <hr>
    <h2>📈 Data Collected</h2>
    <div class="stat"><span class="label">Trades:</span> <span class="value">{}</span></div>
    <div class="stat"><span class="label">Tickers:</span> <span class="value">{}</span></div>
    <div class="stat"><span class="label">Orderbooks:</span> <span class="value">{}</span></div>
    <div class="stat"><span class="label">OHLC:</span> <span class="value">{}</span></div>
    <div class="stat"><span class="label">Last Sample:</span> <span class="value">{}</span></div>
    <hr>
    <h2>⚙️ Configuration</h2>
    <div class="stat"><span class="label">Crypto Pairs:</span> <span class="value">{}</span></div>
    <div class="stat"><span class="label">Stock Pairs:</span> <span class="value">{}</span></div>
    <div class="stat"><span class="label">US Market:</span> <span class="{}">{}</span></div>
    <hr>
    <p style="color:#666">Auto-refresh every 10s | <a href="/stats" style="color:#00ff88">JSON API</a></p>
</body>
</html>"#,
        if connected { "connected" } else { "disconnected" },
        if connected { "🟢 Connected" } else { "🔴 Disconnected" },
        hours, mins,
        state.started_at.format("%Y-%m-%d %H:%M:%S UTC"),
        state.counters.trades.load(Ordering::Relaxed),
        state.counters.tickers.load(Ordering::Relaxed),
        state.counters.orderbooks.load(Ordering::Relaxed),
        state.counters.ohlc.load(Ordering::Relaxed),
        last_sample.format("%H:%M:%S UTC"),
        state.crypto_pairs,
        state.stock_pairs,
        if is_us_market_open() { "market-open" } else { "market-closed" },
        if is_us_market_open() { "🟢 Open" } else { "🔴 Closed" },
    );
    Html(html)
}

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

    // Create stats counters
    let counters = Arc::new(StatsCounters::new());

    // Create HTTP server state
    let app_state = Arc::new(AppState {
        start_time: Instant::now(),
        started_at: Utc::now(),
        crypto_pairs: config.trading.crypto_pairs.len(),
        stock_pairs: config.trading.stock_pairs.len(),
        counters: Arc::clone(&counters),
    });

    // Start HTTP server
    let http_state = Arc::clone(&app_state);
    tokio::spawn(async move {
        let app = Router::new()
            .route("/", get(stats_html))
            .route("/stats", get(stats_json))
            .with_state(http_state);

        let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
        let addr = format!("0.0.0.0:{}", port);
        info!("Starting HTTP stats server on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

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

    // Create Kraken OHLC fetcher if enabled
    let mut ohlc_fetcher = if config.kraken_ohlc.enabled {
        info!(
            "Kraken OHLC REST capture enabled for {} pairs",
            config.kraken_ohlc.pairs.len()
        );
        Some(KrakenOhlcFetcher::new(
            config.kraken_ohlc.clone(),
            &config.recording.data_dir,
        ))
    } else {
        None
    };
    let mut last_ohlc_check = Instant::now();
    let ohlc_check_interval = Duration::from_secs(60); // Check every minute

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
                            counters.connected.store(true, Ordering::Relaxed);
                        }
                        MarketEvent::Disconnected => {
                            warn!("Disconnected from Kraken WebSocket");
                            counters.connected.store(false, Ordering::Relaxed);
                        }
                        MarketEvent::TradeUpdate(trade) => {
                            recorder.record_trade(
                                &trade.pair,
                                &trade.side,
                                trade.price,
                                trade.qty,
                                trade.trade_id,
                            );
                            counters.trades.fetch_add(1, Ordering::Relaxed);
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
            let mut ticker_count = 0u64;
            let mut orderbook_count = 0u64;
            let mut ohlc_count = 0u64;
            if let Ok(tickers) = ws_tickers.try_lock() {
                if let Ok(orderbooks) = ws_orderbooks.try_lock() {
                    if let Ok(candles) = ws_candles.try_lock() {
                        for pair in &config.trading.crypto_pairs {
                            if let Some(ticker) = tickers.get(pair) {
                                recorder.record_ticker(ticker);
                                sampled += 1;
                                ticker_count += 1;
                            }
                            if let Some(orderbook) = orderbooks.get(pair) {
                                recorder.record_orderbook(pair, orderbook);
                                orderbook_count += 1;
                            }
                            if let Some(pair_candles) = candles.get(pair) {
                                if let Some(candle) = pair_candles.back() {
                                    recorder.record_ohlc(pair, candle);
                                    ohlc_count += 1;
                                }
                            }
                        }
                    }
                }
            }
            recorder.mark_crypto_sampled();
            if sampled > 0 {
                counters.tickers.fetch_add(ticker_count, Ordering::Relaxed);
                counters.orderbooks.fetch_add(orderbook_count, Ordering::Relaxed);
                counters.ohlc.fetch_add(ohlc_count, Ordering::Relaxed);
                *counters.last_sample.write().await = Utc::now();
                info!("Sampled crypto data: {} pairs", sampled);
            }
        }

        // Sample stock data (only during US market hours)
        if recorder.should_sample_stock() {
            recorder.mark_stock_sampled();

            // Only record stock data when US market is open (NYSE: 9:30 AM - 4:00 PM ET, Mon-Fri)
            if is_us_market_open() {
                let mut sampled = 0;
                let mut stock_ticker_count = 0u64;
                let mut stock_orderbook_count = 0u64;
                let mut stock_ohlc_count = 0u64;
                let ticker_map_count = ws_tickers.try_lock().map(|t| t.all().count()).unwrap_or(0);
                if let Ok(tickers) = ws_tickers.try_lock() {
                    if let Ok(orderbooks) = ws_orderbooks.try_lock() {
                        if let Ok(candles) = ws_candles.try_lock() {
                            for pair in &config.trading.stock_pairs {
                                if let Some(ticker) = tickers.get(pair) {
                                    recorder.record_ticker(ticker);
                                    sampled += 1;
                                    stock_ticker_count += 1;
                                }
                                if let Some(orderbook) = orderbooks.get(pair) {
                                    recorder.record_orderbook(pair, orderbook);
                                    stock_orderbook_count += 1;
                                }
                                if let Some(pair_candles) = candles.get(pair) {
                                    if let Some(candle) = pair_candles.back() {
                                        recorder.record_ohlc(pair, candle);
                                        stock_ohlc_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                if sampled > 0 {
                    counters.tickers.fetch_add(stock_ticker_count, Ordering::Relaxed);
                    counters.orderbooks.fetch_add(stock_orderbook_count, Ordering::Relaxed);
                    counters.ohlc.fetch_add(stock_ohlc_count, Ordering::Relaxed);
                    *counters.last_sample.write().await = Utc::now();
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
                } else if ticker_map_count > 0 {
                    // Only warn if we have some data but no stocks matched
                    warn!("Stock sample: 0/{} stock pairs (ticker map has {} entries, data may still be loading)",
                        config.trading.stock_pairs.len(), ticker_map_count);
                }
            }
        }

        // Periodic flush to disk
        if recorder.should_flush() {
            let stats = recorder.buffer_stats();
            if stats.0 > 0 || stats.3 > 0 || stats.4 > 0 || stats.7 > 0 {
                info!(
                    "Periodic flush - crypto: {}t/{}b/{}o/{}tr, stocks: {}t/{}b/{}o/{}tr",
                    stats.0, stats.1, stats.2, stats.3,
                    stats.4, stats.5, stats.6, stats.7
                );
                if let Err(e) = recorder.flush() {
                    error!("Failed to flush: {}", e);
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

        // Periodic Kraken REST API OHLC fetch
        if let Some(ref mut fetcher) = ohlc_fetcher {
            if last_ohlc_check.elapsed() >= ohlc_check_interval {
                if fetcher.should_fetch() {
                    info!("Starting Kraken OHLC REST fetch...");
                    match fetcher.fetch_all_pending().await {
                        Ok(count) => {
                            if count > 0 {
                                info!("Fetched {} total OHLC candles from Kraken REST API", count);
                            }
                            let (pairs, fetched) = fetcher.stats();
                            info!("OHLC REST stats: {} pairs configured, {} fetch records", pairs, fetched);
                        }
                        Err(e) => error!("Kraken OHLC REST fetch failed: {}", e),
                    }
                }
                last_ohlc_check = Instant::now();
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
