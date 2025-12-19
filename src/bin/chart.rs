//! Chart viewer for OHLC data
//!
//! Serves a web interface with Lightweight Charts to visualize market data.
//! Run with: cargo run --bin chart
//!
//! Then open http://localhost:3000 in your browser.

use anyhow::Result;
use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringBuilder, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use axum::{
    extract::{Query, State},
    response::Html,
    routing::get,
    Json, Router,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

/// OHLC candle for JSON response
#[derive(Debug, Clone, Serialize)]
struct OhlcCandle {
    time: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

/// App state
struct AppState {
    data_dir: PathBuf,
    kraken: KrakenClient,
}

/// Query parameters for OHLC endpoint
#[derive(Debug, Deserialize)]
struct OhlcQuery {
    pair: Option<String>,
    date: Option<String>,
    category: Option<String>,
}

/// Query parameters for pairs endpoint
#[derive(Debug, Deserialize)]
struct PairsQuery {
    category: Option<String>,
}

/// Query parameters for Kraken OHLC endpoint
#[derive(Debug, Deserialize)]
struct KrakenOhlcQuery {
    pair: String,
    interval: Option<u32>,
}

/// HTTP client for Kraken API
struct KrakenClient {
    client: reqwest::Client,
}

impl KrakenClient {
    fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn fetch_ohlc(&self, pair: &str, interval: u32) -> Result<Vec<OhlcCandle>, String> {
        // Kraken API needs pair without slash (e.g., XBTUSD, not XBT/USD)
        let pair_clean = pair.replace("/", "");
        let url = format!(
            "https://api.kraken.com/0/public/OHLC?pair={}&interval={}",
            pair_clean, interval
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        let data: Value = response
            .json()
            .await
            .map_err(|e| format!("JSON parse failed: {}", e))?;

        // Check for errors
        if let Some(errors) = data.get("error").and_then(|e| e.as_array()) {
            if !errors.is_empty() {
                return Err(format!("Kraken API error: {:?}", errors));
            }
        }

        // Parse result - the key is the pair name (might be different from input)
        let result = data.get("result").ok_or("No result in response")?;

        // Find the OHLC data (first key that's not "last")
        let ohlc_data = result
            .as_object()
            .and_then(|obj| {
                obj.iter()
                    .find(|(k, _)| *k != "last")
                    .map(|(_, v)| v)
            })
            .and_then(|v| v.as_array())
            .ok_or("No OHLC data found")?;

        let candles: Vec<OhlcCandle> = ohlc_data
            .iter()
            .filter_map(|candle| {
                let arr = candle.as_array()?;
                if arr.len() < 7 {
                    return None;
                }
                Some(OhlcCandle {
                    time: arr[0].as_i64()?,
                    open: arr[1].as_str()?.parse().ok()?,
                    high: arr[2].as_str()?.parse().ok()?,
                    low: arr[3].as_str()?.parse().ok()?,
                    close: arr[4].as_str()?.parse().ok()?,
                    volume: arr[6].as_str()?.parse().ok()?,
                })
            })
            .collect();

        Ok(candles)
    }

    async fn fetch_pairs(&self) -> Result<Vec<String>, String> {
        let url = "https://api.kraken.com/0/public/AssetPairs";

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        let data: Value = response
            .json()
            .await
            .map_err(|e| format!("JSON parse failed: {}", e))?;

        // Check for errors
        if let Some(errors) = data.get("error").and_then(|e| e.as_array()) {
            if !errors.is_empty() {
                return Err(format!("Kraken API error: {:?}", errors));
            }
        }

        let result = data
            .get("result")
            .and_then(|r| r.as_object())
            .ok_or("No result in response")?;

        // Filter tradeable pairs and get wsname (websocket name) which is more readable
        let mut pairs: Vec<String> = result
            .iter()
            .filter_map(|(_, info)| {
                // Only include tradeable pairs
                let status = info.get("status")?.as_str()?;
                if status != "online" {
                    return None;
                }
                // Use wsname if available, otherwise altname
                info.get("wsname")
                    .or_else(|| info.get("altname"))
                    .and_then(|n| n.as_str())
                    .map(|s| s.to_string())
            })
            .collect();

        pairs.sort();
        pairs.dedup();
        Ok(pairs)
    }
}

/// Save OHLC candles to parquet file
fn save_kraken_ohlc(data_dir: &PathBuf, pair: &str, interval: u32, candles: &[OhlcCandle]) -> Result<(), String> {
    if candles.is_empty() {
        return Ok(());
    }

    // Create directory: data/kraken/{pair_safe}/
    let pair_safe = pair.replace("/", "_");
    let dir = data_dir.join("kraken").join(&pair_safe);
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create directory: {}", e))?;

    // File path: data/kraken/{pair_safe}/{interval}.parquet
    let file_path = dir.join(format!("{}.parquet", interval));

    // Build schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, false),
        Field::new("pair", DataType::Utf8, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
    ]));

    // Build arrays
    let ts_array: ArrayRef = Arc::new(Int64Array::from(
        candles.iter().map(|c| c.time * 1000).collect::<Vec<_>>(), // Convert to ms
    ));

    let mut pair_builder = StringBuilder::new();
    for _ in candles {
        pair_builder.append_value(pair);
    }
    let pair_array: ArrayRef = Arc::new(pair_builder.finish());

    let open_array: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.open).collect::<Vec<_>>(),
    ));
    let high_array: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.high).collect::<Vec<_>>(),
    ));
    let low_array: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.low).collect::<Vec<_>>(),
    ));
    let close_array: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.close).collect::<Vec<_>>(),
    ));
    let volume_array: ArrayRef = Arc::new(Float64Array::from(
        candles.iter().map(|c| c.volume).collect::<Vec<_>>(),
    ));

    // Create record batch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![ts_array, pair_array, open_array, high_array, low_array, close_array, volume_array],
    ).map_err(|e| format!("Failed to create record batch: {}", e))?;

    // Write to parquet
    let file = File::create(&file_path).map_err(|e| format!("Failed to create file: {}", e))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| format!("Failed to create parquet writer: {}", e))?;
    writer.write(&batch).map_err(|e| format!("Failed to write batch: {}", e))?;
    writer.close().map_err(|e| format!("Failed to close writer: {}", e))?;

    info!("Saved {} candles to {}", candles.len(), file_path.display());
    Ok(())
}

/// List available pairs (optionally filtered by category)
async fn list_pairs(
    State(state): State<Arc<AppState>>,
    Query(query): Query<PairsQuery>,
) -> Json<Vec<String>> {
    let mut pairs = std::collections::HashSet::new();

    // Determine which categories to scan
    let categories: Vec<&str> = match query.category.as_deref() {
        Some("crypto") => vec!["crypto"],
        Some("stocks") => vec!["stocks"],
        _ => vec!["crypto", "stocks"],
    };

    for category in categories {
        let ohlc_dir = state.data_dir.join(category).join("ohlc");
        if let Ok(dates) = std::fs::read_dir(&ohlc_dir) {
            for date_entry in dates.flatten() {
                if let Ok(files) = std::fs::read_dir(date_entry.path()) {
                    for file_entry in files.flatten() {
                        let path = file_entry.path();
                        if path.extension().map(|e| e == "parquet").unwrap_or(false) {
                            if let Ok(file) = File::open(&path) {
                                if let Ok(builder) = ParquetRecordBatchReaderBuilder::try_new(file) {
                                    if let Ok(reader) = builder.build() {
                                        for batch in reader.flatten() {
                                            if let Some(pair_col) = batch.column_by_name("pair") {
                                                if let Some(arr) = pair_col.as_any().downcast_ref::<StringArray>() {
                                                    for i in 0..arr.len() {
                                                        if !arr.is_null(i) {
                                                            pairs.insert(arr.value(i).to_string());
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut pairs: Vec<_> = pairs.into_iter().collect();
    pairs.sort();
    Json(pairs)
}

/// Get OHLC data for a pair
async fn get_ohlc(
    State(state): State<Arc<AppState>>,
    Query(query): Query<OhlcQuery>,
) -> Json<Vec<OhlcCandle>> {
    let pair = query.pair.unwrap_or_else(|| "BTC/USD".to_string());
    let category = query.category.unwrap_or_else(|| "crypto".to_string());

    let mut candles: HashMap<i64, OhlcCandle> = HashMap::new();

    let ohlc_dir = state.data_dir.join(&category).join("ohlc");

    // Read all date directories or specific date
    let dates: Vec<PathBuf> = if let Some(ref date) = query.date {
        vec![ohlc_dir.join(date)]
    } else {
        std::fs::read_dir(&ohlc_dir)
            .map(|entries| entries.flatten().map(|e| e.path()).collect())
            .unwrap_or_default()
    };

    for date_dir in dates {
        if !date_dir.is_dir() {
            continue;
        }

        if let Ok(files) = std::fs::read_dir(&date_dir) {
            for file_entry in files.flatten() {
                let path = file_entry.path();
                if path.extension().map(|e| e == "parquet").unwrap_or(false) {
                    if let Ok(file) = File::open(&path) {
                        if let Ok(builder) = ParquetRecordBatchReaderBuilder::try_new(file) {
                            if let Ok(reader) = builder.build() {
                                for batch in reader.flatten() {
                                    let pair_col = batch.column_by_name("pair");
                                    let ts_col = batch.column_by_name("ts");
                                    let open_col = batch.column_by_name("open");
                                    let high_col = batch.column_by_name("high");
                                    let low_col = batch.column_by_name("low");
                                    let close_col = batch.column_by_name("close");
                                    let volume_col = batch.column_by_name("volume");

                                    if let (Some(pair_arr), Some(ts_arr), Some(open_arr), Some(high_arr), Some(low_arr), Some(close_arr), Some(vol_arr)) = (
                                        pair_col.and_then(|c| c.as_any().downcast_ref::<StringArray>()),
                                        ts_col.and_then(|c| c.as_any().downcast_ref::<Int64Array>()),
                                        open_col.and_then(|c| c.as_any().downcast_ref::<Float64Array>()),
                                        high_col.and_then(|c| c.as_any().downcast_ref::<Float64Array>()),
                                        low_col.and_then(|c| c.as_any().downcast_ref::<Float64Array>()),
                                        close_col.and_then(|c| c.as_any().downcast_ref::<Float64Array>()),
                                        volume_col.and_then(|c| c.as_any().downcast_ref::<Float64Array>()),
                                    ) {
                                        for i in 0..batch.num_rows() {
                                            if !pair_arr.is_null(i) && pair_arr.value(i) == pair {
                                                let ts = ts_arr.value(i);
                                                // Convert ms to seconds for Lightweight Charts
                                                let time = ts / 1000;

                                                candles.insert(time, OhlcCandle {
                                                    time,
                                                    open: open_arr.value(i),
                                                    high: high_arr.value(i),
                                                    low: low_arr.value(i),
                                                    close: close_arr.value(i),
                                                    volume: vol_arr.value(i),
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut result: Vec<_> = candles.into_values().collect();
    result.sort_by_key(|c| c.time);
    Json(result)
}

/// Get OHLC data from Kraken API
async fn get_kraken_ohlc(
    State(state): State<Arc<AppState>>,
    Query(query): Query<KrakenOhlcQuery>,
) -> Json<Vec<OhlcCandle>> {
    let interval = query.interval.unwrap_or(60);

    match state.kraken.fetch_ohlc(&query.pair, interval).await {
        Ok(candles) => {
            // Save to parquet for persistence
            if let Err(e) = save_kraken_ohlc(&state.data_dir, &query.pair, interval, &candles) {
                warn!("Failed to save Kraken OHLC: {}", e);
            }
            Json(candles)
        }
        Err(e) => {
            warn!("Kraken OHLC fetch failed: {}", e);
            Json(vec![])
        }
    }
}

/// Get available pairs from Kraken API
async fn get_kraken_pairs(State(state): State<Arc<AppState>>) -> Json<Vec<String>> {
    match state.kraken.fetch_pairs().await {
        Ok(pairs) => Json(pairs),
        Err(e) => {
            warn!("Kraken pairs fetch failed: {}", e);
            Json(vec![])
        }
    }
}

/// Main HTML page with chart
async fn index_page() -> Html<String> {

    let html = format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scalper Chart Viewer</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #131722;
            color: #d1d4dc;
            min-height: 100vh;
        }}
        .header {{
            padding: 12px 24px;
            background: #1e222d;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .header h1 {{
            font-size: 18px;
            color: #00c853;
            font-weight: 600;
        }}
        .controls {{
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}
        select, button, input {{
            background: #2a2e39;
            border: 1px solid #363c4e;
            color: #d1d4dc;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
        }}
        input[type="number"] {{
            width: 60px;
            cursor: text;
        }}
        select:hover, button:hover {{
            border-color: #00c853;
        }}
        button {{
            background: #00c853;
            color: #131722;
            font-weight: 600;
        }}
        button:hover {{
            background: #00e676;
        }}
        .indicator-group {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            background: #252930;
            border-radius: 4px;
            font-size: 12px;
        }}
        .indicator-group label {{
            color: #787b86;
        }}
        .indicator-group input[type="checkbox"] {{
            width: 14px;
            height: 14px;
            cursor: pointer;
        }}
        #chart-container {{
            width: 100%;
            height: calc(100vh - 180px);
        }}
        #rsi-container {{
            width: 100%;
            height: 120px;
            border-top: 1px solid #2a2e39;
        }}
        .stats {{
            position: absolute;
            top: 80px;
            right: 24px;
            background: rgba(30, 34, 45, 0.95);
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            z-index: 100;
            min-width: 160px;
        }}
        .stats div {{ margin: 4px 0; }}
        .stats .label {{ color: #787b86; }}
        .stats .value {{ color: #d1d4dc; font-weight: 500; }}
        .stats .up {{ color: #00c853; }}
        .stats .down {{ color: #ff5252; }}
        .stats .sma {{ color: #2196f3; }}
        .stats .ema {{ color: #ff9800; }}
        .stats .rsi {{ color: #e91e63; }}
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 18px;
            color: #787b86;
        }}
        .legend {{
            position: absolute;
            top: 75px;
            left: 24px;
            display: flex;
            gap: 16px;
            font-size: 12px;
            z-index: 100;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .legend-color {{
            width: 12px;
            height: 3px;
            border-radius: 1px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Scalper Charts</h1>
        <div class="controls">
            <select id="data-source">
                <option value="captured">Captured</option>
                <option value="kraken">Kraken API</option>
            </select>
            <select id="category">
                <option value="crypto">Crypto</option>
                <option value="stocks">Stocks</option>
            </select>
            <select id="pair">
                <option value="">Loading...</option>
            </select>
            <select id="timeframe-select">
                <option value="1">1m</option>
                <option value="5">5m</option>
                <option value="15">15m</option>
                <option value="30">30m</option>
                <option value="60" selected>1h</option>
                <option value="240">4h</option>
                <option value="1440">1d</option>
                <option value="10080">1w</option>
            </select>
            <span id="candle-count" style="color: #787b86; font-size: 12px; margin-left: 8px;"></span>
            <div class="indicator-group">
                <input type="checkbox" id="show-sma" checked>
                <label for="show-sma">SMA</label>
                <input type="number" id="sma-period" value="20" min="2" max="200">
            </div>
            <div class="indicator-group">
                <input type="checkbox" id="show-ema" checked>
                <label for="show-ema">EMA</label>
                <input type="number" id="ema-period" value="50" min="2" max="200">
            </div>
            <div class="indicator-group">
                <input type="checkbox" id="show-rsi" checked>
                <label for="show-rsi">RSI</label>
                <input type="number" id="rsi-period" value="14" min="2" max="50">
            </div>
            <button onclick="loadChart()">Load</button>
        </div>
    </div>
    <div class="legend" id="legend" style="display: none;">
        <div class="legend-item"><div class="legend-color" style="background: #2196f3;"></div><span id="sma-label">SMA(20)</span></div>
        <div class="legend-item"><div class="legend-color" style="background: #ff9800;"></div><span id="ema-label">EMA(50)</span></div>
    </div>
    <div id="chart-container">
        <div class="loading" id="loading">Loading chart data...</div>
    </div>
    <div id="rsi-container"></div>
    <div class="stats" id="stats" style="display: none;">
        <div><span class="label">Open:</span> <span class="value" id="stat-open">-</span></div>
        <div><span class="label">High:</span> <span class="value" id="stat-high">-</span></div>
        <div><span class="label">Low:</span> <span class="value" id="stat-low">-</span></div>
        <div><span class="label">Close:</span> <span class="value" id="stat-close">-</span></div>
        <div><span class="label">Volume:</span> <span class="value" id="stat-volume">-</span></div>
        <hr style="border-color: #363c4e; margin: 8px 0;">
        <div><span class="label">SMA:</span> <span class="value sma" id="stat-sma">-</span></div>
        <div><span class="label">EMA:</span> <span class="value ema" id="stat-ema">-</span></div>
        <div><span class="label">RSI:</span> <span class="value rsi" id="stat-rsi">-</span></div>
    </div>

    <script>
        let chart = null;
        let rsiChart = null;
        let candleSeries = null;
        let volumeSeries = null;
        let smaSeries = null;
        let emaSeries = null;
        let rsiSeries = null;
        let rsiOverbought = null;
        let rsiOversold = null;

        // Calculate SMA (Simple Moving Average)
        function calculateSMA(data, period) {{
            const result = [];
            for (let i = 0; i < data.length; i++) {{
                if (i < period - 1) {{
                    continue;
                }}
                let sum = 0;
                for (let j = 0; j < period; j++) {{
                    sum += data[i - j].close;
                }}
                result.push({{
                    time: data[i].time,
                    value: sum / period
                }});
            }}
            return result;
        }}

        // Calculate EMA (Exponential Moving Average)
        function calculateEMA(data, period) {{
            const result = [];
            const multiplier = 2 / (period + 1);
            let ema = null;

            for (let i = 0; i < data.length; i++) {{
                if (ema === null) {{
                    // Start with SMA for first EMA value
                    if (i >= period - 1) {{
                        let sum = 0;
                        for (let j = 0; j < period; j++) {{
                            sum += data[i - j].close;
                        }}
                        ema = sum / period;
                        result.push({{ time: data[i].time, value: ema }});
                    }}
                }} else {{
                    ema = (data[i].close - ema) * multiplier + ema;
                    result.push({{ time: data[i].time, value: ema }});
                }}
            }}
            return result;
        }}

        // Calculate RSI (Relative Strength Index)
        function calculateRSI(data, period) {{
            const result = [];
            const gains = [];
            const losses = [];

            for (let i = 1; i < data.length; i++) {{
                const change = data[i].close - data[i - 1].close;
                gains.push(change > 0 ? change : 0);
                losses.push(change < 0 ? -change : 0);

                if (i >= period) {{
                    let avgGain, avgLoss;

                    if (i === period) {{
                        // First RSI: simple average
                        avgGain = gains.slice(-period).reduce((a, b) => a + b, 0) / period;
                        avgLoss = losses.slice(-period).reduce((a, b) => a + b, 0) / period;
                    }} else {{
                        // Subsequent RSI: smoothed average
                        const prevAvgGain = result.length > 0 ? result[result.length - 1].avgGain : 0;
                        const prevAvgLoss = result.length > 0 ? result[result.length - 1].avgLoss : 0;
                        avgGain = (prevAvgGain * (period - 1) + gains[gains.length - 1]) / period;
                        avgLoss = (prevAvgLoss * (period - 1) + losses[losses.length - 1]) / period;
                    }}

                    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
                    const rsi = 100 - (100 / (1 + rs));

                    result.push({{
                        time: data[i].time,
                        value: rsi,
                        avgGain: avgGain,
                        avgLoss: avgLoss
                    }});
                }}
            }}
            return result;
        }}

        function formatPrice(price) {{
            if (price >= 1000) return price.toFixed(2);
            if (price >= 1) return price.toFixed(4);
            return price.toFixed(6);
        }}

        function formatVolume(vol) {{
            if (vol >= 1e9) return (vol / 1e9).toFixed(2) + 'B';
            if (vol >= 1e6) return (vol / 1e6).toFixed(2) + 'M';
            if (vol >= 1e3) return (vol / 1e3).toFixed(2) + 'K';
            return vol.toFixed(2);
        }}

        async function loadChart() {{
            const dataSource = document.getElementById('data-source').value;
            const pair = document.getElementById('pair').value;
            const category = document.getElementById('category').value;
            const showSMA = document.getElementById('show-sma').checked;
            const showEMA = document.getElementById('show-ema').checked;
            const showRSI = document.getElementById('show-rsi').checked;
            const smaPeriod = parseInt(document.getElementById('sma-period').value) || 20;
            const emaPeriod = parseInt(document.getElementById('ema-period').value) || 50;
            const rsiPeriod = parseInt(document.getElementById('rsi-period').value) || 14;
            const targetMinutes = parseInt(document.getElementById('timeframe-select').value) || 60;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('stats').style.display = 'none';
            document.getElementById('legend').style.display = 'none';

            try {{
                let data;
                let rawCount = 0;

                if (dataSource === 'kraken') {{
                    // Fetch directly from Kraken API with interval
                    const response = await fetch(`/api/kraken-ohlc?pair=${{encodeURIComponent(pair)}}&interval=${{targetMinutes}}`);
                    data = await response.json();
                    rawCount = data.length;
                }} else {{
                    // Fetch from captured data and resample
                    const response = await fetch(`/api/ohlc?pair=${{encodeURIComponent(pair)}}&category=${{category}}`);
                    const rawData = await response.json();
                    rawCount = rawData.length;
                    data = resampleOHLC(rawData, targetMinutes);
                }}

                if (data.length === 0) {{
                    document.getElementById('loading').textContent = 'No data available for ' + pair;
                    document.getElementById('candle-count').textContent = '';
                    return;
                }}

                // Display candle count
                const sourceLabel = dataSource === 'kraken' ? 'Kraken' : 'Captured';
                document.getElementById('candle-count').textContent = dataSource === 'kraken'
                    ? `${{data.length}} candles (${{sourceLabel}})`
                    : `${{data.length}} candles (from ${{rawCount}} raw)`;

                document.getElementById('loading').style.display = 'none';
                document.getElementById('stats').style.display = 'block';
                document.getElementById('legend').style.display = 'flex';

                // Update legend labels
                document.getElementById('sma-label').textContent = `SMA(${{smaPeriod}})`;
                document.getElementById('ema-label').textContent = `EMA(${{emaPeriod}})`;

                // Clear existing charts
                const container = document.getElementById('chart-container');
                const rsiContainer = document.getElementById('rsi-container');
                if (chart) {{ chart.remove(); }}
                if (rsiChart) {{ rsiChart.remove(); }}

                // Adjust heights based on RSI visibility
                if (showRSI) {{
                    container.style.height = 'calc(100vh - 180px)';
                    rsiContainer.style.display = 'block';
                }} else {{
                    container.style.height = 'calc(100vh - 60px)';
                    rsiContainer.style.display = 'none';
                }}

                // Create main chart
                chart = LightweightCharts.createChart(container, {{
                    layout: {{
                        background: {{ type: 'solid', color: '#131722' }},
                        textColor: '#d1d4dc',
                    }},
                    grid: {{
                        vertLines: {{ color: '#1e222d' }},
                        horzLines: {{ color: '#1e222d' }},
                    }},
                    crosshair: {{
                        mode: LightweightCharts.CrosshairMode.Normal,
                    }},
                    rightPriceScale: {{
                        borderColor: '#2a2e39',
                    }},
                    timeScale: {{
                        borderColor: '#2a2e39',
                        timeVisible: true,
                        secondsVisible: false,
                    }},
                }});

                // Add candlestick series
                candleSeries = chart.addCandlestickSeries({{
                    upColor: '#00c853',
                    downColor: '#ff5252',
                    borderDownColor: '#ff5252',
                    borderUpColor: '#00c853',
                    wickDownColor: '#ff5252',
                    wickUpColor: '#00c853',
                }});

                // Add volume series
                volumeSeries = chart.addHistogramSeries({{
                    color: '#26a69a',
                    priceFormat: {{ type: 'volume' }},
                    priceScaleId: '',
                    scaleMargins: {{ top: 0.85, bottom: 0 }},
                }});

                // Add SMA line
                if (showSMA) {{
                    smaSeries = chart.addLineSeries({{
                        color: '#2196f3',
                        lineWidth: 2,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});
                    const smaData = calculateSMA(data, smaPeriod);
                    smaSeries.setData(smaData);
                }}

                // Add EMA line
                if (showEMA) {{
                    emaSeries = chart.addLineSeries({{
                        color: '#ff9800',
                        lineWidth: 2,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});
                    const emaData = calculateEMA(data, emaPeriod);
                    emaSeries.setData(emaData);
                }}

                // Format candle data
                const candleData = data.map(d => ({{
                    time: d.time,
                    open: d.open,
                    high: d.high,
                    low: d.low,
                    close: d.close,
                }}));

                const volumeData = data.map(d => ({{
                    time: d.time,
                    value: d.volume,
                    color: d.close >= d.open ? 'rgba(0, 200, 83, 0.3)' : 'rgba(255, 82, 82, 0.3)',
                }}));

                candleSeries.setData(candleData);
                volumeSeries.setData(volumeData);

                // Create RSI chart
                let rsiData = [];
                if (showRSI) {{
                    rsiChart = LightweightCharts.createChart(rsiContainer, {{
                        layout: {{
                            background: {{ type: 'solid', color: '#131722' }},
                            textColor: '#d1d4dc',
                        }},
                        grid: {{
                            vertLines: {{ color: '#1e222d' }},
                            horzLines: {{ color: '#1e222d' }},
                        }},
                        crosshair: {{
                            mode: LightweightCharts.CrosshairMode.Normal,
                        }},
                        rightPriceScale: {{
                            borderColor: '#2a2e39',
                            scaleMargins: {{ top: 0.1, bottom: 0.1 }},
                        }},
                        timeScale: {{
                            borderColor: '#2a2e39',
                            timeVisible: false,
                            visible: false,
                        }},
                    }});

                    // RSI overbought line (70)
                    rsiOverbought = rsiChart.addLineSeries({{
                        color: 'rgba(255, 82, 82, 0.5)',
                        lineWidth: 1,
                        lineStyle: 2,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});

                    // RSI oversold line (30)
                    rsiOversold = rsiChart.addLineSeries({{
                        color: 'rgba(0, 200, 83, 0.5)',
                        lineWidth: 1,
                        lineStyle: 2,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    }});

                    // RSI line
                    rsiSeries = rsiChart.addLineSeries({{
                        color: '#e91e63',
                        lineWidth: 2,
                        priceLineVisible: false,
                        lastValueVisible: true,
                    }});

                    rsiData = calculateRSI(data, rsiPeriod);
                    rsiSeries.setData(rsiData.map(d => ({{ time: d.time, value: d.value }})));

                    // Set overbought/oversold lines
                    const times = rsiData.map(d => d.time);
                    if (times.length > 0) {{
                        rsiOverbought.setData(times.map(t => ({{ time: t, value: 70 }})));
                        rsiOversold.setData(times.map(t => ({{ time: t, value: 30 }})));
                    }}

                    // Sync time scales
                    chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                        if (range && rsiChart) {{
                            rsiChart.timeScale().setVisibleLogicalRange(range);
                        }}
                    }});
                    rsiChart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                        if (range && chart) {{
                            chart.timeScale().setVisibleLogicalRange(range);
                        }}
                    }});
                }}

                // Fit content
                chart.timeScale().fitContent();

                // Prepare indicator data maps for crosshair
                const smaMap = new Map();
                const emaMap = new Map();
                const rsiMap = new Map();
                if (showSMA) calculateSMA(data, smaPeriod).forEach(d => smaMap.set(d.time, d.value));
                if (showEMA) calculateEMA(data, emaPeriod).forEach(d => emaMap.set(d.time, d.value));
                if (showRSI) rsiData.forEach(d => rsiMap.set(d.time, d.value));

                // Update stats on crosshair move
                chart.subscribeCrosshairMove(param => {{
                    if (param.time) {{
                        const candle = param.seriesData.get(candleSeries);
                        const volume = param.seriesData.get(volumeSeries);
                        if (candle) {{
                            document.getElementById('stat-open').textContent = formatPrice(candle.open);
                            document.getElementById('stat-high').textContent = formatPrice(candle.high);
                            document.getElementById('stat-low').textContent = formatPrice(candle.low);
                            document.getElementById('stat-close').textContent = formatPrice(candle.close);
                            const closeEl = document.getElementById('stat-close');
                            closeEl.className = 'value ' + (candle.close >= candle.open ? 'up' : 'down');
                        }}
                        if (volume) {{
                            document.getElementById('stat-volume').textContent = formatVolume(volume.value);
                        }}
                        // Update indicator stats
                        const smaVal = smaMap.get(param.time);
                        const emaVal = emaMap.get(param.time);
                        const rsiVal = rsiMap.get(param.time);
                        document.getElementById('stat-sma').textContent = smaVal ? formatPrice(smaVal) : '-';
                        document.getElementById('stat-ema').textContent = emaVal ? formatPrice(emaVal) : '-';
                        document.getElementById('stat-rsi').textContent = rsiVal ? rsiVal.toFixed(2) : '-';
                    }}
                }});

                // Show last values
                if (data.length > 0) {{
                    const last = data[data.length - 1];
                    document.getElementById('stat-open').textContent = formatPrice(last.open);
                    document.getElementById('stat-high').textContent = formatPrice(last.high);
                    document.getElementById('stat-low').textContent = formatPrice(last.low);
                    document.getElementById('stat-close').textContent = formatPrice(last.close);
                    document.getElementById('stat-volume').textContent = formatVolume(last.volume);
                    const closeEl = document.getElementById('stat-close');
                    closeEl.className = 'value ' + (last.close >= last.open ? 'up' : 'down');

                    // Last indicator values
                    const lastSMA = showSMA ? calculateSMA(data, smaPeriod).slice(-1)[0] : null;
                    const lastEMA = showEMA ? calculateEMA(data, emaPeriod).slice(-1)[0] : null;
                    const lastRSI = showRSI ? rsiData.slice(-1)[0] : null;
                    document.getElementById('stat-sma').textContent = lastSMA ? formatPrice(lastSMA.value) : '-';
                    document.getElementById('stat-ema').textContent = lastEMA ? formatPrice(lastEMA.value) : '-';
                    document.getElementById('stat-rsi').textContent = lastRSI ? lastRSI.value.toFixed(2) : '-';
                }}

            }} catch (error) {{
                document.getElementById('loading').textContent = 'Error loading data: ' + error.message;
                console.error('Error:', error);
            }}
        }}

        // Handle window resize
        window.addEventListener('resize', () => {{
            if (chart) {{
                chart.applyOptions({{
                    width: document.getElementById('chart-container').clientWidth,
                    height: document.getElementById('chart-container').clientHeight,
                }});
            }}
            if (rsiChart) {{
                rsiChart.applyOptions({{
                    width: document.getElementById('rsi-container').clientWidth,
                    height: document.getElementById('rsi-container').clientHeight,
                }});
            }}
        }});

        // Load pairs for selected category
        async function loadPairs() {{
            const dataSource = document.getElementById('data-source').value;
            const category = document.getElementById('category').value;
            const pairSelect = document.getElementById('pair');
            const categorySelect = document.getElementById('category');

            // Show/hide category based on data source
            categorySelect.style.display = dataSource === 'captured' ? 'block' : 'none';

            try {{
                let url;
                if (dataSource === 'kraken') {{
                    url = '/api/kraken-pairs';
                }} else {{
                    url = `/api/pairs?category=${{category}}`;
                }}

                const response = await fetch(url);
                const pairs = await response.json();

                pairSelect.innerHTML = pairs.map(p =>
                    `<option value="${{p}}">${{p}}</option>`
                ).join('');

                // Load chart with first pair
                if (pairs.length > 0) {{
                    loadChart();
                }}
            }} catch (error) {{
                console.error('Error loading pairs:', error);
                pairSelect.innerHTML = '<option value="">Error loading pairs</option>';
            }}
        }}

        // Resample OHLC data to a larger timeframe
        function resampleOHLC(data, targetMinutes) {{
            if (data.length === 0) return [];

            const targetSeconds = targetMinutes * 60;
            const result = [];
            let currentCandle = null;

            for (const candle of data) {{
                // Round down to the start of the target period
                const periodStart = Math.floor(candle.time / targetSeconds) * targetSeconds;

                if (currentCandle === null || currentCandle.time !== periodStart) {{
                    // Start new candle
                    if (currentCandle !== null) {{
                        result.push(currentCandle);
                    }}
                    currentCandle = {{
                        time: periodStart,
                        open: candle.open,
                        high: candle.high,
                        low: candle.low,
                        close: candle.close,
                        volume: candle.volume
                    }};
                }} else {{
                    // Update existing candle
                    currentCandle.high = Math.max(currentCandle.high, candle.high);
                    currentCandle.low = Math.min(currentCandle.low, candle.low);
                    currentCandle.close = candle.close;
                    currentCandle.volume += candle.volume;
                }}
            }}

            // Don't forget the last candle
            if (currentCandle !== null) {{
                result.push(currentCandle);
            }}

            return result;
        }}

        // Update data source handler
        document.getElementById('data-source').addEventListener('change', loadPairs);

        // Update category handler
        document.getElementById('category').addEventListener('change', loadPairs);

        // Update timeframe handler
        document.getElementById('timeframe-select').addEventListener('change', loadChart);

        // Load pairs and chart on page load
        window.onload = loadPairs;
    </script>
</body>
</html>"##);

    Html(html)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("chart=info")
        .init();

    // Get data directory from env or default
    let data_dir = std::env::var("DATA_DIR")
        .unwrap_or_else(|_| "data".to_string());
    let data_dir = PathBuf::from(data_dir);

    if !data_dir.exists() {
        anyhow::bail!("Data directory '{}' does not exist", data_dir.display());
    }

    info!("Starting chart viewer with data from: {}", data_dir.display());

    let state = Arc::new(AppState {
        data_dir,
        kraken: KrakenClient::new(),
    });

    let app = Router::new()
        .route("/", get(index_page))
        .route("/api/pairs", get(list_pairs))
        .route("/api/ohlc", get(get_ohlc))
        .route("/api/kraken-pairs", get(get_kraken_pairs))
        .route("/api/kraken-ohlc", get(get_kraken_ohlc))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    info!("Chart viewer running at http://localhost:{}", port);
    info!("Open your browser to view the charts");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
