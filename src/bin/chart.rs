//! Chart viewer for OHLC data
//!
//! Serves a web interface with Lightweight Charts to visualize market data.
//! Run with: cargo run --bin chart
//!
//! Then open http://localhost:3000 in your browser.

use anyhow::Result;
use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use axum::{
    extract::{Query, State},
    response::Html,
    routing::get,
    Json, Router,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

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
}

/// Query parameters for OHLC endpoint
#[derive(Debug, Deserialize)]
struct OhlcQuery {
    pair: Option<String>,
    date: Option<String>,
    category: Option<String>,
}

/// List available pairs
async fn list_pairs(State(state): State<Arc<AppState>>) -> Json<Vec<String>> {
    let mut pairs = std::collections::HashSet::new();

    // Scan crypto OHLC files
    for category in ["crypto", "stocks"] {
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

/// Main HTML page with chart
async fn index_page(State(state): State<Arc<AppState>>) -> Html<String> {
    // Get list of pairs for dropdown
    let pairs = list_pairs(State(state)).await.0;

    let pair_options: String = pairs
        .iter()
        .map(|p| format!(r#"<option value="{}">{}</option>"#, p, p))
        .collect::<Vec<_>>()
        .join("\n");

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
            padding: 16px 24px;
            background: #1e222d;
            border-bottom: 1px solid #2a2e39;
            display: flex;
            align-items: center;
            gap: 16px;
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
        }}
        select, button {{
            background: #2a2e39;
            border: 1px solid #363c4e;
            color: #d1d4dc;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
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
        #chart-container {{
            width: 100%;
            height: calc(100vh - 70px);
        }}
        .stats {{
            position: absolute;
            top: 80px;
            right: 24px;
            background: rgba(30, 34, 45, 0.9);
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            z-index: 100;
        }}
        .stats div {{ margin: 4px 0; }}
        .stats .label {{ color: #787b86; }}
        .stats .value {{ color: #d1d4dc; font-weight: 500; }}
        .stats .up {{ color: #00c853; }}
        .stats .down {{ color: #ff5252; }}
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 18px;
            color: #787b86;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Scalper Charts</h1>
        <div class="controls">
            <select id="category">
                <option value="crypto">Crypto</option>
                <option value="stocks">Stocks</option>
            </select>
            <select id="pair">
                {pair_options}
            </select>
            <button onclick="loadChart()">Load</button>
        </div>
    </div>
    <div id="chart-container">
        <div class="loading" id="loading">Loading chart data...</div>
    </div>
    <div class="stats" id="stats" style="display: none;">
        <div><span class="label">Open:</span> <span class="value" id="stat-open">-</span></div>
        <div><span class="label">High:</span> <span class="value" id="stat-high">-</span></div>
        <div><span class="label">Low:</span> <span class="value" id="stat-low">-</span></div>
        <div><span class="label">Close:</span> <span class="value" id="stat-close">-</span></div>
        <div><span class="label">Volume:</span> <span class="value" id="stat-volume">-</span></div>
    </div>

    <script>
        let chart = null;
        let candleSeries = null;
        let volumeSeries = null;

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
            const pair = document.getElementById('pair').value;
            const category = document.getElementById('category').value;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('stats').style.display = 'none';

            try {{
                const response = await fetch(`/api/ohlc?pair=${{encodeURIComponent(pair)}}&category=${{category}}`);
                const data = await response.json();

                if (data.length === 0) {{
                    document.getElementById('loading').textContent = 'No data available for ' + pair;
                    return;
                }}

                document.getElementById('loading').style.display = 'none';
                document.getElementById('stats').style.display = 'block';

                // Clear existing chart
                const container = document.getElementById('chart-container');
                if (chart) {{
                    chart.remove();
                }}

                // Create new chart
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
                    priceFormat: {{
                        type: 'volume',
                    }},
                    priceScaleId: '',
                    scaleMargins: {{
                        top: 0.8,
                        bottom: 0,
                    }},
                }});

                // Format data for Lightweight Charts
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
                    color: d.close >= d.open ? 'rgba(0, 200, 83, 0.5)' : 'rgba(255, 82, 82, 0.5)',
                }}));

                candleSeries.setData(candleData);
                volumeSeries.setData(volumeData);

                // Fit content
                chart.timeScale().fitContent();

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
                    }}
                }});

                // Show last candle stats
                if (data.length > 0) {{
                    const last = data[data.length - 1];
                    document.getElementById('stat-open').textContent = formatPrice(last.open);
                    document.getElementById('stat-high').textContent = formatPrice(last.high);
                    document.getElementById('stat-low').textContent = formatPrice(last.low);
                    document.getElementById('stat-close').textContent = formatPrice(last.close);
                    document.getElementById('stat-volume').textContent = formatVolume(last.volume);

                    const closeEl = document.getElementById('stat-close');
                    closeEl.className = 'value ' + (last.close >= last.open ? 'up' : 'down');
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
        }});

        // Load chart on page load
        window.onload = loadChart;
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

    let state = Arc::new(AppState { data_dir });

    let app = Router::new()
        .route("/", get(index_page))
        .route("/api/pairs", get(list_pairs))
        .route("/api/ohlc", get(get_ohlc))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{}", port);

    info!("Chart viewer running at http://localhost:{}", port);
    info!("Open your browser to view the charts");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
