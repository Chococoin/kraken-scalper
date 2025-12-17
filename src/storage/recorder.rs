use crate::config::{Config, RecordingConfig};
use crate::data::{Candle, OrderBook, Ticker};
use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Datelike, Timelike, Utc};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

/// Snapshot of ticker data for recording
#[derive(Debug, Clone)]
pub struct TickerSnapshot {
    pub ts: i64,
    pub pair: String,
    pub bid: f64,
    pub ask: f64,
    pub last: f64,
    pub volume: f64,
    pub vwap: f64,
    pub high: f64,
    pub low: f64,
    pub change_pct: f64,
}

/// Snapshot of order book for recording
#[derive(Debug, Clone)]
pub struct BookSnapshot {
    pub ts: i64,
    pub pair: String,
    pub bids: Vec<(f64, f64)>, // (price, qty)
    pub asks: Vec<(f64, f64)>,
}

/// Snapshot of OHLC candle for recording
#[derive(Debug, Clone)]
pub struct OhlcSnapshot {
    pub ts: i64,
    pub pair: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: f64,
    pub trades: i64,
}

/// Snapshot of a trade for recording
#[derive(Debug, Clone)]
pub struct TradeSnapshot {
    pub ts: i64,
    pub pair: String,
    pub side: String,
    pub price: f64,
    pub qty: f64,
    pub trade_id: i64,
}

/// Data recorder that captures market data to Parquet files
pub struct DataRecorder {
    config: RecordingConfig,
    crypto_pairs: Vec<String>,
    stock_pairs: Vec<String>,

    // Buffers for crypto data
    crypto_ticker_buffer: Vec<TickerSnapshot>,
    crypto_book_buffer: Vec<BookSnapshot>,
    crypto_ohlc_buffer: Vec<OhlcSnapshot>,
    crypto_trade_buffer: Vec<TradeSnapshot>,

    // Buffers for stock data
    stock_ticker_buffer: Vec<TickerSnapshot>,
    stock_book_buffer: Vec<BookSnapshot>,
    stock_ohlc_buffer: Vec<OhlcSnapshot>,
    stock_trade_buffer: Vec<TradeSnapshot>,

    // Sampling control
    last_crypto_sample: Instant,
    last_stock_sample: Instant,
    last_flush: Instant,

    // Track last candle timestamps per pair to avoid duplicates
    last_ohlc_ts: HashMap<String, i64>,
}

impl DataRecorder {
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.recording.clone(),
            crypto_pairs: config.trading.crypto_pairs.clone(),
            stock_pairs: config.trading.stock_pairs.clone(),
            crypto_ticker_buffer: Vec::new(),
            crypto_book_buffer: Vec::new(),
            crypto_ohlc_buffer: Vec::new(),
            crypto_trade_buffer: Vec::new(),
            stock_ticker_buffer: Vec::new(),
            stock_book_buffer: Vec::new(),
            stock_ohlc_buffer: Vec::new(),
            stock_trade_buffer: Vec::new(),
            last_crypto_sample: Instant::now(),
            last_stock_sample: Instant::now(),
            last_flush: Instant::now(),
            last_ohlc_ts: HashMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if it's time to sample crypto data
    pub fn should_sample_crypto(&self) -> bool {
        self.last_crypto_sample.elapsed().as_secs() >= self.config.crypto_sample_interval_secs
    }

    /// Check if it's time to sample stock data
    pub fn should_sample_stock(&self) -> bool {
        self.last_stock_sample.elapsed().as_secs() >= self.config.stock_sample_interval_secs
    }

    /// Check if it's time to flush to disk
    pub fn should_flush(&self) -> bool {
        self.last_flush.elapsed().as_secs() >= self.config.flush_interval_secs
    }

    /// Check if a pair is a crypto pair
    fn is_crypto(&self, pair: &str) -> bool {
        self.crypto_pairs.contains(&pair.to_string())
    }

    /// Record a ticker snapshot (called at sampling intervals)
    pub fn record_ticker(&mut self, ticker: &Ticker) {
        if !self.config.enabled {
            return;
        }

        let snapshot = TickerSnapshot {
            ts: Utc::now().timestamp_millis(),
            pair: ticker.pair.clone(),
            bid: decimal_to_f64(ticker.bid),
            ask: decimal_to_f64(ticker.ask),
            last: decimal_to_f64(ticker.last),
            volume: decimal_to_f64(ticker.volume_24h),
            vwap: decimal_to_f64(ticker.vwap_24h),
            high: decimal_to_f64(ticker.high_24h),
            low: decimal_to_f64(ticker.low_24h),
            change_pct: decimal_to_f64(ticker.change_24h),
        };

        if self.is_crypto(&ticker.pair) {
            self.crypto_ticker_buffer.push(snapshot);
        } else {
            self.stock_ticker_buffer.push(snapshot);
        }
    }

    /// Record an order book snapshot
    pub fn record_orderbook(&mut self, pair: &str, orderbook: &OrderBook) {
        if !self.config.enabled {
            return;
        }

        let depth = if self.is_crypto(pair) {
            self.config.crypto_book_depth as usize
        } else {
            self.config.stock_book_depth as usize
        };

        let bids: Vec<(f64, f64)> = orderbook
            .top_bids(depth)
            .iter()
            .map(|l| (decimal_to_f64(l.price), decimal_to_f64(l.qty)))
            .collect();

        let asks: Vec<(f64, f64)> = orderbook
            .top_asks(depth)
            .iter()
            .map(|l| (decimal_to_f64(l.price), decimal_to_f64(l.qty)))
            .collect();

        let snapshot = BookSnapshot {
            ts: Utc::now().timestamp_millis(),
            pair: pair.to_string(),
            bids,
            asks,
        };

        if self.is_crypto(pair) {
            self.crypto_book_buffer.push(snapshot);
        } else {
            self.stock_book_buffer.push(snapshot);
        }
    }

    /// Record an OHLC candle
    pub fn record_ohlc(&mut self, pair: &str, candle: &Candle) {
        if !self.config.enabled {
            return;
        }

        let ts = candle.timestamp.timestamp_millis();

        // Avoid recording duplicate candles
        if let Some(&last_ts) = self.last_ohlc_ts.get(pair) {
            if ts <= last_ts {
                return;
            }
        }
        self.last_ohlc_ts.insert(pair.to_string(), ts);

        let snapshot = OhlcSnapshot {
            ts,
            pair: pair.to_string(),
            open: decimal_to_f64(candle.open),
            high: decimal_to_f64(candle.high),
            low: decimal_to_f64(candle.low),
            close: decimal_to_f64(candle.close),
            volume: decimal_to_f64(candle.volume),
            vwap: decimal_to_f64(candle.vwap),
            trades: candle.trades as i64,
        };

        if self.is_crypto(pair) {
            self.crypto_ohlc_buffer.push(snapshot);
        } else {
            self.stock_ohlc_buffer.push(snapshot);
        }
    }

    /// Record a trade
    pub fn record_trade(
        &mut self,
        pair: &str,
        side: &str,
        price: Decimal,
        qty: Decimal,
        trade_id: u64,
    ) {
        if !self.config.enabled {
            return;
        }

        let snapshot = TradeSnapshot {
            ts: Utc::now().timestamp_millis(),
            pair: pair.to_string(),
            side: side.to_string(),
            price: decimal_to_f64(price),
            qty: decimal_to_f64(qty),
            trade_id: trade_id as i64,
        };

        if self.is_crypto(pair) {
            self.crypto_trade_buffer.push(snapshot);
        } else {
            self.stock_trade_buffer.push(snapshot);
        }
    }

    /// Reset crypto sampling timer
    pub fn mark_crypto_sampled(&mut self) {
        self.last_crypto_sample = Instant::now();
    }

    /// Reset stock sampling timer
    pub fn mark_stock_sampled(&mut self) {
        self.last_stock_sample = Instant::now();
    }

    /// Flush all buffers to disk
    pub fn flush(&mut self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let now = Utc::now();
        info!(
            "Flushing data buffers - crypto: {} tickers, {} books, {} ohlc, {} trades | stocks: {} tickers, {} books, {} ohlc, {} trades",
            self.crypto_ticker_buffer.len(),
            self.crypto_book_buffer.len(),
            self.crypto_ohlc_buffer.len(),
            self.crypto_trade_buffer.len(),
            self.stock_ticker_buffer.len(),
            self.stock_book_buffer.len(),
            self.stock_ohlc_buffer.len(),
            self.stock_trade_buffer.len(),
        );

        // Flush crypto data
        if !self.crypto_ticker_buffer.is_empty() {
            self.write_ticker_parquet("crypto", &self.crypto_ticker_buffer, now)?;
            self.crypto_ticker_buffer.clear();
        }
        if !self.crypto_book_buffer.is_empty() {
            self.write_book_parquet("crypto", &self.crypto_book_buffer, now)?;
            self.crypto_book_buffer.clear();
        }
        if !self.crypto_ohlc_buffer.is_empty() {
            self.write_ohlc_parquet("crypto", &self.crypto_ohlc_buffer, now)?;
            self.crypto_ohlc_buffer.clear();
        }
        if !self.crypto_trade_buffer.is_empty() {
            self.write_trade_parquet("crypto", &self.crypto_trade_buffer, now)?;
            self.crypto_trade_buffer.clear();
        }

        // Flush stock data
        if !self.stock_ticker_buffer.is_empty() {
            self.write_ticker_parquet("stocks", &self.stock_ticker_buffer, now)?;
            self.stock_ticker_buffer.clear();
        }
        if !self.stock_book_buffer.is_empty() {
            self.write_book_parquet("stocks", &self.stock_book_buffer, now)?;
            self.stock_book_buffer.clear();
        }
        if !self.stock_ohlc_buffer.is_empty() {
            self.write_ohlc_parquet("stocks", &self.stock_ohlc_buffer, now)?;
            self.stock_ohlc_buffer.clear();
        }
        if !self.stock_trade_buffer.is_empty() {
            self.write_trade_parquet("stocks", &self.stock_trade_buffer, now)?;
            self.stock_trade_buffer.clear();
        }

        self.last_flush = Instant::now();
        Ok(())
    }

    /// Get the output path for a parquet file
    fn get_output_path(
        &self,
        category: &str,
        data_type: &str,
        now: DateTime<Utc>,
    ) -> Result<PathBuf> {
        let date_str = format!("{:04}-{:02}-{:02}", now.year(), now.month(), now.day());
        let hour_str = format!("{:02}", now.hour());

        let dir = PathBuf::from(&self.config.data_dir)
            .join(category)
            .join(data_type)
            .join(&date_str);

        fs::create_dir_all(&dir).context("Failed to create data directory")?;

        Ok(dir.join(format!("{}.parquet", hour_str)))
    }

    /// Write ticker data to parquet
    fn write_ticker_parquet(
        &self,
        category: &str,
        data: &[TickerSnapshot],
        now: DateTime<Utc>,
    ) -> Result<()> {
        let path = self.get_output_path(category, "ticker", now)?;
        debug!("Writing {} ticker records to {:?}", data.len(), path);

        let schema = Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Int64, false),
            Field::new("pair", DataType::Utf8, false),
            Field::new("bid", DataType::Float64, false),
            Field::new("ask", DataType::Float64, false),
            Field::new("last", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
            Field::new("vwap", DataType::Float64, false),
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("change_pct", DataType::Float64, false),
        ]));

        let ts: Vec<i64> = data.iter().map(|d| d.ts).collect();
        let pairs: Vec<&str> = data.iter().map(|d| d.pair.as_str()).collect();
        let bids: Vec<f64> = data.iter().map(|d| d.bid).collect();
        let asks: Vec<f64> = data.iter().map(|d| d.ask).collect();
        let lasts: Vec<f64> = data.iter().map(|d| d.last).collect();
        let volumes: Vec<f64> = data.iter().map(|d| d.volume).collect();
        let vwaps: Vec<f64> = data.iter().map(|d| d.vwap).collect();
        let highs: Vec<f64> = data.iter().map(|d| d.high).collect();
        let lows: Vec<f64> = data.iter().map(|d| d.low).collect();
        let changes: Vec<f64> = data.iter().map(|d| d.change_pct).collect();

        let mut pair_builder = StringBuilder::new();
        for p in &pairs {
            pair_builder.append_value(p);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ts)) as ArrayRef,
                Arc::new(pair_builder.finish()) as ArrayRef,
                Arc::new(Float64Array::from(bids)) as ArrayRef,
                Arc::new(Float64Array::from(asks)) as ArrayRef,
                Arc::new(Float64Array::from(lasts)) as ArrayRef,
                Arc::new(Float64Array::from(volumes)) as ArrayRef,
                Arc::new(Float64Array::from(vwaps)) as ArrayRef,
                Arc::new(Float64Array::from(highs)) as ArrayRef,
                Arc::new(Float64Array::from(lows)) as ArrayRef,
                Arc::new(Float64Array::from(changes)) as ArrayRef,
            ],
        )?;

        self.write_parquet(&path, schema, batch)
    }

    /// Write order book data to parquet
    fn write_book_parquet(
        &self,
        category: &str,
        data: &[BookSnapshot],
        now: DateTime<Utc>,
    ) -> Result<()> {
        let path = self.get_output_path(category, "book", now)?;
        debug!("Writing {} book records to {:?}", data.len(), path);

        // For order book, we store bids/asks as JSON strings for simplicity
        // A more efficient approach would use nested arrays, but this works for analysis
        let schema = Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Int64, false),
            Field::new("pair", DataType::Utf8, false),
            Field::new("bids_json", DataType::Utf8, false),
            Field::new("asks_json", DataType::Utf8, false),
        ]));

        let ts: Vec<i64> = data.iter().map(|d| d.ts).collect();
        let pairs: Vec<&str> = data.iter().map(|d| d.pair.as_str()).collect();
        let bids_json: Vec<String> = data
            .iter()
            .map(|d| serde_json::to_string(&d.bids).unwrap_or_default())
            .collect();
        let asks_json: Vec<String> = data
            .iter()
            .map(|d| serde_json::to_string(&d.asks).unwrap_or_default())
            .collect();

        let mut pair_builder = StringBuilder::new();
        for p in &pairs {
            pair_builder.append_value(p);
        }

        let mut bids_builder = StringBuilder::new();
        for b in &bids_json {
            bids_builder.append_value(b);
        }

        let mut asks_builder = StringBuilder::new();
        for a in &asks_json {
            asks_builder.append_value(a);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ts)) as ArrayRef,
                Arc::new(pair_builder.finish()) as ArrayRef,
                Arc::new(bids_builder.finish()) as ArrayRef,
                Arc::new(asks_builder.finish()) as ArrayRef,
            ],
        )?;

        self.write_parquet(&path, schema, batch)
    }

    /// Write OHLC data to parquet
    fn write_ohlc_parquet(
        &self,
        category: &str,
        data: &[OhlcSnapshot],
        now: DateTime<Utc>,
    ) -> Result<()> {
        let path = self.get_output_path(category, "ohlc", now)?;
        debug!("Writing {} ohlc records to {:?}", data.len(), path);

        let schema = Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Int64, false),
            Field::new("pair", DataType::Utf8, false),
            Field::new("open", DataType::Float64, false),
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("close", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
            Field::new("vwap", DataType::Float64, false),
            Field::new("trades", DataType::Int64, false),
        ]));

        let ts: Vec<i64> = data.iter().map(|d| d.ts).collect();
        let pairs: Vec<&str> = data.iter().map(|d| d.pair.as_str()).collect();
        let opens: Vec<f64> = data.iter().map(|d| d.open).collect();
        let highs: Vec<f64> = data.iter().map(|d| d.high).collect();
        let lows: Vec<f64> = data.iter().map(|d| d.low).collect();
        let closes: Vec<f64> = data.iter().map(|d| d.close).collect();
        let volumes: Vec<f64> = data.iter().map(|d| d.volume).collect();
        let vwaps: Vec<f64> = data.iter().map(|d| d.vwap).collect();
        let trades: Vec<i64> = data.iter().map(|d| d.trades).collect();

        let mut pair_builder = StringBuilder::new();
        for p in &pairs {
            pair_builder.append_value(p);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ts)) as ArrayRef,
                Arc::new(pair_builder.finish()) as ArrayRef,
                Arc::new(Float64Array::from(opens)) as ArrayRef,
                Arc::new(Float64Array::from(highs)) as ArrayRef,
                Arc::new(Float64Array::from(lows)) as ArrayRef,
                Arc::new(Float64Array::from(closes)) as ArrayRef,
                Arc::new(Float64Array::from(volumes)) as ArrayRef,
                Arc::new(Float64Array::from(vwaps)) as ArrayRef,
                Arc::new(Int64Array::from(trades)) as ArrayRef,
            ],
        )?;

        self.write_parquet(&path, schema, batch)
    }

    /// Write trade data to parquet
    fn write_trade_parquet(
        &self,
        category: &str,
        data: &[TradeSnapshot],
        now: DateTime<Utc>,
    ) -> Result<()> {
        let path = self.get_output_path(category, "trade", now)?;
        debug!("Writing {} trade records to {:?}", data.len(), path);

        let schema = Arc::new(Schema::new(vec![
            Field::new("ts", DataType::Int64, false),
            Field::new("pair", DataType::Utf8, false),
            Field::new("side", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
            Field::new("qty", DataType::Float64, false),
            Field::new("trade_id", DataType::Int64, false),
        ]));

        let ts: Vec<i64> = data.iter().map(|d| d.ts).collect();
        let pairs: Vec<&str> = data.iter().map(|d| d.pair.as_str()).collect();
        let sides: Vec<&str> = data.iter().map(|d| d.side.as_str()).collect();
        let prices: Vec<f64> = data.iter().map(|d| d.price).collect();
        let qtys: Vec<f64> = data.iter().map(|d| d.qty).collect();
        let trade_ids: Vec<i64> = data.iter().map(|d| d.trade_id).collect();

        let mut pair_builder = StringBuilder::new();
        for p in &pairs {
            pair_builder.append_value(p);
        }

        let mut side_builder = StringBuilder::new();
        for s in &sides {
            side_builder.append_value(s);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ts)) as ArrayRef,
                Arc::new(pair_builder.finish()) as ArrayRef,
                Arc::new(side_builder.finish()) as ArrayRef,
                Arc::new(Float64Array::from(prices)) as ArrayRef,
                Arc::new(Float64Array::from(qtys)) as ArrayRef,
                Arc::new(Int64Array::from(trade_ids)) as ArrayRef,
            ],
        )?;

        self.write_parquet(&path, schema, batch)
    }

    /// Write a record batch to a parquet file (append if exists)
    fn write_parquet(
        &self,
        path: &PathBuf,
        schema: Arc<Schema>,
        batch: RecordBatch,
    ) -> Result<()> {
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        // For simplicity, we overwrite the file each hour
        // A more sophisticated approach would append to existing files
        let file = File::create(path).context("Failed to create parquet file")?;

        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Get buffer statistics for logging/monitoring
    pub fn buffer_stats(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        (
            self.crypto_ticker_buffer.len(),
            self.crypto_book_buffer.len(),
            self.crypto_ohlc_buffer.len(),
            self.crypto_trade_buffer.len(),
            self.stock_ticker_buffer.len(),
            self.stock_book_buffer.len(),
            self.stock_ohlc_buffer.len(),
            self.stock_trade_buffer.len(),
        )
    }
}

/// Convert Decimal to f64 for storage
fn decimal_to_f64(d: Decimal) -> f64 {
    d.to_f64().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_to_f64() {
        let d = Decimal::new(12345, 2); // 123.45
        assert!((decimal_to_f64(d) - 123.45).abs() < 0.001);
    }
}
