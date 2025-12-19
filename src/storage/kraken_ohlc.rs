//! Kraken REST API OHLC data fetcher with rate limiting and incremental storage
//!
//! Fetches historical OHLC data from Kraken's public REST API and stores it
//! in parquet files, merging with existing data to build up historical records.

use crate::config::KrakenOhlcConfig;
use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// OHLC candle data from Kraken API
#[derive(Debug, Clone)]
pub struct OhlcCandle {
    pub time: i64,      // Unix timestamp in seconds
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: f64,
    pub trades: i64,
}

/// Kraken OHLC fetcher with rate limiting and incremental storage
pub struct KrakenOhlcFetcher {
    config: KrakenOhlcConfig,
    client: reqwest::Client,
    data_dir: PathBuf,
    last_request: Instant,
    last_fetch_times: HashMap<(String, u32), DateTime<Utc>>,
}

impl KrakenOhlcFetcher {
    /// Create a new fetcher
    pub fn new(config: KrakenOhlcConfig, data_dir: &str) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            data_dir: PathBuf::from(data_dir),
            last_request: Instant::now() - Duration::from_secs(10), // Allow immediate first request
            last_fetch_times: HashMap::new(),
        }
    }

    /// Calculate effective fetch interval accounting for overlap
    /// Fetches slightly earlier than scheduled to ensure overlap for gap-free merging
    fn effective_fetch_interval(&self, schedule_hours: u32) -> chrono::Duration {
        let total_minutes = (schedule_hours as i64) * 60;
        let overlap = self.config.overlap_minutes as i64;
        // Subtract overlap to fetch earlier, ensuring we have overlapping data
        let effective_minutes = (total_minutes - overlap).max(60); // At least 1 hour
        chrono::Duration::minutes(effective_minutes)
    }

    /// Check if any fetch is due based on schedules
    pub fn should_fetch(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        let now = Utc::now();
        for pair in &self.config.pairs {
            for schedule in &self.config.schedules {
                let key = (pair.clone(), schedule.interval);
                let fetch_interval = self.effective_fetch_interval(schedule.fetch_every_hours);

                match self.last_fetch_times.get(&key) {
                    None => return true, // Never fetched
                    Some(last) if now.signed_duration_since(*last) >= fetch_interval => return true,
                    _ => continue,
                }
            }
        }
        false
    }

    /// Get list of (pair, interval) that need fetching
    fn get_pending_fetches(&self) -> Vec<(String, u32)> {
        let mut pending = Vec::new();
        let now = Utc::now();

        for pair in &self.config.pairs {
            for schedule in &self.config.schedules {
                let key = (pair.clone(), schedule.interval);
                let fetch_interval = self.effective_fetch_interval(schedule.fetch_every_hours);

                let should_fetch = match self.last_fetch_times.get(&key) {
                    None => true,
                    Some(last) => now.signed_duration_since(*last) >= fetch_interval,
                };

                if should_fetch {
                    pending.push(key);
                }
            }
        }
        pending
    }

    /// Rate-limited wait between API requests
    async fn rate_limit(&mut self) {
        let min_interval = Duration::from_millis(self.config.request_delay_ms);
        let elapsed = self.last_request.elapsed();
        if elapsed < min_interval {
            tokio::time::sleep(min_interval - elapsed).await;
        }
        self.last_request = Instant::now();
    }

    /// Fetch OHLC data from Kraken API
    async fn fetch_ohlc(&mut self, pair: &str, interval: u32) -> Result<Vec<OhlcCandle>> {
        self.rate_limit().await;

        let pair_clean = pair.replace("/", "");
        let url = format!(
            "https://api.kraken.com/0/public/OHLC?pair={}&interval={}",
            pair_clean, interval
        );

        debug!("Fetching OHLC: {} interval={}", pair, interval);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch OHLC from Kraken")?;

        let data: Value = response
            .json()
            .await
            .context("Failed to parse Kraken response")?;

        // Check for errors
        if let Some(errors) = data.get("error").and_then(|e| e.as_array()) {
            if !errors.is_empty() {
                return Err(anyhow::anyhow!("Kraken API error: {:?}", errors));
            }
        }

        // Parse result
        let result = data
            .get("result")
            .ok_or_else(|| anyhow::anyhow!("No result in response"))?;

        // Find the OHLC data (first key that's not "last")
        let ohlc_data = result
            .as_object()
            .and_then(|obj| obj.iter().find(|(k, _)| *k != "last").map(|(_, v)| v))
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("No OHLC data found"))?;

        let candles: Vec<OhlcCandle> = ohlc_data
            .iter()
            .filter_map(|candle| {
                let arr = candle.as_array()?;
                if arr.len() < 8 {
                    return None;
                }
                Some(OhlcCandle {
                    time: arr[0].as_i64()?,
                    open: arr[1].as_str()?.parse().ok()?,
                    high: arr[2].as_str()?.parse().ok()?,
                    low: arr[3].as_str()?.parse().ok()?,
                    close: arr[4].as_str()?.parse().ok()?,
                    vwap: arr[5].as_str()?.parse().ok()?,
                    volume: arr[6].as_str()?.parse().ok()?,
                    trades: arr[7].as_i64()?,
                })
            })
            .collect();

        debug!(
            "Fetched {} candles for {} interval={}",
            candles.len(),
            pair,
            interval
        );
        Ok(candles)
    }

    /// Load existing parquet data for a pair/interval
    fn load_existing(&self, pair: &str, interval: u32) -> Result<Vec<OhlcCandle>> {
        let path = self.get_parquet_path(pair, interval);

        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut candles = Vec::new();
        for batch in reader {
            let batch = batch?;
            let ts_col = batch
                .column_by_name("ts")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let open_col = batch
                .column_by_name("open")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let high_col = batch
                .column_by_name("high")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let low_col = batch
                .column_by_name("low")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let close_col = batch
                .column_by_name("close")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let volume_col = batch
                .column_by_name("volume")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let vwap_col = batch
                .column_by_name("vwap")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let trades_col = batch
                .column_by_name("trades")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

            if let (
                Some(ts),
                Some(open),
                Some(high),
                Some(low),
                Some(close),
                Some(vol),
                Some(vwap),
                Some(trades),
            ) = (
                ts_col, open_col, high_col, low_col, close_col, volume_col, vwap_col, trades_col,
            ) {
                for i in 0..batch.num_rows() {
                    candles.push(OhlcCandle {
                        time: ts.value(i) / 1000, // Convert ms to seconds
                        open: open.value(i),
                        high: high.value(i),
                        low: low.value(i),
                        close: close.value(i),
                        volume: vol.value(i),
                        vwap: vwap.value(i),
                        trades: trades.value(i),
                    });
                }
            }
        }

        debug!(
            "Loaded {} existing candles for {} interval={}",
            candles.len(),
            pair,
            interval
        );
        Ok(candles)
    }

    /// Merge new candles with existing, deduplicating by timestamp
    /// Returns (merged_candles, overlap_count) where overlap_count is the number of duplicates found
    fn merge_candles(&self, existing: Vec<OhlcCandle>, new: Vec<OhlcCandle>) -> (Vec<OhlcCandle>, usize) {
        let existing_count = existing.len();
        let new_count = new.len();

        let mut by_time: HashMap<i64, OhlcCandle> = HashMap::new();

        // Add existing (will be overwritten by new if same timestamp)
        for candle in existing {
            by_time.insert(candle.time, candle);
        }

        // Add new (overwrites existing with same timestamp - newer data is more accurate)
        for candle in new {
            by_time.insert(candle.time, candle);
        }

        // Calculate overlap (duplicates that were merged)
        let merged_count = by_time.len();
        let overlap = (existing_count + new_count).saturating_sub(merged_count);

        // Sort by time
        let mut result: Vec<_> = by_time.into_values().collect();
        result.sort_by_key(|c| c.time);

        (result, overlap)
    }

    /// Get parquet file path for a pair/interval
    fn get_parquet_path(&self, pair: &str, interval: u32) -> PathBuf {
        let pair_safe = pair.replace("/", "_");
        self.data_dir
            .join("kraken")
            .join(&pair_safe)
            .join(format!("{}.parquet", interval))
    }

    /// Save candles to parquet file
    fn save_parquet(&self, pair: &str, interval: u32, candles: &[OhlcCandle]) -> Result<()> {
        if candles.is_empty() {
            return Ok(());
        }

        let path = self.get_parquet_path(pair, interval);

        // Create directory
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

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

        // Build arrays
        let ts_array: ArrayRef = Arc::new(Int64Array::from(
            candles.iter().map(|c| c.time * 1000).collect::<Vec<_>>(), // Store as ms
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
        let vwap_array: ArrayRef = Arc::new(Float64Array::from(
            candles.iter().map(|c| c.vwap).collect::<Vec<_>>(),
        ));
        let trades_array: ArrayRef = Arc::new(Int64Array::from(
            candles.iter().map(|c| c.trades).collect::<Vec<_>>(),
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                ts_array,
                pair_array,
                open_array,
                high_array,
                low_array,
                close_array,
                volume_array,
                vwap_array,
                trades_array,
            ],
        )?;

        let file = File::create(&path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        info!("Saved {} candles to {}", candles.len(), path.display());
        Ok(())
    }

    /// Fetch and save OHLC for a pair/interval, merging with existing data
    async fn fetch_and_save(&mut self, pair: &str, interval: u32) -> Result<usize> {
        let new_candles = self.fetch_ohlc(pair, interval).await?;
        let new_count = new_candles.len();

        if new_candles.is_empty() {
            return Ok(0);
        }

        // Load existing data
        let existing = self.load_existing(pair, interval)?;
        let existing_count = existing.len();

        // Merge and deduplicate
        let (merged, overlap) = self.merge_candles(existing, new_candles);
        let added = merged.len().saturating_sub(existing_count);

        // Log merge stats
        if existing_count > 0 {
            info!(
                "{} interval={}: {} existing + {} new = {} total ({} overlap, {} added)",
                pair, interval, existing_count, new_count, merged.len(), overlap, added
            );
        }

        // Save merged data
        self.save_parquet(pair, interval, &merged)?;

        // Update last fetch time
        self.last_fetch_times
            .insert((pair.to_string(), interval), Utc::now());

        Ok(merged.len())
    }

    /// Fetch all pending OHLC data (call this from recorder main loop)
    pub async fn fetch_all_pending(&mut self) -> Result<usize> {
        let pending = self.get_pending_fetches();

        if pending.is_empty() {
            return Ok(0);
        }

        info!("Fetching {} pending OHLC requests", pending.len());

        let mut total_candles = 0;
        let mut errors = 0;

        for (pair, interval) in pending {
            match self.fetch_and_save(&pair, interval).await {
                Ok(count) => {
                    total_candles += count;
                }
                Err(e) => {
                    error!(
                        "Failed to fetch OHLC for {} interval={}: {}",
                        pair, interval, e
                    );
                    errors += 1;
                }
            }
        }

        if errors > 0 {
            warn!("OHLC fetch completed with {} errors", errors);
        }

        Ok(total_candles)
    }

    /// Get statistics (configured pairs, fetched records)
    pub fn stats(&self) -> (usize, usize) {
        (self.config.pairs.len(), self.last_fetch_times.len())
    }
}
