//! Data loader for reading historical market data from Parquet files

use crate::data::{Candle, OrderBook, Ticker};
use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// A snapshot of market data at a specific point in time
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub ts: DateTime<Utc>,
    pub pair: String,
    pub ticker: Option<Ticker>,
    pub orderbook: Option<OrderBook>,
    pub candle: Option<Candle>,
}

impl MarketSnapshot {
    pub fn new(ts: DateTime<Utc>, pair: String) -> Self {
        Self {
            ts,
            pair,
            ticker: None,
            orderbook: None,
            candle: None,
        }
    }
}

/// Loader for historical market data from Parquet files
pub struct DataLoader {
    data_dir: PathBuf,
}

impl DataLoader {
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
        }
    }

    /// Load market snapshots for a pair within a date range
    pub fn load_range(
        &self,
        pair: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MarketSnapshot>> {
        let mut snapshots: BTreeMap<i64, MarketSnapshot> = BTreeMap::new();

        // Load ticker data
        self.load_tickers(pair, start, end, &mut snapshots)?;

        // Load orderbook data
        self.load_orderbooks(pair, start, end, &mut snapshots)?;

        // Load OHLC data
        self.load_ohlc(pair, start, end, &mut snapshots)?;

        // Convert to sorted vec
        Ok(snapshots.into_values().collect())
    }

    /// Get list of Parquet files for a data type within date range
    fn get_parquet_files(
        &self,
        category: &str,
        data_type: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<PathBuf> {
        let mut files = Vec::new();

        let start_date = start.date_naive();
        let end_date = end.date_naive();

        let mut current_date = start_date;
        while current_date <= end_date {
            let date_str = current_date.format("%Y-%m-%d").to_string();
            let date_dir = self
                .data_dir
                .join(category)
                .join(data_type)
                .join(&date_str);

            if date_dir.exists() {
                // Get all hour files
                if let Ok(entries) = std::fs::read_dir(&date_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().map_or(false, |e| e == "parquet") {
                            files.push(path);
                        }
                    }
                }
            }

            current_date = current_date
                .succ_opt()
                .unwrap_or(current_date);
        }

        files.sort();
        files
    }

    /// Load ticker data from Parquet files
    fn load_tickers(
        &self,
        pair: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let files = self.get_parquet_files("crypto", "ticker", start, end);

        for file_path in files {
            self.load_ticker_file(&file_path, pair, start, end, snapshots)?;
        }

        Ok(())
    }

    fn load_ticker_file(
        &self,
        path: &Path,
        pair: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let file = File::open(path).context("Failed to open ticker parquet file")?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();

        for batch_result in reader {
            let batch: RecordBatch = batch_result?;
            self.process_ticker_batch(&batch, pair, start_ms, end_ms, snapshots)?;
        }

        Ok(())
    }

    fn process_ticker_batch(
        &self,
        batch: &RecordBatch,
        pair: &str,
        start_ms: i64,
        end_ms: i64,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let ts_col = batch
            .column_by_name("ts")
            .context("Missing ts column")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("ts column is not Int64")?;

        let pair_col = batch
            .column_by_name("pair")
            .context("Missing pair column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("pair column is not String")?;

        let bid_col = get_f64_column(batch, "bid")?;
        let ask_col = get_f64_column(batch, "ask")?;
        let last_col = get_f64_column(batch, "last")?;
        let volume_col = get_f64_column(batch, "volume")?;
        let vwap_col = get_f64_column(batch, "vwap")?;
        let high_col = get_f64_column(batch, "high")?;
        let low_col = get_f64_column(batch, "low")?;
        let change_col = get_f64_column(batch, "change_pct")?;

        for i in 0..batch.num_rows() {
            let row_pair = pair_col.value(i);
            if row_pair != pair {
                continue;
            }

            let ts = ts_col.value(i);
            if ts < start_ms || ts > end_ms {
                continue;
            }

            let ticker = Ticker {
                pair: pair.to_string(),
                bid: Decimal::try_from(bid_col.value(i)).unwrap_or_default(),
                ask: Decimal::try_from(ask_col.value(i)).unwrap_or_default(),
                last: Decimal::try_from(last_col.value(i)).unwrap_or_default(),
                volume_24h: Decimal::try_from(volume_col.value(i)).unwrap_or_default(),
                vwap_24h: Decimal::try_from(vwap_col.value(i)).unwrap_or_default(),
                high_24h: Decimal::try_from(high_col.value(i)).unwrap_or_default(),
                low_24h: Decimal::try_from(low_col.value(i)).unwrap_or_default(),
                change_24h: Decimal::try_from(change_col.value(i)).unwrap_or_default(),
                updated_at: Utc.timestamp_millis_opt(ts).unwrap(),
            };

            let snapshot = snapshots
                .entry(ts)
                .or_insert_with(|| MarketSnapshot::new(ticker.updated_at, pair.to_string()));
            snapshot.ticker = Some(ticker);
        }

        Ok(())
    }

    /// Load orderbook data from Parquet files
    fn load_orderbooks(
        &self,
        pair: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let files = self.get_parquet_files("crypto", "book", start, end);

        for file_path in files {
            self.load_orderbook_file(&file_path, pair, start, end, snapshots)?;
        }

        Ok(())
    }

    fn load_orderbook_file(
        &self,
        path: &Path,
        pair: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let file = File::open(path).context("Failed to open book parquet file")?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();

        for batch_result in reader {
            let batch: RecordBatch = batch_result?;
            self.process_orderbook_batch(&batch, pair, start_ms, end_ms, snapshots)?;
        }

        Ok(())
    }

    fn process_orderbook_batch(
        &self,
        batch: &RecordBatch,
        pair: &str,
        start_ms: i64,
        end_ms: i64,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let ts_col = batch
            .column_by_name("ts")
            .context("Missing ts column")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("ts column is not Int64")?;

        let pair_col = batch
            .column_by_name("pair")
            .context("Missing pair column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("pair column is not String")?;

        let bids_col = batch
            .column_by_name("bids_json")
            .context("Missing bids_json column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("bids_json column is not String")?;

        let asks_col = batch
            .column_by_name("asks_json")
            .context("Missing asks_json column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("asks_json column is not String")?;

        for i in 0..batch.num_rows() {
            let row_pair = pair_col.value(i);
            if row_pair != pair {
                continue;
            }

            let ts = ts_col.value(i);
            if ts < start_ms || ts > end_ms {
                continue;
            }

            // Parse orderbook from JSON
            let bids_json = bids_col.value(i);
            let asks_json = asks_col.value(i);

            let mut orderbook = OrderBook::new(pair);
            orderbook.updated_at = Utc.timestamp_millis_opt(ts).unwrap();

            // Parse bids: [[price, qty], ...]
            if let Ok(bids) = serde_json::from_str::<Vec<(f64, f64)>>(bids_json) {
                for (price, qty) in bids {
                    let price_dec = Decimal::try_from(price).unwrap_or_default();
                    let qty_dec = Decimal::try_from(qty).unwrap_or_default();
                    orderbook.bids.insert(price_dec, qty_dec);
                }
            }

            // Parse asks
            if let Ok(asks) = serde_json::from_str::<Vec<(f64, f64)>>(asks_json) {
                for (price, qty) in asks {
                    let price_dec = Decimal::try_from(price).unwrap_or_default();
                    let qty_dec = Decimal::try_from(qty).unwrap_or_default();
                    orderbook.asks.insert(price_dec, qty_dec);
                }
            }

            let snapshot = snapshots
                .entry(ts)
                .or_insert_with(|| MarketSnapshot::new(orderbook.updated_at, pair.to_string()));
            snapshot.orderbook = Some(orderbook);
        }

        Ok(())
    }

    /// Load OHLC data from Parquet files
    fn load_ohlc(
        &self,
        pair: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let files = self.get_parquet_files("crypto", "ohlc", start, end);

        for file_path in files {
            self.load_ohlc_file(&file_path, pair, start, end, snapshots)?;
        }

        Ok(())
    }

    fn load_ohlc_file(
        &self,
        path: &Path,
        pair: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let file = File::open(path).context("Failed to open ohlc parquet file")?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();

        for batch_result in reader {
            let batch: RecordBatch = batch_result?;
            self.process_ohlc_batch(&batch, pair, start_ms, end_ms, snapshots)?;
        }

        Ok(())
    }

    fn process_ohlc_batch(
        &self,
        batch: &RecordBatch,
        pair: &str,
        start_ms: i64,
        end_ms: i64,
        snapshots: &mut BTreeMap<i64, MarketSnapshot>,
    ) -> Result<()> {
        let ts_col = batch
            .column_by_name("ts")
            .context("Missing ts column")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("ts column is not Int64")?;

        let pair_col = batch
            .column_by_name("pair")
            .context("Missing pair column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("pair column is not String")?;

        let open_col = get_f64_column(batch, "open")?;
        let high_col = get_f64_column(batch, "high")?;
        let low_col = get_f64_column(batch, "low")?;
        let close_col = get_f64_column(batch, "close")?;
        let volume_col = get_f64_column(batch, "volume")?;
        let vwap_col = get_f64_column(batch, "vwap")?;
        let trades_col = batch
            .column_by_name("trades")
            .context("Missing trades column")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("trades column is not Int64")?;

        for i in 0..batch.num_rows() {
            let row_pair = pair_col.value(i);
            if row_pair != pair {
                continue;
            }

            let ts = ts_col.value(i);
            if ts < start_ms || ts > end_ms {
                continue;
            }

            let candle = Candle {
                timestamp: Utc.timestamp_millis_opt(ts).unwrap(),
                open: Decimal::try_from(open_col.value(i)).unwrap_or_default(),
                high: Decimal::try_from(high_col.value(i)).unwrap_or_default(),
                low: Decimal::try_from(low_col.value(i)).unwrap_or_default(),
                close: Decimal::try_from(close_col.value(i)).unwrap_or_default(),
                volume: Decimal::try_from(volume_col.value(i)).unwrap_or_default(),
                vwap: Decimal::try_from(vwap_col.value(i)).unwrap_or_default(),
                trades: trades_col.value(i) as u64,
            };

            let snapshot = snapshots
                .entry(ts)
                .or_insert_with(|| MarketSnapshot::new(candle.timestamp, pair.to_string()));
            snapshot.candle = Some(candle);
        }

        Ok(())
    }

    /// Get available date range for a pair
    pub fn get_available_dates(&self, category: &str) -> Vec<NaiveDate> {
        let mut dates = Vec::new();
        let ticker_dir = self.data_dir.join(category).join("ticker");

        if let Ok(entries) = std::fs::read_dir(&ticker_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        if let Ok(date) = NaiveDate::parse_from_str(name, "%Y-%m-%d") {
                            dates.push(date);
                        }
                    }
                }
            }
        }

        dates.sort();
        dates
    }
}

/// Helper to extract f64 column from record batch
fn get_f64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    batch
        .column_by_name(name)
        .context(format!("Missing {} column", name))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .context(format!("{} column is not Float64", name))
}
