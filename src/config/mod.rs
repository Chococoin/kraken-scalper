use anyhow::{Context, Result};
use config::{Config as ConfigLoader, File};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub kraken: KrakenConfig,
    pub trading: TradingConfig,
    pub strategy: StrategyConfig,
    pub recording: RecordingConfig,
    pub ui: UiConfig,
    pub database: DatabaseConfig,
    #[serde(default)]
    pub backtest: BacktestConfigFile,
    #[serde(default)]
    pub huggingface: HuggingFaceConfig,
    #[serde(default)]
    pub kraken_ohlc: KrakenOhlcConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KrakenConfig {
    pub api_key: String,
    pub api_secret: String,
    pub ws_url: String,
    pub ws_auth_url: String,
    #[serde(default = "default_rest_url")]
    pub rest_url: String,
}

fn default_rest_url() -> String {
    "https://api.kraken.com".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TradingConfig {
    pub crypto_pairs: Vec<String>,
    pub stock_pairs: Vec<String>,
    pub mode: TradingMode,
    pub paper_balance: f64,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TradingMode {
    Paper,
    Live,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StrategyConfig {
    pub min_spread_pct: f64,
    pub take_profit_pct: f64,
    pub stop_loss_pct: f64,
    pub max_position_size: f64,
    pub cooldown_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecordingConfig {
    pub enabled: bool,
    pub data_dir: String,
    pub crypto_sample_interval_secs: u64,
    pub stock_sample_interval_secs: u64,
    pub crypto_book_depth: u32,
    pub stock_book_depth: u32,
    pub flush_interval_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UiConfig {
    pub refresh_rate_ms: u64,
    pub chart_candles: usize,
    pub chart_timeframe: String,
}

impl UiConfig {
    /// Convert chart_timeframe string to Kraken API interval (in minutes)
    /// Valid values: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
    pub fn chart_interval(&self) -> u32 {
        match self.chart_timeframe.to_lowercase().as_str() {
            "1m" | "1min" => 1,
            "5m" | "5min" => 5,
            "15m" | "15min" => 15,
            "30m" | "30min" => 30,
            "1h" | "60m" => 60,
            "4h" | "240m" => 240,
            "1d" | "1day" | "1440m" => 1440,
            "1w" | "1week" => 10080,
            "15d" => 21600,
            _ => 1, // Default to 1 minute
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BacktestConfigFile {
    pub default_pair: String,
    pub initial_balance: f64,
    pub commission_pct: f64,
    pub slippage_pct: f64,
}

impl Default for BacktestConfigFile {
    fn default() -> Self {
        Self {
            default_pair: "BTC/USD".to_string(),
            initial_balance: 10000.0,
            commission_pct: 0.26,
            slippage_pct: 0.05,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceConfig {
    pub enabled: bool,
    pub repo_id: String,
    pub upload_interval_secs: u64,
    pub upload_delay_hours: u32,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            repo_id: String::new(),
            upload_interval_secs: 3600,
            upload_delay_hours: 1,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOhlcSchedule {
    pub interval: u32,           // Interval in minutes (1, 5, 15, 60, 1440)
    pub fetch_every_hours: u32,  // How often to fetch this interval
}

#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOhlcConfig {
    pub enabled: bool,
    pub pairs: Vec<String>,           // Pairs to capture
    pub request_delay_ms: u64,        // Delay between requests (min 1000ms)
    pub overlap_minutes: u32,         // Overlap to ensure no gaps (default 60 = 1 hour)
    pub schedules: Vec<KrakenOhlcSchedule>,
}

impl Default for KrakenOhlcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            pairs: vec!["BTC/USD".into(), "ETH/USD".into()],
            request_delay_ms: 1100,
            overlap_minutes: 5, // 5 minutes overlap to ensure no gaps
            schedules: vec![
                KrakenOhlcSchedule { interval: 1, fetch_every_hours: 12 },
            ],
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        let settings = ConfigLoader::builder()
            .add_source(File::from(config_path.clone()).required(false))
            .add_source(config::Environment::with_prefix("SCALPER").separator("_"))
            .build()
            .context("Failed to build configuration")?;

        settings
            .try_deserialize()
            .context("Failed to deserialize configuration")
    }

    fn config_path() -> PathBuf {
        // Try local config first, then default
        let local_config = PathBuf::from("config/default.toml");
        if local_config.exists() {
            return local_config;
        }

        // Fallback to executable directory
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let exe_config = exe_dir.join("config/default.toml");
                if exe_config.exists() {
                    return exe_config;
                }
            }
        }

        local_config
    }

    pub fn is_paper_trading(&self) -> bool {
        self.trading.mode == TradingMode::Paper
    }

    /// Get all pairs (crypto + stocks)
    pub fn all_pairs(&self) -> Vec<String> {
        let mut pairs = self.trading.crypto_pairs.clone();
        pairs.extend(self.trading.stock_pairs.clone());
        pairs
    }

    /// Check if a pair is a crypto pair
    pub fn is_crypto_pair(&self, pair: &str) -> bool {
        self.trading.crypto_pairs.contains(&pair.to_string())
    }

    /// Check if a pair is a stock pair
    pub fn is_stock_pair(&self, pair: &str) -> bool {
        self.trading.stock_pairs.contains(&pair.to_string())
    }
}
