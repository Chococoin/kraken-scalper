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
}

#[derive(Debug, Clone, Deserialize)]
pub struct KrakenConfig {
    pub api_key: String,
    pub api_secret: String,
    pub ws_url: String,
    pub ws_auth_url: String,
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
