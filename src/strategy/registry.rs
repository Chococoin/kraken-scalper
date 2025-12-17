//! Strategy Registry
//!
//! Factory for creating strategies by name.

use super::base::Strategy;
use super::breakout::{BreakoutConfig, BreakoutStrategy};
use super::mean_reversion::{MeanReversionConfig, MeanReversionStrategy};
use super::rsi::{RSIConfig, RSIStrategy};
use super::scalper::ScalperStrategy;
use super::trend_following::{TrendFollowingConfig, TrendFollowingStrategy};
use crate::config::StrategyConfig;

/// Available strategy names
pub const STRATEGY_NAMES: &[&str] = &["scalper", "meanrev", "trend", "breakout", "rsi"];

/// Create a strategy by name
pub fn create_strategy(name: &str, config: &StrategyConfig) -> Option<Box<dyn Strategy>> {
    match name.to_lowercase().as_str() {
        "scalper" => Some(Box::new(ScalperStrategy::new(config.clone()))),

        "meanrev" | "mean_reversion" | "meanreversion" => {
            Some(Box::new(MeanReversionStrategy::new(MeanReversionConfig::default())))
        }

        "trend" | "trend_following" | "trendfollowing" => {
            Some(Box::new(TrendFollowingStrategy::new(TrendFollowingConfig::default())))
        }

        "breakout" => Some(Box::new(BreakoutStrategy::new(BreakoutConfig::default()))),

        "rsi" => Some(Box::new(RSIStrategy::new(RSIConfig::default()))),

        _ => None,
    }
}

/// Get all available strategies
pub fn all_strategies(config: &StrategyConfig) -> Vec<Box<dyn Strategy>> {
    STRATEGY_NAMES
        .iter()
        .filter_map(|name| create_strategy(name, config))
        .collect()
}

/// Parse a comma-separated list of strategy names
pub fn parse_strategy_list(list: &str) -> Vec<String> {
    if list.to_lowercase() == "all" {
        return STRATEGY_NAMES.iter().map(|s| s.to_string()).collect();
    }

    list.split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}
