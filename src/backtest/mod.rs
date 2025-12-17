//! Backtesting module for historical data replay and strategy evaluation

mod comparator;
mod engine;
mod loader;
mod metrics;

pub use comparator::{compare_strategies, ComparisonResult};
pub use engine::{BacktestConfig, BacktestEngine, BacktestResult, TradeRecord};
pub use loader::{DataLoader, MarketSnapshot};
pub use metrics::{MetricsCollector, PerformanceMetrics};
