//! Trading engine - abstraction over paper and live trading

use crate::config::{Config, TradingMode as ConfigTradingMode};
use crate::data::Ticker;
use crate::storage::Database;
use crate::trading::live::LiveTrader;
use crate::trading::order::{Order, Position};
use crate::trading::paper::PaperTrader;
use anyhow::Result;
use rust_decimal::Decimal;
use std::sync::Arc;

/// Trading mode - either paper or live
pub enum TradingMode {
    Paper(PaperTrader),
    Live(LiveTrader),
}

/// Trading engine that abstracts over paper and live trading
pub struct TradingEngine {
    mode: TradingMode,
    config: Config,
}

impl TradingEngine {
    /// Create a new trading engine based on config
    pub async fn new(config: Config, db: Arc<Database>) -> Result<Self> {
        let mode = match config.trading.mode {
            ConfigTradingMode::Paper => {
                TradingMode::Paper(PaperTrader::new(config.trading.paper_balance, db).await?)
            }
            ConfigTradingMode::Live => {
                TradingMode::Live(LiveTrader::new(&config, db).await?)
            }
        };

        Ok(Self { mode, config })
    }

    /// Check if running in paper trading mode
    pub fn is_paper_trading(&self) -> bool {
        matches!(self.mode, TradingMode::Paper(_))
    }

    /// Get current balance (USD)
    pub fn balance(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.balance(),
            TradingMode::Live(trader) => trader.balance_usd(),
        }
    }

    /// Get current equity (balance + unrealized P&L)
    pub fn equity(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.equity(),
            TradingMode::Live(trader) => trader.equity(),
        }
    }

    /// Get initial balance
    pub fn initial_balance(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.initial_balance(),
            TradingMode::Live(trader) => trader.initial_balance(),
        }
    }

    /// Get total realized P&L
    pub fn total_pnl(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.total_pnl(),
            TradingMode::Live(trader) => trader.total_pnl(),
        }
    }

    /// Get total P&L percentage
    pub fn total_pnl_pct(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.total_pnl_pct(),
            TradingMode::Live(trader) => trader.total_pnl_pct(),
        }
    }

    /// Get all positions
    pub fn positions(&self) -> Vec<&Position> {
        match &self.mode {
            TradingMode::Paper(trader) => trader.positions().values().collect(),
            TradingMode::Live(trader) => trader.positions().values().collect(),
        }
    }

    /// Check if there's a position for a pair
    pub fn has_position(&self, pair: &str) -> bool {
        match &self.mode {
            TradingMode::Paper(trader) => trader.has_position(pair),
            TradingMode::Live(trader) => trader.has_position(pair),
        }
    }

    /// Get position for a pair
    pub fn get_position(&self, pair: &str) -> Option<&Position> {
        match &self.mode {
            TradingMode::Paper(trader) => trader.get_position(pair),
            TradingMode::Live(trader) => trader.get_position(pair),
        }
    }

    /// Execute an order
    pub async fn execute_order(&mut self, order: Order, ticker: &Ticker) -> Result<Order> {
        match &mut self.mode {
            TradingMode::Paper(trader) => trader.execute_order(order, ticker).await,
            TradingMode::Live(trader) => trader.execute_order(order, ticker).await,
        }
    }

    /// Update positions with current price
    pub fn update_positions(&mut self, ticker: &Ticker) {
        match &mut self.mode {
            TradingMode::Paper(trader) => trader.update_positions(ticker),
            TradingMode::Live(trader) => trader.update_positions(ticker),
        }
    }

    /// Get maximum position value based on config
    pub fn max_position_value(&self) -> Decimal {
        use rust_decimal::prelude::FromStr;
        let pct = Decimal::from_str(&self.config.strategy.max_position_size.to_string())
            .unwrap_or_else(|_| Decimal::from_str("0.1").unwrap());
        self.balance() * pct
    }
}
