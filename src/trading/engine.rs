use crate::config::Config;
use crate::data::Ticker;
use crate::storage::Database;
use crate::trading::order::{Order, Position};
use crate::trading::paper::PaperTrader;
use anyhow::Result;
use rust_decimal::Decimal;
use std::sync::Arc;

pub enum TradingMode {
    Paper(PaperTrader),
    // Live trading would go here in the future
}

pub struct TradingEngine {
    mode: TradingMode,
    config: Config,
}

impl TradingEngine {
    pub async fn new(config: Config, db: Arc<Database>) -> Result<Self> {
        let mode = TradingMode::Paper(
            PaperTrader::new(config.trading.paper_balance, db).await?,
        );

        Ok(Self { mode, config })
    }

    pub fn is_paper_trading(&self) -> bool {
        matches!(self.mode, TradingMode::Paper(_))
    }

    pub fn balance(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.balance(),
        }
    }

    pub fn equity(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.equity(),
        }
    }

    pub fn initial_balance(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.initial_balance(),
        }
    }

    pub fn total_pnl(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.total_pnl(),
        }
    }

    pub fn total_pnl_pct(&self) -> Decimal {
        match &self.mode {
            TradingMode::Paper(trader) => trader.total_pnl_pct(),
        }
    }

    pub fn positions(&self) -> Vec<&Position> {
        match &self.mode {
            TradingMode::Paper(trader) => trader.positions().values().collect(),
        }
    }

    pub fn has_position(&self, pair: &str) -> bool {
        match &self.mode {
            TradingMode::Paper(trader) => trader.has_position(pair),
        }
    }

    pub fn get_position(&self, pair: &str) -> Option<&Position> {
        match &self.mode {
            TradingMode::Paper(trader) => trader.get_position(pair),
        }
    }

    pub async fn execute_order(&mut self, order: Order, ticker: &Ticker) -> Result<Order> {
        match &mut self.mode {
            TradingMode::Paper(trader) => trader.execute_order(order, ticker).await,
        }
    }

    pub fn update_positions(&mut self, ticker: &Ticker) {
        match &mut self.mode {
            TradingMode::Paper(trader) => trader.update_positions(ticker),
        }
    }

    pub fn max_position_value(&self) -> Decimal {
        let pct = Decimal::from_str_exact(&self.config.strategy.max_position_size.to_string())
            .unwrap_or(Decimal::from_str_exact("0.1").unwrap());
        self.balance() * pct
    }
}
