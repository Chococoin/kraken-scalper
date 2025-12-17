//! RSI Strategy
//!
//! Uses Relative Strength Index to identify overbought/oversold conditions.
//! Buys when RSI is oversold (< 30), sells when RSI is overbought (> 70).

use crate::data::{OrderBook, Ticker};
use crate::strategy::base::{Signal, Strategy};
use crate::strategy::indicators::rsi;
use crate::trading::order::{OrderSide, Position};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use tracing::debug;

/// Configuration for RSI strategy
#[derive(Debug, Clone)]
pub struct RSIConfig {
    pub rsi_period: usize,
    pub oversold_level: f64,
    pub overbought_level: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub position_size: Decimal,
}

impl Default for RSIConfig {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            oversold_level: 30.0,
            overbought_level: 70.0,
            stop_loss_pct: 1.0,
            take_profit_pct: 2.0,
            position_size: dec!(0.001),
        }
    }
}

pub struct RSIStrategy {
    config: RSIConfig,
    price_history: HashMap<String, Vec<Decimal>>,
}

impl RSIStrategy {
    pub fn new(config: RSIConfig) -> Self {
        Self {
            config,
            price_history: HashMap::new(),
        }
    }

    fn update_history(&mut self, pair: &str, price: Decimal) {
        let history = self.price_history.entry(pair.to_string()).or_default();
        history.push(price);

        // Keep enough for RSI calculation
        let max_history = self.config.rsi_period * 3;
        if history.len() > max_history {
            history.remove(0);
        }
    }

    fn get_rsi(&self, pair: &str) -> Option<Decimal> {
        let history = self.price_history.get(pair)?;
        rsi(history, self.config.rsi_period)
    }
}

impl Strategy for RSIStrategy {
    fn analyze(
        &mut self,
        ticker: &Ticker,
        _orderbook: Option<&OrderBook>,
        position: Option<&Position>,
    ) -> Signal {
        self.update_history(&ticker.pair, ticker.last);

        let Some(current_rsi) = self.get_rsi(&ticker.pair) else {
            return Signal::Hold;
        };

        let oversold = Decimal::try_from(self.config.oversold_level).unwrap_or(dec!(30));
        let overbought = Decimal::try_from(self.config.overbought_level).unwrap_or(dec!(70));

        // If we have a position
        if let Some(pos) = position {
            // Sell when RSI is overbought
            if current_rsi > overbought {
                debug!(
                    "{}: RSI overbought {} > {} - selling",
                    ticker.pair, current_rsi, overbought
                );
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            // Check take profit
            if self.should_take_profit(pos, ticker.last) {
                debug!("{}: Take profit triggered", ticker.pair);
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            // Check stop loss
            if self.should_stop_loss(pos, ticker.last) {
                debug!("{}: Stop loss triggered", ticker.pair);
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            return Signal::Hold;
        }

        // No position - check for oversold condition
        if current_rsi < oversold {
            debug!(
                "{}: RSI oversold {} < {} - buying",
                ticker.pair, current_rsi, oversold
            );
            return Signal::Buy {
                pair: ticker.pair.clone(),
                quantity: self.config.position_size,
            };
        }

        Signal::Hold
    }

    fn should_take_profit(&self, position: &Position, current_price: Decimal) -> bool {
        let take_profit_pct = Decimal::try_from(self.config.take_profit_pct).unwrap_or(dec!(2));

        let profit_pct = match position.side {
            OrderSide::Buy => {
                if position.entry_price.is_zero() {
                    Decimal::ZERO
                } else {
                    ((current_price - position.entry_price) / position.entry_price) * dec!(100)
                }
            }
            OrderSide::Sell => {
                if position.entry_price.is_zero() {
                    Decimal::ZERO
                } else {
                    ((position.entry_price - current_price) / position.entry_price) * dec!(100)
                }
            }
        };

        profit_pct >= take_profit_pct
    }

    fn should_stop_loss(&self, position: &Position, current_price: Decimal) -> bool {
        let stop_loss_pct = Decimal::try_from(self.config.stop_loss_pct).unwrap_or(dec!(1));

        let loss_pct = match position.side {
            OrderSide::Buy => {
                if position.entry_price.is_zero() {
                    Decimal::ZERO
                } else {
                    ((position.entry_price - current_price) / position.entry_price) * dec!(100)
                }
            }
            OrderSide::Sell => {
                if position.entry_price.is_zero() {
                    Decimal::ZERO
                } else {
                    ((current_price - position.entry_price) / position.entry_price) * dec!(100)
                }
            }
        };

        loss_pct >= stop_loss_pct
    }

    fn name(&self) -> &str {
        "RSI"
    }

    fn reset(&mut self) {
        self.price_history.clear();
    }
}
