//! Mean Reversion Strategy
//!
//! Buys when price falls significantly below the moving average,
//! sells when price returns to or exceeds the mean.

use crate::data::{OrderBook, Ticker};
use crate::strategy::base::{Signal, Strategy};
use crate::strategy::indicators::{sma, std_dev};
use crate::trading::order::{OrderSide, Position};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use tracing::debug;

/// Configuration for Mean Reversion strategy
#[derive(Debug, Clone)]
pub struct MeanReversionConfig {
    pub lookback_periods: usize,
    pub entry_std_devs: f64,
    pub exit_std_devs: f64,
    pub stop_loss_pct: f64,
    pub position_size: Decimal,
}

impl Default for MeanReversionConfig {
    fn default() -> Self {
        Self {
            lookback_periods: 20,
            entry_std_devs: 2.0,
            exit_std_devs: 0.5,
            stop_loss_pct: 1.0,
            position_size: dec!(0.001),
        }
    }
}

pub struct MeanReversionStrategy {
    config: MeanReversionConfig,
    price_history: HashMap<String, Vec<Decimal>>,
}

impl MeanReversionStrategy {
    pub fn new(config: MeanReversionConfig) -> Self {
        Self {
            config,
            price_history: HashMap::new(),
        }
    }

    fn update_history(&mut self, pair: &str, price: Decimal) {
        let history = self.price_history.entry(pair.to_string()).or_default();
        history.push(price);

        // Keep more than lookback for calculations
        let max_history = self.config.lookback_periods * 2;
        if history.len() > max_history {
            history.remove(0);
        }
    }

    fn get_bands(&self, pair: &str) -> Option<(Decimal, Decimal, Decimal)> {
        let history = self.price_history.get(pair)?;

        if history.len() < self.config.lookback_periods {
            return None;
        }

        let mean = sma(history, self.config.lookback_periods)?;
        let std = std_dev(history, self.config.lookback_periods)?;

        let entry_mult = Decimal::try_from(self.config.entry_std_devs).ok()?;
        let exit_mult = Decimal::try_from(self.config.exit_std_devs).ok()?;

        let lower_entry = mean - std * entry_mult;
        let upper_exit = mean + std * exit_mult;

        Some((lower_entry, mean, upper_exit))
    }
}

impl Strategy for MeanReversionStrategy {
    fn analyze(
        &mut self,
        ticker: &Ticker,
        _orderbook: Option<&OrderBook>,
        position: Option<&Position>,
    ) -> Signal {
        self.update_history(&ticker.pair, ticker.last);

        let Some((lower_entry, mean, _upper_exit)) = self.get_bands(&ticker.pair) else {
            return Signal::Hold;
        };

        // If we have a position, check exit
        if let Some(pos) = position {
            // Exit when price returns to mean
            if ticker.last >= mean {
                debug!(
                    "{}: Mean reversion exit - price {} >= mean {}",
                    ticker.pair, ticker.last, mean
                );
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

        // No position - check entry
        if ticker.last < lower_entry {
            debug!(
                "{}: Mean reversion entry - price {} < lower band {}",
                ticker.pair, ticker.last, lower_entry
            );
            return Signal::Buy {
                pair: ticker.pair.clone(),
                quantity: self.config.position_size,
            };
        }

        Signal::Hold
    }

    fn should_take_profit(&self, position: &Position, current_price: Decimal) -> bool {
        // Mean reversion exits at mean, not fixed take profit
        let history = match self.price_history.get(&position.pair) {
            Some(h) => h,
            None => return false,
        };

        let mean = match sma(history, self.config.lookback_periods) {
            Some(m) => m,
            None => return false,
        };

        current_price >= mean
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
        "MeanReversion"
    }

    fn reset(&mut self) {
        self.price_history.clear();
    }
}
