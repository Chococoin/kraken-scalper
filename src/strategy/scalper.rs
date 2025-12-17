use crate::config::StrategyConfig;
use crate::data::{OrderBook, Ticker};
use crate::strategy::base::{Signal, Strategy};
use crate::trading::order::{OrderSide, Position};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use tracing::debug;

/// Simple scalping strategy based on spread and momentum
pub struct ScalperStrategy {
    config: StrategyConfig,
    last_trade_time: HashMap<String, DateTime<Utc>>,
    price_history: HashMap<String, Vec<Decimal>>,
    max_history: usize,
}

impl ScalperStrategy {
    pub fn new(config: StrategyConfig) -> Self {
        Self {
            config,
            last_trade_time: HashMap::new(),
            price_history: HashMap::new(),
            max_history: 20,
        }
    }

    fn can_trade(&self, pair: &str) -> bool {
        if let Some(last_time) = self.last_trade_time.get(pair) {
            let elapsed = Utc::now().signed_duration_since(*last_time);
            elapsed.num_seconds() >= self.config.cooldown_seconds as i64
        } else {
            true
        }
    }

    fn record_trade(&mut self, pair: &str) {
        self.last_trade_time.insert(pair.to_string(), Utc::now());
    }

    fn update_price_history(&mut self, pair: &str, price: Decimal) {
        let history = self
            .price_history
            .entry(pair.to_string())
            .or_insert_with(Vec::new);

        history.push(price);

        if history.len() > self.max_history {
            history.remove(0);
        }
    }

    fn calculate_momentum(&self, pair: &str) -> Option<Decimal> {
        let history = self.price_history.get(pair)?;

        if history.len() < 5 {
            return None;
        }

        // Simple momentum: (current - average of last 5) / average
        let recent: Vec<_> = history.iter().rev().take(5).collect();
        let sum: Decimal = recent.iter().copied().copied().sum();
        let avg = sum / Decimal::from(recent.len());

        if avg.is_zero() {
            return None;
        }

        let current = *recent[0];
        Some(((current - avg) / avg) * dec!(100))
    }

    fn check_entry_conditions(
        &self,
        ticker: &Ticker,
        _orderbook: Option<&OrderBook>,
    ) -> Option<Signal> {
        // Check spread condition
        let spread_pct = ticker.spread_pct();
        let min_spread = Decimal::from_str_exact(&self.config.min_spread_pct.to_string()).ok()?;

        if spread_pct < min_spread {
            debug!(
                "{}: Spread too tight ({:.4}% < {:.4}%)",
                ticker.pair, spread_pct, min_spread
            );
            return None;
        }

        // Check momentum
        let momentum = self.calculate_momentum(&ticker.pair).unwrap_or(Decimal::ZERO);

        // Entry condition: positive momentum and good spread
        if momentum > dec!(0.05) {
            debug!(
                "{}: Entry signal - momentum: {:.4}%, spread: {:.4}%",
                ticker.pair, momentum, spread_pct
            );

            // Calculate position size (we'll use a simple fixed percentage)
            let quantity = dec!(0.001); // Small size for scalping

            return Some(Signal::Buy {
                pair: ticker.pair.clone(),
                quantity,
            });
        }

        None
    }
}

impl Strategy for ScalperStrategy {
    fn analyze(
        &mut self,
        ticker: &Ticker,
        orderbook: Option<&OrderBook>,
        position: Option<&Position>,
    ) -> Signal {
        // Update price history
        self.update_price_history(&ticker.pair, ticker.last);

        // Check cooldown
        if !self.can_trade(&ticker.pair) {
            return Signal::Hold;
        }

        // If we have a position, check exit conditions
        if let Some(pos) = position {
            // Check take profit
            if self.should_take_profit(pos, ticker.last) {
                debug!(
                    "{}: Take profit triggered at {:.2}",
                    ticker.pair, ticker.last
                );
                self.record_trade(&ticker.pair);
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            // Check stop loss
            if self.should_stop_loss(pos, ticker.last) {
                debug!("{}: Stop loss triggered at {:.2}", ticker.pair, ticker.last);
                self.record_trade(&ticker.pair);
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            // Hold position
            return Signal::Hold;
        }

        // No position - check entry conditions
        if let Some(signal) = self.check_entry_conditions(ticker, orderbook) {
            self.record_trade(&ticker.pair);
            return signal;
        }

        Signal::Hold
    }

    fn should_take_profit(&self, position: &Position, current_price: Decimal) -> bool {
        let take_profit_pct =
            Decimal::from_str_exact(&self.config.take_profit_pct.to_string()).unwrap_or(dec!(0.15));

        let price_change_pct = match position.side {
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

        price_change_pct >= take_profit_pct
    }

    fn should_stop_loss(&self, position: &Position, current_price: Decimal) -> bool {
        let stop_loss_pct =
            Decimal::from_str_exact(&self.config.stop_loss_pct.to_string()).unwrap_or(dec!(0.1));

        let price_change_pct = match position.side {
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

        price_change_pct >= stop_loss_pct
    }

    fn name(&self) -> &str {
        "Scalper"
    }

    fn reset(&mut self) {
        self.last_trade_time.clear();
        self.price_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> StrategyConfig {
        StrategyConfig {
            min_spread_pct: 0.1,
            take_profit_pct: 0.15,
            stop_loss_pct: 0.1,
            max_position_size: 0.1,
            cooldown_seconds: 30,
        }
    }

    #[test]
    fn test_take_profit() {
        let strategy = ScalperStrategy::new(test_config());
        let position = Position::new(1, "BTC/USD", OrderSide::Buy, dec!(100), dec!(1));

        // 0.15% gain should trigger take profit
        assert!(strategy.should_take_profit(&position, dec!(100.15)));

        // 0.10% gain should not trigger
        assert!(!strategy.should_take_profit(&position, dec!(100.10)));
    }

    #[test]
    fn test_stop_loss() {
        let strategy = ScalperStrategy::new(test_config());
        let position = Position::new(1, "BTC/USD", OrderSide::Buy, dec!(100), dec!(1));

        // 0.10% loss should trigger stop loss
        assert!(strategy.should_stop_loss(&position, dec!(99.90)));

        // 0.05% loss should not trigger
        assert!(!strategy.should_stop_loss(&position, dec!(99.95)));
    }
}
