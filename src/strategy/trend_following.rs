//! Trend Following Strategy
//!
//! Uses moving average crossovers to follow the trend.
//! Buys when fast MA crosses above slow MA (golden cross).
//! Sells when fast MA crosses below slow MA (death cross) or trailing stop hit.

use crate::data::{OrderBook, Ticker};
use crate::strategy::base::{Signal, Strategy};
use crate::strategy::indicators::{crossed_above, crossed_below, ema};
use crate::trading::order::{OrderSide, Position};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use tracing::debug;

/// Configuration for Trend Following strategy
#[derive(Debug, Clone)]
pub struct TrendFollowingConfig {
    pub fast_period: usize,
    pub slow_period: usize,
    pub atr_multiplier: f64,
    pub trailing_stop_pct: f64,
    pub position_size: Decimal,
}

impl Default for TrendFollowingConfig {
    fn default() -> Self {
        Self {
            fast_period: 10,
            slow_period: 30,
            atr_multiplier: 2.0,
            trailing_stop_pct: 1.5,
            position_size: dec!(0.001),
        }
    }
}

struct MACrossState {
    prev_fast: Option<Decimal>,
    prev_slow: Option<Decimal>,
    highest_since_entry: Decimal,
}

pub struct TrendFollowingStrategy {
    config: TrendFollowingConfig,
    price_history: HashMap<String, Vec<Decimal>>,
    ma_state: HashMap<String, MACrossState>,
}

impl TrendFollowingStrategy {
    pub fn new(config: TrendFollowingConfig) -> Self {
        Self {
            config,
            price_history: HashMap::new(),
            ma_state: HashMap::new(),
        }
    }

    fn update_history(&mut self, pair: &str, price: Decimal) {
        let history = self.price_history.entry(pair.to_string()).or_default();
        history.push(price);

        // Keep enough history for slow MA
        let max_history = self.config.slow_period * 2;
        if history.len() > max_history {
            history.remove(0);
        }

        // Update highest price since entry
        if let Some(state) = self.ma_state.get_mut(pair) {
            if price > state.highest_since_entry {
                state.highest_since_entry = price;
            }
        }
    }

    fn get_moving_averages(&self, pair: &str) -> Option<(Decimal, Decimal)> {
        let history = self.price_history.get(pair)?;

        if history.len() < self.config.slow_period {
            return None;
        }

        let fast_ma = ema(history, self.config.fast_period)?;
        let slow_ma = ema(history, self.config.slow_period)?;

        Some((fast_ma, slow_ma))
    }

    fn check_trailing_stop(&self, pair: &str, current_price: Decimal, entry_price: Decimal) -> bool {
        let state = match self.ma_state.get(pair) {
            Some(s) => s,
            None => return false,
        };

        let trailing_pct = Decimal::try_from(self.config.trailing_stop_pct).unwrap_or(dec!(1.5));

        // Calculate trailing stop level from highest price
        let stop_level = state.highest_since_entry * (dec!(1) - trailing_pct / dec!(100));

        // Trigger if price falls below trailing stop AND we're in profit territory
        current_price < stop_level && state.highest_since_entry > entry_price
    }
}

impl Strategy for TrendFollowingStrategy {
    fn analyze(
        &mut self,
        ticker: &Ticker,
        _orderbook: Option<&OrderBook>,
        position: Option<&Position>,
    ) -> Signal {
        self.update_history(&ticker.pair, ticker.last);

        let Some((fast_ma, slow_ma)) = self.get_moving_averages(&ticker.pair) else {
            return Signal::Hold;
        };

        // Get previous MA values
        let state = self.ma_state.entry(ticker.pair.clone()).or_insert(MACrossState {
            prev_fast: None,
            prev_slow: None,
            highest_since_entry: ticker.last,
        });

        let prev_fast = state.prev_fast.unwrap_or(fast_ma);
        let prev_slow = state.prev_slow.unwrap_or(slow_ma);

        // Update state for next iteration
        state.prev_fast = Some(fast_ma);
        state.prev_slow = Some(slow_ma);

        // If we have a position
        if let Some(pos) = position {
            // Check for death cross (trend reversal)
            if crossed_below(prev_fast, fast_ma, prev_slow, slow_ma) {
                debug!(
                    "{}: Trend reversal - fast MA {} crossed below slow MA {}",
                    ticker.pair, fast_ma, slow_ma
                );
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            // Check trailing stop
            if self.check_trailing_stop(&ticker.pair, ticker.last, pos.entry_price) {
                debug!(
                    "{}: Trailing stop hit at {}",
                    ticker.pair, ticker.last
                );
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            // Check hard stop loss
            if self.should_stop_loss(pos, ticker.last) {
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            return Signal::Hold;
        }

        // No position - check for golden cross
        if crossed_above(prev_fast, fast_ma, prev_slow, slow_ma) {
            debug!(
                "{}: Golden cross - fast MA {} crossed above slow MA {}",
                ticker.pair, fast_ma, slow_ma
            );

            // Reset highest since entry
            if let Some(state) = self.ma_state.get_mut(&ticker.pair) {
                state.highest_since_entry = ticker.last;
            }

            return Signal::Buy {
                pair: ticker.pair.clone(),
                quantity: self.config.position_size,
            };
        }

        Signal::Hold
    }

    fn should_take_profit(&self, _position: &Position, _current_price: Decimal) -> bool {
        // Trend following uses trailing stop, not fixed take profit
        false
    }

    fn should_stop_loss(&self, position: &Position, current_price: Decimal) -> bool {
        // Hard stop at 3% (larger than trailing to act as emergency stop)
        let stop_loss_pct = dec!(3);

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
        "TrendFollowing"
    }

    fn reset(&mut self) {
        self.price_history.clear();
        self.ma_state.clear();
    }
}
