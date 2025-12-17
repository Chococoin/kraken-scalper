//! Breakout Strategy
//!
//! Buys when price breaks above recent highs.
//! Sells when price breaks below recent lows or trailing stop hit.

use crate::data::{OrderBook, Ticker};
use crate::strategy::base::{Signal, Strategy};
use crate::trading::order::{OrderSide, Position};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use tracing::debug;

/// Configuration for Breakout strategy
#[derive(Debug, Clone)]
pub struct BreakoutConfig {
    pub lookback_periods: usize,
    pub breakout_buffer_pct: f64,
    pub stop_loss_pct: f64,
    pub trailing_stop_pct: f64,
    pub position_size: Decimal,
}

impl Default for BreakoutConfig {
    fn default() -> Self {
        Self {
            lookback_periods: 20,
            breakout_buffer_pct: 0.1,
            stop_loss_pct: 1.0,
            trailing_stop_pct: 1.5,
            position_size: dec!(0.001),
        }
    }
}

struct BreakoutState {
    highest_since_entry: Decimal,
    in_breakout: bool,
}

pub struct BreakoutStrategy {
    config: BreakoutConfig,
    price_history: HashMap<String, Vec<Decimal>>,
    state: HashMap<String, BreakoutState>,
}

impl BreakoutStrategy {
    pub fn new(config: BreakoutConfig) -> Self {
        Self {
            config,
            price_history: HashMap::new(),
            state: HashMap::new(),
        }
    }

    fn update_history(&mut self, pair: &str, price: Decimal) {
        let history = self.price_history.entry(pair.to_string()).or_default();
        history.push(price);

        let max_history = self.config.lookback_periods * 2;
        if history.len() > max_history {
            history.remove(0);
        }

        // Update highest since entry
        if let Some(state) = self.state.get_mut(pair) {
            if price > state.highest_since_entry {
                state.highest_since_entry = price;
            }
        }
    }

    fn get_range(&self, pair: &str) -> Option<(Decimal, Decimal)> {
        let history = self.price_history.get(pair)?;

        if history.len() < self.config.lookback_periods {
            return None;
        }

        // Get high/low excluding the most recent price (to avoid self-referencing)
        let lookback: Vec<_> = history
            .iter()
            .rev()
            .skip(1)
            .take(self.config.lookback_periods)
            .copied()
            .collect();

        if lookback.is_empty() {
            return None;
        }

        let high = *lookback.iter().max()?;
        let low = *lookback.iter().min()?;

        Some((low, high))
    }

    fn check_trailing_stop(&self, pair: &str, current_price: Decimal, entry_price: Decimal) -> bool {
        let state = match self.state.get(pair) {
            Some(s) => s,
            None => return false,
        };

        let trailing_pct = Decimal::try_from(self.config.trailing_stop_pct).unwrap_or(dec!(1.5));
        let stop_level = state.highest_since_entry * (dec!(1) - trailing_pct / dec!(100));

        current_price < stop_level && state.highest_since_entry > entry_price
    }
}

impl Strategy for BreakoutStrategy {
    fn analyze(
        &mut self,
        ticker: &Ticker,
        _orderbook: Option<&OrderBook>,
        position: Option<&Position>,
    ) -> Signal {
        self.update_history(&ticker.pair, ticker.last);

        let Some((range_low, range_high)) = self.get_range(&ticker.pair) else {
            return Signal::Hold;
        };

        let buffer_pct = Decimal::try_from(self.config.breakout_buffer_pct).unwrap_or(dec!(0.1));
        let buffer = range_high * buffer_pct / dec!(100);

        let breakout_level = range_high + buffer;
        let breakdown_level = range_low - buffer;

        // If we have a position
        if let Some(pos) = position {
            // Check for breakdown (price falls below range)
            if ticker.last < breakdown_level {
                debug!(
                    "{}: Breakdown - price {} < level {}",
                    ticker.pair, ticker.last, breakdown_level
                );
                return Signal::Sell {
                    pair: ticker.pair.clone(),
                };
            }

            // Check trailing stop
            if self.check_trailing_stop(&ticker.pair, ticker.last, pos.entry_price) {
                debug!("{}: Trailing stop hit", ticker.pair);
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

        // No position - check for breakout
        let state = self.state.entry(ticker.pair.clone()).or_insert(BreakoutState {
            highest_since_entry: ticker.last,
            in_breakout: false,
        });

        if ticker.last > breakout_level && !state.in_breakout {
            debug!(
                "{}: Breakout - price {} > level {} (range high: {})",
                ticker.pair, ticker.last, breakout_level, range_high
            );

            state.in_breakout = true;
            state.highest_since_entry = ticker.last;

            return Signal::Buy {
                pair: ticker.pair.clone(),
                quantity: self.config.position_size,
            };
        }

        // Reset breakout state if price returns to range
        if ticker.last < range_high {
            state.in_breakout = false;
        }

        Signal::Hold
    }

    fn should_take_profit(&self, _position: &Position, _current_price: Decimal) -> bool {
        // Breakout uses trailing stop, not fixed take profit
        false
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
        "Breakout"
    }

    fn reset(&mut self) {
        self.price_history.clear();
        self.state.clear();
    }
}
