use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub vwap: Decimal,
    pub trades: u64,
}

impl Candle {
    pub fn new(timestamp: DateTime<Utc>, price: Decimal) -> Self {
        Self {
            timestamp,
            open: price,
            high: price,
            low: price,
            close: price,
            volume: Decimal::ZERO,
            vwap: price,
            trades: 0,
        }
    }

    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    pub fn body_size(&self) -> Decimal {
        (self.close - self.open).abs()
    }

    pub fn range(&self) -> Decimal {
        self.high - self.low
    }

    pub fn upper_wick(&self) -> Decimal {
        if self.is_bullish() {
            self.high - self.close
        } else {
            self.high - self.open
        }
    }

    pub fn lower_wick(&self) -> Decimal {
        if self.is_bullish() {
            self.open - self.low
        } else {
            self.close - self.low
        }
    }
}

#[derive(Debug)]
pub struct CandleStore {
    /// Candles per pair, stored as a ring buffer (newest last)
    candles: HashMap<String, VecDeque<Candle>>,
    max_candles: usize,
}

impl CandleStore {
    pub fn new(max_candles: usize) -> Self {
        Self {
            candles: HashMap::new(),
            max_candles,
        }
    }

    pub fn add(&mut self, pair: &str, candle: Candle) {
        let deque = self
            .candles
            .entry(pair.to_string())
            .or_insert_with(VecDeque::new);

        deque.push_back(candle);

        while deque.len() > self.max_candles {
            deque.pop_front();
        }
    }

    pub fn update_last(&mut self, pair: &str, candle: Candle) {
        if let Some(deque) = self.candles.get_mut(pair) {
            if let Some(last) = deque.back_mut() {
                if last.timestamp == candle.timestamp {
                    *last = candle;
                    return;
                }
            }
            self.add(pair, candle);
        } else {
            self.add(pair, candle);
        }
    }

    pub fn get(&self, pair: &str) -> Option<&VecDeque<Candle>> {
        self.candles.get(pair)
    }

    pub fn last(&self, pair: &str) -> Option<&Candle> {
        self.candles.get(pair).and_then(|d| d.back())
    }

    pub fn last_n(&self, pair: &str, n: usize) -> Vec<&Candle> {
        self.candles
            .get(pair)
            .map(|d| d.iter().rev().take(n).rev().collect())
            .unwrap_or_default()
    }
}
