use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    pub pair: String,
    pub bid: Decimal,
    pub ask: Decimal,
    pub last: Decimal,
    pub volume_24h: Decimal,
    pub high_24h: Decimal,
    pub low_24h: Decimal,
    pub vwap_24h: Decimal,
    pub change_24h: Decimal,
    pub updated_at: DateTime<Utc>,
}

impl Ticker {
    pub fn new(pair: &str) -> Self {
        Self {
            pair: pair.to_string(),
            bid: Decimal::ZERO,
            ask: Decimal::ZERO,
            last: Decimal::ZERO,
            volume_24h: Decimal::ZERO,
            high_24h: Decimal::ZERO,
            low_24h: Decimal::ZERO,
            vwap_24h: Decimal::ZERO,
            change_24h: Decimal::ZERO,
            updated_at: Utc::now(),
        }
    }

    pub fn spread(&self) -> Decimal {
        self.ask - self.bid
    }

    pub fn spread_pct(&self) -> Decimal {
        if self.bid.is_zero() {
            return Decimal::ZERO;
        }
        (self.spread() / self.bid) * Decimal::from(100)
    }

    pub fn mid_price(&self) -> Decimal {
        (self.bid + self.ask) / Decimal::from(2)
    }
}

#[derive(Debug, Default)]
pub struct TickerStore {
    tickers: HashMap<String, Ticker>,
}

impl TickerStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, ticker: Ticker) {
        self.tickers.insert(ticker.pair.clone(), ticker);
    }

    pub fn get(&self, pair: &str) -> Option<&Ticker> {
        self.tickers.get(pair)
    }

    pub fn all(&self) -> impl Iterator<Item = &Ticker> {
        self.tickers.values()
    }
}
