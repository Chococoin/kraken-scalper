use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: Decimal,
    pub qty: Decimal,
}

impl PriceLevel {
    pub fn new(price: Decimal, qty: Decimal) -> Self {
        Self { price, qty }
    }
}

#[derive(Debug, Clone)]
pub struct OrderBook {
    pub pair: String,
    /// Bids sorted by price descending (best bid first)
    pub bids: BTreeMap<Decimal, Decimal>,
    /// Asks sorted by price ascending (best ask first)
    pub asks: BTreeMap<Decimal, Decimal>,
    pub updated_at: DateTime<Utc>,
}

impl OrderBook {
    pub fn new(pair: &str) -> Self {
        Self {
            pair: pair.to_string(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            updated_at: Utc::now(),
        }
    }

    pub fn update_bid(&mut self, price: Decimal, qty: Decimal) {
        if qty.is_zero() {
            self.bids.remove(&price);
        } else {
            self.bids.insert(price, qty);
        }
        self.updated_at = Utc::now();
    }

    pub fn update_ask(&mut self, price: Decimal, qty: Decimal) {
        if qty.is_zero() {
            self.asks.remove(&price);
        } else {
            self.asks.insert(price, qty);
        }
        self.updated_at = Utc::now();
    }

    pub fn best_bid(&self) -> Option<PriceLevel> {
        self.bids
            .iter()
            .next_back()
            .map(|(p, q)| PriceLevel::new(*p, *q))
    }

    pub fn best_ask(&self) -> Option<PriceLevel> {
        self.asks
            .iter()
            .next()
            .map(|(p, q)| PriceLevel::new(*p, *q))
    }

    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }

    pub fn spread_pct(&self) -> Option<Decimal> {
        match (self.best_bid(), self.spread()) {
            (Some(bid), Some(spread)) if !bid.price.is_zero() => {
                Some((spread / bid.price) * Decimal::from(100))
            }
            _ => None,
        }
    }

    pub fn mid_price(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / Decimal::from(2)),
            _ => None,
        }
    }

    /// Get top N bids (best first)
    pub fn top_bids(&self, n: usize) -> Vec<PriceLevel> {
        self.bids
            .iter()
            .rev()
            .take(n)
            .map(|(p, q)| PriceLevel::new(*p, *q))
            .collect()
    }

    /// Get top N asks (best first)
    pub fn top_asks(&self, n: usize) -> Vec<PriceLevel> {
        self.asks
            .iter()
            .take(n)
            .map(|(p, q)| PriceLevel::new(*p, *q))
            .collect()
    }

    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }
}

#[derive(Debug, Default)]
pub struct OrderBookStore {
    books: std::collections::HashMap<String, OrderBook>,
}

impl OrderBookStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_create(&mut self, pair: &str) -> &mut OrderBook {
        self.books
            .entry(pair.to_string())
            .or_insert_with(|| OrderBook::new(pair))
    }

    pub fn get(&self, pair: &str) -> Option<&OrderBook> {
        self.books.get(pair)
    }
}
