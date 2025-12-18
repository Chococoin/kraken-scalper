//! View modules for the TUI
//!
//! Each view represents a different screen/layout in the application.

pub mod history;
pub mod portfolio;
pub mod trading;

pub use history::HistoryView;
pub use portfolio::PortfolioView;
pub use trading::TradingView;

use crate::data::{Candle, OrderBook, Ticker};
use crate::storage::TradeRecord;
use crate::trading::order::Position;
use crate::ui::widgets::ai_advisor::{AiRecentTrade, AiSignalData, AiTradeStats};
use ratatui::{layout::Rect, Frame};
use rust_decimal::Decimal;
use std::collections::HashMap;

/// Shared state passed to views for rendering
pub struct ViewState<'a> {
    pub tickers: &'a HashMap<String, Ticker>,
    pub orderbooks: &'a HashMap<String, OrderBook>,
    pub candles: &'a HashMap<String, Vec<Candle>>,
    pub positions: &'a [Position],
    pub recent_trades: &'a [TradeRecord],
    pub balance: Decimal,
    pub equity: Decimal,
    pub total_pnl: Decimal,
    pub total_pnl_pct: Decimal,
    pub is_paper: bool,
    pub connected: bool,
    pub selected_pair: &'a str,
    pub all_pairs: &'a [String],
    pub timeframe: &'a str,
    // AI Advisor data
    pub ai_signal: Option<&'a AiSignalData>,
    pub ai_stats: Option<&'a AiTradeStats>,
    pub ai_recent_trades: &'a [AiRecentTrade],
}

/// Trait for renderable views
pub trait ViewRenderer {
    fn render(&self, f: &mut Frame, area: Rect, state: &ViewState);
}
