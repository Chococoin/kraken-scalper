//! Trading view - main trading interface with chart, orderbook, positions

use super::{ViewRenderer, ViewState};
use crate::ui::charts::PriceChart;
use crate::ui::widgets::{orderbook::OrderBookWidget, positions::PositionsWidget, status::StatusWidget};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    Frame,
};

/// Main trading view
pub struct TradingView;

impl TradingView {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TradingView {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewRenderer for TradingView {
    fn render(&self, f: &mut Frame, area: Rect, state: &ViewState) {
        // Main layout: status bar + chart + bottom row
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),      // Status bar
                Constraint::Percentage(50), // Chart (full width)
                Constraint::Min(10),        // Order book + Positions
            ])
            .split(area);

        // Render status bar
        render_status(f, main_chunks[0], state);

        // Render chart (full width)
        render_chart(f, main_chunks[1], state);

        // Bottom row: order book + positions side by side
        let bottom_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(main_chunks[2]);

        render_orderbook(f, bottom_chunks[0], state);
        render_positions(f, bottom_chunks[1], state);
    }
}

fn render_status(f: &mut Frame, area: Rect, state: &ViewState) {
    let tickers: Vec<&_> = state.tickers.values().collect();

    let widget = StatusWidget::new(
        tickers,
        state.balance,
        state.equity,
        state.total_pnl,
        state.total_pnl_pct,
        state.is_paper,
        state.connected,
        state.timeframe,
    );

    f.render_widget(widget, area);
}

fn render_chart(f: &mut Frame, area: Rect, state: &ViewState) {
    let candles: Vec<&_> = state
        .candles
        .get(state.selected_pair)
        .map(|c| c.iter().collect())
        .unwrap_or_default();

    let widget = PriceChart::new(&candles, state.selected_pair);
    f.render_widget(widget, area);
}

fn render_orderbook(f: &mut Frame, area: Rect, state: &ViewState) {
    let orderbook = state.orderbooks.get(state.selected_pair);

    let widget = OrderBookWidget::new(orderbook).depth(10);
    f.render_widget(widget, area);
}

fn render_positions(f: &mut Frame, area: Rect, state: &ViewState) {
    let positions: Vec<&_> = state.positions.iter().collect();
    let widget = PositionsWidget::new(positions);
    f.render_widget(widget, area);
}
