//! Trading view - main trading interface with chart, orderbook, positions, trades

use super::{ViewRenderer, ViewState};
use crate::ui::charts::PriceChart;
use crate::ui::widgets::{
    orderbook::OrderBookWidget, positions::PositionsWidget, status::StatusWidget,
    trades::TradesWidget,
};
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
        // Main layout: status bar + content
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(10)])
            .split(area);

        // Render status bar
        render_status(f, main_chunks[0], state);

        // Content layout: left (chart + orderbook) + right (positions + trades)
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(main_chunks[1]);

        // Left side: chart on top, orderbook on bottom
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(content_chunks[0]);

        render_chart(f, left_chunks[0], state);
        render_orderbook(f, left_chunks[1], state);

        // Right side: positions on top, trades on bottom
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(content_chunks[1]);

        render_positions(f, right_chunks[0], state);
        render_trades(f, right_chunks[1], state);
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

fn render_trades(f: &mut Frame, area: Rect, state: &ViewState) {
    let widget = TradesWidget::new(state.recent_trades);
    f.render_widget(widget, area);
}
