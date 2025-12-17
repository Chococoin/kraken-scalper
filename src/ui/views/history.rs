//! History view - trade history, order history

use super::{ViewRenderer, ViewState};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table},
    Frame,
};

/// History view
pub struct HistoryView;

impl HistoryView {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HistoryView {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewRenderer for HistoryView {
    fn render(&self, f: &mut Frame, area: Rect, state: &ViewState) {
        // Main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Min(10),    // Trades table
            ])
            .split(area);

        // Title bar
        render_title(f, chunks[0], state);

        // Trades table
        render_trades_table(f, chunks[1], state);
    }
}

fn render_title(f: &mut Frame, area: Rect, state: &ViewState) {
    let mode_text = if state.is_paper { "PAPER" } else { "LIVE" };
    let mode_color = if state.is_paper { Color::Yellow } else { Color::Red };

    let title = Line::from(vec![
        Span::styled(
            " Trade History ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | "),
        Span::styled(mode_text, Style::default().fg(mode_color).add_modifier(Modifier::BOLD)),
        Span::raw(" | Press "),
        Span::styled("1", Style::default().fg(Color::Cyan)),
        Span::raw(" for Trading, "),
        Span::styled("2", Style::default().fg(Color::Cyan)),
        Span::raw(" for Portfolio"),
    ]);

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(title)
        .block(block)
        .alignment(Alignment::Center);

    f.render_widget(paragraph, area);
}

fn render_trades_table(f: &mut Frame, area: Rect, state: &ViewState) {
    let block = Block::default()
        .title(format!(" Recent Trades ({}) ", state.recent_trades.len()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.recent_trades.is_empty() {
        let msg = Paragraph::new("No trade history")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        f.render_widget(msg, inner);
        return;
    }

    // Header
    let header_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);

    let header = Row::new(vec![
        "Time",
        "Pair",
        "Side",
        "Price",
        "Quantity",
        "Total",
        "Mode",
    ])
    .style(header_style)
    .height(1);

    // Rows
    let rows: Vec<Row> = state
        .recent_trades
        .iter()
        .rev() // Most recent first
        .map(|trade| {
            let side_str = trade.side.to_string();
            let _side_color = if matches!(trade.side, crate::trading::order::OrderSide::Buy) {
                Color::Green
            } else {
                Color::Red
            };

            let total = trade.price * trade.quantity;
            let mode = if trade.is_paper { "PAPER" } else { "LIVE" };
            let _mode_color = if trade.is_paper { Color::Yellow } else { Color::Green };

            // Format timestamp
            let time_str = trade
                .executed_at
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();

            Row::new(vec![
                time_str,
                trade.pair.clone(),
                side_str,
                format!("${:.2}", trade.price),
                format!("{:.6}", trade.quantity),
                format!("${:.2}", total),
                mode.to_string(),
            ])
            .style(Style::default().fg(Color::White))
        })
        .collect();

    let widths = [
        Constraint::Length(20),     // Time
        Constraint::Percentage(12), // Pair
        Constraint::Percentage(8),  // Side
        Constraint::Percentage(15), // Price
        Constraint::Percentage(15), // Quantity
        Constraint::Percentage(15), // Total
        Constraint::Percentage(10), // Mode
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(table, inner);
}
