//! Portfolio view - account balances, positions overview, allocation

use super::{ViewRenderer, ViewState};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table},
    Frame,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Portfolio overview view
pub struct PortfolioView;

impl PortfolioView {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PortfolioView {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewRenderer for PortfolioView {
    fn render(&self, f: &mut Frame, area: Rect, state: &ViewState) {
        // Main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Length(8),  // Account summary
                Constraint::Min(10),    // Positions table
            ])
            .split(area);

        // Title bar
        render_title(f, chunks[0], state);

        // Account summary
        render_account_summary(f, chunks[1], state);

        // Positions table
        render_positions_table(f, chunks[2], state);
    }
}

fn render_title(f: &mut Frame, area: Rect, state: &ViewState) {
    let mode_text = if state.is_paper { "PAPER" } else { "LIVE" };
    let mode_color = if state.is_paper { Color::Yellow } else { Color::Red };

    let title = Line::from(vec![
        Span::styled(
            " Portfolio Overview ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | "),
        Span::styled(mode_text, Style::default().fg(mode_color).add_modifier(Modifier::BOLD)),
        Span::raw(" | Press "),
        Span::styled("1", Style::default().fg(Color::Cyan)),
        Span::raw(" for Trading, "),
        Span::styled("3", Style::default().fg(Color::Cyan)),
        Span::raw(" for History"),
    ]);

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(title)
        .block(block)
        .alignment(Alignment::Center);

    f.render_widget(paragraph, area);
}

fn render_account_summary(f: &mut Frame, area: Rect, state: &ViewState) {
    let block = Block::default()
        .title(" Account Summary ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(inner);

    // Balance
    render_metric(f, chunks[0], "Balance", state.balance, None);

    // Equity
    render_metric(f, chunks[1], "Equity", state.equity, None);

    // Total P&L
    let pnl_color = if state.total_pnl >= Decimal::ZERO {
        Color::Green
    } else {
        Color::Red
    };
    render_metric(f, chunks[2], "Total P&L", state.total_pnl, Some(pnl_color));

    // P&L %
    render_metric(f, chunks[3], "P&L %", state.total_pnl_pct, Some(pnl_color));
}

fn render_metric(f: &mut Frame, area: Rect, label: &str, value: Decimal, color: Option<Color>) {
    let value_color = color.unwrap_or(Color::White);
    let is_percentage = label.contains('%');

    let value_str = if is_percentage {
        format!("{:.2}%", value)
    } else {
        format!("${:.2}", value)
    };

    let lines = vec![
        Line::from(Span::styled(
            label,
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            value_str,
            Style::default()
                .fg(value_color)
                .add_modifier(Modifier::BOLD),
        )),
    ];

    let paragraph = Paragraph::new(lines).alignment(Alignment::Center);
    f.render_widget(paragraph, area);
}

fn render_positions_table(f: &mut Frame, area: Rect, state: &ViewState) {
    let block = Block::default()
        .title(format!(" Open Positions ({}) ", state.positions.len()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if state.positions.is_empty() {
        let msg = Paragraph::new("No open positions")
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
        "Pair",
        "Side",
        "Qty",
        "Entry",
        "Current",
        "P&L",
        "P&L %",
    ])
    .style(header_style)
    .height(1);

    // Rows
    let rows: Vec<Row> = state
        .positions
        .iter()
        .map(|pos| {
            let side_str = format!("{:?}", pos.side).to_uppercase();
            let _side_color = match pos.side {
                crate::trading::order::OrderSide::Buy => Color::Green,
                crate::trading::order::OrderSide::Sell => Color::Red,
            };

            let _pnl_color = if pos.unrealized_pnl >= Decimal::ZERO {
                Color::Green
            } else {
                Color::Red
            };

            let pnl_pct = if pos.entry_price > Decimal::ZERO {
                ((pos.current_price - pos.entry_price) / pos.entry_price) * dec!(100)
            } else {
                Decimal::ZERO
            };

            Row::new(vec![
                pos.pair.clone(),
                side_str,
                format!("{:.6}", pos.quantity),
                format!("${:.2}", pos.entry_price),
                format!("${:.2}", pos.current_price),
                format!("${:.2}", pos.unrealized_pnl),
                format!("{:.2}%", pnl_pct),
            ])
            .style(Style::default().fg(Color::White))
        })
        .collect();

    let widths = [
        Constraint::Percentage(15),
        Constraint::Percentage(10),
        Constraint::Percentage(15),
        Constraint::Percentage(15),
        Constraint::Percentage(15),
        Constraint::Percentage(15),
        Constraint::Percentage(15),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    f.render_widget(table, inner);
}
