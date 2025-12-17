//! Help overlay showing all keybindings

use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};

/// Help overlay widget
pub struct HelpOverlay;

impl HelpOverlay {
    pub fn new() -> Self {
        Self
    }

    /// Calculate centered rect for dialog
    fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
        let popup_layout = Layout::vertical([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

        Layout::horizontal([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
    }
}

impl Default for HelpOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for HelpOverlay {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let dialog_area = Self::centered_rect(60, 70, area);

        // Clear background
        Clear.render(dialog_area, buf);

        let block = Block::default()
            .title(" Help - Keybindings ")
            .title_alignment(Alignment::Center)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .style(Style::default().bg(Color::Black));

        let inner = block.inner(dialog_area);
        block.render(dialog_area, buf);

        let header_style = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let key_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let desc_style = Style::default().fg(Color::White);
        let dim_style = Style::default().fg(Color::DarkGray);

        let lines = vec![
            Line::from(vec![Span::styled("NAVIGATION", header_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Tab       ", key_style),
                Span::styled("Cycle through trading pairs", desc_style),
            ]),
            Line::from(vec![
                Span::styled("  ↑ ↓ ← →   ", key_style),
                Span::styled("Navigate UI areas", desc_style),
            ]),
            Line::from(vec![
                Span::styled("  /         ", key_style),
                Span::styled("Search pairs", desc_style),
            ]),
            Line::from(vec![
                Span::styled("  Enter     ", key_style),
                Span::styled("Select / Confirm", desc_style),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled("VIEWS", header_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  1         ", key_style),
                Span::styled("Trading view (chart, orderbook)", desc_style),
            ]),
            Line::from(vec![
                Span::styled("  2         ", key_style),
                Span::styled("Portfolio view (balances, positions)", desc_style),
            ]),
            Line::from(vec![
                Span::styled("  3         ", key_style),
                Span::styled("History view (trades, orders)", desc_style),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled("TRADING", header_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  B         ", key_style),
                Span::styled("Open BUY dialog for selected pair", desc_style),
            ]),
            Line::from(vec![
                Span::styled("  S         ", key_style),
                Span::styled("Open SELL dialog for selected pair", desc_style),
            ]),
            Line::from(vec![
                Span::styled("  C         ", key_style),
                Span::styled("Cancel order", desc_style),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled("OTHER", header_style)]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  ? / H     ", key_style),
                Span::styled("Show this help", desc_style),
            ]),
            Line::from(vec![
                Span::styled("  q / Esc   ", key_style),
                Span::styled("Quit (close dialog or exit)", desc_style),
            ]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "Press any key to close this help",
                dim_style,
            )]),
        ];

        let paragraph = Paragraph::new(lines).alignment(Alignment::Left);

        paragraph.render(inner, buf);
    }
}
