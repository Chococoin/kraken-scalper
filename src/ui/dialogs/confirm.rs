//! Confirmation dialog

use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};

/// Confirmation dialog widget
pub struct ConfirmDialog<'a> {
    title: &'a str,
    message: &'a str,
    selected_yes: bool,
    is_destructive: bool,
}

impl<'a> ConfirmDialog<'a> {
    pub fn new(title: &'a str, message: &'a str, selected_yes: bool) -> Self {
        Self {
            title,
            message,
            selected_yes,
            is_destructive: false,
        }
    }

    pub fn destructive(mut self) -> Self {
        self.is_destructive = true;
        self
    }

    /// Calculate centered rect for dialog
    fn centered_rect(width: u16, height: u16, r: Rect) -> Rect {
        let x = r.x + (r.width.saturating_sub(width)) / 2;
        let y = r.y + (r.height.saturating_sub(height)) / 2;
        Rect::new(x, y, width.min(r.width), height.min(r.height))
    }
}

impl Widget for ConfirmDialog<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let dialog_area = Self::centered_rect(40, 10, area);

        // Clear background
        Clear.render(dialog_area, buf);

        let border_color = if self.is_destructive {
            Color::Red
        } else {
            Color::Yellow
        };

        let block = Block::default()
            .title(format!(" {} ", self.title))
            .title_alignment(Alignment::Center)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .style(Style::default().bg(Color::Black));

        let inner = block.inner(dialog_area);
        block.render(dialog_area, buf);

        let button_style = Style::default().fg(Color::White);
        let button_focused_style = Style::default()
            .fg(Color::Black)
            .bg(if self.is_destructive { Color::Red } else { Color::Yellow })
            .add_modifier(Modifier::BOLD);

        let yes_style = if self.selected_yes {
            button_focused_style
        } else {
            button_style
        };
        let no_style = if !self.selected_yes {
            button_focused_style
        } else {
            button_style
        };

        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                self.message,
                Style::default().fg(Color::White),
            )),
            Line::from(""),
            Line::from(""),
            Line::from(vec![
                Span::raw("       "),
                Span::styled("  YES  ", yes_style),
                Span::raw("    "),
                Span::styled("  NO  ", no_style),
            ]),
        ];

        let paragraph = Paragraph::new(lines).alignment(Alignment::Center);
        paragraph.render(inner, buf);
    }
}
