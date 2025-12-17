//! Order entry dialog for Buy/Sell

use crate::trading::order::{OrderSide, OrderType};
use crate::ui::input::{OrderDialogField, OrderDialogState};
use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Widget},
};
use rust_decimal::Decimal;

/// Order dialog widget
pub struct OrderDialog<'a> {
    state: &'a OrderDialogState,
    current_price: Option<Decimal>,
    balance: Decimal,
}

impl<'a> OrderDialog<'a> {
    pub fn new(state: &'a OrderDialogState, current_price: Option<Decimal>, balance: Decimal) -> Self {
        Self {
            state,
            current_price,
            balance,
        }
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

impl Widget for OrderDialog<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let dialog_area = Self::centered_rect(50, 50, area);

        // Clear background
        Clear.render(dialog_area, buf);

        let title = match self.state.side {
            OrderSide::Buy => format!(" BUY {} ", self.state.pair),
            OrderSide::Sell => format!(" SELL {} ", self.state.pair),
        };

        let border_color = match self.state.side {
            OrderSide::Buy => Color::Green,
            OrderSide::Sell => Color::Red,
        };

        let block = Block::default()
            .title(title)
            .title_alignment(Alignment::Center)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .style(Style::default().bg(Color::Black));

        let inner = block.inner(dialog_area);
        block.render(dialog_area, buf);

        // Styles
        let label_style = Style::default().fg(Color::DarkGray);
        let value_style = Style::default().fg(Color::White);
        let focused_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let button_style = Style::default().fg(Color::White);
        let button_focused_style = Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let error_style = Style::default().fg(Color::Red);

        let mut lines = vec![];

        // Current price
        if let Some(price) = self.current_price {
            lines.push(Line::from(vec![
                Span::styled("Current Price: ", label_style),
                Span::styled(format!("${:.2}", price), value_style),
            ]));
            lines.push(Line::from(""));
        }

        // Order type selector
        let order_type_style = if self.state.focused_field == OrderDialogField::OrderType {
            focused_style
        } else {
            value_style
        };
        let market_style = if self.state.order_type == OrderType::Market {
            order_type_style.add_modifier(Modifier::REVERSED)
        } else {
            order_type_style
        };
        let limit_style = if self.state.order_type == OrderType::Limit {
            order_type_style.add_modifier(Modifier::REVERSED)
        } else {
            order_type_style
        };

        lines.push(Line::from(vec![
            Span::styled("Type: ", label_style),
            Span::styled(" Market ", market_style),
            Span::raw(" "),
            Span::styled(" Limit ", limit_style),
        ]));
        lines.push(Line::from(""));

        // Quantity input
        let qty_style = if self.state.focused_field == OrderDialogField::Quantity {
            focused_style
        } else {
            value_style
        };
        let qty_display = if self.state.quantity.is_empty() {
            "0.0".to_string()
        } else {
            self.state.quantity.clone()
        };
        lines.push(Line::from(vec![
            Span::styled("Quantity: ", label_style),
            Span::styled(format!("[{}]", qty_display), qty_style),
        ]));

        // Quantity shortcuts
        lines.push(Line::from(vec![
            Span::styled("          ", label_style),
            Span::styled("(1)25% ", Style::default().fg(Color::DarkGray)),
            Span::styled("(2)50% ", Style::default().fg(Color::DarkGray)),
            Span::styled("(3)75% ", Style::default().fg(Color::DarkGray)),
            Span::styled("(4)100%", Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(""));

        // Price input (for limit orders)
        if self.state.order_type == OrderType::Limit {
            let price_style = if self.state.focused_field == OrderDialogField::Price {
                focused_style
            } else {
                value_style
            };
            let price_display = if self.state.price.is_empty() {
                "0.00".to_string()
            } else {
                self.state.price.clone()
            };
            lines.push(Line::from(vec![
                Span::styled("Price:    ", label_style),
                Span::styled(format!("[${}]", price_display), price_style),
            ]));
            lines.push(Line::from(""));
        }

        // Estimated total
        let qty: Decimal = self.state.quantity.parse().unwrap_or(Decimal::ZERO);
        let price = if self.state.order_type == OrderType::Limit {
            self.state.price.parse().unwrap_or(Decimal::ZERO)
        } else {
            self.current_price.unwrap_or(Decimal::ZERO)
        };
        let total = qty * price;

        lines.push(Line::from(vec![
            Span::styled("Est. Total: ", label_style),
            Span::styled(format!("${:.2}", total), value_style),
        ]));

        // Balance info
        lines.push(Line::from(vec![
            Span::styled("Balance:    ", label_style),
            Span::styled(format!("${:.2}", self.balance), Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(""));

        // Error message
        if let Some(ref error) = self.state.error_message {
            lines.push(Line::from(vec![Span::styled(error, error_style)]));
            lines.push(Line::from(""));
        }

        // Buttons
        let submit_style = if self.state.focused_field == OrderDialogField::Submit {
            button_focused_style
        } else {
            button_style
        };
        let cancel_style = if self.state.focused_field == OrderDialogField::Cancel {
            button_focused_style
        } else {
            button_style
        };

        let submit_text = match self.state.side {
            OrderSide::Buy => " CONFIRM BUY ",
            OrderSide::Sell => " CONFIRM SELL ",
        };

        lines.push(Line::from(vec![
            Span::styled("      ", label_style),
            Span::styled(" CANCEL ", cancel_style),
            Span::raw("    "),
            Span::styled(submit_text, submit_style),
        ]));

        let paragraph = Paragraph::new(lines).alignment(Alignment::Left);

        // Add some padding
        let padded = Rect {
            x: inner.x + 2,
            y: inner.y + 1,
            width: inner.width.saturating_sub(4),
            height: inner.height.saturating_sub(2),
        };

        paragraph.render(padded, buf);
    }
}
