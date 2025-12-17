use crate::trading::order::{OrderSide, Position};
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Widget},
};
use rust_decimal::Decimal;

pub struct PositionsWidget<'a> {
    positions: Vec<&'a Position>,
}

impl<'a> PositionsWidget<'a> {
    pub fn new(positions: Vec<&'a Position>) -> Self {
        Self { positions }
    }
}

impl Widget for PositionsWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Positions ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.width < 30 || inner.height < 3 {
            return;
        }

        if self.positions.is_empty() {
            let msg = "No open positions";
            let x = inner.x + (inner.width.saturating_sub(msg.len() as u16)) / 2;
            let y = inner.y + inner.height / 2;
            buf.set_string(x, y, msg, Style::default().fg(Color::DarkGray));
            return;
        }

        // Header
        let header_style = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);

        let cols = ["PAIR", "SIDE", "ENTRY", "CURRENT", "QTY", "P&L"];
        let col_widths = [10u16, 6, 12, 12, 12, 12];
        let mut x_offset = inner.x;

        for (col, width) in cols.iter().zip(col_widths.iter()) {
            buf.set_string(x_offset, inner.y, *col, header_style);
            x_offset += width;
        }

        // Positions
        for (i, pos) in self.positions.iter().enumerate() {
            let y = inner.y + 1 + i as u16;
            if y >= inner.y + inner.height {
                break;
            }

            let side_color = match pos.side {
                OrderSide::Buy => Color::Green,
                OrderSide::Sell => Color::Red,
            };

            let pnl_color = if pos.unrealized_pnl >= Decimal::ZERO {
                Color::Green
            } else {
                Color::Red
            };

            let mut x_offset = inner.x;

            // Pair
            buf.set_string(x_offset, y, &pos.pair, Style::default().fg(Color::White));
            x_offset += col_widths[0];

            // Side
            let side_str = match pos.side {
                OrderSide::Buy => "LONG",
                OrderSide::Sell => "SHORT",
            };
            buf.set_string(x_offset, y, side_str, Style::default().fg(side_color));
            x_offset += col_widths[1];

            // Entry price
            let entry = format!("{:.2}", pos.entry_price);
            buf.set_string(x_offset, y, &entry, Style::default().fg(Color::DarkGray));
            x_offset += col_widths[2];

            // Current price
            let current = format!("{:.2}", pos.current_price);
            buf.set_string(x_offset, y, &current, Style::default().fg(Color::White));
            x_offset += col_widths[3];

            // Quantity
            let qty = format!("{:.6}", pos.quantity);
            buf.set_string(x_offset, y, &qty, Style::default().fg(Color::DarkGray));
            x_offset += col_widths[4];

            // P&L
            let pnl_sign = if pos.unrealized_pnl >= Decimal::ZERO {
                "+"
            } else {
                ""
            };
            let pnl = format!(
                "{}{:.2} ({}{:.2}%)",
                pnl_sign, pos.unrealized_pnl, pnl_sign, pos.unrealized_pnl_pct
            );
            buf.set_string(x_offset, y, &pnl, Style::default().fg(pnl_color));
        }
    }
}
