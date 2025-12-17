use crate::storage::TradeRecord;
use crate::trading::OrderSide;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Widget},
};

pub struct TradesWidget<'a> {
    trades: &'a [TradeRecord],
}

impl<'a> TradesWidget<'a> {
    pub fn new(trades: &'a [TradeRecord]) -> Self {
        Self { trades }
    }
}

impl Widget for TradesWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Recent Trades ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.width < 40 || inner.height < 3 {
            return;
        }

        if self.trades.is_empty() {
            let msg = "No trades yet";
            let x = inner.x + (inner.width.saturating_sub(msg.len() as u16)) / 2;
            let y = inner.y + inner.height / 2;
            buf.set_string(x, y, msg, Style::default().fg(Color::DarkGray));
            return;
        }

        // Header
        let header_style = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);

        let cols = ["TIME", "PAIR", "SIDE", "PRICE", "QTY"];
        let col_widths = [10u16, 12, 6, 14, 14];
        let mut x_offset = inner.x;

        for (col, width) in cols.iter().zip(col_widths.iter()) {
            buf.set_string(x_offset, inner.y, *col, header_style);
            x_offset += width;
        }

        // Trades
        for (i, trade) in self.trades.iter().enumerate() {
            let y = inner.y + 1 + i as u16;
            if y >= inner.y + inner.height {
                break;
            }

            let side_color = match trade.side {
                OrderSide::Buy => Color::Green,
                OrderSide::Sell => Color::Red,
            };

            let mut x_offset = inner.x;

            // Time
            let time = trade.executed_at.format("%H:%M:%S").to_string();
            buf.set_string(x_offset, y, &time, Style::default().fg(Color::DarkGray));
            x_offset += col_widths[0];

            // Pair
            buf.set_string(
                x_offset,
                y,
                &trade.pair,
                Style::default().fg(Color::White),
            );
            x_offset += col_widths[1];

            // Side
            let side_str = match trade.side {
                OrderSide::Buy => "BUY",
                OrderSide::Sell => "SELL",
            };
            buf.set_string(x_offset, y, side_str, Style::default().fg(side_color));
            x_offset += col_widths[2];

            // Price
            let price = format!("{:.2}", trade.price);
            buf.set_string(x_offset, y, &price, Style::default().fg(Color::White));
            x_offset += col_widths[3];

            // Quantity
            let qty = format!("{:.6}", trade.quantity);
            buf.set_string(x_offset, y, &qty, Style::default().fg(Color::DarkGray));
        }
    }
}
