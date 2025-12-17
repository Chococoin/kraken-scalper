use crate::data::OrderBook;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Widget},
};
use rust_decimal::prelude::ToPrimitive;

pub struct OrderBookWidget<'a> {
    orderbook: Option<&'a OrderBook>,
    depth: usize,
}

impl<'a> OrderBookWidget<'a> {
    pub fn new(orderbook: Option<&'a OrderBook>) -> Self {
        Self {
            orderbook,
            depth: 10,
        }
    }

    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }
}

impl Widget for OrderBookWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let title = self
            .orderbook
            .map(|ob| format!(" {} Order Book ", ob.pair))
            .unwrap_or_else(|| " Order Book ".to_string());

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.width < 20 || inner.height < 4 {
            return;
        }

        let Some(orderbook) = self.orderbook else {
            let msg = "Waiting for data...";
            let x = inner.x + (inner.width.saturating_sub(msg.len() as u16)) / 2;
            let y = inner.y + inner.height / 2;
            buf.set_string(x, y, msg, Style::default().fg(Color::DarkGray));
            return;
        };

        let asks = orderbook.top_asks(self.depth);
        let bids = orderbook.top_bids(self.depth);

        // Find max quantity for bar scaling
        let max_qty = asks
            .iter()
            .chain(bids.iter())
            .map(|l| l.qty.to_f64().unwrap_or(0.0))
            .fold(0.0f64, f64::max);

        let half_height = (inner.height / 2) as usize;
        let col_width = inner.width / 2;

        // Header
        let header_style = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
        buf.set_string(inner.x + 1, inner.y, "PRICE", header_style);
        buf.set_string(inner.x + col_width, inner.y, "QTY", header_style);

        // Draw asks (top half, reversed so best ask is at bottom)
        let asks_to_show: Vec<_> = asks.iter().take(half_height.saturating_sub(1)).collect();
        for (i, level) in asks_to_show.iter().rev().enumerate() {
            let y = inner.y + 1 + i as u16;
            if y >= inner.y + inner.height {
                break;
            }

            let price = format!("{:.2}", level.price);
            let qty = format!("{:.6}", level.qty);

            // Draw depth bar
            let bar_width = if max_qty > 0.0 {
                ((level.qty.to_f64().unwrap_or(0.0) / max_qty) * (col_width as f64 * 0.3)) as u16
            } else {
                0
            };
            for x in 0..bar_width {
                if inner.x + inner.width - 1 - x < inner.x + col_width + qty.len() as u16 + 1 {
                    continue;
                }
                buf[(inner.x + inner.width - 1 - x, y)]
                    .set_char('█')
                    .set_fg(Color::Rgb(60, 20, 20));
            }

            buf.set_string(inner.x + 1, y, &price, Style::default().fg(Color::Red));
            buf.set_string(inner.x + col_width, y, &qty, Style::default().fg(Color::DarkGray));
        }

        // Spread indicator
        let spread_y = inner.y + half_height as u16;
        if let (Some(bid), Some(ask)) = (orderbook.best_bid(), orderbook.best_ask()) {
            let spread = ask.price - bid.price;
            let spread_pct = if !bid.price.is_zero() {
                (spread / bid.price) * rust_decimal::Decimal::from(100)
            } else {
                rust_decimal::Decimal::ZERO
            };
            let spread_str = format!("── Spread: {:.2} ({:.3}%) ──", spread, spread_pct);
            let x = inner.x + (inner.width.saturating_sub(spread_str.len() as u16)) / 2;
            buf.set_string(x, spread_y, &spread_str, Style::default().fg(Color::Yellow));
        }

        // Draw bids (bottom half)
        for (i, level) in bids.iter().take(half_height.saturating_sub(1)).enumerate() {
            let y = inner.y + half_height as u16 + 1 + i as u16;
            if y >= inner.y + inner.height {
                break;
            }

            let price = format!("{:.2}", level.price);
            let qty = format!("{:.6}", level.qty);

            // Draw depth bar
            let bar_width = if max_qty > 0.0 {
                ((level.qty.to_f64().unwrap_or(0.0) / max_qty) * (col_width as f64 * 0.3)) as u16
            } else {
                0
            };
            for x in 0..bar_width {
                if inner.x + inner.width - 1 - x < inner.x + col_width + qty.len() as u16 + 1 {
                    continue;
                }
                buf[(inner.x + inner.width - 1 - x, y)]
                    .set_char('█')
                    .set_fg(Color::Rgb(20, 60, 20));
            }

            buf.set_string(inner.x + 1, y, &price, Style::default().fg(Color::Green));
            buf.set_string(inner.x + col_width, y, &qty, Style::default().fg(Color::DarkGray));
        }
    }
}
