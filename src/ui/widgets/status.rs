use crate::data::Ticker;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Widget},
};
use rust_decimal::Decimal;

pub struct StatusWidget<'a> {
    tickers: Vec<&'a Ticker>,
    balance: Decimal,
    equity: Decimal,
    total_pnl: Decimal,
    total_pnl_pct: Decimal,
    is_paper: bool,
    connected: bool,
}

impl<'a> StatusWidget<'a> {
    pub fn new(
        tickers: Vec<&'a Ticker>,
        balance: Decimal,
        equity: Decimal,
        total_pnl: Decimal,
        total_pnl_pct: Decimal,
        is_paper: bool,
        connected: bool,
    ) -> Self {
        Self {
            tickers,
            balance,
            equity,
            total_pnl,
            total_pnl_pct,
            is_paper,
            connected,
        }
    }
}

impl Widget for StatusWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mode_str = if self.is_paper { "PAPER" } else { "LIVE" };
        let mode_color = if self.is_paper {
            Color::Yellow
        } else {
            Color::Red
        };

        let title = format!(" Kraken Scalper [{}] ", mode_str);

        let block = Block::default()
            .title(title)
            .title_style(Style::default().fg(mode_color).add_modifier(Modifier::BOLD))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.width < 40 || inner.height < 2 {
            return;
        }

        let mut y = inner.y;

        // Connection status
        let conn_status = if self.connected { "● CONNECTED" } else { "○ DISCONNECTED" };
        let conn_color = if self.connected {
            Color::Green
        } else {
            Color::Red
        };
        buf.set_string(inner.x + 1, y, conn_status, Style::default().fg(conn_color));

        // Ticker prices (on the same line)
        let mut x_offset = inner.x + 20;
        for ticker in &self.tickers {
            if x_offset + 25 >= inner.x + inner.width {
                break;
            }

            let change_color = if ticker.change_24h >= Decimal::ZERO {
                Color::Green
            } else {
                Color::Red
            };
            let change_sign = if ticker.change_24h >= Decimal::ZERO {
                "+"
            } else {
                ""
            };

            let ticker_str = format!(
                "{}: ${:.2} ({}{:.2}%)",
                ticker.pair, ticker.last, change_sign, ticker.change_24h
            );
            buf.set_string(x_offset, y, &ticker_str, Style::default().fg(change_color));
            x_offset += ticker_str.len() as u16 + 3;
        }

        y += 1;
        if y >= inner.y + inner.height {
            return;
        }

        // Balance and P&L
        let pnl_color = if self.total_pnl >= Decimal::ZERO {
            Color::Green
        } else {
            Color::Red
        };
        let pnl_sign = if self.total_pnl >= Decimal::ZERO { "+" } else { "" };

        let balance_str = format!("Balance: ${:.2}", self.balance);
        buf.set_string(
            inner.x + 1,
            y,
            &balance_str,
            Style::default().fg(Color::White),
        );

        let equity_str = format!("Equity: ${:.2}", self.equity);
        buf.set_string(
            inner.x + 25,
            y,
            &equity_str,
            Style::default().fg(Color::Cyan),
        );

        let pnl_str = format!(
            "P&L: {}{:.2} ({}{:.2}%)",
            pnl_sign, self.total_pnl, pnl_sign, self.total_pnl_pct
        );
        buf.set_string(inner.x + 50, y, &pnl_str, Style::default().fg(pnl_color));

        // Help text at the end
        if inner.width > 90 {
            let help = "Press 'q' to quit";
            buf.set_string(
                inner.x + inner.width - help.len() as u16 - 1,
                y,
                help,
                Style::default().fg(Color::DarkGray),
            );
        }
    }
}
