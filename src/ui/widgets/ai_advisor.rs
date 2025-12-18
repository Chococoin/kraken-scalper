//! AI Advisor widget - displays ML model signals and paper trade performance

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Widget},
};

/// AI signal data for the widget
#[derive(Debug, Clone, Default)]
pub struct AiSignalData {
    pub signal: String,       // "BUY", "SELL", "HOLD"
    pub confidence: f64,      // 0.0 - 1.0
    pub price: f64,
    pub rsi: Option<f64>,
    pub macd: Option<f64>,
    pub macd_hist: Option<f64>,
    pub model_version: String,
    pub last_update: String,  // Timestamp string
}

/// AI trading statistics
#[derive(Debug, Clone, Default)]
pub struct AiTradeStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_pnl: f64,
    pub open_positions: usize,
}

/// Recent AI paper trade
#[derive(Debug, Clone)]
pub struct AiRecentTrade {
    pub pair: String,
    pub side: String,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub pnl: Option<f64>,
    pub pnl_pct: Option<f64>,
    pub status: String,
}

/// AI Advisor widget for the trading UI
pub struct AiAdvisorWidget<'a> {
    signal: Option<&'a AiSignalData>,
    stats: Option<&'a AiTradeStats>,
    recent_trades: Vec<&'a AiRecentTrade>,
}

impl<'a> AiAdvisorWidget<'a> {
    pub fn new(
        signal: Option<&'a AiSignalData>,
        stats: Option<&'a AiTradeStats>,
        recent_trades: Vec<&'a AiRecentTrade>,
    ) -> Self {
        Self {
            signal,
            stats,
            recent_trades,
        }
    }
}

impl Widget for AiAdvisorWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" AI Advisor [XGBoost] ")
            .title_style(
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            )
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.width < 20 || inner.height < 4 {
            return;
        }

        let mut y = inner.y;

        // Current Signal Section
        if let Some(signal) = self.signal {
            // Signal with confidence bar
            let signal_color = match signal.signal.as_str() {
                "BUY" => Color::Green,
                "SELL" => Color::Red,
                _ => Color::Yellow,
            };

            let signal_str = format!(" {} ", signal.signal);
            buf.set_string(
                inner.x + 1,
                y,
                &signal_str,
                Style::default()
                    .fg(Color::Black)
                    .bg(signal_color)
                    .add_modifier(Modifier::BOLD),
            );

            // Confidence percentage
            let conf_str = format!("{:.0}%", signal.confidence * 100.0);
            buf.set_string(
                inner.x + signal_str.len() as u16 + 2,
                y,
                &conf_str,
                Style::default().fg(signal_color).add_modifier(Modifier::BOLD),
            );

            // Price
            let price_str = format!("@ ${:.2}", signal.price);
            buf.set_string(
                inner.x + signal_str.len() as u16 + conf_str.len() as u16 + 4,
                y,
                &price_str,
                Style::default().fg(Color::White),
            );

            y += 1;

            // Confidence bar
            if y < inner.y + inner.height {
                let bar_width = (inner.width - 4).min(30) as usize;
                let filled = ((signal.confidence * bar_width as f64) as usize).min(bar_width);
                let empty = bar_width - filled;

                buf.set_string(inner.x + 1, y, "[", Style::default().fg(Color::DarkGray));
                buf.set_string(
                    inner.x + 2,
                    y,
                    &"█".repeat(filled),
                    Style::default().fg(signal_color),
                );
                buf.set_string(
                    inner.x + 2 + filled as u16,
                    y,
                    &"░".repeat(empty),
                    Style::default().fg(Color::DarkGray),
                );
                buf.set_string(
                    inner.x + 2 + bar_width as u16,
                    y,
                    "]",
                    Style::default().fg(Color::DarkGray),
                );
            }

            y += 1;

            // Indicators
            if y < inner.y + inner.height {
                let mut ind_str = String::new();

                if let Some(rsi) = signal.rsi {
                    let rsi_color = if rsi > 70.0 {
                        "▲" // Overbought
                    } else if rsi < 30.0 {
                        "▼" // Oversold
                    } else {
                        "●"
                    };
                    ind_str.push_str(&format!("RSI:{:.0}{} ", rsi, rsi_color));
                }

                if let Some(macd_hist) = signal.macd_hist {
                    let macd_sign = if macd_hist > 0.0 { "+" } else { "" };
                    ind_str.push_str(&format!("MACD:{}{:.2}", macd_sign, macd_hist));
                }

                if !ind_str.is_empty() {
                    buf.set_string(
                        inner.x + 1,
                        y,
                        &ind_str,
                        Style::default().fg(Color::Cyan),
                    );
                }
            }

            y += 1;

            // Last update time
            if y < inner.y + inner.height && !signal.last_update.is_empty() {
                let time_str = format!("Updated: {}", signal.last_update);
                buf.set_string(
                    inner.x + 1,
                    y,
                    &time_str,
                    Style::default().fg(Color::DarkGray),
                );
            }

            y += 1;
        } else {
            buf.set_string(
                inner.x + 1,
                y,
                "No signal data",
                Style::default().fg(Color::DarkGray),
            );
            y += 2;
        }

        // Separator
        if y < inner.y + inner.height {
            buf.set_string(
                inner.x + 1,
                y,
                &"─".repeat((inner.width - 2) as usize),
                Style::default().fg(Color::DarkGray),
            );
            y += 1;
        }

        // Stats Section
        if let Some(stats) = self.stats {
            if y < inner.y + inner.height {
                let pnl_color = if stats.total_pnl >= 0.0 {
                    Color::Green
                } else {
                    Color::Red
                };
                let pnl_sign = if stats.total_pnl >= 0.0 { "+" } else { "" };

                let stats_str = format!(
                    "Trades: {} | Win: {:.0}% | PnL: {}${:.2}",
                    stats.total_trades, stats.win_rate, pnl_sign, stats.total_pnl
                );
                buf.set_string(
                    inner.x + 1,
                    y,
                    &stats_str,
                    Style::default().fg(Color::White),
                );

                // Highlight PnL portion
                let pnl_start = stats_str.find("PnL:").unwrap_or(0);
                buf.set_string(
                    inner.x + 1 + pnl_start as u16,
                    y,
                    &stats_str[pnl_start..],
                    Style::default().fg(pnl_color),
                );
            }

            y += 1;

            if y < inner.y + inner.height && stats.open_positions > 0 {
                let open_str = format!("Open: {} position(s)", stats.open_positions);
                buf.set_string(
                    inner.x + 1,
                    y,
                    &open_str,
                    Style::default().fg(Color::Yellow),
                );
            }

            y += 1;
        }

        // Recent Trades
        if !self.recent_trades.is_empty() && y + 2 < inner.y + inner.height {
            // Separator
            buf.set_string(
                inner.x + 1,
                y,
                &"─".repeat((inner.width - 2) as usize),
                Style::default().fg(Color::DarkGray),
            );
            y += 1;

            buf.set_string(
                inner.x + 1,
                y,
                "Recent:",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            );
            y += 1;

            for trade in self.recent_trades.iter().take(3) {
                if y >= inner.y + inner.height {
                    break;
                }

                let side_color = if trade.side == "buy" {
                    Color::Green
                } else {
                    Color::Red
                };

                let side_str = if trade.side == "buy" { "B" } else { "S" };

                buf.set_string(
                    inner.x + 1,
                    y,
                    side_str,
                    Style::default().fg(side_color).add_modifier(Modifier::BOLD),
                );

                let entry_str = format!("${:.0}", trade.entry_price);
                buf.set_string(
                    inner.x + 3,
                    y,
                    &entry_str,
                    Style::default().fg(Color::White),
                );

                if let (Some(pnl), Some(pnl_pct)) = (trade.pnl, trade.pnl_pct) {
                    let pnl_color = if pnl >= 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    };
                    let sign = if pnl >= 0.0 { "+" } else { "" };
                    let pnl_str = format!("{}${:.2} ({}{:.1}%)", sign, pnl, sign, pnl_pct);
                    buf.set_string(
                        inner.x + 12,
                        y,
                        &pnl_str,
                        Style::default().fg(pnl_color),
                    );
                } else {
                    buf.set_string(
                        inner.x + 12,
                        y,
                        &trade.status,
                        Style::default().fg(Color::Yellow),
                    );
                }

                y += 1;
            }
        }
    }
}
