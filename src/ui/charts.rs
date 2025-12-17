use crate::data::Candle;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Widget},
};
use rust_decimal::prelude::ToPrimitive;

pub struct PriceChart<'a> {
    candles: &'a [&'a Candle],
    title: String,
}

impl<'a> PriceChart<'a> {
    pub fn new(candles: &'a [&'a Candle], pair: &str) -> Self {
        Self {
            candles,
            title: format!(" {} Price ", pair),
        }
    }
}

impl Widget for PriceChart<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(self.title.as_str())
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        if self.candles.is_empty() || inner.width < 4 || inner.height < 4 {
            return;
        }

        // Find price range
        let (min_price, max_price) = self.price_range();
        if min_price == max_price {
            return;
        }

        let price_range = max_price - min_price;
        let height = inner.height as f64;

        // Calculate how many candles we can display
        let candle_width = 3; // Each candle takes 3 columns
        let max_candles = (inner.width as usize / candle_width).min(self.candles.len());
        let candles_to_show = &self.candles[self.candles.len().saturating_sub(max_candles)..];

        // Draw candles
        for (i, candle) in candles_to_show.iter().enumerate() {
            let x = inner.x + (i * candle_width) as u16 + 1;
            if x >= inner.x + inner.width - 1 {
                break;
            }

            let open = candle.open.to_f64().unwrap_or(0.0);
            let close = candle.close.to_f64().unwrap_or(0.0);
            let high = candle.high.to_f64().unwrap_or(0.0);
            let low = candle.low.to_f64().unwrap_or(0.0);

            let is_bullish = close >= open;
            let color = if is_bullish { Color::Green } else { Color::Red };

            // Convert prices to Y positions (inverted: higher price = lower Y)
            let y_open = height - ((open - min_price) / price_range * height);
            let y_close = height - ((close - min_price) / price_range * height);
            let y_high = height - ((high - min_price) / price_range * height);
            let y_low = height - ((low - min_price) / price_range * height);

            let body_top = y_open.min(y_close).max(0.0) as u16;
            let body_bottom = y_open.max(y_close).min(height) as u16;
            let wick_top = y_high.max(0.0) as u16;
            let wick_bottom = y_low.min(height) as u16;

            // Draw upper wick
            for y in wick_top..body_top {
                let py = inner.y + y;
                if py < inner.y + inner.height {
                    buf[(x, py)]
                        .set_char('│')
                        .set_fg(color);
                }
            }

            // Draw candle body
            for y in body_top..=body_bottom {
                let py = inner.y + y;
                if py < inner.y + inner.height {
                    let ch = if is_bullish { '█' } else { '▓' };
                    buf[(x, py)]
                        .set_char(ch)
                        .set_fg(color);
                }
            }

            // Draw lower wick
            for y in (body_bottom + 1)..=wick_bottom {
                let py = inner.y + y;
                if py < inner.y + inner.height {
                    buf[(x, py)]
                        .set_char('│')
                        .set_fg(color);
                }
            }
        }

        // Draw price labels on the right
        self.draw_price_labels(inner, buf, min_price, max_price);
    }
}

impl PriceChart<'_> {
    fn price_range(&self) -> (f64, f64) {
        let mut min = f64::MAX;
        let mut max = f64::MIN;

        for candle in self.candles {
            let low = candle.low.to_f64().unwrap_or(0.0);
            let high = candle.high.to_f64().unwrap_or(0.0);
            min = min.min(low);
            max = max.max(high);
        }

        // Add some padding
        let padding = (max - min) * 0.05;
        (min - padding, max + padding)
    }

    fn draw_price_labels(&self, area: Rect, buf: &mut Buffer, min: f64, max: f64) {
        if area.width < 10 {
            return;
        }

        let label_x = area.x + area.width - 8;
        let range = max - min;

        // Draw 3 price labels: top, middle, bottom
        let labels = [
            (0, max),
            (area.height / 2, min + range / 2.0),
            (area.height - 1, min),
        ];

        for (y_offset, price) in labels {
            let y = area.y + y_offset;
            if y < area.y + area.height {
                let label = format!("{:.0}", price);
                let label_len = label.len().min((area.x + area.width - label_x) as usize);
                for (i, ch) in label.chars().take(label_len).enumerate() {
                    let x = label_x + i as u16;
                    if x < area.x + area.width {
                        buf[(x, y)]
                            .set_char(ch)
                            .set_fg(Color::DarkGray);
                    }
                }
            }
        }
    }
}

/// Simple sparkline for showing recent price movement
pub struct Sparkline<'a> {
    data: &'a [f64],
    color: Color,
}

impl<'a> Sparkline<'a> {
    pub fn new(data: &'a [f64]) -> Self {
        Self {
            data,
            color: Color::Cyan,
        }
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

impl Widget for Sparkline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.data.is_empty() || area.width == 0 || area.height == 0 {
            return;
        }

        let min = self.data.iter().copied().fold(f64::MAX, f64::min);
        let max = self.data.iter().copied().fold(f64::MIN, f64::max);
        let range = if max == min { 1.0 } else { max - min };

        let bar_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        let width = area.width as usize;
        let data_len = self.data.len();

        for x in 0..width {
            let data_idx = (x * data_len) / width;
            if data_idx < data_len {
                let value = self.data[data_idx];
                let normalized = ((value - min) / range).clamp(0.0, 1.0);
                let char_idx = (normalized * 7.0) as usize;
                let ch = bar_chars[char_idx.min(7)];

                buf[(area.x + x as u16, area.y)]
                    .set_char(ch)
                    .set_fg(self.color);
            }
        }
    }
}
