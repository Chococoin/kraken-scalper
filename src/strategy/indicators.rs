//! Technical indicators for trading strategies

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;

/// Simple Moving Average
pub fn sma(prices: &[Decimal], period: usize) -> Option<Decimal> {
    if prices.len() < period || period == 0 {
        return None;
    }

    let sum: Decimal = prices.iter().rev().take(period).sum();
    Some(sum / Decimal::from(period))
}

/// Exponential Moving Average
pub fn ema(prices: &[Decimal], period: usize) -> Option<Decimal> {
    if prices.len() < period || period == 0 {
        return None;
    }

    let multiplier = Decimal::from(2) / Decimal::from(period + 1);

    // Start with SMA of first `period` prices
    let initial_sma: Decimal = prices.iter().take(period).sum::<Decimal>() / Decimal::from(period);

    // Calculate EMA for remaining prices
    let mut ema = initial_sma;
    for price in prices.iter().skip(period) {
        ema = (*price - ema) * multiplier + ema;
    }

    Some(ema)
}

/// Standard Deviation
pub fn std_dev(prices: &[Decimal], period: usize) -> Option<Decimal> {
    if prices.len() < period || period == 0 {
        return None;
    }

    let recent: Vec<_> = prices.iter().rev().take(period).copied().collect();
    let mean = sma(prices, period)?;

    let variance: Decimal = recent
        .iter()
        .map(|p| (*p - mean) * (*p - mean))
        .sum::<Decimal>()
        / Decimal::from(period);

    // Square root approximation using Newton's method
    let variance_f64 = variance.to_f64()?;
    let std_dev_f64 = variance_f64.sqrt();

    Decimal::try_from(std_dev_f64).ok()
}

/// Relative Strength Index (RSI)
pub fn rsi(prices: &[Decimal], period: usize) -> Option<Decimal> {
    if prices.len() < period + 1 || period == 0 {
        return None;
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    // Calculate price changes
    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > Decimal::ZERO {
            gains.push(change);
            losses.push(Decimal::ZERO);
        } else {
            gains.push(Decimal::ZERO);
            losses.push(change.abs());
        }
    }

    if gains.len() < period {
        return None;
    }

    // Calculate average gain and loss using EMA-style smoothing
    let mut avg_gain: Decimal = gains.iter().take(period).sum::<Decimal>() / Decimal::from(period);
    let mut avg_loss: Decimal = losses.iter().take(period).sum::<Decimal>() / Decimal::from(period);

    for i in period..gains.len() {
        avg_gain = (avg_gain * Decimal::from(period - 1) + gains[i]) / Decimal::from(period);
        avg_loss = (avg_loss * Decimal::from(period - 1) + losses[i]) / Decimal::from(period);
    }

    if avg_loss.is_zero() {
        return Some(dec!(100));
    }

    let rs = avg_gain / avg_loss;
    let rsi = dec!(100) - (dec!(100) / (dec!(1) + rs));

    Some(rsi)
}

/// Average True Range (ATR)
pub fn atr(
    highs: &[Decimal],
    lows: &[Decimal],
    closes: &[Decimal],
    period: usize,
) -> Option<Decimal> {
    if highs.len() < period + 1 || lows.len() < period + 1 || closes.len() < period + 1 {
        return None;
    }

    let mut true_ranges = Vec::new();

    for i in 1..highs.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i - 1]).abs();
        let low_close = (lows[i] - closes[i - 1]).abs();

        let tr = high_low.max(high_close).max(low_close);
        true_ranges.push(tr);
    }

    if true_ranges.len() < period {
        return None;
    }

    // Calculate ATR using EMA-style smoothing
    let mut atr: Decimal = true_ranges.iter().take(period).sum::<Decimal>() / Decimal::from(period);

    for tr in true_ranges.iter().skip(period) {
        atr = (atr * Decimal::from(period - 1) + *tr) / Decimal::from(period);
    }

    Some(atr)
}

/// Bollinger Bands - returns (lower, middle, upper)
pub fn bollinger_bands(
    prices: &[Decimal],
    period: usize,
    num_std_devs: f64,
) -> Option<(Decimal, Decimal, Decimal)> {
    let middle = sma(prices, period)?;
    let std = std_dev(prices, period)?;
    let std_multiplier = Decimal::try_from(num_std_devs).ok()?;

    let lower = middle - std * std_multiplier;
    let upper = middle + std * std_multiplier;

    Some((lower, middle, upper))
}

/// Highest value in last N periods
pub fn highest(prices: &[Decimal], period: usize) -> Option<Decimal> {
    if prices.len() < period || period == 0 {
        return None;
    }

    prices.iter().rev().take(period).max().copied()
}

/// Lowest value in last N periods
pub fn lowest(prices: &[Decimal], period: usize) -> Option<Decimal> {
    if prices.len() < period || period == 0 {
        return None;
    }

    prices.iter().rev().take(period).min().copied()
}

/// Check if fast MA crossed above slow MA (golden cross)
pub fn crossed_above(fast_prev: Decimal, fast_curr: Decimal, slow_prev: Decimal, slow_curr: Decimal) -> bool {
    fast_prev <= slow_prev && fast_curr > slow_curr
}

/// Check if fast MA crossed below slow MA (death cross)
pub fn crossed_below(fast_prev: Decimal, fast_curr: Decimal, slow_prev: Decimal, slow_curr: Decimal) -> bool {
    fast_prev >= slow_prev && fast_curr < slow_curr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let prices = vec![dec!(10), dec!(11), dec!(12), dec!(13), dec!(14)];
        let result = sma(&prices, 3);
        assert_eq!(result, Some(dec!(13))); // (12 + 13 + 14) / 3
    }

    #[test]
    fn test_rsi_overbought() {
        // Prices going up consistently should give high RSI
        let prices: Vec<Decimal> = (0..20).map(|i| Decimal::from(100 + i)).collect();
        let result = rsi(&prices, 14);
        assert!(result.is_some());
        assert!(result.unwrap() > dec!(70));
    }

    #[test]
    fn test_highest() {
        let prices = vec![dec!(10), dec!(15), dec!(12), dec!(18), dec!(14)];
        assert_eq!(highest(&prices, 3), Some(dec!(18)));
    }

    #[test]
    fn test_crossed_above() {
        assert!(crossed_above(dec!(9), dec!(11), dec!(10), dec!(10)));
        assert!(!crossed_above(dec!(11), dec!(12), dec!(10), dec!(10)));
    }
}
