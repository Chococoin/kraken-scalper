//! Performance metrics calculation for backtesting

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::VecDeque;

/// Performance metrics from a backtest run
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    // Basic stats
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub win_rate: f64,

    // P&L
    pub total_pnl: Decimal,
    pub total_pnl_pct: f64,
    pub gross_profit: Decimal,
    pub gross_loss: Decimal,
    pub avg_win: Decimal,
    pub avg_loss: Decimal,
    pub profit_factor: f64,
    pub expectancy: Decimal,
    pub largest_win: Decimal,
    pub largest_loss: Decimal,

    // Risk
    pub max_drawdown: f64,
    pub max_drawdown_abs: Decimal,
    pub max_drawdown_duration: Duration,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,

    // Time
    pub avg_trade_duration: Duration,
    pub start_balance: Decimal,
    pub end_balance: Decimal,
    pub total_duration: Duration,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            total_pnl: Decimal::ZERO,
            total_pnl_pct: 0.0,
            gross_profit: Decimal::ZERO,
            gross_loss: Decimal::ZERO,
            avg_win: Decimal::ZERO,
            avg_loss: Decimal::ZERO,
            profit_factor: 0.0,
            expectancy: Decimal::ZERO,
            largest_win: Decimal::ZERO,
            largest_loss: Decimal::ZERO,
            max_drawdown: 0.0,
            max_drawdown_abs: Decimal::ZERO,
            max_drawdown_duration: Duration::zero(),
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            avg_trade_duration: Duration::zero(),
            start_balance: Decimal::ZERO,
            end_balance: Decimal::ZERO,
            total_duration: Duration::zero(),
        }
    }
}

/// Collects data during backtest and calculates final metrics
pub struct MetricsCollector {
    start_balance: Decimal,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,

    // Equity curve
    equity_curve: Vec<(DateTime<Utc>, Decimal)>,

    // Trade P&Ls
    trade_pnls: Vec<Decimal>,
    trade_durations: Vec<Duration>,

    // For drawdown calculation
    peak_equity: Decimal,
    current_drawdown_start: Option<DateTime<Utc>>,
    max_drawdown: f64,
    max_drawdown_abs: Decimal,
    max_drawdown_duration: Duration,

    // For Sharpe/Sortino
    returns: VecDeque<f64>,
    last_equity: Decimal,
}

impl MetricsCollector {
    pub fn new(start_balance: Decimal) -> Self {
        Self {
            start_balance,
            start_time: None,
            end_time: None,
            equity_curve: Vec::new(),
            trade_pnls: Vec::new(),
            trade_durations: Vec::new(),
            peak_equity: start_balance,
            current_drawdown_start: None,
            max_drawdown: 0.0,
            max_drawdown_abs: Decimal::ZERO,
            max_drawdown_duration: Duration::zero(),
            returns: VecDeque::new(),
            last_equity: start_balance,
        }
    }

    /// Record equity at a point in time
    pub fn record_equity(&mut self, ts: DateTime<Utc>, equity: Decimal) {
        if self.start_time.is_none() {
            self.start_time = Some(ts);
        }
        self.end_time = Some(ts);

        // Calculate return
        if !self.last_equity.is_zero() {
            let ret = ((equity - self.last_equity) / self.last_equity)
                .to_f64()
                .unwrap_or(0.0);
            self.returns.push_back(ret);

            // Keep last 1000 returns for ratio calculations
            if self.returns.len() > 1000 {
                self.returns.pop_front();
            }
        }
        self.last_equity = equity;

        // Update peak and drawdown
        if equity > self.peak_equity {
            self.peak_equity = equity;
            // End of drawdown period
            if let Some(dd_start) = self.current_drawdown_start.take() {
                let dd_duration = ts - dd_start;
                if dd_duration > self.max_drawdown_duration {
                    self.max_drawdown_duration = dd_duration;
                }
            }
        } else if !self.peak_equity.is_zero() {
            // In drawdown
            let drawdown_abs = self.peak_equity - equity;
            let drawdown_pct = (drawdown_abs / self.peak_equity)
                .to_f64()
                .unwrap_or(0.0)
                * 100.0;

            if drawdown_pct > self.max_drawdown {
                self.max_drawdown = drawdown_pct;
                self.max_drawdown_abs = drawdown_abs;
            }

            if self.current_drawdown_start.is_none() {
                self.current_drawdown_start = Some(ts);
            }
        }

        self.equity_curve.push((ts, equity));
    }

    /// Record a completed trade
    pub fn record_trade(&mut self, pnl: Decimal, duration: Duration) {
        self.trade_pnls.push(pnl);
        self.trade_durations.push(duration);
    }

    /// Get the equity curve
    pub fn equity_curve(&self) -> &[(DateTime<Utc>, Decimal)] {
        &self.equity_curve
    }

    /// Calculate final metrics
    pub fn calculate(&self) -> PerformanceMetrics {
        let total_trades = self.trade_pnls.len() as u32;

        if total_trades == 0 {
            return PerformanceMetrics {
                start_balance: self.start_balance,
                end_balance: self.last_equity,
                ..Default::default()
            };
        }

        // Separate wins and losses
        let mut wins: Vec<Decimal> = Vec::new();
        let mut losses: Vec<Decimal> = Vec::new();

        for &pnl in &self.trade_pnls {
            if pnl > Decimal::ZERO {
                wins.push(pnl);
            } else if pnl < Decimal::ZERO {
                losses.push(pnl);
            }
        }

        let winning_trades = wins.len() as u32;
        let losing_trades = losses.len() as u32;
        let win_rate = if total_trades > 0 {
            (winning_trades as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        // P&L calculations
        let gross_profit: Decimal = wins.iter().sum();
        let gross_loss: Decimal = losses.iter().sum(); // Will be negative
        let total_pnl = gross_profit + gross_loss;

        let avg_win = if !wins.is_empty() {
            gross_profit / Decimal::from(wins.len())
        } else {
            Decimal::ZERO
        };

        let avg_loss = if !losses.is_empty() {
            gross_loss / Decimal::from(losses.len())
        } else {
            Decimal::ZERO
        };

        let profit_factor = if !gross_loss.is_zero() {
            (gross_profit / gross_loss.abs()).to_f64().unwrap_or(0.0)
        } else if gross_profit > Decimal::ZERO {
            f64::INFINITY
        } else {
            0.0
        };

        // Expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
        let win_rate_dec = Decimal::try_from(win_rate / 100.0).unwrap_or_default();
        let loss_rate_dec = Decimal::ONE - win_rate_dec;
        let expectancy = (win_rate_dec * avg_win) + (loss_rate_dec * avg_loss);

        let largest_win = wins.iter().max().copied().unwrap_or(Decimal::ZERO);
        let largest_loss = losses.iter().min().copied().unwrap_or(Decimal::ZERO);

        let total_pnl_pct = if !self.start_balance.is_zero() {
            (total_pnl / self.start_balance).to_f64().unwrap_or(0.0) * 100.0
        } else {
            0.0
        };

        // Average trade duration
        let total_duration_secs: i64 = self.trade_durations.iter().map(|d| d.num_seconds()).sum();
        let avg_trade_duration = if total_trades > 0 {
            Duration::seconds(total_duration_secs / total_trades as i64)
        } else {
            Duration::zero()
        };

        // Sharpe Ratio (annualized, assuming daily returns)
        let sharpe_ratio = calculate_sharpe(&self.returns);

        // Sortino Ratio (using only downside deviation)
        let sortino_ratio = calculate_sortino(&self.returns);

        // Total duration
        let total_duration = match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => end - start,
            _ => Duration::zero(),
        };

        PerformanceMetrics {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            total_pnl,
            total_pnl_pct,
            gross_profit,
            gross_loss,
            avg_win,
            avg_loss,
            profit_factor,
            expectancy,
            largest_win,
            largest_loss,
            max_drawdown: self.max_drawdown,
            max_drawdown_abs: self.max_drawdown_abs,
            max_drawdown_duration: self.max_drawdown_duration,
            sharpe_ratio,
            sortino_ratio,
            avg_trade_duration,
            start_balance: self.start_balance,
            end_balance: self.last_equity,
            total_duration,
        }
    }
}

/// Calculate Sharpe Ratio from returns
fn calculate_sharpe(returns: &VecDeque<f64>) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 0.0;
    }

    // Annualize (assuming ~252 trading periods per year for crypto 24/7 with 30s intervals)
    // For 30-second intervals: 2880 periods per day * 365 days
    let periods_per_year: f64 = 2880.0 * 365.0;
    (mean / std_dev) * periods_per_year.sqrt()
}

/// Calculate Sortino Ratio from returns (downside deviation only)
fn calculate_sortino(returns: &VecDeque<f64>) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

    // Only consider negative returns for downside deviation
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    if downside_returns.is_empty() {
        return if mean > 0.0 { f64::INFINITY } else { 0.0 };
    }

    let downside_variance: f64 =
        downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
    let downside_dev = downside_variance.sqrt();

    if downside_dev == 0.0 {
        return 0.0;
    }

    let periods_per_year: f64 = 2880.0 * 365.0;
    (mean / downside_dev) * periods_per_year.sqrt()
}
