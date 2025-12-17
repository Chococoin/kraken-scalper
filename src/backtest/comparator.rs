//! Strategy Comparator
//!
//! Runs multiple strategies on the same data and compares results.

use super::engine::{BacktestConfig, BacktestResult};
use super::loader::DataLoader;
use super::metrics::MetricsCollector;
use crate::data::Ticker;
use crate::strategy::{create_strategy, Signal, Strategy};
use crate::trading::order::{OrderSide, Position};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::fs::File;
use std::io::Write;
use tracing::info;

/// Result of comparing multiple strategies
#[derive(Debug)]
pub struct ComparisonResult {
    pub config: BacktestConfig,
    pub results: Vec<(String, BacktestResult)>,
    pub snapshots_count: usize,
}

impl ComparisonResult {
    /// Get the best strategy by P&L percentage
    pub fn best_by_pnl(&self) -> Option<&(String, BacktestResult)> {
        self.results
            .iter()
            .max_by(|a, b| {
                a.1.metrics
                    .total_pnl_pct
                    .partial_cmp(&b.1.metrics.total_pnl_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get the best strategy by Sharpe ratio
    pub fn best_by_sharpe(&self) -> Option<&(String, BacktestResult)> {
        self.results
            .iter()
            .filter(|(_, r)| !r.metrics.sharpe_ratio.is_nan())
            .max_by(|a, b| {
                a.1.metrics
                    .sharpe_ratio
                    .partial_cmp(&b.1.metrics.sharpe_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Export results to CSV
    pub fn export_csv(&self, path: &str) -> Result<()> {
        let mut file = File::create(path).context("Failed to create CSV file")?;

        // Header
        writeln!(
            file,
            "strategy,trades,win_rate,pnl,pnl_pct,max_drawdown,sharpe_ratio,profit_factor,avg_win,avg_loss"
        )?;

        // Data rows
        for (name, result) in &self.results {
            let m = &result.metrics;
            writeln!(
                file,
                "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                name,
                m.total_trades,
                m.win_rate,
                m.total_pnl.to_f64().unwrap_or(0.0),
                m.total_pnl_pct,
                m.max_drawdown,
                if m.sharpe_ratio.is_nan() { 0.0 } else { m.sharpe_ratio },
                if m.profit_factor.is_infinite() { 0.0 } else { m.profit_factor },
                m.avg_win.to_f64().unwrap_or(0.0),
                m.avg_loss.to_f64().unwrap_or(0.0),
            )?;
        }

        Ok(())
    }
}

/// Simple in-memory trader for backtesting
struct BacktestTrader {
    balance: Decimal,
    position: Option<Position>,
    commission_pct: Decimal,
    slippage_pct: Decimal,
    next_position_id: i64,
    highest_since_entry: Decimal,
}

impl BacktestTrader {
    fn new(initial_balance: Decimal, commission_pct: Decimal, slippage_pct: Decimal) -> Self {
        Self {
            balance: initial_balance,
            position: None,
            commission_pct,
            slippage_pct,
            next_position_id: 1,
            highest_since_entry: Decimal::ZERO,
        }
    }

    fn has_position(&self) -> bool {
        self.position.is_some()
    }

    fn get_position(&self) -> Option<&Position> {
        self.position.as_ref()
    }

    fn equity(&self) -> Decimal {
        let position_value = self
            .position
            .as_ref()
            .map(|p| p.current_price * p.quantity)
            .unwrap_or(Decimal::ZERO);
        self.balance + position_value
    }

    fn update_position(&mut self, ticker: &Ticker) {
        if let Some(pos) = &mut self.position {
            pos.update_price(ticker.last);
            if ticker.last > self.highest_since_entry {
                self.highest_since_entry = ticker.last;
            }
        }
    }

    fn execute_buy(
        &mut self,
        pair: &str,
        quantity: Decimal,
        ticker: &Ticker,
        ts: DateTime<Utc>,
    ) -> Option<Decimal> {
        let slippage = ticker.ask * self.slippage_pct;
        let execution_price = ticker.ask + slippage;
        let order_value = execution_price * quantity;
        let commission = order_value * self.commission_pct;
        let total_cost = order_value + commission;

        if total_cost > self.balance {
            return None;
        }

        self.balance -= total_cost;

        let position_id = self.next_position_id;
        self.next_position_id += 1;

        let mut position = Position::new(position_id, pair, OrderSide::Buy, execution_price, quantity);
        position.opened_at = ts;
        self.position = Some(position);
        self.highest_since_entry = execution_price;

        Some(execution_price)
    }

    fn execute_sell(&mut self, ticker: &Ticker, _ts: DateTime<Utc>) -> Option<Decimal> {
        let position = self.position.take()?;

        let slippage = ticker.bid * self.slippage_pct;
        let execution_price = ticker.bid - slippage;
        let order_value = execution_price * position.quantity;
        let commission = order_value * self.commission_pct;

        let gross_pnl = position.calculate_pnl(execution_price);
        let entry_commission = (position.entry_price * position.quantity) * self.commission_pct;
        let net_pnl = gross_pnl - commission - entry_commission;

        self.balance += order_value - commission;

        Some(net_pnl)
    }
}

/// Run a single strategy backtest
fn run_strategy_backtest(
    strategy: &mut dyn Strategy,
    snapshots: &[(DateTime<Utc>, Ticker)],
    config: &BacktestConfig,
) -> BacktestResult {
    let mut trader = BacktestTrader::new(
        config.initial_balance,
        config.commission_pct,
        config.slippage_pct,
    );
    let mut metrics = MetricsCollector::new(config.initial_balance);
    let mut trades = Vec::new();

    let max_position_value = config.initial_balance * Decimal::try_from(0.1).unwrap_or(Decimal::ONE);

    for (ts, ticker) in snapshots {
        trader.update_position(ticker);
        metrics.record_equity(*ts, trader.equity());

        let position = trader.get_position();

        // Check exit conditions
        if let Some(pos) = position {
            let should_exit = strategy.should_take_profit(pos, ticker.last)
                || strategy.should_stop_loss(pos, ticker.last);

            if should_exit {
                let opened_at = pos.opened_at;
                if let Some(pnl) = trader.execute_sell(ticker, *ts) {
                    let duration = *ts - opened_at;
                    metrics.record_trade(pnl, duration);
                    trades.push(pnl);
                }
                continue;
            }
        }

        let signal = strategy.analyze(ticker, None, trader.get_position());

        match signal {
            Signal::Buy { pair, quantity } => {
                if !trader.has_position() {
                    let price = ticker.ask;
                    let max_qty = if !price.is_zero() {
                        max_position_value / price
                    } else {
                        quantity
                    };
                    let actual_qty = quantity.min(max_qty);
                    trader.execute_buy(&pair, actual_qty, ticker, *ts);
                }
            }
            Signal::Sell { pair: _ } => {
                if let Some(pos) = trader.get_position() {
                    let opened_at = pos.opened_at;
                    if let Some(pnl) = trader.execute_sell(ticker, *ts) {
                        let duration = *ts - opened_at;
                        metrics.record_trade(pnl, duration);
                        trades.push(pnl);
                    }
                }
            }
            Signal::Hold => {}
        }
    }

    // Close remaining position
    if let Some((ts, ticker)) = snapshots.last() {
        if let Some(pos) = trader.get_position() {
            let opened_at = pos.opened_at;
            if let Some(pnl) = trader.execute_sell(ticker, *ts) {
                let duration = *ts - opened_at;
                metrics.record_trade(pnl, duration);
                trades.push(pnl);
            }
        }
    }

    let performance = metrics.calculate();
    let equity_curve = metrics.equity_curve().to_vec();

    BacktestResult {
        config: config.clone(),
        trades: Vec::new(), // We don't need detailed trades for comparison
        metrics: performance,
        equity_curve,
    }
}

/// Compare multiple strategies on the same data
pub fn compare_strategies(
    strategy_names: &[String],
    config: BacktestConfig,
    strategy_config: &crate::config::StrategyConfig,
    data_dir: &str,
) -> Result<ComparisonResult> {
    info!(
        "Comparing {} strategies for {} from {} to {}",
        strategy_names.len(),
        config.pair,
        config.start_date,
        config.end_date
    );

    // Load data once
    let loader = DataLoader::new(data_dir);
    let snapshots = loader
        .load_range(&config.pair, config.start_date, config.end_date)
        .context("Failed to load historical data")?;

    info!("Loaded {} market snapshots", snapshots.len());

    if snapshots.is_empty() {
        return Ok(ComparisonResult {
            config,
            results: Vec::new(),
            snapshots_count: 0,
        });
    }

    // Convert to ticker-only snapshots
    let ticker_snapshots: Vec<(DateTime<Utc>, Ticker)> = snapshots
        .into_iter()
        .filter_map(|s| s.ticker.map(|t| (s.ts, t)))
        .collect();

    let snapshots_count = ticker_snapshots.len();

    // Run each strategy
    let mut results = Vec::new();

    for name in strategy_names {
        let mut strategy = match create_strategy(name, strategy_config) {
            Some(s) => s,
            None => {
                info!("Unknown strategy: {}", name);
                continue;
            }
        };

        info!("Running strategy: {}", strategy.name());
        let result = run_strategy_backtest(strategy.as_mut(), &ticker_snapshots, &config);

        info!(
            "  {} trades, {:.2}% P&L",
            result.metrics.total_trades, result.metrics.total_pnl_pct
        );

        results.push((strategy.name().to_string(), result));
    }

    // Sort by P&L descending
    results.sort_by(|a, b| {
        b.1.metrics
            .total_pnl_pct
            .partial_cmp(&a.1.metrics.total_pnl_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ComparisonResult {
        config,
        results,
        snapshots_count,
    })
}
