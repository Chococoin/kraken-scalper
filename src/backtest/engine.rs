//! Backtest engine for replaying historical data through strategies

use super::loader::DataLoader;
use super::metrics::{MetricsCollector, PerformanceMetrics};
use crate::config::StrategyConfig;
use crate::data::Ticker;
use crate::strategy::{ScalperStrategy, Signal, Strategy};
use crate::trading::order::{OrderSide, Position};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use tracing::{debug, info};

/// Configuration for a backtest run
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub pair: String,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub initial_balance: Decimal,
    pub commission_pct: Decimal,
    pub slippage_pct: Decimal,
}

impl BacktestConfig {
    pub fn new(pair: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self {
            pair: pair.to_string(),
            start_date: start,
            end_date: end,
            initial_balance: Decimal::from(10000),
            commission_pct: Decimal::try_from(0.0026).unwrap(), // 0.26% Kraken taker fee
            slippage_pct: Decimal::try_from(0.0005).unwrap(),   // 0.05% slippage
        }
    }

    pub fn with_balance(mut self, balance: Decimal) -> Self {
        self.initial_balance = balance;
        self
    }

    pub fn with_commission(mut self, commission_pct: f64) -> Self {
        self.commission_pct = Decimal::try_from(commission_pct / 100.0).unwrap_or_default();
        self
    }

    pub fn with_slippage(mut self, slippage_pct: f64) -> Self {
        self.slippage_pct = Decimal::try_from(slippage_pct / 100.0).unwrap_or_default();
        self
    }
}

/// Record of a trade executed during backtest
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub ts: DateTime<Utc>,
    pub pair: String,
    pub side: OrderSide,
    pub price: Decimal,
    pub quantity: Decimal,
    pub value: Decimal,
    pub commission: Decimal,
    pub pnl: Option<Decimal>,
}

/// Result of a backtest run
#[derive(Debug)]
pub struct BacktestResult {
    pub config: BacktestConfig,
    pub trades: Vec<TradeRecord>,
    pub metrics: PerformanceMetrics,
    pub equity_curve: Vec<(DateTime<Utc>, Decimal)>,
}

/// Simple in-memory trader for backtesting (no async DB)
struct BacktestTrader {
    balance: Decimal,
    position: Option<Position>,
    commission_pct: Decimal,
    slippage_pct: Decimal,
    next_position_id: i64,
}

impl BacktestTrader {
    fn new(initial_balance: Decimal, commission_pct: Decimal, slippage_pct: Decimal) -> Self {
        Self {
            balance: initial_balance,
            position: None,
            commission_pct,
            slippage_pct,
            next_position_id: 1,
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
        }
    }

    fn execute_buy(
        &mut self,
        pair: &str,
        quantity: Decimal,
        ticker: &Ticker,
        ts: DateTime<Utc>,
    ) -> Option<TradeRecord> {
        // Apply slippage (buy at slightly higher price)
        let slippage = ticker.ask * self.slippage_pct;
        let execution_price = ticker.ask + slippage;
        let order_value = execution_price * quantity;
        let commission = order_value * self.commission_pct;
        let total_cost = order_value + commission;

        if total_cost > self.balance {
            debug!("Insufficient balance for buy: need {}, have {}", total_cost, self.balance);
            return None;
        }

        // Deduct from balance
        self.balance -= total_cost;

        // Create position
        let position_id = self.next_position_id;
        self.next_position_id += 1;

        let mut position = Position::new(position_id, pair, OrderSide::Buy, execution_price, quantity);
        position.opened_at = ts;
        self.position = Some(position);

        debug!(
            "BUY {} {} @ {} (cost: {}, commission: {})",
            quantity, pair, execution_price, order_value, commission
        );

        Some(TradeRecord {
            ts,
            pair: pair.to_string(),
            side: OrderSide::Buy,
            price: execution_price,
            quantity,
            value: order_value,
            commission,
            pnl: None,
        })
    }

    fn execute_sell(&mut self, ticker: &Ticker, ts: DateTime<Utc>) -> Option<TradeRecord> {
        let position = self.position.take()?;

        // Apply slippage (sell at slightly lower price)
        let slippage = ticker.bid * self.slippage_pct;
        let execution_price = ticker.bid - slippage;
        let order_value = execution_price * position.quantity;
        let commission = order_value * self.commission_pct;

        // Calculate P&L
        let gross_pnl = position.calculate_pnl(execution_price);
        let entry_commission = (position.entry_price * position.quantity) * self.commission_pct;
        let net_pnl = gross_pnl - commission - entry_commission;

        // Add to balance
        self.balance += order_value - commission;

        debug!(
            "SELL {} {} @ {} (value: {}, pnl: {})",
            position.quantity, position.pair, execution_price, order_value, net_pnl
        );

        Some(TradeRecord {
            ts,
            pair: position.pair.clone(),
            side: OrderSide::Sell,
            price: execution_price,
            quantity: position.quantity,
            value: order_value,
            commission,
            pnl: Some(net_pnl),
        })
    }
}

/// Main backtest engine
pub struct BacktestEngine {
    config: BacktestConfig,
    strategy_config: StrategyConfig,
    data_dir: String,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig, strategy_config: StrategyConfig, data_dir: &str) -> Self {
        Self {
            config,
            strategy_config,
            data_dir: data_dir.to_string(),
        }
    }

    /// Run the backtest
    pub fn run(&self) -> Result<BacktestResult> {
        info!(
            "Starting backtest for {} from {} to {}",
            self.config.pair, self.config.start_date, self.config.end_date
        );

        // Load data
        let loader = DataLoader::new(&self.data_dir);
        let snapshots = loader
            .load_range(&self.config.pair, self.config.start_date, self.config.end_date)
            .context("Failed to load historical data")?;

        info!("Loaded {} market snapshots", snapshots.len());

        if snapshots.is_empty() {
            return Ok(BacktestResult {
                config: self.config.clone(),
                trades: Vec::new(),
                metrics: PerformanceMetrics::default(),
                equity_curve: Vec::new(),
            });
        }

        // Initialize components
        let mut strategy = ScalperStrategy::new(self.strategy_config.clone());
        let mut trader = BacktestTrader::new(
            self.config.initial_balance,
            self.config.commission_pct,
            self.config.slippage_pct,
        );
        let mut metrics = MetricsCollector::new(self.config.initial_balance);
        let mut trades: Vec<TradeRecord> = Vec::new();

        // Calculate position size
        let max_position_value = self.config.initial_balance
            * Decimal::try_from(self.strategy_config.max_position_size).unwrap_or(Decimal::ONE);

        // Process each snapshot
        for snapshot in &snapshots {
            // We need a ticker for trading
            let ticker = match &snapshot.ticker {
                Some(t) => t,
                None => continue,
            };

            // Update position mark-to-market
            trader.update_position(ticker);

            // Record equity
            metrics.record_equity(snapshot.ts, trader.equity());

            // Get current position
            let position = trader.get_position();

            // Check for exit signals first
            if let Some(pos) = position {
                let should_exit = strategy.should_take_profit(pos, ticker.last)
                    || strategy.should_stop_loss(pos, ticker.last);
                let opened_at = pos.opened_at; // Copy before mutable borrow

                if should_exit {
                    if let Some(trade) = trader.execute_sell(ticker, snapshot.ts) {
                        if let Some(pnl) = trade.pnl {
                            let duration = snapshot.ts - opened_at;
                            metrics.record_trade(pnl, duration);
                        }
                        trades.push(trade);
                    }
                    continue;
                }
            }

            // Analyze for new signals
            let signal = strategy.analyze(ticker, snapshot.orderbook.as_ref(), trader.get_position());

            match signal {
                Signal::Buy { pair, quantity } => {
                    if !trader.has_position() {
                        // Calculate actual quantity based on max position size
                        let price = ticker.ask;
                        let max_qty = if !price.is_zero() {
                            max_position_value / price
                        } else {
                            quantity
                        };
                        let actual_qty = quantity.min(max_qty);

                        if let Some(trade) = trader.execute_buy(&pair, actual_qty, ticker, snapshot.ts) {
                            trades.push(trade);
                        }
                    }
                }
                Signal::Sell { pair: _ } => {
                    if let Some(pos) = trader.get_position() {
                        let opened_at = pos.opened_at;
                        if let Some(trade) = trader.execute_sell(ticker, snapshot.ts) {
                            if let Some(pnl) = trade.pnl {
                                let duration = snapshot.ts - opened_at;
                                metrics.record_trade(pnl, duration);
                            }
                            trades.push(trade);
                        }
                    }
                }
                Signal::Hold => {}
            }
        }

        // Close any remaining position at end
        if let Some(snapshot) = snapshots.last() {
            if let Some(ticker) = &snapshot.ticker {
                if let Some(pos) = trader.get_position() {
                    let opened_at = pos.opened_at;
                    if let Some(trade) = trader.execute_sell(ticker, snapshot.ts) {
                        if let Some(pnl) = trade.pnl {
                            let duration = snapshot.ts - opened_at;
                            metrics.record_trade(pnl, duration);
                        }
                        trades.push(trade);
                    }
                }
            }
        }

        // Calculate final metrics
        let performance = metrics.calculate();
        let equity_curve = metrics.equity_curve().to_vec();

        info!(
            "Backtest complete: {} trades, {:.2}% P&L",
            trades.len(),
            performance.total_pnl_pct
        );

        Ok(BacktestResult {
            config: self.config.clone(),
            trades,
            metrics: performance,
            equity_curve,
        })
    }
}
