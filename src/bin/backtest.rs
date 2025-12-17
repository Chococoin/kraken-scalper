//! Backtest CLI
//!
//! Run backtests on historical data.
//! Usage:
//!   cargo run --bin backtest -- run --pair BTC/USD --start 2025-12-17
//!   cargo run --bin backtest -- compare --pair BTC/USD --start 2025-12-17 --strategies all

use anyhow::Result;
use chrono::{NaiveDate, TimeZone, Utc};
use clap::{Parser, Subcommand};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use scalper::backtest::{compare_strategies, BacktestConfig, BacktestEngine, ComparisonResult, PerformanceMetrics};
use scalper::config::Config;
use scalper::strategy::{parse_strategy_list, STRATEGY_NAMES};

#[derive(Parser, Debug)]
#[command(name = "backtest")]
#[command(about = "Run backtests on historical market data")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single strategy backtest
    Run {
        /// Trading pair to backtest (e.g., BTC/USD)
        #[arg(short, long, default_value = "BTC/USD")]
        pair: String,

        /// Start date (YYYY-MM-DD)
        #[arg(short, long)]
        start: String,

        /// End date (YYYY-MM-DD), defaults to today
        #[arg(short, long)]
        end: Option<String>,

        /// Initial balance in USD
        #[arg(short, long, default_value = "10000")]
        balance: f64,

        /// Commission percentage (default: 0.26 for Kraken taker)
        #[arg(short, long, default_value = "0.26")]
        commission: f64,

        /// Show individual trades
        #[arg(long)]
        trades: bool,

        /// Show equity curve
        #[arg(long)]
        equity: bool,
    },

    /// Compare multiple strategies
    Compare {
        /// Trading pair to backtest (e.g., BTC/USD)
        #[arg(short, long, default_value = "BTC/USD")]
        pair: String,

        /// Start date (YYYY-MM-DD)
        #[arg(short, long)]
        start: String,

        /// End date (YYYY-MM-DD), defaults to today
        #[arg(short, long)]
        end: Option<String>,

        /// Initial balance in USD
        #[arg(short, long, default_value = "10000")]
        balance: f64,

        /// Strategies to compare (comma-separated, or "all")
        #[arg(short = 'S', long, default_value = "all")]
        strategies: String,

        /// Export results to CSV file
        #[arg(long)]
        export: Option<String>,
    },

    /// List available strategies
    List,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("scalper=info".parse().unwrap())
                .add_directive("backtest=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            pair,
            start,
            end,
            balance,
            commission,
            trades,
            equity,
        } => run_single_backtest(&pair, &start, end.as_deref(), balance, commission, trades, equity),

        Commands::Compare {
            pair,
            start,
            end,
            balance,
            strategies,
            export,
        } => run_comparison(&pair, &start, end.as_deref(), balance, &strategies, export.as_deref()),

        Commands::List => {
            println!("\nAvailable strategies:");
            for name in STRATEGY_NAMES {
                println!("  - {}", name);
            }
            println!("\nUsage: backtest compare --strategies scalper,meanrev,trend");
            println!("       backtest compare --strategies all");
            Ok(())
        }
    }
}

fn parse_dates(start: &str, end: Option<&str>) -> Result<(NaiveDate, NaiveDate)> {
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .map_err(|e| anyhow::anyhow!("Invalid start date: {}", e))?;

    let end_date = if let Some(end_str) = end {
        NaiveDate::parse_from_str(end_str, "%Y-%m-%d")
            .map_err(|e| anyhow::anyhow!("Invalid end date: {}", e))?
    } else {
        Utc::now().date_naive()
    };

    Ok((start_date, end_date))
}

fn run_single_backtest(
    pair: &str,
    start: &str,
    end: Option<&str>,
    balance: f64,
    commission: f64,
    show_trades: bool,
    show_equity: bool,
) -> Result<()> {
    let (start_date, end_date) = parse_dates(start, end)?;
    let start_dt = Utc.from_utc_datetime(&start_date.and_hms_opt(0, 0, 0).unwrap());
    let end_dt = Utc.from_utc_datetime(&end_date.and_hms_opt(23, 59, 59).unwrap());

    let config = Config::load()?;

    let backtest_config = BacktestConfig::new(pair, start_dt, end_dt)
        .with_balance(Decimal::try_from(balance).unwrap_or(Decimal::from(10000)))
        .with_commission(commission);

    let engine = BacktestEngine::new(backtest_config, config.strategy.clone(), &config.recording.data_dir);
    let result = engine.run()?;

    print_header(pair);
    print_period(start_date, end_date);
    print_performance(&result.metrics);
    print_pnl(&result.metrics);
    print_risk(&result.metrics);
    print_equity_summary(&result.metrics);
    print_footer();

    if show_trades && !result.trades.is_empty() {
        println!();
        print_trades(&result.trades);
    }

    if show_equity && !result.equity_curve.is_empty() {
        println!();
        print_equity_curve(&result.equity_curve);
    }

    Ok(())
}

fn run_comparison(
    pair: &str,
    start: &str,
    end: Option<&str>,
    balance: f64,
    strategies: &str,
    export: Option<&str>,
) -> Result<()> {
    let (start_date, end_date) = parse_dates(start, end)?;
    let start_dt = Utc.from_utc_datetime(&start_date.and_hms_opt(0, 0, 0).unwrap());
    let end_dt = Utc.from_utc_datetime(&end_date.and_hms_opt(23, 59, 59).unwrap());

    let config = Config::load()?;
    let strategy_list = parse_strategy_list(strategies);

    let backtest_config = BacktestConfig::new(pair, start_dt, end_dt)
        .with_balance(Decimal::try_from(balance).unwrap_or(Decimal::from(10000)));

    let comparison = compare_strategies(
        &strategy_list,
        backtest_config,
        &config.strategy,
        &config.recording.data_dir,
    )?;

    print_comparison_results(&comparison, start_date, end_date);

    if let Some(path) = export {
        comparison.export_csv(path)?;
        println!("\nExported to: {}", path);
    }

    Ok(())
}

fn print_comparison_results(result: &ComparisonResult, start: NaiveDate, end: NaiveDate) {
    println!();
    println!("\x1b[1;36m{}\x1b[0m", "═".repeat(78));
    println!(
        "\x1b[1;36m                    STRATEGY COMPARISON: {}\x1b[0m",
        result.config.pair
    );
    println!("\x1b[1;36m{}\x1b[0m", "═".repeat(78));
    println!(
        "Period: {} → {} | Snapshots: {}",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        result.snapshots_count
    );
    println!();

    // Table header
    println!(
        "┌{:─<17}┬{:─>8}┬{:─>10}┬{:─>9}┬{:─>11}┬{:─>11}┐",
        "", "", "", "", "", ""
    );
    println!(
        "│ {:15} │ {:>6} │ {:>8} │ {:>7} │ {:>9} │ {:>9} │",
        "Strategy", "Trades", "Win Rate", "P&L %", "Drawdown", "Sharpe"
    );
    println!(
        "├{:─<17}┼{:─>8}┼{:─>10}┼{:─>9}┼{:─>11}┼{:─>11}┤",
        "", "", "", "", "", ""
    );

    for (name, backtest_result) in &result.results {
        let m = &backtest_result.metrics;

        let pnl_color = if m.total_pnl_pct >= 0.0 {
            "\x1b[32m"
        } else {
            "\x1b[31m"
        };

        let sharpe_str = if m.sharpe_ratio.is_nan() || m.sharpe_ratio.is_infinite() {
            "N/A".to_string()
        } else {
            format!("{:.2}", m.sharpe_ratio)
        };

        println!(
            "│ {:15} │ {:>6} │ {:>7.1}% │ {}{:>+6.2}%\x1b[0m │ {:>8.1}% │ {:>9} │",
            name,
            m.total_trades,
            m.win_rate,
            pnl_color,
            m.total_pnl_pct,
            -m.max_drawdown,
            sharpe_str
        );
    }

    println!(
        "└{:─<17}┴{:─>8}┴{:─>10}┴{:─>9}┴{:─>11}┴{:─>11}┘",
        "", "", "", "", "", ""
    );

    // Best strategy
    if let Some((name, best)) = result.best_by_pnl() {
        println!();
        let m = &best.metrics;
        let sharpe_str = if m.sharpe_ratio.is_nan() || m.sharpe_ratio.is_infinite() {
            "N/A".to_string()
        } else {
            format!("{:.2}", m.sharpe_ratio)
        };
        println!(
            "\x1b[1;32mBest Strategy: {} ({:+.2}% P&L, {} Sharpe)\x1b[0m",
            name, m.total_pnl_pct, sharpe_str
        );
    }

    println!("\x1b[1;36m{}\x1b[0m", "═".repeat(78));
}

fn print_header(pair: &str) {
    println!();
    println!("\x1b[1;36m{}\x1b[0m", "═".repeat(55));
    println!("\x1b[1;36m         BACKTEST RESULTS: {}\x1b[0m", pair);
    println!("\x1b[1;36m{}\x1b[0m", "═".repeat(55));
}

fn print_period(start: NaiveDate, end: NaiveDate) {
    println!(
        "Period: {} → {}",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d")
    );
    println!();
}

fn print_performance(metrics: &PerformanceMetrics) {
    println!("\x1b[1;33mPERFORMANCE\x1b[0m");
    println!("  Total Trades:     {}", metrics.total_trades);
    println!(
        "  Win Rate:         {:.1}%  ({} wins / {} losses)",
        metrics.win_rate, metrics.winning_trades, metrics.losing_trades
    );

    let pf_str = if metrics.profit_factor.is_infinite() {
        "∞".to_string()
    } else {
        format!("{:.2}", metrics.profit_factor)
    };
    println!("  Profit Factor:    {}", pf_str);
    println!();
}

fn print_pnl(metrics: &PerformanceMetrics) {
    println!("\x1b[1;33mP&L\x1b[0m");

    let pnl_color = if metrics.total_pnl >= Decimal::ZERO {
        "\x1b[32m"
    } else {
        "\x1b[31m"
    };

    println!(
        "  Total P&L:        {}${:.2} ({:+.2}%)\x1b[0m",
        pnl_color,
        metrics.total_pnl.to_f64().unwrap_or(0.0),
        metrics.total_pnl_pct
    );
    println!(
        "  Avg Win:          ${:.2}",
        metrics.avg_win.to_f64().unwrap_or(0.0)
    );
    println!(
        "  Avg Loss:         ${:.2}",
        metrics.avg_loss.to_f64().unwrap_or(0.0)
    );
    println!(
        "  Largest Win:      ${:.2}",
        metrics.largest_win.to_f64().unwrap_or(0.0)
    );
    println!(
        "  Largest Loss:     ${:.2}",
        metrics.largest_loss.to_f64().unwrap_or(0.0)
    );
    println!(
        "  Expectancy:       ${:.2}",
        metrics.expectancy.to_f64().unwrap_or(0.0)
    );
    println!();
}

fn print_risk(metrics: &PerformanceMetrics) {
    println!("\x1b[1;33mRISK\x1b[0m");
    println!("  Max Drawdown:     {:.2}%", metrics.max_drawdown);

    let dd_duration = format_duration(metrics.max_drawdown_duration);
    println!("  DD Duration:      {}", dd_duration);

    let sharpe_str = if metrics.sharpe_ratio.is_infinite() || metrics.sharpe_ratio.is_nan() {
        "N/A".to_string()
    } else {
        format!("{:.2}", metrics.sharpe_ratio)
    };
    println!("  Sharpe Ratio:     {}", sharpe_str);

    let sortino_str = if metrics.sortino_ratio.is_infinite() || metrics.sortino_ratio.is_nan() {
        "N/A".to_string()
    } else {
        format!("{:.2}", metrics.sortino_ratio)
    };
    println!("  Sortino Ratio:    {}", sortino_str);
    println!();
}

fn print_equity_summary(metrics: &PerformanceMetrics) {
    println!("\x1b[1;33mEQUITY\x1b[0m");
    println!(
        "  Start:            ${:.2}",
        metrics.start_balance.to_f64().unwrap_or(0.0)
    );
    println!(
        "  End:              ${:.2}",
        metrics.end_balance.to_f64().unwrap_or(0.0)
    );

    let duration = format_duration(metrics.total_duration);
    println!("  Duration:         {}", duration);

    let avg_trade = format_duration(metrics.avg_trade_duration);
    println!("  Avg Trade:        {}", avg_trade);
}

fn print_footer() {
    println!("\x1b[1;36m{}\x1b[0m", "═".repeat(55));
}

fn print_trades(trades: &[scalper::backtest::TradeRecord]) {
    println!("\x1b[1;33mTRADES\x1b[0m");
    println!(
        "{:<20} {:>6} {:>12} {:>10} {:>12}",
        "Timestamp", "Side", "Price", "Qty", "P&L"
    );
    println!("{}", "-".repeat(65));

    for trade in trades {
        let pnl_str = trade
            .pnl
            .map(|p| format!("{:+.2}", p.to_f64().unwrap_or(0.0)))
            .unwrap_or_else(|| "-".to_string());

        let side_color = match trade.side {
            scalper::trading::order::OrderSide::Buy => "\x1b[32m",
            scalper::trading::order::OrderSide::Sell => "\x1b[31m",
        };

        println!(
            "{:<20} {}{:>6}\x1b[0m {:>12.2} {:>10.6} {:>12}",
            trade.ts.format("%Y-%m-%d %H:%M:%S"),
            side_color,
            format!("{}", trade.side),
            trade.price.to_f64().unwrap_or(0.0),
            trade.quantity.to_f64().unwrap_or(0.0),
            pnl_str
        );
    }
}

fn print_equity_curve(curve: &[(chrono::DateTime<Utc>, Decimal)]) {
    println!("\x1b[1;33mEQUITY CURVE (sampled)\x1b[0m");

    let sample_rate = (curve.len() / 20).max(1);

    for (i, (ts, equity)) in curve.iter().enumerate() {
        if i % sample_rate == 0 || i == curve.len() - 1 {
            println!(
                "  {} ${:.2}",
                ts.format("%Y-%m-%d %H:%M"),
                equity.to_f64().unwrap_or(0.0)
            );
        }
    }
}

fn format_duration(duration: chrono::Duration) -> String {
    let total_secs = duration.num_seconds();

    if total_secs < 60 {
        format!("{}s", total_secs)
    } else if total_secs < 3600 {
        format!("{}m {}s", total_secs / 60, total_secs % 60)
    } else if total_secs < 86400 {
        format!("{}h {}m", total_secs / 3600, (total_secs % 3600) / 60)
    } else {
        format!("{}d {}h", total_secs / 86400, (total_secs % 86400) / 3600)
    }
}
