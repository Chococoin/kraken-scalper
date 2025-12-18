//! MongoDB storage for AI trading signals and executions
//!
//! Stores ML model predictions and paper trade executions for analysis.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use mongodb::{
    bson::{doc, oid::ObjectId, Document},
    options::ClientOptions,
    Client, Collection, Database,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tracing::{error, info};

/// AI trading signal from ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSignal {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<ObjectId>,
    pub timestamp: DateTime<Utc>,
    pub pair: String,
    pub signal: SignalType,
    pub confidence: f64,
    pub model_version: String,
    /// Technical indicators at time of signal
    pub indicators: SignalIndicators,
    /// Whether this signal was acted upon
    pub executed: bool,
    /// Execution details if acted upon
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<SignalExecution>,
}

/// Type of trading signal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::Hold => write!(f, "HOLD"),
        }
    }
}

/// Technical indicators captured with the signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalIndicators {
    pub price: f64,
    pub rsi: Option<f64>,
    pub macd: Option<f64>,
    pub macd_signal: Option<f64>,
    pub macd_hist: Option<f64>,
    pub ema_20: Option<f64>,
    pub ema_50: Option<f64>,
    pub bb_upper: Option<f64>,
    pub bb_lower: Option<f64>,
    pub volume: Option<f64>,
    pub bid_ask_spread: Option<f64>,
    pub order_imbalance: Option<f64>,
}

impl Default for SignalIndicators {
    fn default() -> Self {
        Self {
            price: 0.0,
            rsi: None,
            macd: None,
            macd_signal: None,
            macd_hist: None,
            ema_20: None,
            ema_50: None,
            bb_upper: None,
            bb_lower: None,
            volume: None,
            bid_ask_spread: None,
            order_imbalance: None,
        }
    }
}

/// Execution details for an AI signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalExecution {
    pub executed_at: DateTime<Utc>,
    pub side: String,
    pub price: f64,
    pub quantity: f64,
    pub cost: f64,
    pub paper_trade: bool,
    /// Outcome tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_price: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_time: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pnl: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pnl_pct: Option<f64>,
}

/// Paper trade record for tracking AI performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiPaperTrade {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<ObjectId>,
    pub signal_id: ObjectId,
    pub timestamp: DateTime<Utc>,
    pub pair: String,
    pub side: String,
    pub entry_price: f64,
    pub quantity: f64,
    pub cost: f64,
    pub status: TradeStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_price: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_time: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pnl: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pnl_pct: Option<f64>,
    /// Model confidence at entry
    pub entry_confidence: f64,
    /// Technical indicators at entry
    pub entry_indicators: SignalIndicators,
}

/// Status of a paper trade
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TradeStatus {
    Open,
    Closed,
    StopLoss,
    TakeProfit,
}

/// MongoDB client for AI trading data
pub struct MongoStore {
    client: Client,
    db: Database,
}

impl MongoStore {
    /// Create a new MongoDB connection
    pub async fn new(uri: &str, db_name: &str) -> Result<Self> {
        let client_options = ClientOptions::parse(uri)
            .await
            .context("Failed to parse MongoDB URI")?;

        let client = Client::with_options(client_options)
            .context("Failed to create MongoDB client")?;

        // Verify connection
        client
            .database("admin")
            .run_command(doc! { "ping": 1 })
            .await
            .context("Failed to connect to MongoDB")?;

        let db = client.database(db_name);
        info!("Connected to MongoDB database: {}", db_name);

        Ok(Self { client, db })
    }

    /// Create with default local connection
    pub async fn new_local() -> Result<Self> {
        Self::new("mongodb://localhost:27017", "scalper_ai").await
    }

    /// Get the signals collection
    fn signals(&self) -> Collection<AiSignal> {
        self.db.collection("signals")
    }

    /// Get the paper trades collection
    fn paper_trades(&self) -> Collection<AiPaperTrade> {
        self.db.collection("paper_trades")
    }

    /// Store a new AI signal
    pub async fn store_signal(&self, signal: &AiSignal) -> Result<ObjectId> {
        let result = self
            .signals()
            .insert_one(signal)
            .await
            .context("Failed to insert AI signal")?;

        let id = result
            .inserted_id
            .as_object_id()
            .context("Failed to get inserted signal ID")?;

        info!(
            "Stored AI signal: {} {} @ {} (confidence: {:.1}%)",
            signal.signal,
            signal.pair,
            signal.indicators.price,
            signal.confidence * 100.0
        );

        Ok(id)
    }

    /// Store a paper trade execution
    pub async fn store_paper_trade(&self, trade: &AiPaperTrade) -> Result<ObjectId> {
        let result = self
            .paper_trades()
            .insert_one(trade)
            .await
            .context("Failed to insert paper trade")?;

        let id = result
            .inserted_id
            .as_object_id()
            .context("Failed to get inserted trade ID")?;

        info!(
            "Stored paper trade: {} {} {} @ {}",
            trade.side, trade.quantity, trade.pair, trade.entry_price
        );

        Ok(id)
    }

    /// Mark a signal as executed
    pub async fn mark_signal_executed(
        &self,
        signal_id: ObjectId,
        execution: SignalExecution,
    ) -> Result<()> {
        self.signals()
            .update_one(
                doc! { "_id": signal_id },
                doc! {
                    "$set": {
                        "executed": true,
                        "execution": mongodb::bson::to_bson(&execution)?
                    }
                },
            )
            .await
            .context("Failed to update signal execution")?;

        Ok(())
    }

    /// Close a paper trade
    pub async fn close_paper_trade(
        &self,
        trade_id: ObjectId,
        exit_price: f64,
        status: TradeStatus,
    ) -> Result<()> {
        // Get the trade to calculate PnL
        let trade = self
            .paper_trades()
            .find_one(doc! { "_id": trade_id })
            .await?
            .context("Trade not found")?;

        let pnl = if trade.side == "buy" {
            (exit_price - trade.entry_price) * trade.quantity
        } else {
            (trade.entry_price - exit_price) * trade.quantity
        };
        let pnl_pct = pnl / trade.cost * 100.0;

        self.paper_trades()
            .update_one(
                doc! { "_id": trade_id },
                doc! {
                    "$set": {
                        "status": mongodb::bson::to_bson(&status)?,
                        "exit_price": exit_price,
                        "exit_time": mongodb::bson::DateTime::now(),
                        "pnl": pnl,
                        "pnl_pct": pnl_pct
                    }
                },
            )
            .await
            .context("Failed to close paper trade")?;

        info!(
            "Closed paper trade {}: PnL ${:.2} ({:.2}%)",
            trade.pair, pnl, pnl_pct
        );

        Ok(())
    }

    /// Get recent signals for a pair
    pub async fn get_recent_signals(&self, pair: &str, limit: i64) -> Result<Vec<AiSignal>> {
        use futures_util::TryStreamExt;

        let signals: Vec<AiSignal> = self
            .signals()
            .find(doc! { "pair": pair })
            .sort(doc! { "timestamp": -1 })
            .limit(limit)
            .await?
            .try_collect()
            .await?;

        Ok(signals)
    }

    /// Get open paper trades
    pub async fn get_open_trades(&self) -> Result<Vec<AiPaperTrade>> {
        use futures_util::TryStreamExt;

        let trades: Vec<AiPaperTrade> = self
            .paper_trades()
            .find(doc! { "status": "open" })
            .await?
            .try_collect()
            .await?;

        Ok(trades)
    }

    /// Get trade statistics
    pub async fn get_trade_stats(&self) -> Result<TradeStats> {
        use futures_util::TryStreamExt;

        let closed_trades: Vec<AiPaperTrade> = self
            .paper_trades()
            .find(doc! { "status": { "$ne": "open" } })
            .await?
            .try_collect()
            .await?;

        let total_trades = closed_trades.len();
        let winning_trades = closed_trades
            .iter()
            .filter(|t| t.pnl.unwrap_or(0.0) > 0.0)
            .count();
        let total_pnl: f64 = closed_trades.iter().filter_map(|t| t.pnl).sum();
        let avg_pnl = if total_trades > 0 {
            total_pnl / total_trades as f64
        } else {
            0.0
        };
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };

        Ok(TradeStats {
            total_trades,
            winning_trades,
            losing_trades: total_trades - winning_trades,
            win_rate,
            total_pnl,
            avg_pnl,
        })
    }

    /// Get the latest signal for a pair
    pub async fn get_latest_signal(&self, pair: &str) -> Result<Option<AiSignal>> {
        let signal = self
            .signals()
            .find_one(doc! { "pair": pair })
            .sort(doc! { "timestamp": -1 })
            .await?;

        Ok(signal)
    }
}

/// Trading statistics
#[derive(Debug, Clone, Default)]
pub struct TradeStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_pnl: f64,
}
