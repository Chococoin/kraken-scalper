use crate::trading::OrderSide;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Row, Sqlite};
use std::str::FromStr;

pub struct Database {
    pool: Pool<Sqlite>,
}

#[derive(Debug, Clone)]
pub struct PositionRecord {
    pub id: i64,
    pub pair: String,
    pub side: OrderSide,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub quantity: Decimal,
    pub pnl: Option<Decimal>,
    pub status: String,
    pub opened_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub id: i64,
    pub position_id: Option<i64>,
    pub pair: String,
    pub side: OrderSide,
    pub price: Decimal,
    pub quantity: Decimal,
    pub fee: Option<Decimal>,
    pub is_paper: bool,
    pub executed_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct MetricsRecord {
    pub period: String,
    pub total_trades: i64,
    pub winning_trades: i64,
    pub losing_trades: i64,
    pub total_pnl: Decimal,
    pub balance: Decimal,
    pub recorded_at: DateTime<Utc>,
}

impl Database {
    pub async fn new(path: &str) -> Result<Self> {
        let url = format!("sqlite:{}?mode=rwc", path);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&url)
            .await
            .context("Failed to connect to SQLite database")?;

        // Enable WAL mode for better concurrent access from multiple processes
        sqlx::query("PRAGMA journal_mode=WAL")
            .execute(&pool)
            .await
            .context("Failed to enable WAL mode")?;

        // Set busy timeout to 5 seconds to handle lock contention
        sqlx::query("PRAGMA busy_timeout=5000")
            .execute(&pool)
            .await
            .context("Failed to set busy timeout")?;

        let db = Self { pool };
        db.run_migrations().await?;

        Ok(db)
    }

    async fn run_migrations(&self) -> Result<()> {
        let migration_sql = include_str!("../../migrations/001_initial.sql");

        sqlx::raw_sql(migration_sql)
            .execute(&self.pool)
            .await
            .context("Failed to run migrations")?;

        Ok(())
    }

    pub async fn insert_position(
        &self,
        pair: &str,
        side: OrderSide,
        entry_price: Decimal,
        quantity: Decimal,
    ) -> Result<i64> {
        let side_str = match side {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        };

        let result = sqlx::query(
            r#"
            INSERT INTO positions (pair, side, entry_price, quantity, status, opened_at)
            VALUES (?, ?, ?, ?, 'open', datetime('now'))
            "#,
        )
        .bind(pair)
        .bind(side_str)
        .bind(entry_price.to_string())
        .bind(quantity.to_string())
        .execute(&self.pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    pub async fn close_position(
        &self,
        position_id: i64,
        exit_price: Decimal,
        pnl: Decimal,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE positions
            SET exit_price = ?, pnl = ?, status = 'closed', closed_at = datetime('now')
            WHERE id = ?
            "#,
        )
        .bind(exit_price.to_string())
        .bind(pnl.to_string())
        .bind(position_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_open_positions(&self) -> Result<Vec<PositionRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, pair, side, entry_price, exit_price, quantity, pnl, status, opened_at, closed_at
            FROM positions
            WHERE status = 'open'
            ORDER BY opened_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut positions = Vec::new();
        for row in rows {
            positions.push(self.row_to_position(&row)?);
        }

        Ok(positions)
    }

    pub async fn get_recent_positions(&self, limit: i64) -> Result<Vec<PositionRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, pair, side, entry_price, exit_price, quantity, pnl, status, opened_at, closed_at
            FROM positions
            ORDER BY opened_at DESC
            LIMIT ?
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut positions = Vec::new();
        for row in rows {
            positions.push(self.row_to_position(&row)?);
        }

        Ok(positions)
    }

    fn row_to_position(&self, row: &sqlx::sqlite::SqliteRow) -> Result<PositionRecord> {
        let side_str: String = row.get("side");
        let side = match side_str.as_str() {
            "buy" => OrderSide::Buy,
            "sell" => OrderSide::Sell,
            _ => OrderSide::Buy,
        };

        let entry_price_str: String = row.get("entry_price");
        let exit_price_str: Option<String> = row.get("exit_price");
        let quantity_str: String = row.get("quantity");
        let pnl_str: Option<String> = row.get("pnl");
        let opened_at_str: String = row.get("opened_at");
        let closed_at_str: Option<String> = row.get("closed_at");

        Ok(PositionRecord {
            id: row.get("id"),
            pair: row.get("pair"),
            side,
            entry_price: Decimal::from_str(&entry_price_str)?,
            exit_price: exit_price_str.map(|s| Decimal::from_str(&s).ok()).flatten(),
            quantity: Decimal::from_str(&quantity_str)?,
            pnl: pnl_str.map(|s| Decimal::from_str(&s).ok()).flatten(),
            status: row.get("status"),
            opened_at: DateTime::parse_from_rfc3339(&format!("{}Z", opened_at_str))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            closed_at: closed_at_str
                .map(|s| {
                    DateTime::parse_from_rfc3339(&format!("{}Z", s))
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()
                })
                .flatten(),
        })
    }

    pub async fn insert_trade(
        &self,
        position_id: Option<i64>,
        pair: &str,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
        is_paper: bool,
    ) -> Result<i64> {
        let side_str = match side {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        };

        let result = sqlx::query(
            r#"
            INSERT INTO trades (position_id, pair, side, price, quantity, is_paper, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            "#,
        )
        .bind(position_id)
        .bind(pair)
        .bind(side_str)
        .bind(price.to_string())
        .bind(quantity.to_string())
        .bind(is_paper)
        .execute(&self.pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    pub async fn get_recent_trades(&self, limit: i64) -> Result<Vec<TradeRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, position_id, pair, side, price, quantity, fee, is_paper, executed_at
            FROM trades
            ORDER BY executed_at DESC
            LIMIT ?
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut trades = Vec::new();
        for row in rows {
            let side_str: String = row.get("side");
            let side = match side_str.as_str() {
                "buy" => OrderSide::Buy,
                "sell" => OrderSide::Sell,
                _ => OrderSide::Buy,
            };

            let price_str: String = row.get("price");
            let quantity_str: String = row.get("quantity");
            let fee_str: Option<String> = row.get("fee");
            let executed_at_str: String = row.get("executed_at");

            trades.push(TradeRecord {
                id: row.get("id"),
                position_id: row.get("position_id"),
                pair: row.get("pair"),
                side,
                price: Decimal::from_str(&price_str)?,
                quantity: Decimal::from_str(&quantity_str)?,
                fee: fee_str.map(|s| Decimal::from_str(&s).ok()).flatten(),
                is_paper: row.get("is_paper"),
                executed_at: DateTime::parse_from_rfc3339(&format!("{}Z", executed_at_str))
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            });
        }

        Ok(trades)
    }

    pub async fn save_metrics(&self, metrics: &MetricsRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO metrics (period, total_trades, winning_trades, losing_trades, total_pnl, balance, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&metrics.period)
        .bind(metrics.total_trades)
        .bind(metrics.winning_trades)
        .bind(metrics.losing_trades)
        .bind(metrics.total_pnl.to_string())
        .bind(metrics.balance.to_string())
        .bind(metrics.recorded_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_total_pnl(&self) -> Result<Decimal> {
        let row = sqlx::query(
            r#"
            SELECT COALESCE(SUM(CAST(pnl AS REAL)), 0) as total
            FROM positions
            WHERE status = 'closed'
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        let total: f64 = row.get("total");
        Ok(Decimal::from_str(&total.to_string()).unwrap_or(Decimal::ZERO))
    }

    pub async fn get_trade_stats(&self) -> Result<(i64, i64, i64)> {
        let row = sqlx::query(
            r#"
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN CAST(pnl AS REAL) > 0 THEN 1 ELSE 0 END) as winning,
                SUM(CASE WHEN CAST(pnl AS REAL) < 0 THEN 1 ELSE 0 END) as losing
            FROM positions
            WHERE status = 'closed'
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok((
            row.get::<i64, _>("total"),
            row.get::<i64, _>("winning"),
            row.get::<i64, _>("losing"),
        ))
    }
}
