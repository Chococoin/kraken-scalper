//! Live trader implementation using Kraken REST API

use crate::api::KrakenRestClient;
use crate::config::Config;
use crate::data::Ticker;
use crate::storage::Database;
use crate::trading::order::{Order, OrderSide, OrderStatus, OrderType, Position};
use anyhow::{anyhow, Result};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};

/// Live trader that executes real orders on Kraken
pub struct LiveTrader {
    client: KrakenRestClient,
    balances: HashMap<String, Decimal>,
    positions: HashMap<String, Position>,
    open_orders: HashMap<String, Order>,
    db: Arc<Database>,
    total_pnl: Decimal,
    initial_equity: Decimal,
}

impl LiveTrader {
    /// Create a new live trader
    pub async fn new(config: &Config, db: Arc<Database>) -> Result<Self> {
        if config.kraken.api_key.is_empty() || config.kraken.api_secret.is_empty() {
            return Err(anyhow!(
                "API credentials required for live trading. Set api_key and api_secret in config."
            ));
        }

        let client = KrakenRestClient::new(
            &config.kraken.api_key,
            &config.kraken.api_secret,
            &config.kraken.rest_url,
        )?;

        let mut trader = Self {
            client,
            balances: HashMap::new(),
            positions: HashMap::new(),
            open_orders: HashMap::new(),
            db,
            total_pnl: Decimal::ZERO,
            initial_equity: Decimal::ZERO,
        };

        // Initial sync
        trader.sync_balance().await?;
        trader.initial_equity = trader.total_usd_balance();

        info!(
            "Live trader initialized with balance: ${:.2}",
            trader.initial_equity
        );

        Ok(trader)
    }

    /// Sync balance from exchange
    pub async fn sync_balance(&mut self) -> Result<()> {
        match self.client.get_balance().await {
            Ok(balances) => {
                self.balances = balances;
                info!("Synced {} asset balances", self.balances.len());
                Ok(())
            }
            Err(e) => {
                error!("Failed to sync balance: {}", e);
                Err(e)
            }
        }
    }

    /// Sync open positions from exchange
    pub async fn sync_positions(&mut self) -> Result<()> {
        match self.client.get_open_positions().await {
            Ok(positions) => {
                self.positions.clear();
                for (id, pos_info) in positions {
                    let side = if pos_info.position_type == "buy" {
                        OrderSide::Buy
                    } else {
                        OrderSide::Sell
                    };

                    let entry_price = pos_info.cost.parse().unwrap_or(Decimal::ZERO);
                    let quantity = pos_info.vol.parse().unwrap_or(Decimal::ZERO);

                    // Calculate entry price from cost
                    let entry = if quantity > Decimal::ZERO {
                        entry_price / quantity
                    } else {
                        Decimal::ZERO
                    };

                    let position = Position::new(
                        id.parse().unwrap_or(0),
                        &pos_info.pair,
                        side,
                        entry,
                        quantity,
                    );

                    self.positions.insert(pos_info.pair.clone(), position);
                }
                info!("Synced {} open positions", self.positions.len());
                Ok(())
            }
            Err(e) => {
                error!("Failed to sync positions: {}", e);
                Err(e)
            }
        }
    }

    /// Sync open orders from exchange
    pub async fn sync_orders(&mut self) -> Result<()> {
        match self.client.get_open_orders().await {
            Ok(orders_result) => {
                self.open_orders.clear();
                for (txid, order_info) in orders_result.open {
                    let side = if order_info.descr.order_type == "buy" {
                        OrderSide::Buy
                    } else {
                        OrderSide::Sell
                    };

                    let order_type = match order_info.descr.ordertype.as_str() {
                        "limit" => OrderType::Limit,
                        "stop-loss" => OrderType::StopLoss,
                        "take-profit" => OrderType::TakeProfit,
                        _ => OrderType::Market,
                    };

                    let order = Order {
                        id: txid.clone(),
                        pair: order_info.descr.pair.clone(),
                        side,
                        order_type,
                        quantity: order_info.vol.parse().unwrap_or(Decimal::ZERO),
                        price: order_info.price.parse().ok(),
                        filled_quantity: order_info.vol_exec.parse().unwrap_or(Decimal::ZERO),
                        filled_price: None,
                        status: OrderStatus::Open,
                        created_at: chrono::Utc::now(),
                        updated_at: chrono::Utc::now(),
                    };

                    self.open_orders.insert(txid, order);
                }
                info!("Synced {} open orders", self.open_orders.len());
                Ok(())
            }
            Err(e) => {
                error!("Failed to sync orders: {}", e);
                Err(e)
            }
        }
    }

    /// Get balance for a specific asset
    pub fn balance(&self, asset: &str) -> Decimal {
        self.balances.get(asset).copied().unwrap_or(Decimal::ZERO)
    }

    /// Get total balance in USD
    pub fn total_usd_balance(&self) -> Decimal {
        // USD-like assets
        let usd = self.balance("ZUSD")
            + self.balance("USD")
            + self.balance("USDT")
            + self.balance("USDC");

        // For crypto, we'd need current prices to convert
        // For now, just return USD balance
        usd
    }

    /// Get balance (USD)
    pub fn balance_usd(&self) -> Decimal {
        self.total_usd_balance()
    }

    /// Get equity (balance + unrealized P&L)
    pub fn equity(&self) -> Decimal {
        let unrealized_pnl: Decimal = self
            .positions
            .values()
            .map(|p| p.unrealized_pnl)
            .sum();

        self.total_usd_balance() + unrealized_pnl
    }

    /// Get initial balance
    pub fn initial_balance(&self) -> Decimal {
        self.initial_equity
    }

    /// Get total realized P&L
    pub fn total_pnl(&self) -> Decimal {
        self.total_pnl
    }

    /// Get total P&L percentage
    pub fn total_pnl_pct(&self) -> Decimal {
        if self.initial_equity.is_zero() {
            Decimal::ZERO
        } else {
            (self.total_pnl / self.initial_equity) * Decimal::from(100)
        }
    }

    /// Get all positions
    pub fn positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Check if there's a position for a pair
    pub fn has_position(&self, pair: &str) -> bool {
        self.positions.contains_key(pair)
    }

    /// Get position for a pair
    pub fn get_position(&self, pair: &str) -> Option<&Position> {
        self.positions.get(pair)
    }

    /// Execute an order on the exchange
    pub async fn execute_order(&mut self, order: Order, _ticker: &Ticker) -> Result<Order> {
        info!(
            "Executing LIVE order: {} {} {} @ {:?}",
            order.side, order.quantity, order.pair, order.price
        );

        // Place order on exchange
        let result = self
            .client
            .add_order(
                &order.pair,
                order.side.clone(),
                order.order_type.clone(),
                order.quantity,
                order.price,
            )
            .await?;

        let txid = result
            .txid
            .and_then(|ids| ids.first().cloned())
            .ok_or_else(|| anyhow!("No transaction ID returned from exchange"))?;

        info!("Order placed successfully: {}", txid);

        // Create executed order
        let mut executed_order = order;
        executed_order.id = txid.clone();
        executed_order.status = OrderStatus::Open;

        // Record trade in database
        if let Err(e) = self
            .db
            .insert_trade(
                None,
                &executed_order.pair,
                executed_order.side.clone(),
                executed_order.price.unwrap_or(Decimal::ZERO),
                executed_order.quantity,
                false, // is_paper = false
            )
            .await
        {
            warn!("Failed to record trade in database: {}", e);
        }

        // Sync state after order
        let _ = self.sync_balance().await;
        let _ = self.sync_orders().await;

        Ok(executed_order)
    }

    /// Cancel an order
    pub async fn cancel_order(&mut self, txid: &str) -> Result<()> {
        info!("Cancelling order: {}", txid);

        self.client.cancel_order(txid).await?;
        self.open_orders.remove(txid);

        info!("Order cancelled: {}", txid);
        Ok(())
    }

    /// Update positions with current prices
    pub fn update_positions(&mut self, ticker: &Ticker) {
        if let Some(position) = self.positions.get_mut(&ticker.pair) {
            position.update_price(ticker.last);
        }
    }
}
