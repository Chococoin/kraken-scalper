use crate::data::Ticker;
use crate::storage::Database;
use crate::trading::order::{Order, OrderSide, OrderStatus, Position};
use anyhow::Result;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

pub struct PaperTrader {
    balance: Decimal,
    initial_balance: Decimal,
    positions: HashMap<String, Position>,
    db: Arc<Database>,
    next_position_id: i64,
}

impl PaperTrader {
    pub async fn new(initial_balance: f64, db: Arc<Database>) -> Result<Self> {
        let balance = Decimal::from_str_exact(&initial_balance.to_string())?;

        // Load open positions from DB
        let open_positions = db.get_open_positions().await?;
        let mut positions = HashMap::new();
        let mut max_id = 0i64;

        for pos in open_positions {
            max_id = max_id.max(pos.id);
            positions.insert(
                pos.pair.clone(),
                Position::new(pos.id, &pos.pair, pos.side, pos.entry_price, pos.quantity),
            );
        }

        Ok(Self {
            balance,
            initial_balance: balance,
            positions,
            db,
            next_position_id: max_id + 1,
        })
    }

    pub fn balance(&self) -> Decimal {
        self.balance
    }

    pub fn initial_balance(&self) -> Decimal {
        self.initial_balance
    }

    pub fn total_pnl(&self) -> Decimal {
        self.balance - self.initial_balance
    }

    pub fn total_pnl_pct(&self) -> Decimal {
        if self.initial_balance.is_zero() {
            return Decimal::ZERO;
        }
        (self.total_pnl() / self.initial_balance) * Decimal::from(100)
    }

    pub fn positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    pub fn has_position(&self, pair: &str) -> bool {
        self.positions.contains_key(pair)
    }

    pub fn get_position(&self, pair: &str) -> Option<&Position> {
        self.positions.get(pair)
    }

    pub async fn execute_order(&mut self, mut order: Order, ticker: &Ticker) -> Result<Order> {
        let execution_price = match order.side {
            OrderSide::Buy => ticker.ask,
            OrderSide::Sell => ticker.bid,
        };

        let order_value = execution_price * order.quantity;

        match order.side {
            OrderSide::Buy => {
                if order_value > self.balance {
                    order.status = OrderStatus::Rejected;
                    return Ok(order);
                }

                // Deduct from balance
                self.balance -= order_value;

                // Create position
                let position_id = self.next_position_id;
                self.next_position_id += 1;

                let position = Position::new(
                    position_id,
                    &order.pair,
                    OrderSide::Buy,
                    execution_price,
                    order.quantity,
                );

                // Save to DB
                self.db
                    .insert_position(&order.pair, OrderSide::Buy, execution_price, order.quantity)
                    .await?;

                self.db
                    .insert_trade(
                        Some(position_id),
                        &order.pair,
                        OrderSide::Buy,
                        execution_price,
                        order.quantity,
                        true,
                    )
                    .await?;

                self.positions.insert(order.pair.clone(), position);

                info!(
                    "Paper BUY {} {} @ {} (value: {})",
                    order.quantity, order.pair, execution_price, order_value
                );
            }
            OrderSide::Sell => {
                // Close existing position
                if let Some(position) = self.positions.remove(&order.pair) {
                    let pnl = position.calculate_pnl(execution_price);
                    let exit_value = execution_price * position.quantity;

                    // Add to balance
                    self.balance += exit_value;

                    // Update DB
                    self.db
                        .close_position(position.id, execution_price, pnl)
                        .await?;

                    self.db
                        .insert_trade(
                            Some(position.id),
                            &order.pair,
                            OrderSide::Sell,
                            execution_price,
                            position.quantity,
                            true,
                        )
                        .await?;

                    info!(
                        "Paper SELL {} {} @ {} (PnL: {})",
                        position.quantity, order.pair, execution_price, pnl
                    );
                } else {
                    order.status = OrderStatus::Rejected;
                    return Ok(order);
                }
            }
        }

        order.status = OrderStatus::Filled;
        order.filled_quantity = order.quantity;
        order.filled_price = Some(execution_price);
        order.updated_at = chrono::Utc::now();

        Ok(order)
    }

    pub fn update_positions(&mut self, ticker: &Ticker) {
        if let Some(position) = self.positions.get_mut(&ticker.pair) {
            position.update_price(ticker.last);
        }
    }

    pub fn equity(&self) -> Decimal {
        let positions_value: Decimal = self
            .positions
            .values()
            .map(|p| p.current_price * p.quantity)
            .sum();

        self.balance + positions_value
    }
}
