use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Market => write!(f, "MARKET"),
            OrderType::Limit => write!(f, "LIMIT"),
            OrderType::StopLoss => write!(f, "STOP_LOSS"),
            OrderType::TakeProfit => write!(f, "TAKE_PROFIT"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Open,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected,
}

impl std::fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderStatus::Pending => write!(f, "PENDING"),
            OrderStatus::Open => write!(f, "OPEN"),
            OrderStatus::Filled => write!(f, "FILLED"),
            OrderStatus::PartiallyFilled => write!(f, "PARTIAL"),
            OrderStatus::Cancelled => write!(f, "CANCELLED"),
            OrderStatus::Rejected => write!(f, "REJECTED"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub pair: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub filled_quantity: Decimal,
    pub filled_price: Option<Decimal>,
    pub status: OrderStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Order {
    pub fn market(pair: &str, side: OrderSide, quantity: Decimal) -> Self {
        let now = Utc::now();
        Self {
            id: generate_order_id(),
            pair: pair.to_string(),
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            filled_quantity: Decimal::ZERO,
            filled_price: None,
            status: OrderStatus::Pending,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn limit(pair: &str, side: OrderSide, quantity: Decimal, price: Decimal) -> Self {
        let now = Utc::now();
        Self {
            id: generate_order_id(),
            pair: pair.to_string(),
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            filled_quantity: Decimal::ZERO,
            filled_price: None,
            status: OrderStatus::Pending,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Pending | OrderStatus::Open | OrderStatus::PartiallyFilled
        )
    }
}

fn generate_order_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("ORD-{}", timestamp)
}

#[derive(Debug, Clone)]
pub struct Position {
    pub id: i64,
    pub pair: String,
    pub side: OrderSide,
    pub entry_price: Decimal,
    pub quantity: Decimal,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub unrealized_pnl_pct: Decimal,
    pub opened_at: DateTime<Utc>,
}

impl Position {
    pub fn new(
        id: i64,
        pair: &str,
        side: OrderSide,
        entry_price: Decimal,
        quantity: Decimal,
    ) -> Self {
        Self {
            id,
            pair: pair.to_string(),
            side,
            entry_price,
            quantity,
            current_price: entry_price,
            unrealized_pnl: Decimal::ZERO,
            unrealized_pnl_pct: Decimal::ZERO,
            opened_at: Utc::now(),
        }
    }

    pub fn update_price(&mut self, price: Decimal) {
        self.current_price = price;
        self.unrealized_pnl = self.calculate_pnl(price);
        self.unrealized_pnl_pct = if !self.entry_price.is_zero() {
            (self.unrealized_pnl / (self.entry_price * self.quantity)) * Decimal::from(100)
        } else {
            Decimal::ZERO
        };
    }

    pub fn calculate_pnl(&self, exit_price: Decimal) -> Decimal {
        let diff = exit_price - self.entry_price;
        match self.side {
            OrderSide::Buy => diff * self.quantity,
            OrderSide::Sell => -diff * self.quantity,
        }
    }
}
