use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Kraken WebSocket v2 message wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[serde(rename_all = "snake_case")]
pub enum WsMessage {
    Subscribe(SubscribeRequest),
    Unsubscribe(UnsubscribeRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeRequest {
    pub params: SubscribeParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub req_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeParams {
    pub channel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsubscribeRequest {
    pub params: SubscribeParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub req_id: Option<u64>,
}

/// Incoming messages from Kraken WebSocket
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum WsResponse {
    Channel(ChannelMessage),
    Status(StatusMessage),
    Error(ErrorMessage),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChannelMessage {
    pub channel: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub data: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StatusMessage {
    pub channel: String,
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(default)]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorMessage {
    pub error: String,
    pub success: bool,
}

/// Ticker data from Kraken
#[derive(Debug, Clone, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    pub bid: Decimal,
    pub bid_qty: Decimal,
    pub ask: Decimal,
    pub ask_qty: Decimal,
    pub last: Decimal,
    pub volume: Decimal,
    pub vwap: Decimal,
    pub low: Decimal,
    pub high: Decimal,
    pub change: Decimal,
    pub change_pct: Decimal,
}

/// Order book data from Kraken
#[derive(Debug, Clone, Deserialize)]
pub struct BookData {
    pub symbol: String,
    #[serde(default)]
    pub bids: Vec<BookLevel>,
    #[serde(default)]
    pub asks: Vec<BookLevel>,
    #[serde(default)]
    pub checksum: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BookLevel {
    pub price: Decimal,
    pub qty: Decimal,
}

/// OHLC candle data from Kraken
#[derive(Debug, Clone, Deserialize)]
pub struct OhlcData {
    pub symbol: String,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub vwap: Decimal,
    pub volume: Decimal,
    pub trades: u64,
    pub interval_begin: String,
    pub interval: u32,
}

/// Trade data from Kraken
#[derive(Debug, Clone, Deserialize)]
pub struct TradeData {
    pub symbol: String,
    pub side: String,
    pub price: Decimal,
    pub qty: Decimal,
    pub ord_type: String,
    pub trade_id: u64,
    pub timestamp: String,
}
