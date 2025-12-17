use crate::api::models::*;
use crate::config::KrakenConfig;
use crate::data::{Candle, CandleStore, OrderBookStore, Ticker, TickerStore};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

pub enum MarketEvent {
    TickerUpdate(Ticker),
    OrderBookUpdate(String),
    CandleUpdate(String),
    Connected,
    Disconnected,
    Error(String),
}

pub struct KrakenWebSocket {
    config: KrakenConfig,
    pairs: Vec<String>,
    pub tickers: Arc<Mutex<TickerStore>>,
    pub orderbooks: Arc<Mutex<OrderBookStore>>,
    pub candles: Arc<Mutex<CandleStore>>,
}

impl KrakenWebSocket {
    pub fn new(config: KrakenConfig, pairs: Vec<String>, max_candles: usize) -> Self {
        Self {
            config,
            pairs,
            tickers: Arc::new(Mutex::new(TickerStore::new())),
            orderbooks: Arc::new(Mutex::new(OrderBookStore::new())),
            candles: Arc::new(Mutex::new(CandleStore::new(max_candles))),
        }
    }

    pub async fn connect(&self, event_tx: mpsc::Sender<MarketEvent>) -> Result<()> {
        let url = &self.config.ws_url;
        info!("Connecting to Kraken WebSocket: {}", url);

        let (ws_stream, _) = connect_async(url)
            .await
            .context("Failed to connect to Kraken WebSocket")?;

        let (mut write, mut read) = ws_stream.split();

        // Send subscription requests
        self.subscribe_all(&mut write).await?;

        let _ = event_tx.send(MarketEvent::Connected).await;

        let tickers = Arc::clone(&self.tickers);
        let orderbooks = Arc::clone(&self.orderbooks);
        let candles = Arc::clone(&self.candles);

        // Process incoming messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = Self::process_message(
                        &text,
                        &tickers,
                        &orderbooks,
                        &candles,
                        &event_tx,
                    )
                    .await
                    {
                        warn!("Error processing message: {}", e);
                    }
                }
                Ok(Message::Ping(data)) => {
                    if let Err(e) = write.send(Message::Pong(data)).await {
                        error!("Failed to send pong: {}", e);
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket closed by server");
                    let _ = event_tx.send(MarketEvent::Disconnected).await;
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    let _ = event_tx.send(MarketEvent::Error(e.to_string())).await;
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn subscribe_all<S>(&self, write: &mut S) -> Result<()>
    where
        S: SinkExt<Message> + Unpin,
        S::Error: std::error::Error + Send + Sync + 'static,
    {
        // Subscribe to ticker
        let ticker_sub = serde_json::json!({
            "method": "subscribe",
            "params": {
                "channel": "ticker",
                "symbol": self.pairs
            }
        });
        write
            .send(Message::Text(ticker_sub.to_string().into()))
            .await
            .context("Failed to send ticker subscription")?;

        // Subscribe to order book (depth 10)
        let book_sub = serde_json::json!({
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": self.pairs,
                "depth": 10
            }
        });
        write
            .send(Message::Text(book_sub.to_string().into()))
            .await
            .context("Failed to send book subscription")?;

        // Subscribe to OHLC (1 minute candles)
        let ohlc_sub = serde_json::json!({
            "method": "subscribe",
            "params": {
                "channel": "ohlc",
                "symbol": self.pairs,
                "interval": 1
            }
        });
        write
            .send(Message::Text(ohlc_sub.to_string().into()))
            .await
            .context("Failed to send ohlc subscription")?;

        info!("Subscribed to ticker, book, and ohlc for {:?}", self.pairs);
        Ok(())
    }

    async fn process_message(
        text: &str,
        tickers: &Arc<Mutex<TickerStore>>,
        orderbooks: &Arc<Mutex<OrderBookStore>>,
        candles: &Arc<Mutex<CandleStore>>,
        event_tx: &mpsc::Sender<MarketEvent>,
    ) -> Result<()> {
        let json: serde_json::Value = serde_json::from_str(text)?;

        // Check if it's a channel message
        let channel = json.get("channel").and_then(|c| c.as_str());
        let msg_type = json.get("type").and_then(|t| t.as_str());

        match (channel, msg_type) {
            (Some("ticker"), Some("update") | Some("snapshot")) => {
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    for item in data {
                        if let Ok(ticker_data) = serde_json::from_value::<TickerData>(item.clone())
                        {
                            let ticker = Ticker {
                                pair: ticker_data.symbol.clone(),
                                bid: ticker_data.bid,
                                ask: ticker_data.ask,
                                last: ticker_data.last,
                                volume_24h: ticker_data.volume,
                                high_24h: ticker_data.high,
                                low_24h: ticker_data.low,
                                vwap_24h: ticker_data.vwap,
                                change_24h: ticker_data.change_pct,
                                updated_at: Utc::now(),
                            };
                            debug!("Ticker update: {} @ {}", ticker.pair, ticker.last);
                            let _ = event_tx.send(MarketEvent::TickerUpdate(ticker.clone())).await;
                            tickers.lock().await.update(ticker);
                        }
                    }
                }
            }
            (Some("book"), Some("snapshot")) => {
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    for item in data {
                        if let Ok(book_data) = serde_json::from_value::<BookData>(item.clone()) {
                            let mut store = orderbooks.lock().await;
                            let book = store.get_or_create(&book_data.symbol);
                            book.clear();
                            for level in &book_data.bids {
                                book.update_bid(level.price, level.qty);
                            }
                            for level in &book_data.asks {
                                book.update_ask(level.price, level.qty);
                            }
                            let _ = event_tx
                                .send(MarketEvent::OrderBookUpdate(book_data.symbol))
                                .await;
                        }
                    }
                }
            }
            (Some("book"), Some("update")) => {
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    for item in data {
                        if let Ok(book_data) = serde_json::from_value::<BookData>(item.clone()) {
                            let mut store = orderbooks.lock().await;
                            let book = store.get_or_create(&book_data.symbol);
                            for level in &book_data.bids {
                                book.update_bid(level.price, level.qty);
                            }
                            for level in &book_data.asks {
                                book.update_ask(level.price, level.qty);
                            }
                            let _ = event_tx
                                .send(MarketEvent::OrderBookUpdate(book_data.symbol))
                                .await;
                        }
                    }
                }
            }
            (Some("ohlc"), Some("update") | Some("snapshot")) => {
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    for item in data {
                        if let Ok(ohlc_data) = serde_json::from_value::<OhlcData>(item.clone()) {
                            let timestamp = parse_timestamp(&ohlc_data.interval_begin)
                                .unwrap_or_else(Utc::now);
                            let candle = Candle {
                                timestamp,
                                open: ohlc_data.open,
                                high: ohlc_data.high,
                                low: ohlc_data.low,
                                close: ohlc_data.close,
                                volume: ohlc_data.volume,
                                vwap: ohlc_data.vwap,
                                trades: ohlc_data.trades,
                            };
                            candles.lock().await.update_last(&ohlc_data.symbol, candle);
                            let _ = event_tx
                                .send(MarketEvent::CandleUpdate(ohlc_data.symbol))
                                .await;
                        }
                    }
                }
            }
            (Some("status"), _) => {
                debug!("Status message: {}", text);
            }
            (Some("heartbeat"), _) => {
                // Ignore heartbeats
            }
            _ => {
                debug!("Unknown message: {}", text);
            }
        }

        Ok(())
    }
}

fn parse_timestamp(s: &str) -> Option<DateTime<Utc>> {
    // Kraken sends timestamps like "2024-01-15T10:30:00.000000Z"
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}
