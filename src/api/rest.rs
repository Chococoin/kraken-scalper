//! Kraken REST API client
//!
//! Implements authenticated endpoints for trading operations.

use crate::api::auth::KrakenAuth;
use crate::trading::order::{OrderSide, OrderType};
use anyhow::{anyhow, Result};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{de::DeserializeOwned, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, error, info};

/// Rate limit: minimum time between requests (Kraken allows ~15 requests in 3 seconds for private API)
const MIN_REQUEST_INTERVAL: Duration = Duration::from_millis(200);

/// Kraken REST API client
pub struct KrakenRestClient {
    client: Client,
    auth: KrakenAuth,
    base_url: String,
    last_request: Instant,
}

/// Generic Kraken API response wrapper
#[derive(Debug, Deserialize)]
pub struct KrakenResponse<T> {
    pub error: Vec<String>,
    pub result: Option<T>,
}

/// Balance response
pub type BalanceResult = HashMap<String, String>;

/// Add order response
#[derive(Debug, Deserialize)]
pub struct AddOrderResult {
    pub descr: OrderDescription,
    pub txid: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct OrderDescription {
    pub order: String,
    pub close: Option<String>,
}

/// Cancel order response
#[derive(Debug, Deserialize)]
pub struct CancelOrderResult {
    pub count: u32,
    pub pending: Option<bool>,
}

/// Open orders response
#[derive(Debug, Deserialize)]
pub struct OpenOrdersResult {
    pub open: HashMap<String, OrderInfo>,
}

#[derive(Debug, Deserialize)]
pub struct OrderInfo {
    pub status: String,
    pub opentm: f64,
    pub starttm: f64,
    pub expiretm: f64,
    pub descr: OrderDescr,
    pub vol: String,
    pub vol_exec: String,
    pub cost: String,
    pub fee: String,
    pub price: String,
    pub stopprice: String,
    pub limitprice: String,
    pub misc: String,
    pub oflags: String,
}

#[derive(Debug, Deserialize)]
pub struct OrderDescr {
    pub pair: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub ordertype: String,
    pub price: String,
    pub price2: String,
    pub leverage: String,
    pub order: String,
    pub close: Option<String>,
}

/// Open positions response
#[derive(Debug, Deserialize)]
pub struct OpenPositionsResult(pub HashMap<String, PositionInfo>);

#[derive(Debug, Deserialize)]
pub struct PositionInfo {
    pub ordertxid: String,
    pub posstatus: String,
    pub pair: String,
    #[serde(rename = "type")]
    pub position_type: String,
    pub ordertype: String,
    pub cost: String,
    pub fee: String,
    pub vol: String,
    pub vol_closed: String,
    pub margin: String,
    pub value: String,
    pub net: String,
    pub terms: String,
    pub rollovertm: String,
    pub misc: String,
    pub oflags: String,
}

/// Trades history response
#[derive(Debug, Deserialize)]
pub struct TradesHistoryResult {
    pub trades: HashMap<String, TradeInfo>,
    pub count: u32,
}

#[derive(Debug, Deserialize)]
pub struct TradeInfo {
    pub ordertxid: String,
    pub postxid: String,
    pub pair: String,
    pub time: f64,
    #[serde(rename = "type")]
    pub trade_type: String,
    pub ordertype: String,
    pub price: String,
    pub cost: String,
    pub fee: String,
    pub vol: String,
    pub margin: String,
    pub misc: String,
}

/// WebSocket token response
#[derive(Debug, Deserialize)]
pub struct WebSocketTokenResult {
    pub token: String,
    pub expires: u64,
}

impl KrakenRestClient {
    /// Create a new Kraken REST API client
    pub fn new(api_key: &str, api_secret: &str, base_url: &str) -> Result<Self> {
        let auth = KrakenAuth::new(api_key, api_secret)?;
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            auth,
            base_url: base_url.to_string(),
            last_request: Instant::now() - MIN_REQUEST_INTERVAL,
        })
    }

    /// Wait for rate limiting if necessary
    async fn rate_limit(&mut self) {
        let elapsed = self.last_request.elapsed();
        if elapsed < MIN_REQUEST_INTERVAL {
            tokio::time::sleep(MIN_REQUEST_INTERVAL - elapsed).await;
        }
        self.last_request = Instant::now();
    }

    /// Make a private (authenticated) API request
    async fn private_request<T: DeserializeOwned>(
        &mut self,
        endpoint: &str,
        params: &[(&str, &str)],
    ) -> Result<T> {
        self.rate_limit().await;

        let nonce = KrakenAuth::generate_nonce();
        let uri_path = format!("/0/private/{}", endpoint);

        // Build POST data
        let mut post_data = format!("nonce={}", nonce);
        for (key, value) in params {
            post_data.push_str(&format!("&{}={}", key, value));
        }

        // Sign request
        let signature = self.auth.sign_request(&uri_path, nonce, &post_data)?;

        debug!("Kraken API request: {} with params: {:?}", endpoint, params);

        // Make request
        let url = format!("{}{}", self.base_url, uri_path);
        let response = self
            .client
            .post(&url)
            .header("API-Key", self.auth.api_key())
            .header("API-Sign", &signature)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(post_data)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        debug!("Kraken API response: {} - {}", status, &body[..body.len().min(200)]);

        if !status.is_success() {
            return Err(anyhow!("API request failed with status {}: {}", status, body));
        }

        let result: KrakenResponse<T> = serde_json::from_str(&body)?;

        if !result.error.is_empty() {
            let errors = result.error.join(", ");
            error!("Kraken API error: {}", errors);
            return Err(anyhow!("Kraken API error: {}", errors));
        }

        result
            .result
            .ok_or_else(|| anyhow!("No result in API response"))
    }

    /// Get account balance
    pub async fn get_balance(&mut self) -> Result<HashMap<String, Decimal>> {
        let result: BalanceResult = self.private_request("Balance", &[]).await?;

        let mut balances = HashMap::new();
        for (asset, amount) in result {
            if let Ok(value) = amount.parse::<Decimal>() {
                if value > Decimal::ZERO {
                    balances.insert(asset, value);
                }
            }
        }

        info!("Fetched {} non-zero balances", balances.len());
        Ok(balances)
    }

    /// Place a new order
    pub async fn add_order(
        &mut self,
        pair: &str,
        side: OrderSide,
        order_type: OrderType,
        volume: Decimal,
        price: Option<Decimal>,
    ) -> Result<AddOrderResult> {
        let side_str = match side {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        };

        let type_str = match order_type {
            OrderType::Market => "market",
            OrderType::Limit => "limit",
            OrderType::StopLoss => "stop-loss",
            OrderType::TakeProfit => "take-profit",
        };

        let volume_str = volume.to_string();
        let mut params = vec![
            ("pair", pair),
            ("type", side_str),
            ("ordertype", type_str),
            ("volume", &volume_str),
        ];

        let price_str;
        if let Some(p) = price {
            price_str = p.to_string();
            params.push(("price", &price_str));
        }

        info!(
            "Placing {} {} order for {} {} @ {:?}",
            side_str, type_str, volume, pair, price
        );

        self.private_request("AddOrder", &params).await
    }

    /// Cancel an open order
    pub async fn cancel_order(&mut self, txid: &str) -> Result<CancelOrderResult> {
        info!("Cancelling order: {}", txid);
        self.private_request("CancelOrder", &[("txid", txid)]).await
    }

    /// Get open orders
    pub async fn get_open_orders(&mut self) -> Result<OpenOrdersResult> {
        self.private_request("OpenOrders", &[]).await
    }

    /// Get open positions
    pub async fn get_open_positions(&mut self) -> Result<HashMap<String, PositionInfo>> {
        let result: OpenPositionsResult = self.private_request("OpenPositions", &[]).await?;
        Ok(result.0)
    }

    /// Get trade history
    pub async fn get_trades_history(&mut self, start: Option<i64>) -> Result<TradesHistoryResult> {
        let mut params = Vec::new();
        let start_str;

        if let Some(s) = start {
            start_str = s.to_string();
            params.push(("start", start_str.as_str()));
        }

        self.private_request("TradesHistory", &params).await
    }

    /// Get WebSocket authentication token
    pub async fn get_websocket_token(&mut self) -> Result<String> {
        let result: WebSocketTokenResult = self.private_request("GetWebSocketsToken", &[]).await?;
        info!("Got WebSocket token, expires in {} seconds", result.expires);
        Ok(result.token)
    }

    /// Check if API credentials are valid by making a simple request
    pub async fn validate_credentials(&mut self) -> Result<bool> {
        match self.get_balance().await {
            Ok(_) => Ok(true),
            Err(e) => {
                if e.to_string().contains("EAPI:Invalid key") {
                    Ok(false)
                } else {
                    Err(e)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_parsing() {
        let json = r#"{
            "error": [],
            "result": {
                "ZUSD": "1000.00",
                "XXBT": "0.5"
            }
        }"#;

        let response: KrakenResponse<BalanceResult> = serde_json::from_str(json).unwrap();
        assert!(response.error.is_empty());
        assert!(response.result.is_some());

        let result = response.result.unwrap();
        assert_eq!(result.get("ZUSD"), Some(&"1000.00".to_string()));
    }

    #[test]
    fn test_error_parsing() {
        let json = r#"{
            "error": ["EAPI:Invalid key"],
            "result": null
        }"#;

        let response: KrakenResponse<BalanceResult> = serde_json::from_str(json).unwrap();
        assert!(!response.error.is_empty());
        assert_eq!(response.error[0], "EAPI:Invalid key");
    }
}
