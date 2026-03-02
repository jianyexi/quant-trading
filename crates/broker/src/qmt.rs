/// QMT live broker — communicates with the QMT Python bridge sidecar.
///
/// The Python bridge (`qmt_bridge.py`) wraps the `xtquant` SDK and exposes
/// a local HTTP API. This broker sends HTTP requests to the bridge to execute
/// real trades via QMT (迅投量化, miniQMT mode).

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug};
use uuid::Uuid;

use quant_core::error::{QuantError, Result};
use quant_core::models::*;
use quant_core::traits::Broker;
use quant_core::types::*;

// ── Bridge API types ────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct BridgeOrderRequest {
    stock_code: String,
    price: f64,
    amount: i64,
    side: String,
    price_type: String,
}

#[derive(Debug, Deserialize)]
struct BridgeOrderResponse {
    order_id: Option<i64>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BridgeCancelResponse {
    #[allow(dead_code)]
    result: Option<i32>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BridgePosition {
    stock_code: String,
    volume: f64,
    #[allow(dead_code)]
    can_use_volume: f64,
    #[allow(dead_code)]
    frozen_volume: f64,
    #[allow(dead_code)]
    open_price: f64,
    market_value: f64,
    cost_price: f64,
}

#[derive(Debug, Deserialize)]
struct BridgePositionsResponse {
    positions: Option<Vec<BridgePosition>>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BridgeAccountResponse {
    total_asset: Option<f64>,
    cash: Option<f64>,
    #[allow(dead_code)]
    market_value: Option<f64>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BridgeHealthResponse {
    pub status: String,
    pub connected: bool,
}

// ── QMT Broker ──────────────────────────────────────────────────────

/// Configuration for connecting to the QMT bridge.
#[derive(Debug, Clone)]
pub struct QmtConfig {
    pub bridge_url: String,
    pub account: String,
}

pub struct QmtBroker {
    config: QmtConfig,
    client: Client,
    /// Map Uuid order IDs → QMT integer order IDs
    order_map: std::sync::Mutex<HashMap<Uuid, i64>>,
}

impl QmtBroker {
    pub fn new(config: QmtConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .connect_timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            config,
            client,
            order_map: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Check if the bridge is healthy and connected.
    pub async fn check_connection(&self) -> Result<bool> {
        let url = format!("{}/health", self.config.bridge_url);
        debug!(url=%url, "QMT bridge health check");
        let resp = self.client.get(&url).send().await
            .map_err(|e| QuantError::BrokerError(format!("Bridge unreachable: {}", e)))?;
        let health: BridgeHealthResponse = resp.json().await
            .map_err(|e| QuantError::BrokerError(format!("Bad health response: {}", e)))?;
        debug!(status=%health.status, connected=%health.connected, "QMT bridge health response");
        Ok(health.status == "ok" && health.connected)
    }

    /// Map order side + type to bridge params.
    fn map_order(order: &Order) -> (String, String) {
        let side = match order.side {
            OrderSide::Buy => "buy".to_string(),
            OrderSide::Sell => "sell".to_string(),
        };
        let price_type = match order.order_type {
            OrderType::Market => "market".to_string(),
            OrderType::Limit => "limit".to_string(),
        };
        (side, price_type)
    }

    /// Sync order status from QMT bridge. Returns updated OrderStatus.
    pub async fn sync_order_status(&self, order_id: Uuid) -> Result<OrderStatus> {
        let qmt_id = {
            let map = self.order_map.lock()
                .map_err(|e| QuantError::BrokerError(e.to_string()))?;
            map.get(&order_id).copied()
                .ok_or_else(|| QuantError::BrokerError(
                    format!("No QMT order ID mapped for {}", order_id)
                ))?
        };

        let url = format!("{}/order_result/{}", self.config.bridge_url, qmt_id);
        let max_retries = 3u32;
        let mut last_err = String::new();

        for attempt in 0..max_retries {
            match self.client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    #[derive(Deserialize)]
                    #[allow(dead_code)]
                    struct OrderResult {
                        order_status: Option<i32>,
                        traded_volume: Option<f64>,
                        error_msg: Option<String>,
                    }
                    match resp.json::<OrderResult>().await {
                        Ok(r) => {
                            // QMT order_status codes: typically 5=filled, 6=cancelled, 8=rejected
                            let status = match r.order_status.unwrap_or(-1) {
                                5 => OrderStatus::Filled,
                                6 => OrderStatus::Cancelled,
                                8 => OrderStatus::Rejected,
                                _ => OrderStatus::Submitted,
                            };
                            return Ok(status);
                        }
                        Err(e) => last_err = format!("Parse error: {}", e),
                    }
                }
                Ok(resp) if resp.status().as_u16() == 404 => {
                    return Ok(OrderStatus::Submitted); // no callback yet
                }
                Ok(resp) => last_err = format!("HTTP {}", resp.status()),
                Err(e) => last_err = format!("Request failed: {}", e),
            }

            if attempt < max_retries - 1 {
                let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt));
                tokio::time::sleep(delay).await;
            }
        }

        Err(QuantError::BrokerError(format!(
            "sync_order_status failed after {} retries: {}", max_retries, last_err
        )))
    }
}

#[async_trait]
impl Broker for QmtBroker {
    async fn submit_order(&self, order: &Order) -> Result<Order> {
        let (side, price_type) = Self::map_order(order);
        let amount = order.quantity as i64;

        let req = BridgeOrderRequest {
            stock_code: order.symbol.clone(),
            price: order.price,
            amount,
            side,
            price_type,
        };

        let url = format!("{}/order", self.config.bridge_url);
        debug!(url=%url, symbol=%order.symbol, side=%req.side, price=%req.price, amount=%req.amount, "QMT submit_order request");
        let resp = self.client.post(&url).json(&req).send().await
            .map_err(|e| QuantError::BrokerError(format!("Bridge request failed: {}", e)))?;

        let status = resp.status();
        let body: BridgeOrderResponse = resp.json().await
            .map_err(|e| QuantError::BrokerError(format!("Bad order response: {}", e)))?;

        if !status.is_success() || body.order_id.is_none() {
            let msg = body.error.unwrap_or_else(|| format!("HTTP {}", status));
            error!("QMT order failed: {}", msg);
            return Err(QuantError::BrokerError(msg));
        }

        let qmt_id = body.order_id.unwrap();
        info!("QMT order submitted: {} → qmt_id={}", order.symbol, qmt_id);

        // Track the mapping
        if let Ok(mut map) = self.order_map.lock() {
            map.insert(order.id, qmt_id);
        }

        let mut submitted = order.clone();
        submitted.status = OrderStatus::Submitted;
        submitted.updated_at = Utc::now().naive_utc();
        Ok(submitted)
    }

    async fn cancel_order(&self, order_id: Uuid) -> Result<()> {
        let qmt_id = {
            let map = self.order_map.lock()
                .map_err(|e| QuantError::BrokerError(e.to_string()))?;
            map.get(&order_id).copied()
                .ok_or_else(|| QuantError::BrokerError(
                    format!("No QMT order ID mapped for {}", order_id)
                ))?
        };

        let url = format!("{}/cancel", self.config.bridge_url);
        debug!(url=%url, order_id=%order_id, qmt_id=%qmt_id, "QMT cancel_order request");
        let body = serde_json::json!({ "order_id": qmt_id });
        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| QuantError::BrokerError(format!("Cancel request failed: {}", e)))?;

        let cancel: BridgeCancelResponse = resp.json().await
            .map_err(|e| QuantError::BrokerError(format!("Bad cancel response: {}", e)))?;

        if let Some(err) = cancel.error {
            return Err(QuantError::BrokerError(err));
        }

        info!("QMT order cancelled: uuid={} qmt_id={}", order_id, qmt_id);
        Ok(())
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        let url = format!("{}/positions", self.config.bridge_url);
        debug!(url=%url, "QMT get_positions request");
        let resp = self.client.get(&url).send().await
            .map_err(|e| QuantError::BrokerError(format!("Positions request failed: {}", e)))?;

        let body: BridgePositionsResponse = resp.json().await
            .map_err(|e| QuantError::BrokerError(format!("Bad positions response: {}", e)))?;

        if let Some(err) = body.error {
            return Err(QuantError::BrokerError(err));
        }

        let positions = body.positions.unwrap_or_default()
            .into_iter()
            .filter(|p| p.volume > 0.0)
            .map(|p| {
                let unrealized = (p.market_value / p.volume - p.cost_price) * p.volume;
                Position {
                    symbol: p.stock_code,
                    quantity: p.volume,
                    avg_cost: p.cost_price,
                    current_price: if p.volume > 0.0 { p.market_value / p.volume } else { 0.0 },
                    unrealized_pnl: unrealized,
                    realized_pnl: 0.0,
                    entry_time: chrono::Utc::now().naive_utc(),
                    scale_level: 1,
                    target_weight: 0.0,
                }
            })
            .collect();

        Ok(positions)
    }

    async fn get_account(&self) -> Result<Account> {
        let url = format!("{}/account", self.config.bridge_url);
        debug!(url=%url, "QMT get_account request");
        let resp = self.client.get(&url).send().await
            .map_err(|e| QuantError::BrokerError(format!("Account request failed: {}", e)))?;

        let body: BridgeAccountResponse = resp.json().await
            .map_err(|e| QuantError::BrokerError(format!("Bad account response: {}", e)))?;

        if let Some(err) = body.error {
            return Err(QuantError::BrokerError(err));
        }

        let cash = body.cash.unwrap_or(0.0);
        let total = body.total_asset.unwrap_or(cash);

        // Also fetch positions to build portfolio
        let positions_list = self.get_positions().await.unwrap_or_default();
        let mut positions_map = HashMap::new();
        for p in positions_list {
            positions_map.insert(p.symbol.clone(), p);
        }

        Ok(Account {
            id: Uuid::new_v4(),
            name: format!("QMT-{}", self.config.account),
            portfolio: Portfolio {
                positions: positions_map,
                cash,
                total_value: total,
            },
            initial_capital: total, // QMT doesn't track initial capital
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ── QMT Market Data ────────────────────────────────────────────────

/// Market data kline bar returned from QMT bridge.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QmtKline {
    pub datetime: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Real-time quote snapshot from QMT bridge.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QmtQuote {
    pub stock_code: String,
    pub last_price: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub pre_close: f64,
    pub volume: f64,
    pub amount: f64,
    #[serde(default)]
    pub bid_prices: Vec<f64>,
    #[serde(default)]
    pub ask_prices: Vec<f64>,
    #[serde(default)]
    pub bid_vols: Vec<i64>,
    #[serde(default)]
    pub ask_vols: Vec<i64>,
    #[serde(default)]
    pub timestamp: i64,
}

impl QmtBroker {
    /// Fetch historical kline data from the QMT bridge via xtdata.
    ///
    /// `period`: "1d", "1m", "5m", "15m", "30m", "60m"
    pub async fn fetch_klines(
        &self,
        stock_code: &str,
        period: &str,
        start_time: &str,
        end_time: &str,
    ) -> Result<Vec<QmtKline>> {
        let url = format!(
            "{}/market/kline?stock_code={}&period={}&start_time={}&end_time={}",
            self.config.bridge_url, stock_code, period, start_time, end_time
        );
        debug!(url=%url, stock_code=%stock_code, period=%period, "QMT fetch_klines request");
        let resp = self.client.get(&url).send().await
            .map_err(|e| QuantError::DataError(format!("QMT kline request failed: {}", e)))?;

        #[derive(Deserialize)]
        struct KlineResp {
            klines: Option<Vec<QmtKline>>,
            error: Option<String>,
        }

        let body: KlineResp = resp.json().await
            .map_err(|e| QuantError::DataError(format!("Bad kline response: {}", e)))?;

        if let Some(err) = body.error {
            return Err(QuantError::DataError(err));
        }
        let klines = body.klines.unwrap_or_default();
        debug!(count=%klines.len(), "QMT fetch_klines response");
        Ok(klines)
    }

    /// Fetch real-time quotes for one or more stocks.
    pub async fn fetch_quotes(&self, stock_codes: &[&str]) -> Result<Vec<QmtQuote>> {
        let codes = stock_codes.join(",");
        let url = format!(
            "{}/market/quote?stock_codes={}",
            self.config.bridge_url, codes
        );
        debug!(url=%url, count=%stock_codes.len(), "QMT fetch_quotes request");
        let resp = self.client.get(&url).send().await
            .map_err(|e| QuantError::DataError(format!("QMT quote request failed: {}", e)))?;

        #[derive(Deserialize)]
        struct QuoteResp {
            quotes: Option<Vec<QmtQuote>>,
            error: Option<String>,
        }

        let body: QuoteResp = resp.json().await
            .map_err(|e| QuantError::DataError(format!("Bad quote response: {}", e)))?;

        if let Some(err) = body.error {
            return Err(QuantError::DataError(err));
        }
        let quotes = body.quotes.unwrap_or_default();
        debug!(count=%quotes.len(), "QMT fetch_quotes response");
        Ok(quotes)
    }

    /// Subscribe to real-time market data on the bridge side.
    pub async fn subscribe_market(&self, stock_codes: &[&str], period: &str) -> Result<usize> {
        let url = format!("{}/market/subscribe", self.config.bridge_url);
        let body = serde_json::json!({
            "stock_codes": stock_codes,
            "period": period,
        });
        debug!(url=%url, count=%stock_codes.len(), period=%period, "QMT subscribe_market request");
        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| QuantError::DataError(format!("QMT subscribe request failed: {}", e)))?;

        #[derive(Deserialize)]
        struct SubResp {
            count: Option<usize>,
            error: Option<String>,
        }

        let result: SubResp = resp.json().await
            .map_err(|e| QuantError::DataError(format!("Bad subscribe response: {}", e)))?;

        if let Some(err) = result.error {
            return Err(QuantError::DataError(err));
        }
        let count = result.count.unwrap_or(0);
        debug!(count=%count, "QMT subscribe_market response");
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Test that QmtBroker correctly maps order fields and handles responses.
    /// Uses a tiny mock HTTP server to avoid needing a real QMT bridge.
    #[tokio::test]
    async fn test_qmt_broker_submit_order() {
        // Start a mock server
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        // Spawn mock responder
        let handle = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let n = stream.read(&mut buf).await.unwrap();
            let req = String::from_utf8_lossy(&buf[..n]);
            assert!(req.contains("POST /order"));
            assert!(req.contains("000001.SZ"));

            let body = r#"{"order_id": 12345}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
                body.len(), body
            );
            stream.write_all(resp.as_bytes()).await.unwrap();
        });

        let config = QmtConfig {
            bridge_url: format!("http://127.0.0.1:{}", port),
            account: "test_account".into(),
        };
        let broker = QmtBroker::new(config);

        let order = Order {
            id: Uuid::new_v4(),
            symbol: "000001.SZ".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            price: 10.5,
            quantity: 100.0,
            filled_qty: 0.0,
            status: OrderStatus::Pending,
            created_at: Utc::now().naive_utc(),
            updated_at: Utc::now().naive_utc(),
        };

        let result = broker.submit_order(&order).await;
        assert!(result.is_ok());
        let submitted = result.unwrap();
        assert_eq!(submitted.status, OrderStatus::Submitted);

        // Verify the mapping was recorded
        let map = broker.order_map.lock().unwrap();
        assert_eq!(map.get(&order.id), Some(&12345i64));

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_qmt_broker_get_positions() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _n = stream.read(&mut buf).await.unwrap();

            let body = r#"{"positions":[{"stock_code":"000001.SZ","volume":1000,"can_use_volume":1000,"frozen_volume":0,"open_price":10.0,"market_value":11000,"cost_price":10.0}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
                body.len(), body
            );
            stream.write_all(resp.as_bytes()).await.unwrap();
        });

        let config = QmtConfig {
            bridge_url: format!("http://127.0.0.1:{}", port),
            account: "test".into(),
        };
        let broker = QmtBroker::new(config);

        let positions = broker.get_positions().await.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].symbol, "000001.SZ");
        assert_eq!(positions[0].quantity, 1000.0);
        assert_eq!(positions[0].avg_cost, 10.0);

        handle.await.unwrap();
    }
}
