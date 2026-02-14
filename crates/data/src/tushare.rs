use chrono::NaiveDateTime;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::debug;

use quant_core::error::{QuantError, Result};
use quant_core::models::{Kline, StockInfo, Tick};
use quant_core::types::Market;

pub struct TushareClient {
    client: Client,
    token: String,
    base_url: String,
}

impl TushareClient {
    pub fn new(token: &str, base_url: &str) -> Self {
        Self {
            client: Client::new(),
            token: token.to_string(),
            base_url: base_url.to_string(),
        }
    }

    async fn request(
        &self,
        api_name: &str,
        params: Value,
        fields: &str,
    ) -> Result<TushareResponse> {
        let body = json!({
            "api_name": api_name,
            "token": self.token,
            "params": params,
            "fields": fields,
        });

        debug!("Tushare request: api_name={}, params={}", api_name, params);

        let resp = self
            .client
            .post(&self.base_url)
            .json(&body)
            .send()
            .await?;

        let value: Value = resp.json().await?;

        let data = value
            .get("data")
            .ok_or_else(|| QuantError::DataError("Missing 'data' in Tushare response".into()))?;

        let fields_arr: Vec<String> = data
            .get("fields")
            .and_then(|f| serde_json::from_value(f.clone()).ok())
            .unwrap_or_default();

        let items: Vec<Vec<Value>> = data
            .get("items")
            .and_then(|i| serde_json::from_value(i.clone()).ok())
            .unwrap_or_default();

        Ok(TushareResponse {
            fields: fields_arr,
            items,
        })
    }

    pub async fn fetch_daily(
        &self,
        symbol: &str,
        start: &str,
        end: &str,
    ) -> Result<Vec<Kline>> {
        let params = json!({
            "ts_code": symbol,
            "start_date": start,
            "end_date": end,
        });
        let fields = "ts_code,trade_date,open,high,low,close,vol";
        let resp = self.request("daily", params, fields).await?;

        let field_idx = FieldIndex::new(&resp.fields);

        resp.items
            .iter()
            .map(|row| {
                let symbol = field_idx.get_str(row, "ts_code")?;
                let trade_date = field_idx.get_str(row, "trade_date")?;
                let datetime = parse_trade_date(&trade_date)?;

                Ok(Kline {
                    symbol,
                    datetime,
                    open: field_idx.get_f64(row, "open")?,
                    high: field_idx.get_f64(row, "high")?,
                    low: field_idx.get_f64(row, "low")?,
                    close: field_idx.get_f64(row, "close")?,
                    volume: field_idx.get_f64(row, "vol")?,
                })
            })
            .collect()
    }

    pub async fn fetch_stock_basic(&self) -> Result<Vec<StockInfo>> {
        let params = json!({
            "list_status": "L",
        });
        let fields = "ts_code,name,market,industry,list_date";
        let resp = self.request("stock_basic", params, fields).await?;

        let field_idx = FieldIndex::new(&resp.fields);

        resp.items
            .iter()
            .map(|row| {
                let ts_code = field_idx.get_str(row, "ts_code")?;
                let market_str = field_idx.get_str(row, "market")?;
                let market = parse_market(&market_str);

                Ok(StockInfo {
                    symbol: ts_code,
                    name: field_idx.get_str(row, "name")?,
                    market,
                    industry: field_idx.get_str(row, "industry")?,
                    list_date: field_idx.get_str(row, "list_date")?,
                })
            })
            .collect()
    }

    pub async fn fetch_realtime_quote(&self, symbol: &str) -> Result<Tick> {
        let params = json!({
            "ts_code": symbol,
        });
        let fields = "ts_code,trade_time,price,vol,bid_price1,ask_price1";
        let resp = self.request("realtime_quote", params, fields).await?;

        let field_idx = FieldIndex::new(&resp.fields);

        let row = resp
            .items
            .first()
            .ok_or_else(|| QuantError::DataError(format!("No quote data for {}", symbol)))?;

        let trade_time = field_idx.get_str(row, "trade_time")?;
        let datetime = NaiveDateTime::parse_from_str(&trade_time, "%Y-%m-%d %H:%M:%S")
            .map_err(|e| QuantError::DataError(format!("Failed to parse trade_time: {}", e)))?;

        Ok(Tick {
            symbol: field_idx.get_str(row, "ts_code")?,
            datetime,
            price: field_idx.get_f64(row, "price")?,
            volume: field_idx.get_f64(row, "vol")?,
            bid: field_idx.get_f64(row, "bid_price1")?,
            ask: field_idx.get_f64(row, "ask_price1")?,
        })
    }
}

// ── Internal helpers ─────────────────────────────────────────

struct TushareResponse {
    fields: Vec<String>,
    items: Vec<Vec<Value>>,
}

struct FieldIndex {
    names: Vec<String>,
}

impl FieldIndex {
    fn new(fields: &[String]) -> Self {
        Self {
            names: fields.to_vec(),
        }
    }

    fn index_of(&self, name: &str) -> Result<usize> {
        self.names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| QuantError::DataError(format!("Field '{}' not found in response", name)))
    }

    fn get_str(&self, row: &[Value], name: &str) -> Result<String> {
        let idx = self.index_of(name)?;
        row.get(idx)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .or_else(|| row.get(idx).map(|v| v.to_string()))
            .ok_or_else(|| QuantError::DataError(format!("Missing value for field '{}'", name)))
    }

    fn get_f64(&self, row: &[Value], name: &str) -> Result<f64> {
        let idx = self.index_of(name)?;
        row.get(idx)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| QuantError::DataError(format!("Missing f64 for field '{}'", name)))
    }
}

fn parse_trade_date(date_str: &str) -> Result<NaiveDateTime> {
    // Tushare dates are "YYYYMMDD" format
    chrono::NaiveDate::parse_from_str(date_str, "%Y%m%d")
        .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        .map_err(|e| QuantError::DataError(format!("Failed to parse trade_date '{}': {}", date_str, e)))
}

fn parse_market(market_str: &str) -> Market {
    match market_str {
        "主板" | "SH" => Market::SH,
        _ => Market::SZ,
    }
}
