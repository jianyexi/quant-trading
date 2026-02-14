use chrono::NaiveDateTime;
use reqwest::Client;
use serde::Deserialize;
use tracing::debug;

use quant_core::error::{QuantError, Result};
use quant_core::models::{Kline, Tick};

pub struct AkshareClient {
    client: Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct AkDailyRecord {
    #[serde(alias = "日期", alias = "date")]
    pub date: String,
    #[serde(alias = "开盘", alias = "open")]
    pub open: f64,
    #[serde(alias = "最高", alias = "high")]
    pub high: f64,
    #[serde(alias = "最低", alias = "low")]
    pub low: f64,
    #[serde(alias = "收盘", alias = "close")]
    pub close: f64,
    #[serde(alias = "成交量", alias = "volume")]
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct AkRealtimeRecord {
    #[serde(alias = "代码", alias = "code")]
    pub code: String,
    #[serde(alias = "时间", alias = "time")]
    pub time: Option<String>,
    #[serde(alias = "最新价", alias = "price")]
    pub price: f64,
    #[serde(alias = "成交量", alias = "volume")]
    pub volume: f64,
    #[serde(alias = "买一", alias = "bid")]
    pub bid: f64,
    #[serde(alias = "卖一", alias = "ask")]
    pub ask: f64,
}

impl AkshareClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub async fn fetch_daily(
        &self,
        symbol: &str,
        start: &str,
        end: &str,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/api/public/stock_zh_a_hist?symbol={}&start_date={}&end_date={}&adjust=qfq",
            self.base_url, symbol, start, end
        );

        debug!("AKShare daily request: {}", url);

        let resp = self.client.get(&url).send().await?;
        let records: Vec<AkDailyRecord> = resp.json().await?;

        records
            .into_iter()
            .map(|r| {
                let datetime = parse_ak_date(&r.date)?;
                Ok(Kline {
                    symbol: symbol.to_string(),
                    datetime,
                    open: r.open,
                    high: r.high,
                    low: r.low,
                    close: r.close,
                    volume: r.volume,
                })
            })
            .collect()
    }

    pub async fn fetch_realtime_quote(&self, symbol: &str) -> Result<Tick> {
        let url = format!(
            "{}/api/public/stock_zh_a_spot_em?symbol={}",
            self.base_url, symbol
        );

        debug!("AKShare realtime request: {}", url);

        let resp = self.client.get(&url).send().await?;
        let records: Vec<AkRealtimeRecord> = resp.json().await?;

        let record = records
            .into_iter()
            .next()
            .ok_or_else(|| QuantError::DataError(format!("No realtime data for {}", symbol)))?;

        let datetime = match &record.time {
            Some(t) => NaiveDateTime::parse_from_str(t, "%Y-%m-%d %H:%M:%S")
                .unwrap_or_else(|_| chrono::Local::now().naive_local()),
            None => chrono::Local::now().naive_local(),
        };

        Ok(Tick {
            symbol: record.code,
            datetime,
            price: record.price,
            volume: record.volume,
            bid: record.bid,
            ask: record.ask,
        })
    }
}

fn parse_ak_date(date_str: &str) -> Result<NaiveDateTime> {
    // Try "YYYY-MM-DD" first, then "YYYYMMDD"
    chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .or_else(|_| chrono::NaiveDate::parse_from_str(date_str, "%Y%m%d"))
        .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        .map_err(|e| QuantError::DataError(format!("Failed to parse AKShare date '{}': {}", date_str, e)))
}
