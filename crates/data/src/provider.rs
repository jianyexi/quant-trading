use async_trait::async_trait;
use sqlx::PgPool;
use tracing::{info, warn};

use quant_core::error::Result;
use quant_core::models::{Kline, StockInfo, Tick};
use quant_core::traits::DataProvider;
use quant_core::types::TimeFrame;

use crate::akshare::AkshareClient;
use crate::tushare::TushareClient;

pub struct DataProviderImpl {
    tushare: TushareClient,
    akshare: AkshareClient,
    pool: PgPool,
}

impl DataProviderImpl {
    pub fn new(tushare: TushareClient, akshare: AkshareClient, pool: PgPool) -> Self {
        Self {
            tushare,
            akshare,
            pool,
        }
    }

    async fn store_klines(&self, klines: &[Kline]) {
        for kline in klines {
            let result = sqlx::query(
                "INSERT INTO klines (symbol, datetime, open, high, low, close, volume)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)
                 ON CONFLICT (symbol, datetime) DO NOTHING",
            )
            .bind(&kline.symbol)
            .bind(kline.datetime)
            .bind(kline.open)
            .bind(kline.high)
            .bind(kline.low)
            .bind(kline.close)
            .bind(kline.volume)
            .execute(&self.pool)
            .await;

            if let Err(e) = result {
                warn!("Failed to store kline: {}", e);
            }
        }
    }
}

#[async_trait]
impl DataProvider for DataProviderImpl {
    async fn fetch_kline(
        &self,
        symbol: &str,
        start: &str,
        end: &str,
        _timeframe: TimeFrame,
    ) -> Result<Vec<Kline>> {
        // Try Tushare first, fall back to AKShare
        let result = self.tushare.fetch_daily(symbol, start, end).await;
        let klines = match result {
            Ok(data) => {
                info!("Fetched {} klines from Tushare for {}", data.len(), symbol);
                data
            }
            Err(e) => {
                warn!("Tushare failed for {}: {}, falling back to AKShare", symbol, e);
                self.akshare.fetch_daily(symbol, start, end).await?
            }
        };

        self.store_klines(&klines).await;
        Ok(klines)
    }

    async fn fetch_stock_list(&self) -> Result<Vec<StockInfo>> {
        self.tushare.fetch_stock_basic().await
    }

    async fn fetch_realtime_quote(&self, symbol: &str) -> Result<Tick> {
        let result = self.tushare.fetch_realtime_quote(symbol).await;
        match result {
            Ok(tick) => Ok(tick),
            Err(e) => {
                warn!("Tushare realtime failed for {}: {}, falling back to AKShare", symbol, e);
                self.akshare.fetch_realtime_quote(symbol).await
            }
        }
    }
}
