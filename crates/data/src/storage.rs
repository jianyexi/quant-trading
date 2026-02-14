use chrono::NaiveDateTime;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};

use quant_core::models::{Kline, StockInfo};
use quant_core::types::Market;

pub async fn create_pool(database_url: &str, max_connections: u32) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(max_connections)
        .connect(database_url)
        .await
}

pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::Error> {
    sqlx::migrate!("../../migrations")
        .run(pool)
        .await?;
    Ok(())
}

pub struct MarketDataStore {
    pool: PgPool,
}

impl MarketDataStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn save_klines(&self, klines: &[Kline]) -> Result<(), sqlx::Error> {
        for kline in klines {
            sqlx::query(
                "INSERT INTO klines (symbol, datetime, open, high, low, close, volume)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)
                 ON CONFLICT (symbol, datetime) DO UPDATE SET
                   open = EXCLUDED.open,
                   high = EXCLUDED.high,
                   low = EXCLUDED.low,
                   close = EXCLUDED.close,
                   volume = EXCLUDED.volume",
            )
            .bind(&kline.symbol)
            .bind(kline.datetime)
            .bind(kline.open)
            .bind(kline.high)
            .bind(kline.low)
            .bind(kline.close)
            .bind(kline.volume)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    pub async fn get_klines(
        &self,
        symbol: &str,
        start: &str,
        end: &str,
    ) -> Result<Vec<Kline>, sqlx::Error> {
        let start_dt = NaiveDateTime::parse_from_str(
            &format!("{} 00:00:00", start),
            "%Y%m%d %H:%M:%S",
        )
        .unwrap_or_default();
        let end_dt = NaiveDateTime::parse_from_str(
            &format!("{} 23:59:59", end),
            "%Y%m%d %H:%M:%S",
        )
        .unwrap_or_default();

        let rows = sqlx::query(
            "SELECT symbol, datetime, open, high, low, close, volume
             FROM klines
             WHERE symbol = $1 AND datetime >= $2 AND datetime <= $3
             ORDER BY datetime ASC",
        )
        .bind(symbol)
        .bind(start_dt)
        .bind(end_dt)
        .fetch_all(&self.pool)
        .await?;

        let klines = rows
            .iter()
            .map(|row| Kline {
                symbol: row.get("symbol"),
                datetime: row.get("datetime"),
                open: row.get("open"),
                high: row.get("high"),
                low: row.get("low"),
                close: row.get("close"),
                volume: row.get("volume"),
            })
            .collect();

        Ok(klines)
    }

    pub async fn save_stock_info(&self, stocks: &[StockInfo]) -> Result<(), sqlx::Error> {
        for stock in stocks {
            sqlx::query(
                "INSERT INTO stock_info (symbol, name, market, industry, list_date)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (symbol) DO UPDATE SET
                   name = EXCLUDED.name,
                   market = EXCLUDED.market,
                   industry = EXCLUDED.industry,
                   list_date = EXCLUDED.list_date",
            )
            .bind(&stock.symbol)
            .bind(&stock.name)
            .bind(stock.market.to_string())
            .bind(&stock.industry)
            .bind(&stock.list_date)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    pub async fn get_stock_list(&self) -> Result<Vec<StockInfo>, sqlx::Error> {
        let rows = sqlx::query(
            "SELECT symbol, name, market, industry, list_date
             FROM stock_info
             ORDER BY symbol ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let stocks = rows
            .iter()
            .map(|row| {
                let market_str: String = row.get("market");
                let market = match market_str.as_str() {
                    "SH" => Market::SH,
                    "SZ" => Market::SZ,
                    _ => Market::SH,
                };
                StockInfo {
                    symbol: row.get("symbol"),
                    name: row.get("name"),
                    market,
                    industry: row.get("industry"),
                    list_date: row.get("list_date"),
                }
            })
            .collect();

        Ok(stocks)
    }
}
