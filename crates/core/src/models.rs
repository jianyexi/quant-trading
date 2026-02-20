use std::collections::HashMap;

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{Market, OrderSide, OrderStatus, OrderType};

// ── Market Data ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub symbol: String,
    pub datetime: NaiveDateTime,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub symbol: String,
    pub datetime: NaiveDateTime,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

// ── Level-2 Market Data ──────────────────────────────────────

/// L2 逐笔成交 (tick-by-tick trade)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickData {
    pub symbol: String,
    pub datetime: NaiveDateTime,
    pub price: f64,
    pub volume: f64,
    /// 'B' = buyer-initiated, 'S' = seller-initiated, ' ' = unknown
    pub direction: char,
    /// Sequence number from exchange
    pub seq: u64,
    /// Best bid at time of trade
    pub bid1: f64,
    /// Best ask at time of trade
    pub ask1: f64,
}

/// Single price level in L2 depth (盘口档位)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DepthLevel {
    pub price: f64,
    pub volume: f64,
    /// Number of orders at this level (if available)
    pub order_count: u32,
}

/// L2 depth snapshot — 5/10 档盘口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthData {
    pub symbol: String,
    pub datetime: NaiveDateTime,
    /// Bid levels (index 0 = best bid, highest price)
    pub bids: Vec<DepthLevel>,
    /// Ask levels (index 0 = best ask, lowest price)
    pub asks: Vec<DepthLevel>,
    /// Last trade price
    pub last_price: f64,
    /// Total volume today
    pub total_volume: f64,
    /// Total turnover today
    pub total_turnover: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockInfo {
    pub symbol: String,
    pub name: String,
    pub market: Market,
    pub industry: String,
    pub list_date: String,
}

// ── Sentiment ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SentimentLevel {
    VeryBullish,
    Bullish,
    Neutral,
    Bearish,
    VeryBearish,
}

impl SentimentLevel {
    pub fn from_score(score: f64) -> Self {
        if score > 0.6 { Self::VeryBullish }
        else if score > 0.2 { Self::Bullish }
        else if score > -0.2 { Self::Neutral }
        else if score > -0.6 { Self::Bearish }
        else { Self::VeryBearish }
    }
}

impl std::fmt::Display for SentimentLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VeryBullish => write!(f, "非常看多"),
            Self::Bullish => write!(f, "看多"),
            Self::Neutral => write!(f, "中性"),
            Self::Bearish => write!(f, "看空"),
            Self::VeryBearish => write!(f, "非常看空"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentItem {
    pub id: Uuid,
    pub symbol: String,
    pub source: String,
    pub title: String,
    pub content: String,
    /// Sentiment score in [-1.0, +1.0]: positive = bullish, negative = bearish
    pub sentiment_score: f64,
    pub published_at: NaiveDateTime,
    pub created_at: NaiveDateTime,
}

impl SentimentItem {
    pub fn level(&self) -> SentimentLevel {
        SentimentLevel::from_score(self.sentiment_score)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentSummary {
    pub symbol: String,
    pub count: usize,
    pub avg_score: f64,
    pub level: SentimentLevel,
    pub bullish_count: usize,
    pub bearish_count: usize,
    pub neutral_count: usize,
    pub latest_title: String,
    pub latest_at: Option<NaiveDateTime>,
}

// ── Trading ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: Uuid,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub price: f64,
    pub quantity: f64,
    pub filled_qty: f64,
    pub status: OrderStatus,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: Uuid,
    pub order_id: Uuid,
    pub symbol: String,
    pub side: OrderSide,
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub timestamp: NaiveDateTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_cost: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    /// When the position was first opened
    #[serde(default = "default_entry_time")]
    pub entry_time: NaiveDateTime,
    /// Scaling level: 1 = initial, 2 = added, 3 = full
    #[serde(default = "default_scale_level")]
    pub scale_level: u32,
    /// Target weight in portfolio (0.0 - 1.0), for rebalancing
    #[serde(default)]
    pub target_weight: f64,
}

fn default_entry_time() -> NaiveDateTime {
    chrono::Utc::now().naive_utc()
}

fn default_scale_level() -> u32 {
    1
}

/// A closed position archived for history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedPosition {
    pub symbol: String,
    pub entry_time: NaiveDateTime,
    pub exit_time: NaiveDateTime,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub realized_pnl: f64,
    pub holding_days: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub positions: HashMap<String, Position>,
    pub cash: f64,
    pub total_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: Uuid,
    pub name: String,
    pub portfolio: Portfolio,
    pub initial_capital: f64,
}
