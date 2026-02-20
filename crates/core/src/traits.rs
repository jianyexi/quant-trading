use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::models::{Account, Kline, Order, Position, StockInfo, Tick, TickData, DepthData};
use crate::types::{Signal, TimeFrame};

use crate::models::Portfolio;

#[async_trait]
pub trait DataProvider: Send + Sync {
    async fn fetch_kline(
        &self,
        symbol: &str,
        start: &str,
        end: &str,
        timeframe: TimeFrame,
    ) -> Result<Vec<Kline>>;

    async fn fetch_stock_list(&self) -> Result<Vec<StockInfo>>;

    async fn fetch_realtime_quote(&self, symbol: &str) -> Result<Tick>;
}

pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn on_init(&mut self);
    fn on_bar(&mut self, kline: &Kline) -> Option<Signal>;
    /// Called on each L2 tick (逐笔成交). Default: no-op.
    fn on_tick(&mut self, _tick: &TickData) -> Option<Signal> { None }
    /// Called on each L2 depth update (盘口变化). Default: no-op.
    fn on_depth(&mut self, _depth: &DepthData) -> Option<Signal> { None }
    fn on_stop(&mut self);
}

#[async_trait]
pub trait Broker: Send + Sync {
    async fn submit_order(&self, order: &Order) -> Result<Order>;
    async fn cancel_order(&self, order_id: Uuid) -> Result<()>;
    async fn get_positions(&self) -> Result<Vec<Position>>;
    async fn get_account(&self) -> Result<Account>;
    /// Downcast support for concrete broker type access
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait RiskManager: Send + Sync {
    fn check_order(&self, order: &Order, portfolio: &Portfolio) -> Result<()>;
    fn calculate_position_size(&self, signal: &Signal, portfolio: &Portfolio) -> f64;
}
