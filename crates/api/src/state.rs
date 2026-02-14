use std::sync::Arc;
use tokio::sync::Mutex;
use quant_broker::engine::TradingEngine;
use quant_config::AppConfig;

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub engine: Arc<Mutex<Option<TradingEngine>>>,
}
