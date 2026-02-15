use std::sync::Arc;
use tokio::sync::Mutex;
use quant_broker::engine::TradingEngine;
use quant_broker::journal::JournalStore;
use quant_config::AppConfig;
use quant_strategy::sentiment::SentimentStore;

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub engine: Arc<Mutex<Option<TradingEngine>>>,
    pub sentiment_store: SentimentStore,
    pub journal: Arc<JournalStore>,
}
