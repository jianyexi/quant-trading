use std::sync::Arc;
use tokio::sync::Mutex;
use quant_broker::engine::TradingEngine;
use quant_broker::journal::JournalStore;
use quant_config::AppConfig;
use quant_strategy::sentiment::SentimentStore;
use quant_strategy::collector::SentimentCollector;
use crate::log_store::LogStore;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub engine: Arc<Mutex<Option<TradingEngine>>>,
    pub sentiment_store: SentimentStore,
    pub sentiment_collector: Arc<Mutex<SentimentCollector>>,
    pub journal: Arc<JournalStore>,
    pub log_store: Arc<LogStore>,
}
