use std::sync::Arc;
use tokio::sync::Mutex;
use quant_broker::engine::TradingEngine;
use quant_broker::journal::JournalStore;
use quant_broker::notifier::Notifier;
use quant_config::AppConfig;
use quant_strategy::sentiment::SentimentStore;
use quant_strategy::collector::SentimentCollector;
use crate::log_store::LogStore;
use crate::task_store::TaskStore;

/// Tracks a managed sidecar subprocess (e.g. ml_serve.py).
pub struct ManagedProcess {
    pub name: String,
    pub child: std::process::Child,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub args: Vec<String>,
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub engine: Arc<Mutex<Option<TradingEngine>>>,
    pub sentiment_store: SentimentStore,
    pub sentiment_collector: Arc<Mutex<SentimentCollector>>,
    pub journal: Arc<JournalStore>,
    pub log_store: Arc<LogStore>,
    pub notifier: Arc<Notifier>,
    pub db: Option<sqlx::PgPool>,
    pub task_store: Arc<TaskStore>,
    /// Managed sidecar processes (ml_serve, qmt_bridge, etc.)
    pub managed_processes: Arc<Mutex<std::collections::HashMap<String, ManagedProcess>>>,
}
