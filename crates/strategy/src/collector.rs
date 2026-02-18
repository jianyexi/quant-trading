// Background sentiment collector service
//
// Periodically fetches financial news for watched symbols,
// analyzes sentiment via LLM, and stores results in SentimentStore.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use quant_config::SentimentConfig;
use quant_data::news::NewsFetcher;
use quant_llm::client::LlmClient;
use quant_llm::sentiment::SentimentAnalyzer;

use crate::sentiment::SentimentStore;

/// Status of the sentiment collector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorStatus {
    pub running: bool,
    pub total_cycles: u64,
    pub total_articles_fetched: u64,
    pub total_articles_analyzed: u64,
    pub total_items_stored: u64,
    pub last_cycle_at: Option<String>,
    pub last_error: Option<String>,
    pub watch_symbols: Vec<String>,
}

/// Shared collector stats, updated by the background task.
#[derive(Debug, Clone)]
pub struct CollectorStats {
    inner: Arc<RwLock<CollectorStatsInner>>,
}

#[derive(Debug)]
struct CollectorStatsInner {
    total_cycles: u64,
    total_articles_fetched: u64,
    total_articles_analyzed: u64,
    total_items_stored: u64,
    last_cycle_at: Option<String>,
    last_error: Option<String>,
}

impl CollectorStats {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(CollectorStatsInner {
                total_cycles: 0,
                total_articles_fetched: 0,
                total_articles_analyzed: 0,
                total_items_stored: 0,
                last_cycle_at: None,
                last_error: None,
            })),
        }
    }

    pub async fn snapshot(&self, running: bool, symbols: &[String]) -> CollectorStatus {
        let inner = self.inner.read().await;
        CollectorStatus {
            running,
            total_cycles: inner.total_cycles,
            total_articles_fetched: inner.total_articles_fetched,
            total_articles_analyzed: inner.total_articles_analyzed,
            total_items_stored: inner.total_items_stored,
            last_cycle_at: inner.last_cycle_at.clone(),
            last_error: inner.last_error.clone(),
            watch_symbols: symbols.to_vec(),
        }
    }
}

/// The sentiment collector handle — start/stop the background collection loop.
#[derive(Clone)]
pub struct SentimentCollector {
    running: Arc<AtomicBool>,
    stats: CollectorStats,
    config: SentimentConfig,
}

impl SentimentCollector {
    pub fn new(config: SentimentConfig) -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            stats: CollectorStats::new(),
            config,
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub async fn status(&self) -> CollectorStatus {
        self.stats
            .snapshot(self.is_running(), &self.config.watch_symbols)
            .await
    }

    /// Start the background collection loop.
    /// Returns Err if already running.
    pub fn start(
        &self,
        store: SentimentStore,
        llm_client: LlmClient,
        akshare_base: String,
    ) -> Result<(), &'static str> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err("Collector is already running");
        }

        let running = self.running.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            info!(
                "Sentiment collector started — interval={}s, symbols={:?}",
                config.collect_interval_secs, config.watch_symbols
            );

            let fetcher = NewsFetcher::new(&akshare_base);
            let analyzer = SentimentAnalyzer::new(llm_client);

            loop {
                if !running.load(Ordering::Relaxed) {
                    break;
                }

                // Run one collection cycle
                let cycle_result = run_cycle(
                    &fetcher,
                    &analyzer,
                    &store,
                    &config,
                    &stats,
                ).await;

                if let Err(e) = cycle_result {
                    error!("Sentiment collection cycle failed: {}", e);
                    let mut inner = stats.inner.write().await;
                    inner.last_error = Some(e.to_string());
                }

                // Sleep until next cycle
                let interval = std::time::Duration::from_secs(config.collect_interval_secs);
                tokio::time::sleep(interval).await;
            }

            info!("Sentiment collector stopped.");
        });

        Ok(())
    }

    /// Stop the background collection loop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Update the watch symbols list.
    pub fn update_symbols(&mut self, symbols: Vec<String>) {
        self.config.watch_symbols = symbols;
    }
}

/// Execute a single collection cycle: fetch news → analyze → store.
async fn run_cycle(
    fetcher: &NewsFetcher,
    analyzer: &SentimentAnalyzer,
    store: &SentimentStore,
    config: &SentimentConfig,
    stats: &CollectorStats,
) -> anyhow::Result<()> {
    info!("Starting sentiment collection cycle for {} symbols", config.watch_symbols.len());

    let mut total_fetched = 0u64;
    let mut total_analyzed = 0u64;
    let mut total_stored = 0u64;
    let mut llm_calls_remaining = config.max_llm_calls_per_cycle;

    for symbol in &config.watch_symbols {
        if llm_calls_remaining == 0 {
            info!("LLM call budget exhausted, stopping cycle");
            break;
        }

        // 1. Fetch news
        let articles = fetcher
            .fetch_news(symbol, &config.news_sources, config.max_news_per_symbol)
            .await;

        total_fetched += articles.len() as u64;

        if articles.is_empty() {
            continue;
        }

        // 2. Prepare batch for LLM analysis
        let batch: Vec<(String, String, String)> = articles
            .iter()
            .map(|a| (a.title.clone(), a.content.clone(), a.symbol.clone()))
            .collect();

        let calls_for_this = batch.len().min(llm_calls_remaining);
        let results = analyzer.analyze_batch(&batch, calls_for_this).await;

        // 3. Store results
        for (idx, result) in results {
            match result {
                Ok(analysis) => {
                    let article = &articles[idx];
                    store.submit(
                        &article.symbol,
                        &article.source,
                        &article.title,
                        &format!("{}\n\nLLM分析: {}", article.content, analysis.reasoning),
                        analysis.score,
                        article.published_at,
                    );
                    total_stored += 1;
                }
                Err(e) => {
                    warn!("Failed to analyze article: {}", e);
                }
            }
            total_analyzed += 1;
        }

        llm_calls_remaining = llm_calls_remaining.saturating_sub(calls_for_this);
    }

    // Update stats
    {
        let mut inner = stats.inner.write().await;
        inner.total_cycles += 1;
        inner.total_articles_fetched += total_fetched;
        inner.total_articles_analyzed += total_analyzed;
        inner.total_items_stored += total_stored;
        inner.last_cycle_at = Some(Utc::now().format("%Y-%m-%d %H:%M:%S").to_string());
        inner.last_error = None;
    }

    info!(
        "Collection cycle complete: fetched={}, analyzed={}, stored={}",
        total_fetched, total_analyzed, total_stored
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_new() {
        let config = SentimentConfig::default();
        let collector = SentimentCollector::new(config);
        assert!(!collector.is_running());
    }

    #[tokio::test]
    async fn test_collector_status() {
        let config = SentimentConfig::default();
        let collector = SentimentCollector::new(config);
        let status = collector.status().await;
        assert!(!status.running);
        assert_eq!(status.total_cycles, 0);
    }
}
