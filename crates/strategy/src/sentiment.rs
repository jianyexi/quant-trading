// Sentiment data store and sentiment-aware strategy wrapper

use std::sync::{Arc, RwLock};
use chrono::NaiveDateTime;
use uuid::Uuid;

use quant_core::models::{Kline, SentimentItem, SentimentLevel, SentimentSummary};
use quant_core::traits::Strategy;
use quant_core::types::Signal;

// ── Sentiment Store ─────────────────────────────────────────────────

/// Thread-safe in-memory sentiment data store.
#[derive(Debug, Clone)]
pub struct SentimentStore {
    items: Arc<RwLock<Vec<SentimentItem>>>,
}

impl SentimentStore {
    pub fn new() -> Self {
        Self {
            items: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add a sentiment item to the store.
    pub fn add(&self, item: SentimentItem) {
        let mut items = self.items.write().unwrap();
        items.push(item);
    }

    /// Submit a new sentiment item with auto-generated id and created_at.
    pub fn submit(
        &self,
        symbol: &str,
        source: &str,
        title: &str,
        content: &str,
        sentiment_score: f64,
        published_at: NaiveDateTime,
    ) -> SentimentItem {
        let item = SentimentItem {
            id: Uuid::new_v4(),
            symbol: symbol.to_string(),
            source: source.to_string(),
            title: title.to_string(),
            content: content.to_string(),
            sentiment_score: sentiment_score.max(-1.0).min(1.0),
            published_at,
            created_at: chrono::Utc::now().naive_utc(),
        };
        self.add(item.clone());
        item
    }

    /// Query sentiment items for a given symbol, optionally filtered by date range.
    pub fn query_by_symbol(
        &self,
        symbol: &str,
        start: Option<NaiveDateTime>,
        end: Option<NaiveDateTime>,
        limit: Option<usize>,
    ) -> Vec<SentimentItem> {
        let items = self.items.read().unwrap();
        let mut result: Vec<_> = items
            .iter()
            .filter(|it| it.symbol == symbol)
            .filter(|it| start.map_or(true, |s| it.published_at >= s))
            .filter(|it| end.map_or(true, |e| it.published_at <= e))
            .cloned()
            .collect();
        result.sort_by(|a, b| b.published_at.cmp(&a.published_at));
        if let Some(n) = limit {
            result.truncate(n);
        }
        result
    }

    /// Get a summary of sentiment for a symbol.
    pub fn summary(&self, symbol: &str) -> SentimentSummary {
        let items = self.items.read().unwrap();
        let relevant: Vec<_> = items.iter().filter(|it| it.symbol == symbol).collect();
        let count = relevant.len();

        if count == 0 {
            return SentimentSummary {
                symbol: symbol.to_string(),
                count: 0,
                avg_score: 0.0,
                level: SentimentLevel::Neutral,
                bullish_count: 0,
                bearish_count: 0,
                neutral_count: 0,
                latest_title: String::new(),
                latest_at: None,
            };
        }

        let total_score: f64 = relevant.iter().map(|it| it.sentiment_score).sum();
        let avg_score = total_score / count as f64;
        let bullish_count = relevant.iter().filter(|it| it.sentiment_score > 0.2).count();
        let bearish_count = relevant.iter().filter(|it| it.sentiment_score < -0.2).count();
        let neutral_count = count - bullish_count - bearish_count;

        let latest = relevant.iter().max_by_key(|it| it.published_at).unwrap();

        SentimentSummary {
            symbol: symbol.to_string(),
            count,
            avg_score,
            level: SentimentLevel::from_score(avg_score),
            bullish_count,
            bearish_count,
            neutral_count,
            latest_title: latest.title.clone(),
            latest_at: Some(latest.published_at),
        }
    }

    /// Get summaries for all symbols that have sentiment data.
    pub fn all_summaries(&self) -> Vec<SentimentSummary> {
        let items = self.items.read().unwrap();
        let mut symbols: Vec<String> = items.iter().map(|it| it.symbol.clone()).collect();
        symbols.sort();
        symbols.dedup();
        drop(items);

        symbols.iter().map(|s| self.summary(s)).collect()
    }

    /// Get aggregate sentiment score for a symbol (recent N items).
    /// Returns None if no data available.
    pub fn aggregate_score(&self, symbol: &str, recent_n: usize) -> Option<f64> {
        let items = self.items.read().unwrap();
        let mut relevant: Vec<_> = items
            .iter()
            .filter(|it| it.symbol == symbol)
            .collect();
        if relevant.is_empty() {
            return None;
        }
        relevant.sort_by(|a, b| b.published_at.cmp(&a.published_at));
        let take_n = relevant.len().min(recent_n);
        let total: f64 = relevant[..take_n].iter().map(|it| it.sentiment_score).sum();
        Some(total / take_n as f64)
    }

    pub fn count(&self) -> usize {
        self.items.read().unwrap().len()
    }
}

impl Default for SentimentStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Sentiment-Aware Strategy Wrapper ────────────────────────────────

/// Wraps any `Strategy` and adjusts signals based on sentiment data.
///
/// Behavior:
/// - On BUY signal: if sentiment is bearish (avg < -0.3), reduce confidence or suppress
/// - On SELL signal: if sentiment is bullish (avg > 0.3), reduce confidence or suppress
/// - If strong sentiment agrees with signal, boost confidence
/// - Adds a "sentiment_boost" factor to signal confidence
pub struct SentimentAwareStrategy {
    inner: Box<dyn Strategy>,
    store: SentimentStore,
    /// How much sentiment can adjust confidence (0.0 to 0.5)
    sentiment_weight: f64,
    /// Min sentiment items required before applying adjustment
    min_items: usize,
    /// How many recent items to aggregate
    recent_n: usize,
}

impl SentimentAwareStrategy {
    pub fn new(
        inner: Box<dyn Strategy>,
        store: SentimentStore,
        sentiment_weight: f64,
        min_items: usize,
    ) -> Self {
        Self {
            inner,
            store,
            sentiment_weight: sentiment_weight.max(0.0).min(0.5),
            min_items,
            recent_n: 20,
        }
    }

    pub fn with_defaults(inner: Box<dyn Strategy>, store: SentimentStore) -> Self {
        Self::new(inner, store, 0.2, 3)
    }
}

impl Strategy for SentimentAwareStrategy {
    fn name(&self) -> &str {
        "SentimentAware"
    }

    fn on_init(&mut self) {
        self.inner.on_init();
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        let base_signal = self.inner.on_bar(kline)?;

        // Get sentiment score
        let sentiment = self.store.aggregate_score(&kline.symbol, self.recent_n);

        match sentiment {
            Some(score) if self.store.query_by_symbol(&kline.symbol, None, None, None).len() >= self.min_items => {
                let mut adjusted = base_signal;

                if adjusted.is_buy() {
                    // Bullish sentiment → boost, bearish → dampen
                    let adjustment = score * self.sentiment_weight;
                    adjusted.confidence = (adjusted.confidence + adjustment).max(0.0).min(1.0);

                    // Strong bearish sentiment suppresses buy signals with low confidence
                    if score < -0.5 && adjusted.confidence < 0.3 {
                        return None; // Suppress weak buy in bearish sentiment
                    }
                } else if adjusted.is_sell() {
                    // Bearish sentiment → boost sell, bullish → dampen
                    let adjustment = -score * self.sentiment_weight;
                    adjusted.confidence = (adjusted.confidence + adjustment).max(0.0).min(1.0);

                    // Strong bullish sentiment suppresses sell signals with low confidence
                    if score > 0.5 && adjusted.confidence < 0.3 {
                        return None; // Suppress weak sell in bullish sentiment
                    }
                }

                Some(adjusted)
            }
            _ => Some(base_signal), // Not enough sentiment data, pass through
        }
    }

    fn on_stop(&mut self) {
        self.inner.on_stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn now() -> NaiveDateTime {
        Utc::now().naive_utc()
    }

    #[test]
    fn test_store_add_and_query() {
        let store = SentimentStore::new();

        store.submit("600519.SH", "新闻", "茅台业绩超预期", "内容...", 0.8, now());
        store.submit("600519.SH", "研报", "机构上调目标价", "内容...", 0.6, now());
        store.submit("000858.SZ", "新闻", "五粮液销量下滑", "内容...", -0.5, now());

        assert_eq!(store.count(), 3);

        let mt_items = store.query_by_symbol("600519.SH", None, None, None);
        assert_eq!(mt_items.len(), 2);

        let wly_items = store.query_by_symbol("000858.SZ", None, None, None);
        assert_eq!(wly_items.len(), 1);
    }

    #[test]
    fn test_store_summary() {
        let store = SentimentStore::new();

        store.submit("600519.SH", "新闻", "利好消息1", "", 0.8, now());
        store.submit("600519.SH", "研报", "利好消息2", "", 0.6, now());
        store.submit("600519.SH", "社交", "中性评论", "", 0.0, now());

        let summary = store.summary("600519.SH");
        assert_eq!(summary.count, 3);
        assert!((summary.avg_score - 0.4667).abs() < 0.01);
        assert_eq!(summary.bullish_count, 2);
        assert_eq!(summary.neutral_count, 1);
        assert_eq!(summary.bearish_count, 0);
    }

    #[test]
    fn test_aggregate_score() {
        let store = SentimentStore::new();

        store.submit("600519.SH", "src", "t1", "", 0.8, now());
        store.submit("600519.SH", "src", "t2", "", 0.6, now());

        let score = store.aggregate_score("600519.SH", 10);
        assert!(score.is_some());
        assert!((score.unwrap() - 0.7).abs() < 0.01);

        assert!(store.aggregate_score("999999.SH", 10).is_none());
    }

    #[test]
    fn test_sentiment_aware_boosts_buy_in_bullish() {
        let store = SentimentStore::new();
        // Add bullish sentiment
        for _ in 0..5 {
            store.submit("TEST", "src", "bullish", "", 0.7, now());
        }

        // Create a simple strategy that always buys
        struct AlwaysBuy;
        impl Strategy for AlwaysBuy {
            fn name(&self) -> &str { "AlwaysBuy" }
            fn on_init(&mut self) {}
            fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
                Some(Signal::buy(&kline.symbol, 0.5, kline.datetime))
            }
            fn on_stop(&mut self) {}
        }

        let mut strat = SentimentAwareStrategy::with_defaults(Box::new(AlwaysBuy), store);
        strat.on_init();

        let kline = Kline {
            symbol: "TEST".to_string(),
            datetime: now(),
            open: 100.0, high: 101.0, low: 99.0, close: 100.5, volume: 1000.0,
        };

        let signal = strat.on_bar(&kline).unwrap();
        assert!(signal.is_buy());
        assert!(signal.confidence > 0.5); // Boosted by bullish sentiment
    }

    #[test]
    fn test_sentiment_aware_suppresses_buy_in_bearish() {
        let store = SentimentStore::new();
        // Add strongly bearish sentiment
        for _ in 0..5 {
            store.submit("TEST", "src", "bearish", "", -0.8, now());
        }

        // Strategy with low-confidence buy
        struct WeakBuy;
        impl Strategy for WeakBuy {
            fn name(&self) -> &str { "WeakBuy" }
            fn on_init(&mut self) {}
            fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
                Some(Signal::buy(&kline.symbol, 0.2, kline.datetime))
            }
            fn on_stop(&mut self) {}
        }

        let mut strat = SentimentAwareStrategy::with_defaults(Box::new(WeakBuy), store);
        strat.on_init();

        let kline = Kline {
            symbol: "TEST".to_string(),
            datetime: now(),
            open: 100.0, high: 101.0, low: 99.0, close: 100.5, volume: 1000.0,
        };

        let signal = strat.on_bar(&kline);
        assert!(signal.is_none()); // Suppressed due to bearish sentiment + low confidence
    }
}
