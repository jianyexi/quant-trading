//! LLM-based trading signal strategy.
//!
//! Mirrors `MlFactorStrategy`: buffers Kline bars, computes technical indicators,
//! sends market context to `llm_signal_serve.py` via HTTP, and converts the LLM
//! response into a `Signal(Buy/Sell/Hold)` for the trading engine.
//!
//! **Performance note:** LLM inference is slow (1-30s per call), so this strategy
//! uses a background thread for HTTP requests. `on_bar()` returns the latest cached
//! signal instantly, while the background thread fetches new signals asynchronously.
//! This avoids blocking the trading engine's event loop.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::sync::mpsc;

use quant_core::models::Kline;
use quant_core::traits::Strategy;
use quant_core::types::Signal;

use crate::ml_factor::{FEATURE_NAMES, NUM_FEATURES};

/// Max response body size (10 KB) to prevent memory exhaustion from malformed server.
const MAX_RESPONSE_SIZE: usize = 10_000;

// ── Configuration ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LlmSignalConfig {
    /// URL of the llm_signal_serve.py signal endpoint.
    pub signal_url: String,
    /// Minimum confidence from LLM to trigger a BUY signal.
    pub buy_threshold: f64,
    /// Minimum confidence from LLM to trigger a SELL signal.
    pub sell_threshold: f64,
    /// Number of recent bars to include in the prompt context.
    pub context_bars: usize,
    /// Bars to buffer before first signal (needs 60 for MA60).
    pub lookback: usize,
    /// HTTP timeout in seconds for each signal request.
    pub timeout_secs: u64,
}

impl Default for LlmSignalConfig {
    fn default() -> Self {
        Self {
            signal_url: "http://127.0.0.1:18095/signal".to_string(),
            buy_threshold: 0.65,
            sell_threshold: 0.65,
            context_bars: 10,
            lookback: 61,
            timeout_secs: 10,
        }
    }
}

// ── Cached signal from background inference ─────────────────────

#[derive(Debug, Clone)]
struct CachedSignal {
    action: String,
    confidence: f64,
}

/// Request payload sent to the background inference thread.
struct InferenceRequest {
    body: serde_json::Value,
}

// ── Strategy ────────────────────────────────────────────────────

pub struct LlmSignalStrategy {
    cfg: LlmSignalConfig,
    bar_buffer: VecDeque<Kline>,
    incr_engine: crate::fast_factors::IncrementalFactorEngine,
    prev_action: Option<String>,
    /// Bars to skip after a signal to avoid signal spam.
    cooldown: usize,
    /// Whether the LLM server is reachable.
    server_available: bool,
    /// Bars since the last health check (retries every 100 bars).
    bars_since_check: usize,
    /// Latest signal from the background inference thread.
    cached_signal: Arc<Mutex<Option<CachedSignal>>>,
    /// Channel to send requests to the background thread.
    request_tx: mpsc::Sender<InferenceRequest>,
    /// Whether the background thread has a pending request.
    pending: Arc<Mutex<bool>>,
}

impl LlmSignalStrategy {
    pub fn new(cfg: LlmSignalConfig) -> Self {
        let http_agent = ureq::AgentBuilder::new()
            .timeout_read(std::time::Duration::from_secs(cfg.timeout_secs))
            .timeout_write(std::time::Duration::from_secs(5))
            .build();

        let server_available = Self::check_health_sync(&http_agent, &cfg.signal_url);
        if !server_available {
            tracing::warn!(
                url = %cfg.signal_url,
                "LLM signal server not reachable — will retry every 100 bars"
            );
        }

        let cached_signal: Arc<Mutex<Option<CachedSignal>>> = Arc::new(Mutex::new(None));
        let pending: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
        let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>();

        // Spawn background inference thread
        {
            let cached = Arc::clone(&cached_signal);
            let pending_flag = Arc::clone(&pending);
            let signal_url = cfg.signal_url.clone();
            let agent = http_agent;
            std::thread::Builder::new()
                .name("llm-signal-infer".into())
                .spawn(move || {
                    Self::inference_loop(agent, &signal_url, request_rx, cached, pending_flag);
                })
                .expect("Failed to spawn LLM inference thread");
        }

        Self {
            cfg,
            bar_buffer: VecDeque::with_capacity(80),
            incr_engine: crate::fast_factors::IncrementalFactorEngine::new(),
            prev_action: None,
            cooldown: 0,
            server_available,
            bars_since_check: 0,
            cached_signal,
            request_tx,
            pending,
        }
    }

    fn check_health_sync(agent: &ureq::Agent, signal_url: &str) -> bool {
        let health_url = signal_url.replace("/signal", "/health");
        matches!(agent.get(&health_url).call(), Ok(resp) if resp.status() == 200)
    }

    /// Background thread: reads requests from channel, calls server, caches result.
    fn inference_loop(
        agent: ureq::Agent,
        signal_url: &str,
        rx: mpsc::Receiver<InferenceRequest>,
        cached: Arc<Mutex<Option<CachedSignal>>>,
        pending: Arc<Mutex<bool>>,
    ) {
        while let Ok(req) = rx.recv() {
            let result = match agent.post(signal_url).send_json(&req.body) {
                Ok(resp) => {
                    let text = match resp.into_string() {
                        Ok(t) if t.len() <= MAX_RESPONSE_SIZE => t,
                        Ok(t) => {
                            tracing::warn!("LLM response too large: {} bytes", t.len());
                            *pending.lock().unwrap() = false;
                            continue;
                        }
                        Err(_) => {
                            *pending.lock().unwrap() = false;
                            continue;
                        }
                    };
                    match serde_json::from_str::<serde_json::Value>(&text) {
                        Ok(json) => {
                            let action = json
                                .get("action")
                                .and_then(|v| v.as_str())
                                .unwrap_or("hold")
                                .to_lowercase();
                            let confidence = json
                                .get("confidence")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0);
                            Some(CachedSignal { action, confidence })
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse LLM response: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("LLM signal request failed: {}", e);
                    None
                }
            };

            if let Some(sig) = result {
                *cached.lock().unwrap() = Some(sig);
            }
            *pending.lock().unwrap() = false;
        }
    }

    /// Build the JSON request body for the signal server.
    fn build_request_body(
        &self,
        kline: &Kline,
        features: &[f32; NUM_FEATURES],
    ) -> serde_json::Value {
        let bars: Vec<serde_json::Value> = self
            .bar_buffer
            .iter()
            .rev()
            .take(self.cfg.context_bars)
            .rev()
            .map(|b| {
                serde_json::json!({
                    "datetime": b.datetime.format("%Y-%m-%d %H:%M:%S").to_string(),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                })
            })
            .collect();

        let mut indicators = serde_json::Map::new();
        for (i, &val) in features.iter().enumerate() {
            if i < FEATURE_NAMES.len() {
                indicators.insert(FEATURE_NAMES[i].to_string(), (val as f64).into());
            }
        }

        serde_json::json!({
            "symbol": kline.symbol,
            "bars": bars,
            "indicators": indicators,
        })
    }

    /// Submit an inference request to the background thread (non-blocking).
    fn submit_request(&self, kline: &Kline, features: &[f32; NUM_FEATURES]) {
        let mut is_pending = self.pending.lock().unwrap();
        if *is_pending {
            return; // Previous request still in flight
        }
        *is_pending = true;
        let body = self.build_request_body(kline, features);
        let _ = self.request_tx.send(InferenceRequest { body });
    }

    /// Read the latest cached signal (non-blocking).
    fn read_cached_signal(&self) -> Option<CachedSignal> {
        self.cached_signal.lock().unwrap().clone()
    }
}

impl Strategy for LlmSignalStrategy {
    fn name(&self) -> &str {
        "llm_signal"
    }

    fn on_init(&mut self) {
        tracing::info!(url = %self.cfg.signal_url, "LlmSignalStrategy initialised");
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        // Buffer bars in ring buffer
        self.bar_buffer.push_back(kline.clone());
        if self.bar_buffer.len() > self.cfg.lookback + 10 {
            self.bar_buffer.pop_front();
        }

        // Compute features via incremental engine (returns None until enough data)
        let features = self.incr_engine.update(kline)?;

        // Cooldown after signal
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return None;
        }

        // Periodic server availability check
        if !self.server_available {
            self.bars_since_check += 1;
            if self.bars_since_check >= 100 {
                self.bars_since_check = 0;
                // Submit a request to test connectivity
                self.submit_request(kline, &features);
                if self.read_cached_signal().is_some() {
                    self.server_available = true;
                    tracing::info!("LLM signal server reconnected");
                }
            }
            return None;
        }

        // Submit async inference request (non-blocking)
        self.submit_request(kline, &features);

        // Read the latest cached signal (non-blocking)
        let cached = match self.read_cached_signal() {
            Some(c) => c,
            None => return None,
        };

        tracing::debug!(
            symbol = %kline.symbol,
            action = %cached.action,
            confidence = %cached.confidence,
            "LLM signal (cached)"
        );

        // Generate signal based on action + confidence thresholds
        let signal = match cached.action.as_str() {
            "buy"
                if cached.confidence >= self.cfg.buy_threshold
                    && self.prev_action.as_deref() != Some("buy") =>
            {
                self.cooldown = 5;
                Some(Signal::buy(&kline.symbol, cached.confidence, kline.datetime))
            }
            "sell"
                if cached.confidence >= self.cfg.sell_threshold
                    && self.prev_action.as_deref() != Some("sell") =>
            {
                self.cooldown = 5;
                Some(Signal::sell(&kline.symbol, cached.confidence, kline.datetime))
            }
            _ => None,
        };

        if signal.is_some() {
            self.prev_action = Some(cached.action);
        }

        signal
    }

    fn on_stop(&mut self) {
        tracing::info!("LlmSignalStrategy stopped");
    }
}
