//! LLM-based trading signal strategy.
//!
//! Mirrors `MlFactorStrategy`: buffers Kline bars, computes technical indicators,
//! sends market context to `llm_signal_serve.py` via HTTP, and converts the LLM
//! response into a `Signal(Buy/Sell/Hold)` for the trading engine.

use std::collections::VecDeque;

use quant_core::models::Kline;
use quant_core::traits::Strategy;
use quant_core::types::Signal;

use crate::ml_factor::{FEATURE_NAMES, NUM_FEATURES};

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
            timeout_secs: 30,
        }
    }
}

// ── Strategy ────────────────────────────────────────────────────

pub struct LlmSignalStrategy {
    cfg: LlmSignalConfig,
    http_agent: ureq::Agent,
    bar_buffer: VecDeque<Kline>,
    incr_engine: crate::fast_factors::IncrementalFactorEngine,
    prev_action: Option<String>,
    /// Bars to skip after a signal to avoid signal spam.
    cooldown: usize,
    /// Whether the LLM server is reachable.
    server_available: bool,
    /// Bars since the last health check (retries every 100 bars).
    bars_since_check: usize,
}

impl LlmSignalStrategy {
    pub fn new(cfg: LlmSignalConfig) -> Self {
        let http_agent = ureq::AgentBuilder::new()
            .timeout_read(std::time::Duration::from_secs(cfg.timeout_secs))
            .timeout_write(std::time::Duration::from_secs(10))
            .build();

        let server_available = Self::check_health(&http_agent, &cfg.signal_url);
        if !server_available {
            tracing::warn!(
                url = %cfg.signal_url,
                "LLM signal server not reachable — will retry every 100 bars"
            );
        }

        Self {
            cfg,
            http_agent,
            bar_buffer: VecDeque::with_capacity(80),
            incr_engine: crate::fast_factors::IncrementalFactorEngine::new(),
            prev_action: None,
            cooldown: 0,
            server_available,
            bars_since_check: 0,
        }
    }

    fn check_health(agent: &ureq::Agent, signal_url: &str) -> bool {
        let health_url = signal_url.replace("/signal", "/health");
        matches!(agent.get(&health_url).call(), Ok(resp) if resp.status() == 200)
    }

    /// Send market context to the LLM signal server and parse the response.
    fn call_server(
        &self,
        kline: &Kline,
        features: &[f32; NUM_FEATURES],
    ) -> Option<(String, f64)> {
        // Build recent bars context
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

        // Build indicators map from computed features
        let mut indicators = serde_json::Map::new();
        for (i, &val) in features.iter().enumerate() {
            if i < FEATURE_NAMES.len() {
                indicators.insert(FEATURE_NAMES[i].to_string(), (val as f64).into());
            }
        }

        let body = serde_json::json!({
            "symbol": kline.symbol,
            "bars": bars,
            "indicators": indicators,
        });

        match self.http_agent.post(&self.cfg.signal_url).send_json(&body) {
            Ok(resp) => {
                let text = resp.into_string().ok()?;
                let json: serde_json::Value = serde_json::from_str(&text).ok()?;
                let action = json.get("action")?.as_str()?.to_lowercase();
                let confidence = json.get("confidence")?.as_f64()?;
                tracing::debug!(
                    symbol = %kline.symbol,
                    action = %action,
                    confidence = %confidence,
                    "LLM signal received"
                );
                Some((action, confidence))
            }
            Err(e) => {
                tracing::warn!("LLM signal request failed: {}", e);
                None
            }
        }
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
                self.server_available =
                    Self::check_health(&self.http_agent, &self.cfg.signal_url);
                if self.server_available {
                    tracing::info!("LLM signal server reconnected");
                }
            }
            return None;
        }

        // Call LLM signal server
        let (action, confidence) = match self.call_server(kline, &features) {
            Some(r) => r,
            None => {
                self.server_available = false;
                self.bars_since_check = 0;
                return None;
            }
        };

        // Generate signal based on action + confidence thresholds
        let signal = match action.as_str() {
            "buy"
                if confidence >= self.cfg.buy_threshold
                    && self.prev_action.as_deref() != Some("buy") =>
            {
                self.cooldown = 5;
                Some(Signal::buy(&kline.symbol, confidence, kline.datetime))
            }
            "sell"
                if confidence >= self.cfg.sell_threshold
                    && self.prev_action.as_deref() != Some("sell") =>
            {
                self.cooldown = 5;
                Some(Signal::sell(&kline.symbol, confidence, kline.datetime))
            }
            _ => None,
        };

        if signal.is_some() {
            self.prev_action = Some(action);
        }

        signal
    }

    fn on_stop(&mut self) {
        tracing::info!("LlmSignalStrategy stopped");
    }
}
