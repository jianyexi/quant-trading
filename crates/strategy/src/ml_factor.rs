// Machine Learning Factor Extraction Strategy
//
// Architecture:
//   Kline bars → Feature Engineering (24 features in Rust) → ML Inference (Python sidecar with GPU) → Signal
//
// The Python sidecar (`ml_models/ml_serve.py`) runs a Flask HTTP server that:
//   1. Loads an ONNX/LightGBM/PyTorch model with GPU acceleration (CUDA)
//   2. Accepts feature vectors via POST /predict
//   3. Returns prediction probability
//
// When the Python sidecar is not available, falls back to rule-based scoring in Rust.

use std::sync::Mutex;

use quant_core::models::Kline;
use quant_core::traits::Strategy;
use quant_core::types::Signal;

// ── Feature Engineering ─────────────────────────────────────────────

/// Number of features produced by compute_features
pub const NUM_FEATURES: usize = 24;

/// Feature names in the same order as the feature vector
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "volatility_5d", "volatility_20d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma60_ratio", "ma5_ma20_cross",
    "rsi_14",
    "macd_histogram", "macd_normalized",
    "volume_ratio_5_20", "volume_change",
    "price_position",
    "gap",
    "intraday_range",
    "upper_shadow_ratio", "lower_shadow_ratio",
    "bollinger_pctb",
    "body_ratio",
    "close_to_open",
];

/// Computes feature vector from a window of Kline bars.
/// Requires at least 60 bars for MA(60). Returns None if insufficient data.
pub fn compute_features(bars: &[Kline]) -> Option<[f32; NUM_FEATURES]> {
    if bars.len() < 61 {
        return None;
    }

    let n = bars.len();
    let cur = &bars[n - 1];
    let closes: Vec<f64> = bars.iter().map(|k| k.close).collect();
    let highs: Vec<f64> = bars.iter().map(|k| k.high).collect();
    let lows: Vec<f64> = bars.iter().map(|k| k.low).collect();
    let opens: Vec<f64> = bars.iter().map(|k| k.open).collect();
    let volumes: Vec<f64> = bars.iter().map(|k| k.volume).collect();

    let c = cur.close;
    let o = cur.open;
    let h = cur.high;
    let l = cur.low;

    // Returns
    let ret_1d = safe_div(c - closes[n - 2], closes[n - 2]);
    let ret_5d = if n >= 6 { safe_div(c - closes[n - 6], closes[n - 6]) } else { 0.0 };
    let ret_10d = if n >= 11 { safe_div(c - closes[n - 11], closes[n - 11]) } else { 0.0 };
    let ret_20d = if n >= 21 { safe_div(c - closes[n - 21], closes[n - 21]) } else { 0.0 };

    // Volatility
    let volatility_5d = std_of_returns(&closes[n.saturating_sub(6)..n]);
    let volatility_20d = std_of_returns(&closes[n.saturating_sub(21)..n]);

    // Moving averages
    let ma5 = mean(&closes[n - 5..n]);
    let ma10 = mean(&closes[n - 10..n]);
    let ma20 = mean(&closes[n - 20..n]);
    let ma60 = mean(&closes[n - 60..n]);
    let ma5_ratio = safe_div(c - ma5, ma5);
    let ma10_ratio = safe_div(c - ma10, ma10);
    let ma20_ratio = safe_div(c - ma20, ma20);
    let ma60_ratio = safe_div(c - ma60, ma60);
    let ma5_ma20_cross = safe_div(ma5 - ma20, ma20);

    // RSI(14)
    let rsi_14 = compute_rsi(&closes[n.saturating_sub(15)..n], 14);

    // MACD
    let (macd_hist, macd_norm) = compute_macd_features(&closes, c);

    // Volume features
    let vol_ma5 = mean(&volumes[n - 5..n]);
    let vol_ma20 = mean(&volumes[n - 20..n]);
    let volume_ratio_5_20 = safe_div(vol_ma5, vol_ma20);
    let volume_change = if volumes[n - 2] > 0.0 {
        safe_div(volumes[n - 1] - volumes[n - 2], volumes[n - 2])
    } else { 0.0 };

    // Price range position (20-day)
    let high_20 = highs[n - 20..n].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let low_20 = lows[n - 20..n].iter().cloned().fold(f64::INFINITY, f64::min);
    let range_20 = high_20 - low_20;
    let price_position = if range_20 > f64::EPSILON { (c - low_20) / range_20 } else { 0.5 };

    // Gap
    let gap = safe_div(o - closes[n - 2], closes[n - 2]);

    // Intraday range
    let intraday_range = if o > f64::EPSILON { (h - l) / o } else { 0.0 };

    // Shadow ratios
    let total_range = h - l;
    let body_top = c.max(o);
    let body_bottom = c.min(o);
    let upper_shadow = if total_range > f64::EPSILON { (h - body_top) / total_range } else { 0.0 };
    let lower_shadow = if total_range > f64::EPSILON { (body_bottom - l) / total_range } else { 0.0 };

    // Bollinger %B
    let bb_std = std_dev(&closes[n - 20..n]);
    let bb_upper = ma20 + 2.0 * bb_std;
    let bb_lower = ma20 - 2.0 * bb_std;
    let bb_range = bb_upper - bb_lower;
    let bollinger_pctb = if bb_range > f64::EPSILON { (c - bb_lower) / bb_range } else { 0.5 };

    // Body ratio
    let body_ratio = if total_range > f64::EPSILON { (c - o).abs() / total_range } else { 0.0 };

    // Close to open
    let close_to_open = safe_div(c - o, o);

    Some([
        ret_1d as f32, ret_5d as f32, ret_10d as f32, ret_20d as f32,
        volatility_5d as f32, volatility_20d as f32,
        ma5_ratio as f32, ma10_ratio as f32, ma20_ratio as f32, ma60_ratio as f32, ma5_ma20_cross as f32,
        rsi_14 as f32,
        macd_hist as f32, macd_norm as f32,
        volume_ratio_5_20 as f32, volume_change as f32,
        price_position as f32,
        gap as f32,
        intraday_range as f32,
        upper_shadow as f32, lower_shadow as f32,
        bollinger_pctb as f32,
        body_ratio as f32,
        close_to_open as f32,
    ])
}

// ── Math helpers ────────────────────────────────────────────────────

fn safe_div(a: f64, b: f64) -> f64 {
    if b.abs() < f64::EPSILON { 0.0 } else { a / b }
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_dev(vals: &[f64]) -> f64 {
    if vals.len() < 2 { return 0.0; }
    let m = mean(vals);
    let variance = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    variance.sqrt()
}

fn std_of_returns(closes: &[f64]) -> f64 {
    if closes.len() < 3 { return 0.0; }
    let rets: Vec<f64> = closes.windows(2)
        .map(|w| if w[0].abs() > f64::EPSILON { w[1] / w[0] - 1.0 } else { 0.0 })
        .collect();
    std_dev(&rets)
}

fn compute_rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 { return 50.0; }
    let mut gains = 0.0;
    let mut losses = 0.0;
    for w in closes[closes.len() - period - 1..].windows(2) {
        let delta = w[1] - w[0];
        if delta > 0.0 { gains += delta; } else { losses -= delta; }
    }
    gains /= period as f64;
    losses /= period as f64;
    if losses < f64::EPSILON { return 100.0; }
    100.0 - 100.0 / (1.0 + gains / losses)
}

fn compute_macd_features(closes: &[f64], current_close: f64) -> (f64, f64) {
    // Simplified MACD: use recent closes for EMA approximation
    let n = closes.len();
    if n < 26 { return (0.0, 0.0); }

    let ema12 = ema_approx(&closes[n.saturating_sub(30)..n], 12);
    let ema26 = ema_approx(&closes[n.saturating_sub(40)..n], 26);
    let macd_line = ema12 - ema26;

    // Signal line would need more history; approximate
    let hist = macd_line * 0.5; // simplified
    let norm = safe_div(hist, current_close);
    (hist, norm)
}

fn ema_approx(vals: &[f64], period: usize) -> f64 {
    if vals.is_empty() { return 0.0; }
    let k = 2.0 / (period as f64 + 1.0);
    let mut ema = vals[0];
    for &v in &vals[1..] {
        ema = v * k + ema * (1.0 - k);
    }
    ema
}

// ── ML Inference via Python Sidecar ─────────────────────────────────

/// HTTP client for the ML inference Python sidecar.
/// Runs on http://127.0.0.1:18091 by default.
pub struct MlInferenceClient {
    bridge_url: String,
    client: reqwest::blocking::Client,
    /// Cache: is the bridge available?
    available: Mutex<Option<bool>>,
}

impl MlInferenceClient {
    pub fn new(bridge_url: &str) -> Self {
        Self {
            bridge_url: bridge_url.trim_end_matches('/').to_string(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
            available: Mutex::new(None),
        }
    }

    /// Check if the ML bridge is available.
    pub fn check_health(&self) -> bool {
        match self.client.get(format!("{}/health", self.bridge_url)).send() {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Run prediction on a feature vector. Returns probability [0, 1].
    pub fn predict(&self, features: &[f32; NUM_FEATURES]) -> Result<f64, String> {
        // Check cached availability
        {
            let cached = self.available.lock().unwrap();
            if let Some(false) = *cached {
                return Err("ML bridge not available (cached)".to_string());
            }
        }

        let body = serde_json::json!({
            "features": features.to_vec(),
            "feature_names": FEATURE_NAMES.to_vec(),
        });

        match self.client
            .post(format!("{}/predict", self.bridge_url))
            .json(&body)
            .send()
        {
            Ok(resp) if resp.status().is_success() => {
                let result: serde_json::Value = resp.json()
                    .map_err(|e| format!("Parse error: {}", e))?;
                let prob = result["probability"].as_f64().unwrap_or(0.5);
                *self.available.lock().unwrap() = Some(true);
                Ok(prob)
            }
            Ok(resp) => {
                Err(format!("ML bridge error: HTTP {}", resp.status()))
            }
            Err(e) => {
                *self.available.lock().unwrap() = Some(false);
                Err(format!("ML bridge unreachable: {}", e))
            }
        }
    }

    /// Get model info from the bridge.
    pub fn model_info(&self) -> Result<serde_json::Value, String> {
        self.client
            .get(format!("{}/model_info", self.bridge_url))
            .send()
            .map_err(|e| format!("Request error: {}", e))?
            .json()
            .map_err(|e| format!("Parse error: {}", e))
    }
}

// ── Fallback Rule-Based Scoring ─────────────────────────────────────

/// When no ONNX model is available, use a rule-based scoring of the features.
fn fallback_score(features: &[f32; NUM_FEATURES]) -> f64 {
    let mut score = 0.5; // neutral baseline

    // Index mapping matches FEATURE_NAMES
    let ret_5d = features[1] as f64;
    let ret_20d = features[3] as f64;
    let ma5_ratio = features[6] as f64;
    let ma5_ma20_cross = features[10] as f64;
    let rsi = features[11] as f64;
    let volume_ratio = features[14] as f64;
    let price_position = features[16] as f64;
    let bollinger = features[21] as f64;

    // Trend: MA5 > MA20, price above MA5
    if ma5_ma20_cross > 0.005 { score += 0.08; }
    if ma5_ma20_cross < -0.005 { score -= 0.08; }
    if ma5_ratio > 0.01 { score += 0.05; }
    if ma5_ratio < -0.01 { score -= 0.05; }

    // Momentum: 5-day and 20-day returns
    score += (ret_5d * 2.0).max(-0.1).min(0.1);
    score += (ret_20d * 1.0).max(-0.05).min(0.05);

    // RSI
    if rsi < 30.0 { score += 0.08; } // oversold → buy signal
    else if rsi > 70.0 { score -= 0.08; } // overbought → sell signal
    else { score += ((rsi - 50.0) / 200.0).max(-0.03).min(0.03); }

    // Volume
    if volume_ratio > 1.5 { score += 0.05 * ret_5d.signum(); }

    // Price position
    if price_position > 0.9 { score += 0.03; } // near high → momentum
    if price_position < 0.1 { score += 0.05; } // near low → potential reversal

    // Bollinger
    if bollinger < 0.1 { score += 0.05; } // near lower band
    if bollinger > 0.9 { score -= 0.05; } // near upper band

    score.max(0.0).min(1.0)
}

// ── ML Factor Strategy ──────────────────────────────────────────────

/// Configuration for the ML Factor strategy.
#[derive(Debug, Clone)]
pub struct MlFactorConfig {
    /// URL of the ML inference Python sidecar (e.g. http://127.0.0.1:18091)
    pub bridge_url: String,
    /// Minimum prediction probability to trigger BUY signal
    pub buy_threshold: f64,
    /// Maximum prediction probability to trigger SELL signal
    pub sell_threshold: f64,
    /// Number of bars to buffer before generating signals
    pub lookback: usize,
}

impl Default for MlFactorConfig {
    fn default() -> Self {
        Self {
            bridge_url: "http://127.0.0.1:18091".to_string(),
            buy_threshold: 0.60,
            sell_threshold: 0.35,
            lookback: 61, // need 60 bars for MA(60) + 1 for current
        }
    }
}

/// ML Factor Extraction Strategy.
///
/// Computes 24 factor features from a rolling window of Kline bars,
/// sends to Python sidecar for GPU-accelerated model inference,
/// and generates trading signals based on predicted probability.
/// Falls back to rule-based scoring when the sidecar is unavailable.
pub struct MlFactorStrategy {
    cfg: MlFactorConfig,
    client: Option<MlInferenceClient>,
    bar_buffer: Vec<Kline>,
    prev_prediction: Option<f64>,
    using_fallback: bool,
}

impl MlFactorStrategy {
    pub fn new(cfg: MlFactorConfig) -> Self {
        let client = MlInferenceClient::new(&cfg.bridge_url);
        let available = client.check_health();
        if available {
            tracing::info!("ML Factor Strategy: Python bridge connected at {}", cfg.bridge_url);
        } else {
            tracing::warn!("ML Factor Strategy: Python bridge not available at {}, using fallback scoring", cfg.bridge_url);
        }

        Self {
            client: Some(client),
            bar_buffer: Vec::with_capacity(70),
            prev_prediction: None,
            using_fallback: !available,
            cfg,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(MlFactorConfig::default())
    }

    pub fn with_bridge(bridge_url: &str) -> Self {
        Self::new(MlFactorConfig {
            bridge_url: bridge_url.to_string(),
            ..Default::default()
        })
    }

    /// Get the predicted probability for the current bar.
    fn predict(&self, features: &[f32; NUM_FEATURES]) -> f64 {
        if !self.using_fallback {
            if let Some(client) = &self.client {
                match client.predict(features) {
                    Ok(prob) => return prob,
                    Err(e) => {
                        tracing::warn!("ML inference failed: {}, using fallback", e);
                    }
                }
            }
        }
        fallback_score(features)
    }

    pub fn is_using_fallback(&self) -> bool {
        self.using_fallback
    }
}

impl Strategy for MlFactorStrategy {
    fn name(&self) -> &str {
        "MlFactor"
    }

    fn on_init(&mut self) {
        self.bar_buffer.clear();
        self.prev_prediction = None;
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        // Buffer bars
        self.bar_buffer.push(kline.clone());
        if self.bar_buffer.len() > self.cfg.lookback + 10 {
            self.bar_buffer.remove(0);
        }

        // Need enough bars for feature computation
        if self.bar_buffer.len() < self.cfg.lookback {
            return None;
        }

        // Compute features
        let features = compute_features(&self.bar_buffer)?;

        // Run prediction
        let prediction = self.predict(&features);

        // Generate signal on threshold crossings (avoids repeated signals)
        let signal = match self.prev_prediction {
            Some(prev) => {
                if prediction > self.cfg.buy_threshold && prev <= self.cfg.buy_threshold {
                    let confidence = ((prediction - self.cfg.buy_threshold) / (1.0 - self.cfg.buy_threshold))
                        .max(0.1).min(1.0);
                    Some(Signal::buy(&kline.symbol, confidence, kline.datetime))
                } else if prediction < self.cfg.sell_threshold && prev >= self.cfg.sell_threshold {
                    let confidence = ((self.cfg.sell_threshold - prediction) / self.cfg.sell_threshold)
                        .max(0.1).min(1.0);
                    Some(Signal::sell(&kline.symbol, confidence, kline.datetime))
                } else {
                    None
                }
            }
            None => {
                // First prediction — signal if strong
                if prediction > self.cfg.buy_threshold + 0.1 {
                    Some(Signal::buy(&kline.symbol, prediction.min(1.0), kline.datetime))
                } else if prediction < self.cfg.sell_threshold - 0.1 {
                    Some(Signal::sell(&kline.symbol, (1.0 - prediction).min(1.0), kline.datetime))
                } else {
                    None
                }
            }
        };

        self.prev_prediction = Some(prediction);
        signal
    }

    fn on_stop(&mut self) {
        self.bar_buffer.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn make_bar(close: f64, high: f64, low: f64, open: f64, volume: f64, day: u32) -> Kline {
        Kline {
            symbol: "TEST".to_string(),
            datetime: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()
                .and_hms_opt(15, 0, 0).unwrap()
                + chrono::Duration::days(day as i64),
            open, high, low, close, volume,
        }
    }

    fn generate_bars(n: usize, base_price: f64, trend: f64) -> Vec<Kline> {
        let mut bars = Vec::with_capacity(n);
        let mut price = base_price;
        for i in 0..n {
            price *= 1.0 + trend + (i as f64 * 0.001).sin() * 0.01;
            let h = price * 1.01;
            let l = price * 0.99;
            let o = price * (1.0 + 0.002 * (i % 3) as f64 - 0.003);
            let vol = 1e7 * (1.0 + (i as f64 * 0.1).sin() * 0.3);
            bars.push(make_bar(price, h, l, o, vol, i as u32));
        }
        bars
    }

    #[test]
    fn test_feature_computation_requires_warmup() {
        let bars = generate_bars(30, 100.0, 0.001);
        assert!(compute_features(&bars).is_none());
    }

    #[test]
    fn test_feature_computation_with_sufficient_bars() {
        let bars = generate_bars(70, 100.0, 0.001);
        let features = compute_features(&bars);
        assert!(features.is_some());
        let f = features.unwrap();
        assert_eq!(f.len(), NUM_FEATURES);
        // All values should be finite
        for (i, &v) in f.iter().enumerate() {
            assert!(v.is_finite(), "Feature {} ({}) is not finite: {}", i, FEATURE_NAMES[i], v);
        }
    }

    #[test]
    fn test_fallback_scoring() {
        let bars = generate_bars(70, 100.0, 0.005); // uptrend
        let features = compute_features(&bars).unwrap();
        let score = fallback_score(&features);
        assert!(score >= 0.0 && score <= 1.0, "Score out of range: {}", score);
        // Uptrend should produce score > 0.5
        assert!(score > 0.45, "Uptrend should produce bullish score, got {}", score);
    }

    #[test]
    fn test_fallback_scoring_downtrend() {
        let bars = generate_bars(70, 100.0, -0.005); // downtrend
        let features = compute_features(&bars).unwrap();
        let score = fallback_score(&features);
        assert!(score >= 0.0 && score <= 1.0);
        // Downtrend should produce score < 0.5
        assert!(score < 0.55, "Downtrend should produce bearish score, got {}", score);
    }

    #[test]
    fn test_ml_strategy_needs_warmup() {
        let mut strat = MlFactorStrategy::with_defaults();
        strat.on_init();
        assert!(strat.is_using_fallback());

        let bars = generate_bars(50, 100.0, 0.001);
        for bar in &bars {
            assert!(strat.on_bar(bar).is_none());
        }
    }

    #[test]
    fn test_ml_strategy_generates_signal_uptrend() {
        let mut strat = MlFactorStrategy::new(MlFactorConfig {
            buy_threshold: 0.52,
            sell_threshold: 0.45,
            ..Default::default()
        });
        strat.on_init();

        // Strong uptrend should eventually trigger a buy
        let bars = generate_bars(100, 100.0, 0.008);
        let mut signals = Vec::new();
        for bar in &bars {
            if let Some(sig) = strat.on_bar(bar) {
                signals.push(sig);
            }
        }
        // Should have at least one signal
        assert!(!signals.is_empty(), "Expected signals from uptrend data");
        assert!(signals.iter().any(|s| s.is_buy()), "Expected at least one buy signal");
    }

    #[test]
    fn test_ml_strategy_generates_signal_downtrend() {
        let mut strat = MlFactorStrategy::new(MlFactorConfig {
            buy_threshold: 0.55,
            sell_threshold: 0.48,
            ..Default::default()
        });
        strat.on_init();

        // Strong downtrend should eventually trigger a sell
        let bars = generate_bars(100, 100.0, -0.008);
        let mut signals = Vec::new();
        for bar in &bars {
            if let Some(sig) = strat.on_bar(bar) {
                signals.push(sig);
            }
        }
        assert!(!signals.is_empty(), "Expected signals from downtrend data");
        assert!(signals.iter().any(|s| s.is_sell()), "Expected at least one sell signal");
    }

    #[test]
    fn test_feature_names_count() {
        assert_eq!(FEATURE_NAMES.len(), NUM_FEATURES);
    }
}
