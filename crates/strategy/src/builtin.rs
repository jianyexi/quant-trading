// Built-in strategies

use crate::indicators::{SMA, EMA, MACD, RSI, BollingerBands, KDJ};
use quant_core::models::Kline;
use quant_core::traits::Strategy;
use quant_core::types::Signal;


/// Dual Moving Average Crossover Strategy
pub struct DualMaCrossover {
    fast_period: usize,
    slow_period: usize,
    fast_ma: SMA,
    slow_ma: SMA,
    prev_fast: Option<f64>,
    prev_slow: Option<f64>,
}

impl DualMaCrossover {
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            fast_ma: SMA::new(fast_period),
            slow_ma: SMA::new(slow_period),
            prev_fast: None,
            prev_slow: None,
        }
    }
}

impl Strategy for DualMaCrossover {
    fn name(&self) -> &str {
        "DualMaCrossover"
    }

    fn on_init(&mut self) {
        self.fast_ma = SMA::new(self.fast_period);
        self.slow_ma = SMA::new(self.slow_period);
        self.prev_fast = None;
        self.prev_slow = None;
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        let fast_val = self.fast_ma.update(kline.close);
        let slow_val = self.slow_ma.update(kline.close);

        let result = match (fast_val, slow_val, self.prev_fast, self.prev_slow) {
            (Some(fast), Some(slow), Some(pf), Some(ps)) => {
                if pf <= ps && fast > slow {
                    // Golden cross: fast crosses above slow
                    let confidence = (fast - slow) / slow;
                    Some(Signal::buy(&kline.symbol, confidence, kline.datetime))
                } else if pf >= ps && fast < slow {
                    // Death cross: fast crosses below slow
                    let confidence = (slow - fast) / slow;
                    Some(Signal::sell(&kline.symbol, confidence, kline.datetime))
                } else {
                    None
                }
            }
            _ => None,
        };

        self.prev_fast = fast_val;
        self.prev_slow = slow_val;
        result
    }

    fn on_stop(&mut self) {}
}

/// RSI Mean Reversion Strategy
pub struct RsiMeanReversion {
    period: usize,
    overbought: f64,
    oversold: f64,
    rsi: RSI,
}

impl RsiMeanReversion {
    pub fn new(period: usize, overbought: f64, oversold: f64) -> Self {
        Self {
            period,
            overbought,
            oversold,
            rsi: RSI::new(period),
        }
    }
}

impl Strategy for RsiMeanReversion {
    fn name(&self) -> &str {
        "RsiMeanReversion"
    }

    fn on_init(&mut self) {
        self.rsi = RSI::new(self.period);
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        let rsi_val = self.rsi.update(kline.close)?;

        if rsi_val < self.oversold {
            // Oversold — expect reversion upward
            let confidence = (self.oversold - rsi_val) / self.oversold;
            Some(Signal::buy(&kline.symbol, confidence, kline.datetime))
        } else if rsi_val > self.overbought {
            // Overbought — expect reversion downward
            let confidence = (rsi_val - self.overbought) / (100.0 - self.overbought);
            Some(Signal::sell(&kline.symbol, confidence, kline.datetime))
        } else {
            None
        }
    }

    fn on_stop(&mut self) {}
}

/// Momentum Strategy using MACD
pub struct MacdMomentum {
    fast: usize,
    slow: usize,
    signal: usize,
    macd: MACD,
    prev_histogram: Option<f64>,
}

impl MacdMomentum {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast,
            slow,
            signal,
            macd: MACD::new(fast, slow, signal),
            prev_histogram: None,
        }
    }
}

impl Strategy for MacdMomentum {
    fn name(&self) -> &str {
        "MacdMomentum"
    }

    fn on_init(&mut self) {
        self.macd = MACD::new(self.fast, self.slow, self.signal);
        self.prev_histogram = None;
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        self.macd.update(kline.close);
        let histogram = self.macd.histogram();

        let result = match (histogram, self.prev_histogram) {
            (Some(curr), Some(prev)) => {
                if prev <= 0.0 && curr > 0.0 {
                    // Histogram crosses above zero — bullish momentum
                    Some(Signal::buy(&kline.symbol, curr.abs(), kline.datetime))
                } else if prev >= 0.0 && curr < 0.0 {
                    // Histogram crosses below zero — bearish momentum
                    Some(Signal::sell(&kline.symbol, curr.abs(), kline.datetime))
                } else {
                    None
                }
            }
            _ => None,
        };

        self.prev_histogram = histogram;
        result
    }

    fn on_stop(&mut self) {}
}

// ── Multi-Factor Model Strategy ─────────────────────────────────────
//
// Combines 6 factor groups into a composite score each bar:
//   1. Trend       — SMA crossover + EMA slope
//   2. Momentum    — RSI + MACD histogram direction
//   3. Volatility  — Bollinger Bands %B + bandwidth
//   4. Osc/Reversal— KDJ golden/dead cross
//   5. Volume      — Volume ratio vs moving average
//   6. Price Action— Close vs N-day high/low range
//
// Each factor group produces a sub-score in [-1, +1].
// The composite score is a weighted sum → mapped to BUY / SELL / HOLD.

/// Configuration for the multi-factor model.
#[derive(Debug, Clone)]
pub struct MultiFactorConfig {
    // Trend
    pub sma_fast: usize,
    pub sma_slow: usize,
    pub ema_period: usize,
    // Momentum
    pub rsi_period: usize,
    pub macd_fast: usize,
    pub macd_slow: usize,
    pub macd_signal: usize,
    // Volatility
    pub bb_period: usize,
    pub bb_std: f64,
    // KDJ
    pub kdj_period: usize,
    pub kdj_k: usize,
    pub kdj_d: usize,
    // Volume
    pub vol_fast: usize,
    pub vol_slow: usize,
    // Price action
    pub price_range_period: usize,
    // Signal thresholds (composite score)
    pub buy_threshold: f64,
    pub sell_threshold: f64,
    // Factor weights (must sum to 1.0)
    pub w_trend: f64,
    pub w_momentum: f64,
    pub w_volatility: f64,
    pub w_oscillator: f64,
    pub w_volume: f64,
    pub w_price_action: f64,
}

impl Default for MultiFactorConfig {
    fn default() -> Self {
        Self {
            sma_fast: 5,
            sma_slow: 20,
            ema_period: 10,
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            bb_period: 20,
            bb_std: 2.0,
            kdj_period: 9,
            kdj_k: 3,
            kdj_d: 3,
            vol_fast: 5,
            vol_slow: 20,
            price_range_period: 20,
            buy_threshold: 0.15,
            sell_threshold: -0.15,
            w_trend: 0.25,
            w_momentum: 0.25,
            w_volatility: 0.15,
            w_oscillator: 0.15,
            w_volume: 0.10,
            w_price_action: 0.10,
        }
    }
}

// ── Dynamic Weight Adapter ──────────────────────────────────────────
// Tracks each factor's directional accuracy over a rolling window
// and adjusts weights using exponential smoothing.

const NUM_FACTORS: usize = 6;

/// Tracks factor performance and adapts weights dynamically.
#[derive(Debug, Clone)]
pub struct DynamicWeightAdapter {
    /// Whether adaptive weighting is enabled.
    pub enabled: bool,
    /// Rolling window size for accuracy tracking.
    pub window: usize,
    /// Smoothing factor for weight updates (0..1, higher = faster adaptation).
    pub alpha: f64,
    /// Minimum weight floor (prevents any factor from going to zero).
    pub min_weight: f64,
    /// Base weights (initial/default).
    base_weights: [f64; NUM_FACTORS],
    /// Current adaptive weights.
    current_weights: [f64; NUM_FACTORS],
    /// Rolling accuracy scores for each factor (correct direction / total).
    accuracy_ring: Vec<[f64; NUM_FACTORS]>,
    /// Previous bar's factor scores (for accuracy evaluation).
    prev_scores: Option<[f64; NUM_FACTORS]>,
    /// Previous bar's close price (for direction evaluation).
    prev_close: Option<f64>,
    /// How many bars have been evaluated.
    eval_count: usize,
}

impl Default for DynamicWeightAdapter {
    fn default() -> Self {
        let base = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10];
        Self {
            enabled: true,
            window: 50,
            alpha: 0.1,
            min_weight: 0.03,
            base_weights: base,
            current_weights: base,
            accuracy_ring: Vec::new(),
            prev_scores: None,
            prev_close: None,
            eval_count: 0,
        }
    }
}

impl DynamicWeightAdapter {
    pub fn new(base_weights: [f64; NUM_FACTORS], window: usize, alpha: f64) -> Self {
        Self {
            enabled: true,
            window,
            alpha,
            min_weight: 0.03,
            base_weights,
            current_weights: base_weights,
            accuracy_ring: Vec::new(),
            prev_scores: None,
            prev_close: None,
            eval_count: 0,
        }
    }

    /// Call each bar with the 6 factor scores and current close price.
    /// Returns the current adaptive weights.
    pub fn update(&mut self, scores: [f64; NUM_FACTORS], close: f64) -> [f64; NUM_FACTORS] {
        if !self.enabled {
            return self.base_weights;
        }

        // Evaluate previous bar's factor predictions
        if let (Some(prev_scores), Some(prev_close)) = (self.prev_scores, self.prev_close) {
            let actual_direction = (close - prev_close).signum(); // +1 up, -1 down, 0 flat

            // Each factor gets +1 if it predicted the correct direction, 0 otherwise
            let mut hits = [0.0f64; NUM_FACTORS];
            for i in 0..NUM_FACTORS {
                let predicted_dir = prev_scores[i].signum();
                // correct if same sign, or factor was neutral (0)
                if predicted_dir != 0.0 && predicted_dir == actual_direction {
                    hits[i] = 1.0;
                } else if predicted_dir == 0.0 {
                    hits[i] = 0.5; // neutral = half credit
                }
            }

            self.accuracy_ring.push(hits);
            if self.accuracy_ring.len() > self.window {
                self.accuracy_ring.remove(0);
            }
            self.eval_count += 1;

            // Recompute accuracy-based weights when we have enough data
            if self.accuracy_ring.len() >= 10 {
                let n = self.accuracy_ring.len() as f64;
                let mut accuracy = [0.0f64; NUM_FACTORS];
                for row in &self.accuracy_ring {
                    for i in 0..NUM_FACTORS {
                        accuracy[i] += row[i];
                    }
                }
                for a in accuracy.iter_mut() {
                    *a /= n;
                }

                // Exponential smoothing: new_weight = alpha * accuracy_weight + (1-alpha) * base_weight
                let accuracy_sum: f64 = accuracy.iter().sum();
                if accuracy_sum > 0.0 {
                    for i in 0..NUM_FACTORS {
                        let target = accuracy[i] / accuracy_sum;
                        self.current_weights[i] =
                            self.alpha * target + (1.0 - self.alpha) * self.current_weights[i];
                    }
                }

                // Apply minimum floor and renormalize
                for w in self.current_weights.iter_mut() {
                    if *w < self.min_weight {
                        *w = self.min_weight;
                    }
                }
                let sum: f64 = self.current_weights.iter().sum();
                for w in self.current_weights.iter_mut() {
                    *w /= sum;
                }
            }
        }

        self.prev_scores = Some(scores);
        self.prev_close = Some(close);

        self.current_weights
    }

    /// Get current weights (for reporting).
    pub fn weights(&self) -> [f64; NUM_FACTORS] {
        self.current_weights
    }

    /// Get factor accuracy over the window.
    pub fn factor_accuracy(&self) -> [f64; NUM_FACTORS] {
        if self.accuracy_ring.is_empty() {
            return [0.0; NUM_FACTORS];
        }
        let n = self.accuracy_ring.len() as f64;
        let mut acc = [0.0f64; NUM_FACTORS];
        for row in &self.accuracy_ring {
            for i in 0..NUM_FACTORS {
                acc[i] += row[i];
            }
        }
        for a in acc.iter_mut() {
            *a /= n;
        }
        acc
    }

    /// Reset adapter state.
    pub fn reset(&mut self) {
        self.current_weights = self.base_weights;
        self.accuracy_ring.clear();
        self.prev_scores = None;
        self.prev_close = None;
        self.eval_count = 0;
    }
}

/// Multi-Factor Model trading strategy.
///
/// Computes 6 factor sub-scores on each bar, produces a composite signal
/// in [-1, +1], and generates BUY when > buy_threshold or SELL when < sell_threshold.
///
/// When `adaptive_weights` is enabled, the strategy dynamically adjusts factor
/// weights based on each factor's recent directional accuracy — factors that
/// correctly predicted the next bar's direction get more weight.
pub struct MultiFactorStrategy {
    cfg: MultiFactorConfig,
    // Trend indicators
    sma_fast: SMA,
    sma_slow: SMA,
    ema: EMA,
    prev_ema: Option<f64>,
    prev_sma_fast: Option<f64>,
    prev_sma_slow: Option<f64>,
    // Momentum
    rsi: RSI,
    macd: MACD,
    prev_histogram: Option<f64>,
    // Volatility
    bb: BollingerBands,
    // KDJ
    kdj: KDJ,
    prev_kdj_k: Option<f64>,
    prev_kdj_d: Option<f64>,
    // Volume
    vol_fast_buf: Vec<f64>,
    vol_slow_buf: Vec<f64>,
    // Price action
    close_buf: Vec<f64>,
    high_buf: Vec<f64>,
    low_buf: Vec<f64>,
    // State
    prev_composite: Option<f64>,
    bar_count: usize,
    // Dynamic weight adapter
    weight_adapter: DynamicWeightAdapter,
}

impl MultiFactorStrategy {
    pub fn new(cfg: MultiFactorConfig) -> Self {
        let base_weights = [
            cfg.w_trend, cfg.w_momentum, cfg.w_volatility,
            cfg.w_oscillator, cfg.w_volume, cfg.w_price_action,
        ];
        Self {
            sma_fast: SMA::new(cfg.sma_fast),
            sma_slow: SMA::new(cfg.sma_slow),
            ema: EMA::new(cfg.ema_period),
            prev_ema: None,
            prev_sma_fast: None,
            prev_sma_slow: None,
            rsi: RSI::new(cfg.rsi_period),
            macd: MACD::new(cfg.macd_fast, cfg.macd_slow, cfg.macd_signal),
            prev_histogram: None,
            bb: BollingerBands::new(cfg.bb_period, cfg.bb_std),
            kdj: KDJ::new(cfg.kdj_period, cfg.kdj_k, cfg.kdj_d),
            prev_kdj_k: None,
            prev_kdj_d: None,
            vol_fast_buf: Vec::with_capacity(cfg.vol_fast),
            vol_slow_buf: Vec::with_capacity(cfg.vol_slow),
            close_buf: Vec::with_capacity(cfg.price_range_period),
            high_buf: Vec::with_capacity(cfg.price_range_period),
            low_buf: Vec::with_capacity(cfg.price_range_period),
            prev_composite: None,
            bar_count: 0,
            weight_adapter: DynamicWeightAdapter::new(base_weights, 50, 0.1),
            cfg,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(MultiFactorConfig::default())
    }

    // ── Factor 1: Trend ──────────────────────────────────────────
    // SMA crossover direction + EMA slope → [-1, +1]
    fn score_trend(&self, sma_f: f64, sma_s: f64, ema: f64) -> f64 {
        let cross_raw = (sma_f - sma_s) / sma_s;
        let cross_score = cross_raw.max(-0.05).min(0.05) / 0.05;

        let slope_score = if let Some(prev) = self.prev_ema {
            let slope = (ema - prev) / prev;
            (slope * 200.0).max(-1.0).min(1.0)
        } else {
            0.0
        };

        0.6 * cross_score + 0.4 * slope_score
    }

    // ── Factor 2: Momentum ───────────────────────────────────────
    // RSI position + MACD histogram direction → [-1, +1]
    fn score_momentum(&self, rsi_val: Option<f64>, histogram: Option<f64>) -> f64 {
        let rsi_score = match rsi_val {
            Some(r) => ((r - 50.0) / 50.0).max(-1.0).min(1.0),
            None => 0.0,
        };

        let macd_score = match (histogram, self.prev_histogram) {
            (Some(h), Some(ph)) => {
                let direction: f64 = if h > ph { 0.5 } else { -0.5 };
                let level: f64 = if h > 0.0 { 0.5 } else { -0.5 };
                (direction + level).max(-1.0).min(1.0)
            }
            (Some(h), None) => {
                if h > 0.0 { 0.3 } else { -0.3 }
            }
            _ => 0.0,
        };

        0.5 * rsi_score + 0.5 * macd_score
    }

    // ── Factor 3: Volatility (Bollinger) ─────────────────────────
    // %B position: near lower band = buy (+), near upper band = sell (-)
    fn score_volatility(&self, close: f64) -> f64 {
        match (self.bb.upper(), self.bb.lower(), self.bb.middle()) {
            (Some(upper), Some(lower), Some(_mid)) => {
                let range = upper - lower;
                if range < f64::EPSILON { return 0.0; }
                let pct_b = (close - lower) / range;
                (1.0 - 2.0 * pct_b).max(-1.0).min(1.0)
            }
            _ => 0.0,
        }
    }

    // ── Factor 4: KDJ Oscillator ─────────────────────────────────
    // K-D cross + J extreme → [-1, +1]
    fn score_oscillator(&self) -> f64 {
        match (self.kdj.k(), self.kdj.d(), self.kdj.j()) {
            (Some(k), Some(d), Some(j)) => {
                let kd_score = match (self.prev_kdj_k, self.prev_kdj_d) {
                    (Some(pk), Some(pd)) => {
                        if pk <= pd && k > d { 0.8 }       // golden cross
                        else if pk >= pd && k < d { -0.8 }  // dead cross
                        else if k > d { 0.3 }
                        else { -0.3 }
                    }
                    _ => if k > d { 0.2 } else { -0.2 },
                };

                let j_score = if j < 20.0 { 0.5 }
                    else if j > 80.0 { -0.5 }
                    else { 0.0 };

                0.6 * kd_score + 0.4 * j_score
            }
            _ => 0.0,
        }
    }

    // ── Factor 5: Volume ─────────────────────────────────────────
    // Short-term vol vs long-term avg, signed by price direction → [-1, +1]
    fn score_volume(&self, close: f64) -> f64 {
        if self.vol_fast_buf.len() < self.cfg.vol_fast || self.vol_slow_buf.len() < self.cfg.vol_slow {
            return 0.0;
        }
        let fast_avg: f64 = self.vol_fast_buf.iter().sum::<f64>() / self.vol_fast_buf.len() as f64;
        let slow_avg: f64 = self.vol_slow_buf.iter().sum::<f64>() / self.vol_slow_buf.len() as f64;
        if slow_avg < f64::EPSILON { return 0.0; }
        let ratio = fast_avg / slow_avg;

        let price_dir = if self.close_buf.len() >= 2 {
            let prev = self.close_buf[self.close_buf.len() - 2];
            if close > prev { 1.0 } else { -1.0 }
        } else {
            0.0
        };

        let vol_signal: f64 = if ratio > 2.0 { 0.8 * price_dir }
            else if ratio > 1.5 { 0.5 * price_dir }
            else if ratio > 1.0 { 0.2 * price_dir }
            else { 0.1 * price_dir };

        vol_signal.max(-1.0).min(1.0)
    }

    // ── Factor 6: Price Action ───────────────────────────────────
    // Close position in N-day high/low range → [-1, +1]
    fn score_price_action(&self, close: f64) -> f64 {
        if self.high_buf.len() < self.cfg.price_range_period {
            return 0.0;
        }
        let highest = self.high_buf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lowest = self.low_buf.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = highest - lowest;
        if range < f64::EPSILON { return 0.0; }
        let position = (close - lowest) / range;

        if position > 0.95 { 0.6 }          // breakout → bullish
        else if position > 0.8 { -0.2 }     // near top → slightly overbought
        else if position < 0.05 { -0.6 }    // breakdown → bearish
        else if position < 0.2 { 0.3 }      // near bottom → potential reversal
        else { (position - 0.5) * 0.4 }     // middle → mild directional
    }
}

impl Strategy for MultiFactorStrategy {
    fn name(&self) -> &str {
        "MultiFactorModel"
    }

    fn on_init(&mut self) {
        self.sma_fast = SMA::new(self.cfg.sma_fast);
        self.sma_slow = SMA::new(self.cfg.sma_slow);
        self.ema = EMA::new(self.cfg.ema_period);
        self.prev_ema = None;
        self.prev_sma_fast = None;
        self.prev_sma_slow = None;
        self.rsi = RSI::new(self.cfg.rsi_period);
        self.macd = MACD::new(self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal);
        self.prev_histogram = None;
        self.bb = BollingerBands::new(self.cfg.bb_period, self.cfg.bb_std);
        self.kdj = KDJ::new(self.cfg.kdj_period, self.cfg.kdj_k, self.cfg.kdj_d);
        self.prev_kdj_k = None;
        self.prev_kdj_d = None;
        self.vol_fast_buf.clear();
        self.vol_slow_buf.clear();
        self.close_buf.clear();
        self.high_buf.clear();
        self.low_buf.clear();
        self.prev_composite = None;
        self.bar_count = 0;
        self.weight_adapter.reset();
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        self.bar_count += 1;

        // Update all indicators
        let sma_f = self.sma_fast.update(kline.close);
        let sma_s = self.sma_slow.update(kline.close);
        let ema_val = self.ema.update(kline.close);
        let rsi_val = self.rsi.update(kline.close);
        self.macd.update(kline.close);
        let histogram = self.macd.histogram();
        self.bb.update(kline.close);
        self.kdj.update(kline.high, kline.low, kline.close);

        // Update volume buffers
        self.vol_fast_buf.push(kline.volume);
        if self.vol_fast_buf.len() > self.cfg.vol_fast {
            self.vol_fast_buf.remove(0);
        }
        self.vol_slow_buf.push(kline.volume);
        if self.vol_slow_buf.len() > self.cfg.vol_slow {
            self.vol_slow_buf.remove(0);
        }

        // Update price buffers
        self.close_buf.push(kline.close);
        if self.close_buf.len() > self.cfg.price_range_period {
            self.close_buf.remove(0);
        }
        self.high_buf.push(kline.high);
        if self.high_buf.len() > self.cfg.price_range_period {
            self.high_buf.remove(0);
        }
        self.low_buf.push(kline.low);
        if self.low_buf.len() > self.cfg.price_range_period {
            self.low_buf.remove(0);
        }

        // Need slow SMA warmup before generating signals
        let result = if let (Some(sf), Some(ss), Some(ev)) = (sma_f, sma_s, ema_val) {
            // Compute all factor scores
            let s_trend = self.score_trend(sf, ss, ev);
            let s_momentum = self.score_momentum(rsi_val, histogram);
            let s_volatility = self.score_volatility(kline.close);
            let s_oscillator = self.score_oscillator();
            let s_volume = self.score_volume(kline.close);
            let s_price_action = self.score_price_action(kline.close);

            let factor_scores = [s_trend, s_momentum, s_volatility, s_oscillator, s_volume, s_price_action];

            // Get adaptive weights (updates adapter state and returns current weights)
            let w = self.weight_adapter.update(factor_scores, kline.close);

            // Weighted composite (weights are normalized to sum=1.0 by adapter)
            let composite = w[0] * s_trend
                + w[1] * s_momentum
                + w[2] * s_volatility
                + w[3] * s_oscillator
                + w[4] * s_volume
                + w[5] * s_price_action;

            // Signal on threshold crossings to avoid repeated signals
            let signal = match self.prev_composite {
                Some(prev) => {
                    if composite > self.cfg.buy_threshold && prev <= self.cfg.buy_threshold {
                        let confidence = ((composite - self.cfg.buy_threshold) / (1.0 - self.cfg.buy_threshold))
                            .max(0.0).min(1.0);
                        Some(Signal::buy(&kline.symbol, confidence, kline.datetime))
                    } else if composite < self.cfg.sell_threshold && prev >= self.cfg.sell_threshold {
                        let confidence = ((self.cfg.sell_threshold - composite) / (1.0 + self.cfg.sell_threshold))
                            .max(0.0).min(1.0);
                        Some(Signal::sell(&kline.symbol, confidence, kline.datetime))
                    } else {
                        None
                    }
                }
                None => {
                    // First bar after warmup: use same threshold (no crossing required)
                    if composite > self.cfg.buy_threshold {
                        Some(Signal::buy(&kline.symbol, composite.max(0.0).min(1.0), kline.datetime))
                    } else if composite < self.cfg.sell_threshold {
                        Some(Signal::sell(&kline.symbol, composite.abs().min(1.0), kline.datetime))
                    } else {
                        None
                    }
                }
            };

            self.prev_composite = Some(composite);
            signal
        } else {
            None
        };

        // Save state for next bar
        self.prev_ema = ema_val;
        self.prev_sma_fast = sma_f;
        self.prev_sma_slow = sma_s;
        self.prev_histogram = histogram;
        self.prev_kdj_k = self.kdj.k();
        self.prev_kdj_d = self.kdj.d();

        result
    }

    fn on_stop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_kline(symbol: &str, close: f64, high: f64, low: f64, volume: f64) -> Kline {
        Kline {
            symbol: symbol.to_string(),
            datetime: Utc::now().naive_utc(),
            open: close,
            high,
            low,
            close,
            volume,
        }
    }

    #[test]
    fn test_multifactor_needs_warmup() {
        let mut strat = MultiFactorStrategy::with_defaults();
        strat.on_init();

        for i in 1..=19 {
            let k = make_kline("TEST", 100.0 + i as f64 * 0.1, 101.0, 99.0, 1_000_000.0);
            assert!(strat.on_bar(&k).is_none(), "Expected None on bar {}", i);
        }
    }

    #[test]
    fn test_multifactor_generates_buy_on_uptrend() {
        let mut strat = MultiFactorStrategy::with_defaults();
        strat.on_init();

        // Warm up with flat prices
        for _ in 0..30 {
            let k = make_kline("TEST", 100.0, 100.5, 99.5, 1_000_000.0);
            strat.on_bar(&k);
        }

        // Strong uptrend with high volume
        let mut got_buy = false;
        for i in 1..=30 {
            let price = 100.0 + i as f64 * 0.8;
            let k = make_kline("TEST", price, price + 0.5, price - 0.3, 3_000_000.0);
            if let Some(sig) = strat.on_bar(&k) {
                if sig.is_buy() {
                    got_buy = true;
                    break;
                }
            }
        }
        assert!(got_buy, "Should generate BUY during strong uptrend");
    }

    #[test]
    fn test_multifactor_generates_sell_on_downtrend() {
        let mut strat = MultiFactorStrategy::with_defaults();
        strat.on_init();

        // Warm up with uptrend
        for i in 0..35 {
            let price = 100.0 + i as f64 * 0.5;
            let k = make_kline("TEST", price, price + 0.3, price - 0.3, 1_000_000.0);
            strat.on_bar(&k);
        }

        // Strong downtrend
        let mut got_sell = false;
        let peak = 100.0 + 35.0 * 0.5;
        for i in 1..=30 {
            let price = peak - i as f64 * 1.0;
            let k = make_kline("TEST", price, price + 0.2, price - 0.5, 3_000_000.0);
            if let Some(sig) = strat.on_bar(&k) {
                if sig.is_sell() {
                    got_sell = true;
                    break;
                }
            }
        }
        assert!(got_sell, "Should generate SELL during strong downtrend");
    }

    #[test]
    fn test_multifactor_no_signal_on_flat() {
        let mut strat = MultiFactorStrategy::with_defaults();
        strat.on_init();

        let mut signals = 0;
        for i in 0..60 {
            let noise = ((i as f64) * 0.7).sin() * 0.2;
            let price = 100.0 + noise;
            let k = make_kline("TEST", price, price + 0.3, price - 0.3, 1_000_000.0);
            if strat.on_bar(&k).is_some() {
                signals += 1;
            }
        }
        assert!(signals <= 3, "Expected few signals on flat market, got {}", signals);
    }

    #[test]
    fn test_dynamic_weight_adapter_initial() {
        let adapter = DynamicWeightAdapter::default();
        let w = adapter.weights();
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Initial weights should sum to ~1.0, got {}", sum);
        assert_eq!(w[0], 0.25); // trend
        assert_eq!(w[4], 0.10); // volume
    }

    #[test]
    fn test_dynamic_weight_adapter_adapts() {
        let base = [0.167, 0.167, 0.167, 0.167, 0.166, 0.166]; // nearly equal
        let mut adapter = DynamicWeightAdapter::new(base, 20, 0.5); // high alpha for fast adaptation

        // Feed 50 bars where factor[0] (trend) is always correct (+1 when price goes up)
        // and factor[5] (price action) is always wrong (-1 when price goes up)
        for i in 0..50 {
            let close = 100.0 + i as f64 * 0.5; // steadily rising
            let scores = [1.0, 0.5, 0.0, 0.0, 0.0, -1.0]; // trend correct, price_action wrong
            adapter.update(scores, close);
        }

        let w = adapter.weights();
        // Trend weight (always correct) should be > price_action (always wrong)
        assert!(w[0] > w[5], "Trend weight should be > price_action: {:.3} vs {:.3}", w[0], w[5]);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Weights should still sum to ~1.0, got {}", sum);
    }

    #[test]
    fn test_dynamic_weight_adapter_min_floor() {
        let base = [0.50, 0.50, 0.0, 0.0, 0.0, 0.0];
        let mut adapter = DynamicWeightAdapter::new(base, 10, 0.5);
        adapter.min_weight = 0.05;

        for i in 0..20 {
            let close = 100.0 + i as f64;
            let scores = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
            adapter.update(scores, close);
        }

        let w = adapter.weights();
        for &wi in &w {
            assert!(wi >= 0.03, "No weight should be below min_weight floor, got {:.4}", wi);
        }
    }

    #[test]
    fn test_dynamic_weight_adapter_reset() {
        let mut adapter = DynamicWeightAdapter::default();
        for i in 0..20 {
            adapter.update([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 100.0 + i as f64);
        }
        let before = adapter.weights();
        adapter.reset();
        let after = adapter.weights();
        // After reset, weights should be back to base
        assert_eq!(after, adapter.base_weights);
        assert_ne!(before, after, "Weights should have changed before reset");
    }

    #[test]
    fn test_dynamic_weight_disabled() {
        let mut adapter = DynamicWeightAdapter::default();
        adapter.enabled = false;
        for i in 0..30 {
            adapter.update([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 100.0 + i as f64);
        }
        let w = adapter.weights();
        assert_eq!(w, adapter.base_weights, "Disabled adapter should return base weights");
    }
}
