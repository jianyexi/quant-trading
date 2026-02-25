/// Active risk enforcement: stop-loss, daily loss limit, max drawdown, circuit breaker.
///
/// These are stateful runtime checks that the engine evaluates continuously,
/// as opposed to the pre-trade checks in `checks.rs`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

// ── Configuration ──────────────────────────────────────────────────

/// Risk enforcement configuration. All thresholds are fractions (0.05 = 5%).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Stop-loss threshold per position (unrealized loss / cost).
    pub stop_loss_pct: f64,
    /// Maximum daily loss as fraction of initial capital.
    pub max_daily_loss_pct: f64,
    /// Maximum drawdown from peak portfolio value.
    pub max_drawdown_pct: f64,
    /// Consecutive order failures before circuit breaker trips.
    pub circuit_breaker_failures: u32,
    /// Whether the engine should auto-halt on max drawdown.
    pub halt_on_drawdown: bool,
    /// Max holding days before timeout exit (0 = disabled)
    #[serde(default)]
    pub max_holding_days: u32,
    /// Min profit % to keep position past timeout (e.g. 0.02 = 2%)
    #[serde(default = "default_timeout_min_profit")]
    pub timeout_min_profit_pct: f64,
    /// Rebalance threshold: trigger when weight drifts this much from target (e.g. 0.05 = 5%)
    #[serde(default = "default_rebalance_threshold")]
    pub rebalance_threshold: f64,
    /// Volatility spike ratio: if short-term vol / long-term vol exceeds this,
    /// trigger deleverage (e.g. 2.0 = short vol is 2x long vol).
    #[serde(default = "default_vol_spike_ratio")]
    pub vol_spike_ratio: f64,
    /// Position scale-down factor during volatility spike (e.g. 0.5 = halve all positions).
    #[serde(default = "default_vol_deleverage_factor")]
    pub vol_deleverage_factor: f64,
    /// Maximum total exposure to correlated positions (correlation > 0.7).
    /// Expressed as fraction of portfolio (e.g. 0.4 = 40%).
    #[serde(default = "default_max_correlated_exposure")]
    pub max_correlated_exposure: f64,
    /// VaR confidence level (e.g. 0.95 = 95% VaR)
    #[serde(default = "default_var_confidence")]
    pub var_confidence: f64,
    /// Maximum acceptable VaR as fraction of portfolio (e.g. 0.05 = 5%)
    #[serde(default = "default_max_var_pct")]
    pub max_var_pct: f64,
}

fn default_timeout_min_profit() -> f64 { 0.02 }
fn default_rebalance_threshold() -> f64 { 0.05 }
fn default_vol_spike_ratio() -> f64 { 2.0 }
fn default_vol_deleverage_factor() -> f64 { 0.5 }
fn default_max_correlated_exposure() -> f64 { 0.4 }
fn default_var_confidence() -> f64 { 0.95 }
fn default_max_var_pct() -> f64 { 0.05 }

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            stop_loss_pct: 0.05,
            max_daily_loss_pct: 0.03,
            max_drawdown_pct: 0.10,
            circuit_breaker_failures: 5,
            halt_on_drawdown: true,
            max_holding_days: 30,
            timeout_min_profit_pct: 0.02,
            rebalance_threshold: 0.05,
            vol_spike_ratio: 2.0,
            vol_deleverage_factor: 0.5,
            max_correlated_exposure: 0.4,
            var_confidence: 0.95,
            max_var_pct: 0.05,
        }
    }
}

// ── Position for stop-loss ─────────────────────────────────────────

/// Minimal position info for stop-loss checks.
#[derive(Debug, Clone)]
pub struct PositionInfo {
    pub symbol: String,
    pub quantity: f64,
    pub avg_cost: f64,
    pub current_price: f64,
    pub entry_time: chrono::NaiveDateTime,
    pub target_weight: f64,
}

impl PositionInfo {
    pub fn unrealized_pnl(&self) -> f64 {
        (self.current_price - self.avg_cost) * self.quantity
    }

    pub fn unrealized_pnl_pct(&self) -> f64 {
        if self.avg_cost <= 0.0 { return 0.0; }
        (self.current_price - self.avg_cost) / self.avg_cost
    }

    pub fn holding_days(&self) -> i64 {
        let now = chrono::Utc::now().naive_utc();
        (now - self.entry_time).num_days()
    }

    pub fn market_value(&self) -> f64 {
        self.current_price * self.quantity
    }
}

// ── Timeout Signal ─────────────────────────────────────────────────

/// A timeout exit signal when position held too long without sufficient profit.
#[derive(Debug, Clone, Serialize)]
pub struct TimeoutSignal {
    pub symbol: String,
    pub quantity: f64,
    pub holding_days: i64,
    pub profit_pct: f64,
    pub max_days: u32,
}

// ── Rebalance Signal ───────────────────────────────────────────────

/// A rebalance signal when position weight drifts from target.
#[derive(Debug, Clone, Serialize)]
pub struct RebalanceSignal {
    pub symbol: String,
    pub current_weight: f64,
    pub target_weight: f64,
    pub drift: f64,
    /// Positive = need to buy more, negative = need to sell
    pub adjustment_shares: f64,
}

// ── Volatility Spike Signal ────────────────────────────────────────

/// Signal to deleverage when short-term volatility spikes above long-term.
#[derive(Debug, Clone, Serialize)]
pub struct VolatilitySpikeSignal {
    pub short_vol: f64,
    pub long_vol: f64,
    pub spike_ratio: f64,
    pub deleverage_factor: f64,
    /// Recommended position reductions: (symbol, reduce_by_shares)
    pub reductions: Vec<(String, f64)>,
}

// ── Correlation Warning Signal ────────────────────────────────────

/// Warning when correlated positions exceed concentration limit.
#[derive(Debug, Clone, Serialize)]
pub struct CorrelationWarning {
    /// Groups of correlated symbols and their combined exposure.
    pub groups: Vec<CorrelatedGroup>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrelatedGroup {
    pub symbols: Vec<String>,
    pub total_exposure: f64,
    pub max_allowed: f64,
    pub excess: f64,
}

// ── Tail Risk (VaR) Signal ────────────────────────────────────────

/// Value-at-Risk estimate from historical simulation.
#[derive(Debug, Clone, Serialize)]
pub struct TailRiskSignal {
    pub var_pct: f64,      // VaR as % of portfolio (positive = loss)
    pub cvar_pct: f64,     // CVaR (expected shortfall)
    pub confidence: f64,
    pub max_allowed: f64,
    pub breach: bool,
}

// ── Risk Event Log ─────────────────────────────────────────────────

/// A timestamped risk event for the monitoring timeline.
#[derive(Debug, Clone, Serialize)]
pub struct RiskEvent {
    pub timestamp: DateTime<Utc>,
    pub severity: RiskSeverity,
    pub event_type: String,
    pub message: String,
    /// Optional structured payload (serialized signal)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum RiskSeverity {
    Info,
    Warning,
    Critical,
}

// ── Stop-Loss Signal ───────────────────────────────────────────────

/// A stop-loss signal generated when a position breaches the threshold.
#[derive(Debug, Clone, Serialize)]
pub struct StopLossSignal {
    pub symbol: String,
    pub quantity: f64,
    pub current_price: f64,
    pub avg_cost: f64,
    pub loss_pct: f64,
    pub threshold: f64,
}

// ── Risk Enforcer ──────────────────────────────────────────────────

/// Stateful risk enforcer that tracks daily PnL, drawdown, and circuit breaker.
#[derive(Debug)]
pub struct RiskEnforcer {
    config: RiskConfig,
    initial_capital: f64,

    // Daily loss tracking
    daily_pnl: Mutex<f64>,
    daily_paused: AtomicBool,

    // Drawdown tracking
    peak_value: Mutex<f64>,
    drawdown_halted: AtomicBool,

    // Circuit breaker
    consecutive_failures: AtomicU32,
    circuit_open: AtomicBool,

    // Volatility tracking: ring buffer of recent portfolio returns
    return_history: Mutex<Vec<f64>>,
    prev_portfolio_value: Mutex<f64>,

    // Volatility spike state
    vol_spike_active: AtomicBool,

    // Risk event log (ring buffer, max 200 entries)
    event_log: Mutex<Vec<RiskEvent>>,
}

impl RiskEnforcer {
    pub fn new(config: RiskConfig, initial_capital: f64) -> Self {
        Self {
            config,
            initial_capital,
            daily_pnl: Mutex::new(0.0),
            daily_paused: AtomicBool::new(false),
            peak_value: Mutex::new(initial_capital),
            drawdown_halted: AtomicBool::new(false),
            consecutive_failures: AtomicU32::new(0),
            circuit_open: AtomicBool::new(false),
            return_history: Mutex::new(Vec::with_capacity(256)),
            prev_portfolio_value: Mutex::new(initial_capital),
            vol_spike_active: AtomicBool::new(false),
            event_log: Mutex::new(Vec::with_capacity(200)),
        }
    }

    // ── Stop-Loss ──────────────────────────────────────────────────

    /// Check all positions and return stop-loss sell signals for any that breach the threshold.
    pub fn check_stop_losses(&self, positions: &[PositionInfo]) -> Vec<StopLossSignal> {
        let mut signals = Vec::new();
        for pos in positions {
            let loss_pct = pos.unrealized_pnl_pct();
            if loss_pct < -self.config.stop_loss_pct {
                signals.push(StopLossSignal {
                    symbol: pos.symbol.clone(),
                    quantity: pos.quantity,
                    current_price: pos.current_price,
                    avg_cost: pos.avg_cost,
                    loss_pct,
                    threshold: self.config.stop_loss_pct,
                });
            }
        }
        signals
    }

    // ── Daily Loss ─────────────────────────────────────────────────

    /// Record a realized PnL event (fill). Updates daily_pnl and checks threshold.
    pub fn record_realized_pnl(&self, pnl: f64) {
        let mut daily = self.daily_pnl.lock().unwrap();
        *daily += pnl;
        let threshold = self.initial_capital * self.config.max_daily_loss_pct;
        if *daily < -threshold {
            if !self.daily_paused.swap(true, Ordering::SeqCst) {
                self.push_event(
                    RiskSeverity::Critical,
                    "daily_loss_limit",
                    &format!("日内亏损触发暂停: ¥{:.0}, 限额: ¥{:.0}", *daily, -threshold),
                    None,
                );
            }
        }
    }

    /// Check if new buys are paused due to daily loss limit.
    pub fn is_daily_paused(&self) -> bool {
        self.daily_paused.load(Ordering::SeqCst)
    }

    /// Reset daily PnL (call at start of each trading day).
    pub fn reset_daily(&self) {
        *self.daily_pnl.lock().unwrap() = 0.0;
        self.daily_paused.store(false, Ordering::SeqCst);
    }

    /// Get current daily PnL.
    pub fn get_daily_pnl(&self) -> f64 {
        *self.daily_pnl.lock().unwrap()
    }

    // ── Max Drawdown ───────────────────────────────────────────────

    /// Update portfolio value and check drawdown. Returns true if halted.
    pub fn update_portfolio_value(&self, current_value: f64) -> bool {
        let mut peak = self.peak_value.lock().unwrap();
        if current_value > *peak {
            *peak = current_value;
        }
        let drawdown = (*peak - current_value) / *peak;
        if drawdown > self.config.max_drawdown_pct && self.config.halt_on_drawdown {
            if !self.drawdown_halted.swap(true, Ordering::SeqCst) {
                self.push_event(
                    RiskSeverity::Critical,
                    "max_drawdown",
                    &format!("最大回撤触发停止: {:.1}% > {:.1}%", drawdown * 100.0, self.config.max_drawdown_pct * 100.0),
                    None,
                );
            }
            return true;
        }
        false
    }

    /// Check if engine is halted due to max drawdown.
    pub fn is_drawdown_halted(&self) -> bool {
        self.drawdown_halted.load(Ordering::SeqCst)
    }

    /// Get current drawdown percentage.
    pub fn get_drawdown_pct(&self) -> f64 {
        let peak = *self.peak_value.lock().unwrap();
        // Caller would pass current value; for now just expose peak
        peak
    }

    /// Get peak value.
    pub fn get_peak_value(&self) -> f64 {
        *self.peak_value.lock().unwrap()
    }

    // ── Circuit Breaker ────────────────────────────────────────────

    /// Record a successful order execution. Resets failure counter and circuit breaker.
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::SeqCst);
        // Auto-recover circuit breaker on success
        if self.circuit_open.load(Ordering::SeqCst) {
            self.circuit_open.store(false, Ordering::SeqCst);
            tracing::info!("🟢 Circuit breaker auto-recovered after successful order");
        }
    }

    /// Record a failed order. Increments counter and may trip circuit breaker.
    pub fn record_failure(&self) -> bool {
        let n = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
        if n >= self.config.circuit_breaker_failures {
            if !self.circuit_open.swap(true, Ordering::SeqCst) {
                self.push_event(
                    RiskSeverity::Critical,
                    "circuit_breaker",
                    &format!("熔断器触发: 连续失败{}次", n),
                    None,
                );
            }
            return true; // circuit breaker tripped
        }
        false
    }

    /// Check if circuit breaker is open (trading paused).
    pub fn is_circuit_open(&self) -> bool {
        self.circuit_open.load(Ordering::SeqCst)
    }

    /// Manually reset circuit breaker.
    pub fn reset_circuit_breaker(&self) {
        self.circuit_open.store(false, Ordering::SeqCst);
        self.consecutive_failures.store(0, Ordering::SeqCst);
    }

    // ── Position Timeout ─────────────────────────────────────────────

    /// Check positions that exceeded max holding period without sufficient profit.
    pub fn check_timeouts(&self, positions: &[PositionInfo]) -> Vec<TimeoutSignal> {
        if self.config.max_holding_days == 0 {
            return Vec::new();
        }
        let mut signals = Vec::new();
        for pos in positions {
            let days = pos.holding_days();
            if days > self.config.max_holding_days as i64 {
                let profit_pct = pos.unrealized_pnl_pct();
                if profit_pct < self.config.timeout_min_profit_pct {
                    signals.push(TimeoutSignal {
                        symbol: pos.symbol.clone(),
                        quantity: pos.quantity,
                        holding_days: days,
                        profit_pct,
                        max_days: self.config.max_holding_days,
                    });
                }
            }
        }
        signals
    }

    // ── Rebalance ──────────────────────────────────────────────────

    /// Check if any positions need rebalancing (weight drift from target).
    pub fn check_rebalance(
        &self,
        positions: &[PositionInfo],
        portfolio_value: f64,
    ) -> Vec<RebalanceSignal> {
        if portfolio_value <= 0.0 {
            return Vec::new();
        }
        let mut signals = Vec::new();
        for pos in positions {
            if pos.target_weight <= 0.0 {
                continue; // no target set
            }
            let current_weight = pos.market_value() / portfolio_value;
            let drift = current_weight - pos.target_weight;
            if drift.abs() > self.config.rebalance_threshold {
                let target_value = portfolio_value * pos.target_weight;
                let current_value = pos.market_value();
                let diff_value = target_value - current_value;
                let adjustment = if pos.current_price > 0.0 {
                    (diff_value / pos.current_price).round()
                } else {
                    0.0
                };
                signals.push(RebalanceSignal {
                    symbol: pos.symbol.clone(),
                    current_weight,
                    target_weight: pos.target_weight,
                    drift,
                    adjustment_shares: adjustment,
                });
            }
        }
        signals
    }

    // ── Volatility Spike Detection ─────────────────────────────────

    /// Record a portfolio return observation and check for volatility spike.
    /// Call once per bar/day with current portfolio value.
    /// Returns Some(signal) if short-term vol exceeds long-term vol by threshold.
    pub fn check_volatility_spike(
        &self,
        current_value: f64,
        positions: &[PositionInfo],
    ) -> Option<VolatilitySpikeSignal> {
        // Record portfolio return
        let mut prev = self.prev_portfolio_value.lock().unwrap();
        let ret = if *prev > 0.0 { (current_value - *prev) / *prev } else { 0.0 };
        *prev = current_value;

        let mut history = self.return_history.lock().unwrap();
        history.push(ret);
        // Keep at most 252 trading days (1 year)
        if history.len() > 252 {
            let excess = history.len() - 252;
            history.drain(0..excess);
        }

        // Need at least 60 days to compare short vs long vol
        if history.len() < 60 {
            return None;
        }

        // Short-term vol (5-day) vs long-term vol (60-day)
        let n = history.len();
        let short_window = 5.min(n);
        let long_window = 60.min(n);

        let short_slice = &history[n - short_window..];
        let long_slice = &history[n - long_window..];

        let short_vol = std_dev(short_slice);
        let long_vol = std_dev(long_slice);

        if long_vol < 1e-10 {
            return None;
        }

        let spike_ratio = short_vol / long_vol;

        if spike_ratio > self.config.vol_spike_ratio {
            if !self.vol_spike_active.swap(true, Ordering::SeqCst) {
                self.push_event(
                    RiskSeverity::Warning,
                    "vol_spike",
                    &format!("波动率突变: 短期{:.4} / 长期{:.4} = {:.1}x", short_vol, long_vol, spike_ratio),
                    serde_json::to_value(&serde_json::json!({
                        "short_vol": short_vol, "long_vol": long_vol, "spike_ratio": spike_ratio
                    })).ok(),
                );
            }

            // Calculate position reductions
            let factor = self.config.vol_deleverage_factor;
            let reductions: Vec<(String, f64)> = positions
                .iter()
                .filter(|p| p.quantity > 0.0)
                .map(|p| {
                    let reduce = (p.quantity * (1.0 - factor)).round();
                    (p.symbol.clone(), reduce)
                })
                .filter(|(_, q)| *q > 0.0)
                .collect();

            return Some(VolatilitySpikeSignal {
                short_vol,
                long_vol,
                spike_ratio,
                deleverage_factor: factor,
                reductions,
            });
        } else {
            self.vol_spike_active.store(false, Ordering::SeqCst);
        }

        None
    }

    /// Check if volatility spike deleverage is active.
    pub fn is_vol_spike_active(&self) -> bool {
        self.vol_spike_active.load(Ordering::SeqCst)
    }

    // ── Correlation Monitoring ─────────────────────────────────────

    /// Check if groups of correlated positions exceed the max exposure limit.
    /// `return_series` maps symbol → recent daily returns (at least 20 days).
    pub fn check_correlated_exposure(
        &self,
        positions: &[PositionInfo],
        portfolio_value: f64,
        return_series: &std::collections::HashMap<String, Vec<f64>>,
    ) -> Option<CorrelationWarning> {
        if positions.len() < 2 || portfolio_value <= 0.0 {
            return None;
        }

        let corr_threshold = 0.7;
        let max_exposure = self.config.max_correlated_exposure;

        // Build adjacency: which positions are highly correlated
        let syms: Vec<&str> = positions.iter().map(|p| p.symbol.as_str()).collect();
        let n = syms.len();
        let mut adj = vec![vec![false; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                if let (Some(ri), Some(rj)) = (return_series.get(syms[i]), return_series.get(syms[j])) {
                    let min_len = ri.len().min(rj.len());
                    if min_len >= 20 {
                        let corr = pearson_corr(
                            &ri[ri.len() - min_len..],
                            &rj[rj.len() - min_len..],
                        );
                        if corr > corr_threshold {
                            adj[i][j] = true;
                            adj[j][i] = true;
                        }
                    }
                }
            }
        }

        // Group correlated positions via simple union-find
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            if p[x] != x { p[x] = find(p, p[x]); }
            p[x]
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] {
                    let (pi, pj) = (find(&mut parent, i), find(&mut parent, j));
                    if pi != pj { parent[pi] = pj; }
                }
            }
        }

        // Aggregate exposure per group
        let mut groups_map: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            groups_map.entry(root).or_default().push(i);
        }

        let mut warning_groups = Vec::new();
        for (_root, members) in &groups_map {
            if members.len() < 2 { continue; }
            let total_exposure: f64 = members.iter()
                .map(|&i| positions[i].market_value() / portfolio_value)
                .sum();
            if total_exposure > max_exposure {
                warning_groups.push(CorrelatedGroup {
                    symbols: members.iter().map(|&i| positions[i].symbol.clone()).collect(),
                    total_exposure,
                    max_allowed: max_exposure,
                    excess: total_exposure - max_exposure,
                });
            }
        }

        if warning_groups.is_empty() {
            None
        } else {
            Some(CorrelationWarning { groups: warning_groups })
        }
    }

    // ── Tail Risk (VaR/CVaR) ──────────────────────────────────────

    /// Estimate portfolio VaR and CVaR using historical simulation.
    /// Returns a signal if estimated VaR exceeds the configured threshold.
    pub fn estimate_tail_risk(&self) -> Option<TailRiskSignal> {
        let history = self.return_history.lock().unwrap();
        if history.len() < 30 {
            return None;
        }

        let confidence = self.config.var_confidence;
        let max_var = self.config.max_var_pct;

        // Sort returns ascending (worst first)
        let mut sorted: Vec<f64> = history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // VaR at confidence level: the (1-confidence) percentile of returns
        let var_idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        let var_idx = var_idx.min(sorted.len() - 1);
        let var_pct = -sorted[var_idx]; // positive = loss

        // CVaR (Expected Shortfall): average of returns worse than VaR
        let cvar_pct = if var_idx > 0 {
            -sorted[..=var_idx].iter().sum::<f64>() / (var_idx + 1) as f64
        } else {
            var_pct
        };

        let breach = var_pct > max_var;

        Some(TailRiskSignal {
            var_pct,
            cvar_pct,
            confidence,
            max_allowed: max_var,
            breach,
        })
    }

    // ── Combined Check ─────────────────────────────────────────────

    /// Check if a new BUY order should be blocked. Returns reason string or Ok.
    pub fn can_buy(&self) -> Result<(), String> {
        if self.is_circuit_open() {
            return Err("Circuit breaker is open: too many consecutive failures".into());
        }
        if self.is_drawdown_halted() {
            return Err(format!(
                "Max drawdown exceeded: limit {:.1}%",
                self.config.max_drawdown_pct * 100.0
            ));
        }
        if self.is_daily_paused() {
            return Err(format!(
                "Daily loss limit reached: limit {:.1}% of capital",
                self.config.max_daily_loss_pct * 100.0
            ));
        }
        if self.is_vol_spike_active() {
            return Err("Volatility spike active: new buys blocked until vol normalizes".into());
        }
        Ok(())
    }

    /// Get current status for API/dashboard.
    pub fn status(&self) -> RiskStatus {
        RiskStatus {
            daily_pnl: self.get_daily_pnl(),
            daily_paused: self.is_daily_paused(),
            peak_value: self.get_peak_value(),
            drawdown_halted: self.is_drawdown_halted(),
            consecutive_failures: self.consecutive_failures.load(Ordering::SeqCst),
            circuit_open: self.is_circuit_open(),
            vol_spike_active: self.is_vol_spike_active(),
            config: self.config.clone(),
        }
    }

    // ── Event Log ──────────────────────────────────────────────────

    /// Push a risk event into the ring buffer (max 200 entries).
    pub fn push_event(&self, severity: RiskSeverity, event_type: &str, message: &str, detail: Option<serde_json::Value>) {
        let event = RiskEvent {
            timestamp: Utc::now(),
            severity,
            event_type: event_type.to_string(),
            message: message.to_string(),
            detail,
        };
        let mut log = self.event_log.lock().unwrap();
        if log.len() >= 200 {
            log.remove(0);
        }
        log.push(event);
    }

    /// Get recent risk events (newest last).
    pub fn recent_events(&self, limit: usize) -> Vec<RiskEvent> {
        let log = self.event_log.lock().unwrap();
        let n = log.len();
        let skip = n.saturating_sub(limit);
        log[skip..].to_vec()
    }

    /// Get a comprehensive risk signals snapshot (for monitoring dashboard).
    pub fn risk_signals_snapshot(&self) -> RiskSignalsSnapshot {
        let tail_risk = self.estimate_tail_risk();
        let return_count = self.return_history.lock().unwrap().len();

        RiskSignalsSnapshot {
            status: self.status(),
            vol_spike_active: self.is_vol_spike_active(),
            tail_risk,
            return_history_len: return_count,
            recent_events: self.recent_events(50),
        }
    }

    /// Get config reference.
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }
}

// ── Helper Functions ──────────────────────────────────────────────

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 { return 0.0; }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 { return 0.0; }
    let mean_a = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b = b[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 { 0.0 } else { cov / denom }
}

/// Risk enforcer status snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct RiskStatus {
    pub daily_pnl: f64,
    pub daily_paused: bool,
    pub peak_value: f64,
    pub drawdown_halted: bool,
    pub consecutive_failures: u32,
    pub circuit_open: bool,
    pub vol_spike_active: bool,
    pub config: RiskConfig,
}

/// Comprehensive risk signals snapshot for monitoring dashboard.
#[derive(Debug, Clone, Serialize)]
pub struct RiskSignalsSnapshot {
    pub status: RiskStatus,
    pub vol_spike_active: bool,
    pub tail_risk: Option<TailRiskSignal>,
    pub return_history_len: usize,
    pub recent_events: Vec<RiskEvent>,
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_enforcer() -> RiskEnforcer {
        RiskEnforcer::new(RiskConfig::default(), 1_000_000.0)
    }

    #[test]
    fn test_stop_loss_triggers() {
        let enforcer = default_enforcer();
        let now = chrono::Utc::now().naive_utc();
        let positions = vec![
            PositionInfo {
                symbol: "600519.SH".into(),
                quantity: 100.0,
                avg_cost: 100.0,
                current_price: 93.0, // -7% loss, threshold is 5%
                entry_time: now,
                target_weight: 0.0,
            },
            PositionInfo {
                symbol: "000858.SZ".into(),
                quantity: 200.0,
                avg_cost: 50.0,
                current_price: 48.0, // -4% loss, below threshold
                entry_time: now,
                target_weight: 0.0,
            },
        ];
        let signals = enforcer.check_stop_losses(&positions);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].symbol, "600519.SH");
        assert!(signals[0].loss_pct < -0.05);
    }

    #[test]
    fn test_stop_loss_no_trigger() {
        let enforcer = default_enforcer();
        let now = chrono::Utc::now().naive_utc();
        let positions = vec![PositionInfo {
            symbol: "600519.SH".into(),
            quantity: 100.0,
            avg_cost: 100.0,
            current_price: 98.0, // -2%, within threshold
            entry_time: now,
            target_weight: 0.0,
        }];
        assert!(enforcer.check_stop_losses(&positions).is_empty());
    }

    #[test]
    fn test_daily_loss_limit() {
        let enforcer = default_enforcer(); // 3% of 1M = 30k
        assert!(!enforcer.is_daily_paused());
        enforcer.record_realized_pnl(-20_000.0);
        assert!(!enforcer.is_daily_paused());
        enforcer.record_realized_pnl(-15_000.0); // total -35k > 30k
        assert!(enforcer.is_daily_paused());
        assert!(enforcer.can_buy().is_err());
    }

    #[test]
    fn test_daily_reset() {
        let enforcer = default_enforcer();
        enforcer.record_realized_pnl(-50_000.0);
        assert!(enforcer.is_daily_paused());
        enforcer.reset_daily();
        assert!(!enforcer.is_daily_paused());
        assert_eq!(enforcer.get_daily_pnl(), 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let enforcer = default_enforcer(); // 10% drawdown limit
        // Portfolio goes up to 1.1M
        assert!(!enforcer.update_portfolio_value(1_100_000.0));
        // Then drops to 980k — drawdown = (1.1M - 980k) / 1.1M = 10.9%
        assert!(enforcer.update_portfolio_value(980_000.0));
        assert!(enforcer.is_drawdown_halted());
        assert!(enforcer.can_buy().is_err());
    }

    #[test]
    fn test_drawdown_within_limit() {
        let enforcer = default_enforcer();
        enforcer.update_portfolio_value(1_000_000.0);
        // Drop to 920k — drawdown = 8% < 10%
        assert!(!enforcer.update_portfolio_value(920_000.0));
        assert!(!enforcer.is_drawdown_halted());
    }

    #[test]
    fn test_circuit_breaker() {
        let config = RiskConfig {
            circuit_breaker_failures: 3,
            ..Default::default()
        };
        let enforcer = RiskEnforcer::new(config, 1_000_000.0);

        assert!(!enforcer.is_circuit_open());
        assert!(!enforcer.record_failure()); // 1
        assert!(!enforcer.record_failure()); // 2
        assert!(enforcer.record_failure());  // 3 — trips
        assert!(enforcer.is_circuit_open());
        assert!(enforcer.can_buy().is_err());

        enforcer.reset_circuit_breaker();
        assert!(!enforcer.is_circuit_open());
        assert!(enforcer.can_buy().is_ok());
    }

    #[test]
    fn test_circuit_breaker_reset_on_success() {
        let config = RiskConfig {
            circuit_breaker_failures: 3,
            ..Default::default()
        };
        let enforcer = RiskEnforcer::new(config, 1_000_000.0);
        enforcer.record_failure(); // 1
        enforcer.record_failure(); // 2
        enforcer.record_success(); // resets to 0
        assert!(!enforcer.record_failure()); // 1 again
        assert!(!enforcer.is_circuit_open());
    }

    #[test]
    fn test_can_buy_all_clear() {
        let enforcer = default_enforcer();
        assert!(enforcer.can_buy().is_ok());
    }

    #[test]
    fn test_status_snapshot() {
        let enforcer = default_enforcer();
        enforcer.record_realized_pnl(-5_000.0);
        let status = enforcer.status();
        assert_eq!(status.daily_pnl, -5_000.0);
        assert!(!status.daily_paused);
        assert!(!status.circuit_open);
        assert!(!status.drawdown_halted);
        assert!(!status.vol_spike_active);
    }

    #[test]
    fn test_volatility_spike_detection() {
        let enforcer = default_enforcer();
        let now = chrono::Utc::now().naive_utc();
        let positions = vec![PositionInfo {
            symbol: "600519.SH".into(),
            quantity: 1000.0,
            avg_cost: 100.0,
            current_price: 100.0,
            entry_time: now,
            target_weight: 0.0,
        }];

        // Feed 60 days of calm returns (gentle random walk ±0.3%)
        let mut val = 1_000_000.0;
        for i in 0..60 {
            // Small daily moves: +0.3% or -0.2% alternating
            val *= if i % 2 == 0 { 1.003 } else { 0.998 };
            assert!(enforcer.check_volatility_spike(val, &positions).is_none());
        }
        assert!(!enforcer.is_vol_spike_active());

        // Crash: alternating -5%/+1% days (high dispersion = high vol)
        let mut triggered = false;
        for i in 0..5 {
            val *= if i % 2 == 0 { 0.95 } else { 1.01 };
            if enforcer.check_volatility_spike(val, &positions).is_some() {
                triggered = true;
            }
        }
        assert!(triggered, "Volatility spike should have triggered");
        assert!(enforcer.is_vol_spike_active());
        assert!(enforcer.can_buy().is_err());
    }

    #[test]
    fn test_tail_risk_var() {
        let enforcer = default_enforcer();
        // Feed 60 returns including some bad days
        for i in 0..55 {
            let ret = if i % 10 == 0 { 0.97 } else { 1.001 }; // occasional -3% day
            let val = 1_000_000.0 * ret;
            enforcer.check_volatility_spike(val, &[]);
        }
        // Add a few crash days
        for _ in 0..5 {
            enforcer.check_volatility_spike(950_000.0, &[]);
        }

        let signal = enforcer.estimate_tail_risk();
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert!(s.var_pct > 0.0, "VaR should be positive (representing loss)");
        assert!(s.cvar_pct >= s.var_pct, "CVaR should be >= VaR");
    }

    #[test]
    fn test_correlation_monitoring() {
        let enforcer = default_enforcer();
        let now = chrono::Utc::now().naive_utc();
        let positions = vec![
            PositionInfo {
                symbol: "A".into(), quantity: 1000.0, avg_cost: 100.0,
                current_price: 100.0, entry_time: now, target_weight: 0.0,
            },
            PositionInfo {
                symbol: "B".into(), quantity: 1000.0, avg_cost: 50.0,
                current_price: 50.0, entry_time: now, target_weight: 0.0,
            },
        ];
        let portfolio_value = 150_000.0;

        // Create highly correlated return series
        let base: Vec<f64> = (0..30).map(|i| 0.01 * (i as f64).sin()).collect();
        let mut returns = std::collections::HashMap::new();
        returns.insert("A".to_string(), base.clone());
        returns.insert("B".to_string(), base.iter().map(|r| r * 1.1 + 0.001).collect());

        let warning = enforcer.check_correlated_exposure(&positions, portfolio_value, &returns);
        assert!(warning.is_some(), "Should detect correlated exposure");
        let w = warning.unwrap();
        assert!(!w.groups.is_empty());
        assert_eq!(w.groups[0].symbols.len(), 2);
    }

    #[test]
    fn test_event_log_and_snapshot() {
        let enforcer = default_enforcer();
        // Trigger daily loss → event logged
        enforcer.record_realized_pnl(-50_000.0);
        let events = enforcer.recent_events(10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].severity, RiskSeverity::Critical);
        assert_eq!(events[0].event_type, "daily_loss_limit");

        // Trigger circuit breaker → event logged
        for _ in 0..5 {
            enforcer.record_failure();
        }
        let events = enforcer.recent_events(10);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].event_type, "circuit_breaker");

        // Snapshot includes status + events
        let snap = enforcer.risk_signals_snapshot();
        assert!(snap.status.daily_paused);
        assert!(snap.status.circuit_open);
        assert_eq!(snap.recent_events.len(), 2);
    }

    #[test]
    fn test_event_log_ring_buffer() {
        let enforcer = default_enforcer();
        for i in 0..210 {
            enforcer.push_event(RiskSeverity::Info, "test", &format!("event {}", i), None);
        }
        let events = enforcer.recent_events(300);
        assert_eq!(events.len(), 200); // capped at 200
        assert!(events[0].message.contains("10")); // oldest should be event 10
    }
}
