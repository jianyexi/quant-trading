/// Active risk enforcement: stop-loss, daily loss limit, max drawdown, circuit breaker.
///
/// These are stateful runtime checks that the engine evaluates continuously,
/// as opposed to the pre-trade checks in `checks.rs`.

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
}

fn default_timeout_min_profit() -> f64 { 0.02 }
fn default_rebalance_threshold() -> f64 { 0.05 }

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
            self.daily_paused.store(true, Ordering::SeqCst);
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
            self.drawdown_halted.store(true, Ordering::SeqCst);
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

    /// Record a successful order execution. Resets failure counter.
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::SeqCst);
    }

    /// Record a failed order. Increments counter and may trip circuit breaker.
    pub fn record_failure(&self) -> bool {
        let n = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
        if n >= self.config.circuit_breaker_failures {
            self.circuit_open.store(true, Ordering::SeqCst);
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
            config: self.config.clone(),
        }
    }

    /// Get config reference.
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }
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
    pub config: RiskConfig,
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
    }
}
