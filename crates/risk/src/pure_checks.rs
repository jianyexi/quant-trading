/// Pure, stateless risk check functions.
///
/// These are the canonical implementations shared by both the live `RiskEnforcer`
/// and the `BacktestEngine`. Each returns `Some(value)` when a threshold is
/// breached, or `None` when within limits.

/// Check if a position should be stopped out.
/// `price` can be current_price (live) or kline.low (backtest worst-case).
/// Returns the loss fraction (negative) if breached.
pub fn check_stop_loss(price: f64, avg_cost: f64, threshold: f64) -> Option<f64> {
    if threshold <= 0.0 || avg_cost <= 0.0 {
        return None;
    }
    let loss_pct = (price - avg_cost) / avg_cost;
    if loss_pct < -threshold {
        Some(loss_pct)
    } else {
        None
    }
}

/// Check if daily loss limit is breached.
/// Returns the daily PnL fraction (negative) if breached.
pub fn check_daily_loss(current_value: f64, day_start_value: f64, threshold: f64) -> Option<f64> {
    if threshold <= 0.0 || day_start_value <= 0.0 {
        return None;
    }
    let pnl_pct = (current_value - day_start_value) / day_start_value;
    if pnl_pct < -threshold {
        Some(pnl_pct)
    } else {
        None
    }
}

/// Check if max drawdown is breached.
/// Returns the drawdown fraction (positive) if breached.
pub fn check_drawdown(current_value: f64, peak_value: f64, threshold: f64) -> Option<f64> {
    if threshold <= 0.0 || peak_value <= 0.0 {
        return None;
    }
    let dd = (peak_value - current_value) / peak_value;
    if dd > threshold {
        Some(dd)
    } else {
        None
    }
}

/// Check if holding timeout is reached.
/// Returns holding days if max exceeded.
pub fn check_holding_timeout(holding_days: i64, max_days: u32) -> Option<i64> {
    if max_days == 0 {
        return None;
    }
    if holding_days > max_days as i64 {
        Some(holding_days)
    } else {
        None
    }
}

/// Cap buy budget by concentration limit.
/// Returns `(capped_budget, was_capped)`.
pub fn cap_by_concentration(
    budget: f64,
    existing_value: f64,
    total_value: f64,
    max_concentration: f64,
) -> (f64, bool) {
    let max_allowed = total_value * max_concentration - existing_value;
    let capped = budget.min(max_allowed.max(0.0));
    (capped, capped < budget - 1e-10) // small epsilon for float comparison
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_loss_triggered() {
        // Price dropped 10% from cost, threshold 8%
        assert!(check_stop_loss(90.0, 100.0, 0.08).is_some());
        let loss = check_stop_loss(90.0, 100.0, 0.08).unwrap();
        assert!((loss - (-0.10)).abs() < 1e-10);
    }

    #[test]
    fn stop_loss_not_triggered() {
        // Price dropped 3%, threshold 8%
        assert!(check_stop_loss(97.0, 100.0, 0.08).is_none());
    }

    #[test]
    fn stop_loss_disabled() {
        assert!(check_stop_loss(50.0, 100.0, 0.0).is_none());
    }

    #[test]
    fn daily_loss_triggered() {
        // Lost 4%, threshold 3%
        assert!(check_daily_loss(96000.0, 100000.0, 0.03).is_some());
    }

    #[test]
    fn daily_loss_not_triggered() {
        assert!(check_daily_loss(98000.0, 100000.0, 0.03).is_none());
    }

    #[test]
    fn drawdown_triggered() {
        // 12% drawdown, threshold 10%
        assert!(check_drawdown(88000.0, 100000.0, 0.10).is_some());
        let dd = check_drawdown(88000.0, 100000.0, 0.10).unwrap();
        assert!((dd - 0.12).abs() < 1e-10);
    }

    #[test]
    fn drawdown_not_triggered() {
        assert!(check_drawdown(95000.0, 100000.0, 0.10).is_none());
    }

    #[test]
    fn holding_timeout_triggered() {
        assert_eq!(check_holding_timeout(35, 30), Some(35));
    }

    #[test]
    fn holding_timeout_disabled() {
        assert!(check_holding_timeout(35, 0).is_none());
    }

    #[test]
    fn concentration_cap() {
        // Budget 50k, existing 20k, total 100k, max 30%
        let (capped, was_capped) = cap_by_concentration(50000.0, 20000.0, 100000.0, 0.30);
        assert!((capped - 10000.0).abs() < 1e-10); // max_allowed = 100k * 0.3 - 20k = 10k
        assert!(was_capped);
    }

    #[test]
    fn concentration_no_cap() {
        let (capped, was_capped) = cap_by_concentration(5000.0, 0.0, 100000.0, 0.30);
        assert!((capped - 5000.0).abs() < 1e-10);
        assert!(!was_capped);
    }
}
