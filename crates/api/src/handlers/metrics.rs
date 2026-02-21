use axum::{extract::State, Json};
use serde_json::{json, Value};
use crate::state::AppState;

/// GET /api/metrics — comprehensive system metrics for debugging & monitoring
pub async fn get_metrics(State(state): State<AppState>) -> Json<Value> {
    let engine_guard = state.engine.lock().await;

    // Engine metrics
    let engine_metrics = if let Some(ref engine) = *engine_guard {
        let status = engine.status().await;
        let risk = engine.risk_enforcer().status();
        json!({
            "running": status.running,
            "strategy": status.strategy,
            "symbols": status.symbols,
            "throughput": {
                "total_bars": status.latency.total_bars_processed,
                "total_signals": status.total_signals,
                "total_orders": status.total_orders,
                "total_fills": status.total_fills,
                "total_rejected": status.total_rejected,
                "signal_rate": if status.latency.total_bars_processed > 0 {
                    status.total_signals as f64 / status.latency.total_bars_processed as f64
                } else { 0.0 },
                "fill_rate": if status.total_orders > 0 {
                    status.total_fills as f64 / status.total_orders as f64 * 100.0
                } else { 0.0 },
                "reject_rate": if status.total_orders > 0 {
                    status.total_rejected as f64 / status.total_orders as f64 * 100.0
                } else { 0.0 },
            },
            "latency": {
                "last_factor_us": status.latency.last_factor_compute_us,
                "avg_factor_us": status.latency.avg_factor_compute_us,
                "last_risk_us": status.latency.last_risk_check_us,
                "last_order_us": status.latency.last_order_submit_us,
            },
            "performance": {
                "portfolio_value": status.performance.portfolio_value,
                "pnl": status.pnl,
                "total_return_pct": status.performance.total_return_pct,
                "max_drawdown_pct": status.performance.max_drawdown_pct,
                "win_rate": status.performance.win_rate,
                "profit_factor": status.performance.profit_factor,
                "wins": status.performance.wins,
                "losses": status.performance.losses,
            },
            "risk": {
                "daily_pnl": risk.daily_pnl,
                "daily_paused": risk.daily_paused,
                "circuit_open": risk.circuit_open,
                "drawdown_halted": risk.drawdown_halted,
                "peak_value": risk.peak_value,
                "consecutive_failures": risk.consecutive_failures,
            },
        })
    } else {
        json!({ "running": false })
    };
    drop(engine_guard);

    // API request metrics from LogStore
    let (info_count, warn_count, error_count) = state.log_store.summary();
    let total_requests = info_count + warn_count + error_count;
    let recent_logs = state.log_store.query(None, None, 100);
    let avg_duration_ms = if recent_logs.is_empty() {
        0.0
    } else {
        recent_logs.iter().map(|l| l.duration_ms as f64).sum::<f64>() / recent_logs.len() as f64
    };
    let p99_duration_ms = if recent_logs.is_empty() {
        0
    } else {
        let mut durations: Vec<u64> = recent_logs.iter().map(|l| l.duration_ms).collect();
        durations.sort_unstable();
        durations[durations.len() * 99 / 100]
    };

    // Endpoint breakdown: group by path, count calls & avg latency
    let mut endpoint_stats: std::collections::HashMap<String, (u64, u64)> = std::collections::HashMap::new();
    for log in &recent_logs {
        let entry = endpoint_stats.entry(log.path.clone()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += log.duration_ms;
    }
    let endpoints: Vec<Value> = endpoint_stats.iter()
        .map(|(path, (count, total_ms))| {
            json!({ "path": path, "calls": count, "avg_ms": *total_ms as f64 / *count as f64 })
        })
        .collect();

    // DB status
    let db_status = if let Some(ref pool) = state.db {
        let pool_size = pool.size();
        let idle = pool.num_idle();
        json!({
            "backend": "postgresql",
            "pool_size": pool_size,
            "idle_connections": idle,
            "active_connections": pool_size - idle as u32,
        })
    } else {
        json!({ "backend": "sqlite" })
    };

    // System info
    let uptime = {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let start = START.get_or_init(std::time::Instant::now);
        start.elapsed().as_secs()
    };

    Json(json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_secs": uptime,
        "engine": engine_metrics,
        "api": {
            "total_requests": total_requests,
            "info": info_count,
            "warnings": warn_count,
            "errors": error_count,
            "avg_duration_ms": (avg_duration_ms * 100.0).round() / 100.0,
            "p99_duration_ms": p99_duration_ms,
            "endpoints": endpoints,
        },
        "database": db_status,
    }))
}
