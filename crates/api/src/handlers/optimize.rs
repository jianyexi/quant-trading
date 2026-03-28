use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::state::AppState;
use super::market::fetch_real_klines_with_period;

#[derive(Debug, Deserialize)]
pub struct OptimizeRequest {
    pub strategy: String,
    pub symbol: String,
    pub start: String,
    pub end: String,
    pub capital: Option<f64>,
    pub period: Option<String>,
    pub param1_name: String,
    pub param1_values: Vec<f64>,
    pub param2_name: String,
    pub param2_values: Vec<f64>,
}

pub async fn run_optimization(
    State(state): State<AppState>,
    Json(req): Json<OptimizeRequest>,
) -> (StatusCode, Json<Value>) {
    use std::sync::Arc;

    let capital = req.capital.unwrap_or(1_000_000.0);
    let period = req.period.clone().unwrap_or_else(|| "daily".to_string());

    let total_combinations = req.param1_values.len() * req.param2_values.len();
    if total_combinations == 0 {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "参数值列表不能为空"
        })));
    }

    let ts = state.task_store.clone();
    let params_json = serde_json::to_string(&json!({
        "strategy": req.strategy,
        "symbol": req.symbol,
        "start": req.start,
        "end": req.end,
        "capital": capital,
        "period": period,
        "param1_name": req.param1_name,
        "param1_values": req.param1_values,
        "param2_name": req.param2_name,
        "param2_values": req.param2_values,
        "total_combinations": total_combinations,
    })).unwrap_or_default();

    let task_id = ts.create_with_params("optimize", Some(&params_json));
    let tid = task_id.clone();

    let ts2 = Arc::clone(&ts);
    tokio::task::spawn_blocking(move || {
        run_optimization_task(&ts2, &tid, &req, capital, &period);
    });

    (StatusCode::OK, Json(json!({
        "task_id": task_id,
        "status": "Running",
        "total_combinations": total_combinations,
    })))
}

fn run_optimization_task(
    ts: &crate::task_store::TaskStore,
    tid: &str,
    req: &OptimizeRequest,
    capital: f64,
    period: &str,
) {
    use quant_backtest::engine::{BacktestConfig, BacktestEngine};
    use quant_strategy::factory::create_strategy_with_params;
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Stage 1: Fetch data once (shared across all combos)
    ts.set_progress(tid, "📊 Fetching market data...");

    let klines = match fetch_real_klines_with_period(&req.symbol, &req.start, &req.end, period) {
        Ok(k) if !k.is_empty() => k,
        _ => {
            // Fallback: generate synthetic data
            let generated = super::market::generate_backtest_klines(&req.symbol, &req.start, &req.end);
            if generated.is_empty() {
                ts.fail(tid, &format!("无法获取 {} 的行情数据，生成模拟数据也失败。", req.symbol));
                return;
            }
            generated
        }
    };

    let total = req.param1_values.len() * req.param2_values.len();
    ts.set_progress(tid, &format!(
        "🔧 Data loaded ({} bars). Running {} parameter combinations...",
        klines.len(), total
    ));

    let bt_config = BacktestConfig {
        initial_capital: capital,
        commission_rate: 0.001,
        stamp_tax_rate: 0.001,
        slippage_ticks: 1,
        position_size_pct: 0.3,
        max_concentration: 0.30,
        stop_loss_pct: 0.08,
        max_holding_days: 30,
        daily_loss_limit: 0.03,
        max_drawdown_limit: 0.15,
        use_atr_sizing: true,
        atr_period: 14,
        risk_per_trade: 0.02,
        benchmark_symbol: None,
    };

    // Determine whether to skip invalid combos (param1 >= param2 for MA cross)
    let is_ma_cross = matches!(req.strategy.as_str(), "sma_cross" | "macd_trend");
    let is_fast_slow_pair = req.param1_name.contains("fast") && req.param2_name.contains("slow");

    // Build all parameter combinations for parallel iteration
    let combinations: Vec<(f64, f64)> = req.param1_values.iter()
        .flat_map(|&p1| req.param2_values.iter().map(move |&p2| (p1, p2)))
        .collect();

    let completed_count = AtomicUsize::new(0);

    let grid: Vec<Value> = combinations.par_iter().map(|&(p1, p2)| {
        // Skip invalid combos where fast >= slow for MA-based strategies
        if is_ma_cross && is_fast_slow_pair && p1 >= p2 {
            completed_count.fetch_add(1, Ordering::Relaxed);
            return json!({
                "param1": p1,
                "param2": p2,
                "total_return": null,
                "sharpe": null,
                "max_drawdown": null,
                "win_rate": null,
                "skipped": true,
            });
        }

        // Build strategy config with these params
        let params_map = json!({
            req.param1_name.clone(): p1,
            req.param2_name.clone(): p2,
        });

        let entry = match create_strategy_with_params(&req.strategy, &params_map) {
            Ok(created) => {
                let mut strategy = created.strategy;
                let engine = BacktestEngine::new(bt_config.clone());
                let result = engine.run_with_benchmark(strategy.as_mut(), &klines, None);

                let m = &result.metrics;
                json!({
                    "param1": p1,
                    "param2": p2,
                    "total_return": (m.total_return * 10000.0).round() / 100.0,
                    "sharpe": (m.sharpe_ratio * 100.0).round() / 100.0,
                    "max_drawdown": (m.max_drawdown * 10000.0).round() / 100.0,
                    "win_rate": (m.win_rate * 10000.0).round() / 100.0,
                })
            }
            Err(e) => {
                json!({
                    "param1": p1,
                    "param2": p2,
                    "total_return": null,
                    "sharpe": null,
                    "max_drawdown": null,
                    "win_rate": null,
                    "error": e,
                })
            }
        };

        let done = completed_count.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 5 == 0 || done == total {
            ts.set_progress(tid, &format!(
                "🔄 Optimizing... {}/{} combinations completed",
                done, total
            ));
        }

        entry
    }).collect();

    // Find the best entry by sharpe ratio after parallel collection
    let best_entry = grid.iter()
        .filter(|e| e.get("sharpe").and_then(|s| s.as_f64()).is_some())
        .max_by(|a, b| {
            let sa = a["sharpe"].as_f64().unwrap_or(f64::NEG_INFINITY);
            let sb = b["sharpe"].as_f64().unwrap_or(f64::NEG_INFINITY);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|e| json!({
            "param1": e["param1"],
            "param2": e["param2"],
            "total_return": e["total_return"],
            "sharpe": e["sharpe"],
        }));

    let report = json!({
        "status": "completed",
        "grid": grid,
        "best": best_entry,
        "param1_name": req.param1_name,
        "param2_name": req.param2_name,
        "strategy": req.strategy,
        "symbol": req.symbol,
        "total_combinations": total,
        "completed_combinations": grid.len(),
    });

    ts.complete(tid, &report.to_string());
}
