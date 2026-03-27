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

    // Stage 1: Fetch data once (shared across all combos)
    ts.set_progress(tid, "📊 Fetching market data...");

    let klines = match fetch_real_klines_with_period(&req.symbol, &req.start, &req.end, period) {
        Ok(k) if !k.is_empty() => k,
        Ok(_) => {
            ts.fail(tid, &format!(
                "无法获取 {} 的行情数据：数据源返回空数据。请检查股票代码是否正确。",
                req.symbol
            ));
            return;
        }
        Err(reason) => {
            ts.fail(tid, &format!(
                "无法获取 {} 的行情数据：{}。请检查数据源连接。",
                req.symbol, reason
            ));
            return;
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

    let mut grid: Vec<Value> = Vec::with_capacity(total);
    let mut best_sharpe = f64::NEG_INFINITY;
    let mut best_entry: Option<Value> = None;
    let mut completed = 0usize;

    for &p1 in &req.param1_values {
        for &p2 in &req.param2_values {
            // Skip invalid combos where fast >= slow for MA-based strategies
            if is_ma_cross && is_fast_slow_pair && p1 >= p2 {
                let entry = json!({
                    "param1": p1,
                    "param2": p2,
                    "total_return": null,
                    "sharpe": null,
                    "max_drawdown": null,
                    "win_rate": null,
                    "skipped": true,
                });
                grid.push(entry);
                completed += 1;
                continue;
            }

            // Build strategy config with these params
            let params_map = json!({
                req.param1_name.clone(): p1,
                req.param2_name.clone(): p2,
            });

            let created = match create_strategy_with_params(&req.strategy, &params_map) {
                Ok(c) => c,
                Err(e) => {
                    let entry = json!({
                        "param1": p1,
                        "param2": p2,
                        "total_return": null,
                        "sharpe": null,
                        "max_drawdown": null,
                        "win_rate": null,
                        "error": e,
                    });
                    grid.push(entry);
                    completed += 1;
                    continue;
                }
            };

            let mut strategy = created.strategy;
            let engine = BacktestEngine::new(bt_config.clone());
            let result = engine.run_with_benchmark(strategy.as_mut(), &klines, None);

            let m = &result.metrics;
            let total_return = (m.total_return * 10000.0).round() / 100.0;
            let sharpe = (m.sharpe_ratio * 100.0).round() / 100.0;
            let max_drawdown = (m.max_drawdown * 10000.0).round() / 100.0;
            let win_rate = (m.win_rate * 10000.0).round() / 100.0;

            let entry = json!({
                "param1": p1,
                "param2": p2,
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
            });

            if sharpe > best_sharpe {
                best_sharpe = sharpe;
                best_entry = Some(json!({
                    "param1": p1,
                    "param2": p2,
                    "total_return": total_return,
                    "sharpe": sharpe,
                }));
            }

            grid.push(entry);
            completed += 1;

            // Update progress periodically
            if completed % 5 == 0 || completed == total {
                ts.set_progress(tid, &format!(
                    "🔄 Optimizing... {}/{} combinations completed",
                    completed, total
                ));
            }
        }
    }

    let report = json!({
        "status": "completed",
        "grid": grid,
        "best": best_entry,
        "param1_name": req.param1_name,
        "param2_name": req.param2_name,
        "strategy": req.strategy,
        "symbol": req.symbol,
        "total_combinations": total,
        "completed_combinations": completed,
    });

    ts.complete(tid, &report.to_string());
}
