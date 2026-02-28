use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde_json::{json, Value};

use tracing::debug;

use crate::state::AppState;
use super::BacktestRequest;
use super::market::fetch_real_klines_with_period;

pub async fn run_backtest(
    State(state): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> (StatusCode, Json<Value>) {
    use std::sync::Arc;

    let capital = req.capital.unwrap_or(1_000_000.0);
    let period = req.period.clone().unwrap_or_else(|| "daily".to_string());
    debug!(symbol=%req.symbol, "Backtest started");

    let ts = state.task_store.clone();
    let task_id = ts.create("backtest");
    let tid = task_id.clone();

    let ts2 = Arc::clone(&ts);
    tokio::task::spawn_blocking(move || {
        run_backtest_task(&ts2, &tid, &req, capital, &period);
    });

    (StatusCode::OK, Json(json!({
        "task_id": task_id,
        "status": "Running",
        "progress": "Backtest started..."
    })))
}

fn run_backtest_task(
    ts: &crate::task_store::TaskStore,
    tid: &str,
    req: &BacktestRequest,
    capital: f64,
    period: &str,
) {
    use quant_backtest::engine::{BacktestConfig, BacktestEngine};
    use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum, MultiFactorStrategy, MultiFactorConfig};
    use quant_strategy::ml_factor::{MlFactorStrategy, MlFactorConfig};

    // Stage 1: Fetch data
    ts.set_progress(tid, "📊 Fetching market data...");

    let (klines, data_source) = match fetch_real_klines_with_period(&req.symbol, &req.start, &req.end, period) {
        Ok(k) if !k.is_empty() => {
            let n = k.len();
            let label = if period == "daily" { "日线" } else { &format!("{}分钟线", period) };
            (k, format!("akshare ({}条真实{})", n, label))
        }
        Ok(_) | Err(_) if period != "daily" => {
            ts.fail(tid, &format!("无法获取{}分钟级数据。分钟K线仅支持近5个交易日。", period));
            return;
        }
        Ok(_) => {
            ts.fail(tid, &format!("无法获取 {} 的行情数据：数据源返回空数据。请检查股票代码是否正确，或先同步缓存数据。", req.symbol));
            return;
        }
        Err(reason) => {
            ts.fail(tid, &format!("无法获取 {} 的行情数据：{}。请检查数据源连接或先同步缓存数据。", req.symbol, reason));
            return;
        }
    };

    if klines.is_empty() {
        ts.fail(tid, "No kline data for date range");
        return;
    }

    ts.set_progress(tid, &format!("📊 Data loaded ({} bars). Initializing strategy...", klines.len()));

    // Stage 2: Build strategy
    let mut active_inference_mode = String::new();
    let mut strategy: Box<dyn quant_core::traits::Strategy> = match req.strategy.as_str() {
        "sma_cross" | "DualMaCrossover" => Box::new(DualMaCrossover::new(5, 20)),
        "rsi_reversal" | "RsiMeanReversion" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
        "macd_trend" | "MacdMomentum" => Box::new(MacdMomentum::new(12, 26, 9)),
        "multi_factor" | "MultiFactorModel" => Box::new(MultiFactorStrategy::new(MultiFactorConfig::default())),
        "ml_factor" | "MlFactor" => {
            let mode = req.inference_mode.as_deref().unwrap_or("embedded");
            let ml_cfg = MlFactorConfig {
                inference_mode: quant_strategy::ml_factor::MlInferenceMode::from_str(mode),
                ..Default::default()
            };
            let ml_strategy = MlFactorStrategy::new(ml_cfg);
            active_inference_mode = format!("{}", ml_strategy.active_mode());
            Box::new(ml_strategy)
        }
        _ => Box::new(DualMaCrossover::new(5, 20)),
    };

    // Stage 3: Run backtest
    ts.set_progress(tid, &format!("🚀 Running backtest on {} bars...", klines.len()));

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
    };

    let engine = BacktestEngine::new(bt_config);
    let result = engine.run(strategy.as_mut(), &klines);

    // Stage 4: Compute metrics
    ts.set_progress(tid, "📈 Computing metrics and building report...");

    let eq_fmt = if period == "daily" { "%Y-%m-%d" } else { "%Y-%m-%d %H:%M" };
    let equity_curve: Vec<Value> = result.equity_curve.iter().map(|(dt, val)| {
        json!({ "date": dt.format(eq_fmt).to_string(), "value": (*val * 100.0).round() / 100.0 })
    }).collect();

    let trades: Vec<Value> = result.trades.iter().map(|t| {
        json!({
            "date": t.timestamp.format("%Y-%m-%d %H:%M").to_string(),
            "symbol": t.symbol,
            "side": if t.side == quant_core::types::OrderSide::Buy { "BUY" } else { "SELL" },
            "price": (t.price * 100.0).round() / 100.0,
            "quantity": t.quantity as i64,
            "commission": (t.commission * 100.0).round() / 100.0,
        })
    }).collect();

    // Serialize events (only action events, skip PortfolioSnapshot to keep response manageable)
    let events: Vec<Value> = result.events.iter().filter(|ev| {
        !matches!(ev, quant_backtest::engine::BacktestEvent::PortfolioSnapshot { .. })
    }).map(|ev| serde_json::to_value(ev).unwrap_or_default()).collect();

    // Portfolio snapshots as a separate array (sampled for large datasets)
    let snapshots: Vec<Value> = result.events.iter().filter_map(|ev| {
        if let quant_backtest::engine::BacktestEvent::PortfolioSnapshot {
            timestamp, cash, total_value, n_positions, unrealized_pnl, drawdown_pct, ..
        } = ev {
            Some(json!({
                "date": timestamp.format(eq_fmt).to_string(),
                "cash": (*cash * 100.0).round() / 100.0,
                "total_value": (*total_value * 100.0).round() / 100.0,
                "n_positions": n_positions,
                "unrealized_pnl": (*unrealized_pnl * 100.0).round() / 100.0,
                "drawdown_pct": (*drawdown_pct * 100.0).round() / 100.0,
            }))
        } else {
            None
        }
    }).collect();

    // Per-symbol metrics
    let symbol_metrics: Vec<Value> = result.symbol_metrics.iter().map(|sm| {
        json!({
            "symbol": sm.symbol,
            "total_trades": sm.total_trades,
            "winning_trades": sm.winning_trades,
            "losing_trades": sm.losing_trades,
            "total_pnl": (sm.total_pnl * 100.0).round() / 100.0,
            "win_rate_pct": (sm.win_rate * 10000.0).round() / 100.0,
            "avg_holding_days": (sm.avg_holding_days * 10.0).round() / 10.0,
            "max_win": (sm.max_win * 100.0).round() / 100.0,
            "max_loss": (sm.max_loss * 100.0).round() / 100.0,
        })
    }).collect();

    // Event summary counts
    let n_signals = result.events.iter().filter(|e| matches!(e, quant_backtest::engine::BacktestEvent::Signal { .. })).count();
    let n_risk = result.events.iter().filter(|e| matches!(e, quant_backtest::engine::BacktestEvent::RiskTriggered { .. })).count();
    let n_rejected = result.events.iter().filter(|e| matches!(e, quant_backtest::engine::BacktestEvent::OrderRejected { .. })).count();
    let n_pos_opened = result.events.iter().filter(|e| matches!(e, quant_backtest::engine::BacktestEvent::PositionOpened { .. })).count();
    let n_pos_closed = result.events.iter().filter(|e| matches!(e, quant_backtest::engine::BacktestEvent::PositionClosed { .. })).count();

    let m = &result.metrics;
    let id = format!("bt-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());

    let report = json!({
        "id": id,
        "strategy": req.strategy,
        "symbol": req.symbol,
        "start": req.start,
        "end": req.end,
        "initial_capital": capital,
        "final_value": (result.final_portfolio.total_value * 100.0).round() / 100.0,
        "total_return_percent": (m.total_return * 10000.0).round() / 100.0,
        "annual_return_percent": (m.annual_return * 10000.0).round() / 100.0,
        "sharpe_ratio": (m.sharpe_ratio * 100.0).round() / 100.0,
        "sortino_ratio": (m.sortino_ratio * 100.0).round() / 100.0,
        "calmar_ratio": (m.calmar_ratio * 100.0).round() / 100.0,
        "max_drawdown_percent": (m.max_drawdown * 10000.0).round() / 100.0,
        "max_drawdown_duration_days": m.max_drawdown_duration,
        "win_rate_percent": (m.win_rate * 10000.0).round() / 100.0,
        "total_trades": m.total_trades,
        "winning_trades": m.winning_trades,
        "losing_trades": m.losing_trades,
        "profit_factor": (m.profit_factor * 100.0).round() / 100.0,
        "avg_win": (m.avg_win * 100.0).round() / 100.0,
        "avg_loss": (m.avg_loss * 100.0).round() / 100.0,
        "avg_holding_days": (m.avg_holding_days * 10.0).round() / 10.0,
        "total_commission": (m.total_commission * 100.0).round() / 100.0,
        "turnover_rate": (m.turnover_rate * 100.0).round() / 100.0,
        "equity_curve": equity_curve,
        "trades": trades,
        "events": events,
        "snapshots": snapshots,
        "symbol_metrics": symbol_metrics,
        "event_summary": {
            "total_signals": n_signals,
            "risk_triggers": n_risk,
            "orders_rejected": n_rejected,
            "positions_opened": n_pos_opened,
            "positions_closed": n_pos_closed,
            "total_events": result.events.len(),
        },
        "data_source": data_source,
        "period": period,
        "active_inference_mode": active_inference_mode,
        "status": "completed"
    });

    ts.complete(tid, &report.to_string());
}

pub async fn get_backtest_results(
    Path(id): Path<String>,
    State(state): State<AppState>,
) -> (StatusCode, Json<Value>) {
    match state.task_store.get(&id) {
        Some(task) => {
            let mut resp = json!({
                "task_id": task.id,
                "status": task.status,
                "progress": task.progress,
            });
            if task.status == crate::task_store::TaskStatus::Completed {
                if let Some(result_str) = &task.result {
                    if let Ok(result_json) = serde_json::from_str::<Value>(result_str) {
                        resp = result_json;
                        resp["task_id"] = json!(task.id);
                    }
                }
            } else if task.status == crate::task_store::TaskStatus::Failed {
                resp["error"] = json!(task.error);
            }
            (StatusCode::OK, Json(resp))
        }
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "Task not found"}))),
    }
}

pub async fn walk_forward(
    State(_state): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> (StatusCode, Json<Value>) {
    use quant_backtest::engine::BacktestConfig;
    use quant_backtest::walk_forward::walk_forward_validate;
    use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum, MultiFactorStrategy};
    use quant_strategy::ml_factor::MlFactorStrategy;

    let capital = req.capital.unwrap_or(1_000_000.0);
    let symbol = &req.symbol;
    let strategy_name = &req.strategy;

    let (klines, _data_source) = match fetch_real_klines_with_period(symbol, &req.start, &req.end, "daily") {
        Ok(k) if !k.is_empty() => (k, "akshare".to_string()),
        Ok(_) => {
            return (StatusCode::BAD_REQUEST, Json(json!({
                "error": format!("无法获取 {} 的行情数据：数据源返回空数据。请检查股票代码或先同步缓存。", symbol)
            })));
        }
        Err(reason) => {
            return (StatusCode::BAD_REQUEST, Json(json!({
                "error": format!("无法获取 {} 的行情数据：{}。请检查数据源连接或先同步缓存。", symbol, reason)
            })));
        }
    };

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
    };

    let n_folds = 5;
    let factory: Box<dyn Fn() -> Box<dyn quant_core::traits::Strategy> + Send> = match strategy_name.as_str() {
        "dual_ma" | "sma_cross" => Box::new(|| Box::new(DualMaCrossover::new(5, 20))),
        "rsi" | "rsi_reversal" => Box::new(|| Box::new(RsiMeanReversion::new(14, 70.0, 30.0))),
        "macd" | "macd_trend" => Box::new(|| Box::new(MacdMomentum::new(12, 26, 9))),
        "multi_factor" => Box::new(|| Box::new(MultiFactorStrategy::with_defaults())),
        "ml_factor" => Box::new(|| Box::new(MlFactorStrategy::with_defaults())),
        other => {
            return (StatusCode::BAD_REQUEST, Json(json!({"error": format!("Unknown strategy: {}", other)})));
        }
    };

    let result = walk_forward_validate(
        strategy_name,
        factory.as_ref(),
        &klines,
        &bt_config,
        n_folds,
        0.7,
    );

    (StatusCode::OK, Json(json!(result)))
}
