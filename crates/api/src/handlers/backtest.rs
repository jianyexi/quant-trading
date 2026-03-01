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

    // Build full symbol list: primary symbol + optional additional symbols
    let all_symbols = {
        let mut syms = vec![req.symbol.clone()];
        if let Some(extra) = &req.symbols {
            for s in extra {
                let s = s.trim().to_string();
                if !s.is_empty() && !syms.contains(&s) {
                    syms.push(s);
                }
            }
        }
        syms
    };
    let is_multi = all_symbols.len() > 1;

    debug!(symbols=?all_symbols, "Backtest started");

    let ts = state.task_store.clone();
    let params_json = serde_json::to_string(&json!({
        "strategy": req.strategy, "symbols": all_symbols,
        "start": req.start, "end": req.end,
        "capital": capital, "period": period,
        "inference_mode": req.inference_mode,
    })).unwrap_or_default();
    let task_id = ts.create_with_params("backtest", Some(&params_json));
    let tid = task_id.clone();

    let ts2 = Arc::clone(&ts);
    tokio::task::spawn_blocking(move || {
        if is_multi {
            run_multi_backtest_task(&ts2, &tid, &req, &all_symbols, capital, &period);
        } else {
            run_backtest_task(&ts2, &tid, &req, capital, &period);
        }
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
    use quant_strategy::factory::{create_strategy, StrategyOptions};

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

    // Track actual data date range (may differ from requested range)
    let actual_start = klines.first().unwrap().datetime.format("%Y-%m-%d").to_string();
    let actual_end = klines.last().unwrap().datetime.format("%Y-%m-%d").to_string();

    // Validate data coverage: refuse to run on severely incomplete data
    {
        use chrono::NaiveDate;
        let req_start = NaiveDate::parse_from_str(&req.start, "%Y-%m-%d");
        let req_end_raw = NaiveDate::parse_from_str(&req.end, "%Y-%m-%d");
        let act_start = NaiveDate::parse_from_str(&actual_start, "%Y-%m-%d");
        let act_end = NaiveDate::parse_from_str(&actual_end, "%Y-%m-%d");

        if let (Ok(rs), Ok(re), Ok(a_s), Ok(a_e)) = (req_start, req_end_raw, act_start, act_end) {
            // Cap requested end to today (don't penalize for requesting future dates)
            let today = chrono::Local::now().date_naive();
            let re_capped = re.min(today);

            let req_days = (re_capped - rs).num_days().max(1);
            let act_days = (a_e - a_s).num_days().max(1);
            let coverage = act_days as f64 / req_days as f64;

            // Start too late (>7 calendar days gap)
            let start_gap = (a_s - rs).num_days();
            // End too early (>7 calendar days gap, ignoring future dates)
            let end_gap = (re_capped - a_e).num_days();

            if coverage < 0.5 || (start_gap > 30 && end_gap > 30) {
                ts.fail(tid, &format!(
                    "数据覆盖不足，无法回测。\n\
                     请求范围: {} ~ {}\n\
                     实际数据: {} ~ {} (仅 {} 条K线，覆盖率 {:.0}%)\n\n\
                     可能原因:\n\
                     • 该股票在请求期间未上市或已退市\n\
                     • 数据源缺少该时段历史数据\n\
                     • 缓存未同步，请先在「数据管理」页面同步该股票数据\n\n\
                     建议: 缩小日期范围，或先同步缓存后重试。",
                    req.start, req.end, actual_start, actual_end,
                    klines.len(), coverage * 100.0
                ));
                return;
            }

            if start_gap > 7 || end_gap > 7 {
                ts.set_progress(tid, &format!(
                    "⚠️ 数据覆盖部分缺失: 请求 {} ~ {}, 实际 {} ~ {} ({} 条)。继续回测...",
                    req.start, req.end, actual_start, actual_end, klines.len()
                ));
            }
        }
    }

    ts.set_progress(tid, &format!("📊 Data loaded ({} bars, {} ~ {}). Initializing strategy...", klines.len(), actual_start, actual_end));

    // Stage 2: Build strategy
    let created = match create_strategy(&req.strategy, StrategyOptions {
        inference_mode: req.inference_mode.clone(),
        ..Default::default()
    }) {
        Ok(c) => c,
        Err(e) => {
            ts.fail(tid, &e);
            return;
        }
    };
    let mut strategy = created.strategy;
    let active_inference_mode = created.active_inference_mode;

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
        "actual_start": actual_start,
        "actual_end": actual_end,
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

/// Run backtest across multiple symbols, producing per-symbol results + portfolio aggregate
fn run_multi_backtest_task(
    ts: &crate::task_store::TaskStore,
    tid: &str,
    req: &BacktestRequest,
    symbols: &[String],
    capital: f64,
    period: &str,
) {
    use quant_backtest::engine::{BacktestConfig, BacktestEngine};
    use quant_strategy::factory::{create_strategy, StrategyOptions};

    let n_symbols = symbols.len();
    let capital_per_symbol = capital / n_symbols as f64;

    ts.set_progress(tid, &format!("📊 多股回测: 并行加载 {} 只股票数据...", n_symbols));

    // Stage 1: Fetch data for all symbols in parallel
    let fetch_results: Vec<(String, Result<Vec<quant_core::models::Kline>, String>)> =
        std::thread::scope(|s| {
            let handles: Vec<_> = symbols.iter().map(|sym| {
                let sym = sym.clone();
                let start = req.start.clone();
                let end = req.end.clone();
                let period = period.to_string();
                s.spawn(move || {
                    let res = fetch_real_klines_with_period(&sym, &start, &end, &period);
                    (sym, res)
                })
            }).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

    let mut symbol_klines: Vec<(String, Vec<quant_core::models::Kline>)> = Vec::new();
    let mut failed_symbols: Vec<(String, String)> = Vec::new();
    for (sym, res) in fetch_results {
        match res {
            Ok(k) if !k.is_empty() => symbol_klines.push((sym, k)),
            Ok(_) => failed_symbols.push((sym, "空数据".into())),
            Err(e) => failed_symbols.push((sym, e)),
        }
    }

    if symbol_klines.is_empty() {
        let reasons: Vec<String> = failed_symbols.iter().map(|(s, e)| format!("{}: {}", s, e)).collect();
        ts.fail(tid, &format!("所有股票数据获取失败:\n{}", reasons.join("\n")));
        return;
    }

    // Stage 2: Run backtest for each symbol in parallel
    ts.set_progress(tid, &format!("🚀 并行回测 {} 只股票...", symbol_klines.len()));

    let bt_config = BacktestConfig {
        initial_capital: capital_per_symbol,
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

    let eq_fmt = if period == "daily" { "%Y-%m-%d" } else { "%Y-%m-%d %H:%M" };

    // Each thread returns either Ok(result_tuple) or Err(symbol, error)
    type BtResult = Result<(String, Value, Vec<Value>, Vec<(String, f64)>, f64, f64, u64, u64, u64, f64), (String, String)>;

    let bt_results: Vec<BtResult> = std::thread::scope(|s| {
        let handles: Vec<_> = symbol_klines.iter().map(|(sym, klines)| {
            let strategy_name = req.strategy.clone();
            let inference_mode = req.inference_mode.clone();
            let config = bt_config.clone();
            let sym = sym.clone();
            let klines = klines.clone();
            let eq_fmt = eq_fmt;
            s.spawn(move || {
                let created = match create_strategy(&strategy_name, StrategyOptions {
                    inference_mode,
                    ..Default::default()
                }) {
                    Ok(c) => c,
                    Err(e) => return Err((sym, e)),
                };
                let mut strategy = created.strategy;
                let engine = BacktestEngine::new(config);
                let result = engine.run(strategy.as_mut(), &klines);

                let m = &result.metrics;
                let ec: Vec<(String, f64)> = result.equity_curve.iter().map(|(dt, val)| {
                    (dt.format(eq_fmt).to_string(), (*val * 100.0).round() / 100.0)
                }).collect();

                let sym_trades: Vec<Value> = result.trades.iter().map(|t| {
                    json!({
                        "date": t.timestamp.format("%Y-%m-%d %H:%M").to_string(),
                        "symbol": t.symbol,
                        "side": if t.side == quant_core::types::OrderSide::Buy { "BUY" } else { "SELL" },
                        "price": (t.price * 100.0).round() / 100.0,
                        "quantity": t.quantity as i64,
                        "commission": (t.commission * 100.0).round() / 100.0,
                    })
                }).collect();

                let actual_start = klines.first().map(|k| k.datetime.format("%Y-%m-%d").to_string()).unwrap_or_default();
                let actual_end = klines.last().map(|k| k.datetime.format("%Y-%m-%d").to_string()).unwrap_or_default();

                let sym_json = json!({
                    "symbol": sym,
                    "initial_capital": result.config.initial_capital,
                    "final_value": (result.final_portfolio.total_value * 100.0).round() / 100.0,
                    "total_return_percent": (m.total_return * 10000.0).round() / 100.0,
                    "annual_return_percent": (m.annual_return * 10000.0).round() / 100.0,
                    "sharpe_ratio": (m.sharpe_ratio * 100.0).round() / 100.0,
                    "max_drawdown_percent": (m.max_drawdown * 10000.0).round() / 100.0,
                    "total_trades": m.total_trades,
                    "win_rate_percent": (m.win_rate * 10000.0).round() / 100.0,
                    "profit_factor": (m.profit_factor * 100.0).round() / 100.0,
                    "data_bars": klines.len(),
                    "actual_start": actual_start,
                    "actual_end": actual_end,
                    "status": "completed"
                });

                Ok((sym, sym_json, sym_trades, ec,
                    result.final_portfolio.total_value, m.total_commission,
                    m.total_trades as u64, m.winning_trades as u64, m.losing_trades as u64,
                    m.max_drawdown))
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Aggregate results
    let mut per_symbol_results: Vec<Value> = Vec::new();
    let mut all_trades: Vec<Value> = Vec::new();
    let mut total_final_value = 0.0;
    let mut total_commission = 0.0;
    let mut total_trades_count = 0u64;
    let mut total_winning = 0u64;
    let mut total_losing = 0u64;
    let mut portfolio_max_dd = 0.0_f64;
    let mut equity_curves: Vec<(String, Vec<(String, f64)>)> = Vec::new();

    for r in bt_results {
        match r {
            Ok((sym, sym_json, sym_trades, ec, fv, comm, trades, wins, losses, dd)) => {
                per_symbol_results.push(sym_json);
                all_trades.extend(sym_trades);
                equity_curves.push((sym, ec));
                total_final_value += fv;
                total_commission += comm;
                total_trades_count += trades;
                total_winning += wins;
                total_losing += losses;
                portfolio_max_dd = portfolio_max_dd.max(dd);
            }
            Err((sym, e)) => {
                per_symbol_results.push(json!({
                    "symbol": sym,
                    "error": e,
                    "status": "failed"
                }));
            }
        }
    }

    // Stage 3: Build combined equity curve (sum of all symbol equity values by date)
    ts.set_progress(tid, "📈 计算组合指标...");

    // Merge equity curves by date
    let mut date_value_map: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    for (_sym, ec) in &equity_curves {
        for (date, val) in ec {
            *date_value_map.entry(date.clone()).or_insert(0.0) += val;
        }
    }
    let combined_equity: Vec<Value> = date_value_map.iter().map(|(d, v)| {
        json!({ "date": d, "value": (*v * 100.0).round() / 100.0 })
    }).collect();

    // Per-symbol equity curves for individual lines
    let per_symbol_curves: Vec<Value> = equity_curves.iter().map(|(sym, ec)| {
        json!({
            "symbol": sym,
            "data": ec.iter().map(|(d, v)| json!({"date": d, "value": v})).collect::<Vec<_>>()
        })
    }).collect();

    // Sort trades by date
    all_trades.sort_by(|a, b| {
        let da = a["date"].as_str().unwrap_or("");
        let db = b["date"].as_str().unwrap_or("");
        da.cmp(db)
    });

    // Portfolio-level metrics
    let portfolio_return = (total_final_value - capital) / capital;
    let actual_years = {
        let start_d = chrono::NaiveDate::parse_from_str(&req.start, "%Y-%m-%d").unwrap_or_default();
        let end_d = chrono::NaiveDate::parse_from_str(&req.end, "%Y-%m-%d").unwrap_or_default();
        ((end_d - start_d).num_days() as f64 / 365.25).max(0.01)
    };
    let annual_return = (1.0 + portfolio_return).powf(1.0 / actual_years) - 1.0;
    let win_rate = if total_trades_count > 0 { total_winning as f64 / total_trades_count as f64 } else { 0.0 };

    let id = format!("bt-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());

    let report = json!({
        "id": id,
        "strategy": req.strategy,
        "symbol": symbols.join(", "),
        "symbols": symbols,
        "start": req.start,
        "end": req.end,
        "initial_capital": capital,
        "final_value": (total_final_value * 100.0).round() / 100.0,
        "total_return_percent": (portfolio_return * 10000.0).round() / 100.0,
        "annual_return_percent": (annual_return * 10000.0).round() / 100.0,
        "max_drawdown_percent": (portfolio_max_dd * 10000.0).round() / 100.0,
        "win_rate_percent": (win_rate * 10000.0).round() / 100.0,
        "total_trades": total_trades_count,
        "winning_trades": total_winning,
        "losing_trades": total_losing,
        "total_commission": (total_commission * 100.0).round() / 100.0,
        "sharpe_ratio": 0,  // Simplified; per-symbol sharpe available in breakdown
        "sortino_ratio": 0,
        "profit_factor": 0,
        "equity_curve": combined_equity,
        "trades": all_trades,
        "per_symbol_results": per_symbol_results,
        "per_symbol_curves": per_symbol_curves,
        "failed_symbols": failed_symbols.iter().map(|(s, e)| json!({"symbol": s, "error": e})).collect::<Vec<_>>(),
        "data_source": format!("{}只股票组合回测", symbol_klines.len()),
        "period": period,
        "is_multi": true,
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
    use quant_strategy::factory::{create_strategy, StrategyOptions};

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
    let sname = strategy_name.clone();
    let inference_mode = req.inference_mode.clone();
    let factory: Box<dyn Fn() -> Box<dyn quant_core::traits::Strategy> + Send> = Box::new(move || {
        create_strategy(&sname, StrategyOptions {
            inference_mode: inference_mode.clone(),
            ..Default::default()
        }).map(|c| c.strategy)
         .unwrap_or_else(|_| Box::new(quant_strategy::builtin::DualMaCrossover::new(5, 20)))
    });

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
