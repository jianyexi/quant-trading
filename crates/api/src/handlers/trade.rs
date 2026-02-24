use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use tracing::{debug, warn};

use crate::state::AppState;
use super::find_python;

// ── Auto-Trade Request ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TradeStartRequest {
    pub strategy: Option<String>,
    pub symbols: Option<Vec<String>>,
    pub interval: Option<u64>,
    pub position_size: Option<f64>,
    /// "paper" (default), "live" (real data + paper broker), "qmt" for live trading, or "replay" for historical replay
    pub mode: Option<String>,
    /// Slippage in basis points for paper/live modes (default: 0, recommended: 3-10 for realism)
    pub slippage_bps: Option<u32>,
    /// Replay start date (YYYY-MM-DD), required for mode="replay"
    pub replay_start: Option<String>,
    /// Replay end date (YYYY-MM-DD), required for mode="replay"
    pub replay_end: Option<String>,
    /// Replay speed multiplier (0=max speed, 1=real-time, 10=10x), default=0
    pub replay_speed: Option<f64>,
    /// K-line period for replay: "daily", "1", "5", "15", "30", "60" (minutes), default="daily"
    pub replay_period: Option<String>,
    /// ML inference mode: "embedded" (default), "tcp_mq", "http"
    pub inference_mode: Option<String>,
}

pub async fn trade_start(
    State(state): State<AppState>,
    Json(req): Json<TradeStartRequest>,
) -> (StatusCode, Json<Value>) {
    use quant_broker::engine::{EngineConfig, TradingEngine};
    use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum};

    let mut engine_guard = state.engine.lock().await;

    if let Some(ref eng) = *engine_guard {
        if eng.is_running() {
            return (StatusCode::CONFLICT, Json(json!({
                "error": "Engine already running. Stop it first."
            })));
        }
    }

    let strategy_name = req.strategy.unwrap_or_else(|| "sma_cross".into());
    let symbols = req.symbols.unwrap_or_else(|| vec!["600519.SH".into()]);
    let interval = req.interval.unwrap_or(5);
    let position_size = req.position_size.unwrap_or(0.15);
    let mode = req.mode.as_deref().unwrap_or("paper");

    debug!(mode=%mode, strategy=%strategy_name, symbols=?symbols, interval=%interval, "Trade start request");

    let config = EngineConfig {
        strategy_name: strategy_name.clone(),
        symbols: symbols.clone(),
        interval_secs: interval,
        initial_capital: state.config.trading.initial_capital,
        commission_rate: state.config.trading.commission_rate,
        stamp_tax_rate: state.config.trading.stamp_tax_rate,
        max_concentration: state.config.risk.max_concentration,
        position_size_pct: position_size,
        data_mode: if mode == "qmt" || mode == "live" {
            quant_broker::engine::DataMode::Live {
                tushare_url: state.config.tushare.base_url.clone(),
                tushare_token: state.config.tushare.token.clone(),
                akshare_url: state.config.akshare.base_url.clone(),
            }
        } else if mode == "replay" {
            let start = match req.replay_start {
                Some(ref s) if !s.is_empty() => s.clone(),
                _ => return (StatusCode::BAD_REQUEST, Json(json!({
                    "error": "replay_start (YYYY-MM-DD) is required for replay mode"
                }))),
            };
            let end = req.replay_end.clone()
                .unwrap_or_else(|| chrono::Local::now().format("%Y-%m-%d").to_string());
            let speed = req.replay_speed.unwrap_or(0.0);
            let period = req.replay_period.clone().unwrap_or_else(|| "daily".to_string());
            quant_broker::engine::DataMode::HistoricalReplay {
                start_date: start,
                end_date: end,
                speed,
                period,
            }
        } else if mode == "l2" {
            quant_broker::engine::DataMode::L2 {
                l2_addr: "127.0.0.1:18095".to_string(),
            }
        } else {
            quant_broker::engine::DataMode::LowLatency {
                server_url: "http://127.0.0.1:18092".to_string(),
            }
        },
        risk_config: quant_risk::enforcement::RiskConfig {
            stop_loss_pct: state.config.risk.max_drawdown.min(0.10),
            max_daily_loss_pct: state.config.risk.max_daily_loss,
            max_drawdown_pct: state.config.risk.max_drawdown,
            circuit_breaker_failures: 5,
            halt_on_drawdown: true,
            max_holding_days: 30,
            timeout_min_profit_pct: 0.02,
            rebalance_threshold: 0.05,
            ..Default::default()
        },
        db_pool: state.db.clone(),
    };

    let strat_name = strategy_name.clone();

    let mut engine = if mode == "qmt" {
        use quant_broker::qmt::{QmtBroker, QmtConfig};
        let qmt_config = QmtConfig {
            bridge_url: state.config.qmt.bridge_url.clone(),
            account: state.config.qmt.account.clone(),
        };
        let qmt_broker = std::sync::Arc::new(QmtBroker::new(qmt_config));

        match qmt_broker.check_connection().await {
            Ok(true) => {},
            Ok(false) => {
                return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
                    "error": "QMT bridge is running but not connected to QMT client"
                })));
            },
            Err(e) => {
                warn!(error=%e, "Engine start failed");
                return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
                    "error": format!("Cannot reach QMT bridge: {}", e)
                })));
            }
        }

        TradingEngine::new_with_broker(config, qmt_broker)
    } else {
        let eng = TradingEngine::new(config);
        // Apply slippage for paper/live modes
        if let Some(bps) = req.slippage_bps {
            use quant_broker::paper::PaperBroker;
            if let Some(paper) = eng.broker().as_any().downcast_ref::<PaperBroker>() {
                paper.set_slippage_bps(bps);
            }
        } else if mode == "live" {
            // Default 5 bps slippage for live paper trading
            use quant_broker::paper::PaperBroker;
            if let Some(paper) = eng.broker().as_any().downcast_ref::<PaperBroker>() {
                paper.set_slippage_bps(5);
            }
        }
        eng
    };

    engine.set_journal(state.journal.clone());
    engine.set_notifier(state.notifier.clone());

    let sentiment_store = state.sentiment_store.clone();
    let inference_mode_str = req.inference_mode.unwrap_or_else(|| "embedded".into());
    engine.start(move || -> Box<dyn quant_core::traits::Strategy> {
        match strat_name.as_str() {
            "rsi_reversal" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
            "macd_trend" => Box::new(MacdMomentum::new(12, 26, 9)),
            "multi_factor" => Box::new(quant_strategy::builtin::MultiFactorStrategy::with_defaults()),
            "sentiment_aware" => Box::new(quant_strategy::sentiment::SentimentAwareStrategy::with_defaults(
                Box::new(quant_strategy::builtin::MultiFactorStrategy::with_defaults()),
                sentiment_store.clone(),
            )),
            "ml_factor" => {
                let ml_cfg = quant_strategy::ml_factor::MlFactorConfig {
                    inference_mode: quant_strategy::ml_factor::MlInferenceMode::from_str(&inference_mode_str),
                    ..Default::default()
                };
                Box::new(quant_strategy::ml_factor::MlFactorStrategy::new(ml_cfg))
            }
            _ => Box::new(DualMaCrossover::new(5, 20)),
        }
    }).await;

    *engine_guard = Some(engine);

    (StatusCode::OK, Json(json!({
        "status": "started",
        "mode": mode,
        "strategy": strategy_name,
        "symbols": symbols,
        "interval": interval,
        "position_size": position_size,
        "slippage_bps": if mode == "live" { req.slippage_bps.unwrap_or(5) } else { req.slippage_bps.unwrap_or(0) },
        "replay_start": req.replay_start,
        "replay_end": req.replay_end,
        "replay_speed": req.replay_speed,
        "replay_period": req.replay_period
    })))
}

pub async fn trade_stop(
    State(state): State<AppState>,
) -> Json<Value> {
    debug!("Trade stop request");
    let mut engine_guard = state.engine.lock().await;
    if let Some(ref mut eng) = *engine_guard {
        eng.stop().await;
        let status = eng.status().await;
        Json(json!({
            "status": "stopped",
            "total_signals": status.total_signals,
            "total_fills": status.total_fills,
            "pnl": status.pnl
        }))
    } else {
        Json(json!({ "status": "not_running" }))
    }
}

pub async fn trade_status(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine_guard = state.engine.lock().await;
    if let Some(ref eng) = *engine_guard {
        let status = eng.status().await;
        Json(json!(status))
    } else {
        Json(json!({
            "running": false,
            "strategy": "",
            "symbols": [],
            "total_signals": 0,
            "total_orders": 0,
            "total_fills": 0,
            "total_rejected": 0,
            "pnl": 0.0,
            "recent_trades": []
        }))
    }
}

pub async fn risk_status(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine_guard = state.engine.lock().await;
    if let Some(ref eng) = *engine_guard {
        let status = eng.risk_enforcer().status();
        Json(serde_json::to_value(&status).unwrap_or(json!({"error": "serialize"})))
    } else {
        Json(json!({
            "daily_pnl": 0.0,
            "daily_paused": false,
            "drawdown_halted": false,
            "circuit_open": false,
            "consecutive_failures": 0,
            "peak_value": 0.0,
            "config": quant_risk::enforcement::RiskConfig::default()
        }))
    }
}

pub async fn risk_reset_circuit(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine_guard = state.engine.lock().await;
    if let Some(ref eng) = *engine_guard {
        eng.risk_enforcer().reset_circuit_breaker();
        Json(json!({ "status": "circuit_breaker_reset" }))
    } else {
        Json(json!({ "error": "engine_not_running" }))
    }
}

pub async fn risk_reset_daily(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine_guard = state.engine.lock().await;
    if let Some(ref eng) = *engine_guard {
        eng.risk_enforcer().reset_daily();
        Json(json!({ "status": "daily_loss_reset" }))
    } else {
        Json(json!({ "error": "engine_not_running" }))
    }
}

pub async fn trade_performance(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine_guard = state.engine.lock().await;
    if let Some(ref eng) = *engine_guard {
        let status = eng.status().await;
        Json(serde_json::to_value(&status.performance).unwrap_or(json!({})))
    } else {
        Json(json!(quant_broker::engine::PerformanceMetrics::default()))
    }
}

// ── QMT Bridge Status ───────────────────────────────────────────────

pub async fn qmt_bridge_status(
    State(state): State<AppState>,
) -> Json<Value> {
    let url = format!("{}/health", state.config.qmt.bridge_url);
    match reqwest::get(&url).await {
        Ok(resp) => {
            match resp.json::<serde_json::Value>().await {
                Ok(v) => Json(v),
                Err(_) => Json(json!({ "status": "error", "message": "Invalid bridge response" })),
            }
        }
        Err(e) => Json(json!({
            "status": "offline",
            "message": format!("Cannot reach QMT bridge: {}", e),
            "bridge_url": state.config.qmt.bridge_url
        })),
    }
}

// ── ML Model Retrain ────────────────────────────────────────────────

pub async fn ml_retrain(
    State(_state): State<AppState>,
    body: Option<Json<Value>>,
) -> (StatusCode, Json<Value>) {
    let body_val = body.map(|b| b.0).unwrap_or(json!({}));
    let algorithms = body_val.get("algorithms").and_then(|a| a.as_str())
        .unwrap_or("lgb").to_string();
    let data_source = body_val.get("data_source").and_then(|a| a.as_str())
        .unwrap_or("synthetic").to_string();
    let symbols = body_val.get("symbols").and_then(|a| a.as_str())
        .unwrap_or("").to_string();
    let start_date = body_val.get("start_date").and_then(|a| a.as_str())
        .unwrap_or("2022-01-01").to_string();
    let end_date = body_val.get("end_date").and_then(|a| a.as_str())
        .unwrap_or("2024-12-31").to_string();
    let horizon = body_val.get("horizon").and_then(|a| a.as_i64())
        .unwrap_or(5).to_string();
    let threshold = body_val.get("threshold").and_then(|a| a.as_f64())
        .unwrap_or(0.01).to_string();

    let result = tokio::task::spawn_blocking(move || {
        let retrain_script = std::path::Path::new("ml_models/auto_retrain.py");
        if !retrain_script.exists() {
            return Err(format!("auto_retrain.py not found (cwd={:?})", std::env::current_dir()));
        }

        let python = find_python().ok_or("Python not found. Install Python 3.12+ or set PYTHON_PATH env var.")?;

        let mut args = vec![
            "ml_models/auto_retrain.py".to_string(),
            "--no-notify".to_string(),
            "--algorithms".to_string(), algorithms,
            "--horizon".to_string(), horizon,
            "--threshold".to_string(), threshold,
        ];

        if data_source == "akshare" {
            args.push("--data-source".to_string());
            args.push("akshare".to_string());
            if !symbols.is_empty() {
                args.push("--symbols".to_string());
                args.push(symbols);
            }
            args.push("--start-date".to_string());
            args.push(start_date);
            args.push("--end-date".to_string());
            args.push(end_date);
        }

        let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let output = std::process::Command::new(&python)
            .args(&str_args)
            .env("PYTHONIOENCODING", "utf-8")
            .output()
            .map_err(|e| format!("Failed to start python '{}': {}", python, e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(serde_json::json!({
                "status": "completed",
                "stdout": stdout,
                "stderr": stderr,
            }))
        } else {
            let detail = if stderr.is_empty() { &stdout } else { &stderr };
            Err(format!("Retrain exit {}: {}", output.status, detail))
        }
    }).await;

    match result {
        Ok(Ok(report)) => (StatusCode::OK, Json(report)),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("Task error: {}", e)}))),
    }
}

pub async fn ml_model_info() -> Json<Value> {
    let reports_dir = std::path::Path::new("ml_models");
    let mut latest_report = None;

    if reports_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(reports_dir) {
            let mut reports: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("retrain_report_"))
                .collect();
            reports.sort_by_key(|e| e.file_name());
            if let Some(latest) = reports.last() {
                if let Ok(content) = std::fs::read_to_string(latest.path()) {
                    latest_report = serde_json::from_str::<Value>(&content).ok();
                }
            }
        }
    }

    let info = json!({
        "model_dir": "ml_models/",
        "default_model": "ml_models/factor_model.lgb.txt",
        "retrain_script": "ml_models/auto_retrain.py",
        "latest_report": latest_report,
        "supported_algorithms": ["lgb", "xgb", "catboost", "lstm", "transformer"],
    });

    Json(info)
}

// ── Position Management ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ClosePositionRequest {
    pub symbol: String,
    pub price: Option<f64>,
}

pub async fn close_position(
    State(state): State<AppState>,
    Json(req): Json<ClosePositionRequest>,
) -> (StatusCode, Json<Value>) {
    let engine = state.engine.lock().await;
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return (StatusCode::BAD_REQUEST, Json(json!({"error": "Engine not running"}))),
    };

    // Get current price: from request, quote, or last known
    let price = if let Some(p) = req.price {
        p
    } else {
        super::market::fetch_real_quote(&req.symbol)
            .ok()
            .and_then(|q| q["price"].as_f64())
            .unwrap_or_else(|| {
                // Fallback to position's current_price
                0.0
            })
    };

    if price <= 0.0 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "Cannot determine current price"})));
    }

    // Downcast broker to PaperBroker
    use quant_broker::paper::PaperBroker;
    let broker = eng.broker();
    let paper = match broker.as_any().downcast_ref::<PaperBroker>() {
        Some(p) => p,
        None => return (StatusCode::BAD_REQUEST, Json(json!({"error": "Close only supported for paper trading"}))),
    };

    match paper.force_close_position(&req.symbol, price) {
        Ok(closed) => (StatusCode::OK, Json(json!({
            "status": "closed",
            "closed_position": closed,
        }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e}))),
    }
}

pub async fn get_closed_positions(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine = state.engine.lock().await;
    if let Some(eng) = engine.as_ref() {
        use quant_broker::paper::PaperBroker;
        let broker = eng.broker();
        if let Some(paper) = broker.as_any().downcast_ref::<PaperBroker>() {
            let closed = paper.closed_positions();
            return Json(json!({"closed_positions": closed, "count": closed.len()}));
        }
    }
    Json(json!({"closed_positions": [], "count": 0}))
}

// ── Tick Recording ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TickQuery {
    pub symbol: Option<String>,
    pub limit: Option<u32>,
    pub since: Option<String>,
}

pub async fn get_recorded_ticks(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<TickQuery>,
) -> Json<Value> {
    let limit = q.limit.unwrap_or(200).min(1000) as i64;

    // Try PostgreSQL first
    if let Some(ref pool) = state.db {
        let mut sql = String::from(
            "SELECT symbol, datetime::text, open, high, low, close, volume, recorded_at::text FROM market_ticks"
        );
        let mut conditions = Vec::new();
        let mut idx = 1u32;
        let mut binds: Vec<String> = Vec::new();

        if let Some(ref sym) = q.symbol {
            conditions.push(format!("symbol = ${}", idx)); idx += 1;
            binds.push(sym.clone());
        }
        if let Some(ref since) = q.since {
            conditions.push(format!("datetime >= ${}::timestamp", idx)); idx += 1;
            binds.push(since.clone());
        }
        let _ = idx;
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(&format!(" ORDER BY datetime DESC LIMIT {}", limit));

        let mut query = sqlx::query(&sql);
        for b in &binds {
            query = query.bind(b);
        }

        match query.fetch_all(pool).await {
            Ok(rows) => {
                use sqlx::Row;
                let ticks: Vec<serde_json::Value> = rows.iter().map(|row| {
                    json!({
                        "symbol": row.get::<String, _>("symbol"),
                        "datetime": row.get::<String, _>("datetime"),
                        "open": row.get::<f64, _>("open"),
                        "high": row.get::<f64, _>("high"),
                        "low": row.get::<f64, _>("low"),
                        "close": row.get::<f64, _>("close"),
                        "volume": row.get::<f64, _>("volume"),
                        "recorded_at": row.get::<String, _>("recorded_at"),
                    })
                }).collect();
                let count = ticks.len();

                let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM market_ticks")
                    .fetch_one(pool).await.unwrap_or(0);
                let symbols: Vec<String> = sqlx::query_scalar("SELECT DISTINCT symbol FROM market_ticks ORDER BY symbol")
                    .fetch_all(pool).await.unwrap_or_default();

                return Json(json!({
                    "ticks": ticks,
                    "count": count,
                    "total_recorded": total,
                    "symbols": symbols,
                }));
            }
            Err(e) => {
                return Json(json!({"ticks": [], "count": 0, "error": e.to_string()}));
            }
        }
    }

    // Fallback to SQLite
    let conn = match rusqlite::Connection::open("data/market_ticks.db") {
        Ok(c) => c,
        Err(_) => return Json(json!({"ticks": [], "count": 0, "error": "No tick database found"})),
    };

    let mut sql = String::from(
        "SELECT symbol, datetime, open, high, low, close, volume, recorded_at FROM ticks"
    );
    let mut conditions = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref sym) = q.symbol {
        conditions.push(format!("symbol = ?{}", params.len() + 1));
        params.push(Box::new(sym.clone()));
    }
    if let Some(ref since) = q.since {
        conditions.push(format!("datetime >= ?{}", params.len() + 1));
        params.push(Box::new(since.clone()));
    }
    if !conditions.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&conditions.join(" AND "));
    }
    sql.push_str(&format!(" ORDER BY datetime DESC LIMIT {}", limit));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => return Json(json!({"ticks": [], "count": 0, "error": e.to_string()})),
    };
    let rows: Vec<serde_json::Value> = stmt.query_map(param_refs.as_slice(), |row| {
        Ok(serde_json::json!({
            "symbol": row.get::<_, String>(0)?,
            "datetime": row.get::<_, String>(1)?,
            "open": row.get::<_, f64>(2)?,
            "high": row.get::<_, f64>(3)?,
            "low": row.get::<_, f64>(4)?,
            "close": row.get::<_, f64>(5)?,
            "volume": row.get::<_, f64>(6)?,
            "recorded_at": row.get::<_, String>(7)?,
        }))
    }).ok().map(|r| r.filter_map(|x| x.ok()).collect()).unwrap_or_default();

    let count = rows.len();

    let total: i64 = conn.query_row("SELECT COUNT(*) FROM ticks", [], |r| r.get(0)).unwrap_or(0);
    let symbols: Vec<String> = conn.prepare("SELECT DISTINCT symbol FROM ticks ORDER BY symbol")
        .ok()
        .map(|mut s| s.query_map([], |r| r.get(0)).ok()
            .map(|r| r.filter_map(|x| x.ok()).collect())
            .unwrap_or_default())
        .unwrap_or_default();

    Json(json!({
        "ticks": rows,
        "count": count,
        "total_recorded": total,
        "symbols": symbols,
    }))
}
