use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use quant_core::models::Kline;
use crate::state::AppState;

// ── Query Parameters ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct KlineQuery {
    pub start: Option<String>,
    pub end: Option<String>,
    pub limit: Option<usize>,
}

// ── Backtest Request ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BacktestRequest {
    pub strategy: String,
    pub symbol: String,
    pub start: String,
    pub end: String,
    pub capital: Option<f64>,
}

// ── Chat Request ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub reply: String,
}

// ── Health ──────────────────────────────────────────────────────────

pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "version": "0.1.0" }))
}

// ── Strategies ──────────────────────────────────────────────────────

pub async fn list_strategies() -> Json<Value> {
    Json(json!({
        "strategies": [
            {
                "name": "sma_cross",
                "display_name": "SMA Crossover",
                "description": "Dual Simple Moving Average crossover strategy",
                "parameters": [
                    {"key": "fast_period", "label": "Fast Period", "type": "number", "default": 5, "min": 2, "max": 60},
                    {"key": "slow_period", "label": "Slow Period", "type": "number", "default": 20, "min": 5, "max": 120}
                ]
            },
            {
                "name": "rsi_reversal",
                "display_name": "RSI Mean Reversion",
                "description": "RSI oversold/overbought mean reversion strategy",
                "parameters": [
                    {"key": "period", "label": "RSI Period", "type": "number", "default": 14, "min": 5, "max": 50},
                    {"key": "oversold", "label": "Oversold Level", "type": "number", "default": 30, "min": 10, "max": 40},
                    {"key": "overbought", "label": "Overbought Level", "type": "number", "default": 70, "min": 60, "max": 90}
                ]
            },
            {
                "name": "macd_trend",
                "display_name": "MACD Trend Following",
                "description": "MACD histogram crossover trend strategy",
                "parameters": [
                    {"key": "fast_period", "label": "Fast EMA", "type": "number", "default": 12, "min": 5, "max": 30},
                    {"key": "slow_period", "label": "Slow EMA", "type": "number", "default": 26, "min": 15, "max": 60},
                    {"key": "signal_period", "label": "Signal Period", "type": "number", "default": 9, "min": 3, "max": 20}
                ]
            },
            {
                "name": "bollinger_bands",
                "display_name": "Bollinger Bands",
                "description": "Bollinger Bands breakout/reversion strategy",
                "parameters": [
                    {"key": "period", "label": "Period", "type": "number", "default": 20, "min": 10, "max": 50},
                    {"key": "std_dev", "label": "Std Deviation", "type": "number", "default": 2, "min": 1, "max": 3}
                ]
            },
            {
                "name": "dual_momentum",
                "display_name": "Dual Momentum",
                "description": "Absolute + relative momentum strategy",
                "parameters": [
                    {"key": "lookback", "label": "Lookback Period", "type": "number", "default": 60, "min": 20, "max": 120}
                ]
            }
        ]
    }))
}

// ── Dashboard ───────────────────────────────────────────────────────

pub async fn get_dashboard(
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "portfolio_value": 1_284_305.00,
        "daily_pnl": 32_411.20,
        "daily_pnl_percent": 2.59,
        "open_positions": 5,
        "win_rate": 68.5,
        "recent_trades": [
            {"time": "14:32:01", "symbol": "600519.SH", "name": "贵州茅台", "side": "BUY", "quantity": 100, "price": 1689.25, "pnl": 3125.00},
            {"time": "13:45:22", "symbol": "000858.SZ", "name": "五粮液", "side": "SELL", "quantity": 300, "price": 148.10, "pnl": -873.00},
            {"time": "12:18:45", "symbol": "601318.SH", "name": "中国平安", "side": "BUY", "quantity": 500, "price": 52.60, "pnl": 1420.00},
            {"time": "11:05:33", "symbol": "600036.SH", "name": "招商银行", "side": "SELL", "quantity": 400, "price": 35.30, "pnl": 560.00},
            {"time": "10:22:17", "symbol": "000001.SZ", "name": "平安银行", "side": "BUY", "quantity": 1000, "price": 12.90, "pnl": -225.00}
        ]
    }))
}

// ── Market Handlers ─────────────────────────────────────────────────

fn generate_kline_data(symbol: &str, limit: usize) -> Vec<Value> {
    let (base_price, name) = match symbol {
        "600519.SH" => (1650.0, "贵州茅台"),
        "000858.SZ" => (148.0, "五粮液"),
        "601318.SH" => (52.0, "中国平安"),
        "000001.SZ" => (12.5, "平安银行"),
        "600036.SH" => (35.0, "招商银行"),
        "300750.SZ" => (220.0, "宁德时代"),
        "600276.SH" => (28.0, "恒瑞医药"),
        _ => (100.0, "未知"),
    };
    let _ = name;
    let mut data = Vec::with_capacity(limit);
    let mut price = base_price;
    let base_date = chrono::NaiveDate::from_ymd_opt(2024, 6, 1).unwrap();
    for i in 0..limit {
        let change = (((i as f64 * 7.3 + 13.7).sin() * 0.02)
            + ((i as f64 * 3.1).cos() * 0.008))
            * price;
        let open = price;
        let close = price + change;
        let high = open.max(close) + (((i as f64 * 5.1).sin().abs()) * 0.005 * price);
        let low = open.min(close) - (((i as f64 * 4.3).cos().abs()) * 0.005 * price);
        let volume = (5_000_000.0 + ((i as f64 * 2.7).sin() * 3_000_000.0).abs()) as u64;
        let date = base_date + chrono::Duration::days(i as i64);
        // Skip weekends
        if date.weekday() == chrono::Weekday::Sat || date.weekday() == chrono::Weekday::Sun {
            continue;
        }
        data.push(json!({
            "date": date.format("%Y-%m-%d").to_string(),
            "open": (open * 100.0).round() / 100.0,
            "high": (high * 100.0).round() / 100.0,
            "low": (low * 100.0).round() / 100.0,
            "close": (close * 100.0).round() / 100.0,
            "volume": volume
        }));
        price = close;
    }
    data
}

pub async fn get_kline(
    Path(symbol): Path<String>,
    Query(params): Query<KlineQuery>,
    State(_state): State<AppState>,
) -> Json<Value> {
    let limit = params.limit.unwrap_or(60);
    let data = generate_kline_data(&symbol, limit);
    Json(json!({
        "symbol": symbol,
        "start": params.start.unwrap_or_default(),
        "end": params.end.unwrap_or_default(),
        "data": data
    }))
}

pub async fn get_quote(
    Path(symbol): Path<String>,
    State(_state): State<AppState>,
) -> Json<Value> {
    let (price, name) = match symbol.as_str() {
        "600519.SH" => (1688.50, "贵州茅台"),
        "000858.SZ" => (142.85, "五粮液"),
        "601318.SH" => (52.36, "中国平安"),
        "000001.SZ" => (12.58, "平安银行"),
        "600036.SH" => (35.72, "招商银行"),
        "300750.SZ" => (225.40, "宁德时代"),
        _ => (100.0, "未知"),
    };
    Json(json!({
        "symbol": symbol,
        "name": name,
        "price": price,
        "change": price * 0.012,
        "change_percent": 1.19,
        "volume": 12_580_000,
        "turnover": price * 12_580_000.0,
        "timestamp": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()
    }))
}

// ── Stock list ──────────────────────────────────────────────────────

pub async fn list_stocks() -> Json<Value> {
    Json(json!({
        "stocks": [
            {"symbol": "600519.SH", "name": "贵州茅台", "industry": "白酒", "market": "SSE"},
            {"symbol": "000858.SZ", "name": "五粮液", "industry": "白酒", "market": "SZSE"},
            {"symbol": "601318.SH", "name": "中国平安", "industry": "保险", "market": "SSE"},
            {"symbol": "000001.SZ", "name": "平安银行", "industry": "银行", "market": "SZSE"},
            {"symbol": "600036.SH", "name": "招商银行", "industry": "银行", "market": "SSE"},
            {"symbol": "300750.SZ", "name": "宁德时代", "industry": "电池", "market": "ChiNext"},
            {"symbol": "600276.SH", "name": "恒瑞医药", "industry": "医药", "market": "SSE"},
            {"symbol": "000333.SZ", "name": "美的集团", "industry": "家电", "market": "SZSE"},
            {"symbol": "601888.SH", "name": "中国中免", "industry": "零售", "market": "SSE"},
            {"symbol": "002594.SZ", "name": "比亚迪", "industry": "汽车", "market": "SZSE"}
        ]
    }))
}

// ── Backtest Handlers ───────────────────────────────────────────────

pub async fn run_backtest(
    State(_state): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> (StatusCode, Json<Value>) {
    let capital = req.capital.unwrap_or(1_000_000.0);
    // Simulate slightly different results per strategy
    let (ret, sharpe, dd, wr, trades, pf) = match req.strategy.as_str() {
        "sma_cross" => (18.5, 1.32, 10.2, 55.0, 38, 1.65),
        "rsi_reversal" => (22.3, 1.58, 8.7, 62.0, 45, 1.92),
        "macd_trend" => (15.8, 1.15, 14.5, 52.0, 32, 1.48),
        "bollinger_bands" => (20.1, 1.45, 11.3, 58.0, 42, 1.78),
        "dual_momentum" => (25.0, 1.72, 9.5, 60.0, 28, 2.05),
        _ => (12.0, 0.95, 15.0, 48.0, 50, 1.20),
    };
    (StatusCode::OK, Json(json!({
        "id": format!("bt-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()),
        "strategy": req.strategy,
        "symbol": req.symbol,
        "start": req.start,
        "end": req.end,
        "initial_capital": capital,
        "final_value": capital * (1.0 + ret / 100.0),
        "total_return_percent": ret,
        "sharpe_ratio": sharpe,
        "max_drawdown_percent": dd,
        "win_rate_percent": wr,
        "total_trades": trades,
        "profit_factor": pf,
        "status": "completed"
    })))
}

pub async fn get_backtest_results(
    Path(id): Path<String>,
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "id": id,
        "status": "completed",
        "total_return_percent": 25.0,
        "annualized_return_percent": 18.5,
        "sharpe_ratio": 1.45,
        "max_drawdown_percent": 12.3,
        "win_rate_percent": 58.0,
        "total_trades": 42,
        "profit_factor": 1.85
    }))
}

// ── Order Handlers ──────────────────────────────────────────────────

pub async fn list_orders(
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "orders": [
            {"id": "ORD-001", "time": "2024-06-14 14:32:01", "symbol": "600519.SH", "side": "buy", "price": 1689.25, "quantity": 100, "status": "filled"},
            {"id": "ORD-002", "time": "2024-06-14 13:45:22", "symbol": "000858.SZ", "side": "sell", "price": 148.10, "quantity": 300, "status": "filled"},
            {"id": "ORD-003", "time": "2024-06-14 11:05:33", "symbol": "600036.SH", "side": "sell", "price": 35.30, "quantity": 400, "status": "filled"}
        ]
    }))
}

// ── Portfolio Handlers ──────────────────────────────────────────────

pub async fn get_portfolio(
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "total_value": 1_284_305.00,
        "cash": 245_680.00,
        "total_pnl": 284_305.00,
        "positions": [
            {"symbol": "600519.SH", "name": "贵州茅台", "shares": 100, "avg_cost": 1620.00, "current_price": 1688.50, "pnl": 6850.00},
            {"symbol": "000858.SZ", "name": "五粮液", "shares": 500, "avg_cost": 148.30, "current_price": 142.85, "pnl": -2725.00},
            {"symbol": "601318.SH", "name": "中国平安", "shares": 1000, "avg_cost": 49.80, "current_price": 52.36, "pnl": 2560.00},
            {"symbol": "000001.SZ", "name": "平安银行", "shares": 2000, "avg_cost": 13.10, "current_price": 12.58, "pnl": -1040.00},
            {"symbol": "600036.SH", "name": "招商银行", "shares": 800, "avg_cost": 32.50, "current_price": 35.72, "pnl": 2576.00}
        ]
    }))
}

// ── Chat Handlers ───────────────────────────────────────────────────

pub async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    use quant_llm::{client::LlmClient, context::ConversationContext, tools::{get_all_tools, ToolExecutor}};

    let llm_config = &state.config.llm;

    // If no API key configured, return a helpful stub
    if llm_config.api_key.is_empty() {
        return Json(ChatResponse {
            reply: format!(
                "💡 LLM API key not configured. To enable AI chat, set `llm.api_key` in config/default.toml.\n\n\
                Your message: \"{}\"",
                req.message
            ),
        });
    }

    let client = LlmClient::new(
        &llm_config.api_url,
        &llm_config.api_key,
        &llm_config.model,
        llm_config.temperature,
        llm_config.max_tokens,
    );
    let mut context = ConversationContext::new("You are a quantitative trading assistant for Chinese A-shares.", 50);
    let tools = get_all_tools();
    let executor = ToolExecutor::new();

    context.add_user_message(&req.message);

    // Chat loop with tool-call handling (max 5 rounds)
    for _ in 0..5 {
        let messages = context.get_messages();
        match client.chat(&messages, Some(&tools)).await {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    let msg = &choice.message;
                    if let Some(tool_calls) = &msg.tool_calls {
                        context.add_assistant_tool_calls(tool_calls.clone());
                        for tc in tool_calls {
                            match executor.execute(tc).await {
                                Ok(result) => context.add_tool_result(&tc.id, &result),
                                Err(e) => context.add_tool_result(&tc.id, &format!("Error: {e}")),
                            }
                        }
                        continue;
                    }
                    if let Some(content) = &msg.content {
                        return Json(ChatResponse { reply: content.clone() });
                    }
                }
                return Json(ChatResponse { reply: "No response from LLM.".to_string() });
            }
            Err(e) => {
                return Json(ChatResponse {
                    reply: format!("⚠️ LLM error: {e}"),
                });
            }
        }
    }

    Json(ChatResponse { reply: "Tool call loop exceeded maximum iterations.".to_string() })
}

pub async fn chat_history(
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "sessions": []
    }))
}

// ── Auto-Trade Handlers ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TradeStartRequest {
    pub strategy: Option<String>,
    pub symbols: Option<Vec<String>>,
    pub interval: Option<u64>,
    pub position_size: Option<f64>,
    /// "paper" (default) or "qmt" for live trading via QMT bridge
    pub mode: Option<String>,
}

pub async fn trade_start(
    State(state): State<AppState>,
    Json(req): Json<TradeStartRequest>,
) -> (StatusCode, Json<Value>) {
    use quant_broker::engine::{EngineConfig, TradingEngine};
    use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum};

    let mut engine_guard = state.engine.lock().await;

    // Check if already running
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

    let config = EngineConfig {
        strategy_name: strategy_name.clone(),
        symbols: symbols.clone(),
        interval_secs: interval,
        initial_capital: state.config.trading.initial_capital,
        commission_rate: state.config.trading.commission_rate,
        stamp_tax_rate: state.config.trading.stamp_tax_rate,
        max_concentration: state.config.risk.max_concentration,
        position_size_pct: position_size,
    };

    let strat_name = strategy_name.clone();

    let mut engine = if mode == "qmt" {
        // QMT live trading via Python bridge
        use quant_broker::qmt::{QmtBroker, QmtConfig};
        let qmt_config = QmtConfig {
            bridge_url: state.config.qmt.bridge_url.clone(),
            account: state.config.qmt.account.clone(),
        };
        let qmt_broker = std::sync::Arc::new(QmtBroker::new(qmt_config));

        // Verify bridge connectivity
        match qmt_broker.check_connection().await {
            Ok(true) => {},
            Ok(false) => {
                return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
                    "error": "QMT bridge is running but not connected to QMT client"
                })));
            },
            Err(e) => {
                return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({
                    "error": format!("Cannot reach QMT bridge: {}", e)
                })));
            }
        }

        TradingEngine::new_with_broker(config, qmt_broker)
    } else {
        TradingEngine::new(config)
    };

    engine.start(move || -> Box<dyn quant_core::traits::Strategy> {
        match strat_name.as_str() {
            "rsi_reversal" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
            "macd_trend" => Box::new(MacdMomentum::new(12, 26, 9)),
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
        "position_size": position_size
    })))
}

pub async fn trade_stop(
    State(state): State<AppState>,
) -> Json<Value> {
    let mut engine_guard = state.engine.lock().await;
    if let Some(ref mut eng) = *engine_guard {
        eng.stop();
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

// ── Screener Handlers ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ScreenRequest {
    pub top_n: Option<usize>,
    pub min_votes: Option<u32>,
}

pub async fn screen_scan(
    State(_state): State<AppState>,
    Json(req): Json<ScreenRequest>,
) -> Json<Value> {
    use std::collections::HashMap;
    use quant_strategy::screener::{ScreenerConfig, StockScreener};

    let top_n = req.top_n.unwrap_or(10);
    let min_votes = req.min_votes.unwrap_or(2);

    let stocks: Vec<(&str, &str, f64)> = vec![
        ("600519.SH", "贵州茅台", 1700.0),
        ("000858.SZ", "五粮液", 150.0),
        ("601318.SH", "中国平安", 48.0),
        ("000001.SZ", "平安银行", 11.5),
        ("600036.SH", "招商银行", 34.0),
        ("300750.SZ", "宁德时代", 195.0),
        ("600276.SH", "恒瑞医药", 45.0),
        ("000333.SZ", "美的集团", 60.0),
        ("601888.SH", "中国中免", 85.0),
        ("002594.SZ", "比亚迪", 230.0),
        ("601012.SH", "隆基绿能", 22.0),
        ("600900.SH", "长江电力", 28.0),
        ("000568.SZ", "泸州老窖", 185.0),
        ("600809.SH", "山西汾酒", 220.0),
        ("002475.SZ", "立讯精密", 32.0),
        ("600030.SH", "中信证券", 20.0),
        ("601166.SH", "兴业银行", 17.0),
        ("000661.SZ", "长春高新", 165.0),
        ("002714.SZ", "牧原股份", 42.0),
        ("600585.SH", "海螺水泥", 26.0),
    ];

    let end_date = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
    let start_date = end_date - chrono::Duration::days(120);

    let mut stock_data: HashMap<String, (String, Vec<Kline>)> = HashMap::new();
    for (symbol, name, base_price) in &stocks {
        let klines = generate_screening_klines(symbol, name, *base_price, start_date, end_date);
        stock_data.insert(symbol.to_string(), (name.to_string(), klines));
    }

    let config = ScreenerConfig {
        top_n,
        phase1_cutoff: 20,
        min_consensus: min_votes,
        ..ScreenerConfig::default()
    };

    let screener = StockScreener::new(config);
    let result = screener.screen(&stock_data);

    Json(json!(result))
}

pub async fn screen_factors(
    State(_state): State<AppState>,
    Path(symbol): Path<String>,
) -> Json<Value> {
    use std::collections::HashMap;
    use quant_strategy::screener::{ScreenerConfig, StockScreener};

    let (name, base_price) = match symbol.as_str() {
        "600519.SH" => ("贵州茅台", 1700.0),
        "000858.SZ" => ("五粮液", 150.0),
        "601318.SH" => ("中国平安", 48.0),
        "000001.SZ" => ("平安银行", 11.5),
        "600036.SH" => ("招商银行", 34.0),
        "300750.SZ" => ("宁德时代", 195.0),
        _ => ("未知", 100.0),
    };

    let end_date = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
    let start_date = end_date - chrono::Duration::days(120);
    let klines = generate_screening_klines(&symbol, name, base_price, start_date, end_date);

    let mut stock_data: HashMap<String, (String, Vec<Kline>)> = HashMap::new();
    stock_data.insert(symbol.clone(), (name.to_string(), klines));

    let config = ScreenerConfig {
        top_n: 1,
        phase1_cutoff: 1,
        min_consensus: 0,
        ..ScreenerConfig::default()
    };

    let screener = StockScreener::new(config);
    let result = screener.screen(&stock_data);

    if let Some(c) = result.candidates.first() {
        Json(json!(c))
    } else {
        Json(json!({"error": "No data for symbol", "symbol": symbol}))
    }
}

/// Generate kline data for API screening
fn generate_screening_klines(
    symbol: &str,
    _name: &str,
    base_price: f64,
    start: NaiveDate,
    end: NaiveDate,
) -> Vec<Kline> {
    use chrono::Datelike;

    let mut klines = Vec::new();
    let mut current = start;
    let daily_vol = 0.015;
    let amplitude = 0.15;

    let seed: u64 = symbol.bytes().fold(42u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut rng_state = seed;

    let total_days = {
        let mut d = start;
        let mut count = 0;
        while d <= end {
            if d.weekday() != chrono::Weekday::Sat && d.weekday() != chrono::Weekday::Sun {
                count += 1;
            }
            d += chrono::Duration::days(1);
        }
        count as f64
    };

    let mut bar_idx: f64 = 0.0;
    let mut close = base_price;

    while current <= end {
        if current.weekday() == chrono::Weekday::Sat || current.weekday() == chrono::Weekday::Sun {
            current += chrono::Duration::days(1);
            continue;
        }

        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r1 = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r2 = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r3 = ((rng_state >> 33) as f64) / (u32::MAX as f64);

        let t = bar_idx / total_days;
        let cycle1 = (t * std::f64::consts::PI * 6.0).sin();
        let cycle2 = (t * std::f64::consts::PI * 14.0).sin();
        let cycle3 = (t * std::f64::consts::PI * 2.0).sin();
        let target = base_price * (1.0 + amplitude * (0.5 * cycle1 + 0.3 * cycle2 + 0.2 * cycle3));

        let pull = 0.03 * (target - close) / close;
        let noise = daily_vol * r1 * 2.0;
        let daily_return = pull + noise;

        let open = close;
        close = open * (1.0 + daily_return);

        let intra = open.abs() * daily_vol * (0.3 + r3 * 0.5);
        let high = open.max(close) + intra * (0.3 + r2.abs());
        let low = (open.min(close) - intra * (0.3 + (0.5 - r2).abs())).max(open.min(close) * 0.95);

        let base_vol = if base_price > 500.0 { 5e6 } else if base_price > 50.0 { 20e6 } else { 60e6 };
        let volume = base_vol * (0.6 + r3 * 0.8) * (1.0 + daily_return.abs() * 15.0);

        klines.push(Kline {
            symbol: symbol.to_string(),
            datetime: current.and_hms_opt(15, 0, 0).unwrap(),
            open, high, low, close, volume,
        });

        bar_idx += 1.0;
        current += chrono::Duration::days(1);
    }

    klines
}
