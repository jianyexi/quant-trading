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
            },
            {
                "name": "multi_factor",
                "display_name": "多因子模型",
                "description": "6因子综合评分策略: 趋势+动量+波动率+KDJ+量价+价格行为",
                "parameters": [
                    {"key": "buy_threshold", "label": "买入阈值", "type": "number", "default": 0.30, "min": 0.1, "max": 0.6},
                    {"key": "sell_threshold", "label": "卖出阈值", "type": "number", "default": -0.30, "min": -0.6, "max": -0.1}
                ]
            },
            {
                "name": "sentiment_aware",
                "display_name": "舆情增强策略",
                "description": "基于舆情数据增强的多因子策略，结合市场情绪调整交易信号强度",
                "parameters": [
                    {"key": "sentiment_weight", "label": "舆情权重", "type": "number", "default": 0.20, "min": 0.05, "max": 0.50},
                    {"key": "min_items", "label": "最少舆情条数", "type": "number", "default": 3, "min": 1, "max": 20}
                ]
            },
            {
                "name": "ml_factor",
                "display_name": "ML因子模型",
                "description": "机器学习因子提取策略，24维特征工程 + GPU模型推理(Python sidecar)",
                "parameters": [
                    {"key": "buy_threshold", "label": "买入阈值", "type": "number", "default": 0.60, "min": 0.50, "max": 0.80},
                    {"key": "sell_threshold", "label": "卖出阈值", "type": "number", "default": 0.35, "min": 0.20, "max": 0.50},
                    {"key": "bridge_url", "label": "推理服务地址", "type": "string", "default": "http://127.0.0.1:18091"}
                ]
            }
        ]
    }))
}

// ── Dashboard ───────────────────────────────────────────────────────

pub async fn get_dashboard(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine_guard = state.engine.lock().await;
    if let Some(ref eng) = *engine_guard {
        let status = eng.status().await;
        let perf = &status.performance;
        let trades: Vec<Value> = status.recent_trades.iter().map(|t| {
            json!({
                "time": t.timestamp.format("%H:%M:%S").to_string(),
                "symbol": t.symbol,
                "side": if t.side == quant_core::types::OrderSide::Buy { "BUY" } else { "SELL" },
                "quantity": t.quantity as i64,
                "price": t.price,
                "status": t.status,
            })
        }).collect();
        Json(json!({
            "portfolio_value": perf.portfolio_value,
            "daily_pnl": perf.risk_daily_pnl,
            "daily_pnl_percent": if perf.initial_capital > 0.0 {
                perf.risk_daily_pnl / perf.initial_capital * 100.0
            } else { 0.0 },
            "open_positions": eng.broker().get_positions().await.map(|p| p.len()).unwrap_or(0),
            "win_rate": perf.win_rate,
            "total_return_pct": perf.total_return_pct,
            "drawdown_pct": perf.drawdown_pct,
            "max_drawdown_pct": perf.max_drawdown_pct,
            "profit_factor": perf.profit_factor,
            "engine_running": status.running,
            "strategy": status.strategy,
            "total_fills": status.total_fills,
            "recent_trades": trades,
        }))
    } else {
        Json(json!({
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
            "open_positions": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "drawdown_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "engine_running": false,
            "strategy": "",
            "total_fills": 0,
            "recent_trades": [],
        }))
    }
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

fn generate_backtest_klines(symbol: &str, start: &str, end: &str) -> Vec<Kline> {
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 1, 1).unwrap());
    let end_date = NaiveDate::parse_from_str(end, "%Y-%m-%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 12, 31).unwrap());

    let base_price = match symbol {
        "600519.SH" => 1650.0,
        "000858.SZ" => 148.0,
        "601318.SH" => 52.0,
        "000001.SZ" => 12.5,
        "600036.SH" => 35.0,
        "300750.SZ" => 220.0,
        "600276.SH" => 28.0,
        _ => 100.0,
    };

    let mut klines = Vec::new();
    let mut price = base_price;
    let mut date = start_date;

    while date <= end_date {
        if date.weekday() == chrono::Weekday::Sat || date.weekday() == chrono::Weekday::Sun {
            date += chrono::Duration::days(1);
            continue;
        }
        let i = klines.len() as f64;
        let change = ((i * 7.3 + 13.7).sin() * 0.02 + (i * 3.1).cos() * 0.008) * price;
        let open = price;
        let close = price + change;
        let high = open.max(close) + ((i * 5.1).sin().abs()) * 0.005 * price;
        let low = open.min(close) - ((i * 4.3).cos().abs()) * 0.005 * price;
        let volume = 5_000_000.0 + ((i * 2.7).sin() * 3_000_000.0).abs();

        let datetime = date.and_hms_opt(15, 0, 0).unwrap();
        klines.push(Kline {
            symbol: symbol.to_string(),
            datetime,
            open: (open * 100.0).round() / 100.0,
            high: (high * 100.0).round() / 100.0,
            low: (low * 100.0).round() / 100.0,
            close: (close * 100.0).round() / 100.0,
            volume,
        });
        price = close;
        date += chrono::Duration::days(1);
    }
    klines
}

pub async fn run_backtest(
    State(_state): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> (StatusCode, Json<Value>) {
    use quant_backtest::engine::{BacktestConfig, BacktestEngine};
    use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum, MultiFactorStrategy, MultiFactorConfig};

    let capital = req.capital.unwrap_or(1_000_000.0);

    // Generate kline data for the requested date range
    let klines = generate_backtest_klines(&req.symbol, &req.start, &req.end);
    if klines.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "No kline data for date range"})));
    }

    let bt_config = BacktestConfig {
        initial_capital: capital,
        commission_rate: 0.001,
        stamp_tax_rate: 0.001,
        slippage_ticks: 1,
    };

    let engine = BacktestEngine::new(bt_config);

    // Instantiate strategy
    let mut strategy: Box<dyn quant_core::traits::Strategy> = match req.strategy.as_str() {
        "sma_cross" | "DualMaCrossover" => Box::new(DualMaCrossover::new(5, 20)),
        "rsi_reversal" | "RsiMeanReversion" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
        "macd_trend" | "MacdMomentum" => Box::new(MacdMomentum::new(12, 26, 9)),
        "multi_factor" | "MultiFactorModel" => Box::new(MultiFactorStrategy::new(MultiFactorConfig::default())),
        _ => Box::new(DualMaCrossover::new(5, 20)),
    };

    let result = engine.run(strategy.as_mut(), &klines);

    // Build equity curve JSON
    let equity_curve: Vec<Value> = result.equity_curve.iter().map(|(dt, val)| {
        json!({ "date": dt.format("%Y-%m-%d").to_string(), "value": (*val * 100.0).round() / 100.0 })
    }).collect();

    // Build trades JSON
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

    let m = &result.metrics;
    let id = format!("bt-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());

    (StatusCode::OK, Json(json!({
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
        "max_drawdown_percent": (m.max_drawdown * 10000.0).round() / 100.0,
        "max_drawdown_duration_days": m.max_drawdown_duration,
        "win_rate_percent": (m.win_rate * 10000.0).round() / 100.0,
        "total_trades": m.total_trades,
        "winning_trades": m.winning_trades,
        "losing_trades": m.losing_trades,
        "profit_factor": (m.profit_factor * 100.0).round() / 100.0,
        "avg_win": (m.avg_win * 100.0).round() / 100.0,
        "avg_loss": (m.avg_loss * 100.0).round() / 100.0,
        "equity_curve": equity_curve,
        "trades": trades,
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
        data_mode: if mode == "qmt" {
            // Live trading should use live data
            quant_broker::engine::DataMode::Live {
                tushare_url: state.config.tushare.base_url.clone(),
                tushare_token: state.config.tushare.token.clone(),
                akshare_url: state.config.akshare.base_url.clone(),
            }
        } else {
            // Paper mode uses simulated data by default
            quant_broker::engine::DataMode::Simulated
        },
        risk_config: quant_risk::enforcement::RiskConfig {
            stop_loss_pct: state.config.risk.max_drawdown.min(0.10),
            max_daily_loss_pct: state.config.risk.max_daily_loss,
            max_drawdown_pct: state.config.risk.max_drawdown,
            circuit_breaker_failures: 5,
            halt_on_drawdown: true,
        },
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

    let sentiment_store = state.sentiment_store.clone();
    engine.start(move || -> Box<dyn quant_core::traits::Strategy> {
        match strat_name.as_str() {
            "rsi_reversal" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
            "macd_trend" => Box::new(MacdMomentum::new(12, 26, 9)),
            "multi_factor" => Box::new(quant_strategy::builtin::MultiFactorStrategy::with_defaults()),
            "sentiment_aware" => Box::new(quant_strategy::sentiment::SentimentAwareStrategy::with_defaults(
                Box::new(quant_strategy::builtin::MultiFactorStrategy::with_defaults()),
                sentiment_store.clone(),
            )),
            "ml_factor" => Box::new(quant_strategy::ml_factor::MlFactorStrategy::with_defaults()),
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

// ── Sentiment API ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SentimentSubmitRequest {
    pub symbol: String,
    pub source: String,
    pub title: String,
    #[serde(default)]
    pub content: String,
    pub sentiment_score: f64,
    #[serde(default)]
    pub published_at: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SentimentQuery {
    pub start: Option<String>,
    pub end: Option<String>,
    pub limit: Option<usize>,
}

pub async fn sentiment_submit(
    State(state): State<AppState>,
    Json(req): Json<SentimentSubmitRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if req.symbol.is_empty() || req.title.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "symbol and title are required"})),
        ));
    }

    let published = req.published_at
        .as_ref()
        .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .map(|d| d.and_hms_opt(12, 0, 0).unwrap())
        .unwrap_or_else(|| chrono::Utc::now().naive_utc());

    let item = state.sentiment_store.submit(
        &req.symbol,
        &req.source,
        &req.title,
        &req.content,
        req.sentiment_score,
        published,
    );

    Ok(Json(json!({
        "status": "ok",
        "item": {
            "id": item.id.to_string(),
            "symbol": item.symbol,
            "source": item.source,
            "title": item.title,
            "sentiment_score": item.sentiment_score,
            "level": format!("{}", item.level()),
            "published_at": item.published_at.format("%Y-%m-%d %H:%M:%S").to_string(),
        }
    })))
}

pub async fn sentiment_batch_submit(
    State(state): State<AppState>,
    Json(items): Json<Vec<SentimentSubmitRequest>>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut count = 0;
    for req in items {
        if req.symbol.is_empty() || req.title.is_empty() {
            continue;
        }
        let published = req.published_at
            .as_ref()
            .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            .map(|d| d.and_hms_opt(12, 0, 0).unwrap())
            .unwrap_or_else(|| chrono::Utc::now().naive_utc());

        state.sentiment_store.submit(
            &req.symbol, &req.source, &req.title, &req.content,
            req.sentiment_score, published,
        );
        count += 1;
    }

    Ok(Json(json!({
        "status": "ok",
        "submitted": count,
        "total": state.sentiment_store.count(),
    })))
}

pub async fn sentiment_query(
    State(state): State<AppState>,
    Path(symbol): Path<String>,
    Query(q): Query<SentimentQuery>,
) -> Json<Value> {
    let start = q.start.as_ref()
        .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .map(|d| d.and_hms_opt(0, 0, 0).unwrap());
    let end = q.end.as_ref()
        .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .map(|d| d.and_hms_opt(23, 59, 59).unwrap());

    let items = state.sentiment_store.query_by_symbol(&symbol, start, end, q.limit);
    let summary = state.sentiment_store.summary(&symbol);

    Json(json!({
        "symbol": symbol,
        "summary": {
            "count": summary.count,
            "avg_score": summary.avg_score,
            "level": format!("{}", summary.level),
            "bullish_count": summary.bullish_count,
            "bearish_count": summary.bearish_count,
            "neutral_count": summary.neutral_count,
        },
        "items": items.iter().map(|it| json!({
            "id": it.id.to_string(),
            "source": it.source,
            "title": it.title,
            "content": it.content,
            "sentiment_score": it.sentiment_score,
            "level": format!("{}", it.level()),
            "published_at": it.published_at.format("%Y-%m-%d %H:%M:%S").to_string(),
        })).collect::<Vec<_>>(),
    }))
}

pub async fn sentiment_summary(
    State(state): State<AppState>,
) -> Json<Value> {
    let summaries = state.sentiment_store.all_summaries();

    Json(json!({
        "total_items": state.sentiment_store.count(),
        "symbols": summaries.iter().map(|s| json!({
            "symbol": s.symbol,
            "count": s.count,
            "avg_score": s.avg_score,
            "level": format!("{}", s.level),
            "bullish_count": s.bullish_count,
            "bearish_count": s.bearish_count,
            "neutral_count": s.neutral_count,
            "latest_title": s.latest_title,
            "latest_at": s.latest_at.map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string()),
        })).collect::<Vec<_>>(),
    }))
}

// ── Trade Journal ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct JournalQueryParams {
    pub symbol: Option<String>,
    pub entry_type: Option<String>,
    pub start: Option<String>,
    pub end: Option<String>,
    pub limit: Option<usize>,
}

pub async fn get_journal(
    State(state): State<AppState>,
    Query(q): Query<JournalQueryParams>,
) -> Json<Value> {
    let query = quant_broker::journal::JournalQuery {
        symbol: q.symbol,
        entry_type: q.entry_type,
        start: q.start,
        end: q.end,
        limit: q.limit,
    };
    let entries = state.journal.query(&query);
    let stats = state.journal.stats();
    let total = state.journal.count();

    Json(json!({
        "total": total,
        "entries": entries,
        "stats": stats.iter().map(|(t, c)| json!({"type": t, "count": c})).collect::<Vec<_>>(),
    }))
}

pub async fn get_journal_snapshots(
    State(state): State<AppState>,
    Query(q): Query<KlineQuery>,
) -> Json<Value> {
    let limit = q.limit.unwrap_or(30);
    let snapshots = state.journal.get_daily_snapshots(limit);
    Json(json!({ "snapshots": snapshots }))
}

// ── DL Models Research ──────────────────────────────────────────────

pub async fn research_dl_models() -> Json<Value> {
    let kb = quant_strategy::dl_models::build_knowledge_base();
    Json(serde_json::to_value(&kb).unwrap())
}

pub async fn research_dl_models_summary() -> Json<Value> {
    let kb = quant_strategy::dl_models::build_knowledge_base();
    let summary = quant_strategy::dl_models::summarize_knowledge_base(&kb);
    Json(serde_json::to_value(&summary).unwrap())
}

#[derive(Debug, Deserialize)]
pub struct CollectRequest {
    pub topic: Option<String>,
}

pub async fn research_dl_collect(
    State(state): State<AppState>,
    Json(body): Json<CollectRequest>,
) -> (StatusCode, Json<Value>) {
    let topic = body.topic.unwrap_or_else(|| "量化多因子深度学习模型最新进展".into());
    let prompt = quant_strategy::dl_models::build_collection_prompt(&topic);

    let llm = quant_llm::client::LlmClient::new(
        &state.config.llm.api_url,
        &state.config.llm.api_key,
        &state.config.llm.model,
        state.config.llm.temperature,
        state.config.llm.max_tokens,
    );

    let messages = vec![
        quant_llm::client::ChatMessage {
            role: "user".into(),
            content: Some(prompt),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    match llm.chat(&messages, None).await {
        Ok(resp) => {
            let content = resp.choices.first()
                .and_then(|c| c.message.content.as_ref())
                .cloned()
                .unwrap_or_default();

            // Try to parse as JSON array of collected items
            let collected: Vec<quant_strategy::dl_models::CollectedResearch> =
                serde_json::from_str(&content).unwrap_or_else(|_| {
                    // If not valid JSON, wrap the raw text as a single entry
                    vec![quant_strategy::dl_models::CollectedResearch {
                        title: format!("LLM研究摘要: {}", topic),
                        summary: content.clone(),
                        source: "LLM自动收集".into(),
                        relevance: "高".into(),
                        collected_at: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                    }]
                });

            (StatusCode::OK, Json(json!({
                "status": "ok",
                "topic": topic,
                "collected": collected,
                "raw_response": content,
            })))
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({
                "status": "error",
                "message": format!("LLM collection failed: {}", e),
                "hint": "请确保LLM配置正确(config/default.toml中的[llm]部分)"
            })))
        }
    }
}

// ── ML Model Retrain ────────────────────────────────────────────────

pub async fn ml_retrain(
    State(_state): State<AppState>,
    body: Option<Json<Value>>,
) -> (StatusCode, Json<Value>) {
    let algorithms = body
        .and_then(|b| b.get("algorithms").and_then(|a| a.as_str()).map(|s| s.to_string()))
        .unwrap_or_else(|| "lgb".to_string());

    let result = tokio::task::spawn_blocking(move || {
        let retrain_script = std::path::Path::new("ml_models/auto_retrain.py");
        if !retrain_script.exists() {
            return Err("auto_retrain.py not found in ml_models/".to_string());
        }

        let output = std::process::Command::new("python")
            .args([
                "ml_models/auto_retrain.py",
                "--no-notify",
                "--algorithms",
                &algorithms,
            ])
            .output()
            .map_err(|e| format!("Failed to start retrain: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(serde_json::json!({
                "status": "completed",
                "stdout": stdout,
                "stderr": stderr,
            }))
        } else {
            Err(format!("Retrain failed: {}", stderr))
        }
    }).await;

    match result {
        Ok(Ok(report)) => (StatusCode::OK, Json(report)),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("Task error: {}", e)}))),
    }
}

pub async fn ml_model_info() -> Json<Value> {
    // Try to read the latest retrain report
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

// ── Strategy Config Persistence ─────────────────────────────────────

const STRATEGY_CONFIG_PATH: &str = "data/strategy_config.json";

pub async fn save_strategy_config(
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    // Ensure data directory exists
    if let Err(e) = std::fs::create_dir_all("data") {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("Cannot create data dir: {}", e)})));
    }

    match std::fs::write(STRATEGY_CONFIG_PATH, serde_json::to_string_pretty(&body).unwrap_or_default()) {
        Ok(_) => (StatusCode::OK, Json(json!({"status": "saved", "path": STRATEGY_CONFIG_PATH}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("Save failed: {}", e)}))),
    }
}

pub async fn load_strategy_config() -> Json<Value> {
    let path = std::path::Path::new(STRATEGY_CONFIG_PATH);
    if !path.exists() {
        return Json(json!({"config": null, "exists": false}));
    }

    match std::fs::read_to_string(path) {
        Ok(content) => {
            let config: Value = serde_json::from_str(&content).unwrap_or(Value::Null);
            Json(json!({"config": config, "exists": true}))
        }
        Err(e) => Json(json!({"config": null, "exists": false, "error": format!("{}", e)})),
    }
}
