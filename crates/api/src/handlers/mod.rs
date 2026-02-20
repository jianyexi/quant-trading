mod market;
mod backtest;
mod trade;
mod factor;
mod notification;

pub use market::*;
pub use backtest::*;
pub use trade::*;
pub use factor::*;
pub use notification::*;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::NaiveDate;
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
    /// "daily" (default), "1", "5", "15", "30", "60" for minute-level
    pub period: Option<String>,
}

// ── Backtest Request ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BacktestRequest {
    pub strategy: String,
    pub symbol: String,
    pub start: String,
    pub end: String,
    pub capital: Option<f64>,
    /// "daily" (default), "1", "5", "15", "30", "60"
    pub period: Option<String>,
    /// ML inference mode: "embedded" (default), "tcp_mq", "http"
    pub inference_mode: Option<String>,
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

// ── Order Handlers ──────────────────────────────────────────────────

pub async fn list_orders(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine = state.engine.lock().await;
    if let Some(eng) = engine.as_ref() {
        let status = eng.status().await;
        let orders: Vec<Value> = status.recent_trades.iter().map(|t| {
            json!({
                "id": t.order_id.to_string(),
                "time": t.timestamp.format("%Y-%m-%d %H:%M:%S").to_string(),
                "symbol": t.symbol,
                "side": format!("{:?}", t.side).to_lowercase(),
                "price": (t.price * 100.0).round() / 100.0,
                "quantity": t.quantity as i64,
                "status": "filled",
            })
        }).collect();
        return Json(json!({ "orders": orders }));
    }
    Json(json!({ "orders": [] }))
}

// ── Portfolio Handlers ──────────────────────────────────────────────

pub async fn get_portfolio(
    State(state): State<AppState>,
) -> Json<Value> {
    let engine = state.engine.lock().await;
    if let Some(eng) = engine.as_ref() {
        let status = eng.status().await;
        let account = eng.broker().get_account().await.ok();
        if let Some(acct) = account {
            let mut positions_json: Vec<Value> = Vec::new();
            for (sym, pos) in &acct.portfolio.positions {
                let current_price = market::fetch_real_quote(sym)
                    .ok()
                    .and_then(|q| q["price"].as_f64())
                    .unwrap_or(pos.current_price);
                let pnl = (current_price - pos.avg_cost) * pos.quantity;
                positions_json.push(json!({
                    "symbol": sym,
                    "name": "",
                    "shares": pos.quantity as i64,
                    "avg_cost": (pos.avg_cost * 100.0).round() / 100.0,
                    "current_price": (current_price * 100.0).round() / 100.0,
                    "pnl": (pnl * 100.0).round() / 100.0,
                }));
            }
            return Json(json!({
                "total_value": (acct.portfolio.total_value * 100.0).round() / 100.0,
                "cash": (acct.portfolio.cash * 100.0).round() / 100.0,
                "total_pnl": ((acct.portfolio.total_value - acct.initial_capital) * 100.0).round() / 100.0,
                "positions": positions_json,
            }));
        }
        return Json(json!({
            "total_value": status.performance.portfolio_value,
            "cash": 0,
            "total_pnl": status.pnl,
            "positions": []
        }));
    }
    Json(json!({
        "total_value": 0,
        "cash": 0,
        "total_pnl": 0,
        "positions": []
    }))
}

// ── Chat Handlers ───────────────────────────────────────────────────

pub async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    use quant_llm::{client::LlmClient, context::ConversationContext, tools::{get_all_tools, ToolExecutor}};

    let llm_config = &state.config.llm;

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

    let symbols: Vec<(&str, &str)> = vec![
        ("600519.SH", "贵州茅台"), ("000858.SZ", "五粮液"),
        ("601318.SH", "中国平安"), ("000001.SZ", "平安银行"),
        ("600036.SH", "招商银行"), ("300750.SZ", "宁德时代"),
        ("600276.SH", "恒瑞医药"), ("000333.SZ", "美的集团"),
        ("601888.SH", "中国中免"), ("002594.SZ", "比亚迪"),
        ("601012.SH", "隆基绿能"), ("600900.SH", "长江电力"),
        ("000568.SZ", "泸州老窖"), ("600809.SH", "山西汾酒"),
        ("002475.SZ", "立讯精密"), ("600030.SH", "中信证券"),
        ("601166.SH", "兴业银行"), ("000661.SZ", "长春高新"),
        ("002714.SZ", "牧原股份"), ("600585.SH", "海螺水泥"),
    ];

    let today = chrono::Local::now().naive_local().date();
    let end_str = today.format("%Y-%m-%d").to_string();
    let start_date = today - chrono::Duration::days(150);
    let start_str = start_date.format("%Y-%m-%d").to_string();

    let mut stock_data: HashMap<String, (String, Vec<Kline>)> = HashMap::new();
    for (symbol, name) in &symbols {
        match market::fetch_real_klines(symbol, &start_str, &end_str) {
            Ok(klines) if !klines.is_empty() => {
                stock_data.insert(symbol.to_string(), (name.to_string(), klines));
            }
            _ => {
                let klines = market::generate_backtest_klines(symbol, &start_str, &end_str);
                if !klines.is_empty() {
                    stock_data.insert(symbol.to_string(), (name.to_string(), klines));
                }
            }
        }
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

    let today = chrono::Local::now().naive_local().date();
    let end_str = today.format("%Y-%m-%d").to_string();
    let start_date = today - chrono::Duration::days(150);
    let start_str = start_date.format("%Y-%m-%d").to_string();

    let name = market::fetch_real_stock_info(&symbol)
        .ok()
        .and_then(|v| v["name"].as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| symbol.clone());

    let klines = match market::fetch_real_klines(&symbol, &start_str, &end_str) {
        Ok(k) if !k.is_empty() => k,
        _ => market::generate_backtest_klines(&symbol, &start_str, &end_str),
    };

    let mut stock_data: HashMap<String, (String, Vec<Kline>)> = HashMap::new();
    stock_data.insert(symbol.clone(), (name, klines));

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

            let collected: Vec<quant_strategy::dl_models::CollectedResearch> =
                serde_json::from_str(&content).unwrap_or_else(|_| {
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

// ── Strategy Config Persistence ─────────────────────────────────────

const STRATEGY_CONFIG_PATH: &str = "data/strategy_config.json";

pub async fn save_strategy_config(
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
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

// ── Logs ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct LogQueryParams {
    pub level: Option<String>,
    pub path: Option<String>,
    pub limit: Option<usize>,
}

pub async fn get_logs(
    State(state): State<AppState>,
    Query(q): Query<LogQueryParams>,
) -> Json<Value> {
    use crate::log_store::LogLevel;

    let level = q.level.as_deref().and_then(|l| match l {
        "error" => Some(LogLevel::Error),
        "warn" => Some(LogLevel::Warn),
        "info" => Some(LogLevel::Info),
        _ => None,
    });
    let limit = q.limit.unwrap_or(200);
    let entries = state.log_store.query(level, q.path.as_deref(), limit);
    let (info_count, warn_count, error_count) = state.log_store.summary();
    let total = state.log_store.count();

    Json(json!({
        "total": total,
        "entries": entries,
        "summary": {
            "info": info_count,
            "warn": warn_count,
            "error": error_count,
        }
    }))
}

pub async fn clear_logs(
    State(state): State<AppState>,
) -> Json<Value> {
    state.log_store.clear();
    Json(json!({"status": "cleared"}))
}

// ── Sentiment Collector ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CollectorStartRequest {
    /// Override watch symbols (optional)
    pub symbols: Option<Vec<String>>,
}

pub async fn collector_start(
    State(state): State<AppState>,
    body: Option<Json<CollectorStartRequest>>,
) -> (StatusCode, Json<Value>) {
    let mut collector = state.sentiment_collector.lock().await;

    if let Some(Json(req)) = body {
        if let Some(symbols) = req.symbols {
            if !symbols.is_empty() {
                collector.update_symbols(symbols);
            }
        }
    }

    let llm_config = &state.config.llm;
    let llm_client = quant_llm::client::LlmClient::new(
        &llm_config.api_url,
        &llm_config.api_key,
        &llm_config.model,
        llm_config.temperature,
        llm_config.max_tokens,
    );

    match collector.start(
        state.sentiment_store.clone(),
        llm_client,
        state.config.akshare.base_url.clone(),
    ) {
        Ok(()) => (StatusCode::OK, Json(json!({
            "status": "started",
            "message": "Sentiment collector started"
        }))),
        Err(e) => (StatusCode::CONFLICT, Json(json!({
            "status": "error",
            "message": e
        }))),
    }
}

pub async fn collector_stop(
    State(state): State<AppState>,
) -> Json<Value> {
    let collector = state.sentiment_collector.lock().await;
    collector.stop();
    Json(json!({
        "status": "stopped",
        "message": "Sentiment collector stopped"
    }))
}

pub async fn collector_status(
    State(state): State<AppState>,
) -> Json<Value> {
    let collector = state.sentiment_collector.lock().await;
    let status = collector.status().await;
    Json(json!(status))
}

// ── Helper: find Python & run Python script ─────────────────────────

/// Find a working Python 3 interpreter
pub(crate) fn find_python() -> Option<String> {
    quant_core::utils::find_python()
}

/// Run a Python script and capture output
pub(crate) fn run_python_script(python: &str, args: &[String]) -> Result<Value, String> {
    quant_core::utils::run_python_script(python, args)
}

pub(crate) fn flatten_spawn_result(
    result: Result<Result<Value, String>, tokio::task::JoinError>,
) -> (StatusCode, Json<Value>) {
    match result {
        Ok(Ok(val)) => (StatusCode::OK, Json(val)),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e}))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("Task error: {}", e)}))),
    }
}
