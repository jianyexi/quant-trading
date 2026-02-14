use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::state::AppState;

// ── Query Parameters ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct KlineQuery {
    pub start: Option<String>,
    pub end: Option<String>,
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

// ── Market Handlers ─────────────────────────────────────────────────

pub async fn get_kline(
    Path(symbol): Path<String>,
    Query(params): Query<KlineQuery>,
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "symbol": symbol,
        "start": params.start.unwrap_or_default(),
        "end": params.end.unwrap_or_default(),
        "data": [
            {"date": "2024-01-02", "open": 100.0, "high": 102.5, "low": 99.5, "close": 101.8, "volume": 15000000},
            {"date": "2024-01-03", "open": 101.8, "high": 103.2, "low": 101.0, "close": 102.5, "volume": 18000000},
        ]
    }))
}

pub async fn get_quote(
    Path(symbol): Path<String>,
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "symbol": symbol,
        "price": 101.5,
        "change": 1.2,
        "change_percent": 1.19,
        "volume": 12000000,
        "timestamp": "2024-01-03T15:00:00"
    }))
}

// ── Backtest Handlers ───────────────────────────────────────────────

pub async fn run_backtest(
    State(_state): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> (StatusCode, Json<Value>) {
    let capital = req.capital.unwrap_or(1_000_000.0);
    (StatusCode::OK, Json(json!({
        "id": "bt-001",
        "strategy": req.strategy,
        "symbol": req.symbol,
        "start": req.start,
        "end": req.end,
        "initial_capital": capital,
        "final_value": capital * 1.25,
        "total_return_percent": 25.0,
        "sharpe_ratio": 1.45,
        "max_drawdown_percent": 12.3,
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
        "orders": []
    }))
}

// ── Portfolio Handlers ──────────────────────────────────────────────

pub async fn get_portfolio(
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "total_value": 1_250_000.0,
        "cash": 350_000.0,
        "total_pnl": 250_000.0,
        "positions": [
            {"symbol": "600519.SH", "name": "贵州茅台", "shares": 100, "avg_cost": 1800.0, "current_price": 1950.0, "pnl": 15000.0},
            {"symbol": "000858.SZ", "name": "五粮液", "shares": 500, "avg_cost": 160.0, "current_price": 172.0, "pnl": 6000.0},
        ]
    }))
}

// ── Chat Handlers ───────────────────────────────────────────────────

pub async fn chat(
    State(_state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    // Stub: echo the message back with a placeholder response
    Json(ChatResponse {
        reply: format!("Received your message: '{}'. LLM integration pending.", req.message),
    })
}

pub async fn chat_history(
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "sessions": []
    }))
}
