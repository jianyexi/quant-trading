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

fn generate_kline_data(symbol: &str, limit: usize, start: Option<&str>, end: Option<&str>, period: &str) -> Vec<Value> {
    let is_minute = matches!(period, "1" | "5" | "15" | "30" | "60");
    let dt_format = if is_minute { "%Y-%m-%d %H:%M" } else { "%Y-%m-%d" };

    // Determine date range
    let end_date = end
        .and_then(|e| chrono::NaiveDate::parse_from_str(e, "%Y-%m-%d").ok())
        .unwrap_or_else(|| chrono::Local::now().naive_local().date());
    let default_days = if is_minute { 5 } else { (limit as i64) * 7 / 5 + 10 };
    let start_date = start
        .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .unwrap_or_else(|| end_date - chrono::Duration::days(default_days));

    let start_str = start_date.format("%Y-%m-%d").to_string();
    let end_str = end_date.format("%Y-%m-%d").to_string();

    // Try real data
    if let Ok(klines) = fetch_real_klines_with_period(symbol, &start_str, &end_str, period) {
        if !klines.is_empty() {
            let take = klines.len().min(limit);
            let skip = klines.len().saturating_sub(take);
            return klines[skip..].iter().map(|k| {
                json!({
                    "date": k.datetime.format(dt_format).to_string(),
                    "open": k.open,
                    "high": k.high,
                    "low": k.low,
                    "close": k.close,
                    "volume": k.volume as u64
                })
            }).collect();
        }
    }

    // Fallback: GBM synthetic (daily only)
    if is_minute {
        return vec![]; // No synthetic minute data
    }
    let klines = generate_backtest_klines(symbol, &start_str, &end_str);
    let take = klines.len().min(limit);
    let skip = klines.len().saturating_sub(take);
    klines[skip..].iter().map(|k| {
        json!({
            "date": k.datetime.format("%Y-%m-%d").to_string(),
            "open": k.open,
            "high": k.high,
            "low": k.low,
            "close": k.close,
            "volume": k.volume as u64
        })
    }).collect()
}

pub async fn get_kline(
    Path(symbol): Path<String>,
    Query(params): Query<KlineQuery>,
    State(state): State<AppState>,
) -> Json<Value> {
    let limit = params.limit.unwrap_or(60);
    let period = params.period.as_deref().unwrap_or("daily");
    let data = generate_kline_data(&symbol, limit, params.start.as_deref(), params.end.as_deref(), period);
    if data.is_empty() {
        state.log_store.push(crate::log_store::LogLevel::Warn, "GET", &format!("/api/market/kline/{}", symbol), 200, 0,
            &format!("Kline returned 0 bars for {} period={}", symbol, period), None);
    }
    Json(json!({
        "symbol": symbol,
        "period": period,
        "start": params.start.unwrap_or_default(),
        "end": params.end.unwrap_or_default(),
        "data": data
    }))
}

pub async fn get_quote(
    Path(symbol): Path<String>,
    State(state): State<AppState>,
) -> Json<Value> {
    match fetch_real_quote(&symbol) {
        Ok(quote) => {
            Json(json!({
                "symbol": quote["symbol"].as_str().unwrap_or(&symbol),
                "name": quote["name"].as_str().unwrap_or(""),
                "price": quote["price"].as_f64().unwrap_or(0.0),
                "change": quote["change"].as_f64().unwrap_or(0.0),
                "change_percent": quote["change_percent"].as_f64().unwrap_or(0.0),
                "volume": quote["volume"].as_f64().unwrap_or(0.0) as u64,
                "turnover": quote["turnover"].as_f64().unwrap_or(0.0),
                "open": quote["open"].as_f64().unwrap_or(0.0),
                "high": quote["high"].as_f64().unwrap_or(0.0),
                "low": quote["low"].as_f64().unwrap_or(0.0),
                "date": quote["date"].as_str().unwrap_or(""),
                "timestamp": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()
            }))
        }
        Err(e) => {
            state.log_store.push(crate::log_store::LogLevel::Error, "GET", &format!("/api/market/quote/{}", symbol), 200, 0, &format!("Quote failed: {}", e), Some(e.clone()));
            Json(json!({
                "symbol": symbol,
                "name": "",
                "price": 0,
                "error": e
            }))
        }
    }
}

// ── Stock list ──────────────────────────────────────────────────────

pub async fn list_stocks() -> Json<Value> {
    // Try fetching real stock list
    if let Ok(data) = call_market_data(&["stock_list"]) {
        if data.get("stocks").is_some() {
            return Json(data);
        }
    }

    // Fallback: static curated list (names only, no fake prices)
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

// ── Python Market Data Bridge ─────────────────────────────────────

/// Call scripts/market_data.py with given command and args, return parsed JSON.
/// If a `log_store` is provided, errors are recorded with full detail.
fn call_market_data_logged(args: &[&str], log_store: Option<&crate::log_store::LogStore>) -> std::result::Result<Value, String> {
    use std::process::Command;

    let python = find_python().ok_or_else(|| "Python not found".to_string())?;
    let script = std::path::Path::new("scripts/market_data.py");
    if !script.exists() {
        let msg = format!("scripts/market_data.py not found (cwd={:?})", std::env::current_dir());
        if let Some(ls) = log_store {
            ls.push(crate::log_store::LogLevel::Error, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0, 0, &msg, None);
        }
        return Err(msg);
    }

    let start = std::time::Instant::now();
    let output = Command::new(&python)
        .arg(script)
        .args(args)
        .env("PYTHONIOENCODING", "utf-8")
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .output()
        .map_err(|e| {
            let msg = format!("Failed to run Python '{}': {}", python, e);
            if let Some(ls) = log_store {
                ls.push(crate::log_store::LogLevel::Error, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0, 0, &msg, None);
            }
            msg
        })?;
    let duration_ms = start.elapsed().as_millis() as u64;

    let stdout = String::from_utf8_lossy(&output.stdout);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let msg = format!("Python exit {}", output.status);
        if let Some(ls) = log_store {
            ls.push(crate::log_store::LogLevel::Error, "PYTHON", &format!("market_data.py {}", args.join(" ")), 1, duration_ms, &msg,
                Some(format!("stderr: {}\nstdout: {}", stderr, &stdout[..stdout.len().min(500)])));
        }
        return Err(format!("Python exit {}: stdout={}, stderr={}", output.status, stdout, stderr));
    }

    if stdout.trim().is_empty() {
        let msg = "Python returned empty output";
        if let Some(ls) = log_store {
            ls.push(crate::log_store::LogLevel::Warn, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0, duration_ms, msg, None);
        }
        return Err(msg.into());
    }

    let parsed: Value = serde_json::from_str(stdout.trim())
        .map_err(|e| {
            let msg = format!("Invalid JSON: {}", e);
            if let Some(ls) = log_store {
                ls.push(crate::log_store::LogLevel::Error, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0, duration_ms, &msg,
                    Some(format!("raw: {}", &stdout[..stdout.len().min(500)])));
            }
            format!("Invalid JSON: {} (raw: {})", e, &stdout[..stdout.len().min(200)])
        })?;

    if let Some(err) = parsed.get("error") {
        let msg = format!("akshare: {}", err);
        if let Some(ls) = log_store {
            ls.push(crate::log_store::LogLevel::Warn, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0, duration_ms, &msg, None);
        }
        return Err(msg);
    }
    Ok(parsed)
}

/// Convenience wrapper without logging
fn call_market_data(args: &[&str]) -> std::result::Result<Value, String> {
    call_market_data_logged(args, None)
}

/// Fetch real historical klines via akshare. `period`: "daily", "1", "5", "15", "30", "60"
fn fetch_real_klines_with_period(symbol: &str, start: &str, end: &str, period: &str) -> std::result::Result<Vec<Kline>, String> {
    use chrono::NaiveDateTime;

    let parsed = if period == "daily" {
        call_market_data(&["klines", symbol, start, end])?
    } else {
        call_market_data(&["klines", symbol, start, end, period])?
    };
    let arr = parsed.as_array().ok_or("Expected JSON array")?;
    if arr.is_empty() {
        return Err("akshare returned empty data".into());
    }

    let mut klines = Vec::with_capacity(arr.len());
    for item in arr {
        let sym = item["symbol"].as_str().unwrap_or(symbol).to_string();
        let dt_str = item["datetime"].as_str().unwrap_or("");
        let datetime = NaiveDateTime::parse_from_str(dt_str, "%Y-%m-%d %H:%M:%S")
            .map_err(|e| format!("Bad datetime '{}': {}", dt_str, e))?;
        klines.push(Kline {
            symbol: sym, datetime,
            open: item["open"].as_f64().unwrap_or(0.0),
            high: item["high"].as_f64().unwrap_or(0.0),
            low: item["low"].as_f64().unwrap_or(0.0),
            close: item["close"].as_f64().unwrap_or(0.0),
            volume: item["volume"].as_f64().unwrap_or(0.0),
        });
    }
    Ok(klines)
}

/// Convenience wrapper for daily klines (used by backtest/screener)
fn fetch_real_klines(symbol: &str, start: &str, end: &str) -> std::result::Result<Vec<Kline>, String> {
    fetch_real_klines_with_period(symbol, start, end, "daily")
}

/// Fetch real-time quote for a symbol.
fn fetch_real_quote(symbol: &str) -> std::result::Result<Value, String> {
    call_market_data(&["quote", symbol])
}

/// Fetch stock info for a symbol.
fn fetch_real_stock_info(symbol: &str) -> std::result::Result<Value, String> {
    call_market_data(&["stock_info", symbol])
}

/// Find a working Python 3 interpreter
fn find_python() -> Option<String> {
    let candidates = [
        std::env::var("PYTHON_PATH").unwrap_or_default(),
        r"C:\Users\jianyxi\AppData\Local\Programs\Python\Python312\python.exe".into(),
        "python3".into(),
        "python".into(),
    ];
    for path in &candidates {
        if path.is_empty() { continue; }
        if let Ok(output) = std::process::Command::new(path).arg("--version").output() {
            if output.status.success() {
                return Some(path.clone());
            }
        }
    }
    None
}

/// Stock-specific parameters for realistic price simulation
struct StockParams {
    base_price: f64,       // Starting price (yuan)
    annual_drift: f64,     // Annual expected return (e.g. 0.05 = +5%/yr)
    annual_vol: f64,       // Annual volatility (e.g. 0.25 = 25%)
    avg_volume: f64,       // Average daily volume
    volume_vol: f64,       // Volume variability factor (0-1)
}

fn stock_params(symbol: &str) -> StockParams {
    match symbol {
        // 贵州茅台: ~1500 yuan, low vol blue-chip
        "600519.SH" => StockParams { base_price: 1500.0, annual_drift: -0.05, annual_vol: 0.22, avg_volume: 3_500_000.0, volume_vol: 0.4 },
        // 五粮液: ~108 yuan
        "000858.SZ" => StockParams { base_price: 115.0, annual_drift: -0.08, annual_vol: 0.28, avg_volume: 8_000_000.0, volume_vol: 0.5 },
        // 中国平安: ~40 yuan
        "601318.SH" => StockParams { base_price: 42.0, annual_drift: 0.02, annual_vol: 0.25, avg_volume: 15_000_000.0, volume_vol: 0.45 },
        // 平安银行: ~11 yuan
        "000001.SZ" => StockParams { base_price: 11.5, annual_drift: -0.03, annual_vol: 0.22, avg_volume: 25_000_000.0, volume_vol: 0.5 },
        // 招商银行: ~35 yuan
        "600036.SH" => StockParams { base_price: 35.0, annual_drift: 0.04, annual_vol: 0.20, avg_volume: 12_000_000.0, volume_vol: 0.4 },
        // 宁德时代: ~200 yuan, higher vol growth stock
        "300750.SZ" => StockParams { base_price: 195.0, annual_drift: 0.0, annual_vol: 0.35, avg_volume: 6_000_000.0, volume_vol: 0.55 },
        // 恒瑞医药: ~48 yuan
        "600276.SH" => StockParams { base_price: 48.0, annual_drift: 0.03, annual_vol: 0.28, avg_volume: 10_000_000.0, volume_vol: 0.45 },
        _ => StockParams { base_price: 50.0, annual_drift: 0.0, annual_vol: 0.25, avg_volume: 5_000_000.0, volume_vol: 0.5 },
    }
}

/// Simple deterministic PRNG (xorshift64) for reproducible simulation
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Approximate standard normal via Box-Muller
    fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u1 = u1.max(1e-15); // avoid log(0)
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Uniform [0,1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

/// Generate realistic klines using Geometric Brownian Motion.
/// Prices follow: dS = μ·S·dt + σ·S·dW  (daily discretization)
/// Intraday OHLC is simulated from open→close with high/low excursions.
fn generate_backtest_klines(symbol: &str, start: &str, end: &str) -> Vec<Kline> {
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 1, 1).unwrap());
    let end_date = NaiveDate::parse_from_str(end, "%Y-%m-%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 12, 31).unwrap());

    let params = stock_params(symbol);
    let dt = 1.0 / 252.0; // One trading day
    let daily_drift = (params.annual_drift - 0.5 * params.annual_vol.powi(2)) * dt;
    let daily_vol = params.annual_vol * dt.sqrt();

    // Deterministic seed from symbol hash so same symbol always produces same path
    let seed: u64 = symbol.bytes().fold(42u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut rng = Rng::new(seed);

    let mut klines = Vec::new();
    let mut price = params.base_price;
    let mut date = start_date;

    // A-share: ±10% daily limit (±20% for 创业板 300xxx)
    let limit_pct = if symbol.starts_with("300") { 0.20 } else { 0.10 };

    while date <= end_date {
        // Skip weekends
        if date.weekday() == chrono::Weekday::Sat || date.weekday() == chrono::Weekday::Sun {
            date += chrono::Duration::days(1);
            continue;
        }

        let open = price;

        // GBM: log-return ~ N(daily_drift, daily_vol²)
        let z = rng.next_normal();
        let log_return = daily_drift + daily_vol * z;
        let mut close = open * log_return.exp();

        // Enforce daily price limit
        let upper = open * (1.0 + limit_pct);
        let lower = open * (1.0 - limit_pct);
        close = close.clamp(lower, upper);

        // Simulate intraday high/low as excursions beyond open-close range
        let body_high = open.max(close);
        let body_low = open.min(close);
        let body_range = (body_high - body_low).max(open * 0.001);

        // Wicks: random excursion 0–100% of body range beyond OHLC body
        let upper_wick = rng.next_f64() * body_range;
        let lower_wick = rng.next_f64() * body_range;
        let high = (body_high + upper_wick).min(upper);
        let low = (body_low - lower_wick).max(lower).max(0.01);

        // Volume: log-normal around avg with correlation to |return|
        let vol_z = rng.next_normal();
        let return_magnitude = log_return.abs() / daily_vol; // normalized
        let vol_multiplier = (params.volume_vol * vol_z + 0.3 * return_magnitude).exp();
        let volume = params.avg_volume * vol_multiplier;

        let datetime = date.and_hms_opt(15, 0, 0).unwrap();
        klines.push(Kline {
            symbol: symbol.to_string(),
            datetime,
            open: (open * 100.0).round() / 100.0,
            high: (high * 100.0).round() / 100.0,
            low: (low * 100.0).round() / 100.0,
            close: (close * 100.0).round() / 100.0,
            volume: (volume / 100.0).round() * 100.0,
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
    use quant_strategy::ml_factor::{MlFactorStrategy, MlFactorConfig};

    let capital = req.capital.unwrap_or(1_000_000.0);
    let period = req.period.as_deref().unwrap_or("daily");

    // Try fetching real market data first, fall back to synthetic GBM
    let (klines, data_source) = match fetch_real_klines_with_period(&req.symbol, &req.start, &req.end, period) {
        Ok(k) if !k.is_empty() => {
            let n = k.len();
            let label = if period == "daily" { "日线" } else { &format!("{}分钟线", period) };
            (k, format!("akshare ({}条真实{})", n, label))
        }
        Ok(_) | Err(_) if period != "daily" => {
            // Minute data only available for recent ~5 trading days
            return (StatusCode::BAD_REQUEST, Json(json!({
                "error": format!("无法获取{}分钟级数据。分钟K线仅支持近5个交易日，请缩短日期范围或使用日线(daily)。", period)
            })));
        }
        Ok(_) => {
            let k = generate_backtest_klines(&req.symbol, &req.start, &req.end);
            (k, "synthetic (akshare返回空数据)".to_string())
        }
        Err(reason) => {
            let k = generate_backtest_klines(&req.symbol, &req.start, &req.end);
            (k, format!("synthetic ({})", reason))
        }
    };

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
        "ml_factor" | "MlFactor" => Box::new(MlFactorStrategy::new(MlFactorConfig::default())),
        _ => Box::new(DualMaCrossover::new(5, 20)),
    };

    let result = engine.run(strategy.as_mut(), &klines);

    // Build equity curve JSON
    let eq_fmt = if period == "daily" { "%Y-%m-%d" } else { "%Y-%m-%d %H:%M" };
    let equity_curve: Vec<Value> = result.equity_curve.iter().map(|(dt, val)| {
        json!({ "date": dt.format(eq_fmt).to_string(), "value": (*val * 100.0).round() / 100.0 })
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
        "data_source": data_source,
        "period": period,
        "status": "completed"
    })))
}

pub async fn get_backtest_results(
    Path(id): Path<String>,
    State(_state): State<AppState>,
) -> Json<Value> {
    // Results are not persisted yet — return not found
    Json(json!({
        "id": id,
        "status": "not_found",
        "error": "Backtest results are not persisted. Please run a new backtest."
    }))
}

// ── Order Handlers ──────────────────────────────────────────────────

pub async fn list_orders(
    State(state): State<AppState>,
) -> Json<Value> {
    // Return recent trades from trading engine status
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
                // Try to get current price from market
                let current_price = fetch_real_quote(sym)
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
        // Engine exists but no account
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
    /// "paper" (default), "qmt" for live trading, or "replay" for historical replay
    pub mode: Option<String>,
    /// Replay start date (YYYY-MM-DD), required for mode="replay"
    pub replay_start: Option<String>,
    /// Replay end date (YYYY-MM-DD), required for mode="replay"
    pub replay_end: Option<String>,
    /// Replay speed multiplier (0=max speed, 1=real-time, 10=10x), default=0
    pub replay_speed: Option<f64>,
    /// K-line period for replay: "daily", "1", "5", "15", "30", "60" (minutes), default="daily"
    pub replay_period: Option<String>,
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
        } else {
            // Paper mode now uses real akshare data via Python bridge
            quant_broker::engine::DataMode::PythonBridge
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

    // Wire the shared journal from AppState
    engine.set_journal(state.journal.clone());

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
        "position_size": position_size,
        "replay_start": req.replay_start,
        "replay_end": req.replay_end,
        "replay_speed": req.replay_speed,
        "replay_period": req.replay_period
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
        // Fetch real klines; skip if unavailable
        match fetch_real_klines(symbol, &start_str, &end_str) {
            Ok(klines) if !klines.is_empty() => {
                stock_data.insert(symbol.to_string(), (name.to_string(), klines));
            }
            _ => {
                // Fallback to GBM synthetic for this stock
                let klines = generate_backtest_klines(symbol, &start_str, &end_str);
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

    // Get stock name from info API
    let name = fetch_real_stock_info(&symbol)
        .ok()
        .and_then(|v| v["name"].as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| symbol.clone());

    // Fetch real klines
    let klines = match fetch_real_klines(&symbol, &start_str, &end_str) {
        Ok(k) if !k.is_empty() => k,
        _ => generate_backtest_klines(&symbol, &start_str, &end_str),
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
