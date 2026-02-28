use axum::{
    extract::{Path, Query, State},
    Json,
};
use chrono::{Datelike, NaiveDate};
use serde_json::{json, Value};

use quant_core::models::Kline;
use crate::state::AppState;
use super::KlineQuery;

// ── Python Market Data Bridge ─────────────────────────────────────

/// Call scripts/market_data.py with given command and args, return parsed JSON.
/// If a `log_store` is provided, errors are recorded with full detail.
/// Enforces a 15-second timeout to prevent hanging when akshare is unreachable.
fn call_market_data_logged(args: &[&str], log_store: Option<&crate::log_store::LogStore>) -> std::result::Result<Value, String> {
    use std::process::Command;

    let python = super::find_python().ok_or_else(|| "Python not found".to_string())?;
    let script = std::path::Path::new("scripts/market_data.py");
    if !script.exists() {
        let msg = format!("scripts/market_data.py not found (cwd={:?})", std::env::current_dir());
        if let Some(ls) = log_store {
            ls.push(crate::log_store::LogLevel::Error, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0, 0, &msg, None);
        }
        return Err(msg);
    }

    let start = std::time::Instant::now();
    let mut child = Command::new(&python)
        .arg(script)
        .args(args)
        .env("PYTHONIOENCODING", "utf-8")
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            let msg = format!("Failed to run Python '{}': {}", python, e);
            if let Some(ls) = log_store {
                ls.push(crate::log_store::LogLevel::Error, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0, 0, &msg, None);
            }
            msg
        })?;

    // Poll with timeout to prevent hanging when data sources are unreachable
    // Use 45s for klines (gap-filling can be slow), 15s for other commands
    let is_klines = args.first().map_or(false, |a| *a == "klines");
    let timeout = std::time::Duration::from_secs(if is_klines { 45 } else { 15 });
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let msg = format!("Python timed out after {}s", timeout.as_secs());
                    if let Some(ls) = log_store {
                        ls.push(crate::log_store::LogLevel::Warn, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0,
                            start.elapsed().as_millis() as u64, &msg, None);
                    }
                    return Err(msg);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                let _ = child.kill();
                let msg = format!("Failed to wait on Python: {}", e);
                if let Some(ls) = log_store {
                    ls.push(crate::log_store::LogLevel::Error, "PYTHON", &format!("market_data.py {}", args.join(" ")), 0, 0, &msg, None);
                }
                return Err(msg);
            }
        }
    }
    let output = child.wait_with_output().map_err(|e| format!("Failed to read output: {}", e))?;
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
pub(crate) fn call_market_data(args: &[&str]) -> std::result::Result<Value, String> {
    call_market_data_logged(args, None)
}

/// Fetch real historical klines via akshare. `period`: "daily", "1", "5", "15", "30", "60"
/// Retries up to 2 times with backoff on transient failures.
pub(crate) fn fetch_real_klines_with_period(symbol: &str, start: &str, end: &str, period: &str) -> std::result::Result<Vec<Kline>, String> {
    use chrono::NaiveDateTime;

    let max_attempts = 3;
    let mut last_err = String::new();

    for attempt in 1..=max_attempts {
        let parsed = if period == "daily" {
            call_market_data(&["klines", symbol, start, end])
        } else {
            call_market_data(&["klines", symbol, start, end, period])
        };

        match parsed {
            Ok(val) => {
                let arr = match val.as_array() {
                    Some(a) => a,
                    None => return Err("Expected JSON array".into()),
                };
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
                return Ok(klines);
            }
            Err(e) => {
                last_err = e;
                // Don't retry on non-transient errors (bad symbol, empty data)
                if last_err.contains("empty") || last_err.contains("usage:") {
                    return Err(last_err);
                }
                if attempt < max_attempts {
                    tracing::warn!(symbol, attempt, error=%last_err, "Kline fetch failed, retrying...");
                    std::thread::sleep(std::time::Duration::from_millis(500 * attempt as u64));
                }
            }
        }
    }

    Err(format!("Failed after {} attempts: {}", max_attempts, last_err))
}

/// Convenience wrapper for daily klines (used by backtest/screener)
pub(crate) fn fetch_real_klines(symbol: &str, start: &str, end: &str) -> std::result::Result<Vec<Kline>, String> {
    fetch_real_klines_with_period(symbol, start, end, "daily")
}

/// Fetch real-time quote for a symbol.
pub(crate) fn fetch_real_quote(symbol: &str) -> std::result::Result<Value, String> {
    call_market_data(&["quote", symbol])
}

/// Fetch stock info for a symbol.
pub(crate) fn fetch_real_stock_info(symbol: &str) -> std::result::Result<Value, String> {
    call_market_data(&["stock_info", symbol])
}

/// Stock-specific parameters for realistic price simulation
struct StockParams {
    base_price: f64,
    annual_drift: f64,
    annual_vol: f64,
    avg_volume: f64,
    volume_vol: f64,
}

fn stock_params(symbol: &str) -> StockParams {
    match symbol {
        "600519.SH" => StockParams { base_price: 1500.0, annual_drift: -0.05, annual_vol: 0.22, avg_volume: 3_500_000.0, volume_vol: 0.4 },
        "000858.SZ" => StockParams { base_price: 115.0, annual_drift: -0.08, annual_vol: 0.28, avg_volume: 8_000_000.0, volume_vol: 0.5 },
        "601318.SH" => StockParams { base_price: 42.0, annual_drift: 0.02, annual_vol: 0.25, avg_volume: 15_000_000.0, volume_vol: 0.45 },
        "000001.SZ" => StockParams { base_price: 11.5, annual_drift: -0.03, annual_vol: 0.22, avg_volume: 25_000_000.0, volume_vol: 0.5 },
        "600036.SH" => StockParams { base_price: 35.0, annual_drift: 0.04, annual_vol: 0.20, avg_volume: 12_000_000.0, volume_vol: 0.4 },
        "300750.SZ" => StockParams { base_price: 195.0, annual_drift: 0.0, annual_vol: 0.35, avg_volume: 6_000_000.0, volume_vol: 0.55 },
        "600276.SH" => StockParams { base_price: 48.0, annual_drift: 0.03, annual_vol: 0.28, avg_volume: 10_000_000.0, volume_vol: 0.45 },
        "002594.SZ" => StockParams { base_price: 240.0, annual_drift: 0.10, annual_vol: 0.38, avg_volume: 8_000_000.0, volume_vol: 0.55 },
        "688981.SH" => StockParams { base_price: 50.0, annual_drift: -0.05, annual_vol: 0.40, avg_volume: 12_000_000.0, volume_vol: 0.6 },
        "300760.SZ" => StockParams { base_price: 280.0, annual_drift: 0.05, annual_vol: 0.30, avg_volume: 2_000_000.0, volume_vol: 0.45 },
        "000333.SZ" => StockParams { base_price: 60.0, annual_drift: 0.06, annual_vol: 0.25, avg_volume: 10_000_000.0, volume_vol: 0.45 },
        "601888.SH" => StockParams { base_price: 70.0, annual_drift: -0.10, annual_vol: 0.35, avg_volume: 6_000_000.0, volume_vol: 0.5 },
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
pub(crate) fn generate_backtest_klines(symbol: &str, start: &str, end: &str) -> Vec<Kline> {
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 1, 1).unwrap());
    let end_date = NaiveDate::parse_from_str(end, "%Y-%m-%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2024, 12, 31).unwrap());

    let params = stock_params(symbol);
    let dt = 1.0 / 252.0;
    let daily_drift = (params.annual_drift - 0.5 * params.annual_vol.powi(2)) * dt;
    let daily_vol = params.annual_vol * dt.sqrt();

    let seed: u64 = symbol.bytes().fold(42u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut rng = Rng::new(seed);

    let mut klines = Vec::new();
    let mut price = params.base_price;
    let mut date = start_date;

    let limit_pct = if symbol.starts_with("300") || symbol.starts_with("688") { 0.20 } else { 0.10 };

    while date <= end_date {
        if date.weekday() == chrono::Weekday::Sat || date.weekday() == chrono::Weekday::Sun {
            date += chrono::Duration::days(1);
            continue;
        }

        let open = price;
        let z = rng.next_normal();
        let log_return = daily_drift + daily_vol * z;
        let mut close = open * log_return.exp();

        let upper = open * (1.0 + limit_pct);
        let lower = open * (1.0 - limit_pct);
        close = close.clamp(lower, upper);

        let body_high = open.max(close);
        let body_low = open.min(close);
        let body_range = (body_high - body_low).max(open * 0.001);

        let upper_wick = rng.next_f64() * body_range;
        let lower_wick = rng.next_f64() * body_range;
        let high = (body_high + upper_wick).min(upper);
        let low = (body_low - lower_wick).max(lower).max(0.01);

        let vol_z = rng.next_normal();
        let return_magnitude = log_return.abs() / daily_vol;
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

// ── Market Handlers ─────────────────────────────────────────────────

fn generate_kline_data(symbol: &str, limit: usize, start: Option<&str>, end: Option<&str>, period: &str) -> Vec<Value> {
    let is_minute = matches!(period, "1" | "5" | "15" | "30" | "60");
    let dt_format = if is_minute { "%Y-%m-%d %H:%M" } else { "%Y-%m-%d" };

    let end_date = end
        .and_then(|e| chrono::NaiveDate::parse_from_str(e, "%Y-%m-%d").ok())
        .unwrap_or_else(|| chrono::Local::now().naive_local().date());
    let default_days = if is_minute { 5 } else { (limit as i64) * 7 / 5 + 10 };
    let start_date = start
        .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .unwrap_or_else(|| end_date - chrono::Duration::days(default_days));

    let start_str = start_date.format("%Y-%m-%d").to_string();
    let end_str = end_date.format("%Y-%m-%d").to_string();

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

    if is_minute {
        return vec![];
    }
    // No synthetic fallback — return empty if real data unavailable
    vec![]
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
    if let Ok(data) = call_market_data(&["stock_list"]) {
        if data.get("stocks").is_some() {
            return Json(data);
        }
    }

    Json(json!({
        "stocks": [
            {"symbol": "600519.SH", "name": "贵州茅台", "industry": "白酒", "market": "SSE"},
            {"symbol": "000858.SZ", "name": "五粮液", "industry": "白酒", "market": "SZSE"},
            {"symbol": "601318.SH", "name": "中国平安", "industry": "保险", "market": "SSE"},
            {"symbol": "000001.SZ", "name": "平安银行", "industry": "银行", "market": "SZSE"},
            {"symbol": "600036.SH", "name": "招商银行", "industry": "银行", "market": "SSE"},
            {"symbol": "600276.SH", "name": "恒瑞医药", "industry": "医药", "market": "SSE"},
            {"symbol": "000333.SZ", "name": "美的集团", "industry": "家电", "market": "SZSE"},
            {"symbol": "601888.SH", "name": "中国中免", "industry": "零售", "market": "SSE"},
            {"symbol": "002594.SZ", "name": "比亚迪", "industry": "汽车", "market": "SZSE"},
            {"symbol": "000651.SZ", "name": "格力电器", "industry": "家电", "market": "SZSE"},
            {"symbol": "300750.SZ", "name": "宁德时代", "industry": "电池", "market": "创业板"},
            {"symbol": "300760.SZ", "name": "迈瑞医疗", "industry": "医疗器械", "market": "创业板"},
            {"symbol": "300059.SZ", "name": "东方财富", "industry": "券商", "market": "创业板"},
            {"symbol": "300122.SZ", "name": "智飞生物", "industry": "疫苗", "market": "创业板"},
            {"symbol": "300782.SZ", "name": "卓胜微", "industry": "芯片", "market": "创业板"},
            {"symbol": "688981.SH", "name": "中芯国际", "industry": "半导体", "market": "科创板"},
            {"symbol": "688111.SH", "name": "金山办公", "industry": "软件", "market": "科创板"},
            {"symbol": "688036.SH", "name": "传音控股", "industry": "手机", "market": "科创板"},
            {"symbol": "688561.SH", "name": "奇安信", "industry": "网络安全", "market": "科创板"},
            {"symbol": "688005.SH", "name": "容百科技", "industry": "锂电材料", "market": "科创板"}
        ]
    }))
}

// ── Market Data Cache ───────────────────────────────────────────────

/// GET /api/market/data-source — check which data sources are available
pub async fn data_source_status() -> Json<Value> {
    match call_market_data(&["data_source_status"]) {
        Ok(data) => Json(data),
        Err(e) => Json(json!({"primary": null, "available": [], "error": e})),
    }
}

pub async fn cache_status() -> Json<Value> {
    match call_market_data(&["cache_status"]) {
        Ok(data) => Json(data),
        Err(e) => Json(json!({"error": e, "symbols": []})),
    }
}

#[derive(serde::Deserialize)]
pub struct SyncDataRequest {
    pub symbols: Vec<String>,
    #[serde(default = "default_sync_start")]
    pub start_date: String,
    #[serde(default = "default_sync_end")]
    pub end_date: String,
}

fn default_sync_start() -> String { "2020-01-01".to_string() }
fn default_sync_end() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

pub async fn sync_data(
    State(state): State<AppState>,
    Json(req): Json<SyncDataRequest>,
) -> Json<Value> {
    let ts = state.task_store.clone();
    let task_id = ts.create("data_sync");
    ts.set_progress(&task_id, &format!("Syncing {} symbols...", req.symbols.len()));

    let symbols_str = req.symbols.join(",");
    let start = req.start_date.clone();
    let end = req.end_date.clone();
    let tid = task_id.clone();

    tokio::task::spawn_blocking(move || {
        // Use market_data.py sync_cache command
        let result = std::process::Command::new("python3")
            .args(["scripts/market_data.py", "sync_cache", &symbols_str, &start, &end])
            .output();
        match result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                if output.status.success() {
                    ts.complete(&tid, &stdout);
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    ts.fail(&tid, &format!("exit {}: {}", output.status, stderr));
                }
            }
            Err(e) => ts.fail(&tid, &format!("spawn error: {}", e)),
        }
    });

    Json(json!({"task_id": task_id}))
}
