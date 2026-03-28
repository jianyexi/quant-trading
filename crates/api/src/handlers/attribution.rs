use axum::{extract::State, http::StatusCode, Json};
use chrono::Datelike;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

use crate::state::AppState;
use crate::task_store::TaskStatus;

// ── Request / Response types ────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AttributionRequest {
    pub task_id: String,
}

#[derive(Debug, Serialize)]
pub struct AttributionResult {
    pub total_return: f64,
    pub timing_contribution: f64,
    pub selection_contribution: f64,
    pub interaction_contribution: f64,
    pub commission_drag: f64,
    pub slippage_drag: f64,
    pub trade_attribution: Vec<TradeAttribution>,
    pub monthly_attribution: Vec<MonthlyAttribution>,
}

#[derive(Debug, Serialize)]
pub struct TradeAttribution {
    pub symbol: String,
    pub entry_date: String,
    pub exit_date: String,
    pub return_pct: f64,
    pub holding_days: i64,
    pub contribution: f64,
}

#[derive(Debug, Serialize)]
pub struct MonthlyAttribution {
    pub month: String,
    pub return_pct: f64,
    pub best_trade: Option<String>,
    pub worst_trade: Option<String>,
    pub trade_count: u32,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Pair raw trades into round-trips (BUY then SELL for same symbol).
struct RoundTrip {
    symbol: String,
    entry_date: String,
    exit_date: String,
    entry_price: f64,
    exit_price: f64,
    quantity: f64,
    commission: f64,
}

fn pair_trades(trades: &[Value]) -> Vec<RoundTrip> {
    // Group trades by symbol, pair BUY→SELL chronologically
    let mut open: HashMap<String, Vec<&Value>> = HashMap::new();
    let mut trips: Vec<RoundTrip> = Vec::new();

    for t in trades {
        let symbol = t.get("symbol").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let side = t.get("side").and_then(|v| v.as_str()).unwrap_or("");
        let side_upper = side.to_uppercase();

        if side_upper == "BUY" || side_upper == "\"BUY\"" || side_upper.contains("BUY") {
            open.entry(symbol).or_default().push(t);
        } else if side_upper == "SELL" || side_upper.contains("SELL") {
            if let Some(buys) = open.get_mut(&symbol) {
                if let Some(buy) = buys.first().cloned() {
                    buys.remove(0);
                    let entry_price = buy.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let exit_price = t.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let qty = buy.get("quantity").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let comm_buy = buy.get("commission").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let comm_sell = t.get("commission").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    let entry_date = buy.get("date").and_then(|v| v.as_str())
                        .or_else(|| buy.get("timestamp").and_then(|v| v.as_str()))
                        .unwrap_or("").to_string();
                    let exit_date = t.get("date").and_then(|v| v.as_str())
                        .or_else(|| t.get("timestamp").and_then(|v| v.as_str()))
                        .unwrap_or("").to_string();

                    trips.push(RoundTrip {
                        symbol,
                        entry_date,
                        exit_date,
                        entry_price,
                        exit_price,
                        quantity: qty,
                        commission: comm_buy + comm_sell,
                    });
                }
            }
        }
    }
    trips
}

/// Extract YYYY-MM from a date-like string.
fn month_key(date: &str) -> String {
    if date.len() >= 7 {
        date[..7].to_string()
    } else {
        "unknown".to_string()
    }
}

/// Parse a date-like string to days since epoch (approximate, for holding-day calc).
fn approx_days(date: &str) -> i64 {
    // Try parsing common formats
    let d = date.split(['T', ' ']).next().unwrap_or(date);
    if let Ok(nd) = chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d") {
        nd.num_days_from_ce() as i64
    } else {
        0
    }
}

/// Compute buy-and-hold return from equity curve between two dates.
fn buy_and_hold_return(equity: &[(String, f64)]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let first = equity.first().unwrap().1;
    let last = equity.last().unwrap().1;
    if first.abs() < 1e-12 {
        return 0.0;
    }
    (last - first) / first
}

// ── Handler ─────────────────────────────────────────────────────────

pub async fn backtest_attribution(
    State(state): State<AppState>,
    Json(req): Json<AttributionRequest>,
) -> (StatusCode, Json<Value>) {
    // 1. Fetch backtest result from task store
    let task = match state.task_store.get(&req.task_id) {
        Some(t) => t,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "Task not found" })),
            );
        }
    };

    if task.status != TaskStatus::Completed {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Backtest task is not completed yet" })),
        );
    }

    let result_json = match &task.result {
        Some(r) => r.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "Backtest task has no result data" })),
            );
        }
    };

    let bt: Value = match serde_json::from_str(&result_json) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to parse backtest result: {}", e) })),
            );
        }
    };

    // 2. Extract equity curve and trades
    let equity_arr = bt.get("equity_curve").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let trades_arr = bt.get("trades").and_then(|v| v.as_array()).cloned().unwrap_or_default();

    let equity_curve: Vec<(String, f64)> = equity_arr
        .iter()
        .filter_map(|pt| {
            let date = pt.get("date").and_then(|v| v.as_str())?.to_string();
            let value = pt.get("value").and_then(|v| v.as_f64())?;
            Some((date, value))
        })
        .collect();

    let initial_capital = bt.get("initial_capital").and_then(|v| v.as_f64())
        .unwrap_or_else(|| equity_curve.first().map(|(_, v)| *v).unwrap_or(1_000_000.0));

    // 3. Pair trades into round-trips
    let round_trips = pair_trades(&trades_arr);

    // 4. Compute per-trade attribution
    let mut trade_attributions: Vec<TradeAttribution> = Vec::new();
    let mut total_commission = 0.0_f64;
    let mut total_trade_pnl = 0.0_f64;

    for rt in &round_trips {
        let ret = if rt.entry_price.abs() > 1e-12 {
            (rt.exit_price - rt.entry_price) / rt.entry_price
        } else {
            0.0
        };
        let pnl = (rt.exit_price - rt.entry_price) * rt.quantity - rt.commission;
        let contribution = pnl / initial_capital;
        let holding_days = {
            let d1 = approx_days(&rt.entry_date);
            let d2 = approx_days(&rt.exit_date);
            if d1 > 0 && d2 > 0 { (d2 - d1).max(1) } else { 1 }
        };

        total_commission += rt.commission;
        total_trade_pnl += pnl;

        trade_attributions.push(TradeAttribution {
            symbol: rt.symbol.clone(),
            entry_date: rt.entry_date.clone(),
            exit_date: rt.exit_date.clone(),
            return_pct: round4(ret * 100.0),
            holding_days,
            contribution: round4(contribution * 100.0),
        });
    }

    // 5. Compute overall return
    let total_return = buy_and_hold_return(&equity_curve);
    let bnh_return = total_return; // For single-stock, buy-and-hold = equity curve return

    // 6. Compute timing vs selection (Brinson-style, single-stock simplification)
    //    selection_contribution = buy-and-hold return of the stock
    //    timing_contribution   = sum of trade returns - buy-and-hold (value from entry/exit timing)
    //    interaction           = remainder
    let sum_trade_return = total_trade_pnl / initial_capital;
    let commission_drag = total_commission / initial_capital;

    // Selection: the stock's buy-and-hold return
    let selection_contribution = bnh_return;
    // Timing: how much the active trading added/subtracted vs buy-and-hold
    let timing_contribution = sum_trade_return - bnh_return;
    // Interaction: residual (total - selection - timing - commission effect)
    let interaction_contribution = total_return - selection_contribution - timing_contribution;

    // 7. Monthly attribution
    let mut monthly_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, ta) in trade_attributions.iter().enumerate() {
        // Use exit_date for monthly grouping
        let mk = month_key(&ta.exit_date);
        monthly_map.entry(mk).or_default().push(i);
    }

    // Also compute monthly returns from equity curve
    let mut monthly_equity: HashMap<String, (f64, f64)> = HashMap::new(); // month -> (first, last)
    for (date, val) in &equity_curve {
        let mk = month_key(date);
        let entry = monthly_equity.entry(mk).or_insert((*val, *val));
        entry.1 = *val; // update last value
    }

    let mut monthly_attribution: Vec<MonthlyAttribution> = Vec::new();
    let mut all_months: Vec<String> = monthly_equity.keys().cloned().collect();
    all_months.sort();

    for m in &all_months {
        let ret_pct = if let Some(&(first, last)) = monthly_equity.get(m) {
            if first.abs() > 1e-12 { (last - first) / first * 100.0 } else { 0.0 }
        } else {
            0.0
        };

        let trade_indices = monthly_map.get(m).cloned().unwrap_or_default();
        let trade_count = trade_indices.len() as u32;

        let best_trade = trade_indices.iter()
            .max_by(|&&a, &&b| {
                trade_attributions[a].contribution
                    .partial_cmp(&trade_attributions[b].contribution)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|&i| {
                let ta = &trade_attributions[i];
                format!("{} ({:+.2}%)", ta.symbol, ta.contribution)
            });

        let worst_trade = trade_indices.iter()
            .min_by(|&&a, &&b| {
                trade_attributions[a].contribution
                    .partial_cmp(&trade_attributions[b].contribution)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|&i| {
                let ta = &trade_attributions[i];
                format!("{} ({:+.2}%)", ta.symbol, ta.contribution)
            });

        monthly_attribution.push(MonthlyAttribution {
            month: m.clone(),
            return_pct: round4(ret_pct),
            best_trade,
            worst_trade,
            trade_count,
        });
    }

    // 8. Build response
    let result = AttributionResult {
        total_return: round4(total_return * 100.0),
        timing_contribution: round4(timing_contribution * 100.0),
        selection_contribution: round4(selection_contribution * 100.0),
        interaction_contribution: round4(interaction_contribution * 100.0),
        commission_drag: round4(commission_drag * 100.0),
        slippage_drag: 0.0, // Slippage not tracked separately in current backtest engine
        trade_attribution: trade_attributions,
        monthly_attribution,
    };

    match serde_json::to_value(&result) {
        Ok(val) => (StatusCode::OK, Json(val)),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Serialization error: {}", e) })),
        ),
    }
}
