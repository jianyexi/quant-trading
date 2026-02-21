use axum::{extract::State, Json};
use serde_json::{json, Value};
use std::collections::HashMap;

use crate::state::AppState;

/// GET /api/reports — comprehensive statistical report from journal + engine metrics
pub async fn get_reports(State(state): State<AppState>) -> Json<Value> {
    // 1. Journal stats
    let journal_stats = state.journal.stats();
    let stats_map: HashMap<String, u64> = journal_stats.into_iter().collect();

    // 2. All journal entries (for per-symbol analysis)
    let all_entries = state.journal.query(
        &quant_broker::journal::JournalQuery { limit: Some(10000), ..Default::default() },
    );

    // 3. Daily snapshots
    let snapshots = state.journal.get_daily_snapshots(365);

    // 4. Engine live status
    let engine_guard = state.engine.lock().await;
    let (engine_running, engine_perf, engine_latency) = if let Some(ref eng) = *engine_guard {
        let s = eng.status().await;
        (true, Some(s.performance), Some(s.latency))
    } else {
        (false, None, None)
    };
    drop(engine_guard);

    // ── Trading Summary ──
    let total_signals = *stats_map.get("signal").unwrap_or(&0);
    let total_submitted = *stats_map.get("order_submitted").unwrap_or(&0);
    let total_filled = *stats_map.get("order_filled").unwrap_or(&0);
    let total_rejected = *stats_map.get("risk_rejected").unwrap_or(&0)
        + *stats_map.get("order_rejected").unwrap_or(&0);
    let total_events: u64 = stats_map.values().sum();

    // ── Per-Symbol Breakdown ──
    let mut sym_stats: HashMap<String, SymbolStats> = HashMap::new();
    for entry in &all_entries {
        if entry.symbol.is_empty() || entry.symbol == "-" {
            continue;
        }
        let ss = sym_stats.entry(entry.symbol.clone()).or_default();

        match entry.entry_type.as_str() {
            "signal" => ss.signals += 1,
            "order_submitted" => ss.orders += 1,
            "order_filled" => {
                ss.fills += 1;
                if let Some(pnl) = entry.pnl {
                    ss.total_pnl += pnl;
                    if pnl > 0.0 {
                        ss.wins += 1;
                        ss.gross_profit += pnl;
                    } else if pnl < 0.0 {
                        ss.losses += 1;
                        ss.gross_loss += pnl.abs();
                    }
                }
                if let Some(qty) = entry.quantity {
                    ss.total_volume += qty;
                }
                if let Some(price) = entry.price {
                    ss.last_price = price;
                }
                if let Some(ref side) = entry.side {
                    match side.as_str() {
                        "Buy" | "BUY" => ss.buy_count += 1,
                        "Sell" | "SELL" => ss.sell_count += 1,
                        _ => {}
                    }
                }
            }
            "risk_rejected" | "order_rejected" => {
                ss.rejected += 1;
                if let Some(ref reason) = entry.reason {
                    *ss.reject_reasons.entry(reason.clone()).or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }

    let symbol_breakdown: Vec<Value> = {
        let mut syms: Vec<_> = sym_stats.iter().collect();
        syms.sort_by(|a, b| b.1.total_pnl.partial_cmp(&a.1.total_pnl).unwrap_or(std::cmp::Ordering::Equal));
        syms.iter().map(|(sym, ss)| {
            let total_trades = ss.wins + ss.losses;
            json!({
                "symbol": sym,
                "signals": ss.signals,
                "orders": ss.orders,
                "fills": ss.fills,
                "rejected": ss.rejected,
                "buy_count": ss.buy_count,
                "sell_count": ss.sell_count,
                "total_pnl": round2(ss.total_pnl),
                "gross_profit": round2(ss.gross_profit),
                "gross_loss": round2(ss.gross_loss),
                "wins": ss.wins,
                "losses": ss.losses,
                "win_rate": if total_trades > 0 { round2(ss.wins as f64 / total_trades as f64 * 100.0) } else { 0.0 },
                "profit_factor": if ss.gross_loss > 0.0 { round2(ss.gross_profit / ss.gross_loss) } else if ss.gross_profit > 0.0 { f64::INFINITY } else { 0.0 },
                "total_volume": round0(ss.total_volume),
                "last_price": round2(ss.last_price),
                "top_reject_reasons": ss.reject_reasons.iter()
                    .take(3)
                    .map(|(r, c)| json!({"reason": r, "count": c}))
                    .collect::<Vec<_>>(),
            })
        }).collect()
    };

    // ── Daily PnL Series ──
    let daily_pnl: Vec<Value> = snapshots.iter().map(|s| {
        json!({
            "date": s.date,
            "portfolio_value": round2(s.portfolio_value),
            "cash": round2(s.cash),
            "positions": s.positions_count,
            "daily_pnl": round2(s.daily_pnl),
            "cumulative_pnl": round2(s.cumulative_pnl),
            "total_trades": s.total_trades,
        })
    }).collect();

    // ── Risk Events ──
    let risk_entries: Vec<Value> = all_entries.iter()
        .filter(|e| e.entry_type == "risk_rejected" || e.entry_type == "order_rejected")
        .take(100)
        .map(|e| json!({
            "timestamp": e.timestamp,
            "symbol": e.symbol,
            "side": e.side,
            "quantity": e.quantity,
            "price": e.price,
            "reason": e.reason,
        }))
        .collect();

    // ── Reject Reason Summary ──
    let mut reason_counts: HashMap<String, u64> = HashMap::new();
    for entry in &all_entries {
        if entry.entry_type == "risk_rejected" || entry.entry_type == "order_rejected" {
            if let Some(ref reason) = entry.reason {
                let key = reason.split(':').next().unwrap_or(reason).trim().to_string();
                *reason_counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    let mut reject_summary: Vec<_> = reason_counts.into_iter().collect();
    reject_summary.sort_by(|a, b| b.1.cmp(&a.1));

    // ── Order Flow Analysis ──
    let filled_entries: Vec<&quant_broker::journal::JournalEntry> = all_entries.iter()
        .filter(|e| e.entry_type == "order_filled")
        .collect();
    let (mut total_pnl, mut total_profit, mut total_loss) = (0.0f64, 0.0f64, 0.0f64);
    let (mut win_count, mut loss_count) = (0u64, 0u64);
    let mut pnl_values: Vec<f64> = Vec::new();

    for e in &filled_entries {
        if let Some(pnl) = e.pnl {
            total_pnl += pnl;
            pnl_values.push(pnl);
            if pnl > 0.0 {
                total_profit += pnl;
                win_count += 1;
            } else if pnl < 0.0 {
                total_loss += pnl.abs();
                loss_count += 1;
            }
        }
    }

    let total_round_trips = win_count + loss_count;
    let avg_pnl = if total_round_trips > 0 { total_pnl / total_round_trips as f64 } else { 0.0 };
    let avg_win = if win_count > 0 { total_profit / win_count as f64 } else { 0.0 };
    let avg_loss = if loss_count > 0 { total_loss / loss_count as f64 } else { 0.0 };
    let expectancy = if total_round_trips > 0 {
        (win_count as f64 / total_round_trips as f64) * avg_win
            - (loss_count as f64 / total_round_trips as f64) * avg_loss
    } else {
        0.0
    };

    // Max consecutive wins/losses
    let (mut max_consec_win, mut max_consec_loss) = (0u32, 0u32);
    let (mut cur_win, mut cur_loss) = (0u32, 0u32);
    for pnl in &pnl_values {
        if *pnl > 0.0 {
            cur_win += 1;
            cur_loss = 0;
            max_consec_win = max_consec_win.max(cur_win);
        } else if *pnl < 0.0 {
            cur_loss += 1;
            cur_win = 0;
            max_consec_loss = max_consec_loss.max(cur_loss);
        }
    }

    // Max/min single trade PnL
    let max_single_win = pnl_values.iter().cloned().fold(0.0f64, f64::max);
    let max_single_loss = pnl_values.iter().cloned().fold(0.0f64, f64::min);

    // ── Hourly Distribution ──
    let mut hourly_signals: [u64; 24] = [0; 24];
    let mut hourly_fills: [u64; 24] = [0; 24];
    for entry in &all_entries {
        // Parse hour from timestamp "YYYY-MM-DD HH:MM:SS" or ISO format
        let hour = entry.timestamp.get(11..13)
            .and_then(|h| h.parse::<usize>().ok())
            .unwrap_or(0);
        if hour < 24 {
            match entry.entry_type.as_str() {
                "signal" => hourly_signals[hour] += 1,
                "order_filled" => hourly_fills[hour] += 1,
                _ => {}
            }
        }
    }

    let hourly: Vec<Value> = (0..24).map(|h| {
        json!({ "hour": h, "signals": hourly_signals[h], "fills": hourly_fills[h] })
    }).collect();

    // ── Performance card (live engine or computed from journal) ──
    let performance = if let Some(ref perf) = engine_perf {
        json!({
            "source": "live_engine",
            "portfolio_value": round2(perf.portfolio_value),
            "initial_capital": round2(perf.initial_capital),
            "total_return_pct": round2(perf.total_return_pct),
            "max_drawdown_pct": round2(perf.max_drawdown_pct),
            "win_rate": round2(perf.win_rate),
            "profit_factor": round2(perf.profit_factor),
            "wins": perf.wins,
            "losses": perf.losses,
        })
    } else {
        json!({
            "source": "journal",
            "total_pnl": round2(total_pnl),
            "gross_profit": round2(total_profit),
            "gross_loss": round2(total_loss),
            "win_rate": if total_round_trips > 0 { round2(win_count as f64 / total_round_trips as f64 * 100.0) } else { 0.0 },
            "profit_factor": if total_loss > 0.0 { round2(total_profit / total_loss) } else { 0.0 },
            "wins": win_count,
            "losses": loss_count,
        })
    };

    let latency = engine_latency.map(|l| json!({
        "avg_factor_us": l.avg_factor_compute_us,
        "last_factor_us": l.last_factor_compute_us,
        "last_risk_us": l.last_risk_check_us,
        "last_order_us": l.last_order_submit_us,
        "total_bars": l.total_bars_processed,
    }));

    Json(json!({
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "engine_running": engine_running,
        "summary": {
            "total_events": total_events,
            "total_signals": total_signals,
            "total_submitted": total_submitted,
            "total_filled": total_filled,
            "total_rejected": total_rejected,
            "fill_rate_pct": if total_submitted > 0 { round2(total_filled as f64 / total_submitted as f64 * 100.0) } else { 0.0 },
            "event_breakdown": stats_map,
        },
        "performance": performance,
        "order_analysis": {
            "total_round_trips": total_round_trips,
            "total_pnl": round2(total_pnl),
            "avg_pnl": round2(avg_pnl),
            "avg_win": round2(avg_win),
            "avg_loss": round2(avg_loss),
            "expectancy": round2(expectancy),
            "max_consecutive_wins": max_consec_win,
            "max_consecutive_losses": max_consec_loss,
            "best_trade": round2(max_single_win),
            "worst_trade": round2(max_single_loss),
        },
        "symbols": symbol_breakdown,
        "daily_pnl": daily_pnl,
        "risk_events": risk_entries,
        "reject_summary": reject_summary.iter().map(|(r, c)| json!({"reason": r, "count": c})).collect::<Vec<_>>(),
        "hourly_distribution": hourly,
        "latency": latency,
    }))
}

// ── Helpers ──

#[derive(Default)]
struct SymbolStats {
    signals: u64,
    orders: u64,
    fills: u64,
    rejected: u64,
    buy_count: u64,
    sell_count: u64,
    wins: u64,
    losses: u64,
    total_pnl: f64,
    gross_profit: f64,
    gross_loss: f64,
    total_volume: f64,
    last_price: f64,
    reject_reasons: HashMap<String, u64>,
}

fn round2(v: f64) -> f64 { (v * 100.0).round() / 100.0 }
fn round0(v: f64) -> f64 { v.round() }
