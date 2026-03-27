use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::state::AppState;

// ── VaR / CVaR ──────────────────────────────────────────────────────

/// GET /api/risk/var — Portfolio VaR, CVaR, volatility, correlation, histogram
pub async fn risk_var(State(state): State<AppState>) -> (StatusCode, Json<Value>) {
    let engine_guard = state.engine.lock().await;

    // Collect held symbols + weights from live engine positions
    let (symbols, weights, _portfolio_value) = if let Some(ref eng) = *engine_guard {
        let positions = eng.broker().get_positions().await.unwrap_or_default();
        if positions.is_empty() {
            return fallback_var(&state).await;
        }
        let total: f64 = positions
            .iter()
            .map(|p| p.current_price * p.quantity)
            .sum();
        if total <= 0.0 {
            return fallback_var(&state).await;
        }
        let syms: Vec<String> = positions.iter().map(|p| p.symbol.clone()).collect();
        let wts: Vec<f64> = positions
            .iter()
            .map(|p| p.current_price * p.quantity / total)
            .collect();
        let pv = eng.status().await.performance.portfolio_value;
        (syms, wts, pv)
    } else {
        return fallback_var(&state).await;
    };

    // Drop engine lock before potentially long DB queries
    drop(engine_guard);

    let lookback = 252i32;

    // Fetch per-symbol daily close prices from kline_daily (PostgreSQL)
    let pool = match &state.db {
        Some(p) => p,
        None => {
            return (
                StatusCode::OK,
                Json(json!({
                    "error": "database_unavailable",
                    "message": "PostgreSQL连接不可用，无法计算VaR"
                })),
            );
        }
    };

    let mut all_returns: Vec<Vec<f64>> = Vec::new();
    let mut valid_symbols: Vec<String> = Vec::new();
    let mut valid_weights: Vec<f64> = Vec::new();

    for (i, sym) in symbols.iter().enumerate() {
        let rows: Vec<(f64,)> = sqlx::query_as(
            "SELECT close FROM kline_daily \
             WHERE symbol = $1 \
             ORDER BY datetime DESC \
             LIMIT $2",
        )
        .bind(sym)
        .bind(lookback + 1)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        if rows.len() < 2 {
            continue;
        }

        // rows are newest-first; reverse for chronological order
        let closes: Vec<f64> = rows.iter().rev().map(|r| r.0).collect();
        let returns: Vec<f64> = closes
            .windows(2)
            .map(|w| if w[0] > 0.0 { w[1] / w[0] - 1.0 } else { 0.0 })
            .collect();

        all_returns.push(returns);
        valid_symbols.push(sym.clone());
        valid_weights.push(weights[i]);
    }

    // If no per-symbol data, try portfolio-level daily_snapshots
    if all_returns.is_empty() {
        return compute_var_from_snapshots(pool, &symbols).await;
    }

    // Normalise weights so they sum to 1
    let w_sum: f64 = valid_weights.iter().sum();
    if w_sum > 0.0 {
        for w in &mut valid_weights {
            *w /= w_sum;
        }
    }

    // Determine common length (shortest series)
    let n = all_returns.iter().map(|r| r.len()).min().unwrap_or(0);
    if n < 5 {
        return compute_var_from_snapshots(pool, &symbols).await;
    }

    // Weighted portfolio daily returns
    let mut port_returns: Vec<f64> = vec![0.0; n];
    for (idx, rets) in all_returns.iter().enumerate() {
        let offset = rets.len() - n;
        for j in 0..n {
            port_returns[j] += valid_weights[idx] * rets[offset + j];
        }
    }

    // ── VaR / CVaR ──
    let mut sorted = port_returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let var95 = percentile(&sorted, 0.05);
    let var99 = percentile(&sorted, 0.01);
    let cvar95 = cvar_below(&sorted, var95);
    let cvar99 = cvar_below(&sorted, var99);

    // ── Portfolio volatility (annualised) ──
    let mean = port_returns.iter().sum::<f64>() / n as f64;
    let variance =
        port_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let daily_vol = variance.sqrt();
    let annual_vol = daily_vol * (252.0_f64).sqrt();

    // ── Correlation matrix ──
    let corr = correlation_matrix(&all_returns, n);

    // ── Histogram (50 bins) ──
    let histogram = build_histogram(&port_returns, 50);

    (
        StatusCode::OK,
        Json(json!({
            "var_95": round4(var95),
            "var_99": round4(var99),
            "cvar_95": round4(cvar95),
            "cvar_99": round4(cvar99),
            "portfolio_volatility": round4(annual_vol),
            "lookback_days": n,
            "positions": valid_symbols,
            "correlation_matrix": {
                "symbols": valid_symbols,
                "matrix": corr,
            },
            "daily_returns_histogram": histogram,
        })),
    )
}

/// Fallback: compute VaR from portfolio-level daily_snapshots when no
/// positions are available (e.g. engine not running).
async fn fallback_var(state: &AppState) -> (StatusCode, Json<Value>) {
    let pool = match &state.db {
        Some(p) => p,
        None => {
            return (
                StatusCode::OK,
                Json(json!({
                    "error": "no_portfolio",
                    "message": "无持仓且数据库不可用，请先运行回测或启动交易"
                })),
            );
        }
    };

    compute_var_from_snapshots(pool, &[]).await
}

async fn compute_var_from_snapshots(
    pool: &sqlx::PgPool,
    symbols: &[String],
) -> (StatusCode, Json<Value>) {
    let rows: Vec<(f64,)> = sqlx::query_as(
        "SELECT portfolio_value FROM daily_snapshots ORDER BY date DESC LIMIT 253",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    if rows.len() < 2 {
        return (
            StatusCode::OK,
            Json(json!({
                "error": "insufficient_data",
                "message": "历史数据不足，请先运行回测或等待更多交易日数据",
                "available_days": rows.len(),
            })),
        );
    }

    let values: Vec<f64> = rows.iter().rev().map(|r| r.0).collect();
    let returns: Vec<f64> = values
        .windows(2)
        .map(|w| if w[0] > 0.0 { w[1] / w[0] - 1.0 } else { 0.0 })
        .collect();
    let n = returns.len();

    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let var95 = percentile(&sorted, 0.05);
    let var99 = percentile(&sorted, 0.01);
    let cvar95 = cvar_below(&sorted, var95);
    let cvar99 = cvar_below(&sorted, var99);

    let mean = returns.iter().sum::<f64>() / n as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let annual_vol = variance.sqrt() * (252.0_f64).sqrt();

    let histogram = build_histogram(&returns, 50);

    (
        StatusCode::OK,
        Json(json!({
            "var_95": round4(var95),
            "var_99": round4(var99),
            "cvar_95": round4(cvar95),
            "cvar_99": round4(cvar99),
            "portfolio_volatility": round4(annual_vol),
            "lookback_days": n,
            "positions": symbols,
            "correlation_matrix": null,
            "daily_returns_histogram": histogram,
            "source": "daily_snapshots",
        })),
    )
}

// ── Stress Testing ──────────────────────────────────────────────────

struct CrisisScenario {
    name: &'static str,
    description: &'static str,
    market_drawdown: f64,
    duration_days: u32,
    volatility_multiplier: f64,
    correlation_spike: f64,
}

const SCENARIOS: &[CrisisScenario] = &[
    CrisisScenario {
        name: "2008_financial",
        description: "2008全球金融危机",
        market_drawdown: -0.55,
        duration_days: 252,
        volatility_multiplier: 3.0,
        correlation_spike: 0.9,
    },
    CrisisScenario {
        name: "2015_china",
        description: "2015年A股股灾",
        market_drawdown: -0.45,
        duration_days: 60,
        volatility_multiplier: 4.0,
        correlation_spike: 0.95,
    },
    CrisisScenario {
        name: "2020_covid",
        description: "2020新冠疫情冲击",
        market_drawdown: -0.34,
        duration_days: 23,
        volatility_multiplier: 5.0,
        correlation_spike: 0.85,
    },
    CrisisScenario {
        name: "flash_crash",
        description: "闪崩场景",
        market_drawdown: -0.10,
        duration_days: 1,
        volatility_multiplier: 10.0,
        correlation_spike: 1.0,
    },
];

#[derive(Debug, Deserialize)]
pub struct StressTestRequest {
    pub scenarios: Option<Vec<String>>,
    pub custom_drawdown: Option<f64>,
}

/// POST /api/risk/stress-test — apply historical crisis scenarios
pub async fn risk_stress_test(
    State(state): State<AppState>,
    Json(req): Json<StressTestRequest>,
) -> (StatusCode, Json<Value>) {
    let engine_guard = state.engine.lock().await;

    let (portfolio_value, portfolio_beta) = if let Some(ref eng) = *engine_guard {
        let perf = eng.status().await.performance;
        // Estimate beta from drawdown behaviour: rough proxy
        let beta = if perf.max_drawdown_pct > 0.0 && perf.portfolio_value > 0.0 {
            (perf.max_drawdown_pct / 0.20).min(2.0).max(0.5) // normalise vs ~20% market drawdown
        } else {
            1.0
        };
        (perf.portfolio_value, beta)
    } else {
        // No engine — use a hypothetical ¥1,000,000 portfolio
        (1_000_000.0, 1.0)
    };
    drop(engine_guard);

    let requested: Vec<String> = req
        .scenarios
        .unwrap_or_else(|| SCENARIOS.iter().map(|s| s.name.to_string()).collect());

    let mut results: Vec<Value> = Vec::new();

    for scenario_name in &requested {
        if scenario_name == "custom" {
            let drawdown = req.custom_drawdown.unwrap_or(-0.20);
            let drawdown_clamped = drawdown.min(-0.01).max(-0.90);
            let impact = portfolio_beta * drawdown_clamped;
            let loss = portfolio_value * impact.abs();
            let risk_level = classify_risk(impact);
            results.push(json!({
                "name": "自定义场景",
                "description": format!("自定义市场下跌 {:.0}%", drawdown_clamped * 100.0),
                "market_drawdown": format!("{:.0}%", drawdown_clamped * 100.0),
                "estimated_portfolio_impact": format!("{:.1}%", impact * 100.0),
                "estimated_loss": format!("¥{}", format_number(loss)),
                "estimated_loss_value": round2(loss),
                "impact_pct": round4(impact),
                "estimated_duration": "自定义",
                "risk_level": risk_level,
            }));
            continue;
        }

        if let Some(sc) = SCENARIOS.iter().find(|s| s.name == scenario_name.as_str()) {
            // Correlation spike amplifies loss: during crises diversification fails
            let corr_effect = (sc.correlation_spike - 0.5).max(0.0) * 0.2;
            let impact = portfolio_beta * sc.market_drawdown * (1.0 + corr_effect);
            let loss = portfolio_value * impact.abs();
            let recovery = (sc.duration_days as f64 * 1.5) as u32;
            let risk_level = classify_risk(impact);

            results.push(json!({
                "name": sc.description,
                "description": sc.description,
                "market_drawdown": format!("{:.0}%", sc.market_drawdown * 100.0),
                "estimated_portfolio_impact": format!("{:.1}%", impact * 100.0),
                "estimated_loss": format!("¥{}", format_number(loss)),
                "estimated_loss_value": round2(loss),
                "impact_pct": round4(impact),
                "estimated_duration": format!("{} 交易日", recovery),
                "volatility_multiplier": sc.volatility_multiplier,
                "correlation_spike": sc.correlation_spike,
                "risk_level": risk_level,
            }));
        }
    }

    (
        StatusCode::OK,
        Json(json!({
            "portfolio_value": round2(portfolio_value),
            "portfolio_beta": round4(portfolio_beta),
            "scenarios": results,
        })),
    )
}

// ── Helpers ──────────────────────────────────────────────────────────

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * sorted.len() as f64).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn cvar_below(sorted: &[f64], threshold: f64) -> f64 {
    let below: Vec<f64> = sorted.iter().copied().filter(|&r| r <= threshold).collect();
    if below.is_empty() {
        threshold
    } else {
        below.iter().sum::<f64>() / below.len() as f64
    }
}

fn correlation_matrix(all_returns: &[Vec<f64>], common_len: usize) -> Vec<Vec<f64>> {
    let k = all_returns.len();
    let mut corr = vec![vec![0.0; k]; k];

    // Pre-compute means and stddevs for the common tail of each series
    let mut means = vec![0.0; k];
    let mut stds = vec![0.0; k];
    for i in 0..k {
        let offset = all_returns[i].len() - common_len;
        let slice = &all_returns[i][offset..];
        let m = slice.iter().sum::<f64>() / common_len as f64;
        means[i] = m;
        let var = slice.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (common_len as f64 - 1.0);
        stds[i] = var.sqrt();
    }

    for i in 0..k {
        for j in 0..k {
            if i == j {
                corr[i][j] = 1.0;
            } else if stds[i] > 0.0 && stds[j] > 0.0 {
                let oi = all_returns[i].len() - common_len;
                let oj = all_returns[j].len() - common_len;
                let cov: f64 = (0..common_len)
                    .map(|t| {
                        (all_returns[i][oi + t] - means[i]) * (all_returns[j][oj + t] - means[j])
                    })
                    .sum::<f64>()
                    / (common_len as f64 - 1.0);
                corr[i][j] = round4(cov / (stds[i] * stds[j]));
            }
        }
    }
    corr
}

fn build_histogram(returns: &[f64], bins: usize) -> Vec<Value> {
    if returns.is_empty() {
        return vec![];
    }
    let min_r = returns
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_r = returns
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max_r - min_r;
    if range <= 0.0 {
        return vec![json!({"bin": round4(min_r), "count": returns.len()})];
    }
    let step = range / bins as f64;
    let mut counts = vec![0usize; bins];
    for &r in returns {
        let idx = ((r - min_r) / step).floor() as usize;
        counts[idx.min(bins - 1)] += 1;
    }
    counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            json!({
                "bin": round4(min_r + step * i as f64 + step / 2.0),
                "count": c,
            })
        })
        .collect()
}

fn classify_risk(impact: f64) -> &'static str {
    let abs = impact.abs();
    if abs >= 0.40 {
        "extreme"
    } else if abs >= 0.20 {
        "severe"
    } else if abs >= 0.10 {
        "moderate"
    } else {
        "mild"
    }
}

fn format_number(v: f64) -> String {
    let abs = v.abs();
    if abs >= 1_0000.0 {
        format!("{:.1}万", v / 1_0000.0)
    } else {
        format!("{:.0}", v)
    }
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
