use axum::{extract::State, http::StatusCode, Json};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::state::AppState;
use crate::task_store::TaskStatus;

// ── Request / Response types ────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MonteCarloRequest {
    pub task_id: String,
    pub num_simulations: Option<usize>,
    pub num_days: Option<usize>,
}

#[derive(Debug, Serialize)]
struct Distribution {
    percentile_5: f64,
    percentile_25: f64,
    percentile_50: f64,
    percentile_75: f64,
    percentile_95: f64,
    mean: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    std: Option<f64>,
}

#[derive(Debug, Serialize)]
struct PercentilePath {
    percentile: u32,
    equity: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct MonteCarloResult {
    simulations: usize,
    trading_days: usize,
    return_distribution: Distribution,
    drawdown_distribution: Distribution,
    paths: Vec<PercentilePath>,
    probability_of_loss: f64,
    probability_of_ruin: f64,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p / 100.0 * (sorted.len() as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi || hi >= sorted.len() {
        sorted[lo.min(sorted.len() - 1)]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = equity[0];
    let mut worst = 0.0_f64;
    for &v in equity.iter().skip(1) {
        if v > peak {
            peak = v;
        }
        let dd = (v - peak) / peak;
        if dd < worst {
            worst = dd;
        }
    }
    worst
}

fn sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    // Annualised Sharpe (assume 252 trading days, risk-free = 0)
    (mean / std) * 252.0_f64.sqrt()
}

// ── Handler ─────────────────────────────────────────────────────────

pub async fn run_monte_carlo(
    State(state): State<AppState>,
    Json(req): Json<MonteCarloRequest>,
) -> (StatusCode, Json<Value>) {
    let num_sims = req.num_simulations.unwrap_or(1000).min(10_000);
    let num_days = req.num_days.unwrap_or(252).min(1000);

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

    // 2. Parse equity curve
    let bt: Value = match serde_json::from_str(&result_json) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to parse backtest result: {}", e) })),
            );
        }
    };

    let equity_curve = match bt.get("equity_curve").and_then(|v| v.as_array()) {
        Some(arr) => arr.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "Backtest result has no equity_curve" })),
            );
        }
    };

    if equity_curve.len() < 2 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Equity curve too short for simulation" })),
        );
    }

    // Extract values and compute daily returns
    let values: Vec<f64> = equity_curve
        .iter()
        .filter_map(|pt| pt.get("value").and_then(|v| v.as_f64()))
        .collect();

    if values.len() < 2 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Not enough equity data points" })),
        );
    }

    let daily_returns: Vec<f64> = values
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    if daily_returns.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Could not compute daily returns" })),
        );
    }

    let ts = Arc::clone(&state.task_store);
    let task_id = ts.create("monte_carlo");
    let tid = task_id.clone();

    // 3. Run simulation in blocking thread
    tokio::task::spawn_blocking(move || {
        ts.set_progress(&tid, "🎲 Running Monte Carlo simulation...");

        let n_returns = daily_returns.len();
        let mut rng = rand::thread_rng();

        // Store per-simulation results
        let mut final_returns: Vec<f64> = Vec::with_capacity(num_sims);
        let mut max_drawdowns: Vec<f64> = Vec::with_capacity(num_sims);
        let mut sharpe_ratios: Vec<f64> = Vec::with_capacity(num_sims);
        let mut all_paths: Vec<Vec<f64>> = Vec::with_capacity(num_sims);

        for _ in 0..num_sims {
            // Bootstrap: sample daily returns with replacement
            let mut equity = Vec::with_capacity(num_days + 1);
            let mut sim_returns = Vec::with_capacity(num_days);
            equity.push(1.0_f64);

            for _ in 0..num_days {
                let idx = rng.gen_range(0..n_returns);
                let r = daily_returns[idx];
                sim_returns.push(r);
                let prev = *equity.last().unwrap();
                equity.push(prev * (1.0 + r));
            }

            let final_ret = equity.last().unwrap() / equity[0] - 1.0;
            let dd = max_drawdown(&equity);
            let sr = sharpe_ratio(&sim_returns);

            final_returns.push(final_ret);
            max_drawdowns.push(dd);
            sharpe_ratios.push(sr);
            all_paths.push(equity);
        }

        // Sort for percentile calculations
        let mut sorted_returns = final_returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut sorted_dd = max_drawdowns.clone();
        sorted_dd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_ret = sorted_returns.iter().sum::<f64>() / sorted_returns.len() as f64;
        let var_ret = sorted_returns
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / sorted_returns.len() as f64;
        let std_ret = var_ret.sqrt();

        let mean_dd = sorted_dd.iter().sum::<f64>() / sorted_dd.len() as f64;

        let return_dist = Distribution {
            percentile_5: round4(percentile(&sorted_returns, 5.0)),
            percentile_25: round4(percentile(&sorted_returns, 25.0)),
            percentile_50: round4(percentile(&sorted_returns, 50.0)),
            percentile_75: round4(percentile(&sorted_returns, 75.0)),
            percentile_95: round4(percentile(&sorted_returns, 95.0)),
            mean: round4(mean_ret),
            std: Some(round4(std_ret)),
        };

        let drawdown_dist = Distribution {
            percentile_5: round4(percentile(&sorted_dd, 5.0)),
            percentile_25: round4(percentile(&sorted_dd, 25.0)),
            percentile_50: round4(percentile(&sorted_dd, 50.0)),
            percentile_75: round4(percentile(&sorted_dd, 75.0)),
            percentile_95: round4(percentile(&sorted_dd, 95.0)),
            mean: round4(mean_dd),
            std: None,
        };

        // Pick representative paths closest to each percentile
        let target_percentiles = [5u32, 25, 50, 75, 95];
        let target_returns: Vec<f64> = target_percentiles
            .iter()
            .map(|&p| percentile(&sorted_returns, p as f64))
            .collect();

        let paths: Vec<PercentilePath> = target_percentiles
            .iter()
            .zip(target_returns.iter())
            .map(|(&p, &target)| {
                // Find the simulation whose final return is closest to this percentile
                let best_idx = final_returns
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = (*a - target).abs();
                        let db = (*b - target).abs();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                // Downsample path to at most 100 points for JSON efficiency
                let full = &all_paths[best_idx];
                let equity = downsample(full, 100);
                PercentilePath { percentile: p, equity }
            })
            .collect();

        let prob_loss = final_returns.iter().filter(|&&r| r < 0.0).count() as f64
            / final_returns.len() as f64;
        let prob_ruin = max_drawdowns.iter().filter(|&&d| d < -0.5).count() as f64
            / max_drawdowns.len() as f64;

        let mc = MonteCarloResult {
            simulations: num_sims,
            trading_days: num_days,
            return_distribution: return_dist,
            drawdown_distribution: drawdown_dist,
            paths,
            probability_of_loss: round4(prob_loss),
            probability_of_ruin: round4(prob_ruin),
        };

        match serde_json::to_string(&mc) {
            Ok(json_str) => ts.complete(&tid, &json_str),
            Err(e) => ts.fail(&tid, &format!("Serialization error: {}", e)),
        }
    });

    (
        StatusCode::OK,
        Json(json!({
            "task_id": task_id,
            "status": "Running",
            "progress": "Monte Carlo simulation started..."
        })),
    )
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Downsample a path to at most `max_points` evenly spaced points,
/// always keeping the first and last.
fn downsample(data: &[f64], max_points: usize) -> Vec<f64> {
    let n = data.len();
    if n <= max_points {
        return data.iter().map(|v| round4(*v)).collect();
    }
    let mut result = Vec::with_capacity(max_points);
    for i in 0..max_points {
        let idx = (i as f64 * (n - 1) as f64 / (max_points - 1) as f64).round() as usize;
        result.push(round4(data[idx]));
    }
    result
}
