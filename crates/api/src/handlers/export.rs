use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::Value;

use crate::state::AppState;

// ── Request Types ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ExportTaskRequest {
    pub task_id: String,
}

// ── Helpers ────────────────────────────────────────────────────────

fn csv_response(filename: &str, csv: String) -> impl IntoResponse {
    (
        StatusCode::OK,
        [
            ("content-type", "text/csv; charset=utf-8".to_string()),
            (
                "content-disposition",
                format!("attachment; filename=\"{}\"", filename),
            ),
        ],
        csv,
    )
}

fn val_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        _ => v.to_string(),
    }
}

fn error_response(status: StatusCode, msg: &str) -> impl IntoResponse {
    (status, msg.to_string()).into_response()
}

// ── Export Backtest Equity Curve + Trades ───────────────────────────

pub async fn export_backtest_csv(
    State(state): State<AppState>,
    Json(req): Json<ExportTaskRequest>,
) -> impl IntoResponse {
    let task = match state.task_store.get(&req.task_id) {
        Some(t) => t,
        None => return error_response(StatusCode::NOT_FOUND, "Task not found").into_response(),
    };

    let result_json: Value = match &task.result {
        Some(r) => match serde_json::from_str(r) {
            Ok(v) => v,
            Err(_) => {
                return error_response(StatusCode::BAD_REQUEST, "Invalid task result")
                    .into_response()
            }
        },
        None => {
            return error_response(StatusCode::BAD_REQUEST, "Task has no result").into_response()
        }
    };

    // Build equity curve CSV
    let mut csv = String::from("date,portfolio_value,benchmark_value\n");

    if let Some(curve) = result_json.get("equity_curve").and_then(|v| v.as_array()) {
        let bench = result_json
            .get("benchmark_curve")
            .and_then(|v| v.as_array());

        for (i, pt) in curve.iter().enumerate() {
            let date = pt
                .get("date")
                .map(val_str)
                .unwrap_or_default();
            let value = pt
                .get("value")
                .map(val_str)
                .unwrap_or_default();
            let bench_val = bench
                .and_then(|b| b.get(i))
                .and_then(|bp| bp.get("value"))
                .map(val_str)
                .unwrap_or_default();
            csv.push_str(&format!("{},{},{}\n", date, value, bench_val));
        }
    }

    // Append trades section
    csv.push_str("\n\ndate,symbol,side,price,quantity,commission,pnl\n");
    if let Some(trades) = result_json.get("trades").and_then(|v| v.as_array()) {
        for t in trades {
            let date = t.get("date").map(val_str).unwrap_or_default();
            let symbol = t.get("symbol").map(val_str).unwrap_or_default();
            let side = t.get("side").map(val_str).unwrap_or_default();
            let price = t.get("price").map(val_str).unwrap_or_default();
            let qty = t.get("quantity").map(val_str).unwrap_or_default();
            let comm = t.get("commission").map(val_str).unwrap_or_default();
            let pnl = t.get("pnl").map(val_str).unwrap_or_default();
            csv.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                date, symbol, side, price, qty, comm, pnl
            ));
        }
    }

    csv_response("backtest.csv", csv).into_response()
}

// ── Export Trade Journal ────────────────────────────────────────────

pub async fn export_trades_csv(State(state): State<AppState>) -> impl IntoResponse {
    let query = quant_broker::journal::JournalQuery {
        symbol: None,
        entry_type: None,
        start: None,
        end: None,
        limit: Some(10_000),
    };
    let entries = state.journal.query(&query);

    let mut csv =
        String::from("timestamp,type,symbol,side,quantity,price,status,pnl,portfolio_value\n");

    for e in &entries {
        let side = e.side.as_deref().unwrap_or("");
        let qty = e
            .quantity
            .map(|v| v.to_string())
            .unwrap_or_default();
        let price = e
            .price
            .map(|v| v.to_string())
            .unwrap_or_default();
        let status = e.status.as_deref().unwrap_or("");
        let pnl = e
            .pnl
            .map(|v| v.to_string())
            .unwrap_or_default();
        let pv = e
            .portfolio_value
            .map(|v| v.to_string())
            .unwrap_or_default();

        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            e.timestamp, e.entry_type, e.symbol, side, qty, price, status, pnl, pv
        ));
    }

    csv_response("trades.csv", csv).into_response()
}

// ── Export Performance Metrics ──────────────────────────────────────

pub async fn export_metrics_csv(
    State(state): State<AppState>,
    Json(req): Json<ExportTaskRequest>,
) -> impl IntoResponse {
    let task = match state.task_store.get(&req.task_id) {
        Some(t) => t,
        None => return error_response(StatusCode::NOT_FOUND, "Task not found").into_response(),
    };

    let result_json: Value = match &task.result {
        Some(r) => match serde_json::from_str(r) {
            Ok(v) => v,
            Err(_) => {
                return error_response(StatusCode::BAD_REQUEST, "Invalid task result")
                    .into_response()
            }
        },
        None => {
            return error_response(StatusCode::BAD_REQUEST, "Task has no result").into_response()
        }
    };

    let columns = [
        "strategy",
        "symbol",
        "start",
        "end",
        "initial_capital",
        "final_value",
        "total_return_percent",
        "annual_return_percent",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown_percent",
        "max_drawdown_duration_days",
        "win_rate_percent",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "avg_holding_days",
        "total_commission",
        "turnover_rate",
        "alpha",
        "beta",
        "information_ratio",
        "tracking_error",
    ];

    let header = columns.join(",");
    let values: Vec<String> = columns
        .iter()
        .map(|&col| {
            result_json
                .get(col)
                .map(val_str)
                .unwrap_or_default()
        })
        .collect();
    let row = values.join(",");

    let csv = format!("{}\n{}\n", header, row);
    csv_response("metrics.csv", csv).into_response()
}
