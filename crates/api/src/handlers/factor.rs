use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use serde_json::{json, Value};

use super::{find_python, run_python_script};
use crate::state::AppState;

fn build_data_args(args: &mut Vec<String>, data_source: &str, symbols: &str, start_date: &str, end_date: &str, _n_bars: i64) {
    // Always use real data (akshare/tushare) — synthetic data is not allowed
    args.push("--akshare".into());
    if !symbols.is_empty() {
        args.push("--symbols".into());
        args.push(symbols.to_string());
    }
    args.push("--start-date".into());
    args.push(start_date.to_string());
    args.push("--end-date".into());
    args.push(end_date.to_string());
    let _ = data_source; // always real data
}

/// Run Phase-1 parameterized factor mining (async task)
pub async fn factor_mine_parametric(
    State(state): State<AppState>,
    body: Option<Json<Value>>,
) -> (StatusCode, Json<Value>) {
    let body_val = body.map(|b| b.0).unwrap_or(json!({}));
    let n_bars = body_val.get("n_bars").and_then(|v| v.as_i64()).unwrap_or(3000);
    let horizon = body_val.get("horizon").and_then(|v| v.as_i64()).unwrap_or(5);
    let ic_threshold = body_val.get("ic_threshold").and_then(|v| v.as_f64()).unwrap_or(0.02);
    let top_n = body_val.get("top_n").and_then(|v| v.as_i64()).unwrap_or(30);
    let retrain = body_val.get("retrain").and_then(|v| v.as_bool()).unwrap_or(false);
    let cross_stock = body_val.get("cross_stock").and_then(|v| v.as_bool()).unwrap_or(false);
    let data_source = body_val.get("data_source").and_then(|v| v.as_str()).unwrap_or("akshare").to_string();
    let symbols = body_val.get("symbols").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let start_date = body_val.get("start_date").and_then(|v| v.as_str()).unwrap_or("2023-01-01").to_string();
    let end_date = body_val.get("end_date").and_then(|v| v.as_str()).unwrap_or("2024-12-31").to_string();

    let ts = state.task_store.clone();
    let task_id = ts.create("factor_mine_parametric");
    let tid = task_id.clone();

    tokio::task::spawn_blocking(move || {
        let script = std::path::Path::new("ml_models/factor_mining.py");
        if !script.exists() {
            ts.fail(&tid, "factor_mining.py not found");
            return;
        }
        let python = match find_python() {
            Some(p) => p,
            None => { ts.fail(&tid, "Python not found"); return; }
        };

        let mut args = vec!["ml_models/factor_mining.py".to_string()];
        build_data_args(&mut args, &data_source, &symbols, &start_date, &end_date, n_bars);
        args.extend([
            "--horizon".into(), horizon.to_string(),
            "--ic-threshold".into(), ic_threshold.to_string(),
            "--export-top".into(), top_n.to_string(),
        ]);
        if retrain { args.push("--retrain".into()); }
        if cross_stock { args.push("--cross-stock".into()); }

        match run_python_script(&python, &args) {
            Ok(val) => ts.complete(&tid, &val.to_string()),
            Err(e) => ts.fail(&tid, &e),
        }
    });

    (StatusCode::ACCEPTED, Json(json!({ "task_id": task_id, "status": "running" })))
}

/// Run Phase-2 GP factor mining (async task)
pub async fn factor_mine_gp(
    State(state): State<AppState>,
    body: Option<Json<Value>>,
) -> (StatusCode, Json<Value>) {
    let body_val = body.map(|b| b.0).unwrap_or(json!({}));
    let n_bars = body_val.get("n_bars").and_then(|v| v.as_i64()).unwrap_or(3000);
    let pop_size = body_val.get("pop_size").and_then(|v| v.as_i64()).unwrap_or(200);
    let generations = body_val.get("generations").and_then(|v| v.as_i64()).unwrap_or(30);
    let max_depth = body_val.get("max_depth").and_then(|v| v.as_i64()).unwrap_or(6);
    let horizon = body_val.get("horizon").and_then(|v| v.as_i64()).unwrap_or(5);
    let retrain = body_val.get("retrain").and_then(|v| v.as_bool()).unwrap_or(false);
    let data_source = body_val.get("data_source").and_then(|v| v.as_str()).unwrap_or("akshare").to_string();
    let symbols = body_val.get("symbols").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let start_date = body_val.get("start_date").and_then(|v| v.as_str()).unwrap_or("2023-01-01").to_string();
    let end_date = body_val.get("end_date").and_then(|v| v.as_str()).unwrap_or("2024-12-31").to_string();

    let ts = state.task_store.clone();
    let task_id = ts.create("factor_mine_gp");
    let tid = task_id.clone();

    tokio::task::spawn_blocking(move || {
        let script = std::path::Path::new("ml_models/gp_factor_mining.py");
        if !script.exists() {
            ts.fail(&tid, "gp_factor_mining.py not found");
            return;
        }
        let python = match find_python() {
            Some(p) => p,
            None => { ts.fail(&tid, "Python not found"); return; }
        };

        let mut args = vec!["ml_models/gp_factor_mining.py".to_string()];
        build_data_args(&mut args, &data_source, &symbols, &start_date, &end_date, n_bars);
        args.extend([
            "--pop-size".into(), pop_size.to_string(),
            "--generations".into(), generations.to_string(),
            "--max-depth".into(), max_depth.to_string(),
            "--horizon".into(), horizon.to_string(),
        ]);
        if retrain { args.push("--retrain".into()); }

        match run_python_script(&python, &args) {
            Ok(val) => ts.complete(&tid, &val.to_string()),
            Err(e) => ts.fail(&tid, &e),
        }
    });

    (StatusCode::ACCEPTED, Json(json!({ "task_id": task_id, "status": "running" })))
}

/// Get factor registry state
pub async fn factor_registry_get() -> Json<Value> {
    let path = std::path::Path::new("ml_models/factor_registry.json");
    if !path.exists() {
        return Json(json!({
            "factors": {},
            "stats": {"total_discovered": 0, "total_promoted": 0, "total_retired": 0}
        }));
    }
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let val: Value = serde_json::from_str(&content).unwrap_or(json!({}));
            Json(val)
        }
        Err(e) => Json(json!({"error": format!("Read error: {}", e)})),
    }
}

/// Run lifecycle management on factor registry (async task)
pub async fn factor_registry_manage(
    State(state): State<AppState>,
    body: Option<Json<Value>>,
) -> (StatusCode, Json<Value>) {
    let body_val = body.map(|b| b.0).unwrap_or(json!({}));
    let n_bars = body_val.get("n_bars").and_then(|v| v.as_i64()).unwrap_or(3000);
    let data_path = body_val.get("data").and_then(|v| v.as_str()).unwrap_or("").to_string();

    let ts = state.task_store.clone();
    let task_id = ts.create("factor_registry_manage");
    let tid = task_id.clone();

    tokio::task::spawn_blocking(move || {
        let script = std::path::Path::new("ml_models/gp_factor_mining.py");
        if !script.exists() {
            ts.fail(&tid, "gp_factor_mining.py not found");
            return;
        }
        let python = match find_python() {
            Some(p) => p,
            None => { ts.fail(&tid, "Python not found"); return; }
        };

        let mut args = vec!["ml_models/gp_factor_mining.py".to_string(), "--manage".into()];
        if data_path.is_empty() {
            args.push("--akshare".into());
            args.extend(["--n-bars".into(), n_bars.to_string()]);
        } else {
            args.push("--data".into());
            args.push(data_path);
        }

        match run_python_script(&python, &args) {
            Ok(val) => ts.complete(&tid, &val.to_string()),
            Err(e) => ts.fail(&tid, &e),
        }
    });

    (StatusCode::ACCEPTED, Json(json!({ "task_id": task_id, "status": "running" })))
}

/// Export promoted factors (async task)
pub async fn factor_export_promoted(
    State(state): State<AppState>,
    body: Option<Json<Value>>,
) -> (StatusCode, Json<Value>) {
    let body_val = body.map(|b| b.0).unwrap_or(json!({}));
    let retrain = body_val.get("retrain").and_then(|v| v.as_bool()).unwrap_or(false);
    let data_path = body_val.get("data").and_then(|v| v.as_str()).unwrap_or("").to_string();

    let ts = state.task_store.clone();
    let task_id = ts.create("factor_export_promoted");
    let tid = task_id.clone();

    tokio::task::spawn_blocking(move || {
        let script = std::path::Path::new("ml_models/gp_factor_mining.py");
        if !script.exists() {
            ts.fail(&tid, "gp_factor_mining.py not found");
            return;
        }
        let python = match find_python() {
            Some(p) => p,
            None => { ts.fail(&tid, "Python not found"); return; }
        };

        let mut args = vec![
            "ml_models/gp_factor_mining.py".to_string(),
            "--export-promoted".into(),
        ];
        if data_path.is_empty() {
            args.push("--akshare".into());
        } else {
            args.push("--data".into());
            args.push(data_path);
        }
        if retrain { args.push("--retrain".into()); }

        match run_python_script(&python, &args) {
            Ok(val) => ts.complete(&tid, &val.to_string()),
            Err(e) => ts.fail(&tid, &e),
        }
    });

    (StatusCode::ACCEPTED, Json(json!({ "task_id": task_id, "status": "running" })))
}

/// Get Phase-1 mining results (feature list + reports)
pub async fn factor_results() -> Json<Value> {
    let models_dir = std::path::Path::new("ml_models");

    let p1_features = std::fs::read_to_string(models_dir.join("mined_factor_features.txt"))
        .unwrap_or_default()
        .lines().filter(|l| !l.is_empty()).map(String::from).collect::<Vec<_>>();

    let gp_features: Vec<Value> = std::fs::read_to_string(models_dir.join("gp_factor_features.txt"))
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let parts: Vec<&str> = line.splitn(2, '\t').collect();
            json!({"id": parts.first().unwrap_or(&""), "expression": parts.get(1).unwrap_or(&"")})
        })
        .collect();

    let mut latest_report: Option<Value> = None;
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        let mut reports: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("factor_mining_report_"))
            .collect();
        reports.sort_by_key(|e| e.file_name());
        if let Some(latest) = reports.last() {
            if let Ok(content) = std::fs::read_to_string(latest.path()) {
                latest_report = serde_json::from_str(&content).ok();
            }
        }
    }

    let p1_rust = std::fs::read_to_string(models_dir.join("mined_factors_rust_snippet.rs")).unwrap_or_default();
    let gp_rust = std::fs::read_to_string(models_dir.join("gp_factors_rust_snippet.rs")).unwrap_or_default();

    Json(json!({
        "parametric": {
            "features": p1_features,
            "latest_report": latest_report,
            "rust_snippet": p1_rust,
        },
        "gp": {
            "features": gp_features,
            "rust_snippet": gp_rust,
        },
    }))
}
