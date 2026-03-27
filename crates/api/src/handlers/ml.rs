use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

/// GET /api/ml/feature-importance
///
/// Returns ranked feature importance from the latest training report.
/// Falls back to the embedded FEATURE_NAMES when no report files exist.
pub async fn ml_feature_importance() -> (StatusCode, Json<Value>) {
    let models_dir = std::path::Path::new("ml_models");

    // Try report files in priority order:
    //   1. factor_mining_report_*.json
    //   2. retrain_report_*.json
    //   3. mined_model_report_*.json
    let prefixes = [
        "factor_mining_report_",
        "retrain_report_",
        "mined_model_report_",
    ];

    let mut report: Option<Value> = None;

    if models_dir.exists() {
        for prefix in &prefixes {
            if report.is_some() {
                break;
            }
            if let Ok(entries) = std::fs::read_dir(models_dir) {
                let mut matched: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        name.starts_with(prefix) && name.ends_with(".json")
                    })
                    .collect();
                matched.sort_by_key(|e| e.file_name());
                if let Some(latest) = matched.last() {
                    if let Ok(content) = std::fs::read_to_string(latest.path()) {
                        report = serde_json::from_str::<Value>(&content).ok();
                    }
                }
            }
        }
    }

    // Also try training_history.db for the latest run with feature_importance
    if report.is_none() {
        if let Ok(conn) = rusqlite::Connection::open("data/training_history.db") {
            let row: Option<(String, String, Option<f64>, Option<f64>, Option<i64>)> = conn
                .query_row(
                    "SELECT feature_importance, timestamp, auc, accuracy, n_features
                     FROM training_runs
                     WHERE feature_importance IS NOT NULL AND feature_importance != '' AND feature_importance != '[]'
                     ORDER BY timestamp DESC LIMIT 1",
                    [],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1).unwrap_or_default(),
                            row.get::<_, f64>(2).ok(),
                            row.get::<_, f64>(3).ok(),
                            row.get::<_, i64>(4).ok(),
                        ))
                    },
                )
                .ok();

            if let Some((fi_json, ts, auc, accuracy, n_features)) = row {
                if let Ok(fi_val) = serde_json::from_str::<Value>(&fi_json) {
                    report = Some(json!({
                        "feature_importance": fi_val,
                        "timestamp": ts,
                        "auc": auc,
                        "accuracy": accuracy,
                        "n_features": n_features,
                    }));
                }
            }
        }
    }

    if let Some(ref rpt) = report {
        // feature_importance can be either:
        //   dict: { "name": value, ... }
        //   list: [ {"feature": "name", "importance": value}, ... ]
        let mut features: Vec<Value> = Vec::new();

        if let Some(obj) = rpt.get("feature_importance").and_then(|v| v.as_object()) {
            // Dict form
            for (name, val) in obj {
                features.push(json!({
                    "name": name,
                    "importance": val.as_f64().unwrap_or(0.0),
                }));
            }
        } else if let Some(arr) = rpt.get("feature_importance").and_then(|v| v.as_array()) {
            // List form
            for item in arr {
                let name = item
                    .get("feature")
                    .or_else(|| item.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let importance = item
                    .get("importance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                features.push(json!({
                    "name": name,
                    "importance": importance,
                }));
            }
        }

        // Sort descending by importance
        features.sort_by(|a, b| {
            let ia = a["importance"].as_f64().unwrap_or(0.0);
            let ib = b["importance"].as_f64().unwrap_or(0.0);
            ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, f) in features.iter_mut().enumerate() {
            if let Some(obj) = f.as_object_mut() {
                obj.insert("rank".to_string(), json!(i + 1));
            }
        }

        // Extract model metadata
        let model_info = json!({
            "auc": rpt.get("auc").or_else(|| rpt.get("test_auc")).and_then(|v| v.as_f64()),
            "accuracy": rpt.get("accuracy").or_else(|| rpt.get("test_accuracy")).and_then(|v| v.as_f64()),
            "n_features": features.len(),
            "timestamp": rpt.get("timestamp").and_then(|v| v.as_str()).unwrap_or(""),
        });

        return (
            StatusCode::OK,
            Json(json!({
                "features": features,
                "model_info": model_info,
            })),
        );
    }

    // Fallback: use embedded FEATURE_NAMES (no importance values)
    let feature_names = quant_ml::ml_factor::FEATURE_NAMES;
    let features: Vec<Value> = feature_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            json!({
                "name": name,
                "importance": Value::Null,
                "rank": i + 1,
            })
        })
        .collect();

    (
        StatusCode::OK,
        Json(json!({
            "features": features,
            "model_info": {
                "auc": Value::Null,
                "accuracy": Value::Null,
                "n_features": features.len(),
                "timestamp": "",
            },
        })),
    )
}
