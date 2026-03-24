use axum::{
    extract::State,
    extract::Path,
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::state::AppState;
use super::find_python;

// ── Dataset Export ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ExportDatasetRequest {
    /// PostgreSQL connection URL (overrides config/env)
    pub db_url: Option<String>,
    /// Path to trade journal SQLite database
    pub journal_db: Option<String>,
}

/// POST /api/llm/export-dataset — Export training datasets as async task
pub async fn llm_export_dataset(
    State(state): State<AppState>,
    body: Option<Json<ExportDatasetRequest>>,
) -> (StatusCode, Json<Value>) {
    let ts = state.task_store.clone();
    let task_id = ts.create_with_params("llm_dataset_export", None);
    let tid = task_id.clone();

    let db_url = body.as_ref().and_then(|b| b.db_url.clone())
        .or_else(|| std::env::var("DATABASE_URL").ok())
        .unwrap_or_else(|| "postgresql://postgres:postgres@127.0.0.1:5432/quant_trading".into());
    let journal_db = body.as_ref().and_then(|b| b.journal_db.clone())
        .unwrap_or_else(|| "data/trade_journal.db".into());

    tokio::task::spawn_blocking(move || {
        let script = std::path::Path::new("ml_models/llm_dataset_export.py");
        if !script.exists() {
            ts.fail(&tid, "llm_dataset_export.py not found");
            return;
        }
        let python = match find_python() {
            Some(p) => p,
            None => { ts.fail(&tid, "Python not found"); return; }
        };

        ts.set_progress(&tid, "Exporting training datasets...");

        let output = std::process::Command::new(&python)
            .args([
                "ml_models/llm_dataset_export.py",
                "--db-url", &db_url,
                "--journal-db", &journal_db,
                "--output-dir", "data/llm_training",
            ])
            .env("PYTHONIOENCODING", "utf-8")
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                if out.status.success() {
                    // Try to extract JSON manifest from last line of stdout
                    let manifest = stdout.lines().last()
                        .and_then(|l| serde_json::from_str::<Value>(l).ok())
                        .unwrap_or(json!({"stdout": stdout}));
                    ts.complete(&tid, &manifest.to_string());
                } else {
                    let detail = if stderr.is_empty() { &stdout } else { &stderr };
                    ts.fail(&tid, &format!("Export failed (exit {}): {}", out.status, detail));
                }
            }
            Err(e) => ts.fail(&tid, &format!("spawn error: {}", e)),
        }
    });

    (StatusCode::ACCEPTED, Json(json!({ "task_id": task_id, "status": "running" })))
}

// ── Training ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct LlmTrainRequest {
    /// Training type: "sft" or "dpo"
    #[serde(default = "default_train_type")]
    pub train_type: String,
    /// HuggingFace model name or local path
    #[serde(default = "default_base_model")]
    pub base_model: String,
    /// Path to SFT adapter (for DPO training, optional)
    pub sft_adapter: Option<String>,
    /// LoRA rank (8/16/32)
    pub lora_rank: Option<u32>,
    /// Number of training epochs
    pub epochs: Option<u32>,
    /// Batch size
    pub batch_size: Option<u32>,
    /// Learning rate
    pub learning_rate: Option<f64>,
    /// DPO beta (KL penalty strength)
    pub beta: Option<f64>,
}

fn default_train_type() -> String { "sft".into() }
fn default_base_model() -> String { "Qwen/Qwen2.5-7B-Instruct".into() }

/// POST /api/llm/train — Trigger SFT or DPO training as async task
pub async fn llm_train(
    State(state): State<AppState>,
    Json(req): Json<LlmTrainRequest>,
) -> (StatusCode, Json<Value>) {
    let ts = state.task_store.clone();
    let task_type = format!("llm_{}", req.train_type);
    let params_json = serde_json::to_string(&json!({
        "train_type": req.train_type,
        "base_model": req.base_model,
        "lora_rank": req.lora_rank,
        "epochs": req.epochs,
    })).unwrap_or_default();
    let task_id = ts.create_with_params(&task_type, Some(&params_json));
    let tid = task_id.clone();

    tokio::task::spawn_blocking(move || {
        let (script, output_dir) = match req.train_type.as_str() {
            "dpo" => ("ml_models/llm_dpo_train.py", "ml_models/llm_adapters/dpo"),
            _ => ("ml_models/llm_sft_train.py", "ml_models/llm_adapters/sft"),
        };

        if !std::path::Path::new(script).exists() {
            ts.fail(&tid, &format!("{} not found", script));
            return;
        }
        let python = match find_python() {
            Some(p) => p,
            None => { ts.fail(&tid, "Python not found"); return; }
        };

        let mut args = vec![
            script.to_string(),
            "--base-model".into(), req.base_model,
            "--output-dir".into(), output_dir.into(),
            "--data-dir".into(), "data/llm_training".into(),
        ];
        if let Some(r) = req.lora_rank {
            args.extend(["--lora-rank".into(), r.to_string()]);
        }
        if let Some(e) = req.epochs {
            args.extend(["--epochs".into(), e.to_string()]);
        }
        if let Some(b) = req.batch_size {
            args.extend(["--batch-size".into(), b.to_string()]);
        }
        if let Some(lr) = req.learning_rate {
            args.extend(["--learning-rate".into(), lr.to_string()]);
        }
        if let Some(adapter) = req.sft_adapter {
            args.extend(["--sft-adapter".into(), adapter]);
        }
        if let Some(beta) = req.beta {
            args.extend(["--beta".into(), beta.to_string()]);
        }

        ts.set_progress(&tid, &format!("{} training started...", req.train_type.to_uppercase()));

        let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let output = std::process::Command::new(&python)
            .args(&str_args)
            .env("PYTHONIOENCODING", "utf-8")
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                if out.status.success() {
                    let report = stdout.lines().last()
                        .and_then(|l| serde_json::from_str::<Value>(l).ok())
                        .unwrap_or(json!({"stdout": stdout}));
                    ts.complete(&tid, &report.to_string());
                } else {
                    let detail = if stderr.is_empty() { &stdout } else { &stderr };
                    ts.fail(&tid, &format!("Training failed (exit {}): {}", out.status, detail));
                }
            }
            Err(e) => ts.fail(&tid, &format!("spawn error: {}", e)),
        }
    });

    (StatusCode::ACCEPTED, Json(json!({ "task_id": task_id, "status": "running" })))
}

// ── Model Management ────────────────────────────────────────────────

/// GET /api/llm/models — List available LLM adapters
pub async fn llm_list_models() -> Json<Value> {
    let adapters_dir = std::path::Path::new("ml_models/llm_adapters");
    let mut models = Vec::new();

    if adapters_dir.exists() {
        for entry in std::fs::read_dir(adapters_dir).into_iter().flatten() {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if !path.is_dir() { continue; }

            let adapter_path = path.join("adapter");
            let report_path = path.join("training_report.json");

            let report: Option<Value> = if report_path.exists() {
                std::fs::read_to_string(&report_path).ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
            } else {
                None
            };

            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            models.push(json!({
                "name": name,
                "adapter_path": adapter_path.to_string_lossy(),
                "has_adapter": adapter_path.exists(),
                "report": report,
            }));
        }
    }

    // Check dataset status too
    let dataset_dir = std::path::Path::new("data/llm_training");
    let manifest: Option<Value> = if dataset_dir.join("manifest.json").exists() {
        std::fs::read_to_string(dataset_dir.join("manifest.json")).ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    let dataset_stats = json!({
        "sft_chat": file_line_count(&dataset_dir.join("sft_chat.jsonl")),
        "sft_sentiment": file_line_count(&dataset_dir.join("sft_sentiment.jsonl")),
        "dpo_trades": file_line_count(&dataset_dir.join("dpo_trades.jsonl")),
        "manifest": manifest,
    });

    Json(json!({
        "models": models,
        "dataset": dataset_stats,
    }))
}

fn file_line_count(path: &std::path::Path) -> usize {
    if !path.exists() { return 0; }
    std::fs::read_to_string(path)
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

/// POST /api/llm/models/:name/activate — Activate an adapter for inference
pub async fn llm_activate_model(
    Path(name): Path<String>,
) -> Json<Value> {
    let adapter_path = format!("ml_models/llm_adapters/{}/adapter", name);
    if !std::path::Path::new(&adapter_path).exists() {
        return Json(json!({"error": format!("Adapter '{}' not found", name)}));
    }

    // Write active model config for ml_serve.py to pick up on reload
    let config = json!({
        "active_adapter": adapter_path,
        "name": name,
        "activated_at": chrono::Utc::now().to_rfc3339(),
    });

    let config_path = "ml_models/llm_adapters/active.json";
    match std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap()) {
        Ok(_) => Json(json!({
            "status": "activated",
            "adapter": adapter_path,
            "name": name,
            "note": "Restart ml_serve or call POST /reload to apply",
        })),
        Err(e) => Json(json!({"error": format!("Failed to write active config: {}", e)})),
    }
}
