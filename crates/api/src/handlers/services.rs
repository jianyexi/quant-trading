use axum::{extract::State, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, error, warn};

use crate::state::AppState;

// ── ML Serve Management ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MlServeStartRequest {
    /// Path to model file (default: ml_models/factor_model.lgb.txt)
    pub model_path: Option<String>,
    /// HTTP port (default: 18091)
    pub http_port: Option<u16>,
    /// TCP MQ port (default: 18094)
    pub tcp_port: Option<u16>,
    /// Device: "auto", "cpu", "cuda" (default: auto)
    pub device: Option<String>,
}

/// POST /api/services/ml-serve/start — start ml_serve.py as a managed subprocess.
pub async fn ml_serve_start(
    State(state): State<AppState>,
    Json(req): Json<MlServeStartRequest>,
) -> Json<Value> {
    let mut procs = state.managed_processes.lock().await;

    // Check if already running
    if let Some(proc) = procs.get_mut("ml_serve") {
        match proc.child.try_wait() {
            Ok(None) => {
                return Json(json!({
                    "status": "already_running",
                    "pid": proc.child.id(),
                    "started_at": proc.started_at.to_rfc3339(),
                }));
            }
            _ => {
                procs.remove("ml_serve");
            }
        }
    }

    let python = match super::find_python() {
        Some(p) => p,
        None => return Json(json!({ "error": "Python not found" })),
    };

    let model_path = req.model_path.unwrap_or_else(|| "ml_models/factor_model.lgb.txt".to_string());
    let http_port = req.http_port.unwrap_or(18091);
    let tcp_port = req.tcp_port.unwrap_or(18094);
    let device = req.device.unwrap_or_else(|| "auto".to_string());

    let script = "ml_models/ml_serve.py";
    if !std::path::Path::new(script).exists() {
        return Json(json!({ "error": format!("{} not found", script) }));
    }

    let args: Vec<String> = vec![
        script.to_string(),
        "--model".to_string(), model_path.clone(),
        "--port".to_string(), http_port.to_string(),
        "--tcp-port".to_string(), tcp_port.to_string(),
        "--device".to_string(), device.clone(),
    ];

    match std::process::Command::new(&python)
        .args(&args)
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => {
            let pid = child.id();
            info!(pid=%pid, model=%model_path, http_port, tcp_port, device=%device, "ML serve started");
            state.log_store.push(
                crate::log_store::LogLevel::Info, "SERVICE", "ml_serve/start",
                0, 0, &format!("ML serve started (pid={}, model={}, ports={}/{})", pid, model_path, http_port, tcp_port), None,
            );

            let proc = crate::state::ManagedProcess {
                name: "ml_serve".to_string(),
                child,
                started_at: chrono::Utc::now(),
                args: args.clone(),
            };
            procs.insert("ml_serve".to_string(), proc);

            Json(json!({
                "status": "started",
                "pid": pid,
                "model": model_path,
                "http_port": http_port,
                "tcp_port": tcp_port,
                "device": device,
            }))
        }
        Err(e) => {
            error!(error=%e, "Failed to start ml_serve.py");
            Json(json!({ "error": format!("Failed to start: {}", e) }))
        }
    }
}

/// POST /api/services/ml-serve/stop — stop ml_serve.py.
pub async fn ml_serve_stop(State(state): State<AppState>) -> Json<Value> {
    let mut procs = state.managed_processes.lock().await;

    if let Some(mut proc) = procs.remove("ml_serve") {
        let pid = proc.child.id();
        match proc.child.kill() {
            Ok(_) => {
                let _ = proc.child.wait(); // reap zombie
                info!(pid=%pid, "ML serve stopped");
                state.log_store.push(
                    crate::log_store::LogLevel::Info, "SERVICE", "ml_serve/stop",
                    0, 0, &format!("ML serve stopped (pid={})", pid), None,
                );
                Json(json!({ "status": "stopped", "pid": pid }))
            }
            Err(e) => {
                let msg = e.to_string();
                warn!(pid=%pid, error=%msg, "ML serve kill failed");
                Json(json!({ "status": "stop_failed", "error": msg }))
            }
        }
    } else {
        Json(json!({ "status": "not_running" }))
    }
}

/// GET /api/services/ml-serve/status — check ml_serve status + health.
pub async fn ml_serve_status(State(state): State<AppState>) -> Json<Value> {
    let mut procs = state.managed_processes.lock().await;

    let process_info = if let Some(proc) = procs.get_mut("ml_serve") {
        match proc.child.try_wait() {
            Ok(None) => {
                Some(json!({
                    "process": "running",
                    "pid": proc.child.id(),
                    "started_at": proc.started_at.to_rfc3339(),
                    "uptime_secs": (chrono::Utc::now() - proc.started_at).num_seconds(),
                }))
            }
            Ok(Some(exit)) => {
                let code = exit.code().unwrap_or(-1);
                procs.remove("ml_serve");
                Some(json!({
                    "process": "exited",
                    "exit_code": code,
                }))
            }
            Err(e) => {
                let msg = e.to_string();
                procs.remove("ml_serve");
                Some(json!({
                    "process": "error",
                    "error": msg,
                }))
            }
        }
    } else {
        None
    };
    drop(procs);

    let health = probe_ml_serve_health(18091).await;

    Json(json!({
        "service": "ml_serve",
        "managed": process_info.is_some(),
        "process_info": process_info.unwrap_or(json!({"process": "not_managed"})),
        "health": health,
    }))
}

/// GET /api/services/status — status of all managed services.
pub async fn services_status(State(state): State<AppState>) -> Json<Value> {
    let mut procs = state.managed_processes.lock().await;
    let mut services: Vec<Value> = Vec::new();
    let mut to_remove: Vec<String> = Vec::new();

    for (name, proc) in procs.iter_mut() {
        let status: Value = match proc.child.try_wait() {
            Ok(None) => json!({
                "name": name,
                "status": "running",
                "pid": proc.child.id(),
                "started_at": proc.started_at.to_rfc3339(),
                "uptime_secs": (chrono::Utc::now() - proc.started_at).num_seconds(),
            }),
            Ok(Some(exit)) => {
                to_remove.push(name.clone());
                json!({
                    "name": name,
                    "status": "exited",
                    "exit_code": exit.code().unwrap_or(-1),
                })
            }
            Err(e) => {
                to_remove.push(name.clone());
                let msg = e.to_string();
                json!({
                    "name": name,
                    "status": "error",
                    "error": msg,
                })
            }
        };
        services.push(status);
    }

    for name in to_remove {
        procs.remove(&name);
    }
    drop(procs);

    let ml_health = probe_ml_serve_health(18091).await;

    Json(json!({
        "services": services,
        "ml_serve_health": ml_health,
    }))
}

/// Probe ml_serve health endpoint via HTTP.
async fn probe_ml_serve_health(port: u16) -> Value {
    probe_service_health(port).await
}

/// Probe any service health endpoint via HTTP.
async fn probe_service_health(port: u16) -> Value {
    let url = format!("http://127.0.0.1:{}/health", port);
    match reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Value>().await {
                Ok(body) => json!({ "reachable": true, "data": body }),
                Err(_) => json!({ "reachable": true, "data": null }),
            }
        }
        Ok(resp) => {
            let code = resp.status().as_u16();
            json!({ "reachable": false, "status": code })
        }
        Err(_) => json!({ "reachable": false }),
    }
}

// ── LLM Signal Serve Management ────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct LlmSignalServeStartRequest {
    /// HuggingFace model ID or local path (default: Qwen/Qwen2.5-7B-Instruct)
    pub base_model: Option<String>,
    /// Path to LoRA adapter directory (auto-detected if None)
    pub adapter: Option<String>,
    /// HTTP port (default: 18095)
    pub port: Option<u16>,
    /// Device: "auto", "cpu", "cuda" (default: auto)
    pub device: Option<String>,
    /// Disable 4-bit quantization
    pub no_quantize: Option<bool>,
}

/// POST /api/services/llm-signal-serve/start — start llm_signal_serve.py.
pub async fn llm_signal_serve_start(
    State(state): State<AppState>,
    Json(req): Json<LlmSignalServeStartRequest>,
) -> Json<Value> {
    let mut procs = state.managed_processes.lock().await;

    if let Some(proc) = procs.get_mut("llm_signal_serve") {
        match proc.child.try_wait() {
            Ok(None) => {
                return Json(json!({
                    "status": "already_running",
                    "pid": proc.child.id(),
                    "started_at": proc.started_at.to_rfc3339(),
                }));
            }
            _ => {
                procs.remove("llm_signal_serve");
            }
        }
    }

    let python = match super::find_python() {
        Some(p) => p,
        None => return Json(json!({ "error": "Python not found" })),
    };

    let script = "ml_models/llm_signal_serve.py";
    if !std::path::Path::new(script).exists() {
        return Json(json!({ "error": format!("{} not found", script) }));
    }

    let base_model = req.base_model.unwrap_or_else(|| "Qwen/Qwen2.5-7B-Instruct".to_string());
    let port = req.port.unwrap_or(18095);
    let device = req.device.unwrap_or_else(|| "auto".to_string());

    let mut args: Vec<String> = vec![
        script.to_string(),
        "--base-model".to_string(), base_model.clone(),
        "--port".to_string(), port.to_string(),
        "--device".to_string(), device.clone(),
    ];

    if let Some(adapter) = &req.adapter {
        args.push("--adapter".to_string());
        args.push(adapter.clone());
    }
    if req.no_quantize.unwrap_or(false) {
        args.push("--no-quantize".to_string());
    }

    match std::process::Command::new(&python)
        .args(&args)
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => {
            let pid = child.id();
            info!(pid=%pid, base_model=%base_model, port, device=%device, "LLM signal serve started");
            state.log_store.push(
                crate::log_store::LogLevel::Info, "SERVICE", "llm_signal_serve/start",
                0, 0, &format!("LLM signal serve started (pid={}, model={}, port={})", pid, base_model, port), None,
            );

            let proc = crate::state::ManagedProcess {
                name: "llm_signal_serve".to_string(),
                child,
                started_at: chrono::Utc::now(),
                args: args.clone(),
            };
            procs.insert("llm_signal_serve".to_string(), proc);

            Json(json!({
                "status": "started",
                "pid": pid,
                "base_model": base_model,
                "port": port,
                "device": device,
            }))
        }
        Err(e) => {
            error!(error=%e, "Failed to start llm_signal_serve.py");
            Json(json!({ "error": format!("Failed to start: {}", e) }))
        }
    }
}

/// POST /api/services/llm-signal-serve/stop — stop llm_signal_serve.py.
pub async fn llm_signal_serve_stop(State(state): State<AppState>) -> Json<Value> {
    let mut procs = state.managed_processes.lock().await;

    if let Some(mut proc) = procs.remove("llm_signal_serve") {
        let pid = proc.child.id();
        match proc.child.kill() {
            Ok(_) => {
                let _ = proc.child.wait();
                info!(pid=%pid, "LLM signal serve stopped");
                state.log_store.push(
                    crate::log_store::LogLevel::Info, "SERVICE", "llm_signal_serve/stop",
                    0, 0, &format!("LLM signal serve stopped (pid={})", pid), None,
                );
                Json(json!({ "status": "stopped", "pid": pid }))
            }
            Err(e) => {
                let msg = e.to_string();
                warn!(pid=%pid, error=%msg, "LLM signal serve kill failed");
                Json(json!({ "status": "stop_failed", "error": msg }))
            }
        }
    } else {
        Json(json!({ "status": "not_running" }))
    }
}

/// GET /api/services/llm-signal-serve/status — check llm_signal_serve status + health.
pub async fn llm_signal_serve_status(State(state): State<AppState>) -> Json<Value> {
    let mut procs = state.managed_processes.lock().await;

    let process_info = if let Some(proc) = procs.get_mut("llm_signal_serve") {
        match proc.child.try_wait() {
            Ok(None) => {
                Some(json!({
                    "process": "running",
                    "pid": proc.child.id(),
                    "started_at": proc.started_at.to_rfc3339(),
                    "uptime_secs": (chrono::Utc::now() - proc.started_at).num_seconds(),
                }))
            }
            Ok(Some(exit)) => {
                let code = exit.code().unwrap_or(-1);
                procs.remove("llm_signal_serve");
                Some(json!({
                    "process": "exited",
                    "exit_code": code,
                }))
            }
            Err(e) => {
                let msg = e.to_string();
                procs.remove("llm_signal_serve");
                Some(json!({
                    "process": "error",
                    "error": msg,
                }))
            }
        }
    } else {
        None
    };
    drop(procs);

    let health = probe_service_health(18095).await;

    Json(json!({
        "service": "llm_signal_serve",
        "managed": process_info.is_some(),
        "process_info": process_info.unwrap_or(json!({"process": "not_managed"})),
        "health": health,
    }))
}
