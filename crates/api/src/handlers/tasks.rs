use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde_json::{json, Value};

use crate::state::AppState;

/// List recent tasks (newest first)
pub async fn list_tasks(
    State(state): State<AppState>,
) -> Json<Value> {
    let tasks = state.task_store.list(50);
    Json(json!({ "tasks": tasks }))
}

/// List only running tasks
pub async fn list_running_tasks(
    State(state): State<AppState>,
) -> Json<Value> {
    let tasks = state.task_store.list_running();
    Json(json!({ "tasks": tasks }))
}

/// Get a single task by ID
pub async fn get_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> (StatusCode, Json<Value>) {
    match state.task_store.get(&id) {
        Some(task) => (StatusCode::OK, Json(json!(task))),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "Task not found"}))),
    }
}
