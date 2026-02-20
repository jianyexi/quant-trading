use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct NotificationQuery {
    pub limit: Option<usize>,
    pub unread_only: Option<bool>,
}

/// GET /api/notifications — list notifications
pub async fn list_notifications(
    State(state): State<AppState>,
    Query(q): Query<NotificationQuery>,
) -> Json<Value> {
    let limit = q.limit.unwrap_or(50);
    let unread_only = q.unread_only.unwrap_or(false);
    let items = state.notifier.list(limit, unread_only);
    let unread = state.notifier.unread_count();
    Json(json!({ "notifications": items, "unread_count": unread }))
}

/// GET /api/notifications/unread-count — just the count (for badge polling)
pub async fn notification_unread_count(
    State(state): State<AppState>,
) -> Json<Value> {
    Json(json!({ "unread_count": state.notifier.unread_count() }))
}

/// POST /api/notifications/:id/read — mark one as read
pub async fn notification_mark_read(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> StatusCode {
    state.notifier.mark_read(&id);
    StatusCode::OK
}

/// POST /api/notifications/read-all — mark all as read
pub async fn notification_mark_all_read(
    State(state): State<AppState>,
) -> StatusCode {
    state.notifier.mark_all_read();
    StatusCode::OK
}

/// GET /api/notifications/config — get notification config
pub async fn notification_config_get(
    State(state): State<AppState>,
) -> Json<Value> {
    let cfg = state.notifier.get_config();
    Json(serde_json::to_value(cfg).unwrap_or_default())
}

/// POST /api/notifications/config — save notification config
pub async fn notification_config_save(
    State(state): State<AppState>,
    Json(cfg): Json<quant_broker::notifier::NotificationConfig>,
) -> StatusCode {
    state.notifier.set_config(cfg);
    StatusCode::OK
}

/// POST /api/notifications/test — send test notification
pub async fn notification_test(
    State(state): State<AppState>,
) -> Json<Value> {
    let results = state.notifier.send_test().await;
    let items: Vec<Value> = results.into_iter().map(|(ch, ok, msg)| {
        json!({ "channel": ch, "success": ok, "message": msg })
    }).collect();
    Json(json!({ "results": items }))
}
