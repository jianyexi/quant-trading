use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::StreamExt;
use serde_json::json;
use tokio::time::{interval, Duration};

use crate::state::AppState;

/// WebSocket upgrade handler for streaming chat.
pub async fn ws_chat(
    ws: WebSocketUpgrade,
    State(_state): State<AppState>,
) -> Response {
    ws.on_upgrade(handle_ws_chat)
}

async fn handle_ws_chat(mut socket: WebSocket) {
    // Send a welcome message
    let welcome = json!({
        "type": "connected",
        "message": "Connected to Quant Trading AI chat stream"
    });
    if socket
        .send(Message::Text(welcome.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    // Process incoming messages
    while let Some(Ok(msg)) = socket.next().await {
        match msg {
            Message::Text(text) => {
                // Stub: echo back with a placeholder response
                let response = json!({
                    "type": "message",
                    "content": format!("Echo: {}", text),
                    "done": true
                });
                if socket
                    .send(Message::Text(response.to_string().into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

/// WebSocket upgrade handler for real-time risk monitoring.
pub async fn ws_monitor(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_ws_monitor(socket, state))
}

async fn handle_ws_monitor(mut socket: WebSocket, state: AppState) {
    let welcome = json!({
        "type": "connected",
        "message": "Connected to risk monitor stream"
    });
    if socket
        .send(Message::Text(welcome.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    let mut tick = interval(Duration::from_millis(2000));
    let mut last_event_count: usize = 0;

    loop {
        tokio::select! {
            _ = tick.tick() => {
                let engine_guard = state.engine.lock().await;
                let payload = if let Some(ref eng) = *engine_guard {
                    let snapshot = eng.risk_enforcer().risk_signals_snapshot();
                    let status = eng.status().await;
                    let event_count = snapshot.recent_events.len();
                    let has_new_events = event_count > last_event_count;
                    last_event_count = event_count;
                    json!({
                        "type": "risk_update",
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                        "risk": snapshot,
                        "performance": {
                            "portfolio_value": status.performance.portfolio_value,
                            "total_return_pct": status.performance.total_return_pct,
                            "drawdown_pct": status.performance.drawdown_pct,
                            "max_drawdown_pct": status.performance.max_drawdown_pct,
                            "win_rate": status.performance.win_rate,
                            "pnl": status.pnl,
                        },
                        "engine_running": status.running,
                        "has_new_events": has_new_events,
                    })
                } else {
                    json!({
                        "type": "risk_update",
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                        "engine_running": false,
                    })
                };
                drop(engine_guard);

                if socket.send(Message::Text(payload.to_string().into())).await.is_err() {
                    break;
                }
            }
            msg = socket.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(data))) => {
                        if socket.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
