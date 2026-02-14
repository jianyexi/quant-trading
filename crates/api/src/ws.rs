use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::StreamExt;
use serde_json::json;

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
