use crate::client::{ChatMessage, ToolCall};
use chrono::NaiveDateTime;
use sqlx::PgPool;
use uuid::Uuid;

/// PostgreSQL-backed chat history persistence.
pub struct ChatHistoryStore {
    pool: PgPool,
}

#[derive(Debug)]
pub struct ChatSession {
    pub id: Uuid,
    pub title: Option<String>,
    pub created_at: NaiveDateTime,
}

impl ChatHistoryStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Create a new chat session and return its ID.
    pub async fn create_session(&self, title: Option<&str>) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO chat_sessions (id, title) VALUES ($1, $2)",
        )
        .bind(id)
        .bind(title)
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    /// List all chat sessions ordered by most recent first.
    pub async fn list_sessions(&self) -> anyhow::Result<Vec<ChatSession>> {
        let rows = sqlx::query_as::<_, (Uuid, Option<String>, NaiveDateTime)>(
            "SELECT id, title, created_at FROM chat_sessions ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let sessions = rows
            .into_iter()
            .map(|(id, title, created_at)| ChatSession {
                id,
                title,
                created_at,
            })
            .collect();
        Ok(sessions)
    }

    /// Persist a single message to a session.
    pub async fn save_message(
        &self,
        session_id: Uuid,
        message: &ChatMessage,
    ) -> anyhow::Result<()> {
        let content = message.content.clone().unwrap_or_default();
        let tool_calls_json: Option<serde_json::Value> = message
            .tool_calls
            .as_ref()
            .map(|tc| serde_json::to_value(tc))
            .transpose()?;

        sqlx::query(
            "INSERT INTO chat_messages (session_id, role, content, tool_calls, tool_call_id) \
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(session_id)
        .bind(&message.role)
        .bind(&content)
        .bind(&tool_calls_json)
        .bind(&message.tool_call_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Load all messages for a session in chronological order.
    pub async fn get_messages(&self, session_id: Uuid) -> anyhow::Result<Vec<ChatMessage>> {
        let rows = sqlx::query_as::<_, (String, String, Option<serde_json::Value>, Option<String>)>(
            "SELECT role, content, tool_calls, tool_call_id \
             FROM chat_messages WHERE session_id = $1 ORDER BY id ASC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let messages = rows
            .into_iter()
            .map(|(role, content, tool_calls_json, tool_call_id)| {
                let tool_calls: Option<Vec<ToolCall>> = tool_calls_json
                    .and_then(|v| serde_json::from_value(v).ok());

                let content = if content.is_empty() {
                    None
                } else {
                    Some(content)
                };

                ChatMessage {
                    role,
                    content,
                    tool_calls,
                    tool_call_id,
                }
            })
            .collect();

        Ok(messages)
    }

    /// Delete a session and its messages (cascade via FK).
    pub async fn delete_session(&self, session_id: Uuid) -> anyhow::Result<()> {
        // Delete messages first (in case FK has no CASCADE)
        sqlx::query("DELETE FROM chat_messages WHERE session_id = $1")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM chat_sessions WHERE id = $1")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
