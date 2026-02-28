//! Persistent task store for long-running background jobs.
//!
//! Tracks ML training, factor mining, GP evolution, etc.
//! Uses SQLite so tasks survive server restarts and browser refreshes.

use std::sync::Mutex;

use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Task Types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

impl TaskStatus {
    fn from_str(s: &str) -> Self {
        match s {
            "running" => Self::Running,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            _ => Self::Pending,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: String,
    pub task_type: String,
    pub status: TaskStatus,
    pub created_at: String,
    pub updated_at: String,
    pub progress: Option<String>,
    pub result: Option<String>,
    pub error: Option<String>,
}

// ── Task Store ─────────────────────────────────────────────────────

pub struct TaskStore {
    conn: Mutex<Connection>,
}

impl TaskStore {
    /// Open (or create) the task database.
    pub fn open(path: &str) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;
        let store = Self { conn: Mutex::new(conn) };
        store.init_tables()?;
        tracing::info!("📋 TaskStore initialized ({})", path);
        Ok(store)
    }

    fn init_tables(&self) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                progress TEXT,
                result TEXT,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);"
        )?;
        Ok(())
    }

    /// Create a new task. Returns the task ID.
    pub fn create(&self, task_type: &str) -> String {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO tasks (id, task_type, status, created_at, updated_at) VALUES (?1, ?2, 'running', ?3, ?3)",
            params![id, task_type, now],
        );
        id
    }

    /// Update task progress message.
    pub fn set_progress(&self, id: &str, progress: &str) {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "UPDATE tasks SET progress = ?1, updated_at = ?2 WHERE id = ?3",
            params![progress, now, id],
        );
    }

    /// Mark task completed with result.
    pub fn complete(&self, id: &str, result: &str) {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "UPDATE tasks SET status = 'completed', result = ?1, updated_at = ?2 WHERE id = ?3",
            params![result, now, id],
        );
    }

    /// Mark task failed with error.
    pub fn fail(&self, id: &str, error: &str) {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "UPDATE tasks SET status = 'failed', error = ?1, updated_at = ?2 WHERE id = ?3",
            params![error, now, id],
        );
    }

    /// Cancel a running task (mark as failed with cancellation reason).
    pub fn cancel(&self, id: &str) -> bool {
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        let rows = conn.execute(
            "UPDATE tasks SET status = 'failed', error = 'Cancelled by user', updated_at = ?1 WHERE id = ?2 AND status = 'running'",
            params![now, id],
        ).unwrap_or(0);
        rows > 0
    }

    /// Delete a task record entirely.
    pub fn delete(&self, id: &str) -> bool {
        let conn = self.conn.lock().unwrap();
        let rows = conn.execute("DELETE FROM tasks WHERE id = ?1", params![id]).unwrap_or(0);
        rows > 0
    }

    /// Mark stale running tasks as failed (e.g. from server crash).
    pub fn cleanup_stale(&self, max_age_minutes: i64) {
        let cutoff = (Utc::now() - chrono::Duration::minutes(max_age_minutes)).to_rfc3339();
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let rows = conn.execute(
            "UPDATE tasks SET status = 'failed', error = 'Stale: server restarted', updated_at = ?1 WHERE status = 'running' AND updated_at < ?2",
            params![now, cutoff],
        ).unwrap_or(0);
        if rows > 0 {
            tracing::warn!("🧹 Cleaned up {} stale running tasks", rows);
        }
    }

    /// Get a single task by ID.
    pub fn get(&self, id: &str) -> Option<TaskRecord> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, task_type, status, created_at, updated_at, progress, result, error FROM tasks WHERE id = ?1",
            params![id],
            |row| Ok(TaskRecord {
                id: row.get(0)?,
                task_type: row.get(1)?,
                status: TaskStatus::from_str(&row.get::<_, String>(2)?),
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
                progress: row.get(5)?,
                result: row.get(6)?,
                error: row.get(7)?,
            }),
        ).ok()
    }

    /// List recent tasks, newest first.
    pub fn list(&self, limit: u32) -> Vec<TaskRecord> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, task_type, status, created_at, updated_at, progress, result, error
             FROM tasks ORDER BY created_at DESC LIMIT ?1"
        ).unwrap();
        stmt.query_map(params![limit], |row| Ok(TaskRecord {
            id: row.get(0)?,
            task_type: row.get(1)?,
            status: TaskStatus::from_str(&row.get::<_, String>(2)?),
            created_at: row.get(3)?,
            updated_at: row.get(4)?,
            progress: row.get(5)?,
            result: row.get(6)?,
            error: row.get(7)?,
        })).unwrap().filter_map(|r| r.ok()).collect()
    }

    /// List tasks that are currently running.
    pub fn list_running(&self) -> Vec<TaskRecord> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, task_type, status, created_at, updated_at, progress, result, error
             FROM tasks WHERE status = 'running' ORDER BY created_at DESC"
        ).unwrap();
        stmt.query_map([], |row| Ok(TaskRecord {
            id: row.get(0)?,
            task_type: row.get(1)?,
            status: TaskStatus::from_str(&row.get::<_, String>(2)?),
            created_at: row.get(3)?,
            updated_at: row.get(4)?,
            progress: row.get(5)?,
            result: row.get(6)?,
            error: row.get(7)?,
        })).unwrap().filter_map(|r| r.ok()).collect()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_lifecycle() {
        let store = TaskStore::open(":memory:").unwrap();

        let id = store.create("ml_retrain");
        let task = store.get(&id).unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.task_type, "ml_retrain");

        store.set_progress(&id, "Training 50%...");
        let task = store.get(&id).unwrap();
        assert_eq!(task.progress.as_deref(), Some("Training 50%..."));

        store.complete(&id, r#"{"auc": 0.85}"#);
        let task = store.get(&id).unwrap();
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(task.result.unwrap().contains("0.85"));
    }

    #[test]
    fn test_task_failure() {
        let store = TaskStore::open(":memory:").unwrap();
        let id = store.create("gp_mining");
        store.fail(&id, "Python not found");
        let task = store.get(&id).unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(task.error.as_deref(), Some("Python not found"));
    }

    #[test]
    fn test_list_and_running() {
        let store = TaskStore::open(":memory:").unwrap();
        let id1 = store.create("task_a");
        let _id2 = store.create("task_b");
        store.complete(&id1, "done");

        let all = store.list(10);
        assert_eq!(all.len(), 2);

        let running = store.list_running();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].task_type, "task_b");
    }
}
