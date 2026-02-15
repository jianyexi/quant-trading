use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Maximum number of log entries kept in memory.
const MAX_ENTRIES: usize = 2000;

#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub duration_ms: u64,
    /// Short summary or error message
    pub message: String,
    /// Optional detail (e.g. full error, stderr, response body snippet)
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// Thread-safe in-memory ring-buffer for API logs.
#[derive(Debug)]
pub struct LogStore {
    inner: Mutex<Inner>,
}

#[derive(Debug)]
struct Inner {
    entries: VecDeque<LogEntry>,
    next_id: u64,
}

impl LogStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: VecDeque::with_capacity(MAX_ENTRIES),
                next_id: 1,
            }),
        }
    }

    /// Push a new log entry. Auto-assigns ID and evicts oldest if full.
    pub fn push(&self, level: LogLevel, method: &str, path: &str, status: u16, duration_ms: u64, message: &str, detail: Option<String>) {
        let mut inner = self.inner.lock().unwrap();
        let id = inner.next_id;
        inner.next_id += 1;
        if inner.entries.len() >= MAX_ENTRIES {
            inner.entries.pop_front();
        }
        inner.entries.push_back(LogEntry {
            id,
            timestamp: Utc::now(),
            level,
            method: method.to_string(),
            path: path.to_string(),
            status,
            duration_ms,
            message: message.to_string(),
            detail,
        });
    }

    /// Query logs with optional filters. Returns newest first.
    pub fn query(&self, level: Option<LogLevel>, path_contains: Option<&str>, limit: usize) -> Vec<LogEntry> {
        let inner = self.inner.lock().unwrap();
        inner.entries.iter().rev()
            .filter(|e| {
                if let Some(lvl) = level {
                    if e.level != lvl { return false; }
                }
                if let Some(pc) = path_contains {
                    if !e.path.contains(pc) { return false; }
                }
                true
            })
            .take(limit)
            .cloned()
            .collect()
    }

    /// Summary counts by level
    pub fn summary(&self) -> (usize, usize, usize) {
        let inner = self.inner.lock().unwrap();
        let mut info = 0usize;
        let mut warn = 0usize;
        let mut error = 0usize;
        for e in inner.entries.iter() {
            match e.level {
                LogLevel::Info => info += 1,
                LogLevel::Warn => warn += 1,
                LogLevel::Error => error += 1,
            }
        }
        (info, warn, error)
    }

    pub fn count(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    /// Clear all entries
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.clear();
    }
}
