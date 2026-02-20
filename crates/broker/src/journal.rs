//! Trade Journal — persistent storage for all trading activity.
//!
//! Stores signals, orders, fills, rejections, and daily snapshots.
//! Supports PostgreSQL (production) and SQLite in-memory (testing).

use std::path::Path;
use std::sync::Mutex;

use chrono::{NaiveDateTime, Utc};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{error, info};
use uuid::Uuid;

use quant_core::types::OrderSide;

// ── Journal Entry Types ─────────────────────────────────────────────

/// Type of journal entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JournalEntryType {
    Signal,
    OrderSubmitted,
    OrderFilled,
    OrderRejected,
    OrderCancelled,
    RiskRejected,
    DailySnapshot,
    EngineStarted,
    EngineStopped,
}

impl std::fmt::Display for JournalEntryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Signal => write!(f, "signal"),
            Self::OrderSubmitted => write!(f, "order_submitted"),
            Self::OrderFilled => write!(f, "order_filled"),
            Self::OrderRejected => write!(f, "order_rejected"),
            Self::OrderCancelled => write!(f, "order_cancelled"),
            Self::RiskRejected => write!(f, "risk_rejected"),
            Self::DailySnapshot => write!(f, "daily_snapshot"),
            Self::EngineStarted => write!(f, "engine_started"),
            Self::EngineStopped => write!(f, "engine_stopped"),
        }
    }
}

impl JournalEntryType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "signal" => Self::Signal,
            "order_submitted" => Self::OrderSubmitted,
            "order_filled" => Self::OrderFilled,
            "order_rejected" => Self::OrderRejected,
            "order_cancelled" => Self::OrderCancelled,
            "risk_rejected" => Self::RiskRejected,
            "daily_snapshot" => Self::DailySnapshot,
            "engine_started" => Self::EngineStarted,
            "engine_stopped" => Self::EngineStopped,
            _ => Self::Signal,
        }
    }
}

/// A single journal entry recording a trading event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub id: String,
    pub timestamp: String,
    pub entry_type: String,
    pub symbol: String,
    pub side: Option<String>,
    pub quantity: Option<f64>,
    pub price: Option<f64>,
    pub order_id: Option<String>,
    pub status: Option<String>,
    pub reason: Option<String>,
    pub pnl: Option<f64>,
    pub portfolio_value: Option<f64>,
    pub cash: Option<f64>,
    pub details: Option<String>,
}

/// Query filter for journal entries.
#[derive(Debug, Default)]
pub struct JournalQuery {
    pub symbol: Option<String>,
    pub entry_type: Option<String>,
    pub start: Option<String>,
    pub end: Option<String>,
    pub limit: Option<usize>,
}

/// Daily performance snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailySnapshot {
    pub date: String,
    pub portfolio_value: f64,
    pub cash: f64,
    pub positions_count: usize,
    pub daily_pnl: f64,
    pub cumulative_pnl: f64,
    pub total_trades: u64,
}

// ── Journal Store ───────────────────────────────────────────────────

enum Backend {
    Pg(PgPool),
    Sqlite(Mutex<Connection>),
}

/// Thread-safe journal store with PostgreSQL or SQLite backend.
pub struct JournalStore {
    backend: Backend,
}

impl JournalStore {
    /// Create a journal backed by PostgreSQL (production).
    /// Tables are created by sqlx migrations.
    pub fn new(pool: PgPool) -> Self {
        info!("📝 JournalStore initialized (PostgreSQL)");
        Self { backend: Backend::Pg(pool) }
    }

    /// Open (or create) a journal database at the given path (SQLite fallback).
    pub fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;
        let store = Self { backend: Backend::Sqlite(Mutex::new(conn)) };
        store.init_sqlite_tables()?;
        info!("📝 JournalStore initialized (SQLite)");
        Ok(store)
    }

    /// Create an in-memory journal (for testing).
    pub fn in_memory() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { backend: Backend::Sqlite(Mutex::new(conn)) };
        store.init_sqlite_tables()?;
        Ok(store)
    }

    fn init_sqlite_tables(&self) -> anyhow::Result<()> {
        if let Backend::Sqlite(ref conn) = self.backend {
            let conn = conn.lock().unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS journal_entries (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    symbol TEXT NOT NULL DEFAULT '',
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    order_id TEXT,
                    status TEXT,
                    reason TEXT,
                    pnl REAL,
                    portfolio_value REAL,
                    cash REAL,
                    details TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal_entries(timestamp);
                CREATE INDEX IF NOT EXISTS idx_journal_symbol ON journal_entries(symbol);
                CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entries(entry_type);

                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    date TEXT PRIMARY KEY,
                    portfolio_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_count INTEGER NOT NULL DEFAULT 0,
                    daily_pnl REAL NOT NULL DEFAULT 0,
                    cumulative_pnl REAL NOT NULL DEFAULT 0,
                    total_trades INTEGER NOT NULL DEFAULT 0
                );"
            )?;
        }
        Ok(())
    }

    // ── Write Methods ───────────────────────────────────────────────

    /// Record a trading signal.
    pub fn record_signal(
        &self,
        symbol: &str,
        side: &str,
        confidence: f64,
        price: f64,
    ) {
        self.insert_entry(JournalEntryType::Signal, symbol, Some(side), None, Some(price),
            None, None, None, None, None, None, Some(format!("confidence={:.4}", confidence)));
    }

    /// Record an order submission.
    pub fn record_order_submitted(
        &self,
        order_id: Uuid,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
        price: f64,
    ) {
        let side_str = if side == OrderSide::Buy { "BUY" } else { "SELL" };
        self.insert_entry(JournalEntryType::OrderSubmitted, symbol, Some(side_str),
            Some(quantity), Some(price), Some(order_id.to_string()), Some("submitted"),
            None, None, None, None, None);
    }

    /// Record an order fill.
    pub fn record_order_filled(
        &self,
        order_id: Uuid,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
        price: f64,
        commission: f64,
        portfolio_value: f64,
        cash: f64,
    ) {
        let side_str = if side == OrderSide::Buy { "BUY" } else { "SELL" };
        self.insert_entry(JournalEntryType::OrderFilled, symbol, Some(side_str),
            Some(quantity), Some(price), Some(order_id.to_string()), Some("filled"),
            None, None, Some(portfolio_value), Some(cash),
            Some(format!("commission={:.4}", commission)));
    }

    /// Record a risk rejection.
    pub fn record_risk_rejected(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
        price: f64,
        reason: &str,
    ) {
        let side_str = if side == OrderSide::Buy { "BUY" } else { "SELL" };
        self.insert_entry(JournalEntryType::RiskRejected, symbol, Some(side_str),
            Some(quantity), Some(price), None, Some("rejected"),
            Some(reason), None, None, None, None);
    }

    /// Record engine start.
    pub fn record_engine_started(&self, strategy: &str, symbols: &[String]) {
        self.insert_entry(JournalEntryType::EngineStarted, "", None, None, None,
            None, None, None, None, None, None,
            Some(format!("strategy={}, symbols={}", strategy, symbols.join(","))));
    }

    /// Record engine stop.
    pub fn record_engine_stopped(&self, pnl: f64, portfolio_value: f64) {
        self.insert_entry(JournalEntryType::EngineStopped, "", None, None, None,
            None, None, None, Some(pnl), Some(portfolio_value), None, None);
    }

    /// Save a daily snapshot.
    pub fn save_daily_snapshot(&self, snapshot: &DailySnapshot) {
        match &self.backend {
            Backend::Pg(pool) => {
                let pool = pool.clone();
                let snapshot = snapshot.clone();
                tokio::spawn(async move {
                    let ts = NaiveDateTime::parse_from_str(
                        &format!("{} 00:00:00", &snapshot.date), "%Y-%m-%d %H:%M:%S"
                    ).unwrap_or_else(|_| Utc::now().naive_utc());
                    if let Err(e) = sqlx::query(
                        "INSERT INTO daily_snapshots (date, portfolio_value, cash, positions_count, daily_pnl, cumulative_pnl, total_trades)
                         VALUES ($1, $2, $3, $4, $5, $6, $7)
                         ON CONFLICT (date) DO UPDATE SET
                           portfolio_value = EXCLUDED.portfolio_value, cash = EXCLUDED.cash,
                           positions_count = EXCLUDED.positions_count, daily_pnl = EXCLUDED.daily_pnl,
                           cumulative_pnl = EXCLUDED.cumulative_pnl, total_trades = EXCLUDED.total_trades"
                    )
                    .bind(ts.date())
                    .bind(snapshot.portfolio_value)
                    .bind(snapshot.cash)
                    .bind(snapshot.positions_count as i32)
                    .bind(snapshot.daily_pnl)
                    .bind(snapshot.cumulative_pnl)
                    .bind(snapshot.total_trades as i64)
                    .execute(&pool)
                    .await {
                        error!("Failed to save daily snapshot (PG): {}", e);
                    }
                });
            }
            Backend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                if let Err(e) = conn.execute(
                    "INSERT OR REPLACE INTO daily_snapshots (date, portfolio_value, cash, positions_count, daily_pnl, cumulative_pnl, total_trades)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![snapshot.date, snapshot.portfolio_value, snapshot.cash,
                        snapshot.positions_count as i64, snapshot.daily_pnl, snapshot.cumulative_pnl,
                        snapshot.total_trades as i64],
                ) {
                    error!("Failed to save daily snapshot: {}", e);
                }
            }
        }
    }

    fn insert_entry(
        &self,
        entry_type: JournalEntryType,
        symbol: &str,
        side: Option<&str>,
        quantity: Option<f64>,
        price: Option<f64>,
        order_id: Option<String>,
        status: Option<&str>,
        reason: Option<&str>,
        pnl: Option<f64>,
        portfolio_value: Option<f64>,
        cash: Option<f64>,
        details: Option<String>,
    ) {
        let id = Uuid::new_v4().to_string();
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        match &self.backend {
            Backend::Pg(pool) => {
                let pool = pool.clone();
                let entry_type_str = entry_type.to_string();
                let symbol = symbol.to_string();
                let side = side.map(|s| s.to_string());
                let order_id = order_id.clone();
                let status = status.map(|s| s.to_string());
                let reason = reason.map(|s| s.to_string());
                let ts = NaiveDateTime::parse_from_str(&timestamp, "%Y-%m-%d %H:%M:%S")
                    .unwrap_or_else(|_| Utc::now().naive_utc());
                tokio::spawn(async move {
                    if let Err(e) = sqlx::query(
                        "INSERT INTO journal_entries (id, timestamp, entry_type, symbol, side, quantity, price, order_id, status, reason, pnl, portfolio_value, cash, details)
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)"
                    )
                    .bind(&id).bind(ts).bind(&entry_type_str).bind(&symbol).bind(&side)
                    .bind(quantity).bind(price).bind(&order_id).bind(&status).bind(&reason)
                    .bind(pnl).bind(portfolio_value).bind(cash).bind(&details)
                    .execute(&pool)
                    .await {
                        error!("Failed to write journal entry (PG): {}", e);
                    }
                });
            }
            Backend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                if let Err(e) = conn.execute(
                    "INSERT INTO journal_entries (id, timestamp, entry_type, symbol, side, quantity, price, order_id, status, reason, pnl, portfolio_value, cash, details)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                    params![id, timestamp, entry_type.to_string(), symbol, side,
                        quantity, price, order_id, status, reason,
                        pnl, portfolio_value, cash, details],
                ) {
                    error!("Failed to write journal entry: {}", e);
                }
            }
        }
    }

    // ── Read Methods ────────────────────────────────────────────────

    /// Query journal entries with optional filters (sync, SQLite or blocking PG).
    pub fn query(&self, q: &JournalQuery) -> Vec<JournalEntry> {
        match &self.backend {
            Backend::Pg(_) => {
                // For sync callers; async callers should use query_async
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.query_async(q))
                })
            }
            Backend::Sqlite(conn) => self.query_sqlite(conn, q),
        }
    }

    /// Async query for PG backend (called from async handlers).
    pub async fn query_async(&self, q: &JournalQuery) -> Vec<JournalEntry> {
        match &self.backend {
            Backend::Pg(pool) => {
                let mut sql = String::from(
                    "SELECT id, timestamp::text, entry_type, symbol, side, quantity, price, order_id, status, reason, pnl, portfolio_value, cash, details
                     FROM journal_entries WHERE 1=1"
                );
                let mut idx = 1u32;
                let mut binds_str: Vec<String> = vec![];

                if let Some(ref sym) = q.symbol {
                    sql.push_str(&format!(" AND symbol = ${}", idx)); idx += 1;
                    binds_str.push(sym.clone());
                }
                if let Some(ref et) = q.entry_type {
                    sql.push_str(&format!(" AND entry_type = ${}", idx)); idx += 1;
                    binds_str.push(et.clone());
                }
                if let Some(ref start) = q.start {
                    sql.push_str(&format!(" AND timestamp >= ${}::timestamp", idx)); idx += 1;
                    binds_str.push(start.clone());
                }
                if let Some(ref end) = q.end {
                    sql.push_str(&format!(" AND timestamp <= ${}::timestamp", idx)); idx += 1;
                    binds_str.push(end.clone());
                }
                let _ = idx;

                sql.push_str(" ORDER BY timestamp DESC");
                let limit = q.limit.unwrap_or(100) as i64;
                sql.push_str(&format!(" LIMIT {}", limit));

                let mut query = sqlx::query_as::<_, JournalEntryRow>(&sql);
                for b in &binds_str {
                    query = query.bind(b);
                }

                match query.fetch_all(pool).await {
                    Ok(rows) => rows.into_iter().map(|r| r.into()).collect(),
                    Err(e) => {
                        error!("Failed to query journal (PG): {}", e);
                        vec![]
                    }
                }
            }
            Backend::Sqlite(conn) => self.query_sqlite(conn, q),
        }
    }

    fn query_sqlite(&self, conn: &Mutex<Connection>, q: &JournalQuery) -> Vec<JournalEntry> {
        let conn = conn.lock().unwrap();
        let mut sql = String::from(
            "SELECT id, timestamp, entry_type, symbol, side, quantity, price, order_id, status, reason, pnl, portfolio_value, cash, details
             FROM journal_entries WHERE 1=1"
        );
        let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = vec![];

        if let Some(ref sym) = q.symbol {
            sql.push_str(" AND symbol = ?");
            params_vec.push(Box::new(sym.clone()));
        }
        if let Some(ref et) = q.entry_type {
            sql.push_str(" AND entry_type = ?");
            params_vec.push(Box::new(et.clone()));
        }
        if let Some(ref start) = q.start {
            sql.push_str(" AND timestamp >= ?");
            params_vec.push(Box::new(start.clone()));
        }
        if let Some(ref end) = q.end {
            sql.push_str(" AND timestamp <= ?");
            params_vec.push(Box::new(end.clone()));
        }

        sql.push_str(" ORDER BY timestamp DESC");
        let limit = q.limit.unwrap_or(100);
        sql.push_str(&format!(" LIMIT {}", limit));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

        let mut stmt = match conn.prepare(&sql) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to prepare journal query: {}", e);
                return vec![];
            }
        };

        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(JournalEntry {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                entry_type: row.get(2)?,
                symbol: row.get(3)?,
                side: row.get(4)?,
                quantity: row.get(5)?,
                price: row.get(6)?,
                order_id: row.get(7)?,
                status: row.get(8)?,
                reason: row.get(9)?,
                pnl: row.get(10)?,
                portfolio_value: row.get(11)?,
                cash: row.get(12)?,
                details: row.get(13)?,
            })
        });

        match rows {
            Ok(mapped) => mapped.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                error!("Failed to query journal: {}", e);
                vec![]
            }
        }
    }

    /// Get total counts by entry type.
    pub fn stats(&self) -> Vec<(String, u64)> {
        match &self.backend {
            Backend::Pg(_) => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.stats_async())
                })
            }
            Backend::Sqlite(conn) => self.stats_sqlite(conn),
        }
    }

    /// Async stats for PG backend.
    pub async fn stats_async(&self) -> Vec<(String, u64)> {
        match &self.backend {
            Backend::Pg(pool) => {
                match sqlx::query_as::<_, (String, i64)>(
                    "SELECT entry_type, COUNT(*) FROM journal_entries GROUP BY entry_type ORDER BY entry_type"
                ).fetch_all(pool).await {
                    Ok(rows) => rows.into_iter().map(|(t, c)| (t, c as u64)).collect(),
                    Err(e) => {
                        error!("Failed to get stats (PG): {}", e);
                        vec![]
                    }
                }
            }
            Backend::Sqlite(conn) => self.stats_sqlite(conn),
        }
    }

    fn stats_sqlite(&self, conn: &Mutex<Connection>) -> Vec<(String, u64)> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT entry_type, COUNT(*) FROM journal_entries GROUP BY entry_type ORDER BY entry_type"
        ).unwrap();
        stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
        }).unwrap().filter_map(|r| r.ok()).collect()
    }

    /// Get daily snapshots for performance tracking.
    pub fn get_daily_snapshots(&self, limit: usize) -> Vec<DailySnapshot> {
        match &self.backend {
            Backend::Pg(_) => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.get_daily_snapshots_async(limit))
                })
            }
            Backend::Sqlite(conn) => self.get_daily_snapshots_sqlite(conn, limit),
        }
    }

    /// Async daily snapshots for PG backend.
    pub async fn get_daily_snapshots_async(&self, limit: usize) -> Vec<DailySnapshot> {
        match &self.backend {
            Backend::Pg(pool) => {
                match sqlx::query_as::<_, DailySnapshotRow>(
                    "SELECT date::text, portfolio_value, cash, positions_count, daily_pnl, cumulative_pnl, total_trades
                     FROM daily_snapshots ORDER BY date DESC LIMIT $1"
                ).bind(limit as i64).fetch_all(pool).await {
                    Ok(rows) => rows.into_iter().map(|r| r.into()).collect(),
                    Err(e) => {
                        error!("Failed to get snapshots (PG): {}", e);
                        vec![]
                    }
                }
            }
            Backend::Sqlite(conn) => self.get_daily_snapshots_sqlite(conn, limit),
        }
    }

    fn get_daily_snapshots_sqlite(&self, conn: &Mutex<Connection>, limit: usize) -> Vec<DailySnapshot> {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT date, portfolio_value, cash, positions_count, daily_pnl, cumulative_pnl, total_trades
             FROM daily_snapshots ORDER BY date DESC LIMIT ?"
        ).unwrap();
        stmt.query_map(params![limit as i64], |row| {
            Ok(DailySnapshot {
                date: row.get(0)?,
                portfolio_value: row.get(1)?,
                cash: row.get(2)?,
                positions_count: row.get::<_, i64>(3)? as usize,
                daily_pnl: row.get(4)?,
                cumulative_pnl: row.get(5)?,
                total_trades: row.get::<_, i64>(6)? as u64,
            })
        }).unwrap().filter_map(|r| r.ok()).collect()
    }

    /// Count total entries.
    pub fn count(&self) -> u64 {
        match &self.backend {
            Backend::Pg(_) => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.count_async())
                })
            }
            Backend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                conn.query_row("SELECT COUNT(*) FROM journal_entries", [], |row| row.get(0))
                    .unwrap_or(0)
            }
        }
    }

    /// Async count for PG backend.
    pub async fn count_async(&self) -> u64 {
        match &self.backend {
            Backend::Pg(pool) => {
                sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM journal_entries")
                    .fetch_one(pool).await.unwrap_or(0) as u64
            }
            Backend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                conn.query_row("SELECT COUNT(*) FROM journal_entries", [], |row| row.get(0))
                    .unwrap_or(0)
            }
        }
    }
}

// ── sqlx row helpers for PG ─────────────────────────────────────────

#[derive(sqlx::FromRow)]
struct JournalEntryRow {
    id: String,
    timestamp: String,
    entry_type: String,
    symbol: String,
    side: Option<String>,
    quantity: Option<f64>,
    price: Option<f64>,
    order_id: Option<String>,
    status: Option<String>,
    reason: Option<String>,
    pnl: Option<f64>,
    portfolio_value: Option<f64>,
    cash: Option<f64>,
    details: Option<String>,
}

impl From<JournalEntryRow> for JournalEntry {
    fn from(r: JournalEntryRow) -> Self {
        Self {
            id: r.id,
            timestamp: r.timestamp,
            entry_type: r.entry_type,
            symbol: r.symbol,
            side: r.side,
            quantity: r.quantity,
            price: r.price,
            order_id: r.order_id,
            status: r.status,
            reason: r.reason,
            pnl: r.pnl,
            portfolio_value: r.portfolio_value,
            cash: r.cash,
            details: r.details,
        }
    }
}

#[derive(sqlx::FromRow)]
struct DailySnapshotRow {
    date: String,
    portfolio_value: f64,
    cash: f64,
    positions_count: i32,
    daily_pnl: f64,
    cumulative_pnl: f64,
    total_trades: i64,
}

impl From<DailySnapshotRow> for DailySnapshot {
    fn from(r: DailySnapshotRow) -> Self {
        Self {
            date: r.date,
            portfolio_value: r.portfolio_value,
            cash: r.cash,
            positions_count: r.positions_count as usize,
            daily_pnl: r.daily_pnl,
            cumulative_pnl: r.cumulative_pnl,
            total_trades: r.total_trades as u64,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_journal_create_and_query() {
        let store = JournalStore::in_memory().unwrap();

        store.record_signal("600519.SH", "BUY", 0.85, 1650.0);
        store.record_order_submitted(
            Uuid::new_v4(), "600519.SH", OrderSide::Buy, 100.0, 1650.0);
        store.record_risk_rejected(
            "000001.SZ", OrderSide::Buy, 200.0, 12.5, "concentration limit exceeded");

        assert_eq!(store.count(), 3);

        let all = store.query(&JournalQuery::default());
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_journal_query_by_symbol() {
        let store = JournalStore::in_memory().unwrap();

        store.record_signal("600519.SH", "BUY", 0.8, 1650.0);
        store.record_signal("000001.SZ", "SELL", 0.6, 12.5);
        store.record_signal("600519.SH", "SELL", 0.7, 1660.0);

        let q = JournalQuery { symbol: Some("600519.SH".into()), ..Default::default() };
        let results = store.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_journal_query_by_type() {
        let store = JournalStore::in_memory().unwrap();

        store.record_signal("600519.SH", "BUY", 0.8, 1650.0);
        store.record_risk_rejected("600519.SH", OrderSide::Buy, 100.0, 1650.0, "too risky");
        store.record_engine_started("sma_cross", &["600519.SH".into()]);

        let q = JournalQuery { entry_type: Some("risk_rejected".into()), ..Default::default() };
        let results = store.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry_type, "risk_rejected");
    }

    #[test]
    fn test_journal_stats() {
        let store = JournalStore::in_memory().unwrap();

        store.record_signal("A", "BUY", 0.8, 10.0);
        store.record_signal("B", "BUY", 0.7, 20.0);
        store.record_risk_rejected("C", OrderSide::Buy, 100.0, 30.0, "nope");

        let stats = store.stats();
        assert_eq!(stats.len(), 2); // "signal" and "risk_rejected"
    }

    #[test]
    fn test_daily_snapshot() {
        let store = JournalStore::in_memory().unwrap();

        store.save_daily_snapshot(&DailySnapshot {
            date: "2024-12-01".into(),
            portfolio_value: 1_050_000.0,
            cash: 500_000.0,
            positions_count: 3,
            daily_pnl: 5_000.0,
            cumulative_pnl: 50_000.0,
            total_trades: 15,
        });

        let snapshots = store.get_daily_snapshots(10);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].date, "2024-12-01");
        assert!((snapshots[0].portfolio_value - 1_050_000.0).abs() < 0.01);
    }
}
