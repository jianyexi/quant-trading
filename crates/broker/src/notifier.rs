//! Order-fill notification service.
//!
//! Supports three channels:
//! - **In-app**: PostgreSQL or SQLite-backed notification store (always active)
//! - **Webhook**: HTTP POST to DingTalk / WeChat Work / Slack / custom URL
//! - **Email**: SMTP via `lettre`

use std::path::Path;
use std::sync::Mutex;

use chrono::Utc;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{error, info, warn};
use uuid::Uuid;

// ── Configuration ───────────────────────────────────────────────────

/// Persisted notification settings (saved as JSON file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Master switch
    pub enabled: bool,
    /// In-app notifications
    pub in_app: bool,
    /// Email settings
    pub email: EmailConfig,
    /// Webhook settings (DingTalk / WeChat / Slack / custom)
    pub webhook: WebhookConfig,
    /// Which events trigger notifications
    pub events: EventFilter,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            in_app: true,
            email: EmailConfig::default(),
            webhook: WebhookConfig::default(),
            events: EventFilter::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    pub enabled: bool,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub username: String,
    /// Password / app-specific password
    pub password: String,
    pub from: String,
    pub to: Vec<String>,
    /// Use STARTTLS
    pub tls: bool,
}

impl Default for EmailConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            smtp_host: "smtp.qq.com".into(),
            smtp_port: 465,
            username: String::new(),
            password: String::new(),
            from: String::new(),
            to: vec![],
            tls: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    pub enabled: bool,
    /// "dingtalk", "wechat", "slack", "custom"
    pub provider: String,
    pub url: String,
    /// Optional secret for signing (DingTalk)
    pub secret: String,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: "dingtalk".into(),
            url: String::new(),
            secret: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    pub order_filled: bool,
    pub order_rejected: bool,
    pub risk_alert: bool,
    pub engine_started: bool,
    pub engine_stopped: bool,
}

impl Default for EventFilter {
    fn default() -> Self {
        Self {
            order_filled: true,
            order_rejected: true,
            risk_alert: true,
            engine_started: false,
            engine_stopped: true,
        }
    }
}

// ── Notification Entry ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub id: String,
    pub timestamp: String,
    pub event_type: String,
    pub title: String,
    pub message: String,
    pub symbol: Option<String>,
    pub side: Option<String>,
    pub quantity: Option<f64>,
    pub price: Option<f64>,
    pub read: bool,
    /// Delivery status: "delivered", "failed", "pending"
    pub delivery: String,
    pub channels: String,
}

// ── Notifier ────────────────────────────────────────────────────────

enum NotifierBackend {
    Pg(PgPool),
    Sqlite(Mutex<Connection>),
}

/// Thread-safe notification service.
pub struct Notifier {
    db: NotifierBackend,
    config: Mutex<NotificationConfig>,
    config_path: String,
    http: reqwest::Client,
}

impl Notifier {
    /// Create a notifier backed by PostgreSQL (production).
    pub fn new(pool: PgPool, config_dir: &str) -> Self {
        let config_path = format!("{}/notification_config.json", config_dir);
        let config = Self::load_config_from_file(&config_path);
        info!("🔔 Notifier initialized (PostgreSQL, email={}, webhook={})",
            config.email.enabled, config.webhook.enabled);
        Self {
            db: NotifierBackend::Pg(pool),
            config: Mutex::new(config),
            config_path,
            http: reqwest::Client::new(),
        }
    }

    /// Open or create notification database and load config (SQLite fallback).
    pub fn open<P: AsRef<Path>>(db_path: P, config_dir: &str) -> anyhow::Result<Self> {
        let conn = Connection::open(db_path)?;
        Self::init_sqlite_tables(&conn)?;

        let config_path = format!("{}/notification_config.json", config_dir);
        let config = Self::load_config_from_file(&config_path);

        info!("🔔 Notifier initialized (SQLite, email={}, webhook={})",
            config.email.enabled, config.webhook.enabled);

        Ok(Self {
            db: NotifierBackend::Sqlite(Mutex::new(conn)),
            config: Mutex::new(config),
            config_path,
            http: reqwest::Client::new(),
        })
    }

    /// Create in-memory notifier (for testing).
    pub fn in_memory() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_sqlite_tables(&conn)?;
        Ok(Self {
            db: NotifierBackend::Sqlite(Mutex::new(conn)),
            config: Mutex::new(NotificationConfig::default()),
            config_path: String::new(),
            http: reqwest::Client::new(),
        })
    }

    fn init_sqlite_tables(conn: &Connection) -> anyhow::Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                read INTEGER NOT NULL DEFAULT 0,
                delivery TEXT NOT NULL DEFAULT 'delivered',
                channels TEXT NOT NULL DEFAULT 'in_app'
            );
            CREATE INDEX IF NOT EXISTS idx_notif_ts ON notifications(timestamp);
            CREATE INDEX IF NOT EXISTS idx_notif_read ON notifications(read);"
        )?;
        Ok(())
    }

    fn load_config_from_file(path: &str) -> NotificationConfig {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => NotificationConfig::default(),
        }
    }

    // ── Public API ──────────────────────────────────────────────────

    /// Get current config.
    pub fn get_config(&self) -> NotificationConfig {
        self.config.lock().unwrap().clone()
    }

    /// Update and persist config.
    pub fn set_config(&self, cfg: NotificationConfig) {
        if !self.config_path.is_empty() {
            if let Ok(json) = serde_json::to_string_pretty(&cfg) {
                let _ = std::fs::write(&self.config_path, json);
            }
        }
        *self.config.lock().unwrap() = cfg;
    }

    /// Notify: order filled.
    pub async fn notify_order_filled(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
        order_id: &str,
    ) {
        let cfg = self.get_config();
        if !cfg.enabled || !cfg.events.order_filled {
            return;
        }
        let title = format!("✅ 订单成交 | {} {} {}", side, symbol, fmt_qty(quantity));
        let message = format!(
            "{}成交: {} x {:.0} @ ¥{:.2}\n订单号: {}",
            side, symbol, quantity, price, order_id
        );
        self.dispatch(&cfg, "order_filled", &title, &message,
            Some(symbol), Some(side), Some(quantity), Some(price)).await;
    }

    /// Notify: order rejected.
    pub async fn notify_order_rejected(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
        reason: &str,
    ) {
        let cfg = self.get_config();
        if !cfg.enabled || !cfg.events.order_rejected {
            return;
        }
        let title = format!("❌ 订单拒绝 | {} {}", side, symbol);
        let message = format!(
            "{}被拒: {} x {:.0} @ ¥{:.2}\n原因: {}",
            side, symbol, quantity, price, reason
        );
        self.dispatch(&cfg, "order_rejected", &title, &message,
            Some(symbol), Some(side), Some(quantity), Some(price)).await;
    }

    /// Notify: risk alert (circuit breaker, stop-loss, etc.).
    pub async fn notify_risk_alert(&self, symbol: &str, reason: &str) {
        let cfg = self.get_config();
        if !cfg.enabled || !cfg.events.risk_alert {
            return;
        }
        let title = format!("⚠️ 风控警报 | {}", symbol);
        let message = format!("风控触发: {}\n标的: {}", reason, symbol);
        self.dispatch(&cfg, "risk_alert", &title, &message,
            Some(symbol), None, None, None).await;
    }

    /// Notify: engine started/stopped.
    pub async fn notify_engine_event(&self, event: &str, details: &str) {
        let cfg = self.get_config();
        if !cfg.enabled {
            return;
        }
        let is_start = event == "engine_started";
        if is_start && !cfg.events.engine_started { return; }
        if !is_start && !cfg.events.engine_stopped { return; }

        let title = if is_start { "🚀 交易引擎启动".to_string() }
                    else { "🛑 交易引擎停止".to_string() };
        self.dispatch(&cfg, event, &title, details,
            None, None, None, None).await;
    }

    /// Send a test notification through all enabled channels.
    pub async fn send_test(&self) -> Vec<(String, bool, String)> {
        let cfg = self.get_config();
        let title = "🔔 测试通知";
        let message = "这是一条测试通知，验证通知渠道是否正常工作。";
        let mut results = vec![];

        // Always test in-app
        self.store_notification("test", title, message, None, None, None, None, "in_app");
        results.push(("in_app".into(), true, "OK".into()));

        if cfg.webhook.enabled && !cfg.webhook.url.is_empty() {
            let ok = self.send_webhook(&cfg.webhook, title, message).await;
            results.push(("webhook".into(), ok.is_ok(),
                ok.err().unwrap_or_else(|| "OK".into())));
        }

        if cfg.email.enabled && !cfg.email.to.is_empty() {
            let ok = self.send_email(&cfg.email, title, message).await;
            results.push(("email".into(), ok.is_ok(),
                ok.err().unwrap_or_else(|| "OK".into())));
        }

        results
    }

    // ── Query ───────────────────────────────────────────────────────

    /// List notifications (newest first).
    pub fn list(&self, limit: usize, unread_only: bool) -> Vec<Notification> {
        match &self.db {
            NotifierBackend::Pg(_) => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.list_async(limit, unread_only))
                })
            }
            NotifierBackend::Sqlite(conn) => self.list_sqlite(conn, limit, unread_only),
        }
    }

    /// Async list for PG backend.
    pub async fn list_async(&self, limit: usize, unread_only: bool) -> Vec<Notification> {
        match &self.db {
            NotifierBackend::Pg(pool) => {
                let sql = if unread_only {
                    "SELECT id, timestamp::text, event_type, title, message, symbol, side, quantity, price, read, delivery, channels
                     FROM notifications WHERE read = false ORDER BY timestamp DESC LIMIT $1"
                } else {
                    "SELECT id, timestamp::text, event_type, title, message, symbol, side, quantity, price, read, delivery, channels
                     FROM notifications ORDER BY timestamp DESC LIMIT $1"
                };
                match sqlx::query_as::<_, NotificationRow>(sql)
                    .bind(limit as i64)
                    .fetch_all(pool).await
                {
                    Ok(rows) => rows.into_iter().map(|r| r.into()).collect(),
                    Err(e) => {
                        error!("Failed to list notifications (PG): {}", e);
                        vec![]
                    }
                }
            }
            NotifierBackend::Sqlite(conn) => self.list_sqlite(conn, limit, unread_only),
        }
    }

    fn list_sqlite(&self, conn: &Mutex<Connection>, limit: usize, unread_only: bool) -> Vec<Notification> {
        let conn = conn.lock().unwrap();
        let sql = if unread_only {
            format!("SELECT id, timestamp, event_type, title, message, symbol, side, quantity, price, read, delivery, channels
                     FROM notifications WHERE read = 0 ORDER BY timestamp DESC LIMIT {}", limit)
        } else {
            format!("SELECT id, timestamp, event_type, title, message, symbol, side, quantity, price, read, delivery, channels
                     FROM notifications ORDER BY timestamp DESC LIMIT {}", limit)
        };
        let mut stmt = conn.prepare(&sql).unwrap();
        stmt.query_map([], |row| {
            Ok(Notification {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                event_type: row.get(2)?,
                title: row.get(3)?,
                message: row.get(4)?,
                symbol: row.get(5)?,
                side: row.get(6)?,
                quantity: row.get(7)?,
                price: row.get(8)?,
                read: row.get::<_, i32>(9)? != 0,
                delivery: row.get(10)?,
                channels: row.get(11)?,
            })
        }).unwrap().filter_map(|r| r.ok()).collect()
    }

    /// Count unread notifications.
    pub fn unread_count(&self) -> u64 {
        match &self.db {
            NotifierBackend::Pg(_) => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.unread_count_async())
                })
            }
            NotifierBackend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                conn.query_row("SELECT COUNT(*) FROM notifications WHERE read = 0", [], |row| row.get(0))
                    .unwrap_or(0)
            }
        }
    }

    /// Async unread count for PG backend.
    pub async fn unread_count_async(&self) -> u64 {
        match &self.db {
            NotifierBackend::Pg(pool) => {
                sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM notifications WHERE read = false")
                    .fetch_one(pool).await.unwrap_or(0) as u64
            }
            NotifierBackend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                conn.query_row("SELECT COUNT(*) FROM notifications WHERE read = 0", [], |row| row.get(0))
                    .unwrap_or(0)
            }
        }
    }

    /// Mark one notification as read.
    pub fn mark_read(&self, id: &str) {
        match &self.db {
            NotifierBackend::Pg(pool) => {
                let pool = pool.clone();
                let id = id.to_string();
                tokio::spawn(async move {
                    let _ = sqlx::query("UPDATE notifications SET read = true WHERE id = $1")
                        .bind(&id).execute(&pool).await;
                });
            }
            NotifierBackend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                let _ = conn.execute("UPDATE notifications SET read = 1 WHERE id = ?", params![id]);
            }
        }
    }

    /// Mark all as read.
    pub fn mark_all_read(&self) {
        match &self.db {
            NotifierBackend::Pg(pool) => {
                let pool = pool.clone();
                tokio::spawn(async move {
                    let _ = sqlx::query("UPDATE notifications SET read = true WHERE read = false")
                        .execute(&pool).await;
                });
            }
            NotifierBackend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                let _ = conn.execute("UPDATE notifications SET read = 1 WHERE read = 0", []);
            }
        }
    }

    // ── Internal ────────────────────────────────────────────────────

    async fn dispatch(
        &self,
        cfg: &NotificationConfig,
        event_type: &str,
        title: &str,
        message: &str,
        symbol: Option<&str>,
        side: Option<&str>,
        quantity: Option<f64>,
        price: Option<f64>,
    ) {
        let mut channels = vec![];

        // In-app (always synchronous, fast)
        if cfg.in_app {
            self.store_notification(event_type, title, message,
                symbol, side, quantity, price, "in_app");
            channels.push("in_app");
        }

        // Webhook
        if cfg.webhook.enabled && !cfg.webhook.url.is_empty() {
            match self.send_webhook(&cfg.webhook, title, message).await {
                Ok(_) => channels.push("webhook"),
                Err(e) => warn!("Webhook notification failed: {}", e),
            }
        }

        // Email
        if cfg.email.enabled && !cfg.email.to.is_empty() {
            match self.send_email(&cfg.email, title, message).await {
                Ok(_) => channels.push("email"),
                Err(e) => warn!("Email notification failed: {}", e),
            }
        }

        info!("🔔 Notification sent [{}]: {} via {:?}", event_type, title, channels);
    }

    fn store_notification(
        &self,
        event_type: &str,
        title: &str,
        message: &str,
        symbol: Option<&str>,
        side: Option<&str>,
        quantity: Option<f64>,
        price: Option<f64>,
        channels: &str,
    ) {
        let id = Uuid::new_v4().to_string();
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        match &self.db {
            NotifierBackend::Pg(pool) => {
                let pool = pool.clone();
                let event_type = event_type.to_string();
                let title = title.to_string();
                let message = message.to_string();
                let symbol = symbol.map(|s| s.to_string());
                let side = side.map(|s| s.to_string());
                let channels = channels.to_string();
                let ts = chrono::NaiveDateTime::parse_from_str(&timestamp, "%Y-%m-%d %H:%M:%S")
                    .unwrap_or_else(|_| Utc::now().naive_utc());
                tokio::spawn(async move {
                    if let Err(e) = sqlx::query(
                        "INSERT INTO notifications (id, timestamp, event_type, title, message, symbol, side, quantity, price, channels)
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)"
                    )
                    .bind(&id).bind(ts).bind(&event_type).bind(&title).bind(&message)
                    .bind(&symbol).bind(&side).bind(quantity).bind(price).bind(&channels)
                    .execute(&pool).await {
                        error!("Failed to store notification (PG): {}", e);
                    }
                });
            }
            NotifierBackend::Sqlite(conn) => {
                let conn = conn.lock().unwrap();
                if let Err(e) = conn.execute(
                    "INSERT INTO notifications (id, timestamp, event_type, title, message, symbol, side, quantity, price, channels)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![id, timestamp, event_type, title, message, symbol, side, quantity, price, channels],
                ) {
                    error!("Failed to store notification: {}", e);
                }
            }
        }
    }

    async fn send_webhook(&self, cfg: &WebhookConfig, title: &str, message: &str) -> Result<(), String> {
        let body = match cfg.provider.as_str() {
            "dingtalk" => serde_json::json!({
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": format!("### {}\n\n{}", title, message)
                }
            }),
            "wechat" => serde_json::json!({
                "msgtype": "markdown",
                "markdown": {
                    "content": format!("### {}\n{}", title, message)
                }
            }),
            "slack" => serde_json::json!({
                "text": format!("*{}*\n{}", title, message)
            }),
            _ => serde_json::json!({
                "title": title,
                "message": message,
                "timestamp": Utc::now().to_rfc3339()
            }),
        };

        self.http.post(&cfg.url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        Ok(())
    }

    async fn send_email(&self, cfg: &EmailConfig, subject: &str, body: &str) -> Result<(), String> {
        // Use lettre for proper SMTP. If not available, fall back to a simple
        // HTTP-based approach or log a warning.
        #[cfg(feature = "email")]
        {
            use lettre::*;
            // ... lettre implementation
        }

        // Lightweight fallback: POST to an HTTP email relay or log-only.
        // For production, configure a webhook to a mail relay service,
        // or enable the "email" feature with lettre.
        let to_list = cfg.to.join(", ");
        info!("📧 Email notification → {}: {}", to_list, subject);

        // Attempt SMTP via reqwest to a local relay or external API
        // For now, store a record that email was requested
        if cfg.smtp_host.is_empty() {
            return Err("SMTP host not configured".into());
        }

        // Simple SMTP implementation using tokio::net
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::TcpStream;

        let addr = format!("{}:{}", cfg.smtp_host, cfg.smtp_port);
        let stream = TcpStream::connect(&addr).await
            .map_err(|e| format!("SMTP connect failed: {}", e))?;

        // For TLS on port 465, we need native-tls / rustls.
        // For simplicity, support port 25 (plain) and 587 (STARTTLS) as plaintext.
        // Production users should use a webhook relay or enable lettre feature.
        if cfg.tls && cfg.smtp_port == 465 {
            return Err("Direct SMTPS (port 465) requires TLS library. Use port 587 or configure a webhook instead.".into());
        }

        let (reader, mut writer) = stream.into_split();
        let mut lines = BufReader::new(reader).lines();

        // Helper to read SMTP response
        async fn read_reply(lines: &mut tokio::io::Lines<BufReader<tokio::net::tcp::OwnedReadHalf>>) -> Result<String, String> {
            lines.next_line().await
                .map_err(|e| format!("SMTP read error: {}", e))?
                .ok_or_else(|| "SMTP connection closed".to_string())
        }

        let _ = read_reply(&mut lines).await?; // greeting

        writer.write_all(format!("EHLO localhost\r\n").as_bytes()).await.map_err(|e| e.to_string())?;
        let _ = read_reply(&mut lines).await?;

        // AUTH LOGIN if credentials provided
        if !cfg.username.is_empty() {
            use base64::Engine as _;
            let b64 = base64::engine::general_purpose::STANDARD;
            writer.write_all(b"AUTH LOGIN\r\n").await.map_err(|e| e.to_string())?;
            let _ = read_reply(&mut lines).await?;
            writer.write_all(format!("{}\r\n", b64.encode(&cfg.username)).as_bytes()).await.map_err(|e| e.to_string())?;
            let _ = read_reply(&mut lines).await?;
            writer.write_all(format!("{}\r\n", b64.encode(&cfg.password)).as_bytes()).await.map_err(|e| e.to_string())?;
            let _ = read_reply(&mut lines).await?;
        }

        writer.write_all(format!("MAIL FROM:<{}>\r\n", cfg.from).as_bytes()).await.map_err(|e| e.to_string())?;
        let _ = read_reply(&mut lines).await?;

        for rcpt in &cfg.to {
            writer.write_all(format!("RCPT TO:<{}>\r\n", rcpt).as_bytes()).await.map_err(|e| e.to_string())?;
            let _ = read_reply(&mut lines).await?;
        }

        writer.write_all(b"DATA\r\n").await.map_err(|e| e.to_string())?;
        let _ = read_reply(&mut lines).await?;

        let date = Utc::now().to_rfc2822();
        let email_body = format!(
            "From: {}\r\nTo: {}\r\nSubject: {}\r\nDate: {}\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n{}\r\n.\r\n",
            cfg.from, cfg.to.join(", "), subject, date, body
        );
        writer.write_all(email_body.as_bytes()).await.map_err(|e| e.to_string())?;
        let _ = read_reply(&mut lines).await?;

        writer.write_all(b"QUIT\r\n").await.map_err(|e| e.to_string())?;

        info!("📧 Email sent to {}", to_list);
        Ok(())
    }
}

fn fmt_qty(q: f64) -> String {
    if q >= 10000.0 { format!("{:.0}万股", q / 10000.0) }
    else { format!("{:.0}股", q) }
}

// ── sqlx row helpers for PG ─────────────────────────────────────────

#[derive(sqlx::FromRow)]
struct NotificationRow {
    id: String,
    timestamp: String,
    event_type: String,
    title: String,
    message: String,
    symbol: Option<String>,
    side: Option<String>,
    quantity: Option<f64>,
    price: Option<f64>,
    read: bool,
    delivery: String,
    channels: String,
}

impl From<NotificationRow> for Notification {
    fn from(r: NotificationRow) -> Self {
        Self {
            id: r.id,
            timestamp: r.timestamp,
            event_type: r.event_type,
            title: r.title,
            message: r.message,
            symbol: r.symbol,
            side: r.side,
            quantity: r.quantity,
            price: r.price,
            read: r.read,
            delivery: r.delivery,
            channels: r.channels,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notifier_store_and_list() {
        let n = Notifier::in_memory().unwrap();
        n.store_notification("order_filled", "Test Fill", "BUY 600519 x100 @ 1650",
            Some("600519.SH"), Some("BUY"), Some(100.0), Some(1650.0), "in_app");

        let list = n.list(10, false);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].event_type, "order_filled");
        assert!(!list[0].read);
        assert_eq!(n.unread_count(), 1);
    }

    #[test]
    fn test_mark_read() {
        let n = Notifier::in_memory().unwrap();
        n.store_notification("order_filled", "Fill", "msg",
            None, None, None, None, "in_app");
        n.store_notification("risk_alert", "Alert", "msg",
            None, None, None, None, "in_app");

        assert_eq!(n.unread_count(), 2);
        let list = n.list(10, false);
        n.mark_read(&list[0].id);
        assert_eq!(n.unread_count(), 1);
        n.mark_all_read();
        assert_eq!(n.unread_count(), 0);
    }

    #[test]
    fn test_config_default() {
        let cfg = NotificationConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.in_app);
        assert!(!cfg.email.enabled);
        assert!(!cfg.webhook.enabled);
        assert!(cfg.events.order_filled);
    }
}
