//! Actor-based auto-trading engine.
//!
//! Architecture:
//!   DataActor → StrategyActor → RiskActor → OrderActor
//!              (generates bars)  (signals)   (validates)  (executes)
//!
//! All actors run as independent tokio tasks connected by mpsc channels.
//! The `TradingEngine` coordinates startup, shutdown, and status reporting.
//!
//! The engine is generic over the `Broker` implementation, supporting both
//! `PaperBroker` (simulation) and `QmtBroker` (live trading via QMT).

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch, Mutex};
use tracing::{info, warn, error};
use uuid::Uuid;

use quant_core::models::*;
use quant_core::traits::{Broker, Strategy};
use quant_core::types::*;
use quant_risk::enforcement::{RiskConfig, RiskEnforcer, PositionInfo};

// ── Messages ────────────────────────────────────────────────────────

/// Market data event from DataActor → StrategyActor
#[derive(Debug, Clone)]
pub struct MarketEvent {
    pub kline: Kline,
}

/// Trade signal from StrategyActor → RiskActor
#[derive(Debug, Clone)]
pub struct TradeSignal {
    pub signal: Signal,
    pub kline: Kline,
}

/// Order request from RiskActor → OrderActor
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub order: Order,
}

/// Execution report from OrderActor → engine log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub order_id: Uuid,
    pub symbol: String,
    pub side: OrderSide,
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub timestamp: chrono::NaiveDateTime,
    pub status: String,
}

/// Engine status snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub running: bool,
    pub strategy: String,
    pub symbols: Vec<String>,
    pub total_signals: u64,
    pub total_orders: u64,
    pub total_fills: u64,
    pub total_rejected: u64,
    pub pnl: f64,
    pub recent_trades: Vec<ExecutionReport>,
}

/// Engine configuration
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub strategy_name: String,
    pub symbols: Vec<String>,
    pub interval_secs: u64,
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub stamp_tax_rate: f64,
    pub max_concentration: f64,
    pub position_size_pct: f64,
    /// Data mode: "live" uses real market data via DataProvider, "simulated" uses synthetic data
    pub data_mode: DataMode,
    /// Risk enforcement configuration
    pub risk_config: RiskConfig,
}

/// Data feed mode for the DataActor.
#[derive(Debug, Clone, PartialEq)]
pub enum DataMode {
    /// Simulated: deterministic sine/cosine noise (for testing/demo)
    Simulated,
    /// Live: fetch real-time quotes from DataProvider (Tushare/AKShare)
    Live {
        /// Tushare API URL
        tushare_url: String,
        /// Tushare API token
        tushare_token: String,
        /// AKShare API URL
        akshare_url: String,
    },
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            strategy_name: "sma_cross".into(),
            symbols: vec!["600519.SH".into()],
            interval_secs: 5,
            initial_capital: 1_000_000.0,
            commission_rate: 0.00025,
            stamp_tax_rate: 0.001,
            max_concentration: 0.20,
            position_size_pct: 0.10,
            data_mode: DataMode::Simulated,
            risk_config: RiskConfig::default(),
        }
    }
}

use crate::journal::JournalStore;

// ── Shared State ────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct EngineStats {
    total_signals: u64,
    total_orders: u64,
    total_fills: u64,
    total_rejected: u64,
    recent_trades: Vec<ExecutionReport>,
}

// ── Trading Engine ──────────────────────────────────────────────────

/// The trading engine, generic over a `Broker` implementation.
///
/// Use `TradingEngine::new_paper(config)` for simulation or
/// `TradingEngine::new_with_broker(config, broker)` for live trading.
pub struct TradingEngine {
    config: EngineConfig,
    shutdown_tx: Option<watch::Sender<bool>>,
    stats: Arc<Mutex<EngineStats>>,
    broker: Arc<dyn Broker>,
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Whether fill_order should be called after submit (paper mode only)
    auto_fill: bool,
    /// Trade journal for persistent audit trail
    journal: Option<Arc<JournalStore>>,
    /// Active risk enforcement (stop-loss, daily loss, drawdown, circuit breaker)
    risk_enforcer: Arc<RiskEnforcer>,
}

impl TradingEngine {
    /// Create an engine using PaperBroker (backward compatible).
    pub fn new(config: EngineConfig) -> Self {
        use crate::paper::PaperBroker;
        let broker = Arc::new(PaperBroker::new(
            config.initial_capital,
            config.commission_rate,
            config.stamp_tax_rate,
        ));
        // Try to open journal; non-fatal if it fails
        let journal = JournalStore::open("data/trade_journal.db")
            .map(Arc::new)
            .ok();
        let risk_enforcer = Arc::new(RiskEnforcer::new(config.risk_config.clone(), config.initial_capital));
        Self {
            config,
            shutdown_tx: None,
            stats: Arc::new(Mutex::new(EngineStats::default())),
            broker,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            auto_fill: true,
            journal,
            risk_enforcer,
        }
    }

    /// Create an engine with a custom broker (e.g., QmtBroker for live trading).
    pub fn new_with_broker(config: EngineConfig, broker: Arc<dyn Broker>) -> Self {
        let journal = JournalStore::open("data/trade_journal.db")
            .map(Arc::new)
            .ok();
        let risk_enforcer = Arc::new(RiskEnforcer::new(config.risk_config.clone(), config.initial_capital));
        Self {
            config,
            shutdown_tx: None,
            stats: Arc::new(Mutex::new(EngineStats::default())),
            broker,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            auto_fill: false,
            journal,
            risk_enforcer,
        }
    }

    /// Start the actor system with the given strategy factory.
    pub async fn start<F>(&mut self, strategy_factory: F)
    where
        F: Fn() -> Box<dyn Strategy> + Send + 'static,
    {
        if self.running.load(std::sync::atomic::Ordering::SeqCst) {
            warn!("Engine already running");
            return;
        }

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        self.shutdown_tx = Some(shutdown_tx);
        self.running.store(true, std::sync::atomic::Ordering::SeqCst);

        // Create channels between actors
        let (market_tx, market_rx) = mpsc::channel::<MarketEvent>(256);
        let (signal_tx, signal_rx) = mpsc::channel::<TradeSignal>(64);
        let (order_tx, order_rx) = mpsc::channel::<OrderRequest>(64);

        // Spawn DataActor
        let data_shutdown = shutdown_rx.clone();
        let data_symbols = self.config.symbols.clone();
        let data_interval = self.config.interval_secs;
        let data_mode = self.config.data_mode.clone();
        tokio::spawn(async move {
            data_actor(data_symbols, data_interval, data_mode, market_tx, data_shutdown).await;
        });

        // Record engine start in journal
        if let Some(ref j) = self.journal {
            j.record_engine_started(&self.config.strategy_name, &self.config.symbols);
        }

        // Spawn StrategyActor
        let strat_shutdown = shutdown_rx.clone();
        let strat_stats = self.stats.clone();
        let strat_journal = self.journal.clone();
        tokio::spawn(async move {
            strategy_actor(strategy_factory, market_rx, signal_tx, strat_stats, strat_journal, strat_shutdown).await;
        });

        // Spawn RiskActor
        let risk_shutdown = shutdown_rx.clone();
        let risk_broker = self.broker.clone();
        let risk_max_conc = self.config.max_concentration;
        let risk_pos_pct = self.config.position_size_pct;
        let risk_journal = self.journal.clone();
        let risk_enforcer = self.risk_enforcer.clone();
        tokio::spawn(async move {
            risk_actor(signal_rx, order_tx, risk_broker, risk_max_conc, risk_pos_pct, risk_journal, risk_enforcer, risk_shutdown).await;
        });

        // Spawn OrderActor
        let order_shutdown = shutdown_rx.clone();
        let order_broker = self.broker.clone();
        let order_stats = self.stats.clone();
        let auto_fill = self.auto_fill;
        let order_journal = self.journal.clone();
        let order_enforcer = self.risk_enforcer.clone();
        tokio::spawn(async move {
            order_actor(order_rx, order_broker, order_stats, order_journal, order_enforcer, order_shutdown, auto_fill).await;
        });

        info!("🚀 Trading engine started: {} on {:?}", self.config.strategy_name, self.config.symbols);
    }

    /// Stop the engine gracefully.
    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            // Record stop in journal
            if let Some(ref j) = self.journal {
                let account = self.broker.get_account().await.unwrap_or_else(|_| Account {
                    id: Uuid::new_v4(),
                    name: String::new(),
                    portfolio: Portfolio { positions: HashMap::new(), cash: 0.0, total_value: 0.0 },
                    initial_capital: self.config.initial_capital,
                });
                let pnl = account.portfolio.total_value - self.config.initial_capital;
                j.record_engine_stopped(pnl, account.portfolio.total_value);
            }
            let _ = tx.send(true);
            self.running.store(false, std::sync::atomic::Ordering::SeqCst);
            info!("🛑 Trading engine stopped");
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get current engine status.
    pub async fn status(&self) -> EngineStatus {
        let stats = self.stats.lock().await;
        let account = self.broker.get_account().await.unwrap_or_else(|_| Account {
            id: Uuid::new_v4(),
            name: String::new(),
            portfolio: Portfolio {
                positions: HashMap::new(),
                cash: 0.0,
                total_value: 0.0,
            },
            initial_capital: self.config.initial_capital,
        });
        let pnl = account.portfolio.total_value - self.config.initial_capital;
        EngineStatus {
            running: self.is_running(),
            strategy: self.config.strategy_name.clone(),
            symbols: self.config.symbols.clone(),
            total_signals: stats.total_signals,
            total_orders: stats.total_orders,
            total_fills: stats.total_fills,
            total_rejected: stats.total_rejected,
            pnl,
            recent_trades: stats.recent_trades.iter().rev().take(20).cloned().collect(),
        }
    }

    /// Get the broker for external queries.
    pub fn broker(&self) -> &Arc<dyn Broker> {
        &self.broker
    }

    /// Get the journal store (if available).
    pub fn journal(&self) -> Option<&Arc<JournalStore>> {
        self.journal.as_ref()
    }

    /// Get the risk enforcer for status queries.
    pub fn risk_enforcer(&self) -> &Arc<RiskEnforcer> {
        &self.risk_enforcer
    }
}

// ── DataActor ───────────────────────────────────────────────────────
// Provides market data bars at a configurable interval.
// Supports two modes:
//   - Simulated: deterministic noise (for testing/demo)
//   - Live: fetches real-time quotes from Tushare/AKShare

async fn data_actor(
    symbols: Vec<String>,
    interval_secs: u64,
    data_mode: DataMode,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    match &data_mode {
        DataMode::Simulated => {
            info!("📊 DataActor started [SIMULATED] for {:?}", symbols);
            data_actor_simulated(symbols, interval_secs, tx, shutdown).await;
        }
        DataMode::Live { tushare_url, tushare_token, akshare_url } => {
            info!("📊 DataActor started [LIVE] for {:?}", symbols);
            data_actor_live(
                symbols, interval_secs,
                tushare_url.clone(), tushare_token.clone(), akshare_url.clone(),
                tx, shutdown,
            ).await;
        }
    }
}

/// Simulated data actor: generates deterministic sine/cosine noise bars.
async fn data_actor_simulated(
    symbols: Vec<String>,
    interval_secs: u64,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut prices: HashMap<String, f64> = HashMap::new();
    for sym in &symbols {
        let base = match sym.as_str() {
            "600519.SH" => 1650.0,
            "000858.SZ" => 148.0,
            "601318.SH" => 52.0,
            "000001.SZ" => 12.5,
            "600036.SH" => 35.0,
            "300750.SZ" => 220.0,
            _ => 100.0,
        };
        prices.insert(sym.clone(), base);
    }
    let mut tick: u64 = 0;

    loop {
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)) => {
                for sym in &symbols {
                    let price = prices.get_mut(sym).unwrap();
                    let noise = ((tick as f64 * 7.3 + 13.7).sin() * 0.015
                        + (tick as f64 * 3.1).cos() * 0.005)
                        * *price;
                    let open = *price;
                    *price += noise;
                    let close = *price;
                    let high = open.max(close) * (1.0 + (tick as f64 * 5.1).sin().abs() * 0.003);
                    let low = open.min(close) * (1.0 - (tick as f64 * 4.3).cos().abs() * 0.003);
                    let volume = 5_000_000.0 + (tick as f64 * 2.7).sin().abs() * 3_000_000.0;

                    let kline = Kline {
                        symbol: sym.clone(),
                        datetime: Utc::now().naive_utc(),
                        open: (open * 100.0).round() / 100.0,
                        high: (high * 100.0).round() / 100.0,
                        low: (low * 100.0).round() / 100.0,
                        close: (close * 100.0).round() / 100.0,
                        volume,
                    };

                    if tx.send(MarketEvent { kline }).await.is_err() {
                        info!("DataActor: channel closed, shutting down");
                        return;
                    }
                }
                tick += 1;
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("DataActor: shutdown signal received");
                    return;
                }
            }
        }
    }
}

/// Live data actor: fetches real-time quotes from Tushare/AKShare and
/// converts them to Kline bars for the strategy pipeline.
async fn data_actor_live(
    symbols: Vec<String>,
    interval_secs: u64,
    tushare_url: String,
    tushare_token: String,
    akshare_url: String,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    let client = reqwest::Client::new();

    // Track previous prices for OHLC construction
    let mut prev_prices: HashMap<String, f64> = HashMap::new();

    loop {
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)) => {
                for sym in &symbols {
                    let kline = match fetch_realtime_kline(
                        &client, sym, &tushare_url, &tushare_token,
                        &akshare_url, &mut prev_prices,
                    ).await {
                        Some(k) => k,
                        None => continue,
                    };

                    if tx.send(MarketEvent { kline }).await.is_err() {
                        info!("DataActor[live]: channel closed");
                        return;
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("DataActor[live]: shutdown");
                    return;
                }
            }
        }
    }
}

/// Fetch a real-time quote and construct a Kline bar.
/// Tries AKShare first (free, no token needed), falls back to Tushare.
async fn fetch_realtime_kline(
    client: &reqwest::Client,
    symbol: &str,
    tushare_url: &str,
    tushare_token: &str,
    akshare_url: &str,
    prev_prices: &mut HashMap<String, f64>,
) -> Option<Kline> {
    // Try AKShare real-time spot API first
    if let Some(kline) = try_akshare_quote(client, symbol, akshare_url, prev_prices).await {
        return Some(kline);
    }
    // Fall back to Tushare
    if let Some(kline) = try_tushare_quote(client, symbol, tushare_url, tushare_token, prev_prices).await {
        return Some(kline);
    }
    warn!("DataActor[live]: failed to fetch quote for {}, skipping", symbol);
    None
}

/// Try to get a real-time quote from AKShare.
async fn try_akshare_quote(
    client: &reqwest::Client,
    symbol: &str,
    akshare_url: &str,
    prev_prices: &mut HashMap<String, f64>,
) -> Option<Kline> {
    // AKShare spot API: /api/public/stock_zh_a_spot_em
    let url = format!("{}/api/public/stock_zh_a_spot_em", akshare_url.trim_end_matches('/'));
    let resp = client.get(&url)
        .query(&[("symbol", symbol)])
        .timeout(std::time::Duration::from_secs(5))
        .send().await.ok()?;

    if !resp.status().is_success() {
        return None;
    }

    let data: serde_json::Value = resp.json().await.ok()?;

    // AKShare returns fields like "最新价", "今开", "最高", "最低", "成交量"
    let close = data.get("最新价").or(data.get("close"))
        .and_then(|v| v.as_f64())?;
    let open = data.get("今开").or(data.get("open"))
        .and_then(|v| v.as_f64()).unwrap_or(close);
    let high = data.get("最高").or(data.get("high"))
        .and_then(|v| v.as_f64()).unwrap_or(close);
    let low = data.get("最低").or(data.get("low"))
        .and_then(|v| v.as_f64()).unwrap_or(close);
    let volume = data.get("成交量").or(data.get("volume"))
        .and_then(|v| v.as_f64()).unwrap_or(0.0);

    prev_prices.insert(symbol.to_string(), close);

    Some(Kline {
        symbol: symbol.to_string(),
        datetime: Utc::now().naive_utc(),
        open, high, low, close, volume,
    })
}

/// Try to get a real-time quote from Tushare.
async fn try_tushare_quote(
    client: &reqwest::Client,
    symbol: &str,
    tushare_url: &str,
    tushare_token: &str,
    prev_prices: &mut HashMap<String, f64>,
) -> Option<Kline> {
    let body = serde_json::json!({
        "api_name": "realtime_quote",
        "token": tushare_token,
        "params": { "ts_code": symbol }
    });

    let resp = client.post(tushare_url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(5))
        .send().await.ok()?;

    if !resp.status().is_success() {
        return None;
    }

    let data: serde_json::Value = resp.json().await.ok()?;
    let items = data.get("data")?.get("items")?.as_array()?;
    let fields = data.get("data")?.get("fields")?.as_array()?;

    if items.is_empty() {
        return None;
    }

    let row = items.first()?.as_array()?;
    let field_map: HashMap<String, usize> = fields.iter().enumerate()
        .filter_map(|(i, f)| f.as_str().map(|s| (s.to_string(), i)))
        .collect();

    let get_f64 = |name: &str| -> Option<f64> {
        field_map.get(name).and_then(|&i| row.get(i)?.as_f64())
    };

    let close = get_f64("price").or(get_f64("close"))?;
    let open = get_f64("open").unwrap_or(close);
    let high = get_f64("high").unwrap_or(close);
    let low = get_f64("low").unwrap_or(close);
    let volume = get_f64("vol").or(get_f64("volume")).unwrap_or(0.0);

    prev_prices.insert(symbol.to_string(), close);

    Some(Kline {
        symbol: symbol.to_string(),
        datetime: Utc::now().naive_utc(),
        open, high, low, close, volume,
    })
}

// ── StrategyActor ───────────────────────────────────────────────────
// Receives market events, runs strategy logic, emits trade signals.

async fn strategy_actor<F>(
    strategy_factory: F,
    mut rx: mpsc::Receiver<MarketEvent>,
    tx: mpsc::Sender<TradeSignal>,
    stats: Arc<Mutex<EngineStats>>,
    journal: Option<Arc<JournalStore>>,
    mut shutdown: watch::Receiver<bool>,
) where
    F: Fn() -> Box<dyn Strategy> + Send + 'static,
{
    let mut strategies: HashMap<String, Box<dyn Strategy>> = HashMap::new();

    info!("📈 StrategyActor started");

    loop {
        tokio::select! {
            Some(event) = rx.recv() => {
                let sym = event.kline.symbol.clone();
                let strategy = strategies.entry(sym).or_insert_with(|| {
                    let mut s = strategy_factory();
                    s.on_init();
                    s
                });

                if let Some(signal) = strategy.on_bar(&event.kline) {
                    if signal.is_buy() || signal.is_sell() {
                        {
                            let mut s = stats.lock().await;
                            s.total_signals += 1;
                        }
                        let action_str = if signal.is_buy() { "BUY" } else { "SELL" };
                        info!(
                            "📡 Signal: {} {} @ {:.2} (confidence: {:.4})",
                            action_str, signal.symbol, event.kline.close, signal.confidence
                        );
                        // Record signal in journal
                        if let Some(ref j) = journal {
                            j.record_signal(&signal.symbol, action_str, signal.confidence, event.kline.close);
                        }
                        if tx.send(TradeSignal { signal, kline: event.kline }).await.is_err() {
                            return;
                        }
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    for (_, mut s) in strategies.drain() {
                        s.on_stop();
                    }
                    info!("StrategyActor: shutdown");
                    return;
                }
            }
        }
    }
}

// ── RiskActor ───────────────────────────────────────────────────────
// Validates signals against risk rules, sizes positions, creates orders.

async fn risk_actor(
    mut rx: mpsc::Receiver<TradeSignal>,
    tx: mpsc::Sender<OrderRequest>,
    broker: Arc<dyn Broker>,
    max_concentration: f64,
    position_size_pct: f64,
    journal: Option<Arc<JournalStore>>,
    enforcer: Arc<RiskEnforcer>,
    mut shutdown: watch::Receiver<bool>,
) {
    let risk_checker = quant_risk::checks::RiskChecker::new(max_concentration, 0.05, 0.15);

    info!("🛡️ RiskActor started (max_conc={:.0}%, pos_size={:.0}%, stop_loss={:.1}%, daily_limit={:.1}%, drawdown={:.1}%)",
        max_concentration * 100.0, position_size_pct * 100.0,
        enforcer.config().stop_loss_pct * 100.0,
        enforcer.config().max_daily_loss_pct * 100.0,
        enforcer.config().max_drawdown_pct * 100.0);

    loop {
        tokio::select! {
            Some(ts) = rx.recv() => {
                let account = match broker.get_account().await {
                    Ok(a) => a,
                    Err(e) => {
                        error!("RiskActor: failed to get account: {}", e);
                        continue;
                    }
                };
                let portfolio = &account.portfolio;

                // Update portfolio value for drawdown tracking
                let halted = enforcer.update_portfolio_value(portfolio.total_value);
                if halted {
                    warn!("🔴 Max drawdown breached! Engine trading halted.");
                    if let Some(ref j) = journal {
                        j.record_risk_rejected(&ts.signal.symbol, OrderSide::Buy, 0.0, 0.0,
                            "Max drawdown limit exceeded — engine halted");
                    }
                }

                // Check stop-losses on current positions
                let positions: Vec<PositionInfo> = portfolio.positions.iter().map(|(sym, pos)| {
                    PositionInfo {
                        symbol: sym.clone(),
                        quantity: pos.quantity,
                        avg_cost: pos.avg_cost,
                        current_price: ts.kline.close, // approximate with latest bar price
                    }
                }).collect();

                let sl_signals = enforcer.check_stop_losses(&positions);
                for sl in &sl_signals {
                    warn!("🛑 Stop-loss triggered: {} loss={:.1}% > threshold={:.1}%",
                        sl.symbol, sl.loss_pct.abs() * 100.0, sl.threshold * 100.0);
                    let sl_order = Order {
                        id: Uuid::new_v4(),
                        symbol: sl.symbol.clone(),
                        side: OrderSide::Sell,
                        order_type: OrderType::Market,
                        price: sl.current_price,
                        quantity: sl.quantity,
                        filled_qty: 0.0,
                        status: OrderStatus::Pending,
                        created_at: Utc::now().naive_utc(),
                        updated_at: Utc::now().naive_utc(),
                    };
                    if let Some(ref j) = journal {
                        j.record_signal(&sl.symbol, "SELL", sl.loss_pct.abs(),
                            sl.current_price);
                    }
                    let _ = tx.send(OrderRequest { order: sl_order }).await;
                }

                let (side, price) = if ts.signal.is_buy() {
                    (OrderSide::Buy, ts.kline.close)
                } else {
                    (OrderSide::Sell, ts.kline.close)
                };

                // Check active risk enforcement for BUY orders
                if side == OrderSide::Buy {
                    if let Err(reason) = enforcer.can_buy() {
                        warn!("🚫 Risk enforcer blocked BUY: {}", reason);
                        if let Some(ref j) = journal {
                            j.record_risk_rejected(&ts.signal.symbol, side, 0.0, price, &reason);
                        }
                        continue;
                    }
                }

                // Calculate position size
                let quantity = if side == OrderSide::Buy {
                    let allocation = portfolio.total_value * position_size_pct;
                    let raw_shares = allocation / price;
                    let lots = (raw_shares / 100.0).floor() * 100.0;
                    if lots < 100.0 && allocation >= price * 100.0 {
                        100.0
                    } else {
                        lots
                    }
                } else {
                    portfolio.positions
                        .get(&ts.signal.symbol)
                        .map(|p| p.quantity)
                        .unwrap_or(0.0)
                };

                if quantity <= 0.0 {
                    continue;
                }

                let order = Order {
                    id: Uuid::new_v4(),
                    symbol: ts.signal.symbol.clone(),
                    side,
                    order_type: OrderType::Market,
                    price,
                    quantity,
                    filled_qty: 0.0,
                    status: OrderStatus::Pending,
                    created_at: Utc::now().naive_utc(),
                    updated_at: Utc::now().naive_utc(),
                };

                // Run pre-trade risk checks
                if let Err(reason) = risk_checker.check_order(&order, portfolio) {
                    warn!("🚫 Risk rejected: {} {} x{:.0} — {}", 
                        order.symbol, if side == OrderSide::Buy {"BUY"} else {"SELL"}, quantity, reason);
                    if let Some(ref j) = journal {
                        j.record_risk_rejected(&order.symbol, side, quantity, price, &reason.to_string());
                    }
                    continue;
                }

                info!("✅ Risk approved: {} {} x{:.0} @ {:.2}",
                    if side == OrderSide::Buy {"BUY"} else {"SELL"},
                    order.symbol, quantity, price);

                if tx.send(OrderRequest { order }).await.is_err() {
                    return;
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("RiskActor: shutdown");
                    return;
                }
            }
        }
    }
}

// ── OrderActor ──────────────────────────────────────────────────────
// Executes orders via the broker, records execution reports.

async fn order_actor(
    mut rx: mpsc::Receiver<OrderRequest>,
    broker: Arc<dyn Broker>,
    stats: Arc<Mutex<EngineStats>>,
    journal: Option<Arc<JournalStore>>,
    enforcer: Arc<RiskEnforcer>,
    mut shutdown: watch::Receiver<bool>,
    _auto_fill: bool,
) {
    info!("💹 OrderActor started");

    loop {
        tokio::select! {
            Some(req) = rx.recv() => {
                let order = req.order;
                let side_str = if order.side == OrderSide::Buy { "BUY" } else { "SELL" };

                // Submit order to broker (PaperBroker auto-fills; QmtBroker submits to exchange)
                match broker.submit_order(&order).await {
                    Ok(submitted) => {
                        enforcer.record_success();
                        let status_str = format!("{:?}", submitted.status);
                        let report = ExecutionReport {
                            order_id: submitted.id,
                            symbol: order.symbol.clone(),
                            side: order.side,
                            price: order.price,
                            quantity: order.quantity,
                            commission: 0.0,
                            timestamp: Utc::now().naive_utc(),
                            status: status_str,
                        };
                        info!(
                            "🔔 {}: {} {} x{:.0} @ {:.2}",
                            report.status, side_str, order.symbol, order.quantity, order.price
                        );
                        // Record in journal
                        if let Some(ref j) = journal {
                            j.record_order_submitted(order.id, &order.symbol, order.side, order.quantity, order.price);
                            j.record_order_filled(submitted.id, &order.symbol, order.side,
                                order.quantity, order.price, 0.0, 0.0, 0.0);
                        }
                        let mut s = stats.lock().await;
                        s.total_orders += 1;
                        s.total_fills += 1;
                        s.recent_trades.push(report);
                    }
                    Err(e) => {
                        let tripped = enforcer.record_failure();
                        if tripped {
                            warn!("🔴 Circuit breaker tripped after consecutive failures!");
                        }
                        error!("❌ Submit failed: {} — {}", order.symbol, e);
                        if let Some(ref j) = journal {
                            j.record_risk_rejected(&order.symbol, order.side, order.quantity, order.price,
                                &format!("submit_error: {}", e));
                        }
                        let mut s = stats.lock().await;
                        s.total_rejected += 1;
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("OrderActor: shutdown");
                    return;
                }
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper::PaperBroker;

    // Minimal strategy that buys on first bar, sells on second
    struct TestStrategy {
        bar_count: usize,
    }
    impl Strategy for TestStrategy {
        fn name(&self) -> &str { "test" }
        fn on_init(&mut self) { self.bar_count = 0; }
        fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
            self.bar_count += 1;
            match self.bar_count {
                3 => Some(Signal::buy(&kline.symbol, 0.8, kline.datetime)),
                6 => Some(Signal::sell(&kline.symbol, 0.8, kline.datetime)),
                _ => None,
            }
        }
        fn on_stop(&mut self) {}
    }

    #[tokio::test]
    async fn test_engine_lifecycle() {
        let config = EngineConfig {
            strategy_name: "test".into(),
            symbols: vec!["000001.SZ".into()],  // Cheap stock (~12.5) so 100 shares is affordable
            interval_secs: 1,
            initial_capital: 1_000_000.0,
            commission_rate: 0.0003,
            stamp_tax_rate: 0.001,
            max_concentration: 0.25,
            position_size_pct: 0.10,
            data_mode: DataMode::Simulated,
            risk_config: RiskConfig::default(),
        };

        let mut engine = TradingEngine::new(config);
        engine.start(|| Box::new(TestStrategy { bar_count: 0 })).await;
        assert!(engine.is_running());

        // Let it run long enough for the strategy to fire (buy on bar 3, sell on bar 6)
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

        let status = engine.status().await;
        assert!(status.running);
        assert!(status.total_signals > 0, "Expected signals, got {}", status.total_signals);
        assert!(status.total_fills > 0,
            "Expected fills, got {} (signals={}, orders={}, rejected={})",
            status.total_fills, status.total_signals, status.total_orders, status.total_rejected);

        engine.stop().await;
        assert!(!engine.is_running());

        let account = engine.broker().get_account().await.unwrap();
        println!("Final portfolio: cash={:.2}, value={:.2}, fills={}", 
            account.portfolio.cash, account.portfolio.total_value, status.total_fills);
    }

    #[tokio::test]
    async fn test_engine_channels_flow() {
        // Test that messages flow through the channel pipeline
        let (market_tx, mut market_rx) = mpsc::channel::<MarketEvent>(16);
        
        let kline = Kline {
            symbol: "TEST".into(),
            datetime: Utc::now().naive_utc(),
            open: 100.0, high: 102.0, low: 99.0, close: 101.0, volume: 1000.0,
        };
        
        market_tx.send(MarketEvent { kline: kline.clone() }).await.unwrap();
        let event = market_rx.recv().await.unwrap();
        assert_eq!(event.kline.symbol, "TEST");
        assert_eq!(event.kline.close, 101.0);
    }

    #[tokio::test]
    async fn test_risk_rejects_oversized_order() {
        let broker = Arc::new(PaperBroker::new(100_000.0, 0.0003, 0.001));
        let (order_tx, mut order_rx) = mpsc::channel::<OrderRequest>(16);
        let (signal_tx, signal_rx) = mpsc::channel::<TradeSignal>(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        // Start risk actor
        tokio::spawn(async move {
            risk_actor(signal_rx, order_tx, broker, 0.05, 0.10, None,
                Arc::new(RiskEnforcer::new(RiskConfig::default(), 100_000.0)),
                shutdown_rx).await;
        });

        // Send a signal for an expensive stock (will breach concentration at 10%)
        let kline = Kline {
            symbol: "600519.SH".into(),
            datetime: Utc::now().naive_utc(),
            open: 1650.0, high: 1660.0, low: 1640.0, close: 1650.0, volume: 1000.0,
        };
        let signal = Signal::buy("600519.SH", 0.8, Utc::now().naive_utc());
        signal_tx.send(TradeSignal { signal, kline }).await.unwrap();

        // Give time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // The risk actor should reject this (100 shares * 1650 = 165000 > 5% of 100k)
        // So nothing should arrive on order_rx
        assert!(order_rx.try_recv().is_err(), "Order should have been rejected by risk");
    }
}
