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
//!
//! ## Low-Latency Design
//! - **Lock-free stats**: All hot-path counters use `AtomicU64`/`AtomicF64`
//!   (CAS loop). Only `recent_trades` uses `std::sync::Mutex` (cold path).
//! - **No TCP Mutex**: Reader/writer owned directly by single task, no
//!   `Arc<Mutex<>>` wrapper. Writer dropped after subscribe (push-only mode).
//! - **Parallel HTTP**: Multi-symbol quotes fetched concurrently via `join_all`.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};
use tracing::{info, warn, error};
use uuid::Uuid;

use quant_core::models::*;
use quant_core::traits::{Broker, Strategy};
use quant_core::types::*;
use quant_risk::enforcement::{RiskConfig, RiskEnforcer, PositionInfo};

// ── Messages ────────────────────────────────────────────────────────

/// Market data event from DataActor → StrategyActor.
/// Supports L1 (Kline bars) and L2 (tick/depth) data.
#[derive(Debug, Clone)]
pub enum MarketEvent {
    /// L1 K-line bar (OHLCV)
    Bar(Kline),
    /// L2 逐笔成交 (tick-by-tick trade)
    Tick(TickData),
    /// L2 盘口深度快照 (depth/order book)
    Depth(DepthData),
}

impl MarketEvent {
    /// Extract symbol from any variant.
    pub fn symbol(&self) -> &str {
        match self {
            MarketEvent::Bar(k) => &k.symbol,
            MarketEvent::Tick(t) => &t.symbol,
            MarketEvent::Depth(d) => &d.symbol,
        }
    }

    /// Extract latest price from any variant.
    pub fn price(&self) -> f64 {
        match self {
            MarketEvent::Bar(k) => k.close,
            MarketEvent::Tick(t) => t.price,
            MarketEvent::Depth(d) => d.last_price,
        }
    }
}

/// Trade signal from StrategyActor → RiskActor
#[derive(Debug, Clone)]
pub struct TradeSignal {
    pub signal: Signal,
    pub price: f64,
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
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Pipeline latency (microseconds)
    pub latency: PipelineLatency,
}

/// Pipeline latency metrics (microseconds per stage).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineLatency {
    pub last_factor_compute_us: u64,
    pub last_risk_check_us: u64,
    pub last_order_submit_us: u64,
    pub avg_factor_compute_us: u64,
    pub total_bars_processed: u64,
}

/// Real-time performance metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Current portfolio value
    pub portfolio_value: f64,
    /// Initial capital
    pub initial_capital: f64,
    /// Total return percentage
    pub total_return_pct: f64,
    /// Peak portfolio value
    pub peak_value: f64,
    /// Current drawdown percentage
    pub drawdown_pct: f64,
    /// Max drawdown observed
    pub max_drawdown_pct: f64,
    /// Win rate (filled buy orders that were profitable on sell)
    pub win_rate: f64,
    /// Number of winning trades
    pub wins: u64,
    /// Number of losing trades
    pub losses: u64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average trade PnL
    pub avg_trade_pnl: f64,
    /// Risk status
    pub risk_daily_pnl: f64,
    pub risk_daily_paused: bool,
    pub risk_circuit_open: bool,
    pub risk_drawdown_halted: bool,
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
    /// Live: fetch real-time quotes from DataProvider (Tushare/AKShare REST API)
    Live {
        /// Tushare API URL
        tushare_url: String,
        /// Tushare API token
        tushare_token: String,
        /// AKShare API URL
        akshare_url: String,
    },
    /// LowLatency: persistent market data server (HTTP + cache, no fork overhead)
    /// Uses scripts/market_data_server.py on port 18092
    /// Auto-starts the server if not running.
    LowLatency {
        /// Market data server URL (default: http://127.0.0.1:18092)
        server_url: String,
    },
    /// HistoricalReplay: replay real historical klines at configurable speed
    /// Best for strategy verification — deterministic, repeatable
    HistoricalReplay {
        /// Start date (YYYY-MM-DD)
        start_date: String,
        /// End date (YYYY-MM-DD)
        end_date: String,
        /// Replay speed multiplier (1.0 = real-time, 0.0 = as fast as possible)
        speed: f64,
        /// K-line period: "daily", "1", "5", "15", "30", "60" (minutes)
        period: String,
    },
    /// L2: Level-2 tick/depth data via TCP MQ from L2 sidecar
    /// Receives 逐笔成交 + 五档/十档盘口 push data
    L2 {
        /// L2 data server TCP address (default: 127.0.0.1:18095)
        l2_addr: String,
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
use crate::notifier::Notifier;

// ── Lock-Free Shared State ──────────────────────────────────────────
// All hot-path stats use atomics to avoid Mutex contention.
// Only recent_trades uses a Mutex (cold path, infrequent writes).

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic f64 wrapper using u64 bit representation.
#[derive(Debug)]
struct AtomicF64(AtomicU64);

impl AtomicF64 {
    fn new(val: f64) -> Self { Self(AtomicU64::new(val.to_bits())) }
    fn load(&self) -> f64 { f64::from_bits(self.0.load(Ordering::Relaxed)) }
    fn store(&self, val: f64) { self.0.store(val.to_bits(), Ordering::Relaxed); }
    fn fetch_add(&self, val: f64) {
        loop {
            let old = self.0.load(Ordering::Relaxed);
            let new = f64::from_bits(old) + val;
            if self.0.compare_exchange_weak(old, new.to_bits(), Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                break;
            }
        }
    }
    fn fetch_max(&self, val: f64) {
        loop {
            let old = self.0.load(Ordering::Relaxed);
            let cur = f64::from_bits(old);
            if cur >= val { break; }
            if self.0.compare_exchange_weak(old, val.to_bits(), Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                break;
            }
        }
    }
}

impl Default for AtomicF64 {
    fn default() -> Self { Self::new(0.0) }
}

/// Lock-free engine statistics. All counters are atomic for zero-contention updates.
struct EngineStatsAtomic {
    total_signals: AtomicU64,
    total_orders: AtomicU64,
    total_fills: AtomicU64,
    total_rejected: AtomicU64,
    wins: AtomicU64,
    losses: AtomicU64,
    gross_profit: AtomicF64,
    gross_loss: AtomicF64,
    max_drawdown_pct: AtomicF64,
    // Latency tracking
    last_factor_us: AtomicU64,
    last_risk_us: AtomicU64,
    last_order_us: AtomicU64,
    total_factor_us: AtomicU64,
    total_bars: AtomicU64,
    // Recent trades use Mutex (infrequent cold-path writes), bounded VecDeque
    recent_trades: std::sync::Mutex<std::collections::VecDeque<ExecutionReport>>,
}

impl Default for EngineStatsAtomic {
    fn default() -> Self {
        Self {
            total_signals: AtomicU64::new(0),
            total_orders: AtomicU64::new(0),
            total_fills: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            wins: AtomicU64::new(0),
            losses: AtomicU64::new(0),
            gross_profit: AtomicF64::default(),
            gross_loss: AtomicF64::default(),
            max_drawdown_pct: AtomicF64::default(),
            last_factor_us: AtomicU64::new(0),
            last_risk_us: AtomicU64::new(0),
            last_order_us: AtomicU64::new(0),
            total_factor_us: AtomicU64::new(0),
            total_bars: AtomicU64::new(0),
            recent_trades: std::sync::Mutex::new(std::collections::VecDeque::with_capacity(1000)),
        }
    }
}

impl std::fmt::Debug for EngineStatsAtomic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineStatsAtomic")
            .field("total_signals", &self.total_signals.load(Ordering::Relaxed))
            .field("total_bars", &self.total_bars.load(Ordering::Relaxed))
            .finish()
    }
}

// ── Trading Engine ──────────────────────────────────────────────────

/// The trading engine, generic over a `Broker` implementation.
///
/// Use `TradingEngine::new_paper(config)` for simulation or
/// `TradingEngine::new_with_broker(config, broker)` for live trading.
pub struct TradingEngine {
    config: EngineConfig,
    shutdown_tx: Option<watch::Sender<bool>>,
    stats: Arc<EngineStatsAtomic>,
    broker: Arc<dyn Broker>,
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Whether fill_order should be called after submit (paper mode only)
    auto_fill: bool,
    /// Trade journal for persistent audit trail
    journal: Option<Arc<JournalStore>>,
    /// Active risk enforcement (stop-loss, daily loss, drawdown, circuit breaker)
    risk_enforcer: Arc<RiskEnforcer>,
    /// Notification service for order fills, risk alerts, etc.
    notifier: Option<Arc<Notifier>>,
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
        let notifier = Notifier::open("data/notifications.db", "data")
            .map(Arc::new)
            .ok();
        Self {
            config,
            shutdown_tx: None,
            stats: Arc::new(EngineStatsAtomic::default()),
            broker,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            auto_fill: true,
            journal,
            risk_enforcer,
            notifier,
        }
    }

    /// Create an engine with a custom broker (e.g., QmtBroker for live trading).
    pub fn new_with_broker(config: EngineConfig, broker: Arc<dyn Broker>) -> Self {
        let journal = JournalStore::open("data/trade_journal.db")
            .map(Arc::new)
            .ok();
        let risk_enforcer = Arc::new(RiskEnforcer::new(config.risk_config.clone(), config.initial_capital));
        let notifier = Notifier::open("data/notifications.db", "data")
            .map(Arc::new)
            .ok();
        Self {
            config,
            shutdown_tx: None,
            stats: Arc::new(EngineStatsAtomic::default()),
            broker,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            auto_fill: false,
            journal,
            risk_enforcer,
            notifier,
        }
    }

    /// Set an external journal store (e.g., from AppState).
    pub fn set_journal(&mut self, journal: Arc<JournalStore>) {
        self.journal = Some(journal);
    }

    /// Set an external notifier (e.g., from AppState).
    pub fn set_notifier(&mut self, notifier: Arc<Notifier>) {
        self.notifier = Some(notifier);
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

        // Create channels between actors (sized for throughput)
        let (market_tx, market_rx) = mpsc::channel::<MarketEvent>(512);
        let (signal_tx, signal_rx) = mpsc::channel::<TradeSignal>(64);
        let (order_tx, order_rx) = mpsc::channel::<OrderRequest>(64);

        // Spawn DataActor
        let data_shutdown = shutdown_rx.clone();
        let data_symbols = self.config.symbols.clone();
        let data_interval = self.config.interval_secs;
        let data_mode = self.config.data_mode.clone();
        tokio::spawn(async move {
            crate::data_actors::data_actor(data_symbols, data_interval, data_mode, market_tx, data_shutdown).await;
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
        let order_notifier = self.notifier.clone();
        tokio::spawn(async move {
            order_actor(order_rx, order_broker, order_stats, order_journal, order_enforcer, order_notifier, order_shutdown, auto_fill).await;
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

    /// Get current engine status (lock-free reads).
    pub async fn status(&self) -> EngineStatus {
        let stats = &self.stats;
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
        let risk_status = self.risk_enforcer.status();
        let wins = stats.wins.load(Ordering::Relaxed);
        let losses = stats.losses.load(Ordering::Relaxed);
        let total_trades = wins + losses;
        let peak = risk_status.peak_value;
        let drawdown_pct = if peak > 0.0 { (peak - account.portfolio.total_value) / peak } else { 0.0 };
        let drawdown_pct = drawdown_pct.max(0.0);
        let max_dd = stats.max_drawdown_pct.load().max(drawdown_pct);
        let total_bars = stats.total_bars.load(Ordering::Relaxed);

        let recent_trades = stats.recent_trades.lock()
            .map(|t| t.iter().rev().take(20).cloned().collect())
            .unwrap_or_default();

        EngineStatus {
            running: self.is_running(),
            strategy: self.config.strategy_name.clone(),
            symbols: self.config.symbols.clone(),
            total_signals: stats.total_signals.load(Ordering::Relaxed),
            total_orders: stats.total_orders.load(Ordering::Relaxed),
            total_fills: stats.total_fills.load(Ordering::Relaxed),
            total_rejected: stats.total_rejected.load(Ordering::Relaxed),
            pnl,
            recent_trades,
            performance: PerformanceMetrics {
                portfolio_value: account.portfolio.total_value,
                initial_capital: self.config.initial_capital,
                total_return_pct: if self.config.initial_capital > 0.0 {
                    pnl / self.config.initial_capital * 100.0
                } else { 0.0 },
                peak_value: peak,
                drawdown_pct: drawdown_pct * 100.0,
                max_drawdown_pct: max_dd * 100.0,
                win_rate: if total_trades > 0 {
                    wins as f64 / total_trades as f64 * 100.0
                } else { 0.0 },
                wins,
                losses,
                profit_factor: if stats.gross_loss.load().abs() > 0.0 {
                    stats.gross_profit.load() / stats.gross_loss.load().abs()
                } else if stats.gross_profit.load() > 0.0 { f64::INFINITY } else { 0.0 },
                avg_trade_pnl: if total_trades > 0 {
                    (stats.gross_profit.load() - stats.gross_loss.load().abs()) / total_trades as f64
                } else { 0.0 },
                risk_daily_pnl: risk_status.daily_pnl,
                risk_daily_paused: risk_status.daily_paused,
                risk_circuit_open: risk_status.circuit_open,
                risk_drawdown_halted: risk_status.drawdown_halted,
            },
            latency: PipelineLatency {
                last_factor_compute_us: stats.last_factor_us.load(Ordering::Relaxed),
                last_risk_check_us: stats.last_risk_us.load(Ordering::Relaxed),
                last_order_submit_us: stats.last_order_us.load(Ordering::Relaxed),
                avg_factor_compute_us: if total_bars > 0 {
                    stats.total_factor_us.load(Ordering::Relaxed) / total_bars
                } else { 0 },
                total_bars_processed: total_bars,
            },
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

// ── StrategyActor ───────────────────────────────────────────────────
// Receives market events, runs strategy logic, emits trade signals.

async fn strategy_actor<F>(
    strategy_factory: F,
    mut rx: mpsc::Receiver<MarketEvent>,
    tx: mpsc::Sender<TradeSignal>,
    stats: Arc<EngineStatsAtomic>,
    journal: Option<Arc<JournalStore>>,
    mut shutdown: watch::Receiver<bool>,
) where
    F: Fn() -> Box<dyn Strategy> + Send + 'static,
{
    let mut strategies: HashMap<String, Box<dyn Strategy>> = HashMap::new();
    let mut last_seen: HashMap<String, std::time::Instant> = HashMap::new();
    let mut gc_counter = 0u64;

    info!("📈 StrategyActor started (lock-free stats)");

    loop {
        tokio::select! {
            Some(event) = rx.recv() => {
                let sym = event.symbol().to_string();

                // Periodic GC: evict strategies not seen in 10 minutes (before mutable borrow)
                gc_counter += 1;
                if gc_counter % 10000 == 0 && strategies.len() > 20 {
                    let cutoff = std::time::Instant::now() - std::time::Duration::from_secs(600);
                    let stale: Vec<String> = last_seen.iter()
                        .filter(|(_, ts)| **ts < cutoff)
                        .map(|(k, _)| k.clone())
                        .collect();
                    for s in &stale {
                        if let Some(mut old) = strategies.remove(s) { old.on_stop(); }
                        last_seen.remove(s);
                    }
                    if !stale.is_empty() {
                        info!("📈 GC: evicted {} stale strategies", stale.len());
                    }
                }

                if !strategies.contains_key(&sym) {
                    let mut s = strategy_factory();
                    s.on_init();
                    strategies.insert(sym.clone(), s);
                }
                last_seen.insert(sym.clone(), std::time::Instant::now());
                let strategy = strategies.get_mut(&sym).unwrap();

                let t0 = std::time::Instant::now();
                let signal_opt = match &event {
                    MarketEvent::Bar(kline) => strategy.on_bar(kline),
                    MarketEvent::Tick(tick) => strategy.on_tick(tick),
                    MarketEvent::Depth(depth) => strategy.on_depth(depth),
                };
                let factor_us = t0.elapsed().as_micros() as u64;

                // Lock-free latency stats update
                stats.last_factor_us.store(factor_us, Ordering::Relaxed);
                stats.total_factor_us.fetch_add(factor_us, Ordering::Relaxed);
                stats.total_bars.fetch_add(1, Ordering::Relaxed);

                if factor_us > 1000 {
                    warn!("⏱️ Slow factor compute: {}μs for {}", factor_us, sym);
                }

                if let Some(signal) = signal_opt {
                    if signal.is_buy() || signal.is_sell() {
                        stats.total_signals.fetch_add(1, Ordering::Relaxed);
                        let action_str = if signal.is_buy() { "BUY" } else { "SELL" };
                        let price = event.price();
                        info!(
                            "📡 Signal: {} {} @ {:.2} (confidence: {:.4}, factor_time={}μs)",
                            action_str, signal.symbol, price, signal.confidence, factor_us
                        );
                        if let Some(ref j) = journal {
                            j.record_signal(&signal.symbol, action_str, signal.confidence, price);
                        }
                        if tx.send(TradeSignal { signal, price }).await.is_err() {
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
                        current_price: ts.price,
                        entry_time: pos.entry_time,
                        target_weight: pos.target_weight,
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
                    (OrderSide::Buy, ts.price)
                } else {
                    (OrderSide::Sell, ts.price)
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
    stats: Arc<EngineStatsAtomic>,
    journal: Option<Arc<JournalStore>>,
    enforcer: Arc<RiskEnforcer>,
    notifier: Option<Arc<Notifier>>,
    mut shutdown: watch::Receiver<bool>,
    _auto_fill: bool,
) {
    info!("💹 OrderActor started (lock-free stats)");

    loop {
        tokio::select! {
            Some(req) = rx.recv() => {
                let order = req.order;
                let side_str = if order.side == OrderSide::Buy { "BUY" } else { "SELL" };

                let t0 = std::time::Instant::now();
                match broker.submit_order(&order).await {
                    Ok(submitted) => {
                        let order_us = t0.elapsed().as_micros() as u64;
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
                            "🔔 {}: {} {} x{:.0} @ {:.2} (submit={}μs)",
                            report.status, side_str, order.symbol, order.quantity, order.price, order_us
                        );
                        if let Some(ref j) = journal {
                            j.record_order_submitted(order.id, &order.symbol, order.side, order.quantity, order.price);
                            j.record_order_filled(submitted.id, &order.symbol, order.side,
                                order.quantity, order.price, 0.0, 0.0, 0.0);
                        }
                        // Send fill notification
                        if let Some(ref n) = notifier {
                            let sym = order.symbol.clone();
                            let side_s = side_str.to_string();
                            let qty = order.quantity;
                            let px = order.price;
                            let oid = submitted.id.to_string();
                            let n = n.clone();
                            tokio::spawn(async move {
                                n.notify_order_filled(&sym, &side_s, qty, px, &oid).await;
                            });
                        }
                        // Lock-free stats
                        stats.total_orders.fetch_add(1, Ordering::Relaxed);
                        stats.total_fills.fetch_add(1, Ordering::Relaxed);
                        stats.last_order_us.store(order_us, Ordering::Relaxed);
                        if let Ok(mut trades) = stats.recent_trades.lock() {
                            if trades.len() >= 1000 { trades.pop_front(); }
                            trades.push_back(report);
                        }
                    }
                    Err(e) => {
                        let tripped = enforcer.record_failure();
                        if tripped {
                            warn!("🔴 Circuit breaker tripped after consecutive failures!");
                            // Notify risk alert on circuit breaker
                            if let Some(ref n) = notifier {
                                let n = n.clone();
                                let sym = order.symbol.clone();
                                tokio::spawn(async move {
                                    n.notify_risk_alert(&sym, "Circuit breaker tripped after consecutive order failures").await;
                                });
                            }
                        }
                        error!("❌ Submit failed: {} — {}", order.symbol, e);
                        if let Some(ref j) = journal {
                            j.record_risk_rejected(&order.symbol, order.side, order.quantity, order.price,
                                &format!("submit_error: {}", e));
                        }
                        // Notify order rejected
                        if let Some(ref n) = notifier {
                            let sym = order.symbol.clone();
                            let side_s = side_str.to_string();
                            let qty = order.quantity;
                            let px = order.price;
                            let reason = format!("submit_error: {}", e);
                            let n = n.clone();
                            tokio::spawn(async move {
                                n.notify_order_rejected(&sym, &side_s, qty, px, &reason).await;
                            });
                        }
                        stats.total_rejected.fetch_add(1, Ordering::Relaxed);
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
        
        market_tx.send(MarketEvent::Bar(kline.clone())).await.unwrap();
        let event = market_rx.recv().await.unwrap();
        assert_eq!(event.symbol(), "TEST");
        assert_eq!(event.price(), 101.0);
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
        signal_tx.send(TradeSignal { signal, price: kline.close }).await.unwrap();

        // Give time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // The risk actor should reject this (100 shares * 1650 = 165000 > 5% of 100k)
        // So nothing should arrive on order_rx
        assert!(order_rx.try_recv().is_err(), "Order should have been rejected by risk");
    }
}
