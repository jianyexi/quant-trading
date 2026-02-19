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
use tracing::{info, warn, error, debug, trace};
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
            stats: Arc::new(EngineStatsAtomic::default()),
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
            stats: Arc::new(EngineStatsAtomic::default()),
            broker,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            auto_fill: false,
            journal,
            risk_enforcer,
        }
    }

    /// Set an external journal store (e.g., from AppState).
    pub fn set_journal(&mut self, journal: Arc<JournalStore>) {
        self.journal = Some(journal);
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

// ── DataActor ───────────────────────────────────────────────────────
// Provides market data bars at a configurable interval.
// Supports two modes:
//   - Simulated: deterministic noise (for testing/demo)
//   - Live: fetches real-time quotes from Tushare/AKShare

/// Fetch historical klines for warmup from the market data server HTTP endpoint.
/// Falls back gracefully if the server is not available.
async fn warmup_historical(
    symbols: &[String],
    tx: &mpsc::Sender<MarketEvent>,
    warmup_days: usize,
) -> usize {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(_) => return 0,
    };

    // Try market_data_server first (port 18092)
    let base = "http://127.0.0.1:18092";
    let server_ok = client.get(format!("{}/health", base)).send().await
        .map(|r| r.status().is_success()).unwrap_or(false);

    if !server_ok {
        info!("📊 Warmup: market data server not available, skipping HTTP warmup");
        return 0;
    }

    let mut total = 0usize;
    let end = Utc::now().format("%Y-%m-%d").to_string();
    let start = (Utc::now() - chrono::Duration::days(warmup_days as i64 * 2))
        .format("%Y-%m-%d").to_string();

    for sym in symbols {
        let url = format!("{}/klines/{}?start={}&end={}&period=daily", base, sym, start, end);
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(data) = resp.json::<serde_json::Value>().await {
                    if let Some(klines) = data.get("klines").and_then(|v| v.as_array()) {
                        let n = klines.len();
                        for kl in klines {
                            if let Some(kline) = parse_kline_json(sym, kl) {
                                if tx.send(MarketEvent::Bar(kline)).await.is_err() {
                                    return total;
                                }
                                total += 1;
                            }
                        }
                        info!("📊 Warmup: {} loaded {} historical bars", sym, n);
                    }
                }
            }
            _ => {
                warn!("📊 Warmup: failed to fetch history for {}", sym);
            }
        }
    }
    total
}

async fn data_actor(
    symbols: Vec<String>,
    interval_secs: u64,
    data_mode: DataMode,
    tx: mpsc::Sender<MarketEvent>,
    shutdown: watch::Receiver<bool>,
) {
    match &data_mode {
        DataMode::Simulated => {
            info!("📊 DataActor started [SIMULATED] for {:?}", symbols);
            let n = warmup_historical(&symbols, &tx, 80).await;
            if n > 0 { info!("📊 Warmup complete: {} bars pre-loaded", n); }
            data_actor_simulated(symbols, interval_secs, tx, shutdown).await;
        }
        DataMode::Live { tushare_url, tushare_token, akshare_url } => {
            info!("📊 DataActor started [LIVE] for {:?}", symbols);
            let n = warmup_historical(&symbols, &tx, 80).await;
            if n > 0 { info!("📊 Warmup complete: {} bars pre-loaded", n); }
            data_actor_live(
                symbols, interval_secs,
                tushare_url.clone(), tushare_token.clone(), akshare_url.clone(),
                tx, shutdown,
            ).await;
        }
        DataMode::LowLatency { server_url } => {
            info!("📊 DataActor started [LOW_LATENCY] for {:?} (server={})", symbols, server_url);
            data_actor_low_latency(symbols, interval_secs, server_url.clone(), tx, shutdown).await;
        }
        DataMode::HistoricalReplay { start_date, end_date, speed, period } => {
            info!("📊 DataActor started [HISTORICAL_REPLAY] for {:?} ({} → {}, period={}, speed={}x)",
                symbols, start_date, end_date, period, speed);
            data_actor_replay(symbols, start_date.clone(), end_date.clone(), *speed, period.clone(), tx, shutdown).await;
        }
        DataMode::L2 { l2_addr } => {
            info!("📊 DataActor started [L2] for {:?} (addr={})", symbols, l2_addr);
            data_actor_l2(symbols, l2_addr.clone(), tx, shutdown).await;
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

                    if tx.send(MarketEvent::Bar(kline)).await.is_err() {
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

                    if tx.send(MarketEvent::Bar(kline)).await.is_err() {
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

// ── Python Bridge Data Actor ────────────────────────────────────────
// Uses scripts/market_data.py for real-time quotes via akshare library.
// More reliable than REST API which can be down.

/// Find a working Python interpreter.
fn find_python_for_bridge() -> Option<String> {
    let candidates = [
        std::env::var("PYTHON_PATH").unwrap_or_default(),
        #[cfg(windows)]
        format!("{}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
            std::env::var("USERPROFILE").unwrap_or_default()),
        "python3".to_string(),
        "python".to_string(),
    ];
    for c in &candidates {
        if c.is_empty() { continue; }
        if let Ok(output) = std::process::Command::new(c)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            if output.success() { return Some(c.clone()); }
        }
    }
    None
}

/// Fetch a real-time quote via Python subprocess.
fn python_bridge_quote(python: &str, symbol: &str) -> Option<Kline> {
    let script = std::path::Path::new("scripts/market_data.py");
    if !script.exists() {
        warn!("scripts/market_data.py not found");
        return None;
    }

    let t0 = std::time::Instant::now();
    let output = std::process::Command::new(python)
        .args(["scripts/market_data.py", "quote", symbol])
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .ok()?;

    let elapsed_ms = t0.elapsed().as_millis();

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("Python quote failed for {}: {}", symbol, stderr.chars().take(200).collect::<String>());
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let data: serde_json::Value = serde_json::from_str(stdout.trim()).ok()?;

    if data.get("error").is_some() {
        warn!("akshare quote error for {}: {}", symbol, data["error"]);
        return None;
    }

    let close = data["price"].as_f64()?;
    let open = data["open"].as_f64().unwrap_or(close);
    let high = data["high"].as_f64().unwrap_or(close);
    let low = data["low"].as_f64().unwrap_or(close);
    let volume = data["volume"].as_f64().unwrap_or(0.0);

    info!("📡 {} ¥{:.2} (akshare, {}ms)", symbol, close, elapsed_ms);

    Some(Kline {
        symbol: symbol.to_string(),
        datetime: Utc::now().naive_utc(),
        open, high, low, close, volume,
    })
}

/// Fetch historical klines for cache warmup.
fn python_bridge_warmup(python: &str, symbol: &str, days: usize) -> Vec<Kline> {
    use chrono::NaiveDateTime;

    let script = std::path::Path::new("scripts/market_data.py");
    if !script.exists() { return Vec::new(); }

    let end = chrono::Local::now().format("%Y-%m-%d").to_string();
    let start = (chrono::Local::now() - chrono::Duration::days(days as i64 * 2))
        .format("%Y-%m-%d").to_string();

    let output = std::process::Command::new(python)
        .args(["scripts/market_data.py", "klines", symbol, &start, &end])
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let arr: Vec<serde_json::Value> = match serde_json::from_str(stdout.trim()) {
        Ok(a) => a,
        Err(_) => return Vec::new(),
    };

    let mut klines = Vec::with_capacity(arr.len());
    for item in &arr {
        let dt_str = item["datetime"].as_str().unwrap_or("");
        let datetime = NaiveDateTime::parse_from_str(dt_str, "%Y-%m-%d %H:%M:%S")
            .unwrap_or_else(|_| Utc::now().naive_utc());
        klines.push(Kline {
            symbol: symbol.to_string(),
            datetime,
            open: item["open"].as_f64().unwrap_or(0.0),
            high: item["high"].as_f64().unwrap_or(0.0),
            low: item["low"].as_f64().unwrap_or(0.0),
            close: item["close"].as_f64().unwrap_or(0.0),
            volume: item["volume"].as_f64().unwrap_or(0.0),
        });
    }
    // Keep only last N days
    let start_idx = klines.len().saturating_sub(days);
    klines[start_idx..].to_vec()
}

async fn data_actor_python_bridge(
    symbols: Vec<String>,
    interval_secs: u64,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    let python = match find_python_for_bridge() {
        Some(p) => p,
        None => {
            error!("DataActor[python_bridge]: Python not found, cannot start");
            return;
        }
    };

    // Phase 1: Warm up with historical data (60+ bars per symbol)
    info!("📊 Warming up with historical data...");
    for sym in &symbols {
        let warmup_bars = tokio::task::spawn_blocking({
            let python = python.clone();
            let sym = sym.clone();
            move || python_bridge_warmup(&python, &sym, 80)
        }).await.unwrap_or_default();

        if warmup_bars.is_empty() {
            warn!("⚠️ No warmup data for {}", sym);
        } else {
            info!("📊 Warmup: {} loaded {} historical bars ({}..{})",
                sym, warmup_bars.len(),
                warmup_bars.first().map(|k| k.datetime.format("%Y-%m-%d").to_string()).unwrap_or_default(),
                warmup_bars.last().map(|k| k.datetime.format("%Y-%m-%d").to_string()).unwrap_or_default(),
            );
            // Send historical bars rapidly to warm up strategy indicators
            for kline in warmup_bars {
                if tx.send(MarketEvent::Bar(kline)).await.is_err() {
                    return;
                }
            }
        }
    }
    info!("📊 Warmup complete, switching to real-time polling ({}s interval)", interval_secs);

    // Phase 2: Real-time polling
    loop {
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)) => {
                for sym in &symbols {
                    let kline = {
                        let python = python.clone();
                        let sym = sym.clone();
                        tokio::task::spawn_blocking(move || {
                            python_bridge_quote(&python, &sym)
                        }).await.unwrap_or(None)
                    };

                    if let Some(k) = kline {
                        if tx.send(MarketEvent::Bar(k)).await.is_err() {
                            info!("DataActor[python_bridge]: channel closed");
                            return;
                        }
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("DataActor[python_bridge]: shutdown");
                    return;
                }
            }
        }
    }
}

// ── Low-Latency Data Actor (TCP Message Queue) ─────────────────────
// Uses persistent market_data_server.py via TCP binary protocol.
// Protocol: [4 bytes: msg_len (big-endian u32)] [msg_len bytes: JSON]
// Warmup: sends {"cmd":"warmup"} → receives kline stream.
// Real-time: sends {"cmd":"subscribe"} → receives pushed quotes.

async fn data_actor_low_latency(
    symbols: Vec<String>,
    interval_secs: u64,
    server_url: String,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();
    let base = server_url.trim_end_matches('/');

    // Extract host for TCP connection (port + 1)
    let http_port: u16 = base.rsplit(':').next()
        .and_then(|p| p.parse().ok()).unwrap_or(18092);
    let tcp_port = http_port + 1; // TCP MQ on 18093
    let tcp_addr = format!("127.0.0.1:{}", tcp_port);

    // Health check via HTTP — auto-start server if not running
    match client.get(format!("{}/health", base)).send().await {
        Ok(r) if r.status().is_success() => {
            info!("📡 Market data server connected at {}", base);
        }
        _ => {
            info!("📡 Market data server not running, auto-starting...");
            if let Some(python) = find_python_for_bridge() {
                let script = std::path::Path::new("scripts/market_data_server.py");
                if script.exists() {
                    let port_str = http_port.to_string();
                    let tcp_port_str = tcp_port.to_string();
                    let _ = std::process::Command::new(&python)
                        .args(["scripts/market_data_server.py", "--port", &port_str, "--tcp-port", &tcp_port_str])
                        .env("PYTHONIOENCODING", "utf-8")
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .spawn();
                    for i in 0..10 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                        if let Ok(r) = client.get(format!("{}/health", base)).send().await {
                            if r.status().is_success() {
                                info!("📡 Server auto-started after {}ms", (i + 1) * 500);
                                break;
                            }
                        }
                    }
                } else {
                    error!("📡 scripts/market_data_server.py not found");
                    return;
                }
            } else {
                error!("📡 Python not found, cannot start market data server");
                return;
            }
        }
    }

    // Connect TCP message queue
    let stream = match tokio::net::TcpStream::connect(&tcp_addr).await {
        Ok(s) => {
            info!("🔗 TCP MQ connected at {}", tcp_addr);
            s
        }
        Err(e) => {
            warn!("🔗 TCP MQ failed at {}: {}, falling back to HTTP polling", tcp_addr, e);
            data_actor_low_latency_http(symbols, interval_secs, base.to_string(), tx, shutdown).await;
            return;
        }
    };

    // Set TCP_NODELAY for minimum latency
    let _ = stream.set_nodelay(true);
    let (mut reader, mut writer) = stream.into_split();

    // No Mutex: warmup uses writer then drops; push loop uses reader directly

    // Read the "connected" message
    if let Some(msg) = tcp_mq_read(&mut reader).await {
        debug!("TCP MQ: {:?}", msg.get("type"));
    }

    // Phase 1: Warmup via TCP
    info!("📊 Warming up via TCP MQ...");
    for sym in &symbols {
        let warmup_msg = serde_json::json!({
            "cmd": "warmup",
            "symbol": sym,
            "days": 80,
            "period": "daily",
        });
        tcp_mq_write(&mut writer, &warmup_msg).await;

        // Read klines until warmup_done
        let mut count = 0;
        loop {
            match tcp_mq_read(&mut reader).await {
                Some(data) => {
                    let msg_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if msg_type == "kline" {
                        if let Some(kline) = parse_kline_json(sym, &data) {
                            if tx.send(MarketEvent::Bar(kline)).await.is_err() { return; }
                            count += 1;
                        }
                    } else if msg_type == "warmup_done" {
                        break;
                    }
                }
                None => break,
            }
        }
        info!("📊 Warmup: {} loaded {} bars via TCP MQ", sym, count);
    }

    // Phase 2: Subscribe for push updates
    let sub_msg = serde_json::json!({
        "cmd": "subscribe",
        "symbols": symbols,
    });
    tcp_mq_write(&mut writer, &sub_msg).await;
    info!("📊 Subscribed to {} symbols via TCP MQ, receiving push updates", symbols.len());

    // Drop writer — not needed after subscribe (push-only mode)
    drop(writer);

    // Phase 3: Receive pushed quotes (lock-free, no Mutex)
    loop {
        tokio::select! {
            msg = tcp_mq_read(&mut reader) => {
                match msg {
                    Some(data) => {
                        let msg_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if msg_type == "quote" {
                            let close = data.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            if close <= 0.0 { continue; }
                            let sym = data.get("symbol").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            if sym.is_empty() { continue; }
                            let kline = Kline {
                                symbol: sym.clone(),
                                datetime: Utc::now().naive_utc(),
                                open: data.get("open").and_then(|v| v.as_f64()).unwrap_or(close),
                                high: data.get("high").and_then(|v| v.as_f64()).unwrap_or(close),
                                low: data.get("low").and_then(|v| v.as_f64()).unwrap_or(close),
                                close,
                                volume: data.get("volume").and_then(|v| v.as_f64()).unwrap_or(0.0),
                            };
                            trace!("📡 TCP {} ¥{:.2}", sym, close);
                            if tx.send(MarketEvent::Bar(kline)).await.is_err() {
                                info!("DataActor[tcp_mq]: channel closed");
                                return;
                            }
                        }
                    }
                    None => {
                        // TCP connection lost — retry with exponential backoff
                        warn!("DataActor[tcp_mq]: connection lost, retrying...");
                        let mut backoff_ms = 500u64;
                        let max_backoff_ms = 30_000u64;
                        let mut reconnected = false;
                        for attempt in 1..=10 {
                            tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                            if shutdown.has_changed().unwrap_or(false) && *shutdown.borrow() { return; }
                            match tokio::net::TcpStream::connect(&tcp_addr).await {
                                Ok(new_stream) => {
                                    let _ = new_stream.set_nodelay(true);
                                    let (mut new_reader, mut new_writer) = new_stream.into_split();
                                    // Re-read connected msg
                                    let _ = tcp_mq_read(&mut new_reader).await;
                                    // Re-subscribe
                                    let sub = serde_json::json!({"cmd":"subscribe","symbols":symbols});
                                    tcp_mq_write(&mut new_writer, &sub).await;
                                    drop(new_writer);
                                    reader = new_reader;
                                    info!("🔗 TCP MQ reconnected after attempt {}", attempt);
                                    reconnected = true;
                                    break;
                                }
                                Err(e) => {
                                    warn!("🔗 TCP reconnect attempt {}/10 failed: {}", attempt, e);
                                    backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                                }
                            }
                        }
                        if !reconnected {
                            warn!("🔗 TCP MQ failed 10 retries, falling back to HTTP polling");
                            data_actor_low_latency_http(symbols, interval_secs, base.to_string(), tx, shutdown).await;
                            return;
                        }
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("DataActor[tcp_mq]: shutdown");
                    return;
                }
            }
        }
    }
}

/// TCP MQ: write a length-prefixed JSON message (direct, no Mutex).
async fn tcp_mq_write(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    msg: &serde_json::Value,
) {
    use tokio::io::AsyncWriteExt;
    let data = serde_json::to_vec(msg).unwrap_or_default();
    let header = (data.len() as u32).to_be_bytes();
    let _ = writer.write_all(&header).await;
    let _ = writer.write_all(&data).await;
}

/// TCP MQ: read a length-prefixed JSON message (direct, no Mutex).
async fn tcp_mq_read(
    reader: &mut tokio::net::tcp::OwnedReadHalf,
) -> Option<serde_json::Value> {
    use tokio::io::AsyncReadExt;

    // Read 4-byte header
    let mut header = [0u8; 4];
    match tokio::time::timeout(
        tokio::time::Duration::from_secs(10),
        reader.read_exact(&mut header),
    ).await {
        Ok(Ok(_)) => {}
        _ => return None,
    }
    let msg_len = u32::from_be_bytes(header) as usize;
    if msg_len > 1_000_000 { return None; } // 1MB max per message

    // Read body
    let mut body = vec![0u8; msg_len];
    match tokio::time::timeout(
        tokio::time::Duration::from_secs(10),
        reader.read_exact(&mut body),
    ).await {
        Ok(Ok(_)) => {}
        _ => return None,
    }

    serde_json::from_slice(&body).ok()
}

/// HTTP fallback for low-latency mode (when TCP MQ is unavailable).
async fn data_actor_low_latency_http(
    symbols: Vec<String>,
    interval_secs: u64,
    base: String,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    info!("📊 HTTP fallback: polling {} symbols every {}s", symbols.len(), interval_secs);

    // Warmup
    let warmup_end = chrono::Local::now().format("%Y-%m-%d").to_string();
    let warmup_start = (chrono::Local::now() - chrono::Duration::days(160))
        .format("%Y-%m-%d").to_string();
    for sym in &symbols {
        let url = format!("{}/klines/{}?start={}&end={}&period=daily", base, sym, warmup_start, warmup_end);
        if let Ok(resp) = client.get(&url).send().await {
            if let Ok(data) = resp.json::<serde_json::Value>().await {
                if let Some(arr) = data.get("klines").and_then(|k| k.as_array()) {
                    let mut n = 0;
                    for kj in arr {
                        if let Some(kline) = parse_kline_json(sym, kj) {
                            if tx.send(MarketEvent::Bar(kline)).await.is_err() { return; }
                            n += 1;
                        }
                    }
                    info!("📊 HTTP warmup: {} loaded {} bars", sym, n);
                }
            }
        }
    }

    // Poll loop — parallel symbol fetching
    loop {
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)) => {
                // Fetch all symbols in parallel
                let futs: Vec<_> = symbols.iter().map(|sym| {
                    let url = format!("{}/quote/{}", base, sym);
                    let client = client.clone();
                    let sym = sym.clone();
                    async move {
                        if let Ok(resp) = client.get(&url).send().await {
                            if let Ok(data) = resp.json::<serde_json::Value>().await {
                                if data.get("error").is_some() { return None; }
                                let close = data.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                if close <= 0.0 { return None; }
                                return Some(Kline {
                                    symbol: sym,
                                    datetime: Utc::now().naive_utc(),
                                    open: data.get("open").and_then(|v| v.as_f64()).unwrap_or(close),
                                    high: data.get("high").and_then(|v| v.as_f64()).unwrap_or(close),
                                    low: data.get("low").and_then(|v| v.as_f64()).unwrap_or(close),
                                    close,
                                    volume: data.get("volume").and_then(|v| v.as_f64()).unwrap_or(0.0),
                                });
                            }
                        }
                        None
                    }
                }).collect();
                let results = futures::future::join_all(futs).await;
                for kline_opt in results {
                    if let Some(kline) = kline_opt {
                        if tx.send(MarketEvent::Bar(kline)).await.is_err() { return; }
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("DataActor[http_fallback]: shutdown");
                    return;
                }
            }
        }
    }
}

/// Parse a kline JSON object from the market data server.
fn parse_kline_json(symbol: &str, kj: &serde_json::Value) -> Option<Kline> {
    let close = kj.get("close").and_then(|v| v.as_f64())?;
    let date_str = kj.get("date").and_then(|v| v.as_str())?;
    // Parse date without format! allocation — try date+time first, then date-only
    let datetime = if date_str.len() >= 19 {
        chrono::NaiveDateTime::parse_from_str(&date_str[..19], "%Y-%m-%d %H:%M:%S").ok()
    } else {
        None
    }.unwrap_or_else(|| {
        chrono::NaiveDate::parse_from_str(&date_str[..10.min(date_str.len())], "%Y-%m-%d")
            .map(|d| d.and_hms_opt(15, 0, 0).unwrap())
            .unwrap_or_else(|_| Utc::now().naive_utc())
    });

    Some(Kline {
        symbol: symbol.to_string(),
        datetime,
        open: kj.get("open").and_then(|v| v.as_f64()).unwrap_or(close),
        high: kj.get("high").and_then(|v| v.as_f64()).unwrap_or(close),
        low: kj.get("low").and_then(|v| v.as_f64()).unwrap_or(close),
        close,
        volume: kj.get("volume").and_then(|v| v.as_f64()).unwrap_or(0.0),
    })
}

// ── Historical Replay Data Actor ────────────────────────────────────
// Loads real historical klines via Python/akshare and replays them.
// speed=0 → as fast as possible, speed=1 → real-time pace, speed=10 → 10x

async fn data_actor_replay(
    symbols: Vec<String>,
    start_date: String,
    end_date: String,
    speed: f64,
    period: String,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    let python = match find_python_for_bridge() {
        Some(p) => p,
        None => {
            error!("DataActor[replay]: Python not found");
            return;
        }
    };

    // Load all historical klines for each symbol
    let mut all_klines: Vec<Kline> = Vec::new();

    for sym in &symbols {
        let period_label = match period.as_str() {
            "1" => "1分钟", "5" => "5分钟", "15" => "15分钟",
            "30" => "30分钟", "60" => "60分钟", _ => "日线",
        };
        info!("📂 Loading {} historical data for {} ({} → {})", period_label, sym, start_date, end_date);
        let klines = {
            let python = python.clone();
            let sym = sym.clone();
            let start = start_date.clone();
            let end = end_date.clone();
            let p = period.clone();
            tokio::task::spawn_blocking(move || {
                python_bridge_klines_range(&python, &sym, &start, &end, &p)
            }).await.unwrap_or_default()
        };
        if klines.is_empty() {
            warn!("⚠️ No historical data for {} in range {} → {}", sym, start_date, end_date);
        } else {
            info!("📂 {} loaded {} bars ({}..{})",
                sym, klines.len(),
                klines.first().map(|k| k.datetime.format("%Y-%m-%d").to_string()).unwrap_or_default(),
                klines.last().map(|k| k.datetime.format("%Y-%m-%d").to_string()).unwrap_or_default(),
            );
            all_klines.extend(klines);
        }
    }

    if all_klines.is_empty() {
        error!("DataActor[replay]: No data loaded for any symbol");
        return;
    }

    // Sort by datetime for chronological replay
    all_klines.sort_by_key(|k| k.datetime);

    let total = all_klines.len();
    info!("🔄 Replaying {} bars (speed={}x)", total, if speed == 0.0 { "max".to_string() } else { format!("{}", speed) });

    // Calculate delay between bars
    // For daily data: 1 trading day = ~6.5 hours. At speed=1, we compress to interval_secs.
    // We use the actual datetime gaps between bars.
    let mut prev_dt: Option<chrono::NaiveDateTime> = None;

    for (i, kline) in all_klines.into_iter().enumerate() {
        // Check shutdown
        if shutdown.has_changed().unwrap_or(false) && *shutdown.borrow() {
            info!("DataActor[replay]: shutdown at bar {}/{}", i, total);
            return;
        }

        // Compute delay based on time gap and speed
        if speed > 0.0 {
            if let Some(prev) = prev_dt {
                let gap = kline.datetime.signed_duration_since(prev);
                let gap_ms = gap.num_milliseconds().max(0) as f64;
                // Scale: at speed=1, 1 day gap → 1 second delay
                // At speed=0.1, 1 day gap → 10 second delay
                // At speed=100, 1 day gap → 10ms delay
                let delay_ms = (gap_ms / (86_400_000.0 / 1000.0)) / speed;
                let delay_ms = delay_ms.min(5000.0).max(1.0) as u64; // cap at 5s
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            }
        }
        // speed=0 → no delay, full speed

        prev_dt = Some(kline.datetime);

        if tx.send(MarketEvent::Bar(kline)).await.is_err() {
            info!("DataActor[replay]: channel closed at bar {}/{}", i, total);
            return;
        }

        // Log progress every 10%
        if total > 100 && i % (total / 10) == 0 {
            info!("🔄 Replay progress: {}/{} ({:.0}%)", i, total, (i as f64 / total as f64) * 100.0);
        }
    }

    info!("✅ Replay complete: {} bars processed", total);

    // Keep alive until shutdown so status shows "running" until user stops
    loop {
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {}
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("DataActor[replay]: shutdown after completion");
                    return;
                }
            }
        }
    }
}

/// Fetch historical klines for a specific date range via Python bridge.
fn python_bridge_klines_range(python: &str, symbol: &str, start: &str, end: &str, period: &str) -> Vec<Kline> {
    use chrono::NaiveDateTime;

    let script = std::path::Path::new("scripts/market_data.py");
    if !script.exists() { return Vec::new(); }

    let mut cmd_args = vec!["scripts/market_data.py", "klines", symbol, start, end];
    if period != "daily" {
        cmd_args.push(period);
    }
    let output = std::process::Command::new(python)
        .args(&cmd_args)
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            warn!("Python klines failed for {}: {}", symbol, stderr.chars().take(200).collect::<String>());
            return Vec::new();
        }
        Err(e) => {
            warn!("Failed to spawn python for {}: {}", symbol, e);
            return Vec::new();
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let arr: Vec<serde_json::Value> = match serde_json::from_str(stdout.trim()) {
        Ok(a) => a,
        Err(_) => return Vec::new(),
    };

    let mut klines = Vec::with_capacity(arr.len());
    for item in &arr {
        let dt_str = item["datetime"].as_str().unwrap_or("");
        let datetime = NaiveDateTime::parse_from_str(dt_str, "%Y-%m-%d %H:%M:%S")
            .unwrap_or_else(|_| Utc::now().naive_utc());
        klines.push(Kline {
            symbol: symbol.to_string(),
            datetime,
            open: item["open"].as_f64().unwrap_or(0.0),
            high: item["high"].as_f64().unwrap_or(0.0),
            low: item["low"].as_f64().unwrap_or(0.0),
            close: item["close"].as_f64().unwrap_or(0.0),
            volume: item["volume"].as_f64().unwrap_or(0.0),
        });
    }
    klines
}

// ── L2 Data Actor ───────────────────────────────────────────────────
// Receives Level-2 tick/depth data via TCP MQ from L2 sidecar.
// Same binary protocol: [4B len + JSON] with type: "tick" | "depth".
// Also accepts type: "kline" for L1 bars (hybrid mode).

async fn data_actor_l2(
    symbols: Vec<String>,
    l2_addr: String,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    // Connect to L2 TCP MQ server
    let stream = match tokio::net::TcpStream::connect(&l2_addr).await {
        Ok(s) => {
            info!("🔗 L2 TCP connected at {}", l2_addr);
            s
        }
        Err(e) => {
            error!("🔗 L2 TCP failed at {}: {}", l2_addr, e);
            return;
        }
    };
    let _ = stream.set_nodelay(true);
    let (mut reader, mut writer) = stream.into_split();

    // Read welcome message
    if let Some(msg) = tcp_mq_read(&mut reader).await {
        debug!("L2 TCP: {:?}", msg.get("type"));
    }

    // Subscribe to symbols
    let sub_msg = serde_json::json!({
        "cmd": "subscribe",
        "symbols": symbols,
        "level": "l2",
    });
    tcp_mq_write(&mut writer, &sub_msg).await;
    drop(writer);
    info!("📊 L2 subscribed to {} symbols, waiting for tick/depth push", symbols.len());

    // Receive L2 events (lock-free, push-only)
    loop {
        tokio::select! {
            msg = tcp_mq_read(&mut reader) => {
                match msg {
                    Some(data) => {
                        let msg_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        let event = match msg_type {
                            "tick" => parse_l2_tick(&data).map(MarketEvent::Tick),
                            "depth" => parse_l2_depth(&data).map(MarketEvent::Depth),
                            "kline" | "quote" => {
                                // Hybrid: also accept L1 bars
                                let sym = data.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
                                parse_kline_json(sym, &data).map(MarketEvent::Bar)
                            }
                            _ => None,
                        };
                        if let Some(ev) = event {
                            if tx.send(ev).await.is_err() {
                                info!("DataActor[l2]: channel closed");
                                return;
                            }
                        }
                    }
                    None => {
                        warn!("DataActor[l2]: connection lost");
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                        return;
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("DataActor[l2]: shutdown");
                    return;
                }
            }
        }
    }
}

/// Parse L2 tick (逐笔成交) from JSON.
fn parse_l2_tick(data: &serde_json::Value) -> Option<TickData> {
    let symbol = data.get("symbol").and_then(|v| v.as_str())?.to_string();
    let price = data.get("price").and_then(|v| v.as_f64())?;
    let volume = data.get("volume").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let direction = data.get("direction").and_then(|v| v.as_str())
        .and_then(|s| s.chars().next()).unwrap_or(' ');
    let seq = data.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
    let bid1 = data.get("bid1").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let ask1 = data.get("ask1").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let datetime = if let Some(ts) = data.get("datetime").and_then(|v| v.as_str()) {
        chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S%.f")
            .unwrap_or_else(|_| Utc::now().naive_utc())
    } else {
        Utc::now().naive_utc()
    };

    Some(TickData { symbol, datetime, price, volume, direction, seq, bid1, ask1 })
}

/// Parse L2 depth (盘口) from JSON.
fn parse_l2_depth(data: &serde_json::Value) -> Option<DepthData> {
    let symbol = data.get("symbol").and_then(|v| v.as_str())?.to_string();
    let last_price = data.get("last_price").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let total_volume = data.get("total_volume").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let total_turnover = data.get("total_turnover").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let parse_levels = |key: &str| -> Vec<DepthLevel> {
        data.get(key).and_then(|v| v.as_array()).map(|arr| {
            arr.iter().filter_map(|lv| {
                let price = lv.get("price").or(lv.get("p")).and_then(|v| v.as_f64())?;
                let volume = lv.get("volume").or(lv.get("v")).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let order_count = lv.get("order_count").or(lv.get("n"))
                    .and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                Some(DepthLevel { price, volume, order_count })
            }).collect()
        }).unwrap_or_default()
    };

    let bids = parse_levels("bids");
    let asks = parse_levels("asks");

    let datetime = if let Some(ts) = data.get("datetime").and_then(|v| v.as_str()) {
        chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S%.f")
            .unwrap_or_else(|_| Utc::now().naive_utc())
    } else {
        Utc::now().naive_utc()
    };

    Some(DepthData { symbol, datetime, bids, asks, last_price, total_volume, total_turnover })
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
                        current_price: ts.price, // approximate with latest bar price
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
                        }
                        error!("❌ Submit failed: {} — {}", order.symbol, e);
                        if let Some(ref j) = journal {
                            j.record_risk_rejected(&order.symbol, order.side, order.quantity, order.price,
                                &format!("submit_error: {}", e));
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
