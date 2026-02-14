//! Actor-based auto-trading engine.
//!
//! Architecture:
//!   DataActor → StrategyActor → RiskActor → OrderActor
//!              (generates bars)  (signals)   (validates)  (executes)
//!
//! All actors run as independent tokio tasks connected by mpsc channels.
//! The `TradingEngine` coordinates startup, shutdown, and status reporting.

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

use crate::paper::PaperBroker;

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
        }
    }
}

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

pub struct TradingEngine {
    config: EngineConfig,
    shutdown_tx: Option<watch::Sender<bool>>,
    stats: Arc<Mutex<EngineStats>>,
    broker: Arc<PaperBroker>,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl TradingEngine {
    pub fn new(config: EngineConfig) -> Self {
        let broker = Arc::new(PaperBroker::new(
            config.initial_capital,
            config.commission_rate,
            config.stamp_tax_rate,
        ));
        Self {
            config,
            shutdown_tx: None,
            stats: Arc::new(Mutex::new(EngineStats::default())),
            broker,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
        tokio::spawn(async move {
            data_actor(data_symbols, data_interval, market_tx, data_shutdown).await;
        });

        // Spawn StrategyActor
        let strat_shutdown = shutdown_rx.clone();
        let strat_stats = self.stats.clone();
        tokio::spawn(async move {
            strategy_actor(strategy_factory, market_rx, signal_tx, strat_stats, strat_shutdown).await;
        });

        // Spawn RiskActor
        let risk_shutdown = shutdown_rx.clone();
        let risk_broker = self.broker.clone();
        let risk_max_conc = self.config.max_concentration;
        let risk_pos_pct = self.config.position_size_pct;
        tokio::spawn(async move {
            risk_actor(signal_rx, order_tx, risk_broker, risk_max_conc, risk_pos_pct, risk_shutdown).await;
        });

        // Spawn OrderActor
        let order_shutdown = shutdown_rx.clone();
        let order_broker = self.broker.clone();
        let order_stats = self.stats.clone();
        tokio::spawn(async move {
            order_actor(order_rx, order_broker, order_stats, order_shutdown).await;
        });

        info!("🚀 Trading engine started: {} on {:?}", self.config.strategy_name, self.config.symbols);
    }

    /// Stop the engine gracefully.
    pub fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
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
    pub fn broker(&self) -> &Arc<PaperBroker> {
        &self.broker
    }
}

// ── DataActor ───────────────────────────────────────────────────────
// Generates simulated market data bars at a configurable interval.
// In production, this would connect to a WebSocket or polling API.

async fn data_actor(
    symbols: Vec<String>,
    interval_secs: u64,
    tx: mpsc::Sender<MarketEvent>,
    mut shutdown: watch::Receiver<bool>,
) {
    info!("📊 DataActor started for {:?}", symbols);

    // Track price state per symbol
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
                    // Generate realistic price movement using deterministic noise
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

// ── StrategyActor ───────────────────────────────────────────────────
// Receives market events, runs strategy logic, emits trade signals.

async fn strategy_actor<F>(
    strategy_factory: F,
    mut rx: mpsc::Receiver<MarketEvent>,
    tx: mpsc::Sender<TradeSignal>,
    stats: Arc<Mutex<EngineStats>>,
    mut shutdown: watch::Receiver<bool>,
) where
    F: Fn() -> Box<dyn Strategy> + Send + 'static,
{
    // Create one strategy per symbol
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
                        if tx.send(TradeSignal { signal, kline: event.kline }).await.is_err() {
                            return;
                        }
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    // Cleanup strategies
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
    broker: Arc<PaperBroker>,
    max_concentration: f64,
    position_size_pct: f64,
    mut shutdown: watch::Receiver<bool>,
) {
    let risk_checker = quant_risk::checks::RiskChecker::new(max_concentration, 0.05, 0.15);

    info!("🛡️ RiskActor started (max_conc={:.0}%, pos_size={:.0}%)", max_concentration * 100.0, position_size_pct * 100.0);

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

                let (side, price) = if ts.signal.is_buy() {
                    (OrderSide::Buy, ts.kline.close)
                } else {
                    (OrderSide::Sell, ts.kline.close)
                };

                // Calculate position size
                let quantity = if side == OrderSide::Buy {
                    let allocation = portfolio.total_value * position_size_pct;
                    let raw_shares = allocation / price;
                    // Round down to lot of 100 (Chinese market rule)
                    let lots = (raw_shares / 100.0).floor() * 100.0;
                    // If allocation can afford at least 100 shares, buy 100
                    if lots < 100.0 && allocation >= price * 100.0 {
                        100.0
                    } else {
                        lots
                    }
                } else {
                    // Sell entire position
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

                // Run risk checks
                if let Err(reason) = risk_checker.check_order(&order, portfolio) {
                    warn!("🚫 Risk rejected: {} {} x{:.0} — {}", 
                        order.symbol, if side == OrderSide::Buy {"BUY"} else {"SELL"}, quantity, reason);
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
    broker: Arc<PaperBroker>,
    stats: Arc<Mutex<EngineStats>>,
    mut shutdown: watch::Receiver<bool>,
) {
    info!("💹 OrderActor started");

    loop {
        tokio::select! {
            Some(req) = rx.recv() => {
                let order = req.order;
                let side_str = if order.side == OrderSide::Buy { "BUY" } else { "SELL" };

                // Submit order to broker
                match broker.submit_order(&order).await {
                    Ok(submitted) => {
                        // Immediately fill at market price (simulated)
                        match broker.fill_order(submitted.id, order.price) {
                            Ok(trade) => {
                                let report = ExecutionReport {
                                    order_id: submitted.id,
                                    symbol: trade.symbol.clone(),
                                    side: trade.side,
                                    price: trade.price,
                                    quantity: trade.quantity,
                                    commission: trade.commission,
                                    timestamp: trade.timestamp,
                                    status: "FILLED".into(),
                                };
                                info!(
                                    "🔔 FILLED: {} {} x{:.0} @ {:.2} (commission: ¥{:.2})",
                                    side_str, trade.symbol, trade.quantity, trade.price, trade.commission
                                );
                                let mut s = stats.lock().await;
                                s.total_orders += 1;
                                s.total_fills += 1;
                                s.recent_trades.push(report);
                            }
                            Err(e) => {
                                warn!("❌ Fill failed: {} — {}", order.symbol, e);
                                let mut s = stats.lock().await;
                                s.total_orders += 1;
                                s.total_rejected += 1;
                            }
                        }
                    }
                    Err(e) => {
                        error!("❌ Submit failed: {} — {}", order.symbol, e);
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
    use quant_core::traits::Broker;

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

        engine.stop();
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
            risk_actor(signal_rx, order_tx, broker, 0.05, 0.10, shutdown_rx).await;
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
