---
name: backend
description: Expert on the Rust backend — crates, API endpoints, Python sidecars, data pipeline, ML inference, risk management, and trading engine architecture.
tools:
  - thinking
  - editor
  - terminal
  - file_search
---

# Backend Agent Guide — Quant Trading System

> **Purpose**: This document helps AI agents and developers understand the Rust backend architecture, conventions, and patterns for effective development.

## System Overview

A full-stack quantitative trading system targeting **Chinese A-shares** (with US/HK support). The backend is a Rust workspace with 11 crates, Python sidecars for data/ML, and an Axum REST/WebSocket API serving a React frontend.

```
quant-trading/
├── crates/           # 11-crate Rust workspace (core engine)
├── scripts/          # Python data providers & market cache
├── ml_models/        # Python ML training, inference server, factor mining
├── qmt_bridge/       # Python QMT live trading bridge (xtquant SDK)
├── config/           # TOML configuration
├── migrations/       # PostgreSQL schema
├── data/             # SQLite caches, L2 recordings, ML model files
└── tests/            # Integration tests
```

## Crate Dependency Graph

```
cli ──→ api ──→ broker ──→ risk ──→ strategy ──→ ml ──→ core
                  │                    │                    │
                  └──→ llm             └──→ data ──→ config ┘
```

---

## Crate-by-Crate Reference

### 1. `quant-core` — Foundation Types & Traits

**Path**: `crates/core/src/` — Files: `lib.rs`, `types.rs`, `models.rs`, `traits.rs`, `error.rs`, `utils.rs`

#### Key Enumerations (`types.rs`)
| Type | Variants | Notes |
|------|----------|-------|
| `Market` | `SH`, `SZ`, `US`, `HK` | `from_symbol()` auto-detects; `.region()` → "CN"/"US"/"HK" |
| `OrderSide` | `Buy`, `Sell` | |
| `OrderType` | `Market`, `Limit` | |
| `OrderStatus` | `Pending`, `Submitted`, `PartiallyFilled`, `Filled`, `Cancelled`, `Rejected` | |
| `TimeFrame` | `Min1`..`Monthly` | |
| `SignalAction` | `Buy`, `Sell`, `Hold` | |

#### Core Structs (`models.rs`)
- **`Kline`** — OHLCV bar: `{symbol, datetime, open, high, low, close, volume}`
- **`Tick`** — Real-time quote: `{symbol, datetime, price, volume, bid, ask}`
- **`TickData`** — L2 tick-by-tick: `{symbol, datetime, price, volume, direction, seq, bid1, ask1}`
- **`DepthData`** — L2 order book: `{symbol, datetime, bids: Vec<DepthLevel>, asks: Vec<DepthLevel>, last_price, total_volume}`
- **`Signal`** — `{action, confidence: f64, symbol, timestamp}` with convenience constructors `buy()`, `sell()`, `hold()`
- **`Order`** — `{id: Uuid, symbol, side, order_type, price, quantity, filled_qty, status, created_at, updated_at}`
- **`Trade`** — `{id: Uuid, order_id, symbol, side, price, quantity, commission, timestamp}`
- **`Position`** — `{symbol, quantity, avg_cost, current_price, unrealized_pnl, realized_pnl, entry_time, scale_level, target_weight}`
- **`ClosedPosition`** — `{symbol, entry_time, exit_time, entry_price, exit_price, quantity, realized_pnl, holding_days}`
- **`Portfolio`** — `{positions: HashMap<String, Position>, cash, total_value}`
- **`SentimentItem`** — `{id: Uuid, symbol, source, title, content, sentiment_score: f64 [-1,+1], published_at}`
- **`SentimentSummary`** — `{symbol, count, avg_score, level, bullish/bearish/neutral counts}`

#### Core Traits (`traits.rs`)
```rust
pub trait DataProvider: Send + Sync {
    async fn fetch_kline(&self, symbol: &str, start: &str, end: &str, timeframe: TimeFrame) -> Result<Vec<Kline>>;
    async fn fetch_stock_list(&self) -> Result<Vec<StockInfo>>;
    async fn fetch_realtime_quote(&self, symbol: &str) -> Result<Tick>;
}

pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn on_init(&mut self);
    fn on_bar(&mut self, kline: &Kline) -> Option<Signal>;
    fn on_tick(&mut self, _tick: &TickData) -> Option<Signal> { None }   // L2
    fn on_depth(&mut self, _depth: &DepthData) -> Option<Signal> { None } // L2
    fn on_stop(&mut self);
}

pub trait Broker: Send + Sync {
    async fn submit_order(&self, order: &Order) -> Result<Order>;
    async fn cancel_order(&self, order_id: Uuid) -> Result<()>;
    async fn get_positions(&self) -> Result<Vec<Position>>;
    async fn get_account(&self) -> Result<Account>;
    fn as_any(&self) -> &dyn std::any::Any; // downcast support
}

pub trait RiskManager: Send + Sync {
    fn check_order(&self, order: &Order, portfolio: &Portfolio) -> Result<()>;
    fn calculate_position_size(&self, signal: &Signal, portfolio: &Portfolio) -> f64;
}
```

#### Error Handling (`error.rs`)
```rust
pub enum QuantError {
    DataError(String), StrategyError(String), BrokerError(String),
    RiskError(String), ConfigError(String), DatabaseError(String),
    LlmError(String), NetworkError(String),
}
pub type Result<T> = std::result::Result<T, QuantError>;
```
- Implements `From` for `sqlx::Error`, `reqwest::Error`, `serde_json::Error`

#### Utilities (`utils.rs`)
- `find_python()` — Locates Python 3: checks `PYTHON_PATH` env → platform default → `python3` → `python`
- `run_python_script(python, args)` — Execute Python script, return JSON stdout
- `parse_date(s)` / `parse_date_compact(s)` — "YYYY-MM-DD" / "YYYYMMDD"

---

### 2. `quant-config` — Configuration Management

**Path**: `crates/config/src/lib.rs` (211 lines)

#### Root Config: `AppConfig`
```rust
pub struct AppConfig {
    pub database: DatabaseConfig,     // PostgreSQL url, max_connections
    pub tushare: TushareConfig,       // token, base_url
    pub akshare: AkshareConfig,       // base_url
    pub llm: LlmConfig,              // api_url, api_key, model, temperature, max_tokens
    pub trading: TradingConfig,       // commission, stamp_tax, slippage, initial_capital, per-market fees
    pub risk: RiskConfig,             // max_concentration, max_daily_loss, max_drawdown
    pub qmt: QmtConfig,              // bridge_url, account, qmt_path
    pub server: ServerConfig,         // host, port (default 8080)
    pub sentiment: SentimentConfig,   // collect_interval, watch_symbols, news_sources
    pub data_source: DataSourceConfig, // cn/us/hk provider lists, cache_only flag
}
```

**Key patterns**:
- `TradingConfig::fees_for_market("CN")` → `MarketFees` with per-market commission/stamp_tax/slippage
- `TradingConfig::fees_for_symbol("600519.SH")` → auto-detects market, returns appropriate fees
- `DataSourceConfig::providers_for_market("CN")` → `&["tushare", "akshare"]`
- Loading: `AppConfig::from_file(path)` or `AppConfig::from_default()` (reads `config/default.toml`)

**Per-market fee overrides** (config/default.toml):
- CN: commission=0.025%, stamp_tax=0.1% (sell-only)
- US: commission=0.1%, stamp_tax=0%
- HK: commission=0.08%, stamp_tax=0.1%

---

### 3. `quant-data` — Market Data Providers

**Path**: `crates/data/src/` — Files: `lib.rs`, `provider.rs`, `tushare.rs`, `akshare.rs`, `storage.rs`, `news.rs`

- **`DataProviderImpl`**: Wraps TushareClient + AkshareClient + PgPool. Fetches klines (Tushare first, AKShare fallback), caches to PostgreSQL.
- **`TushareClient`**: JSON API to `api.tushare.pro`. Methods: `fetch_daily()`, `fetch_stock_basic()`, `fetch_realtime_quote()`. Uses `FieldIndex` for O(1) field lookup.
- **`AkshareClient`**: REST API. Supports bilingual field aliases (日期/date, 开盘/open). Forward-adjusted (qfq) prices.
- **`MarketDataStore`**: `save_klines()`, `get_klines()` → PostgreSQL with `create_pool()` and `run_migrations()`.
- **`NewsFetcher`**: Aggregates from East Money (via AKShare proxy) and Sina Finance (HTML parsing). Produces `Vec<NewsArticle>`.

---

### 4. `quant-ml` — Machine Learning & Inference

**Path**: `crates/ml/src/` — Files: `lib.rs`, `ml_factor.rs`, `fast_factors.rs`, `lgb_inference.rs`, `dl_models.rs`, `onnx_inference.rs`

#### Feature Engineering (`ml_factor.rs`)
- **NUM_FEATURES = 30** — Returns, volatility, MA ratios, RSI, MACD, volume, price patterns, Bollinger, ATR, z-scores
- `compute_features(bars: &[Kline]) -> Option<[f32; 30]>` — Requires ≥61 bars (for MA60). Efficient incremental computation.

#### Incremental Factors (`fast_factors.rs`)
- **`RingSum`** — O(1) rolling mean via circular buffer
- **`RingStdDev`** — Welford's algorithm for incremental std dev
- **`IncrEMA`** — O(1) exponential moving average
- Performance: ~500ns per bar vs ~10μs batch recomputation

#### LightGBM Inference (`lgb_inference.rs`)
- **`LightGBMModel`**: Loads LightGBM text format natively in Rust. `predict(features: &[f32]) -> f64`. ~0.01ms per prediction.
- **IMPORTANT**: `.lgb.txt` files MUST have LF line endings (CRLF breaks C++ parser — see `.gitattributes`).

#### Inference Modes (`ml_factor.rs`)
```rust
pub enum MlInferenceMode {
    Embedded,  // LightGBM in Rust (~0.01ms)
    Onnx,      // ONNX Runtime (~0.05ms)
    TcpMq,     // TCP binary protocol to ml_serve.py (~0.3ms)
}
```

- **`MlFactorStrategy`**: Implements `Strategy` trait. Computes 30 features → runs inference → generates Signal.
- **Fallback**: When Python sidecar unavailable, uses built-in rule-based scoring on the same 30 features.

---

### 5. `quant-strategy` — Strategy Engine

**Path**: `crates/strategy/src/` — Files: `lib.rs`, `factory.rs`, `engine.rs`, `builtin.rs`, `indicators.rs`, `signals.rs`, `screener.rs`, `sentiment.rs`

#### Strategy Factory (`factory.rs`)
**Single source of truth** — `create_strategy(name, opts) -> Result<CreatedStrategy>`:

| Name | Alias | Description |
|------|-------|-------------|
| `sma_cross` | `DualMaCrossover` | Golden/death cross (fast=5, slow=20) |
| `rsi_reversal` | `RsiMeanReversion` | Buy oversold, sell overbought (period=14) |
| `macd_trend` | `MacdMomentum` | MACD histogram zero-crossing |
| `multi_factor` | `MultiFactorModel` | 6-factor composite scoring |
| `sentiment_aware` | `SentimentAware` | Multi-factor + sentiment fusion |
| `ml_factor` | `MlFactor` | 30-feature ML model with embedded/ONNX/TCP inference |

#### Technical Indicators (`indicators.rs`)
- **SMA** (window buffer), **EMA** (smoothing factor, seeds with SMA), **MACD** (fast/slow/signal EMAs), **RSI** (Wilder's smoothing), **BollingerBands** (SMA ± k*StdDev, %B), **KDJ** (stochastic oscillator, K/D cross)

#### Multi-Factor Strategy
- 6 sub-scores weighted into composite [-1, +1]: Trend (25%), Momentum (25%), Volatility (15%), Oscillator (15%), Volume (10%), Price Action (10%)
- BUY when composite > buy_threshold; SELL when < sell_threshold

#### Stock Screener (`screener.rs`)
- 3-phase pipeline: Multi-factor scoring → Strategy signal voting → LLM analysis (optional)
- `ScreenerConfig` with criteria: `PriceAbove`, `RsiOversold`, `MaCrossAbove`, etc.

---

### 6. `quant-backtest` — Backtesting Engine

**Path**: `crates/backtest/src/` — Files: `lib.rs`, `engine.rs`, `matching.rs`, `metrics.rs`, `report.rs`, `walk_forward.rs`

#### Signal-on-bar-N, Fill-at-bar-N+1-open Paradigm
- Strategy generates signal on current bar's data
- Order created immediately
- Filled on NEXT bar's open price (with slippage) — prevents look-ahead bias

#### Matching Engine (`matching.rs`)
- Market Buy: `fill_price = kline.open + slippage`
- Market Sell: `fill_price = kline.open - slippage`
- Limit orders: Fill only if limit price is within bar's range

#### BacktestConfig
```rust
pub struct BacktestConfig {
    pub initial_capital: f64,          // Default: from config
    pub commission_rate: f64,          // Per-market
    pub stamp_tax_rate: f64,           // CN: 0.1% on sells only
    pub slippage_ticks: u32,           // Price tick slippage
    pub position_size_pct: f64,        // 10% per trade
    pub max_concentration: f64,        // 30% per stock
    pub stop_loss_pct: f64,            // 8%
    pub max_holding_days: u32,         // 30 (0 = disabled)
    pub daily_loss_limit: f64,         // 0 = disabled
    pub max_drawdown_limit: f64,       // 0 = disabled
    pub use_atr_sizing: bool,          // Inverse volatility sizing
    pub atr_period: usize,             // 14
    pub risk_per_trade: f64,           // 2%
}
```

#### BacktestEvent (tagged enum, serde)
`Signal`, `OrderCreated`, `OrderFilled`, `OrderRejected`, `RiskTriggered`, `PositionOpened`, `PositionClosed`, `PortfolioSnapshot`

#### PerformanceMetrics
`total_return`, `annual_return`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `max_drawdown_duration`, `win_rate`, `profit_factor`, `total_trades`, `avg_win/loss`, `avg_holding_days`, `total_commission`, `turnover_rate`

#### Walk-Forward Validation (`walk_forward.rs`)
- Splits data into N folds, trains on fold i, tests on fold i+1
- 5-bar embargo gap to prevent label leakage

---

### 7. `quant-risk` — Risk Management

**Path**: `crates/risk/src/` — Files: `lib.rs`, `pure_checks.rs`, `enforcement.rs`, `position.rs`, `checks.rs`, `rules.rs`

#### Pure Checks (`pure_checks.rs`) — Stateless, shared by BacktestEngine and RiskEnforcer
```rust
pub fn check_stop_loss(price, avg_cost, threshold) -> Option<f64>
pub fn check_daily_loss(current_value, day_start_value, threshold) -> Option<f64>
pub fn check_drawdown(current_value, peak_value, threshold) -> Option<f64>
pub fn check_holding_timeout(holding_days, max_days) -> Option<i64>
pub fn cap_by_concentration(budget, existing_value, total_value, max_concentration) -> (f64, bool)
```

#### Risk Enforcement (`enforcement.rs`) — Stateful runtime enforcer
```rust
pub struct RiskConfig {
    pub stop_loss_pct: f64,               // 5%
    pub max_daily_loss_pct: f64,          // 3%
    pub max_drawdown_pct: f64,            // 10%
    pub circuit_breaker_failures: u32,    // 5 consecutive failures → halt
    pub halt_on_drawdown: bool,           // true
    pub max_holding_days: u32,            // 30
    pub timeout_min_profit_pct: f64,      // 2% (keep profitable positions)
    pub rebalance_threshold: f64,         // 5% drift
    pub vol_spike_ratio: f64,             // 2.0 (short/long vol)
    pub vol_deleverage_factor: f64,       // 0.5 (halve positions on spike)
    pub max_correlated_exposure: f64,     // 40%
    pub var_confidence: f64,              // 95% VaR
    pub max_var_pct: f64,                 // 5%
}
```

#### Chinese Market Rules (`rules.rs`)
- T+1 settlement (cannot sell shares bought today)
- Price limits: ±10% (main board), ±20% (ChiNext/STAR)
- Stamp tax: 0.1% on sell side only
- Minimum lot size: 100 shares
- Commission: default 0.025%, minimum ¥5

---

### 8. `quant-broker` — Trading Brokers

**Path**: `crates/broker/src/` — Files: `lib.rs`, `paper.rs`, `live.rs`, `qmt.rs`, `orders.rs`, `engine.rs`, `journal.rs`, `notifier.rs`, `data_actors.rs`

#### Paper Broker (`paper.rs`)
- In-memory trading with `auto_fill` mode, configurable `slippage_bps`
- `submit_order()`, `cancel_order()`, `get_positions()`, `get_account()`, `force_close_position()`

#### QMT Broker (`qmt.rs`)
- HTTP client to `qmt_bridge.py` sidecar (Flask on port 18090)
- Delegates order submission, cancellation, position queries to QMT (迅投量化)

#### Actor-Based Trading Engine (`engine.rs`)
```
DataActor ──→ StrategyActor ──→ RiskActor ──→ OrderActor
  (bars)        (signals)       (validate)    (execute)
```

**MarketEvent**: `Bar(Kline)` | `Tick(TickData)` | `Depth(DepthData)`

**DataMode**:
- `Live` — Real-time from Tushare/AKShare
- `HistoricalReplay` — Replay cached data at configurable speed
- `L2` — Level-2 tick/depth data via TCP
- `LowLatency` — Lock-free AtomicU64/AtomicF64, no Mutex

#### Trade Journal (`journal.rs`)
- SQLite-backed persistent audit trail (data/trade_journal.db)
- Records: signals, orders, fills, rejections with timestamps
- Daily snapshots: portfolio value, cash, positions, PnL

---

### 9. `quant-llm` — LLM Integration

**Path**: `crates/llm/src/` — Files: `lib.rs`, `client.rs`, `sentiment.rs`, `tools.rs`, `context.rs`, `history.rs`

- **LlmClient**: OpenAI-compatible chat API with tool calling support. Streaming via `stream()`.
- **SentimentAnalyzer**: Sends news articles to LLM with Chinese system prompt. Returns `{score: [-1,+1], reasoning, keywords}`. Rate-limited (200ms between calls).
- **Tool Calling**: `get_kline`, `get_stock_info`, `run_backtest`, `get_portfolio`, `screen_stocks`
- **ConversationContext**: Session-based multi-turn context management.

---

### 10. `quant-api` — REST API Server (Axum)

**Path**: `crates/api/src/` — Files: `lib.rs`, `routes.rs`, `state.rs`, `auth.rs`, `log_store.rs`, `task_store.rs`, `ws.rs`, `handlers/*`

#### AppState
```rust
pub struct AppState {
    pub config: AppConfig,
    pub pool: PgPool,
    pub engine: Arc<Mutex<Option<TradingEngine>>>,
    pub log_store: LogStore,          // In-memory request log (VecDeque)
    pub task_store: TaskStore,        // Async task tracking (SQLite-backed)
    pub sentiment_store: SentimentStore,
}
```

#### Middleware
- `api_key_auth` — `Authorization: Bearer <key>` header check
- `request_logger` — Logs method, path, status, duration to LogStore

#### Complete Route Map
```
/api/health                          GET   — Health check
/api/dashboard                       GET   — Portfolio overview + metrics
/api/strategies                      GET   — List available strategies
/api/strategy/config                 GET/POST — Load/save strategy config
/api/market/kline/:symbol            GET   — K-line data (params: limit, period, start, end)
/api/market/quote/:symbol            GET   — Real-time quote
/api/market/stocks                   GET   — Stock list
/api/market/data-source              GET   — Provider status
/api/market/cache-status             GET   — Cache stats
/api/market/sync-data                POST  — Async data sync task
/api/backtest/run                    POST  — Run backtest (async task)
/api/backtest/walk-forward           POST  — Walk-forward validation (async task)
/api/backtest/results/:id            GET   — Backtest results
/api/orders                          GET   — List orders
/api/portfolio                       GET   — Open positions + cash
/api/portfolio/close                 POST  — Force close position
/api/portfolio/closed                GET   — Closed positions history
/api/trade/start                     POST  — Start auto-trade engine
/api/trade/stop                      POST  — Stop engine
/api/trade/status                    GET   — Engine status
/api/trade/performance               GET   — Performance metrics
/api/trade/risk                      GET   — Risk enforcement status
/api/trade/risk/signals              GET   — Comprehensive risk snapshot
/api/trade/risk/reset-circuit        POST  — Reset circuit breaker
/api/trade/risk/reset-daily          POST  — Reset daily loss
/api/trade/retrain                   POST  — ML model retrain (async task)
/api/trade/model-info                GET   — Model info & training report
/api/trade/qmt/status                GET   — QMT bridge status
/api/trade/ticks                     GET   — Recorded tick data
/api/ml/training-history             GET   — Training run history
/api/ml/training-history/:id         GET   — Training run detail
/api/screen/scan                     POST  — Stock screener
/api/screen/factors/:symbol          GET   — Single symbol factors
/api/sentiment/submit                POST  — Submit sentiment item
/api/sentiment/batch                 POST  — Batch submit
/api/sentiment/summary               GET   — Global sentiment overview
/api/sentiment/:symbol               GET   — Per-symbol sentiment
/api/sentiment/collector/start       POST  — Start collection daemon
/api/sentiment/collector/stop        POST  — Stop daemon
/api/sentiment/collector/status      GET   — Daemon status
/api/research/dl-models              GET   — DL model knowledge base
/api/research/dl-models/summary      GET   — Summary stats
/api/research/dl-models/collect      POST  — LLM-powered research collection
/api/factor/mine/parametric          POST  — Parametric factor mining (async)
/api/factor/mine/gp                  POST  — GP factor mining (async)
/api/factor/evaluate-manual          POST  — Evaluate custom expression (async)
/api/factor/save-manual              POST  — Save to registry
/api/factor/registry                 GET   — Factor registry state
/api/factor/manage                   POST  — Lifecycle management (async)
/api/factor/export                   POST  — Export promoted factors (async)
/api/factor/results                  GET   — Mining results
/api/factor/mining-history           GET   — Mining run history
/api/factor/mining-history/:id       GET   — Mining run detail
/api/tasks                           GET   — List all tasks
/api/tasks/running                   GET   — Running tasks only
/api/tasks/:id                       GET/DELETE — Get/cancel task
/api/services/status                 GET   — All managed services
/api/services/ml-serve/start         POST  — Start ML inference server
/api/services/ml-serve/stop          POST  — Stop ML inference server
/api/services/ml-serve/status        GET   — ML server health
/api/journal                         GET   — Trade journal entries
/api/journal/snapshots               GET   — Daily performance snapshots
/api/notifications                   GET   — List notifications
/api/notifications/unread-count      GET   — Badge count
/api/notifications/read-all          POST  — Mark all read
/api/notifications/config            GET/POST — Notification config
/api/notifications/test              POST  — Test notification channels
/api/notifications/:id/read          POST  — Mark single read
/api/logs                            GET/DELETE — Request logs
/api/metrics                         GET   — System metrics
/api/reports                         GET   — Statistical reports
/api/latency                         GET   — Pipeline latency profiling
/api/chat                            POST  — Chat message
/api/chat/history                    GET   — Chat history
/api/chat/stream                     WS    — Streaming chat
/api/monitor/ws                      WS    — Real-time risk/trade updates
```

#### Async Task Pattern
- Long-running operations return `202 Accepted` with `{ task_id }` immediately
- Frontend polls `GET /api/tasks/:id` via `useTaskPoller` hook
- TaskStore tracks: `Pending` → `Running` → `Completed`/`Failed`
- Progress updates via `set_progress(id, msg)`

#### SPA Fallback
- `/` serves `web/dist/index.html` (built React app)
- `tower-http::fs::ServeDir` for static assets

---

### 11. `quant-cli` — Command-Line Interface

**Path**: `crates/cli/src/main.rs` (400+ lines)

```
quant [--config config/default.toml] <COMMAND>
  serve                                    # Start API server on port 8080
  backtest run --strategy <name> --symbol <sym> --start <date> --end <date>
  backtest walk-forward ...
  backtest strategies                      # List strategies
  trade paper --strategy <name> --symbol <sym>
  trade auto --strategy <name> --symbols <sym1,sym2> --interval 5
  trade qmt --strategy <name> --symbols <sym>
  trade qmt-status
  screen scan --top 10 --min-votes 2
  screen factors --symbol <sym>
  sentiment fetch --symbol <sym>
  sentiment analyze --symbol <sym> --sources eastmoney,sina
  chat                                     # Interactive LLM REPL
  data sync / query / list
```

---

## Python Sidecars

### Market Data Scripts (`scripts/`)

| Script | Purpose | Invocation |
|--------|---------|------------|
| `market_data.py` | Unified JSON data bridge (klines/quote/stock_list) | Called by Rust via subprocess |
| `market_cache.py` | SQLite cache layer (`data/market_cache.db`) | `python market_cache.py warm_symbols 600519 2020-01-01 2024-12-31` |
| `market_data_server.py` | Persistent Flask+WS+TCP server (<5ms quotes) | `python market_data_server.py --port 18092 --tcp-port 18093` |
| `tushare_provider.py` | Primary CN data (token via `TUSHARE_TOKEN` env) | Imported by market_data.py |
| `yfinance_provider.py` | US/HK data (no auth needed) | Imported by market_data.py |
| `data_source_config.py` | Provider priority per market | Imported by market_data.py |
| `fetch_klines.py` | CLI kline fetcher via akshare | `python fetch_klines.py 600519 2024-01-01 2024-12-31` |
| `cache_yfinance.py` | Pre-populate cache with yfinance | `python cache_yfinance.py AAPL 2020-01-01 2024-12-31` |
| `l2_recorder.py` | L2 tick/depth recorder (QMT xtquant) | `python l2_recorder.py --mode record --symbols 600519.SH` |

**Critical convention**: `data_utils.py` loads from cache ONLY (`cache_only=True`). No ML task ever fetches from the network.

### ML Models (`ml_models/`)

| Script | Purpose | Key Output |
|--------|---------|------------|
| `train_factor_model.py` | 24-feature engineering + LightGBM training | `.lgb.txt`, `.onnx` models |
| `auto_retrain.py` | Multi-algorithm competition (LGB/XGB/CatBoost/LSTM/Transformer) | Best model + `retrain_report.json` |
| `ml_serve.py` | GPU inference server (Flask, port 18091) | REST: `/predict`, `/predict_batch`, `/reload` |
| `factor_mining.py` | Parametric factor discovery (200+ candidates) | `factor_mining_report.json`, Rust snippets |
| `gp_factor_mining.py` | Genetic programming factor evolution | `gp_factors_data.csv`, `factor_registry.json` |
| `dl_factor_mining.py` | Neural factor discovery (Autoencoder/Attention/Residual) | Latent factors evaluated via IC |
| `manual_factor_eval.py` | Safe expression evaluator (AST-validated) | IC, IR, turnover, decay metrics |
| `data_utils.py` | Unified data loader (cache-only, 60+ default stocks) | Combined DataFrames |

**ML Features must be lagged by 1 bar** (shift+1) in `auto_retrain.py` to prevent label leakage.

### QMT Bridge (`qmt_bridge/`)

- Flask HTTP server wrapping xtquant SDK for live trading
- Endpoints: `/health`, `/order`, `/cancel`, `/positions`, `/account`, `/orders`
- Async order flow: place → callback → poll `/order_result/<id>`

---

## Database Schema

### PostgreSQL (`migrations/`)
- **`stock_info`** — Master stock registry (symbol PK, name, market, industry)
- **`kline_daily`** — Daily OHLCV (symbol + datetime unique, indexed)
- **`orders`** — Order lifecycle (UUID PK, symbol, side, status)
- **`trades`** — Executed fills (references orders)
- **`positions`** — Open positions

### SQLite (local files)
- **`data/market_cache.db`** — Market data cache (via `market_cache.py`). Has `cache_meta` table for gap detection.
- **`data/trade_journal.db`** — Trade audit trail (via `journal.rs`)
- **`data/tasks.db`** — Task tracking (via `task_store.rs`)

---

## Development Conventions

### Build & Test
```bash
cargo build --release                    # Build all crates
cargo test --release                     # 111 tests (50 strategy, 31 broker, 20 risk, 7 core, 3 integration)
cd web && npm install && npm run build   # Build frontend → web/dist/
```

### Development Mode
```powershell
.\dev.ps1    # Starts Rust backend (port 8080) + Vite frontend (port 3000) in separate windows
```

### Production
```powershell
.\start.ps1  # Builds frontend, runs Rust server serving SPA + API on port 8080
```

### Docker
```bash
docker compose up -d   # Rust app + PostgreSQL 16 with health checks
```
- Dockerfile: `rust:latest` build → `debian:trixie-slim` runtime
- CMD: `quant --config config/default.toml serve`

### Important Rules
1. **Python subprocess stdout** must be read in background threads to avoid Windows pipe buffer deadlock (4KB buffer limit)
2. **Python subprocess calls** (akshare) must use spawn + try_wait polling with 15s timeout — `.output()` blocks forever when akshare is unreachable
3. **LightGBM model files** (`.lgb.txt`) MUST have LF line endings — CRLF breaks tree_sizes byte offsets
4. **Walk-forward CV** uses 5-bar embargo gap to prevent label leakage
5. **Stamp tax** applies only on sells in China A-shares
6. **Signal-on-bar-N, fill-at-bar-N+1-open** — core backtest paradigm to avoid look-ahead bias

### Adding a New Strategy
1. Implement `Strategy` trait in `crates/strategy/src/builtin.rs`
2. Register in `crates/strategy/src/factory.rs` (`create_strategy` match arm)
3. Add tests in the strategy crate
4. Frontend auto-discovers via `GET /api/strategies`

### Adding a New API Endpoint
1. Create handler function in `crates/api/src/handlers/`
2. Mount route in `crates/api/src/routes.rs`
3. Add corresponding API function in `web/src/api/client.ts`
4. For async operations: use `task_store.create_with_params()`, return 202 + task_id

### Adding a New Factor
1. Define computation in `ml_models/factor_mining.py` (template + parameter grid)
2. Or use GP evolution in `ml_models/gp_factor_mining.py`
3. Evaluate via IC/IR metrics
4. Promote in `factor_registry.json`
5. Export Rust snippet for live inference
