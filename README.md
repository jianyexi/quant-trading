# Quant Trading System

A full-featured quantitative trading system built in **Rust**, targeting the **Chinese A-share market**. Supports backtesting, paper trading, **QMT 实盘下单** (live trading via 迅投量化), intelligent stock screening, **sentiment data integration** (舆情数据), **ML-based factor extraction** (机器学习因子策略 with GPU inference), actor-based auto-trading engine, and an integrated **LLM-powered AI assistant** for conversational market analysis.

## ✨ Features

| Category | Highlights |
|----------|------------|
| 📊 **Market Data** | Tushare + AKShare integration for real-time and historical Chinese A-share data |
| 🧪 **Backtesting** | Event-driven engine with Sharpe ratio, max drawdown, win rate, equity curve, trade log |
| 📈 **Indicators** | SMA, EMA, MACD, RSI, Bollinger Bands, KDJ — all composable |
| 🔍 **Stock Screener** | 3-phase pipeline: multi-factor scoring → strategy signal voting → LLM analysis |
| 📰 **Sentiment Data** | Ingest sentiment/news data via API, adjust trading signals based on market mood |
| 🧠 **ML Factor Model** | 24-feature engineering in Rust + GPU-accelerated inference via Python sidecar (LightGBM/XGBoost/CatBoost/ONNX/PyTorch) |
| 🔄 **Auto-Retrain** | Multi-algorithm competition (LightGBM/XGBoost/CatBoost/LSTM/Transformer), walk-forward CV, journal-based labels, hot model reload |
| 🎯 **Ensemble Learning** | Multi-model ensemble (LightGBM + XGBoost + CatBoost + ONNX + PyTorch), weighted average predictions |
| ⚖️ **Dynamic Weights** | Factor weights auto-adapt based on rolling directional accuracy |
| 🤖 **Auto-Trading** | Actor model engine (Data → Strategy → Risk → Order) with real-time status |
| 🔴 **QMT 实盘** | Live trading via QMT (迅投量化) Python bridge — real order placement to broker |
| 📝 **Paper Trading** | Simulated order execution with commission/stamp tax modeling |
| 🧠 **DL Model Research** | Curated knowledge base of 11 latest DL factor models + LLM auto-collection |
| 💬 **LLM Assistant** | OpenAI-compatible AI chat with tool calling for market analysis |
| 🖥️ **Web UI** | React + TypeScript dashboard: 17 pages for market, backtest, screener, sentiment, DL研究, auto-trade, metrics, reports, latency, chat |
| 🌐 **Web API** | REST + WebSocket API (Axum) with SPA fallback |
| 💻 **CLI** | Full subcommand CLI with interactive chat REPL |
| 🛡️ **Risk Management** | T+1, price limits (±10%/±20%), stamp tax, lot sizing, concentration limits |
| 🔒 **Risk Enforcement** | Stop-loss, daily loss limit, max drawdown protection, circuit breaker, all configurable |
| 📋 **Trade Journal** | SQLite-backed persistent audit trail for all signals, orders, fills, rejections |
| 📊 **Performance Metrics** | Real-time portfolio value, return %, drawdown, win rate, profit factor |
| ⏱ **Latency Profiling** | Per-module pipeline latency (data/strategy/risk/order) with bottleneck detection |
| 📈 **System Metrics** | Engine throughput, API request stats, DB pool health, real-time monitoring |
| 📋 **Statistical Reports** | Trading summary, per-symbol PnL, daily P&L, risk events, order analysis |

## 🏗️ Architecture

```
quant-trading/
├── crates/                         # 10-crate Rust workspace
│   ├── core/                       # Domain models, traits (Broker, Strategy, DataProvider), error types
│   ├── config/                     # TOML configuration loading (AppConfig, QmtConfig, etc.)
│   ├── data/                       # Market data fetching (Tushare, AKShare) + PostgreSQL storage
│   ├── strategy/                   # Strategy engine, technical indicators, stock screener
│   │   ├── indicators.rs           #   SMA, EMA, MACD, RSI, Bollinger, KDJ
│   │   ├── builtin.rs              #   DualMaCrossover, RsiMeanReversion, MacdMomentum
│   │   └── screener.rs             #   3-phase stock screening pipeline
│   │   └── dl_models.rs            #   DL factor model knowledge base + auto-collection
│   ├── backtest/                   # Backtesting engine with performance report
│   ├── broker/                     # Order management + execution
│   │   ├── paper.rs                #   PaperBroker (simulated, auto-fill)
│   │   ├── qmt.rs                  #   QmtBroker (live trading via HTTP bridge)
│   │   ├── engine.rs               #   Actor-based TradingEngine (generic over Broker)
│   │   ├── journal.rs              #   SQLite trade journal for persistent audit trail
│   │   └── orders.rs               #   Order state machine
│   ├── risk/                       # Pre-trade risk checks, position sizing, Chinese market rules
│   │   ├── checks.rs               #   Pre-trade order validation (concentration, cash, position)
│   │   ├── enforcement.rs          #   Runtime risk enforcement (stop-loss, daily loss, drawdown, circuit breaker)
│   │   ├── position.rs             #   Position sizing (fixed, percentage, Kelly criterion)
│   │   └── rules.rs                #   Chinese market rules (T+1, price limits, stamp tax, lot sizing)
│   ├── llm/                        # LLM chat client with tool calling + conversation context
│   ├── api/                        # Axum REST API + WebSocket + SPA fallback
│   └── cli/                        # CLI application (clap)
├── web/                            # React + TypeScript + Tailwind WebUI
│   └── src/pages/                  #   Dashboard, Market, Backtest, Strategy, Portfolio, Chat,
│                                   #   Screener, AutoTrade
├── qmt_bridge/                     # Python sidecar wrapping xtquant SDK for QMT live trading
│   ├── qmt_bridge.py               #   Flask HTTP API → xtquant (order, cancel, positions, account)
│   └── requirements.txt            #   flask, xtquant
├── ml_models/                      # ML factor model training & inference sidecar
│   ├── train_factor_model.py       #   LightGBM training + ONNX export
│   ├── auto_retrain.py             #   Multi-algorithm retrain (LGB/XGB/CatBoost/LSTM/Transformer)
│   ├── ml_serve.py                 #   Flask GPU inference server (LGB/XGB/CatBoost/ONNX/PyTorch + CUDA)
│   └── requirements.txt            #   torch, lightgbm, xgboost, catboost, onnxruntime, flask
├── config/default.toml             # System configuration (database, API keys, trading params, QMT)
├── migrations/                     # PostgreSQL schema migrations
├── Dockerfile                      # Container build
└── docker-compose.yml              # Docker Compose (app + PostgreSQL)
```

### Actor-Based Trading Engine

```
DataActor ──→ StrategyActor ──→ RiskActor ──→ OrderActor
(market bars)   (signals)       (validates)    (executes)
                                                  │
                                    ┌─────────────┼─────────────┐
                                    ▼                           ▼
                              PaperBroker                  QmtBroker
                            (simulated fill)         (live order via QMT)
```

### Stock Screener Pipeline

```
Phase 1: Multi-Factor Scoring (9 factors)
  momentum, RSI, MACD, Bollinger, KDJ, MA trend, volume ratio, volatility
  → composite score (0–100) with configurable weights
       ▼
Phase 2: Strategy Signal Voting
  DualMaCrossover + RsiMeanReversion + MacdMomentum → consensus vote
       ▼
Phase 3: LLM Analysis (optional)
  Generate structured prompt with all technical data → AI recommendation
```

### ML Factor Extraction Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust (crates/strategy)                     │
│                                                               │
│  Kline Bars ──→ Feature Engineering (24 features)             │
│                   returns, volatility, MA ratios, RSI,        │
│                   MACD, volume, price patterns, Bollinger     │
│                          │                                     │
│                          ▼                                     │
│              MlInferenceClient (HTTP)                          │
│                          │                                     │
└──────────────────────────│─────────────────────────────────────┘
                           │  POST /predict
                           ▼
┌─────────────────────────────────────────────────────────────┐
│             Python Sidecar (ml_models/ml_serve.py)           │
│                     127.0.0.1:18091                            │
│                                                               │
│   ONNX Runtime (GPU/CPU)  ←── train_factor_model.py           │
│   LightGBM                     (LightGBM → ONNX export)      │
│   PyTorch + CUDA                                              │
└─────────────────────────────────────────────────────────────┘

Fallback: When Python sidecar is unavailable, a rule-based scoring
function evaluates the 24 features to produce a signal.
```

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- Node.js 18+ (for WebUI)
- PostgreSQL 14+ (optional, for data persistence)
- Python 3.8+ (optional, for QMT live trading)
- Python 3.8+ with CUDA (optional, for ML factor model GPU inference)

### Build

```bash
# Build the entire system
cargo build --release

# Build the frontend
cd web && npm install && npm run build && cd ..
```

### Configure

Edit `config/default.toml`:

```toml
[tushare]
token = "YOUR_TUSHARE_TOKEN"

[llm]
api_key = "YOUR_OPENAI_API_KEY"

[trading]
initial_capital = 1000000.0
commission_rate = 0.00025      # 0.025%
stamp_tax_rate = 0.001         # 0.1% (sell only)

[risk]
max_concentration = 0.2        # Max 20% per stock
max_daily_loss = 0.05
max_drawdown = 0.15

[qmt]
bridge_url = "http://127.0.0.1:18090"
account = ""                   # QMT trading account ID
qmt_path = ""                  # miniQMT userdata path
```

### Run

```bash
# Start API server + WebUI → http://localhost:8080
quant serve

# Run backtest (CSI 300 stock with SMA strategy)
quant backtest run --strategy sma_cross --symbol 600519.SH --start 2024-01-01 --end 2024-12-31

# Multi-factor model backtest
quant backtest run --strategy multi_factor --symbol 600519.SH --start 2024-01-01 --end 2024-12-31

# Paper trading (quick simulation)
quant trade paper --strategy sma_cross --symbol 600519.SH

# Auto-trading with actor engine (paper mode)
quant trade auto --strategy sma_cross --symbols "600519.SH,000858.SZ" --interval 5

# Stock screening (top 10 candidates)
quant screen scan --top 10 --min-votes 2

# Factor analysis for a single stock
quant screen factors --symbol 600519.SH

# Interactive AI chat
quant chat

# Check QMT bridge status
quant trade qmt-status

# Live trading via QMT (requires bridge running)
quant trade qmt --strategy sma_cross --symbols "000001.SZ"

# ML factor model backtest (uses rule-based fallback if sidecar not running)
quant backtest run --strategy ml_factor --symbol 600519.SH --start 2024-01-01 --end 2024-12-31
```

## 🔴 QMT Live Trading (实盘交易)

QMT (迅投量化) integration enables real order placement through your broker.

### Setup

1. **Open QMT permission** with your broker (requires miniQMT mode)
2. **Install Python dependencies**:
   ```bash
   cd qmt_bridge && pip install -r requirements.txt
   ```
   > `xtquant` may need to be copied from your QMT client directory
3. **Start QMT client** in miniQMT mode
4. **Start the bridge**:
   ```bash
   python qmt_bridge/qmt_bridge.py --qmt-path "C:/QMT/userdata_mini" --account "YOUR_ACCOUNT"
   ```
5. **Configure** `config/default.toml` with your `[qmt]` settings
6. **Start live trading**:
   ```bash
   # CLI
   quant trade qmt --strategy sma_cross --symbols "000001.SZ,600036.SH"
   
   # Or via WebUI → 自动交易 → QMT 实盘 mode
   ```

### Bridge API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Bridge status + connection state |
| POST | `/order` | Place stock order (buy/sell, limit/market) |
| POST | `/cancel` | Cancel an existing order |
| GET | `/positions` | Query current positions |
| GET | `/account` | Query account assets (cash, market value) |
| GET | `/orders` | List today's orders |

## 🌐 Web API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/dashboard` | Dashboard statistics |
| GET | `/api/strategies` | List available strategies |
| GET | `/api/market/kline/:symbol` | K-line data |
| GET | `/api/market/quote/:symbol` | Real-time quote |
| GET | `/api/market/stocks` | Stock list |
| POST | `/api/backtest/run` | Run backtest |
| GET | `/api/backtest/results/:id` | Backtest results |
| GET | `/api/orders` | List orders |
| GET | `/api/portfolio` | Portfolio status |
| POST | `/api/chat` | Send chat message |
| GET | `/api/chat/history` | Chat history |
| WS | `/api/chat/stream` | Streaming chat via WebSocket |
| POST | `/api/trade/start` | Start auto-trade engine (`mode`: paper / qmt) |
| POST | `/api/trade/stop` | Stop auto-trade engine |
| GET | `/api/trade/status` | Engine status (signals, fills, PnL, performance) |
| GET | `/api/trade/performance` | Real-time performance metrics (return, drawdown, win rate) |
| GET | `/api/trade/risk` | Risk enforcement status (daily PnL, circuit breaker, drawdown) |
| POST | `/api/trade/risk/reset-circuit` | Reset circuit breaker |
| POST | `/api/trade/risk/reset-daily` | Reset daily loss counter |
| POST | `/api/trade/retrain` | Trigger ML model retrain (walk-forward CV + export) |
| GET | `/api/trade/model-info` | Latest retrain report + feature importance |
| GET | `/api/trade/qmt/status` | QMT bridge connection status |
| POST | `/api/screen/scan` | Run stock screener (multi-factor + voting) |
| GET | `/api/screen/factors/:symbol` | Factor scores for a single stock |
| POST | `/api/sentiment/submit` | Submit a single sentiment item |
| POST | `/api/sentiment/batch` | Batch submit sentiment items |
| GET | `/api/sentiment/:symbol` | Query sentiment data for a stock |
| GET | `/api/sentiment/summary` | Global sentiment overview across all stocks |
| GET | `/api/research/dl-models` | Full DL factor model knowledge base |
| GET | `/api/research/dl-models/summary` | Knowledge base summary statistics |
| POST | `/api/research/dl-models/collect` | Auto-collect latest research via LLM |
| GET | `/api/journal` | Trade journal entries (filter by symbol, type, date) |
| GET | `/api/journal/snapshots` | Daily performance snapshots |
| GET | `/api/metrics` | System metrics (throughput, latency, API stats, DB pool) |
| GET | `/api/reports` | Statistical reports (summary, per-symbol, daily PnL, orders) |
| GET | `/api/latency` | Per-module latency profiling with bottleneck detection |
| GET | `/api/notifications` | Notification center |
| GET | `/api/logs` | System logs |

## 🖥️ Web UI Pages

| Page | Path | Description |
|------|------|-------------|
| 仪表盘 | `/` | Portfolio overview, equity curve, positions, journal, auto-refresh |
| 行情 | `/market` | Real-time quotes, K-line charts |
| 回测 | `/backtest` | Run backtests, view performance reports |
| 策略 | `/strategy` | Strategy configuration and management |
| 持仓 | `/portfolio` | Current positions, P&L tracking, closed positions |
| AI 对话 | `/chat` | LLM-powered market analysis chat |
| 智能选股 | `/screener` | Multi-factor scan, strategy votes, LLM analysis |
| 自动交易 | `/autotrade` | Start/stop engine, mode selector (Paper/QMT), real-time stats |
| 风控管理 | `/risk` | Risk configuration, circuit breaker, drawdown limits |
| 舆情数据 | `/sentiment` | Sentiment data submission, overview, per-stock analysis |
| DL模型研究 | `/dl-models` | DL factor model knowledge base, auto-collection |
| 因子挖掘 | `/factor-mining` | Parametric & GP factor mining, factor registry |
| 通知中心 | `/notifications` | System notifications with unread count |
| 系统日志 | `/logs` | Real-time log viewer with level filtering |
| 性能监控 | `/metrics` | Engine throughput, API latency, DB pool, sparklines |
| 统计报表 | `/reports` | Trading summary, per-symbol PnL, daily charts, order analysis |
| 延迟分析 | `/latency` | Per-module latency breakdown, bottleneck detection, health score |

## 📈 Built-in Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| **DualMaCrossover** | Golden/death cross on two moving averages | fast=5, slow=20 |
| **RsiMeanReversion** | Buy oversold, sell overbought | period=14, overbought=70, oversold=30 |
| **MacdMomentum** | MACD histogram zero-crossing | fast=12, slow=26, signal=9 |
| **MultiFactorModel** | 6-factor composite scoring with threshold-crossing signals | buy_threshold=0.30, sell_threshold=-0.30 |
| **SentimentAware** | Multi-factor + sentiment data fusion, adjusts signals based on market mood | sentiment_weight=0.20, min_items=3 |

### Multi-Factor Model Details

The `MultiFactorModel` strategy computes 6 sub-scores on each bar, weighted into a composite signal in [-1, +1]:

| Factor | Weight | Indicators | Signal |
|--------|--------|------------|--------|
| **Trend** | 25% | SMA(5/20) cross + EMA(10) slope | Fast > Slow → bullish |
| **Momentum** | 25% | RSI(14) + MACD(12/26/9) histogram | RSI > 50 + rising MACD → bullish |
| **Volatility** | 15% | Bollinger Bands(20, 2σ) %B | Near lower band → buy, upper → sell |
| **Oscillator** | 15% | KDJ(9,3,3) K/D cross + J extreme | Golden cross + J<20 → buy |
| **Volume** | 10% | Volume MA(5) / MA(20) × price direction | High vol + price up → bullish |
| **Price Action** | 10% | Close position in 20-day H/L range | Breakout → bullish, breakdown → bearish |

- **BUY** when composite crosses above `+0.30`
- **SELL** when composite crosses below `-0.30`
- Threshold-crossing prevents repeated signals in the same direction

### Sentiment-Aware Strategy (舆情增强策略)

The `SentimentAware` strategy wraps any base strategy (default: MultiFactorModel) and adjusts trading signals using external sentiment data:

| Scenario | Action |
|----------|--------|
| **BUY signal + bullish sentiment** | Boost confidence (stronger buy) |
| **BUY signal + bearish sentiment (strong)** | Suppress weak buy signals (confidence < 0.3 → skip) |
| **SELL signal + bearish sentiment** | Boost confidence (stronger sell) |
| **SELL signal + bullish sentiment (strong)** | Suppress weak sell signals |
| **No sentiment data** | Pass through base strategy signal unchanged |

**Sentiment data ingestion** via REST API:
```bash
# Submit single item
curl -X POST http://localhost:8080/api/sentiment/submit \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"600519.SH","source":"新闻","title":"茅台业绩超预期","sentiment_score":0.7}'

# Batch submit
curl -X POST http://localhost:8080/api/sentiment/batch \
  -H 'Content-Type: application/json' \
  -d '[{"symbol":"600519.SH","source":"研报","title":"..","sentiment_score":0.5}]'

# Query sentiment
curl http://localhost:8080/api/sentiment/600519.SH?limit=10

# Get overview
curl http://localhost:8080/api/sentiment/summary
```

### ML Factor Strategy (ML因子模型)

The `MlFactorStrategy` uses a **24-dimensional feature vector** computed in Rust from raw Kline bars, then sends it to a Python inference sidecar for GPU-accelerated prediction.

**Supported Training Algorithms:**

| Algorithm | Type | File Format | GPU Support |
|-----------|------|-------------|-------------|
| LightGBM | Gradient Boosting | `.lgb.txt` | ❌ CPU |
| XGBoost | Gradient Boosting | `.xgb.json` | ✅ CUDA |
| CatBoost | Gradient Boosting | `.catboost.bin` | ✅ CUDA |
| LSTM | Deep Learning (PyTorch) | `.lstm.pt` | ✅ CUDA |
| Transformer | Deep Learning (PyTorch) | `.transformer.pt` | ✅ CUDA |

**Features (24 total):**
- **Returns**: 1d, 5d, 10d, 20d
- **Volatility**: 5d, 20d
- **MA ratios**: close/MA5, close/MA10, close/MA20, close/MA60, MA5/MA20
- **Momentum**: RSI(14), MACD histogram, MACD signal ratio
- **Volume**: 5d volume ratio, 20d volume ratio
- **Price patterns**: price position, gap, intraday range, upper/lower shadow ratios, Bollinger %B, body ratio, close-to-open ratio

**Running the ML inference sidecar:**
```bash
cd ml_models
pip install -r requirements.txt
python ml_serve.py --port 18091  # auto-detects GPU (CUDA)
```

**Training with algorithm competition:**
```bash
cd ml_models
# Single algorithm (default: LightGBM)
python auto_retrain.py --data my_data.csv

# Multi-algorithm competition — best AUC wins
python auto_retrain.py --data my_data.csv --algorithms lgb,xgb,catboost,lstm,transformer

# Deep learning only
python auto_retrain.py --data my_data.csv --algorithms lstm,transformer
```

**Fallback mode**: When the Python sidecar is unavailable, the strategy uses a built-in rule-based scoring function that evaluates the same 24 features to produce trading signals. No ML infrastructure required for basic operation.

## 🔍 Stock Screener Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Momentum (20d) | 25% | 20-day price return |
| RSI (14) | 20% | Relative Strength Index |
| MACD Signal | 20% | MACD histogram direction |
| Bollinger %B | 10% | Position within Bollinger Bands |
| KDJ | 10% | Stochastic oscillator |
| MA Trend | 30% | Price relative to 20-day SMA |
| Volume Ratio | 15% | Recent vs average volume |
| Volatility | 10% | 20-day standard deviation |

## 🛡️ Chinese Market Rules

| Rule | Implementation |
|------|----------------|
| T+1 Settlement | Cannot sell shares bought today |
| Price Limits (Main Board) | ±10% from previous close |
| Price Limits (ChiNext/STAR) | ±20% from previous close |
| Stamp Tax | 0.1% on sell side only |
| Minimum Lot Size | 100 shares |
| Commission | Configurable (default 0.025%, minimum ¥5) |

## 🧪 Tests

```bash
# Run all tests
cargo test --release

# Test breakdown:
# - 50 strategy tests (indicators, screener, multi-factor, sentiment, ml_factor, dl_models, dynamic_weights, factor_mining)
# - 31 broker tests (paper, qmt, engine, orders, journal, data_actors)
# - 20 risk tests (checks, rules, position sizing, enforcement)
# - 7 core tests (models, types, config)
# - 3 integration tests
# Total: 111 tests
```

## 💬 LLM Tool Calling

The AI assistant can invoke system functions during conversation:
- **get_kline** — Fetch K-line data for any symbol and date range
- **get_stock_info** — Get stock fundamentals and metadata
- **run_backtest** — Run a backtest with specified parameters
- **get_portfolio** — View current portfolio and positions
- **screen_stocks** — Screen stocks by multi-factor criteria

## 🐳 Docker

```bash
# One-click deployment (app + PostgreSQL)
docker compose up -d

# View logs
docker compose logs -f quant

# Stop
docker compose down

# With data persistence (volumes preserved)
docker compose down    # keeps pgdata volume
docker compose down -v # removes pgdata volume
```

The Docker setup includes:
- **PostgreSQL 16** with health checks and data persistence
- **Quant server** with auto-restart, health checks, and all migrations
- **Web UI** pre-built and served from the container
- Volumes for config, data, ML models, and scripts

## License

MIT
