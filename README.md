# Quant Trading System

A full-featured quantitative trading system built in **Rust**, targeting the **Chinese A-share market**. Supports backtesting, paper trading, **QMT 实盘下单** (live trading via 迅投量化), intelligent stock screening, actor-based auto-trading engine, and an integrated **LLM-powered AI assistant** for conversational market analysis.

## ✨ Features

| Category | Highlights |
|----------|------------|
| 📊 **Market Data** | Tushare + AKShare integration for real-time and historical Chinese A-share data |
| 🧪 **Backtesting** | Event-driven engine with Sharpe ratio, max drawdown, win rate, equity curve, trade log |
| 📈 **Indicators** | SMA, EMA, MACD, RSI, Bollinger Bands, KDJ — all composable |
| 🔍 **Stock Screener** | 3-phase pipeline: multi-factor scoring → strategy signal voting → LLM analysis |
| 🤖 **Auto-Trading** | Actor model engine (Data → Strategy → Risk → Order) with real-time status |
| 🔴 **QMT 实盘** | Live trading via QMT (迅投量化) Python bridge — real order placement to broker |
| 📝 **Paper Trading** | Simulated order execution with commission/stamp tax modeling |
| 💬 **LLM Assistant** | OpenAI-compatible AI chat with tool calling for market analysis |
| 🖥️ **Web UI** | React + TypeScript dashboard: 8 pages for market, backtest, screener, auto-trade, chat |
| 🌐 **Web API** | REST + WebSocket API (Axum) with SPA fallback |
| 💻 **CLI** | Full subcommand CLI with interactive chat REPL |
| 🛡️ **Risk Management** | T+1, price limits (±10%/±20%), stamp tax, lot sizing, concentration limits |

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
│   ├── backtest/                   # Backtesting engine with performance report
│   ├── broker/                     # Order management + execution
│   │   ├── paper.rs                #   PaperBroker (simulated, auto-fill)
│   │   ├── qmt.rs                  #   QmtBroker (live trading via HTTP bridge)
│   │   ├── engine.rs               #   Actor-based TradingEngine (generic over Broker)
│   │   └── orders.rs               #   Order state machine
│   ├── risk/                       # Pre-trade risk checks, position sizing, Chinese market rules
│   ├── llm/                        # LLM chat client with tool calling + conversation context
│   ├── api/                        # Axum REST API + WebSocket + SPA fallback
│   └── cli/                        # CLI application (clap)
├── web/                            # React + TypeScript + Tailwind WebUI
│   └── src/pages/                  #   Dashboard, Market, Backtest, Strategy, Portfolio, Chat,
│                                   #   Screener, AutoTrade
├── qmt_bridge/                     # Python sidecar wrapping xtquant SDK for QMT live trading
│   ├── qmt_bridge.py               #   Flask HTTP API → xtquant (order, cancel, positions, account)
│   └── requirements.txt            #   flask, xtquant
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

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- Node.js 18+ (for WebUI)
- PostgreSQL 14+ (optional, for data persistence)
- Python 3.8+ (optional, for QMT live trading)

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
| GET | `/api/trade/status` | Engine status (signals, fills, PnL) |
| GET | `/api/trade/qmt/status` | QMT bridge connection status |
| POST | `/api/screen/scan` | Run stock screener (multi-factor + voting) |
| GET | `/api/screen/factors/:symbol` | Factor scores for a single stock |

## 🖥️ Web UI Pages

| Page | Path | Description |
|------|------|-------------|
| 仪表盘 | `/` | Portfolio overview, market summary, equity chart |
| 行情 | `/market` | Real-time quotes, K-line charts |
| 回测 | `/backtest` | Run backtests, view performance reports |
| 策略 | `/strategy` | Strategy configuration and management |
| 持仓 | `/portfolio` | Current positions, P&L tracking |
| AI 对话 | `/chat` | LLM-powered market analysis chat |
| 智能选股 | `/screener` | Multi-factor scan, strategy votes, LLM analysis |
| 自动交易 | `/autotrade` | Start/stop engine, mode selector (Paper/QMT), real-time stats |

## 📈 Built-in Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| **DualMaCrossover** | Golden/death cross on two moving averages | fast=5, slow=20 |
| **RsiMeanReversion** | Buy oversold, sell overbought | period=14, overbought=70, oversold=30 |
| **MacdMomentum** | MACD histogram zero-crossing | fast=12, slow=26, signal=9 |

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
# Run all 37 tests
cargo test --release

# Test breakdown:
# - 12 broker tests (paper, qmt, engine, orders)
# - 15 risk tests (checks, rules, position sizing)
# - 10 strategy tests (indicators, screener)
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
docker-compose up -d    # Starts app + PostgreSQL
```

## License

MIT
