# Quant Trading System

A full-featured quantitative trading system built in **Rust**, targeting the **Chinese A-share market**. Features backtesting, paper trading, live trading interfaces, and an integrated **LLM-powered AI assistant** for market analysis.

## Features

- 📊 **Market Data** — Tushare + AKShare integration for real-time and historical Chinese A-share data
- 🧪 **Backtesting Engine** — Event-driven backtesting with realistic commission/slippage modeling
- 📈 **Technical Indicators** — SMA, EMA, MACD, RSI, Bollinger Bands, KDJ
- 🤖 **LLM Chat Assistant** — OpenAI-compatible AI chat with tool calling for market analysis
- 🖥️ **Web UI** — React + TypeScript dashboard with strategy configuration, backtest visualization, and AI chat
- 🌐 **Web API** — REST + WebSocket API (Axum)
- 💻 **CLI** — Interactive command-line interface with chat REPL
- 🛡️ **Risk Management** — Chinese market rules (T+1, price limits, stamp tax)
- 📝 **Paper Trading** — Simulated order execution for strategy testing

## Quick Start

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- PostgreSQL 14+
- Node.js 18+ (for WebUI)

### Build

```bash
# Build backend
cargo build --release

# Build frontend
cd web && npm install && npm run build
```

### Configure

Copy and edit the configuration file:
```bash
cp config/default.toml config/local.toml
# Edit config/local.toml with your API keys
```

Key settings:
- `database.url` — PostgreSQL connection string
- `tushare.token` — Tushare API token
- `llm.api_key` — OpenAI API key (or Ollama endpoint)
- `llm.api_url` — LLM API URL (default: OpenAI)

### Run

```bash
# Start API server + WebUI (visit http://localhost:8080)
quant serve

# Or use Vite dev server for frontend development
cd web && npm run dev  # http://localhost:3000 (proxies API to :8080)

# Start interactive AI chat (CLI)
quant chat

# Sync market data
quant data sync

# Query market data
quant data query --symbol 600519.SH --start 2024-01-01 --end 2024-12-31

# Run backtest
quant backtest run --strategy DualMaCrossover --symbol 600519.SH --start 2024-01-01 --end 2024-12-31

# Paper trading
quant trade paper --strategy DualMaCrossover

# View portfolio
quant portfolio show
```

## Architecture

```
quant-trading/
├── crates/
│   ├── core/       # Domain models, traits, error types
│   ├── config/     # TOML configuration loading
│   ├── data/       # Market data (Tushare, AKShare, PostgreSQL)
│   ├── strategy/   # Strategy engine, indicators, built-in strategies
│   ├── backtest/   # Backtesting engine with metrics
│   ├── broker/     # Paper & live trading brokers
│   ├── risk/       # Risk management, Chinese market rules
│   ├── llm/        # LLM chat client with tool calling
│   ├── api/        # Axum REST API + WebSocket
│   └── cli/        # CLI application
├── migrations/     # PostgreSQL schema
└── config/         # Configuration files
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/market/kline/:symbol` | K-line data |
| GET | `/api/market/quote/:symbol` | Real-time quote |
| POST | `/api/backtest/run` | Run backtest |
| GET | `/api/backtest/results/:id` | Backtest results |
| GET | `/api/orders` | List orders |
| GET | `/api/portfolio` | Portfolio status |
| POST | `/api/chat` | Send chat message |
| GET | `/api/chat/history` | Chat history |
| WS | `/api/chat/stream` | Streaming chat |

## LLM Tool Calling

The AI assistant can call these tools during conversation:
- **get_kline** — Fetch K-line data for any symbol
- **get_stock_info** — Get stock fundamentals
- **run_backtest** — Run a backtest with parameters
- **get_portfolio** — View current portfolio
- **screen_stocks** — Screen stocks by criteria

## Built-in Strategies

1. **DualMaCrossover** — Golden/death cross with configurable MA periods
2. **RsiMeanReversion** — Buy oversold, sell overbought
3. **MacdMomentum** — MACD histogram zero-crossing

## Chinese Market Rules

- T+1 settlement enforcement
- Price limits: ±10% (main board), ±20% (ChiNext/STAR)
- Stamp tax: 0.1% (sell side only)
- Minimum lot size: 100 shares
- Commission: configurable (default 0.025%)

## License

MIT
