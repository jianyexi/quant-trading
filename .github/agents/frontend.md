---
name: frontend
description: Expert on the React/TypeScript frontend — pages, components, hooks, API client, styling conventions, and state management patterns.
tools:
  - thinking
  - editor
  - terminal
  - file_search
---

# Frontend Agent Guide — Quant Trading System

> **Purpose**: This document helps AI agents and developers understand the React/TypeScript frontend architecture, conventions, and patterns for effective development.

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19.2.0 | UI framework |
| TypeScript | 5.x | Type safety (strict mode) |
| Vite | 7.x | Build tool + dev server |
| TailwindCSS | 4.1.18 | Utility-first CSS (JIT via Vite plugin) |
| React Router DOM | 7.13.0 | Client-side routing |
| Recharts | 3.7.0 | Charts (Area, Bar, Pie, Line) |
| Lucide React | 0.564.0 | Icon library (50+ icons) |
| Playwright | 1.58.x | E2E testing |

## Project Structure

```
web/
├── index.html                 # SPA entry point (<div id="root">)
├── package.json               # Dependencies & scripts
├── vite.config.ts             # Vite + React + Tailwind plugins, proxy config
├── tsconfig.json              # Strict TypeScript, ES2022 target
├── playwright.config.ts       # E2E test config
├── public/                    # Static assets
└── src/
    ├── main.tsx               # React root with StrictMode
    ├── App.tsx                # Router + MarketProvider wrapper
    ├── index.css              # Tailwind import + CSS variables (dark theme)
    ├── api/
    │   └── client.ts          # 900+ lines: Complete API client (100+ functions)
    ├── components/
    │   ├── Layout.tsx         # Sidebar + top bar + persistent page mounting
    │   ├── Sidebar.tsx        # Navigation with collapsible groups + market selector
    │   └── TaskPipeline.tsx   # Shared UI: TaskOutput + ParamGrid components
    ├── contexts/
    │   └── MarketContext.tsx   # Global market region (CN/US/HK/ALL) with localStorage
    ├── hooks/
    │   ├── useTaskPoller.ts   # Low-level task polling (2s interval)
    │   └── useTaskManager.ts  # High-level task lifecycle + sessionStorage persistence
    ├── pages/
    │   ├── Dashboard.tsx      # Portfolio overview, equity curve, metrics (355 lines)
    │   ├── MarketData.tsx     # Kline charts, candlesticks, quotes (480 lines)
    │   ├── StrategyConfig.tsx # Strategy parameter config + persistence
    │   ├── Backtest.tsx       # Backtesting with multi-symbol support
    │   ├── Pipeline.tsx       # Multi-step workflow: sync → mine → train → backtest
    │   ├── History.tsx        # Task history with detailed logs
    │   ├── Portfolio.tsx      # Open/closed positions, pie chart (350 lines)
    │   ├── Chat.tsx           # AI assistant with streaming (197 lines)
    │   ├── Screener.tsx       # Multi-factor stock screening
    │   ├── AutoTrade.tsx      # Live/paper/QMT/replay trading
    │   ├── RiskManagement.tsx # Risk signals, circuit breaker, WebSocket updates
    │   ├── Sentiment.tsx      # Sentiment submission & analysis
    │   ├── DLModels.tsx       # DL model research knowledge base
    │   ├── Notifications.tsx  # Event notifications + channel config
    │   ├── Logs.tsx           # Tasks, API logs, trade journal tabs
    │   ├── Metrics.tsx        # Performance monitoring
    │   ├── Reports.tsx        # Statistical reports
    │   ├── Latency.tsx        # Pipeline latency analysis
    │   ├── Services.tsx       # ML service management (start/stop ml-serve)
    │   └── factor-mining/     # Tabbed factor mining interface
    │       ├── index.tsx      # Main tabbed container (6 tabs)
    │       ├── OverviewTab.tsx
    │       ├── ParametricTab.tsx
    │       ├── GPTab.tsx
    │       ├── ManualTab.tsx
    │       ├── RegistryTab.tsx
    │       ├── ExportTab.tsx
    │       └── DataSourceConfig.tsx
    └── types/
        └── index.ts           # Core TypeScript interfaces (75 lines)
```

---

## Routing & Navigation

### Provider Hierarchy
```tsx
<MarketProvider>                    // Global market context (CN/US/HK/ALL)
  <BrowserRouter>
    <Routes>
      <Route element={<Layout />}>  // Persistent sidebar + top bar
        <Route index element={<Dashboard />} />
        // ... 18 nested routes
      </Route>
    </Routes>
  </BrowserRouter>
</MarketProvider>
```

### Route Table (20 pages)

| Route | Component | Nav Group | Description |
|-------|-----------|-----------|-------------|
| `/` | Dashboard | — | Portfolio overview, equity curve, stats |
| `/market` | MarketData | 交易执行 | K-line charts, stock search, quotes |
| `/strategy` | StrategyConfig | 交易执行 | Strategy parameter config (persistent mount) |
| `/pipeline` | Pipeline | 量化研究 | Multi-step: sync → mine → train → backtest |
| `/history` | History | 量化研究 | Task history with logs |
| `/screener` | Screener | 交易执行 | Multi-factor stock screening |
| `/autotrade` | AutoTrade | 交易执行 | Live/paper/QMT/replay (persistent mount) |
| `/risk` | RiskManagement | 交易执行 | Risk signals, circuit breaker |
| `/portfolio` | Portfolio | 交易执行 | Open/closed positions |
| `/sentiment` | Sentiment | 数据 & 监控 | Sentiment submission & analysis |
| `/dl-models` | DLModels | 量化研究 | DL model knowledge base |
| `/factor-mining` | FactorMining | — | Tabbed factor mining (persistent mount, not in Router) |
| `/notifications` | Notifications | 数据 & 监控 | Event notifications |
| `/logs` | Logs | 数据 & 监控 | Tasks + API logs + journal |
| `/metrics` | Metrics | 数据 & 监控 | System performance monitoring |
| `/reports` | Reports | 数据 & 监控 | Trading statistics |
| `/latency` | Latency | 数据 & 监控 | Pipeline latency analysis |
| `/backtest` | Backtest | — | Strategy backtesting |
| `/services` | Services | 数据 & 监控 | ML service management |
| `/chat` | Chat | — | AI trading assistant |

### Persistent Page Mounting
Pages at `/strategy`, `/factor-mining`, and `/autotrade` stay mounted (hidden via CSS) to preserve task state across navigation. All other pages use normal `<Outlet />` rendering.

### Sidebar Navigation Structure
```
📊 仪表盘 (Dashboard) — standalone
📈 量化研究 (Quant Research) — collapsible group
   ├─ 量化流水线 → /pipeline
   ├─ DL模型研究 → /dl-models
   └─ 任务历史 → /history
💹 交易执行 (Trade Execution) — collapsible group
   ├─ 行情数据 → /market
   ├─ 智能选股 → /screener
   ├─ 策略管理 → /strategy
   ├─ 自动交易 → /autotrade
   ├─ 持仓管理 → /portfolio
   └─ 风控管理 → /risk
📋 数据 & 监控 (Data & Monitoring) — collapsible group
   ├─ 舆情 → /sentiment
   ├─ 通知 → /notifications
   ├─ 日志 → /logs
   ├─ 性能 → /metrics
   ├─ 报表 → /reports
   ├─ 延迟 → /latency
   └─ 服务管理 → /services
💬 AI Chat — standalone
```

---

## Contexts

### MarketContext (`contexts/MarketContext.tsx`)

**Global market region selector** with localStorage persistence.

```tsx
type MarketRegion = 'ALL' | 'CN' | 'US' | 'HK';

interface MarketContextType {
  market: MarketRegion;
  setMarket(m: MarketRegion): void;
  extractMarket(symbol: string): 'CN' | 'US' | 'HK';
  filterByMarket<T>(items: T[], symbolKey: keyof T): T[];
}

// Usage:
const { market, filterByMarket } = useMarket();
const filtered = filterByMarket(positions, 'symbol');
```

**Symbol → Market detection**:
- `.SH` / `.SZ` / 6-digit numeric → CN
- `.HK` → HK
- Everything else → US

**Market selector buttons** in Sidebar: 🌐 ALL | 🇨🇳 CN | 🇺🇸 US | 🇭🇰 HK

---

## Hooks

### useTaskPoller (`hooks/useTaskPoller.ts`)

**Low-level polling** for background task status (used by useTaskManager).

```tsx
const { task, startPolling, reset } = useTaskPoller({ intervalMs: 2000 });
startPolling(taskId);
// task: TaskRecord | null
// Stops automatically when status === 'Completed' | 'Failed'
```

### useTaskManager (`hooks/useTaskManager.ts`)

**High-level task lifecycle management** with sessionStorage persistence for page navigation.

```tsx
const tm = useTaskManager('task_factor_mine');  // storageKey

// Submit a new task:
<button onClick={() => tm.submit(() => factorMineGP(opts))}>Run</button>

// Display status:
<TaskOutput running={tm.running} error={tm.error} output={tm.output} progress={tm.progress} />

// Cancel:
<button onClick={() => tm.cancel()}>Cancel</button>
```

**Returns**: `{ task, running, output, error, progress, submit, cancel, setOutput, setError }`

**Behavior**:
- Restores active task from `sessionStorage` on mount
- Parses JSON results (looks for `stdout`/`stderr` fields)
- Clears sessionStorage on completion/failure
- `submit()` accepts async fn returning `{ task_id: string }`

---

## Shared Components

### Layout (`components/Layout.tsx`)

- Persistent top bar with notification bell (polls `getUnreadCount()` every 15s)
- Badge shows unread count (capped at "99+")
- Sidebar + main content area with `pl-60` left padding
- Persistent mounting for strategy/factor-mining/autotrade pages

### Sidebar (`components/Sidebar.tsx`)

- Fixed left sidebar (`w-60`, `z-40`)
- Brand header: "QuantTrader" + TrendingUp icon
- Market selector: 4 buttons with flag emojis
- Collapsible groups with ChevronDown/Right toggle
- NavLink active state: blue bg + text

### TaskPipeline (`components/TaskPipeline.tsx`)

Exports two reusable components:

**TaskOutput** — Displays task execution status:
```tsx
<TaskOutput
  running={boolean}        // Shows "⏳ 任务运行中..."
  error={string}           // Red error box
  output={string}          // Pre-formatted monospace output
  progress={string | null} // Optional progress text
  runningText={string}     // Custom running message
/>
```

**ParamGrid** — Responsive parameter input grid:
```tsx
<ParamGrid
  fields={[{ key: 'horizon', label: 'Horizon', value: 5, step: 1, min: 1, max: 60 }]}
  onChange={(key, value) => setParams(prev => ({...prev, [key]: value}))}
  columns={3}
/>
```

---

## API Client (`api/client.ts`)

Single file, 900+ lines. All API communication goes through `fetchJson<T>()` with:
- 20-second timeout via AbortController
- Base path: `/api` (proxied to `http://localhost:8080` in dev)
- Error handling: throws on non-OK responses

### Key API Functions by Domain

#### Dashboard & Portfolio
```tsx
getDashboard(): Promise<DashboardData>
getPortfolio(): Promise<PortfolioData>
closePosition(symbol: string, price?: number): Promise<any>
getClosedPositions(): Promise<ClosedPositionRow[]>
getOrders(): Promise<any>
```

#### Market Data
```tsx
getStocks(): Promise<StockInfo[]>
getQuote(symbol: string): Promise<StockQuote>
getKline(symbol: string, start?: string, end?: string, limit?: number): Promise<KlineData[]>
getCacheStatus(): Promise<CacheStatus>
getDataSourceStatus(): Promise<DataSourceStatus>
syncData(symbols: string[], start_date: string, end_date: string): Promise<{ task_id: string }>
```

#### Backtesting
```tsx
runBacktest(params: BacktestParams): Promise<{ task_id: string } | BacktestResultData>
getBacktestResults(taskId: string): Promise<BacktestResultData>
```

#### Strategy
```tsx
getStrategies(): Promise<StrategyInfo[]>
saveStrategyConfig(config: any): Promise<any>
loadStrategyConfig(): Promise<any>
```

#### Auto-Trading
```tsx
tradeStart(params: TradeStartParams): Promise<any>
tradeStop(): Promise<any>
tradeStatus(): Promise<TradeStatus>
getPerformance(): Promise<PerformanceData>
getRiskStatus(): Promise<RiskStatusData>
getRiskSignals(): Promise<RiskSignalsSnapshot>
resetCircuitBreaker(): Promise<any>
resetDailyLoss(): Promise<any>
qmtBridgeStatus(): Promise<QmtBridgeStatus>
```

#### Screening
```tsx
screenScan(topN?, minVotes?, pool?, buyThreshold?, strongBuyThreshold?, minTurnover?, maxPerSector?): Promise<ScreenerResult>
screenFactors(symbol: string): Promise<StockCandidate>
```

#### Factor Mining
```tsx
factorMineParametric(params?): Promise<{ task_id: string }>
factorMineGP(params?): Promise<{ task_id: string }>
factorEvaluateManual(params): Promise<{ task_id: string }>
factorSaveManual(params): Promise<any>
factorRegistryGet(): Promise<FactorRegistry>
factorRegistryManage(params?): Promise<{ task_id: string }>
factorExportPromoted(params?): Promise<{ task_id: string }>
factorResults(): Promise<FactorResults>
```

#### ML Model
```tsx
mlRetrain(opts?): Promise<{ task_id: string }>
mlModelInfo(): Promise<MlModelInfo>
```

#### Sentiment
```tsx
sentimentSubmit(params): Promise<any>
sentimentBatchSubmit(items): Promise<any>
sentimentQuery(symbol, limit?): Promise<SentimentItem[]>
sentimentSummary(): Promise<SentimentSummary[]>
```

#### Background Tasks
```tsx
getTask(taskId: string): Promise<TaskRecord>
listTasks(opts?): Promise<TaskRecord[]>
listRunningTasks(): Promise<TaskRecord[]>
cancelTask(taskId: string): Promise<any>
```

#### Chat
```tsx
sendChat(message: string): Promise<ChatResponse>
getChatHistory(): Promise<ChatMessage[]>
createChatWebSocket(): WebSocket        // ws://host/api/chat/stream
createMonitorWebSocket(): WebSocket     // ws://host/api/monitor/ws
```

#### Notifications
```tsx
getNotifications(limit?, unread_only?): Promise<Notification[]>
getUnreadCount(): Promise<number>
markNotificationRead(id): Promise<any>
markAllNotificationsRead(): Promise<any>
getNotificationConfig(): Promise<NotificationConfig>
saveNotificationConfig(cfg): Promise<any>
testNotification(): Promise<any>
```

#### Services
```tsx
getServiceStatus(): Promise<ServiceStatus>
startMlServe(params?): Promise<any>
stopMlServe(): Promise<any>
getMlServeStatus(): Promise<MlServeStatus>
```

#### Logs & Monitoring
```tsx
getMetrics(): Promise<MetricsData>
getReports(): Promise<ReportsData>
getLatency(): Promise<LatencyData>
getLogs(params?): Promise<LogEntry[]>
getJournal(params?): Promise<JournalEntry[]>
```

---

## TypeScript Types (`types/index.ts`)

### Core Data Types
```tsx
interface KlineData {
  date: string; open: number; high: number; low: number; close: number; volume: number;
}

interface StockQuote {
  symbol: string; price: number; change: number; change_percent: number; volume: number; timestamp: string;
}

interface Position {
  symbol: string; name: string; shares: number; avg_cost: number; current_price: number; pnl: number;
}

interface Portfolio {
  total_value: number; cash: number; total_pnl: number; positions: Position[];
}

interface BacktestResult {
  id: string; strategy: string; symbol: string; start: string; end: string;
  initial_capital: number; final_value: number; total_return_percent: number;
  sharpe_ratio: number; max_drawdown_percent: number; win_rate_percent: number;
  total_trades: number; profit_factor: number; status: string;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'; content: string; timestamp?: Date;
}

interface StrategyConfig {
  name: string; displayName: string; description: string; parameters: StrategyParam[];
}

interface StrategyParam {
  key: string; label: string; type: 'number' | 'select';
  default: number | string; min?: number; max?: number; step?: number;
  options?: { value: string; label: string }[];
}
```

### Task Types (from `api/client.ts`)
```tsx
interface TaskRecord {
  id: string; task_type: string;
  status: 'Running' | 'Pending' | 'Completed' | 'Failed';
  created_at: string; updated_at: string;
  progress: string | null; result: string | null;
  error: string | null; parameters: string | null;
}

interface FactorRegistryEntry {
  expression: string; source: string;
  state: 'candidate' | 'validated' | 'promoted' | 'retired';
  created: string; tree_size: number; tree_depth: number;
  ic_mean: number; ir: number; ic_pos_rate: number;
  turnover: number; decay: number;
  ic_history: Array<{timestamp: string, ic: number, ir: number}>;
  validation_count: number; fail_count: number;
  last_validated: string | null; promoted_at: string | null; retired_at: string | null;
}
```

---

## Styling Conventions

### Dark Theme (CSS Variables in `index.css`)
```css
--bg-primary: #0f172a;      /* Main background (slate-900) */
--bg-secondary: #1e293b;    /* Card/panel background (slate-800) */
--bg-tertiary: #334155;     /* Elevated surfaces (slate-700) */
--text-primary: #f8fafc;    /* Main text (slate-50) */
--text-secondary: #94a3b8;  /* Muted text (slate-400) */
--accent: #3b82f6;          /* Primary blue (blue-500) */
--accent-hover: #2563eb;    /* Blue hover (blue-600) */
--green: #22c55e;           /* Positive/profit (green-500) */
--red: #ef4444;             /* Negative/loss (red-500) */
--yellow: #eab308;          /* Warning (yellow-500) */
```

### Tailwind Patterns Used Everywhere
```
bg-[#0f172a]       → Primary background
bg-[#1e293b]       → Card/panel
text-white          → Primary text
text-gray-400       → Secondary text
text-green-400      → Positive values
text-red-400        → Negative values
rounded-xl          → Card corners
border border-white/10 → Subtle borders
px-6 py-4          → Standard card padding
gap-4              → Standard grid gap
```

### Card Pattern
```tsx
<div className="bg-[#1e293b] rounded-xl p-6 border border-white/10">
  <h3 className="text-white font-semibold mb-4">Title</h3>
  {/* Content */}
</div>
```

### Status Badge Pattern
```tsx
<span className={`px-2 py-0.5 rounded text-xs font-medium ${
  status === 'Completed' ? 'bg-green-500/20 text-green-400' :
  status === 'Failed' ? 'bg-red-500/20 text-red-400' :
  status === 'Running' ? 'bg-blue-500/20 text-blue-400' :
  'bg-gray-500/20 text-gray-400'
}`}>
  {status}
</span>
```

### Table Pattern
```tsx
<table className="w-full text-sm">
  <thead>
    <tr className="text-gray-400 border-b border-white/10">
      <th className="pb-2 text-left">Column</th>
    </tr>
  </thead>
  <tbody>
    {items.map(item => (
      <tr key={item.id} className="border-b border-white/5 hover:bg-white/5">
        <td className="py-2 text-white">{item.value}</td>
      </tr>
    ))}
  </tbody>
</table>
```

---

## Key Page Patterns

### Dashboard — Auto-Refresh + Market Filtering
```tsx
// Polls getDashboard() every 5 seconds
useEffect(() => {
  const load = () => getDashboard().then(setData);
  load();
  const id = setInterval(load, 5000);
  return () => clearInterval(id);
}, []);

// Filters positions by market context
const filtered = filterByMarket(data.positions, 'symbol');
```

### Backtest — Async Task Pattern
```tsx
const [taskId, setTaskId] = useState<string | null>(null);
const [result, setResult] = useState(null);

const handleRun = async () => {
  const resp = await runBacktest(params);
  if (resp.task_id) {
    setTaskId(resp.task_id);
    // Poll for results
    const poll = setInterval(async () => {
      const task = await getTask(resp.task_id);
      if (task.status === 'Completed') {
        clearInterval(poll);
        setResult(JSON.parse(task.result));
      }
    }, 2000);
  }
};
```

### Factor Mining — useTaskManager Pattern
```tsx
const tm = useTaskManager('task_gp_mining');

<button onClick={() => tm.submit(() => factorMineGP(params))} disabled={tm.running}>
  {tm.running ? '运行中...' : '开始挖掘'}
</button>
<TaskOutput running={tm.running} error={tm.error} output={tm.output} progress={tm.progress} />
```

### Risk Management — WebSocket Updates
```tsx
useEffect(() => {
  const ws = createMonitorWebSocket();
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    setRiskStatus(data);
  };
  return () => ws.close();
}, []);
```

### Chart — Custom SVG Equity Curve (Dashboard)
```tsx
// Dashboard uses custom SVG path with linear gradient fill
// Not Recharts — hand-drawn for performance
<svg viewBox={`0 0 ${width} ${height}`}>
  <defs>
    <linearGradient id="eq-grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
      <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
    </linearGradient>
  </defs>
  <path d={areaPath} fill="url(#eq-grad)" />
  <path d={linePath} stroke="#3b82f6" fill="none" strokeWidth="2" />
</svg>
```

### Chart — Recharts (Portfolio Pie, Backtest)
```tsx
import { PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
// Used in: Portfolio (pie), Backtest (equity/drawdown), Overview (pie), etc.
```

---

## Development Workflow

### Scripts
```bash
npm run dev           # Vite dev server on port 3000 (HMR)
npm run build         # tsc -b && vite build → web/dist/
npm run lint          # ESLint
npm run test          # Playwright E2E tests
npm run test:ui       # Interactive Playwright
npm run test:headed   # Headed browser mode
```

### Vite Proxy Config (`vite.config.ts`)
```ts
server: {
  port: 3000,
  proxy: {
    '/api': 'http://localhost:8080',   // Backend API
    '/ws': { target: 'ws://localhost:8080', ws: true }  // WebSocket
  }
}
```

### E2E Tests (`e2e/`)
- Playwright with Chromium
- Tests page navigation, API interactions
- Config in `playwright.config.ts`

---

## Conventions for New Development

### Adding a New Page
1. Create `web/src/pages/NewPage.tsx`
2. Add route in `App.tsx`: `<Route path="/new-page" element={<NewPage />} />`
3. Add navigation item in `Sidebar.tsx` (under appropriate group)
4. If page needs persistent mounting (task state), add to Layout.tsx persistent list

### Adding a New API Call
1. Add function to `web/src/api/client.ts` following existing patterns
2. Use `fetchJson<ReturnType>(url, options?)` helper
3. For POST: pass `{ method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) }`
4. For async tasks: return `{ task_id: string }` and use `useTaskManager` hook

### Adding a New Component
1. Create in `web/src/components/`
2. Use Tailwind classes with dark theme variables
3. Follow card pattern: `bg-[#1e293b] rounded-xl p-6 border border-white/10`
4. Use `text-white` for primary, `text-gray-400` for secondary text
5. Color code values: `text-green-400` (positive), `text-red-400` (negative)

### State Management Pattern
- **Global state**: React Context (`MarketContext`)
- **Page-level state**: `useState` + `useEffect` for data fetching
- **Task state**: `useTaskManager` hook with sessionStorage persistence
- **Real-time data**: WebSocket via `createMonitorWebSocket()` / `createChatWebSocket()`
- **No Redux/Zustand** — Context + hooks is the pattern

### Data Refresh Patterns
| Pattern | Interval | Used By |
|---------|----------|---------|
| Auto-refresh | 5s | Dashboard |
| Auto-refresh | 10s | Portfolio |
| Auto-refresh | 15s | Notification count (Layout) |
| Task polling | 2s | useTaskPoller (background tasks) |
| Auto-refresh | 3s | AutoTrade status (when running) |
| WebSocket | Real-time | Risk management, Chat streaming |

### Strategies (Frontend-Known)
The frontend knows about 6 built-in strategies:
1. **SMA Cross** — `sma_cross` — fast/slow period params
2. **RSI Reversal** — `rsi_reversal` — period/overbought/oversold params
3. **MACD Trend** — `macd_trend` — fast/slow/signal params
4. **Multi-Factor** — `multi_factor` — 6 factor weights + thresholds
5. **Sentiment-Aware** — `sentiment_aware` — wraps multi-factor + sentiment weight
6. **ML Factor** — `ml_factor` — model_path, confidence threshold, inference mode

### Trading Modes (AutoTrade page)
| Mode | Description |
|------|-------------|
| `paper` | Simulated execution with configurable slippage |
| `live` | Real-time data, paper broker |
| `qmt` | Real execution via QMT bridge (Chinese broker) |
| `replay` | Historical data replay (start/end date, speed control) |
| `l2` | Level-2 tick/depth data mode (experimental) |

### Chinese UI Labels
The app uses Chinese labels for navigation and headers:
- 仪表盘 = Dashboard, 行情 = Market, 策略 = Strategy
- 回测 = Backtest, 持仓 = Portfolio, 智能选股 = Smart Screening
- 自动交易 = Auto Trade, 风控 = Risk Control, 舆情 = Sentiment
- 量化流水线 = Quant Pipeline, 任务历史 = Task History
