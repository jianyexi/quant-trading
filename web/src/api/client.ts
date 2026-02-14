const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function getDashboard() {
  return fetchJson('/dashboard');
}

export async function getStrategies() {
  return fetchJson('/strategies');
}

export async function getStocks() {
  return fetchJson('/market/stocks');
}

export async function getKline(symbol: string, start?: string, end?: string, limit?: number) {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  if (limit) params.set('limit', String(limit));
  return fetchJson(`/market/kline/${symbol}?${params}`);
}

export async function getQuote(symbol: string) {
  return fetchJson(`/market/quote/${symbol}`);
}

export async function runBacktest(params: {
  strategy: string;
  symbol: string;
  start: string;
  end: string;
  capital?: number;
}) {
  return fetchJson('/backtest/run', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getBacktestResults(id: string) {
  return fetchJson(`/backtest/results/${id}`);
}

export async function getPortfolio() {
  return fetchJson('/portfolio');
}

export async function getOrders() {
  return fetchJson('/orders');
}

export async function sendChat(message: string) {
  return fetchJson<{ reply: string }>('/chat', {
    method: 'POST',
    body: JSON.stringify({ message }),
  });
}

export async function getChatHistory() {
  return fetchJson('/chat/history');
}

export function createChatWebSocket(): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return new WebSocket(`${protocol}//${window.location.host}/api/chat/stream`);
}

// ── Screener API ────────────────────────────────────────────────────

export interface FactorScores {
  momentum_5d: number;
  momentum_20d: number;
  rsi_14: number;
  macd_histogram: number;
  bollinger_position: number;
  volume_ratio: number;
  ma_trend: number;
  kdj_j: number;
  volatility_20d: number;
}

export interface VoteResult {
  Buy?: number;
  Sell?: number;
  Neutral?: string;
}

export interface StrategyVote {
  sma_cross: VoteResult;
  rsi_reversal: VoteResult;
  macd_trend: VoteResult;
  consensus_count: number;
  avg_confidence: number;
}

export interface StockCandidate {
  symbol: string;
  name: string;
  price: number;
  factor_score: number;
  factors: FactorScores;
  strategy_vote: StrategyVote;
  composite_score: number;
  recommendation: string;
  reasons: string[];
}

export interface ScreenerResult {
  candidates: StockCandidate[];
  total_scanned: number;
  phase1_passed: number;
  phase2_passed: number;
}

export async function screenScan(topN: number = 10, minVotes: number = 2): Promise<ScreenerResult> {
  return fetchJson('/screen/scan', {
    method: 'POST',
    body: JSON.stringify({ top_n: topN, min_votes: minVotes }),
  });
}

export async function screenFactors(symbol: string): Promise<StockCandidate> {
  return fetchJson(`/screen/factors/${symbol}`);
}

// ── Auto-Trade API ──────────────────────────────────────────────────

export interface TradeStatus {
  running: boolean;
  strategy: string;
  symbols: string[];
  total_signals: number;
  total_orders: number;
  total_fills: number;
  total_rejected: number;
  pnl: number;
  recent_trades: Array<{
    side: string;
    symbol: string;
    quantity: number;
    price: number;
    commission: number;
  }>;
}

export async function tradeStart(params: {
  strategy: string;
  symbols: string[];
  interval: number;
  position_size: number;
}) {
  return fetchJson('/trade/start', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function tradeStop() {
  return fetchJson('/trade/stop', { method: 'POST' });
}

export async function tradeStatus(): Promise<TradeStatus> {
  return fetchJson('/trade/status');
}
