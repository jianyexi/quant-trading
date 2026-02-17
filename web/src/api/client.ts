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
  period?: string;
  inference_mode?: string;
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

export interface PipelineLatency {
  last_factor_compute_us: number;
  avg_factor_compute_us: number;
  last_risk_check_us: number;
  last_order_submit_us: number;
  total_bars_processed: number;
}

export interface TradeStatus {
  running: boolean;
  strategy: string;
  symbols: string[];
  total_signals: number;
  total_orders: number;
  total_fills: number;
  total_rejected: number;
  pnl: number;
  latency?: PipelineLatency;
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
  mode?: string;
  replay_start?: string;
  replay_end?: string;
  replay_speed?: number;
  replay_period?: string;
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

export interface QmtBridgeStatus {
  status: string;
  connected?: boolean;
  account?: string;
  message?: string;
  bridge_url?: string;
}

export async function qmtBridgeStatus(): Promise<QmtBridgeStatus> {
  return fetchJson('/trade/qmt/status');
}

// ── Sentiment API ───────────────────────────────────────────────────

export interface SentimentItem {
  id: string;
  source: string;
  title: string;
  content: string;
  sentiment_score: number;
  level: string;
  published_at: string;
}

export interface SentimentSummaryEntry {
  symbol: string;
  count: number;
  avg_score: number;
  level: string;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  latest_title: string;
  latest_at: string | null;
}

export interface SentimentQueryResult {
  symbol: string;
  summary: {
    count: number;
    avg_score: number;
    level: string;
    bullish_count: number;
    bearish_count: number;
    neutral_count: number;
  };
  items: SentimentItem[];
}

export interface SentimentOverview {
  total_items: number;
  symbols: SentimentSummaryEntry[];
}

export async function sentimentSubmit(params: {
  symbol: string;
  source: string;
  title: string;
  content?: string;
  sentiment_score: number;
  published_at?: string;
}) {
  return fetchJson('/sentiment/submit', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function sentimentBatchSubmit(items: Array<{
  symbol: string;
  source: string;
  title: string;
  content?: string;
  sentiment_score: number;
  published_at?: string;
}>) {
  return fetchJson('/sentiment/batch', {
    method: 'POST',
    body: JSON.stringify(items),
  });
}

export async function sentimentQuery(symbol: string, limit?: number): Promise<SentimentQueryResult> {
  const params = new URLSearchParams();
  if (limit) params.set('limit', String(limit));
  return fetchJson(`/sentiment/${symbol}?${params}`);
}

export async function sentimentSummary(): Promise<SentimentOverview> {
  return fetchJson('/sentiment/summary');
}

// ── DL Models Research API ──────────────────────────────────────────

export interface DlModelEntry {
  id: string;
  name: string;
  category: string;
  year: number;
  architecture: string;
  description: string;
  key_innovation: string;
  input_data: string[];
  output: string;
  strengths: string[];
  limitations: string[];
  reference: string;
  reference_url: string;
}

export interface ModelCategory {
  name: string;
  description: string;
  models: DlModelEntry[];
}

export interface CollectedResearch {
  title: string;
  summary: string;
  source: string;
  relevance: string;
  collected_at: string;
}

export interface ResearchKnowledgeBase {
  categories: ModelCategory[];
  collected: CollectedResearch[];
  last_updated: string;
}

export interface KnowledgeBaseSummary {
  total_models: number;
  total_categories: number;
  total_collected: number;
  categories: Array<{ name: string; count: number; description: string }>;
  last_updated: string;
}

export interface CollectResult {
  status: string;
  topic: string;
  collected: CollectedResearch[];
  raw_response?: string;
  message?: string;
  hint?: string;
}

export async function getResearchDlModels(): Promise<ResearchKnowledgeBase> {
  return fetchJson('/research/dl-models');
}

export async function getResearchSummary(): Promise<KnowledgeBaseSummary> {
  return fetchJson('/research/dl-models/summary');
}

export async function collectResearch(topic?: string): Promise<CollectResult> {
  return fetchJson('/research/dl-models/collect', {
    method: 'POST',
    body: JSON.stringify({ topic }),
  });
}

// ── Trade Journal API ───────────────────────────────────────────────

export interface JournalEntry {
  id: string;
  timestamp: string;
  entry_type: string;
  symbol: string;
  side: string | null;
  quantity: number | null;
  price: number | null;
  order_id: string | null;
  status: string | null;
  reason: string | null;
  pnl: number | null;
  portfolio_value: number | null;
  cash: number | null;
  details: string | null;
}

export interface JournalResult {
  total: number;
  entries: JournalEntry[];
  stats: Array<{ type: string; count: number }>;
}

export async function getJournal(params?: {
  symbol?: string;
  entry_type?: string;
  start?: string;
  end?: string;
  limit?: number;
}): Promise<JournalResult> {
  const qs = new URLSearchParams();
  if (params?.symbol) qs.set('symbol', params.symbol);
  if (params?.entry_type) qs.set('entry_type', params.entry_type);
  if (params?.start) qs.set('start', params.start);
  if (params?.end) qs.set('end', params.end);
  if (params?.limit) qs.set('limit', String(params.limit));
  return fetchJson(`/journal?${qs}`);
}

// ── Strategy Config Persistence API ─────────────────────────────────

export async function saveStrategyConfig(config: Record<string, unknown>) {
  return fetchJson('/strategy/config', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function loadStrategyConfig(): Promise<{ config: Record<string, unknown> | null; exists: boolean }> {
  return fetchJson('/strategy/config');
}

// ── ML Model Training API ───────────────────────────────────────────

export interface ModelInfo {
  model_dir: string;
  default_model: string;
  supported_algorithms: string[];
  latest_report: Record<string, unknown> | null;
}

export interface RetrainOptions {
  algorithms?: string;
  data_source?: 'synthetic' | 'akshare';
  symbols?: string;
  start_date?: string;
  end_date?: string;
  horizon?: number;
  threshold?: number;
}

export async function mlRetrain(opts?: RetrainOptions): Promise<{ status: string; stdout: string; stderr: string }> {
  return fetchJson('/trade/retrain', {
    method: 'POST',
    body: JSON.stringify(opts ?? {}),
  });
}

export async function mlModelInfo(): Promise<ModelInfo> {
  return fetchJson('/trade/model-info');
}

// ── Risk Status API ─────────────────────────────────────────────────

export async function getRiskStatus() {
  return fetchJson('/trade/risk');
}

export async function resetCircuitBreaker() {
  return fetchJson('/trade/risk/reset-circuit', { method: 'POST' });
}

export async function resetDailyLoss() {
  return fetchJson('/trade/risk/reset-daily', { method: 'POST' });
}

export async function getPerformance() {
  return fetchJson('/trade/performance');
}
