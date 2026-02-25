const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const { headers: extraHeaders, ...rest } = options ?? {};
  const res = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(extraHeaders instanceof Headers
        ? Object.fromEntries(extraHeaders.entries())
        : (extraHeaders as Record<string, string>) ?? {}),
    },
    ...rest,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API error ${res.status}: ${text || res.statusText}`);
  }
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

export async function closePosition(symbol: string, price?: number) {
  return fetchJson('/portfolio/close', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, price }),
  });
}

export async function getClosedPositions() {
  return fetchJson('/portfolio/closed');
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

export interface RiskMetrics {
  max_drawdown_20d: number;
  consecutive_down_days: number;
  distance_from_high_20d: number;
  atr_ratio: number;
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
  sector: string;
  avg_turnover: number;
  mined_factor_bonus: number;
  risk: RiskMetrics;
}

export interface ScreenerResult {
  candidates: StockCandidate[];
  total_scanned: number;
  phase1_passed: number;
  phase2_passed: number;
  regime: string | null;
}

export async function screenScan(
  topN: number = 10,
  minVotes: number = 2,
  pool?: string,
  buyThreshold?: number,
  strongBuyThreshold?: number,
  minTurnover?: number,
  maxPerSector?: number,
): Promise<ScreenerResult> {
  const body: Record<string, unknown> = { top_n: topN, min_votes: minVotes };
  if (pool) body.pool = pool;
  if (buyThreshold !== undefined) body.buy_threshold = buyThreshold;
  if (strongBuyThreshold !== undefined) body.strong_buy_threshold = strongBuyThreshold;
  if (minTurnover !== undefined) body.min_turnover = minTurnover;
  if (maxPerSector !== undefined) body.max_per_sector = maxPerSector;
  return fetchJson('/screen/scan', {
    method: 'POST',
    body: JSON.stringify(body),
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
  avg_risk_check_us: number;
  avg_order_submit_us: number;
  total_risk_checks: number;
  total_orders_submitted: number;
  last_data_fetch_us: number;
  avg_data_fetch_us: number;
  total_data_fetches: number;
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
  slippage_bps?: number;
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

export async function mlRetrain(opts?: RetrainOptions): Promise<TaskSubmitResult> {
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

export async function getRiskSignals(): Promise<RiskSignalsSnapshot> {
  return fetchJson('/trade/risk/signals');
}

export interface RiskSignalsSnapshot {
  status: RiskStatusData;
  vol_spike_active: boolean;
  tail_risk: TailRiskData | null;
  return_history_len: number;
  recent_events: RiskEvent[];
}

export interface RiskStatusData {
  daily_pnl: number;
  daily_paused: boolean;
  peak_value: number;
  drawdown_halted: boolean;
  consecutive_failures: number;
  circuit_open: boolean;
  vol_spike_active: boolean;
  config: RiskConfigData;
}

export interface RiskConfigData {
  stop_loss_pct: number;
  max_daily_loss_pct: number;
  max_drawdown_pct: number;
  circuit_breaker_failures: number;
  halt_on_drawdown: boolean;
  vol_spike_ratio: number;
  vol_deleverage_factor: number;
  max_correlated_exposure: number;
  var_confidence: number;
  max_var_pct: number;
}

export interface TailRiskData {
  var_pct: number;
  cvar_pct: number;
  confidence: number;
  max_allowed: number;
  breach: boolean;
}

export interface RiskEvent {
  timestamp: string;
  severity: 'Info' | 'Warning' | 'Critical';
  event_type: string;
  message: string;
  detail?: Record<string, unknown>;
}

export function createMonitorWebSocket(): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return new WebSocket(`${protocol}//${window.location.host}/api/monitor/ws`);
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

// ── Factor Mining API ───────────────────────────────────────────────

export interface FactorMiningResult {
  status: string;
  stdout: string;
  stderr: string;
}

export interface FactorRegistryEntry {
  expression: string;
  source: string;
  state: 'candidate' | 'validated' | 'promoted' | 'retired';
  created: string;
  tree_size: number;
  tree_depth: number;
  ic_mean: number;
  ir: number;
  ic_pos_rate: number;
  turnover: number;
  decay: number;
  ic_history: Array<{ timestamp: string; ic: number; ir: number }>;
  validation_count: number;
  fail_count: number;
  last_validated: string | null;
  promoted_at: string | null;
  retired_at: string | null;
}

export interface FactorRegistry {
  factors: Record<string, FactorRegistryEntry>;
  stats: {
    total_discovered: number;
    total_promoted: number;
    total_retired: number;
  };
  updated?: string;
}

export interface GpFeature {
  id: string;
  expression: string;
}

export interface FactorResults {
  parametric: {
    features: string[];
    latest_report: Record<string, unknown> | null;
    rust_snippet: string;
  };
  gp: {
    features: GpFeature[];
    rust_snippet: string;
  };
}

export async function factorMineParametric(params?: {
  n_bars?: number;
  horizon?: number;
  ic_threshold?: number;
  top_n?: number;
  retrain?: boolean;
  cross_stock?: boolean;
  data_source?: string;
  symbols?: string;
  start_date?: string;
  end_date?: string;
}): Promise<TaskSubmitResult> {
  return fetchJson('/factor/mine/parametric', {
    method: 'POST',
    body: JSON.stringify(params ?? {}),
  });
}

export async function factorMineGP(params?: {
  n_bars?: number;
  pop_size?: number;
  generations?: number;
  max_depth?: number;
  horizon?: number;
  retrain?: boolean;
  data_source?: string;
  symbols?: string;
  start_date?: string;
  end_date?: string;
}): Promise<TaskSubmitResult> {
  return fetchJson('/factor/mine/gp', {
    method: 'POST',
    body: JSON.stringify(params ?? {}),
  });
}

export async function factorRegistryGet(): Promise<FactorRegistry> {
  return fetchJson('/factor/registry');
}

export async function factorRegistryManage(params?: {
  n_bars?: number;
  data?: string;
}): Promise<TaskSubmitResult> {
  return fetchJson('/factor/manage', {
    method: 'POST',
    body: JSON.stringify(params ?? {}),
  });
}

export async function factorExportPromoted(params?: {
  retrain?: boolean;
  data?: string;
}): Promise<TaskSubmitResult> {
  return fetchJson('/factor/export', {
    method: 'POST',
    body: JSON.stringify(params ?? {}),
  });
}

export async function factorResults(): Promise<FactorResults> {
  return fetchJson('/factor/results');
}

// ── Notifications ───────────────────────────────────────────────────

export interface NotificationItem {
  id: string;
  timestamp: string;
  event_type: string;
  title: string;
  message: string;
  symbol?: string;
  side?: string;
  quantity?: number;
  price?: number;
  read: boolean;
  delivery: string;
  channels: string;
}

export interface NotificationList {
  notifications: NotificationItem[];
  unread_count: number;
}

export interface NotificationConfig {
  enabled: boolean;
  in_app: boolean;
  email: {
    enabled: boolean;
    smtp_host: string;
    smtp_port: number;
    username: string;
    password: string;
    from: string;
    to: string[];
    tls: boolean;
  };
  webhook: {
    enabled: boolean;
    provider: string;
    url: string;
    secret: string;
  };
  events: {
    order_filled: boolean;
    order_rejected: boolean;
    risk_alert: boolean;
    engine_started: boolean;
    engine_stopped: boolean;
  };
}

export async function getNotifications(limit = 50, unread_only = false): Promise<NotificationList> {
  return fetchJson(`/notifications?limit=${limit}&unread_only=${unread_only}`);
}

export async function getUnreadCount(): Promise<{ unread_count: number }> {
  return fetchJson('/notifications/unread-count');
}

export async function markNotificationRead(id: string): Promise<void> {
  await fetch(`${API_BASE}/notifications/${id}/read`, { method: 'POST' });
}

export async function markAllNotificationsRead(): Promise<void> {
  await fetch(`${API_BASE}/notifications/read-all`, { method: 'POST' });
}

export async function getNotificationConfig(): Promise<NotificationConfig> {
  return fetchJson('/notifications/config');
}

export async function saveNotificationConfig(cfg: NotificationConfig): Promise<void> {
  await fetch(`${API_BASE}/notifications/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
}

export async function testNotification(): Promise<{ results: Array<{ channel: string; success: boolean; message: string }> }> {
  return fetchJson('/notifications/test', { method: 'POST' });
}

// ── Metrics ─────────────────────────────────────────────────────────

export async function getMetrics(): Promise<any> {
  return fetchJson('/metrics');
}

// ── Reports ─────────────────────────────────────────────────────────

export async function getReports(): Promise<any> {
  return fetchJson('/reports');
}

// ── Latency ─────────────────────────────────────────────────────────

export async function getLatency(): Promise<any> {
  return fetchJson('/latency');
}

// ── Background Tasks ────────────────────────────────────────────────

export interface TaskSubmitResult {
  task_id: string;
  status: string;
}

export interface TaskRecord {
  id: string;
  task_type: string;
  status: 'Pending' | 'Running' | 'Completed' | 'Failed';
  created_at: string;
  updated_at: string;
  progress: string | null;
  result: string | null;
  error: string | null;
}

export async function getTask(taskId: string): Promise<TaskRecord> {
  return fetchJson(`/tasks/${taskId}`);
}

export async function listTasks(): Promise<{ tasks: TaskRecord[] }> {
  return fetchJson('/tasks');
}

export async function listRunningTasks(): Promise<{ tasks: TaskRecord[] }> {
  return fetchJson('/tasks/running');
}
