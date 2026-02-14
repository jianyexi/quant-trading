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
