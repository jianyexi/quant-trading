import { useState, useEffect, useCallback } from 'react';

const API_BASE = '/api';

interface LogEntry {
  id: number;
  timestamp: string;
  level: 'info' | 'warn' | 'error';
  method: string;
  path: string;
  status: number;
  duration_ms: number;
  message: string;
  detail: string | null;
}

interface LogSummary {
  info: number;
  warn: number;
  error: number;
}

interface LogResponse {
  total: number;
  entries: LogEntry[];
  summary: LogSummary;
}

const LEVEL_COLORS: Record<string, string> = {
  info: 'text-blue-400',
  warn: 'text-yellow-400',
  error: 'text-red-400',
};

const LEVEL_BG: Record<string, string> = {
  info: 'bg-blue-500/10 border-blue-500/20',
  warn: 'bg-yellow-500/10 border-yellow-500/20',
  error: 'bg-red-500/10 border-red-500/20',
};

const LEVEL_BADGE: Record<string, string> = {
  info: 'bg-blue-500/20 text-blue-300',
  warn: 'bg-yellow-500/20 text-yellow-300',
  error: 'bg-red-500/20 text-red-300',
};

export default function Logs() {
  const [data, setData] = useState<LogResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [levelFilter, setLevelFilter] = useState<string>('all');
  const [pathFilter, setPathFilter] = useState('');
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [limit, setLimit] = useState(200);

  const fetchLogs = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (levelFilter !== 'all') params.set('level', levelFilter);
      if (pathFilter.trim()) params.set('path', pathFilter.trim());
      params.set('limit', String(limit));
      const res = await fetch(`${API_BASE}/logs?${params}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json: LogResponse = await res.json();
      setData(json);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load logs');
    }
  }, [levelFilter, pathFilter, limit]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchLogs, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchLogs]);

  const handleClear = async () => {
    try {
      await fetch(`${API_BASE}/logs`, { method: 'DELETE' });
      fetchLogs();
    } catch { /* ignore */ }
  };

  const formatTime = (ts: string) => {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
        + '.' + String(d.getMilliseconds()).padStart(3, '0');
    } catch {
      return ts;
    }
  };

  const formatDate = (ts: string) => {
    try {
      return new Date(ts).toLocaleDateString('zh-CN');
    } catch {
      return '';
    }
  };

  if (error) return <div className="text-red-400 p-4">Error: {error}</div>;
  if (!data) return <div className="text-[#94a3b8] p-4">Loading logs...</div>;

  const { entries, summary, total } = data;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[#f8fafc]">📋 系统日志</h1>
        <div className="flex items-center gap-3">
          <span className="text-sm text-[#94a3b8]">共 {total} 条</span>
          <label className="flex items-center gap-1.5 text-sm text-[#94a3b8] cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            自动刷新
          </label>
          <button onClick={fetchLogs} className="px-3 py-1.5 rounded bg-[#334155] text-[#f8fafc] text-sm hover:bg-[#475569]">
            🔄 刷新
          </button>
          <button onClick={handleClear} className="px-3 py-1.5 rounded bg-red-500/20 text-red-300 text-sm hover:bg-red-500/30">
            🗑 清除
          </button>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3">
          <div className="text-2xl font-bold text-blue-400">{summary.info}</div>
          <div className="text-sm text-blue-300/70">INFO</div>
        </div>
        <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-3">
          <div className="text-2xl font-bold text-yellow-400">{summary.warn}</div>
          <div className="text-sm text-yellow-300/70">WARN</div>
        </div>
        <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3">
          <div className="text-2xl font-bold text-red-400">{summary.error}</div>
          <div className="text-sm text-red-300/70">ERROR</div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 p-3 rounded-lg bg-[#1e293b] border border-[#334155]">
        <span className="text-sm text-[#94a3b8]">过滤:</span>
        <div className="flex gap-1">
          {['all', 'error', 'warn', 'info'].map((lvl) => (
            <button
              key={lvl}
              onClick={() => setLevelFilter(lvl)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                levelFilter === lvl
                  ? 'bg-[#3b82f6] text-white'
                  : 'bg-[#334155] text-[#94a3b8] hover:text-[#f8fafc]'
              }`}
            >
              {lvl === 'all' ? '全部' : lvl.toUpperCase()}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="路径筛选 (如 /api/market)"
          value={pathFilter}
          onChange={(e) => setPathFilter(e.target.value)}
          className="flex-1 bg-[#0f172a] border border-[#334155] rounded px-3 py-1.5 text-sm text-[#f8fafc] placeholder-[#64748b]"
        />
        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="bg-[#0f172a] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#f8fafc]"
        >
          <option value={50}>50条</option>
          <option value={200}>200条</option>
          <option value={500}>500条</option>
          <option value={2000}>全部</option>
        </select>
      </div>

      {/* Log entries */}
      <div className="rounded-lg border border-[#334155] bg-[#0f172a] divide-y divide-[#1e293b] max-h-[calc(100vh-400px)] overflow-y-auto">
        {entries.length === 0 ? (
          <div className="p-8 text-center text-[#64748b]">暂无日志</div>
        ) : (
          entries.map((entry) => {
            const isExpanded = expandedId === entry.id;
            return (
              <div
                key={entry.id}
                className={`px-4 py-2 hover:bg-[#1e293b]/50 cursor-pointer transition-colors ${
                  entry.level === 'error' ? 'border-l-2 border-l-red-500' :
                  entry.level === 'warn' ? 'border-l-2 border-l-yellow-500' : ''
                }`}
                onClick={() => setExpandedId(isExpanded ? null : entry.id)}
              >
                <div className="flex items-center gap-3 text-sm">
                  {/* Timestamp */}
                  <span className="text-[#64748b] font-mono text-xs w-20 shrink-0">
                    {formatTime(entry.timestamp)}
                  </span>

                  {/* Level badge */}
                  <span className={`px-1.5 py-0.5 rounded text-xs font-bold shrink-0 ${LEVEL_BADGE[entry.level] || ''}`}>
                    {entry.level.toUpperCase()}
                  </span>

                  {/* Method */}
                  <span className={`font-mono text-xs shrink-0 w-12 ${
                    entry.method === 'GET' ? 'text-green-400' :
                    entry.method === 'POST' ? 'text-blue-400' :
                    entry.method === 'DELETE' ? 'text-red-400' :
                    'text-[#94a3b8]'
                  }`}>
                    {entry.method}
                  </span>

                  {/* Path */}
                  <span className="font-mono text-[#e2e8f0] text-xs truncate flex-1 min-w-0">
                    {entry.path}
                  </span>

                  {/* Status */}
                  <span className={`font-mono text-xs shrink-0 ${
                    entry.status >= 500 ? 'text-red-400' :
                    entry.status >= 400 ? 'text-yellow-400' :
                    entry.status > 0 ? 'text-green-400' : 'text-[#64748b]'
                  }`}>
                    {entry.status > 0 ? entry.status : '—'}
                  </span>

                  {/* Duration */}
                  <span className={`font-mono text-xs shrink-0 w-16 text-right ${
                    entry.duration_ms > 5000 ? 'text-red-400' :
                    entry.duration_ms > 1000 ? 'text-yellow-400' : 'text-[#64748b]'
                  }`}>
                    {entry.duration_ms}ms
                  </span>
                </div>

                {/* Expanded detail */}
                {isExpanded && (
                  <div className="mt-2 space-y-1">
                    <div className="text-xs text-[#94a3b8]">
                      <span className="text-[#64748b]">时间:</span> {formatDate(entry.timestamp)} {formatTime(entry.timestamp)}
                    </div>
                    <div className={`text-xs ${LEVEL_COLORS[entry.level] || 'text-[#94a3b8]'}`}>
                      {entry.message}
                    </div>
                    {entry.detail && (
                      <pre className={`mt-1 p-2 rounded text-xs font-mono whitespace-pre-wrap break-all border ${LEVEL_BG[entry.level] || ''}`}>
                        {entry.detail}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
