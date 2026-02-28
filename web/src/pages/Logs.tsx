import { useState, useEffect, useCallback } from 'react';
import { listTasks, cancelTask } from '../api/client';
import type { TaskRecord } from '../api/client';

const API_BASE = '/api';

// ── Types ──────────────────────────────────────────────────────────

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

interface LogSummary { info: number; warn: number; error: number; }
interface LogResponse { total: number; entries: LogEntry[]; summary: LogSummary; }
interface JournalEntry { id: number; timestamp: string; event_type: string; symbol: string; side: string; quantity: number; price: number; pnl: number | null; detail: string | null; }

type TabKey = 'tasks' | 'logs' | 'journal';

const TABS: { key: TabKey; label: string; icon: string }[] = [
  { key: 'tasks', label: '任务', icon: '⚙️' },
  { key: 'logs', label: 'API 日志', icon: '📋' },
  { key: 'journal', label: '交易事件', icon: '📈' },
];

const STATUS_STYLE: Record<string, string> = {
  Running: 'bg-blue-500/20 text-blue-300 animate-pulse',
  Pending: 'bg-yellow-500/20 text-yellow-300',
  Completed: 'bg-green-500/20 text-green-300',
  Failed: 'bg-red-500/20 text-red-300',
};

const LEVEL_BADGE: Record<string, string> = {
  info: 'bg-blue-500/20 text-blue-300',
  warn: 'bg-yellow-500/20 text-yellow-300',
  error: 'bg-red-500/20 text-red-300',
};

// ── Helpers ────────────────────────────────────────────────────────

const fmtTime = (ts: string) => {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
      + '.' + String(d.getMilliseconds()).padStart(3, '0');
  } catch { return ts; }
};

const fmtDateTime = (ts: string) => {
  try {
    const d = new Date(ts);
    return d.toLocaleDateString('zh-CN') + ' ' + d.toLocaleTimeString('zh-CN', { hour12: false });
  } catch { return ts; }
};

const elapsed = (start: string, end: string) => {
  try {
    const ms = new Date(end).getTime() - new Date(start).getTime();
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  } catch { return '—'; }
};

// ── Main Component ─────────────────────────────────────────────────

export default function Logs() {
  const [tab, setTab] = useState<TabKey>('tasks');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Tasks state
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [taskFilter, setTaskFilter] = useState<string>('all');
  const [expandedTask, setExpandedTask] = useState<string | null>(null);

  // Logs state
  const [logData, setLogData] = useState<LogResponse | null>(null);
  const [levelFilter, setLevelFilter] = useState<string>('all');
  const [pathFilter, setPathFilter] = useState('');
  const [expandedLog, setExpandedLog] = useState<number | null>(null);
  const [logLimit, setLogLimit] = useState(200);

  // Journal state
  const [journal, setJournal] = useState<JournalEntry[]>([]);

  const [error, setError] = useState<string | null>(null);

  // ── Fetch functions ──────────────────────────────────────────────

  const fetchTasks = useCallback(async () => {
    try {
      const res = await listTasks();
      setTasks(res.tasks || []);
      setError(null);
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to load tasks'); }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (levelFilter !== 'all') params.set('level', levelFilter);
      if (pathFilter.trim()) params.set('path', pathFilter.trim());
      params.set('limit', String(logLimit));
      const res = await fetch(`${API_BASE}/logs?${params}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      setLogData(await res.json());
      setError(null);
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to load logs'); }
  }, [levelFilter, pathFilter, logLimit]);

  const fetchJournal = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/journal?limit=200`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setJournal(data.events || data.entries || []);
      setError(null);
    } catch (e) { setError(e instanceof Error ? e.message : 'Failed to load journal'); }
  }, []);

  // ── Auto refresh ─────────────────────────────────────────────────

  const fetchCurrent = useCallback(() => {
    if (tab === 'tasks') fetchTasks();
    else if (tab === 'logs') fetchLogs();
    else fetchJournal();
  }, [tab, fetchTasks, fetchLogs, fetchJournal]);

  useEffect(() => { fetchCurrent(); }, [fetchCurrent]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchCurrent, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchCurrent]);

  // ── Task actions ─────────────────────────────────────────────────

  const handleCancelTask = async (id: string) => {
    try {
      await cancelTask(id);
      // Also clear sessionStorage training ref if it matches
      if (sessionStorage.getItem('task_retrain') === id) {
        sessionStorage.removeItem('task_retrain');
      }
      fetchTasks();
    } catch { /* ignore */ }
  };

  const handleClearLogs = async () => {
    try {
      await fetch(`${API_BASE}/logs`, { method: 'DELETE' });
      fetchLogs();
    } catch { /* ignore */ }
  };

  // ── Computed ─────────────────────────────────────────────────────

  const runningCount = tasks.filter(t => t.status === 'Running').length;
  const failedCount = tasks.filter(t => t.status === 'Failed').length;
  const filteredTasks = taskFilter === 'all' ? tasks : tasks.filter(t => t.status === taskFilter);

  // ── Render ───────────────────────────────────────────────────────

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[#f8fafc]">🔍 事件中心</h1>
        <div className="flex items-center gap-3">
          {runningCount > 0 && (
            <span className="px-2 py-1 rounded text-xs bg-blue-500/20 text-blue-300 animate-pulse">
              {runningCount} 运行中
            </span>
          )}
          {failedCount > 0 && (
            <span className="px-2 py-1 rounded text-xs bg-red-500/20 text-red-300">
              {failedCount} 失败
            </span>
          )}
          <label className="flex items-center gap-1.5 text-sm text-[#94a3b8] cursor-pointer">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} className="rounded" />
            自动刷新
          </label>
          <button onClick={fetchCurrent} className="px-3 py-1.5 rounded bg-[#334155] text-[#f8fafc] text-sm hover:bg-[#475569]">
            🔄 刷新
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 rounded-lg bg-[#1e293b] border border-[#334155]">
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              tab === t.key ? 'bg-[#3b82f6] text-white' : 'text-[#94a3b8] hover:text-[#f8fafc] hover:bg-[#334155]'
            }`}
          >
            {t.icon} {t.label}
            {t.key === 'tasks' && runningCount > 0 && (
              <span className="ml-1.5 px-1.5 py-0.5 rounded-full text-xs bg-blue-500/30">{runningCount}</span>
            )}
          </button>
        ))}
      </div>

      {error && <div className="text-red-400 text-sm p-2 rounded bg-red-500/10 border border-red-500/20">⚠️ {error}</div>}

      {/* ── Tasks Tab ─────────────────────────────────────────── */}
      {tab === 'tasks' && (
        <div className="space-y-3">
          {/* Task status filter */}
          <div className="flex items-center gap-2">
            {['all', 'Running', 'Pending', 'Completed', 'Failed'].map(f => (
              <button
                key={f}
                onClick={() => setTaskFilter(f)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  taskFilter === f ? 'bg-[#3b82f6] text-white' : 'bg-[#334155] text-[#94a3b8] hover:text-[#f8fafc]'
                }`}
              >
                {f === 'all' ? '全部' : f}
              </button>
            ))}
            <span className="text-sm text-[#64748b] ml-auto">{filteredTasks.length} 条</span>
          </div>

          {/* Task list */}
          <div className="rounded-lg border border-[#334155] bg-[#0f172a] divide-y divide-[#1e293b] max-h-[calc(100vh-340px)] overflow-y-auto">
            {filteredTasks.length === 0 ? (
              <div className="p-8 text-center text-[#64748b]">暂无任务</div>
            ) : filteredTasks.map(task => {
              const isExpanded = expandedTask === task.id;
              return (
                <div key={task.id} className={`px-4 py-3 hover:bg-[#1e293b]/50 transition-colors ${
                  task.status === 'Failed' ? 'border-l-2 border-l-red-500' :
                  task.status === 'Running' ? 'border-l-2 border-l-blue-500' : ''
                }`}>
                  <div className="flex items-center gap-3 cursor-pointer" onClick={() => setExpandedTask(isExpanded ? null : task.id)}>
                    {/* Status badge */}
                    <span className={`px-2 py-0.5 rounded text-xs font-bold shrink-0 ${STATUS_STYLE[task.status] || 'bg-gray-500/20 text-gray-300'}`}>
                      {task.status}
                    </span>
                    {/* Type */}
                    <span className="text-sm text-[#e2e8f0] font-medium">{task.task_type}</span>
                    {/* Progress */}
                    {task.progress && (
                      <span className="text-xs text-[#94a3b8] truncate max-w-[200px]">{task.progress}</span>
                    )}
                    <span className="flex-1" />
                    {/* Duration */}
                    <span className="text-xs text-[#64748b] font-mono">{elapsed(task.created_at, task.updated_at)}</span>
                    {/* Time */}
                    <span className="text-xs text-[#64748b] font-mono shrink-0">{fmtTime(task.created_at)}</span>
                    {/* Actions */}
                    {(task.status === 'Running' || task.status === 'Pending') && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleCancelTask(task.id); }}
                        className="px-2 py-1 rounded text-xs bg-red-500/20 text-red-300 hover:bg-red-500/30"
                      >
                        ✕ 取消
                      </button>
                    )}
                    {(task.status === 'Completed' || task.status === 'Failed') && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleCancelTask(task.id); }}
                        className="px-2 py-1 rounded text-xs bg-[#334155] text-[#94a3b8] hover:text-[#f8fafc]"
                        title="删除记录"
                      >
                        🗑
                      </button>
                    )}
                  </div>

                  {/* Expanded detail */}
                  {isExpanded && (
                    <div className="mt-2 ml-1 space-y-1.5 text-xs">
                      <div className="text-[#94a3b8]"><span className="text-[#64748b]">ID:</span> <span className="font-mono">{task.id}</span></div>
                      <div className="text-[#94a3b8]"><span className="text-[#64748b]">创建:</span> {fmtDateTime(task.created_at)}</div>
                      <div className="text-[#94a3b8]"><span className="text-[#64748b]">更新:</span> {fmtDateTime(task.updated_at)}</div>
                      {task.result && (
                        <pre className="p-2 rounded bg-green-500/5 border border-green-500/20 text-green-300 font-mono whitespace-pre-wrap break-all max-h-60 overflow-y-auto">
                          {task.result}
                        </pre>
                      )}
                      {task.error && (
                        <pre className="p-2 rounded bg-red-500/5 border border-red-500/20 text-red-300 font-mono whitespace-pre-wrap break-all max-h-60 overflow-y-auto">
                          {task.error}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── API Logs Tab ──────────────────────────────────────── */}
      {tab === 'logs' && (
        <div className="space-y-3">
          {/* Summary + filters */}
          {logData && (
            <>
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-2.5">
                  <div className="text-xl font-bold text-blue-400">{logData.summary.info}</div>
                  <div className="text-xs text-blue-300/70">INFO</div>
                </div>
                <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/5 p-2.5">
                  <div className="text-xl font-bold text-yellow-400">{logData.summary.warn}</div>
                  <div className="text-xs text-yellow-300/70">WARN</div>
                </div>
                <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-2.5">
                  <div className="text-xl font-bold text-red-400">{logData.summary.error}</div>
                  <div className="text-xs text-red-300/70">ERROR</div>
                </div>
              </div>

              <div className="flex items-center gap-3 p-3 rounded-lg bg-[#1e293b] border border-[#334155]">
                <div className="flex gap-1">
                  {['all', 'error', 'warn', 'info'].map(lvl => (
                    <button key={lvl} onClick={() => setLevelFilter(lvl)}
                      className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                        levelFilter === lvl ? 'bg-[#3b82f6] text-white' : 'bg-[#334155] text-[#94a3b8] hover:text-[#f8fafc]'
                      }`}
                    >
                      {lvl === 'all' ? '全部' : lvl.toUpperCase()}
                    </button>
                  ))}
                </div>
                <input type="text" placeholder="路径筛选" value={pathFilter} onChange={e => setPathFilter(e.target.value)}
                  className="flex-1 bg-[#0f172a] border border-[#334155] rounded px-3 py-1.5 text-sm text-[#f8fafc] placeholder-[#64748b]" />
                <select value={logLimit} onChange={e => setLogLimit(Number(e.target.value))}
                  className="bg-[#0f172a] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#f8fafc]">
                  <option value={50}>50条</option>
                  <option value={200}>200条</option>
                  <option value={500}>500条</option>
                  <option value={2000}>全部</option>
                </select>
                <button onClick={handleClearLogs} className="px-3 py-1.5 rounded bg-red-500/20 text-red-300 text-sm hover:bg-red-500/30">
                  🗑 清除
                </button>
              </div>
            </>
          )}

          {/* Log entries */}
          <div className="rounded-lg border border-[#334155] bg-[#0f172a] divide-y divide-[#1e293b] max-h-[calc(100vh-440px)] overflow-y-auto">
            {!logData || logData.entries.length === 0 ? (
              <div className="p-8 text-center text-[#64748b]">暂无日志</div>
            ) : logData.entries.map(entry => {
              const isExpanded = expandedLog === entry.id;
              return (
                <div key={entry.id}
                  className={`px-4 py-2 hover:bg-[#1e293b]/50 cursor-pointer transition-colors ${
                    entry.level === 'error' ? 'border-l-2 border-l-red-500' :
                    entry.level === 'warn' ? 'border-l-2 border-l-yellow-500' : ''
                  }`}
                  onClick={() => setExpandedLog(isExpanded ? null : entry.id)}
                >
                  <div className="flex items-center gap-3 text-sm">
                    <span className="text-[#64748b] font-mono text-xs w-20 shrink-0">{fmtTime(entry.timestamp)}</span>
                    <span className={`px-1.5 py-0.5 rounded text-xs font-bold shrink-0 ${LEVEL_BADGE[entry.level] || ''}`}>
                      {entry.level.toUpperCase()}
                    </span>
                    <span className={`font-mono text-xs shrink-0 w-12 ${
                      entry.method === 'GET' ? 'text-green-400' : entry.method === 'POST' ? 'text-blue-400' :
                      entry.method === 'DELETE' ? 'text-red-400' : 'text-[#94a3b8]'
                    }`}>{entry.method}</span>
                    <span className="font-mono text-[#e2e8f0] text-xs truncate flex-1 min-w-0">{entry.path}</span>
                    <span className={`font-mono text-xs shrink-0 ${
                      entry.status >= 500 ? 'text-red-400' : entry.status >= 400 ? 'text-yellow-400' :
                      entry.status > 0 ? 'text-green-400' : 'text-[#64748b]'
                    }`}>{entry.status > 0 ? entry.status : '—'}</span>
                    <span className={`font-mono text-xs shrink-0 w-16 text-right ${
                      entry.duration_ms > 5000 ? 'text-red-400' : entry.duration_ms > 1000 ? 'text-yellow-400' : 'text-[#64748b]'
                    }`}>{entry.duration_ms}ms</span>
                  </div>
                  {isExpanded && (
                    <div className="mt-2 space-y-1">
                      <div className="text-xs text-[#94a3b8]">{fmtDateTime(entry.timestamp)}</div>
                      <div className={`text-xs ${entry.level === 'error' ? 'text-red-400' : entry.level === 'warn' ? 'text-yellow-400' : 'text-blue-400'}`}>
                        {entry.message}
                      </div>
                      {entry.detail && (
                        <pre className="mt-1 p-2 rounded text-xs font-mono whitespace-pre-wrap break-all border border-[#334155] bg-[#1e293b] max-h-60 overflow-y-auto">
                          {entry.detail}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Journal Tab ───────────────────────────────────────── */}
      {tab === 'journal' && (
        <div className="space-y-3">
          <div className="rounded-lg border border-[#334155] bg-[#0f172a] divide-y divide-[#1e293b] max-h-[calc(100vh-300px)] overflow-y-auto">
            {journal.length === 0 ? (
              <div className="p-8 text-center text-[#64748b]">暂无交易事件</div>
            ) : journal.map((evt, i) => (
              <div key={evt.id || i} className="px-4 py-2.5 hover:bg-[#1e293b]/50">
                <div className="flex items-center gap-3 text-sm">
                  <span className="text-[#64748b] font-mono text-xs w-20 shrink-0">{fmtTime(evt.timestamp)}</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-bold shrink-0 ${
                    evt.event_type === 'trade' || evt.event_type === 'order_filled' ? 'bg-purple-500/20 text-purple-300' :
                    evt.event_type === 'signal' ? 'bg-cyan-500/20 text-cyan-300' :
                    'bg-gray-500/20 text-gray-300'
                  }`}>
                    {evt.event_type}
                  </span>
                  <span className="text-[#e2e8f0] font-mono text-xs font-medium">{evt.symbol}</span>
                  <span className={`text-xs font-bold ${evt.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                    {evt.side?.toUpperCase()}
                  </span>
                  <span className="text-xs text-[#94a3b8]">×{evt.quantity}</span>
                  <span className="text-xs text-[#94a3b8]">@{evt.price}</span>
                  <span className="flex-1" />
                  {evt.pnl != null && (
                    <span className={`text-xs font-mono font-bold ${evt.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {evt.pnl >= 0 ? '+' : ''}{evt.pnl.toFixed(2)}
                    </span>
                  )}
                </div>
                {evt.detail && (
                  <div className="mt-1 text-xs text-[#64748b] ml-24">{evt.detail}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
