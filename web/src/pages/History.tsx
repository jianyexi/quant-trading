import { useState, useEffect, useCallback } from 'react';
import { listTasks, cancelTask, type TaskRecord } from '../api/client';

// ── Constants ─────────────────────────────────────────────────────

const TASK_TYPE_LABELS: Record<string, string> = {
  factor_mine_parametric: '参数化因子挖掘',
  factor_mine_gp: 'GP因子挖掘',
  factor_registry_manage: '因子注册管理',
  factor_export_promoted: '因子导出',
  ml_retrain: 'ML模型训练',
  backtest: '策略回测',
};

const STATUS_COLORS: Record<string, string> = {
  Running: '#3b82f6',
  Completed: '#22c55e',
  Failed: '#ef4444',
  Pending: '#f59e0b',
};

const STATUS_LABELS: Record<string, string> = {
  Running: '运行中',
  Completed: '已完成',
  Failed: '失败',
  Pending: '等待中',
};

// ── Helpers ───────────────────────────────────────────────────────

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString('zh-CN', {
      month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch { return iso; }
}

function duration(created: string, updated: string): string {
  try {
    const ms = new Date(updated).getTime() - new Date(created).getTime();
    if (ms < 0) return '-';
    if (ms < 1000) return `${ms}ms`;
    const s = Math.round(ms / 1000);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    return `${m}m${s % 60}s`;
  } catch { return '-'; }
}

function parseParams(raw: string | null): Record<string, unknown> | null {
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function parseResult(raw: string | null): unknown {
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return raw; }
}

// Pretty-print parameter key
function paramLabel(key: string): string {
  const map: Record<string, string> = {
    strategy: '策略', symbols: '股票', symbol: '股票',
    start: '开始', end: '结束', start_date: '开始日期', end_date: '结束日期',
    capital: '资金', period: '周期', inference_mode: '推理模式',
    n_bars: 'K线数', horizon: '预测周期', ic_threshold: 'IC阈值',
    top_n: 'Top-N', retrain: '重训练', cross_stock: '跨股票',
    data_source: '数据源', algorithms: '算法', threshold: '阈值',
    pop_size: '种群大小', generations: '代数', max_depth: '最大深度',
  };
  return map[key] || key;
}

// ── Component ─────────────────────────────────────────────────────

export default function History() {
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<string>('');
  const [filterStatus, setFilterStatus] = useState<string>('');

  const fetchTasks = useCallback(async () => {
    try {
      const opts: { task_type?: string; status?: string } = {};
      if (filterType) opts.task_type = filterType;
      if (filterStatus) opts.status = filterStatus;
      const data = await listTasks(opts);
      setTasks(data.tasks || []);
    } catch (e) {
      console.error('Failed to load tasks:', e);
    } finally {
      setLoading(false);
    }
  }, [filterType, filterStatus]);

  useEffect(() => { fetchTasks(); }, [fetchTasks]);

  // Auto-refresh running tasks
  useEffect(() => {
    const hasRunning = tasks.some(t => t.status === 'Running');
    if (!hasRunning) return;
    const timer = setInterval(fetchTasks, 3000);
    return () => clearInterval(timer);
  }, [tasks, fetchTasks]);

  const handleCancel = async (id: string) => {
    try {
      await cancelTask(id);
      fetchTasks();
    } catch (e) {
      console.error('Cancel failed:', e);
    }
  };

  const toggle = (id: string) => setExpandedId(prev => prev === id ? null : id);

  // ── Styles ────────────────────────────────────────────────────
  const container: React.CSSProperties = {
    padding: '24px', maxWidth: 1200, margin: '0 auto',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  };
  const card: React.CSSProperties = {
    background: '#1e1e2e', borderRadius: 8, padding: 20,
    marginBottom: 12, cursor: 'pointer', border: '1px solid #2a2a3e',
    transition: 'border-color 0.2s',
  };
  const filterBar: React.CSSProperties = {
    display: 'flex', gap: 12, marginBottom: 20, alignItems: 'center', flexWrap: 'wrap',
  };
  const select: React.CSSProperties = {
    background: '#1e1e2e', color: '#e0e0e0', border: '1px solid #3a3a4e',
    borderRadius: 6, padding: '6px 12px', fontSize: 14,
  };
  const badge = (color: string): React.CSSProperties => ({
    background: color + '22', color, padding: '2px 10px',
    borderRadius: 12, fontSize: 12, fontWeight: 600, whiteSpace: 'nowrap',
  });
  const paramGrid: React.CSSProperties = {
    display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
    gap: '8px 16px', marginTop: 8,
  };
  const paramItem: React.CSSProperties = {
    display: 'flex', justifyContent: 'space-between', padding: '4px 8px',
    background: '#252538', borderRadius: 4, fontSize: 13,
  };
  const resultBox: React.CSSProperties = {
    background: '#0d0d1a', borderRadius: 6, padding: 12, marginTop: 8,
    maxHeight: 400, overflow: 'auto', fontSize: 12, fontFamily: 'monospace',
    whiteSpace: 'pre-wrap', wordBreak: 'break-all',
  };

  return (
    <div style={container}>
      <h2 style={{ color: '#e0e0e0', marginBottom: 4 }}>任务历史</h2>
      <p style={{ color: '#888', fontSize: 14, marginBottom: 16 }}>
        查看因子挖掘、模型训练、策略回测的完整运行记录
      </p>

      {/* Filter bar */}
      <div style={filterBar}>
        <select style={select} value={filterType} onChange={e => setFilterType(e.target.value)}>
          <option value="">全部类型</option>
          {Object.entries(TASK_TYPE_LABELS).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
        <select style={select} value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
          <option value="">全部状态</option>
          <option value="completed">已完成</option>
          <option value="failed">失败</option>
          <option value="running">运行中</option>
        </select>
        <button
          onClick={fetchTasks}
          style={{
            ...select, cursor: 'pointer', background: '#3b82f6', color: '#fff',
            border: 'none', padding: '6px 16px',
          }}
        >
          刷新
        </button>
        <span style={{ color: '#666', fontSize: 13, marginLeft: 'auto' }}>
          共 {tasks.length} 条记录
        </span>
      </div>

      {loading && <p style={{ color: '#888' }}>加载中...</p>}

      {!loading && tasks.length === 0 && (
        <div style={{ ...card, textAlign: 'center', color: '#666', cursor: 'default' }}>
          暂无任务记录。通过流水线页面运行因子挖掘、训练或回测后，记录将显示在此处。
        </div>
      )}

      {tasks.map(task => {
        const expanded = expandedId === task.id;
        const params = parseParams(task.parameters);
        const result = expanded ? parseResult(task.result) : null;
        const isRunning = task.status === 'Running';

        return (
          <div
            key={task.id}
            style={{ ...card, borderColor: expanded ? '#3b82f6' : '#2a2a3e' }}
            onClick={() => toggle(task.id)}
          >
            {/* Header row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <span style={badge(STATUS_COLORS[task.status] || '#888')}>
                {STATUS_LABELS[task.status] || task.status}
              </span>
              <span style={{ color: '#c0c0d0', fontWeight: 600, fontSize: 15 }}>
                {TASK_TYPE_LABELS[task.task_type] || task.task_type}
              </span>

              {/* Quick param summary */}
              {params && (
                <span style={{ color: '#777', fontSize: 12 }}>
                  {(() => {
                    const parts: string[] = [];
                    const syms = params.symbols || params.symbol;
                    if (syms) {
                      const s = Array.isArray(syms) ? (syms as string[]).join(',') : String(syms);
                      if (s) parts.push(s);
                    }
                    if (params.strategy) parts.push(String(params.strategy));
                    if (params.algorithms) parts.push(String(params.algorithms));
                    const sd = params.start || params.start_date;
                    const ed = params.end || params.end_date;
                    if (sd && ed) parts.push(`${sd}~${ed}`);
                    return parts.join(' | ');
                  })()}
                </span>
              )}

              <span style={{ marginLeft: 'auto', color: '#666', fontSize: 12 }}>
                {formatTime(task.created_at)}
                {task.status !== 'Running' && (
                  <span style={{ marginLeft: 8 }}>
                    耗时 {duration(task.created_at, task.updated_at)}
                  </span>
                )}
              </span>

              {isRunning && (
                <button
                  onClick={e => { e.stopPropagation(); handleCancel(task.id); }}
                  style={{
                    background: '#ef4444', color: '#fff', border: 'none',
                    borderRadius: 4, padding: '3px 10px', fontSize: 12, cursor: 'pointer',
                  }}
                >
                  取消
                </button>
              )}

              <span style={{ color: '#555', fontSize: 14, transform: expanded ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }}>
                ▶
              </span>
            </div>

            {/* Progress / error line */}
            {task.progress && !expanded && (
              <div style={{ color: '#888', fontSize: 12, marginTop: 4, marginLeft: 4 }}>
                {task.progress}
              </div>
            )}
            {task.error && !expanded && (
              <div style={{ color: '#ef4444', fontSize: 12, marginTop: 4 }}>
                {task.error}
              </div>
            )}

            {/* Expanded details */}
            {expanded ? (
              <div style={{ marginTop: 12 }} onClick={e => e.stopPropagation()}>
                {/* Parameters */}
                {params ? (
                  <div>
                    <h4 style={{ color: '#aaa', fontSize: 13, margin: '0 0 4px 0' }}>输入参数</h4>
                    <div style={paramGrid}>
                      {Object.entries(params).map(([k, v]) => (
                        <div key={k} style={paramItem}>
                          <span style={{ color: '#888' }}>{paramLabel(k)}</span>
                          <span style={{ color: '#c0c0d0', fontWeight: 500 }}>
                            {typeof v === 'boolean' ? (v ? '是' : '否') :
                             Array.isArray(v) ? (v as string[]).join(', ') :
                             String(v ?? '-')}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                {/* Progress / logs */}
                {task.progress ? (
                  <div style={{ marginTop: 12 }}>
                    <h4 style={{ color: '#aaa', fontSize: 13, margin: '0 0 4px 0' }}>进度日志</h4>
                    <div style={{ ...resultBox, maxHeight: 120, color: '#93c5fd' }}>
                      {task.progress}
                    </div>
                  </div>
                ) : null}

                {/* Error */}
                {task.error ? (
                  <div style={{ marginTop: 12 }}>
                    <h4 style={{ color: '#ef4444', fontSize: 13, margin: '0 0 4px 0' }}>错误信息</h4>
                    <div style={{ ...resultBox, maxHeight: 150, color: '#fca5a5' }}>
                      {task.error}
                    </div>
                  </div>
                ) : null}

                {/* Result */}
                {result ? (
                  <div style={{ marginTop: 12 }}>
                    <h4 style={{ color: '#aaa', fontSize: 13, margin: '0 0 4px 0' }}>运行结果</h4>
                    {renderResult(task.task_type, result)}
                  </div>
                ) : null}

                {/* Task ID */}
                <div style={{ marginTop: 12, color: '#555', fontSize: 11 }}>
                  Task ID: {task.id}
                </div>
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

// ── Result Renderers ──────────────────────────────────────────────

function renderResult(taskType: string, result: unknown) {
  const box: React.CSSProperties = {
    background: '#0d0d1a', borderRadius: 6, padding: 12, marginTop: 4,
    maxHeight: 500, overflow: 'auto', fontSize: 12, fontFamily: 'monospace',
    whiteSpace: 'pre-wrap', wordBreak: 'break-all', color: '#a0d0a0',
  };

  if (taskType === 'backtest') {
    return renderBacktestResult(result, box);
  }

  // Generic JSON display
  const text = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
  return <div style={box}>{text}</div>;
}

function renderBacktestResult(result: unknown, boxStyle: React.CSSProperties) {
  if (!result || typeof result !== 'object') {
    return <div style={boxStyle}>{String(result)}</div>;
  }
  const r = result as Record<string, unknown>;

  // Multi-symbol results
  if (r.symbol_results && Array.isArray(r.symbol_results)) {
    const srs = r.symbol_results as Record<string, unknown>[];
    return (
      <div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #333' }}>
              {['股票', '策略', '总收益', '最大回撤', '夏普', '交易数'].map(h => (
                <th key={h} style={{ padding: '6px 8px', color: '#888', fontWeight: 500, textAlign: 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {srs.map((sr, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #222' }}>
                <td style={{ padding: '6px 8px', color: '#c0c0d0' }}>{String(sr.symbol || '')}</td>
                <td style={{ padding: '6px 8px', color: '#888' }}>{String(sr.strategy || '')}</td>
                <td style={{ padding: '6px 8px', color: Number(sr.total_return) >= 0 ? '#22c55e' : '#ef4444' }}>
                  {(Number(sr.total_return) * 100).toFixed(2)}%
                </td>
                <td style={{ padding: '6px 8px', color: '#f59e0b' }}>
                  {(Number(sr.max_drawdown) * 100).toFixed(2)}%
                </td>
                <td style={{ padding: '6px 8px', color: '#93c5fd' }}>
                  {Number(sr.sharpe_ratio).toFixed(3)}
                </td>
                <td style={{ padding: '6px 8px', color: '#888' }}>{String(sr.total_trades || 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // Single-symbol result
  const metrics = [
    ['总收益', r.total_return, (v: number) => `${(v * 100).toFixed(2)}%`, Number(r.total_return) >= 0 ? '#22c55e' : '#ef4444'],
    ['最大回撤', r.max_drawdown, (v: number) => `${(v * 100).toFixed(2)}%`, '#f59e0b'],
    ['夏普比率', r.sharpe_ratio, (v: number) => v.toFixed(3), '#93c5fd'],
    ['年化收益', r.annual_return, (v: number) => `${(v * 100).toFixed(2)}%`, '#c084fc'],
    ['交易次数', r.total_trades, (v: number) => String(v), '#888'],
    ['胜率', r.win_rate, (v: number) => `${(v * 100).toFixed(1)}%`, '#22c55e'],
  ] as [string, unknown, (v: number) => string, string][];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 8 }}>
      {metrics.filter(([, v]) => v !== undefined && v !== null).map(([label, val, fmt, color]) => (
        <div key={label} style={{ background: '#151525', borderRadius: 6, padding: '8px 12px' }}>
          <div style={{ color: '#666', fontSize: 11 }}>{label}</div>
          <div style={{ color, fontSize: 16, fontWeight: 600 }}>{fmt(Number(val))}</div>
        </div>
      ))}
    </div>
  );
}
