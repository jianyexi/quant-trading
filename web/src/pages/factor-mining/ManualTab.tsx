import { useState, useEffect, useCallback } from 'react';
import {
  factorEvaluateManual,
  factorSaveManual,
  type ManualFactorResult,
  type ManualFactorMetrics,
} from '../../api/client';
import { useTaskManager } from '../../hooks/useTaskManager';
import { TaskOutput } from '../../components/TaskPipeline';
import DataSourceConfig from './DataSourceConfig';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';

/* ── Templates ──────────────────────────────────────────────────────── */

interface Template { name: string; expr: string }
interface TemplateCategory { category: string; icon: string; items: Template[] }

const TEMPLATES: TemplateCategory[] = [
  { category: '动量', icon: '🚀', items: [
    { name: '5日动量', expr: 'pct_change(close, 5)' },
    { name: '20日动量', expr: 'pct_change(close, 20)' },
    { name: '风险调整动量', expr: 'pct_change(close, 20) / rolling_std(pct_change(close, 1), 20)' },
  ]},
  { category: '反转', icon: '🔄', items: [
    { name: '5日反转', expr: '-pct_change(close, 5)' },
    { name: '均值回归', expr: '-(close - rolling_mean(close, 20)) / rolling_std(close, 20)' },
  ]},
  { category: '波动率', icon: '📈', items: [
    { name: '波动率压缩', expr: 'rolling_std(pct_change(close, 1), 5) / rolling_std(pct_change(close, 1), 20)' },
    { name: 'ATR归一化', expr: 'rolling_mean(high - low, 14) / close' },
  ]},
  { category: '量价', icon: '📊', items: [
    { name: '量比', expr: 'volume / rolling_mean(volume, 20)' },
    { name: '量价背离', expr: '-pct_change(close, 10) * (volume / rolling_mean(volume, 20))' },
    { name: '成交量zscore', expr: '(volume - rolling_mean(volume, 20)) / rolling_std(volume, 20)' },
  ]},
  { category: '形态', icon: '🕯️', items: [
    { name: '实体比', expr: 'abs(close - open) / (high - low)' },
    { name: '上影线比', expr: '(high - max(open, close)) / (high - low)' },
    { name: 'MA交叉', expr: 'rolling_mean(close, 5) / rolling_mean(close, 20) - 1' },
  ]},
];

/* ── Experiment History ─────────────────────────────────────────────── */

interface ExperimentEntry {
  name: string;
  expression: string;
  metrics: ManualFactorMetrics;
  timestamp: string;
}

function loadHistory(): ExperimentEntry[] {
  try {
    return JSON.parse(localStorage.getItem('factor_experiments') || '[]');
  } catch { return []; }
}

function saveHistory(entries: ExperimentEntry[]) {
  localStorage.setItem('factor_experiments', JSON.stringify(entries.slice(0, 50)));
}

/* ── Rating ─────────────────────────────────────────────────────────── */

function RatingStars({ rating }: { rating: number }) {
  const labels = ['', '较弱', '一般', '良好', '优秀', '卓越'];
  return (
    <span className="text-yellow-400">
      {'⭐'.repeat(rating)}
      <span className="text-xs text-[#94a3b8] ml-1">{labels[rating] || ''}</span>
    </span>
  );
}

/* ── Main Component ─────────────────────────────────────────────────── */

export default function ManualTab() {
  const [name, setName] = useState('my_factor');
  const [expression, setExpression] = useState('');
  const [symbols, setSymbols] = useState('600519,300750,601318');
  const [startDate, setStartDate] = useState('2022-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [horizon, setHorizon] = useState(5);
  const [result, setResult] = useState<ManualFactorResult | null>(null);
  const [history, setHistory] = useState<ExperimentEntry[]>(loadHistory);
  const [showTemplates, setShowTemplates] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');

  const tm = useTaskManager('task_manual_factor');

  // Parse task output when completed
  useEffect(() => {
    if (tm.output) {
      try {
        // tm.output may be wrapped in {stdout, stderr} from run_python_script
        let parsed = JSON.parse(tm.output);
        if (parsed.stdout) {
          parsed = JSON.parse(parsed.stdout);
        }
        if (parsed.error) {
          setResult(null);
        } else {
          setResult(parsed as ManualFactorResult);
          // Add to history
          if (parsed.metrics) {
            const entry: ExperimentEntry = {
              name: parsed.name || name,
              expression: parsed.expression || expression,
              metrics: parsed.metrics,
              timestamp: new Date().toISOString(),
            };
            const updated = [entry, ...history.filter(h => h.expression !== entry.expression)].slice(0, 50);
            setHistory(updated);
            saveHistory(updated);
          }
        }
      } catch {
        // not JSON, ignore
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tm.output]);

  const handleRun = useCallback(() => {
    if (!expression.trim()) return;
    setResult(null);
    setSaveMsg('');
    tm.submit(() => factorEvaluateManual({
      expression: expression.trim(),
      name,
      symbols: symbols || undefined,
      start_date: startDate,
      end_date: endDate,
      horizon,
    }));
  }, [expression, name, symbols, startDate, endDate, horizon, tm]);

  const handleSave = useCallback(async () => {
    if (!result) return;
    try {
      const res = await factorSaveManual({
        name: result.name || name,
        expression: result.expression || expression,
        metrics: result.metrics,
      });
      setSaveMsg(`✅ 已保存: ${res.factor_id}`);
    } catch (e) {
      setSaveMsg(`❌ 保存失败: ${e}`);
    }
  }, [result, name, expression]);

  const applyTemplate = (t: Template) => {
    setName(t.name);
    setExpression(t.expr);
    setShowTemplates(false);
  };

  const loadExperiment = (entry: ExperimentEntry) => {
    setName(entry.name);
    setExpression(entry.expression);
  };

  return (
    <div className="space-y-6">
      {/* ── Factor Definition ──────────────────────────────────────── */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h3 className="text-base font-bold text-[#f8fafc] mb-1">✏️ 因子定义</h3>
        <p className="text-xs text-[#94a3b8] mb-4">
          输入因子表达式，支持: close, open, high, low, volume + pct_change, rolling_mean, rolling_std, shift, abs, log, sqrt, rank, ema 等
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">因子名称</label>
            <input
              className="w-full rounded bg-[#0f172a] border border-[#334155] text-[#f8fafc] px-3 py-2 text-sm"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="my_momentum_factor"
            />
          </div>
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">前瞻期 (bar)</label>
            <input
              type="number"
              className="w-full rounded bg-[#0f172a] border border-[#334155] text-[#f8fafc] px-3 py-2 text-sm"
              value={horizon}
              onChange={e => setHorizon(Number(e.target.value))}
              min={1} max={60}
            />
          </div>
        </div>

        <div className="mb-4">
          <label className="text-xs text-[#94a3b8] block mb-1">表达式</label>
          <textarea
            className="w-full rounded bg-[#0f172a] border border-[#334155] text-[#f8fafc] px-3 py-2 text-sm font-mono h-20 resize-y"
            value={expression}
            onChange={e => setExpression(e.target.value)}
            placeholder="pct_change(close, 20) / rolling_std(pct_change(close, 1), 20)"
          />
        </div>

        {/* Templates */}
        <div className="mb-4">
          <button
            className="text-xs text-[#3b82f6] hover:text-[#60a5fa] cursor-pointer"
            onClick={() => setShowTemplates(!showTemplates)}
          >
            {showTemplates ? '▼ 收起模板' : '▶ 选择预设模板'}
          </button>
          {showTemplates && (
            <div className="mt-2 grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {TEMPLATES.map(cat => (
                <div key={cat.category} className="rounded-lg bg-[#0f172a] border border-[#334155] p-3">
                  <div className="text-xs font-bold text-[#f8fafc] mb-2">{cat.icon} {cat.category}</div>
                  {cat.items.map(t => (
                    <button
                      key={t.name}
                      onClick={() => applyTemplate(t)}
                      className="block w-full text-left text-xs text-[#94a3b8] hover:text-[#f8fafc] hover:bg-[#1e293b] rounded px-2 py-1 mb-1 cursor-pointer"
                    >
                      {t.name}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Data Config */}
        <DataSourceConfig
          symbols={symbols} setSymbols={setSymbols}
          startDate={startDate} setStartDate={setStartDate}
          endDate={endDate} setEndDate={setEndDate}
        />

        <div className="mt-4">
          <button
            onClick={handleRun}
            disabled={!expression.trim() || tm.running}
            className="px-6 py-2 rounded-lg font-medium text-sm bg-[#3b82f6] hover:bg-[#2563eb] text-white disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            {tm.running ? '⏳ 检验中...' : '▶ 运行检验'}
          </button>
        </div>
      </div>

      {/* ── Task Output ────────────────────────────────────────────── */}
      <TaskOutput
        running={tm.running}
        error={tm.error}
        output={tm.output}
      />

      {/* ── Results ────────────────────────────────────────────────── */}
      {result && result.metrics && (
        <div className="space-y-4">
          {/* Metric Cards */}
          <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-bold text-[#f8fafc]">📊 检验结果</h3>
              <RatingStars rating={result.metrics.rating || 0} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              <MetricCard label="IC均值" value={result.metrics.ic_mean} fmt={v => v.toFixed(4)} good={Math.abs(v(result.metrics.ic_mean)) >= 0.03} />
              <MetricCard label="IR" value={result.metrics.ir} fmt={v => v.toFixed(2)} good={Math.abs(v(result.metrics.ir)) >= 0.5} />
              <MetricCard label="IC正率" value={result.metrics.ic_pos_rate} fmt={v => (v * 100).toFixed(0) + '%'} good={v(result.metrics.ic_pos_rate) >= 0.55} />
              <MetricCard label="换手率" value={result.metrics.turnover} fmt={v => v.toFixed(3)} />
              <MetricCard label="衰减率" value={result.metrics.decay} fmt={v => v.toFixed(2)} />
              <MetricCard label="样本数" value={result.metrics.n_obs} fmt={v => v.toLocaleString()} />
            </div>
          </div>

          {/* IC Time Series */}
          {result.ic_series.length > 0 && (
            <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
              <h3 className="text-sm font-bold text-[#f8fafc] mb-3">📉 月度 Rank IC</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={result.ic_series}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="period" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', fontSize: 12 }}
                    labelStyle={{ color: '#f8fafc' }}
                  />
                  <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                  <Bar dataKey="ic" radius={[2, 2, 0, 0]}>
                    {result.ic_series.map((entry, i) => (
                      <Cell key={i} fill={entry.ic >= 0 ? '#22c55e' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Quintile Returns */}
          {result.quintile_returns.length > 0 && (
            <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
              <h3 className="text-sm font-bold text-[#f8fafc] mb-3">📶 五分位组平均收益</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={result.quintile_returns}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="quintile" tick={{ fontSize: 10, fill: '#94a3b8' }}
                    tickFormatter={(q: number) => `Q${q}`} />
                  <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }}
                    tickFormatter={(v: number) => (v * 100).toFixed(1) + '%'} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', fontSize: 12 }}
                    formatter={(val: number | undefined) => [(((val ?? 0) * 100).toFixed(2) + '%'), '平均收益']}
                  />
                  <Bar dataKey="avg_return" radius={[4, 4, 0, 0]}>
                    {result.quintile_returns.map((entry, i) => (
                      <Cell key={i} fill={entry.avg_return >= 0 ? '#22c55e' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-[#64748b] mt-2">
                Q1=因子值最低组, Q5=因子值最高组。单调递增说明因子有预测力。
              </p>
            </div>
          )}

          {/* Per-stock IC */}
          {Object.keys(result.per_stock_ic).length > 1 && (
            <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
              <h3 className="text-sm font-bold text-[#f8fafc] mb-3">📋 个股 IC</h3>
              <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                {Object.entries(result.per_stock_ic).map(([sym, ic]) => (
                  <div key={sym} className="bg-[#0f172a] rounded px-2 py-1 text-center">
                    <div className="text-xs text-[#94a3b8]">{sym}</div>
                    <div className={`text-sm font-mono ${ic >= 0.03 ? 'text-green-400' : ic >= 0 ? 'text-[#f8fafc]' : 'text-red-400'}`}>
                      {ic.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 flex-wrap">
            <button
              onClick={handleSave}
              className="px-4 py-2 rounded-lg text-sm bg-[#22c55e] hover:bg-[#16a34a] text-white cursor-pointer"
            >
              💾 保存到注册表
            </button>
            <button
              onClick={() => navigator.clipboard.writeText(expression)}
              className="px-4 py-2 rounded-lg text-sm bg-[#334155] hover:bg-[#475569] text-[#f8fafc] cursor-pointer"
            >
              📋 复制表达式
            </button>
            {saveMsg && <span className="text-sm text-[#94a3b8] self-center">{saveMsg}</span>}
          </div>
        </div>
      )}

      {/* ── Experiment History ──────────────────────────────────────── */}
      {history.length > 0 && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h3 className="text-sm font-bold text-[#f8fafc] mb-3">🕐 实验历史</h3>
          <div className="space-y-2">
            {history.map((entry, i) => (
              <div
                key={i}
                className="flex items-center justify-between bg-[#0f172a] rounded-lg px-3 py-2 text-sm"
              >
                <div className="flex-1 min-w-0">
                  <span className="text-[#f8fafc] font-medium">{entry.name}</span>
                  <span className="text-[#64748b] ml-2 font-mono text-xs truncate">{entry.expression}</span>
                </div>
                <div className="flex items-center gap-3 ml-3 shrink-0">
                  <span className={`font-mono text-xs ${Math.abs(entry.metrics.ic_mean) >= 0.03 ? 'text-green-400' : 'text-[#94a3b8]'}`}>
                    IC={entry.metrics.ic_mean.toFixed(4)}
                  </span>
                  <span className="font-mono text-xs text-[#94a3b8]">
                    IR={entry.metrics.ir.toFixed(2)}
                  </span>
                  <RatingStars rating={entry.metrics.rating || 0} />
                  <button
                    onClick={() => loadExperiment(entry)}
                    className="text-xs text-[#3b82f6] hover:text-[#60a5fa] cursor-pointer"
                  >
                    加载
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Helper Components ──────────────────────────────────────────────── */

function v(n: number): number { return n; }

function MetricCard({ label, value, fmt, good }: {
  label: string;
  value: number;
  fmt: (v: number) => string;
  good?: boolean;
}) {
  return (
    <div className="bg-[#0f172a] rounded-lg p-3 text-center">
      <div className="text-xs text-[#64748b] mb-1">{label}</div>
      <div className={`text-lg font-bold font-mono ${good === true ? 'text-green-400' : good === false ? 'text-red-400' : 'text-[#f8fafc]'}`}>
        {fmt(value)}
      </div>
    </div>
  );
}
