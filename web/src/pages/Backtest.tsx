import { useState, useMemo, useEffect } from 'react';
import {
  ChevronDown,
  ChevronUp,
  Play,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Target,
  Activity,
  DollarSign,
  Loader2,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { runBacktest } from '../api/client';

interface BacktestConfig {
  strategy: string;
  symbol: string;
  start: string;
  end: string;
  capital: number;
  period: string;
  inference_mode: string;
}

interface EquityPoint {
  date: string;
  value: number;
}

interface TradeRecord {
  date: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  commission: number;
}

interface BacktestResultData {
  id: string;
  strategy: string;
  symbol: string;
  start: string;
  end: string;
  initial_capital: number;
  final_value: number;
  total_return_percent: number;
  annual_return_percent?: number;
  sharpe_ratio: number;
  max_drawdown_percent: number;
  max_drawdown_duration_days?: number;
  win_rate_percent: number;
  total_trades: number;
  winning_trades?: number;
  losing_trades?: number;
  profit_factor: number;
  avg_win?: number;
  avg_loss?: number;
  equity_curve?: EquityPoint[];
  trades?: TradeRecord[];
  data_source?: string;
  status: string;
}

const STRATEGIES = [
  { value: 'sma_cross', label: 'SMA 交叉 (5/20)' },
  { value: 'rsi_reversal', label: 'RSI 均值回归 (14)' },
  { value: 'macd_trend', label: 'MACD 动量 (12/26/9)' },
  { value: 'multi_factor', label: '多因子模型' },
  { value: 'ml_factor', label: 'ML因子模型' },
];

const PERIODS = [
  { value: 'daily', label: '日线' },
  { value: '60', label: '60分钟' },
  { value: '30', label: '30分钟' },
  { value: '15', label: '15分钟' },
  { value: '5', label: '5分钟' },
  { value: '1', label: '1分钟' },
];

function loadSavedConfig(): Partial<BacktestConfig> {
  try {
    // Try backtest-specific key first, then fall back to strategy config
    const raw = localStorage.getItem('quant-backtest-config')
      ?? localStorage.getItem('quant-strategy-config');
    if (raw) return JSON.parse(raw) as Partial<BacktestConfig>;
  } catch {
    // ignore
  }
  return {};
}

const defaultConfig: BacktestConfig = {
  strategy: 'sma_cross',
  symbol: '600519.SH',
  start: '2024-01-01',
  end: '2024-12-31',
  capital: 1000000,
  period: 'daily',
  inference_mode: 'embedded',
};

export default function Backtest() {
  const saved = useMemo(() => loadSavedConfig(), []);
  const [config, setConfig] = useState<BacktestConfig>({ ...defaultConfig, ...saved });
  const [configOpen, setConfigOpen] = useState(true);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BacktestResultData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Auto-save config to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('quant-backtest-config', JSON.stringify(config));
  }, [config]);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = (await runBacktest({
        strategy: config.strategy,
        symbol: config.symbol,
        start: config.start,
        end: config.end,
        capital: config.capital,
        period: config.period,
        inference_mode: config.inference_mode,
      })) as BacktestResultData;
      setResult(res);
      setConfigOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : '回测失败');
    } finally {
      setRunning(false);
    }
  };

  const updateConfig = (key: keyof BacktestConfig, value: string | number) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const equityCurve = result?.equity_curve ?? [];
  const trades = result?.trades ?? [];

  const metrics = result
    ? [
        {
          label: '总收益率',
          value: `${result.total_return_percent >= 0 ? '+' : ''}${result.total_return_percent.toFixed(2)}%`,
          color: result.total_return_percent >= 0 ? '#22c55e' : '#ef4444',
          icon: result.total_return_percent >= 0 ? TrendingUp : TrendingDown,
        },
        {
          label: '夏普比率',
          value: result.sharpe_ratio.toFixed(2),
          color: result.sharpe_ratio >= 1 ? '#22c55e' : '#eab308',
          icon: BarChart3,
        },
        {
          label: '最大回撤',
          value: `-${result.max_drawdown_percent.toFixed(2)}%`,
          color: '#ef4444',
          icon: TrendingDown,
        },
        {
          label: '胜率',
          value: `${result.win_rate_percent.toFixed(1)}%`,
          color: result.win_rate_percent >= 50 ? '#22c55e' : '#eab308',
          icon: Target,
        },
        {
          label: '交易次数',
          value: result.total_trades.toString(),
          color: '#3b82f6',
          icon: Activity,
        },
        {
          label: '盈亏比',
          value: result.profit_factor === Infinity ? '∞' : result.profit_factor.toFixed(2),
          color: result.profit_factor >= 1 ? '#22c55e' : '#ef4444',
          icon: DollarSign,
        },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Config Form */}
      <div className="bg-[#1e293b] rounded-xl border border-[#334155]">
        <button
          onClick={() => setConfigOpen(!configOpen)}
          className="w-full flex items-center justify-between px-6 py-4 cursor-pointer"
        >
          <h2 className="text-lg font-semibold text-[#f8fafc]">📊 回测配置</h2>
          {configOpen ? (
            <ChevronUp size={20} className="text-[#94a3b8]" />
          ) : (
            <ChevronDown size={20} className="text-[#94a3b8]" />
          )}
        </button>

        {configOpen && (
          <div className="px-6 pb-6 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">策略</label>
                <select
                  value={config.strategy}
                  onChange={(e) => updateConfig('strategy', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]"
                >
                  {STRATEGIES.map((s) => (
                    <option key={s.value} value={s.value}>{s.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">股票代码</label>
                <input type="text" value={config.symbol}
                  onChange={(e) => updateConfig('symbol', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]" />
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">开始日期</label>
                <input type="date" value={config.start}
                  onChange={(e) => updateConfig('start', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]" />
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">结束日期</label>
                <input type="date" value={config.end}
                  onChange={(e) => updateConfig('end', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]" />
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">初始资金 (¥)</label>
                <input type="number" value={config.capital} min={1000} step={10000}
                  onChange={(e) => updateConfig('capital', Number(e.target.value))}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]" />
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">K线周期</label>
                <select
                  value={config.period}
                  onChange={(e) => updateConfig('period', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]"
                >
                  {PERIODS.map((p) => (
                    <option key={p.value} value={p.value}>{p.label}</option>
                  ))}
                </select>
              </div>
              {config.strategy === 'ml_factor' && (
                <div>
                  <label className="block text-sm text-[#94a3b8] mb-1.5">ML推理模式</label>
                  <select
                    value={config.inference_mode}
                    onChange={(e) => updateConfig('inference_mode', e.target.value)}
                    className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]"
                  >
                    <option value="embedded">🦀 内嵌推理 (Rust, ~0.01ms)</option>
                    <option value="tcp_mq">🔗 TCP消息队列 (~0.3ms)</option>
                    <option value="http">🌐 HTTP sidecar (~2-5ms)</option>
                  </select>
                </div>
              )}
            </div>

            {error && (
              <div className="text-sm text-[#ef4444] bg-[#ef4444]/10 border border-[#ef4444]/20 rounded-lg px-4 py-2">{error}</div>
            )}

            <button onClick={() => void handleRun()} disabled={running}
              className="flex items-center gap-2 px-6 py-2.5 bg-[#3b82f6] hover:bg-[#2563eb] text-white rounded-lg text-sm font-medium disabled:opacity-50 transition-colors cursor-pointer">
              {running ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
              {running ? '运行中…' : '开始回测'}
            </button>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Summary */}
          <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-4">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-3">
                <span className="text-[#94a3b8]">
                  {result.strategy} · {result.symbol} · {result.start} ~ {result.end}
                </span>
                {result.data_source != null && (
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    result.data_source.startsWith('akshare') ? 'bg-green-500/15 text-green-400' : 'bg-yellow-500/15 text-yellow-400'
                  }`}>
                    {result.data_source.startsWith('akshare') ? '📡 真实数据' : '🔬 模拟数据'}
                  </span>
                )}
              </div>
              <span className={`font-bold ${result.total_return_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ¥{result.initial_capital.toLocaleString()} → ¥{result.final_value.toLocaleString()}
              </span>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {metrics.map((m) => {
              const Icon = m.icon;
              return (
                <div key={m.label} className="bg-[#1e293b] rounded-xl border border-[#334155] p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Icon size={16} style={{ color: m.color }} />
                    <span className="text-xs text-[#94a3b8]">{m.label}</span>
                  </div>
                  <p className="text-xl font-bold" style={{ color: m.color }}>{m.value}</p>
                </div>
              );
            })}
          </div>

          {/* Extended Metrics */}
          {(result.winning_trades != null || result.avg_win != null) && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {result.annual_return_percent != null && (
                <MiniCard label="年化收益" value={`${result.annual_return_percent.toFixed(2)}%`}
                  color={result.annual_return_percent >= 0 ? '#22c55e' : '#ef4444'} />
              )}
              {result.winning_trades != null && (
                <MiniCard label="盈利次数" value={String(result.winning_trades)} color="#22c55e" />
              )}
              {result.losing_trades != null && (
                <MiniCard label="亏损次数" value={String(result.losing_trades)} color="#ef4444" />
              )}
              {result.avg_win != null && result.avg_win > 0 && (
                <MiniCard label="平均盈利" value={`¥${result.avg_win.toFixed(0)}`} color="#22c55e" />
              )}
              {result.avg_loss != null && result.avg_loss > 0 && (
                <MiniCard label="平均亏损" value={`¥${result.avg_loss.toFixed(0)}`} color="#ef4444" />
              )}
              {result.max_drawdown_duration_days != null && (
                <MiniCard label="最大回撤持续" value={`${result.max_drawdown_duration_days}天`} color="#eab308" />
              )}
            </div>
          )}

          {/* Equity Curve */}
          {equityCurve.length > 0 && (
            <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
              <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📈 净值曲线</h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={equityCurve}>
                  <defs>
                    <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    tickLine={{ stroke: '#334155' }} axisLine={{ stroke: '#334155' }}
                    interval={Math.floor(equityCurve.length / 6)} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }}
                    tickLine={{ stroke: '#334155' }} axisLine={{ stroke: '#334155' }}
                    tickFormatter={(v: number) => `¥${(v / 1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                    formatter={(value: number | undefined) => [`¥${(value ?? 0).toLocaleString()}`, '组合净值']}
                    labelStyle={{ color: '#94a3b8' }} />
                  <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} fill="url(#equityGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Trade List */}
          {trades.length > 0 && (
            <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
              <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📋 交易记录 ({trades.length}笔)</h3>
              <div className="overflow-x-auto max-h-96 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-[#1e293b]">
                    <tr className="border-b border-[#334155]">
                      {['时间', '代码', '方向', '价格', '数量', '手续费'].map((h) => (
                        <th key={h} className="text-left py-3 px-4 text-xs font-medium text-[#94a3b8]">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {trades.map((t, i) => (
                      <tr key={i} className="border-b border-[#334155]/50 hover:bg-[#334155]/30">
                        <td className="py-2 px-4 text-[#94a3b8] font-mono text-xs">{t.date}</td>
                        <td className="py-2 px-4 text-[#3b82f6] font-mono">{t.symbol}</td>
                        <td className="py-2 px-4">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                            t.side === 'BUY' ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'
                          }`}>{t.side === 'BUY' ? '买入' : '卖出'}</span>
                        </td>
                        <td className="py-2 px-4 text-[#f8fafc] font-mono">¥{t.price.toFixed(2)}</td>
                        <td className="py-2 px-4 text-[#f8fafc] font-mono">{t.quantity.toLocaleString()}</td>
                        <td className="py-2 px-4 text-[#94a3b8] font-mono">¥{t.commission.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function MiniCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-[#1e293b] rounded-lg border border-[#334155] p-3">
      <p className="text-xs text-[#94a3b8]">{label}</p>
      <p className="text-sm font-bold mt-0.5" style={{ color }}>{value}</p>
    </div>
  );
}
