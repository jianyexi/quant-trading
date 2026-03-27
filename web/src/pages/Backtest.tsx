import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
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
  X,
  Plus,
  Download,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  LineChart,
  Line,
  Legend,
} from 'recharts';
import { runBacktest, getBacktestResults } from '../api/client';

interface BacktestConfig {
  strategy: string;
  symbol: string;
  start: string;
  end: string;
  capital: number;
  period: string;
  inference_mode: string;
  extraSymbols: string[];
  benchmark_symbol: string;
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
  actual_start?: string;
  actual_end?: string;
  initial_capital: number;
  final_value: number;
  total_return_percent: number;
  annual_return_percent?: number;
  sharpe_ratio: number;
  sortino_ratio?: number;
  calmar_ratio?: number;
  max_drawdown_percent: number;
  max_drawdown_duration_days?: number;
  win_rate_percent: number;
  total_trades: number;
  winning_trades?: number;
  losing_trades?: number;
  profit_factor: number;
  avg_win?: number;
  avg_loss?: number;
  avg_holding_days?: number;
  total_commission?: number;
  turnover_rate?: number;
  equity_curve?: EquityPoint[];
  trades?: TradeRecord[];
  data_source?: string;
  status: string;
  // Multi-stock fields
  is_multi?: boolean;
  symbols?: string[];
  per_symbol_results?: PerSymbolResult[];
  per_symbol_curves?: { symbol: string; data: EquityPoint[] }[];
  failed_symbols?: { symbol: string; error: string }[];
  // Benchmark fields
  benchmark_symbol?: string;
  benchmark_curve?: EquityPoint[];
  alpha?: number;
  beta?: number;
  information_ratio?: number;
  tracking_error?: number;
}

interface PerSymbolResult {
  symbol: string;
  initial_capital: number;
  final_value: number;
  total_return_percent: number;
  annual_return_percent?: number;
  sharpe_ratio: number;
  max_drawdown_percent: number;
  total_trades: number;
  win_rate_percent: number;
  profit_factor?: number;
  data_bars?: number;
  actual_start?: string;
  actual_end?: string;
  status: string;
  error?: string;
}

const STRATEGIES = [
  { value: 'sma_cross', label: 'SMA 交叉 (5/20)' },
  { value: 'rsi_reversal', label: 'RSI 均值回归 (14)' },
  { value: 'macd_trend', label: 'MACD 动量 (12/26/9)' },
  { value: 'multi_factor', label: '多因子模型' },
  { value: 'ml_factor', label: 'ML因子模型' },
];

const MULTI_COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#a855f7', '#06b6d4', '#f97316', '#ec4899'];

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
  extraSymbols: [],
  benchmark_symbol: '',
};

export default function Backtest() {
  const saved = useMemo(() => loadSavedConfig(), []);
  const [config, setConfig] = useState<BacktestConfig>({ ...defaultConfig, ...saved });
  const [configOpen, setConfigOpen] = useState(true);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResultData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollerRef.current) {
      clearInterval(pollerRef.current);
      pollerRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  // Auto-save config to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('quant-backtest-config', JSON.stringify(config));
  }, [config]);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    setProgress('⏳ Submitting backtest...');
    setResult(null);
    stopPolling();
    try {
      const { task_id } = await runBacktest({
        strategy: config.strategy,
        symbol: config.symbol,
        start: config.start,
        end: config.end,
        capital: config.capital,
        period: config.period,
        inference_mode: config.inference_mode,
        symbols: config.extraSymbols.length > 0 ? config.extraSymbols : undefined,
        benchmark_symbol: config.benchmark_symbol || undefined,
      });
      setTaskId(task_id);

      // Poll for progress
      const poll = async () => {
        try {
          const data = await getBacktestResults(task_id) as Record<string, unknown>;
          const status = data.status as string;
          if (status === 'completed' || status === 'Completed') {
            stopPolling();
            setResult(data as unknown as BacktestResultData);
            setProgress(null);
            setRunning(false);
            setConfigOpen(false);
          } else if (status === 'Failed' || status === 'failed') {
            stopPolling();
            setError((data.error as string) ?? 'Backtest failed');
            setProgress(null);
            setRunning(false);
          } else {
            setProgress((data.progress as string) ?? '⏳ Running...');
          }
        } catch {
          // transient error, keep polling
        }
      };

      // First poll after short delay, then every 1s
      setTimeout(() => void poll(), 500);
      pollerRef.current = setInterval(() => void poll(), 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : '回测提交失败');
      setProgress(null);
      setRunning(false);
    }
  };

  const updateConfig = (key: keyof BacktestConfig, value: string | number) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const downloadCsv = async (url: string, body: Record<string, unknown>, filename: string) => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const equityCurve = result?.equity_curve ?? [];
  const trades = result?.trades ?? [];

  const metrics = result
    ? [
        {
          label: '总收益率',
          value: `${(result.total_return_percent ?? 0) >= 0 ? '+' : ''}${(result.total_return_percent ?? 0).toFixed(2)}%`,
          color: (result.total_return_percent ?? 0) >= 0 ? '#22c55e' : '#ef4444',
          icon: (result.total_return_percent ?? 0) >= 0 ? TrendingUp : TrendingDown,
        },
        {
          label: '夏普比率',
          value: (result.sharpe_ratio ?? 0).toFixed(2),
          color: (result.sharpe_ratio ?? 0) >= 1 ? '#22c55e' : '#eab308',
          icon: BarChart3,
        },
        {
          label: 'Sortino比率',
          value: (result.sortino_ratio ?? 0).toFixed(2),
          color: (result.sortino_ratio ?? 0) >= 1.5 ? '#22c55e' : '#eab308',
          icon: BarChart3,
        },
        {
          label: '最大回撤',
          value: `-${(result.max_drawdown_percent ?? 0).toFixed(2)}%`,
          color: '#ef4444',
          icon: TrendingDown,
        },
        {
          label: 'Calmar比率',
          value: (result.calmar_ratio ?? 0).toFixed(2),
          color: (result.calmar_ratio ?? 0) >= 1 ? '#22c55e' : '#eab308',
          icon: BarChart3,
        },
        {
          label: '胜率',
          value: `${(result.win_rate_percent ?? 0).toFixed(1)}%`,
          color: (result.win_rate_percent ?? 0) >= 50 ? '#22c55e' : '#eab308',
          icon: Target,
        },
        {
          label: '交易次数',
          value: (result.total_trades ?? 0).toString(),
          color: '#3b82f6',
          icon: Activity,
        },
        {
          label: '盈亏比',
          value: result.profit_factor == null || result.profit_factor === Infinity ? '∞' : (result.profit_factor ?? 0).toFixed(2),
          color: (result.profit_factor ?? 0) >= 1 ? '#22c55e' : '#ef4444',
          icon: DollarSign,
        },
        {
          label: '平均持仓天数',
          value: (result.avg_holding_days ?? 0).toFixed(1),
          color: '#8b5cf6',
          icon: Activity,
        },
        {
          label: '总手续费',
          value: `¥${(result.total_commission ?? 0).toFixed(0)}`,
          color: '#f97316',
          icon: DollarSign,
        },
        ...(result.alpha != null
          ? [
              {
                label: 'Alpha',
                value: `${result.alpha >= 0 ? '+' : ''}${result.alpha.toFixed(2)}%`,
                color: result.alpha >= 0 ? '#22c55e' : '#ef4444',
                icon: TrendingUp,
              },
            ]
          : []),
        ...(result.beta != null
          ? [
              {
                label: 'Beta',
                value: result.beta.toFixed(2),
                color: Math.abs(result.beta) <= 1 ? '#3b82f6' : '#eab308',
                icon: BarChart3,
              },
            ]
          : []),
        ...(result.information_ratio != null
          ? [
              {
                label: '信息比率',
                value: result.information_ratio.toFixed(2),
                color: result.information_ratio >= 0.5 ? '#22c55e' : '#eab308',
                icon: Target,
              },
            ]
          : []),
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
              <div className="lg:col-span-2">
                <label className="block text-sm text-[#94a3b8] mb-1.5">股票代码 {config.extraSymbols.length > 0 && <span className="text-[#3b82f6]">（组合回测: {1 + config.extraSymbols.length}只）</span>}</label>
                <div className="flex gap-2">
                  <input type="text" value={config.symbol}
                    onChange={(e) => updateConfig('symbol', e.target.value)}
                    placeholder="主股票: 600519.SH"
                    className="flex-1 bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]" />
                  <button onClick={() => {
                    setConfig(prev => ({ ...prev, extraSymbols: [...prev.extraSymbols, ''] }));
                  }}
                    className="flex items-center gap-1 px-3 py-2 bg-[#334155] hover:bg-[#475569] text-[#94a3b8] hover:text-[#f8fafc] rounded-lg text-xs transition-colors cursor-pointer"
                    title="添加更多股票进行组合回测">
                    <Plus size={14} /> 多股
                  </button>
                </div>
                {/* Extra symbol inputs */}
                {config.extraSymbols.map((sym, idx) => (
                  <div key={idx} className="flex gap-2 mt-1.5">
                    <input type="text" value={sym}
                      onChange={(e) => {
                        const updated = [...config.extraSymbols];
                        updated[idx] = e.target.value;
                        setConfig(prev => ({ ...prev, extraSymbols: updated }));
                      }}
                      placeholder={`股票 ${idx + 2}: 如 000001.SZ`}
                      className="flex-1 bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6]" />
                    <button onClick={() => {
                      setConfig(prev => ({ ...prev, extraSymbols: prev.extraSymbols.filter((_, i) => i !== idx) }));
                    }}
                      className="px-2 py-2 text-[#94a3b8] hover:text-red-400 transition-colors cursor-pointer">
                      <X size={14} />
                    </button>
                  </div>
                ))}
                <div className="flex gap-1 mt-1 flex-wrap">
                  {[
                    { sym: '600519.SH', label: '茅台' },
                    { sym: 'AAPL', label: 'Apple' },
                    { sym: 'MSFT', label: 'MSFT' },
                    { sym: '0700.HK', label: '腾讯' },
                    { sym: '9988.HK', label: '阿里' },
                    { sym: '000001.SZ', label: '平安' },
                    { sym: '300750.SZ', label: '宁德' },
                    { sym: '601318.SH', label: '中国平安' },
                  ].map((s) => (
                    <button key={s.sym} onClick={() => {
                      if (config.symbol === s.sym || config.extraSymbols.includes(s.sym)) return;
                      if (!config.symbol) {
                        updateConfig('symbol', s.sym);
                      } else {
                        setConfig(prev => ({ ...prev, extraSymbols: [...prev.extraSymbols, s.sym] }));
                      }
                    }}
                      className={`text-xs px-1.5 py-0.5 rounded transition-colors cursor-pointer ${
                        config.symbol === s.sym || config.extraSymbols.includes(s.sym)
                          ? 'bg-[#3b82f6]/20 text-[#3b82f6] border border-[#3b82f6]/30'
                          : 'bg-[#334155] text-[#94a3b8] hover:bg-[#475569] hover:text-[#f8fafc]'
                      }`}>
                      {s.label}
                    </button>
                  ))}
                </div>
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
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">基准指数 (可选)</label>
                <input type="text" value={config.benchmark_symbol}
                  placeholder="如 000300 (沪深300)"
                  onChange={(e) => updateConfig('benchmark_symbol', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6] placeholder-[#475569]" />
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
                    <option value="onnx">🔮 ONNX Runtime (~0.05ms, WSL)</option>
                    <option value="tcp_mq">🔗 TCP消息队列 (~0.3ms)</option>
                    <option value="http">🌐 HTTP sidecar (~2-5ms)</option>
                  </select>
                </div>
              )}
            </div>

            {error && (
              <div className="text-sm text-[#ef4444] bg-[#ef4444]/10 border border-[#ef4444]/20 rounded-lg px-4 py-2">{error}</div>
            )}

            {running && progress && (
              <div className="flex items-center gap-3 text-sm text-[#60a5fa] bg-[#3b82f6]/10 border border-[#3b82f6]/20 rounded-lg px-4 py-2.5">
                <Loader2 size={16} className="animate-spin flex-shrink-0" />
                <span>{progress}</span>
              </div>
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
              <div className="flex items-center gap-3 flex-wrap">
                <span className="text-[#94a3b8]">
                  {result.strategy} · {result.symbol}
                </span>
                <span className="text-[#f8fafc] font-mono text-xs">
                  {result.actual_start || result.start} ~ {result.actual_end || result.end}
                </span>
                {result.actual_start && result.actual_end && (result.actual_start !== result.start || result.actual_end !== result.end) && (
                  <span className="px-2 py-0.5 rounded text-xs bg-yellow-500/15 text-yellow-400" title={`请求: ${result.start} ~ ${result.end}`}>
                    ⚠️ 实际数据范围与请求不同
                  </span>
                )}
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

          {/* Export Buttons */}
          {taskId && (
            <div className="flex items-center gap-3 flex-wrap">
              <button
                onClick={() => downloadCsv('/api/export/backtest', { task_id: taskId }, 'backtest_equity.csv')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[#334155] text-[#94a3b8] hover:bg-[#475569] hover:text-white transition-colors"
              >
                <Download size={14} />
                导出权益曲线 CSV
              </button>
              <button
                onClick={() => downloadCsv('/api/export/trades', {}, 'trades.csv')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[#334155] text-[#94a3b8] hover:bg-[#475569] hover:text-white transition-colors"
              >
                <Download size={14} />
                导出交易记录 CSV
              </button>
              <button
                onClick={() => downloadCsv('/api/export/metrics', { task_id: taskId }, 'metrics.csv')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[#334155] text-[#94a3b8] hover:bg-[#475569] hover:text-white transition-colors"
              >
                <Download size={14} />
                导出绩效指标 CSV
              </button>
            </div>
          )}

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

          {/* Per-Symbol Comparison (multi-stock only) */}
          {result.is_multi && result.per_symbol_results && result.per_symbol_results.length > 0 && (
            <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
              <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📊 个股对比</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-[#334155]">
                      {['股票', '初始资金', '最终净值', '收益率', '年化', '夏普', '最大回撤', '交易次数', '胜率', '数据'].map((h) => (
                        <th key={h} className="text-left py-3 px-3 text-xs font-medium text-[#94a3b8]">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.per_symbol_results.map((sr: PerSymbolResult) => (
                      <tr key={sr.symbol} className="border-b border-[#334155]/50 hover:bg-[#334155]/30">
                        <td className="py-2 px-3 text-[#3b82f6] font-mono font-medium">{sr.symbol}</td>
                        <td className="py-2 px-3 text-[#f8fafc] font-mono">¥{Math.round(sr.initial_capital ?? 0).toLocaleString()}</td>
                        <td className="py-2 px-3 text-[#f8fafc] font-mono">¥{(sr.final_value ?? 0).toLocaleString(undefined, {maximumFractionDigits: 2})}</td>
                        <td className={`py-2 px-3 font-mono font-bold ${(sr.total_return_percent ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {(sr.total_return_percent ?? 0) >= 0 ? '+' : ''}{(sr.total_return_percent ?? 0).toFixed(2)}%
                        </td>
                        <td className={`py-2 px-3 font-mono ${(sr.annual_return_percent ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {(sr.annual_return_percent ?? 0).toFixed(2)}%
                        </td>
                        <td className={`py-2 px-3 font-mono ${(sr.sharpe_ratio ?? 0) >= 0.5 ? 'text-green-400' : (sr.sharpe_ratio ?? 0) >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {(sr.sharpe_ratio ?? 0).toFixed(2)}
                        </td>
                        <td className="py-2 px-3 text-red-400 font-mono">-{(sr.max_drawdown_percent ?? 0).toFixed(2)}%</td>
                        <td className="py-2 px-3 text-[#f8fafc] font-mono">{sr.total_trades ?? 0}</td>
                        <td className={`py-2 px-3 font-mono ${(sr.win_rate_percent ?? 0) >= 50 ? 'text-green-400' : 'text-[#94a3b8]'}`}>
                          {(sr.win_rate_percent ?? 0).toFixed(1)}%
                        </td>
                        <td className="py-2 px-3 text-[#94a3b8] text-xs">{sr.data_bars ?? 0}条</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {result.failed_symbols && result.failed_symbols.length > 0 && (
                <div className="mt-3 text-xs text-red-400">
                  ⚠️ 失败: {result.failed_symbols.map(f => `${f.symbol} (${f.error})`).join(', ')}
                </div>
              )}
            </div>
          )}

          {/* Multi-stock Equity Curves */}
          {result.is_multi && result.per_symbol_curves && result.per_symbol_curves.length > 0 && (
            <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
              <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📈 个股净值对比</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={(() => {
                  // Merge all curves by date for recharts
                  const dateMap: Record<string, Record<string, number>> = {};
                  for (const curve of (result.per_symbol_curves ?? [])) {
                    for (const pt of curve.data) {
                      if (!dateMap[pt.date]) dateMap[pt.date] = {};
                      dateMap[pt.date][curve.symbol] = pt.value;
                    }
                  }
                  return Object.entries(dateMap).sort(([a], [b]) => a.localeCompare(b)).map(([date, vals]) => ({ date, ...vals }));
                })()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 10 }} tickLine={{ stroke: '#334155' }} axisLine={{ stroke: '#334155' }}
                    interval={Math.floor((result.per_symbol_curves?.[0]?.data?.length ?? 100) / 6)} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickLine={{ stroke: '#334155' }} axisLine={{ stroke: '#334155' }}
                    tickFormatter={(v: number) => `¥${(v / 1000).toFixed(0)}k`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                    formatter={(value: number | undefined, name: string | undefined) => [`¥${(value ?? 0).toLocaleString()}`, name ?? '']}
                    labelStyle={{ color: '#94a3b8' }} />
                  <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} />
                  {(result.per_symbol_curves ?? []).map((curve, i) => (
                    <Line key={curve.symbol} type="monotone" dataKey={curve.symbol}
                      stroke={MULTI_COLORS[i % MULTI_COLORS.length]} strokeWidth={1.5} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Combined Equity Curve (single or multi portfolio total) */}
          {equityCurve.length > 0 && (
            <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
              <h3 className="text-base font-semibold text-[#f8fafc] mb-4">
                {result.is_multi ? '📈 组合净值曲线' : '📈 净值曲线'}
                {result.benchmark_symbol && result.benchmark_curve && (
                  <span className="text-sm font-normal text-[#f97316] ml-2">vs {result.benchmark_symbol}</span>
                )}
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={(() => {
                  if (!result.benchmark_curve || result.benchmark_curve.length === 0) return equityCurve;
                  const benchMap = new Map(result.benchmark_curve.map(p => [p.date, p.value]));
                  return equityCurve.map(p => ({ ...p, benchmark: benchMap.get(p.date) ?? undefined }));
                })()}>
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
                    formatter={(value: number | undefined, name: string) => [
                      `¥${(value ?? 0).toLocaleString()}`,
                      name === 'benchmark' ? `基准 (${result.benchmark_symbol})` : '策略净值',
                    ]}
                    labelStyle={{ color: '#94a3b8' }} />
                  {result.benchmark_curve && result.benchmark_curve.length > 0 && (
                    <Legend formatter={(v: string) => v === 'benchmark' ? `基准 (${result.benchmark_symbol})` : '策略净值'} />
                  )}
                  <Area type="monotone" dataKey="value" name="value" stroke="#3b82f6" strokeWidth={2} fill="url(#equityGradient)" />
                  {result.benchmark_curve && result.benchmark_curve.length > 0 && (
                    <Area type="monotone" dataKey="benchmark" name="benchmark" stroke="#f97316" strokeWidth={2} strokeDasharray="6 3" fill="none" />
                  )}
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
