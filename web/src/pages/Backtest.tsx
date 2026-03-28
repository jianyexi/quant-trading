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
  Grid3X3,
  Layers,
  CheckSquare,
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
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import { runBacktest, getBacktestResults, runOptimization, getBacktestHistory, compareBacktestRuns, runMonteCarloSimulation, getTaskResult, runAttribution } from '../api/client';
import type { BacktestRunSummary, CompareRunData, MonteCarloResultData, AttributionResultData } from '../api/client';

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

interface DrawdownPoint {
  date: string;
  drawdown: number;
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
  drawdown_curve?: DrawdownPoint[];
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

interface OptGridEntry {
  param1: number;
  param2: number;
  total_return: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  skipped?: boolean;
  error?: string;
}

interface OptimizeResultData {
  status: string;
  grid: OptGridEntry[];
  best: { param1: number; param2: number; total_return: number; sharpe: number } | null;
  param1_name: string;
  param2_name: string;
  strategy: string;
  symbol: string;
  total_combinations: number;
  completed_combinations: number;
}

const STRATEGIES = [
  { value: 'sma_cross', label: 'SMA 交叉 (5/20)' },
  { value: 'rsi_reversal', label: 'RSI 均值回归 (14)' },
  { value: 'macd_trend', label: 'MACD 动量 (12/26/9)' },
  { value: 'bollinger_bands', label: '布林带策略' },
  { value: 'multi_factor', label: '多因子模型' },
  { value: 'ml_factor', label: 'ML因子模型' },
  { value: 'composite', label: '策略组合' },
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

  // ── Optimization state ────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<'backtest' | 'optimize' | 'compare'>('backtest');
  const [optParam1Name, setOptParam1Name] = useState('fast_period');
  const [optParam1Values, setOptParam1Values] = useState('3,5,8,10,13,15,20');
  const [optParam2Name, setOptParam2Name] = useState('slow_period');
  const [optParam2Values, setOptParam2Values] = useState('10,15,20,25,30,40,50,60');
  const [optRunning, setOptRunning] = useState(false);
  const [optProgress, setOptProgress] = useState<string | null>(null);
  const [optError, setOptError] = useState<string | null>(null);
  const [optResult, setOptResult] = useState<OptimizeResultData | null>(null);
  const [optMetric, setOptMetric] = useState<'sharpe' | 'total_return' | 'max_drawdown' | 'win_rate'>('sharpe');
  const optPollerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Comparison state ──────────────────────────────────────────────
  const [compareHistory, setCompareHistory] = useState<BacktestRunSummary[]>([]);
  const [compareSelected, setCompareSelected] = useState<Set<string>>(new Set());
  const [compareResult, setCompareResult] = useState<CompareRunData[] | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  // ── Monte Carlo state ─────────────────────────────────────────────
  const [mcRunning, setMcRunning] = useState(false);
  const [mcResult, setMcResult] = useState<MonteCarloResultData | null>(null);
  const [mcError, setMcError] = useState<string | null>(null);
  const mcPollerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Attribution state ───────────────────────────────────────────
  const [attrRunning, setAttrRunning] = useState(false);
  const [attrResult, setAttrResult] = useState<AttributionResultData | null>(null);
  const [attrError, setAttrError] = useState<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollerRef.current) {
      clearInterval(pollerRef.current);
      pollerRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const stopOptPolling = useCallback(() => {
    if (optPollerRef.current) {
      clearInterval(optPollerRef.current);
      optPollerRef.current = null;
    }
  }, []);

  useEffect(() => stopOptPolling, [stopOptPolling]);

  const stopMcPolling = useCallback(() => {
    if (mcPollerRef.current) {
      clearInterval(mcPollerRef.current);
      mcPollerRef.current = null;
    }
  }, []);

  useEffect(() => stopMcPolling, [stopMcPolling]);

  const handleRunMonteCarlo = async () => {
    if (!taskId) return;
    setMcRunning(true);
    setMcResult(null);
    setMcError(null);
    try {
      const { task_id: mcTaskId } = await runMonteCarloSimulation({
        task_id: taskId,
        num_simulations: 1000,
        num_days: 252,
      });
      mcPollerRef.current = setInterval(async () => {
        try {
          const res = await getTaskResult(mcTaskId);
          if (res.status === 'completed' && res.result) {
            stopMcPolling();
            setMcResult(JSON.parse(res.result));
            setMcRunning(false);
          } else if (res.status === 'failed') {
            stopMcPolling();
            setMcError(res.error ?? '模拟失败');
            setMcRunning(false);
          }
        } catch {
          stopMcPolling();
          setMcError('轮询失败');
          setMcRunning(false);
        }
      }, 1000);
    } catch (e) {
      setMcError(e instanceof Error ? e.message : '启动失败');
      setMcRunning(false);
    }
  };

  const handleRunAttribution = async () => {
    if (!taskId) return;
    setAttrRunning(true);
    setAttrResult(null);
    setAttrError(null);
    try {
      const res = await runAttribution({ task_id: taskId });
      setAttrResult(res);
    } catch (e) {
      setAttrError(e instanceof Error ? e.message : '归因分析失败');
    } finally {
      setAttrRunning(false);
    }
  };

  // Auto-save config to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('quant-backtest-config', JSON.stringify(config));
  }, [config]);

  // Load backtest history when comparison tab is activated
  useEffect(() => {
    if (activeTab === 'compare') {
      getBacktestHistory()
        .then(setCompareHistory)
        .catch(() => setCompareError('无法加载回测历史'));
    }
  }, [activeTab]);

  const toggleCompareSelect = (id: string) => {
    setCompareSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < 5) {
        next.add(id);
      }
      return next;
    });
  };

  const handleCompare = async () => {
    const ids = Array.from(compareSelected);
    if (ids.length < 2) {
      setCompareError('请至少选择2个回测进行对比');
      return;
    }
    setCompareLoading(true);
    setCompareError(null);
    setCompareResult(null);
    try {
      const data = await compareBacktestRuns(ids);
      setCompareResult(data.runs);
    } catch (err) {
      setCompareError(err instanceof Error ? err.message : '对比请求失败');
    } finally {
      setCompareLoading(false);
    }
  };

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

  const handleOptimize = async () => {
    const p1vals = optParam1Values.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
    const p2vals = optParam2Values.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
    if (p1vals.length === 0 || p2vals.length === 0) {
      setOptError('请输入有效的参数值（逗号分隔的数字）');
      return;
    }
    setOptRunning(true);
    setOptError(null);
    setOptProgress('⏳ Submitting optimization...');
    setOptResult(null);
    stopOptPolling();
    try {
      const { task_id } = await runOptimization({
        strategy: config.strategy,
        symbol: config.symbol,
        start: config.start,
        end: config.end,
        capital: config.capital,
        period: config.period,
        param1_name: optParam1Name,
        param1_values: p1vals,
        param2_name: optParam2Name,
        param2_values: p2vals,
      });

      const poll = async () => {
        try {
          const data = await getBacktestResults(task_id) as Record<string, unknown>;
          const status = data.status as string;
          if (status === 'completed' || status === 'Completed') {
            stopOptPolling();
            setOptResult(data as unknown as OptimizeResultData);
            setOptProgress(null);
            setOptRunning(false);
          } else if (status === 'Failed' || status === 'failed') {
            stopOptPolling();
            setOptError((data.error as string) ?? 'Optimization failed');
            setOptProgress(null);
            setOptRunning(false);
          } else {
            setOptProgress((data.progress as string) ?? '⏳ Running...');
          }
        } catch {
          // transient error, keep polling
        }
      };

      setTimeout(() => void poll(), 500);
      optPollerRef.current = setInterval(() => void poll(), 1000);
    } catch (err) {
      setOptError(err instanceof Error ? err.message : '优化提交失败');
      setOptProgress(null);
      setOptRunning(false);
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
  const drawdownCurve: DrawdownPoint[] = result?.drawdown_curve ?? [];
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
      {/* Tab Switcher */}
      <div className="flex gap-2">
        <button
          onClick={() => setActiveTab('backtest')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
            activeTab === 'backtest'
              ? 'bg-[#3b82f6] text-white'
              : 'bg-[#1e293b] text-[#94a3b8] hover:text-[#f8fafc] border border-[#334155]'
          }`}
        >
          <Play size={16} /> 回测
        </button>
        <button
          onClick={() => setActiveTab('optimize')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
            activeTab === 'optimize'
              ? 'bg-[#8b5cf6] text-white'
              : 'bg-[#1e293b] text-[#94a3b8] hover:text-[#f8fafc] border border-[#334155]'
          }`}
        >
          <Grid3X3 size={16} /> 参数优化
        </button>
        <button
          onClick={() => setActiveTab('compare')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer ${
            activeTab === 'compare'
              ? 'bg-[#06b6d4] text-white'
              : 'bg-[#1e293b] text-[#94a3b8] hover:text-[#f8fafc] border border-[#334155]'
          }`}
        >
          <Layers size={16} /> 回测对比
        </button>
      </div>

      {activeTab === 'backtest' && (<>
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

          {/* Drawdown (Underwater) Chart */}
          {drawdownCurve.length > 0 && (
            <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
              <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📉 回撤曲线</h3>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={drawdownCurve}>
                  <defs>
                    <linearGradient id="drawdownGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={0.1}/>
                      <stop offset="100%" stopColor="#ef4444" stopOpacity={0.6}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    tickLine={{ stroke: '#334155' }} axisLine={{ stroke: '#334155' }}
                    interval={Math.floor(drawdownCurve.length / 6)} />
                  <YAxis
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    tickLine={{ stroke: '#334155' }} axisLine={{ stroke: '#334155' }}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                    formatter={(v: number) => [`${(Number(v) * 100).toFixed(2)}%`, '回撤']}
                    labelStyle={{ color: '#94a3b8' }} />
                  <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="url(#drawdownGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Monthly Returns Heatmap */}
          {equityCurve.length > 0 && (() => {
            const computeMonthlyReturns = (curve: EquityPoint[]): Map<string, Map<number, number>> => {
              const monthly = new Map<string, { first: number; last: number }>();
              for (const pt of curve) {
                const key = pt.date.substring(0, 7);
                const existing = monthly.get(key);
                if (!existing) monthly.set(key, { first: pt.value, last: pt.value });
                else existing.last = pt.value;
              }
              const res = new Map<string, Map<number, number>>();
              for (const [key, vals] of monthly) {
                const [year, monthStr] = key.split('-');
                const month = parseInt(monthStr);
                const ret = vals.first !== 0 ? (vals.last - vals.first) / vals.first : 0;
                if (!res.has(year)) res.set(year, new Map());
                res.get(year)!.set(month, ret);
              }
              return res;
            };
            const getReturnColor = (ret: number): string => {
              if (ret > 0) return `rgba(34, 197, 94, ${Math.min(Math.abs(ret) * 5, 0.8)})`;
              return `rgba(239, 68, 68, ${Math.min(Math.abs(ret) * 5, 0.8)})`;
            };
            const monthlyReturns = computeMonthlyReturns(equityCurve);
            const years = Array.from(monthlyReturns.keys()).sort();
            const months = [1,2,3,4,5,6,7,8,9,10,11,12];
            const monthLabels = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];

            return (
              <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
                <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📅 月度收益热力图</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[#334155]">
                        <th className="text-left py-2 px-3 text-xs font-medium text-[#94a3b8]">年份</th>
                        {monthLabels.map(m => (
                          <th key={m} className="text-center py-2 px-2 text-xs font-medium text-[#94a3b8]">{m}</th>
                        ))}
                        <th className="text-center py-2 px-3 text-xs font-medium text-[#f59e0b]">全年</th>
                      </tr>
                    </thead>
                    <tbody>
                      {years.map(year => {
                        const yearData = monthlyReturns.get(year)!;
                        const yearVals = Array.from(yearData.values());
                        const annualReturn = yearVals.reduce((acc, r) => acc * (1 + r), 1) - 1;
                        return (
                          <tr key={year} className="border-b border-[#334155]/50">
                            <td className="py-2 px-3 text-xs font-medium text-[#e2e8f0]">{year}</td>
                            {months.map(m => {
                              const ret = yearData.get(m);
                              return (
                                <td key={m} className="text-center py-2 px-2">
                                  {ret !== undefined ? (
                                    <span className="inline-block rounded px-1.5 py-0.5 text-xs font-mono text-white"
                                      style={{ backgroundColor: getReturnColor(ret) }}>
                                      {ret >= 0 ? '+' : ''}{(ret * 100).toFixed(1)}%
                                    </span>
                                  ) : (
                                    <span className="text-[#475569] text-xs">—</span>
                                  )}
                                </td>
                              );
                            })}
                            <td className="text-center py-2 px-3">
                              <span className="inline-block rounded px-1.5 py-0.5 text-xs font-bold font-mono text-white"
                                style={{ backgroundColor: getReturnColor(annualReturn) }}>
                                {annualReturn >= 0 ? '+' : ''}{(annualReturn * 100).toFixed(1)}%
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}

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

      {/* ── Monte Carlo Simulation ──────────────────────────────────── */}
      {result && taskId && (
        <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-[#f8fafc] flex items-center gap-2">
              <Activity size={20} className="text-[#a855f7]" /> 蒙特卡洛模拟
            </h2>
            <button
              onClick={handleRunMonteCarlo}
              disabled={mcRunning}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-[#a855f7] text-white hover:bg-[#9333ea] disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
            >
              {mcRunning ? (
                <><Loader2 size={16} className="animate-spin" /> 模拟中...</>
              ) : (
                <><BarChart3 size={16} /> 运行蒙特卡洛模拟 (1000次)</>
              )}
            </button>
          </div>

          {mcError && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
              {mcError}
            </div>
          )}

          {mcResult && (() => {
            // Build fan chart data from paths
            const fanData = (() => {
              if (!mcResult.paths.length) return [];
              const maxLen = Math.max(...mcResult.paths.map(p => p.equity.length));
              const pathMap: Record<number, number[]> = {};
              mcResult.paths.forEach(p => { pathMap[p.percentile] = p.equity; });
              const rows = [];
              for (let i = 0; i < maxLen; i++) {
                rows.push({
                  day: i,
                  p5: pathMap[5]?.[i] ?? null,
                  p25: pathMap[25]?.[i] ?? null,
                  p50: pathMap[50]?.[i] ?? null,
                  p75: pathMap[75]?.[i] ?? null,
                  p95: pathMap[95]?.[i] ?? null,
                });
              }
              return rows;
            })();

            const rd = mcResult.return_distribution;
            const dd = mcResult.drawdown_distribution;

            // Histogram: divide returns into 20 bins
            const histogramData = (() => {
              const p5 = rd.percentile_5;
              const p95 = rd.percentile_95;
              const range = p95 - p5;
              if (range <= 0) return [];
              const margin = range * 0.2;
              const lo = p5 - margin;
              const hi = p95 + margin;
              const binCount = 20;
              const binWidth = (hi - lo) / binCount;
              const bins = Array.from({ length: binCount }, (_, i) => ({
                label: ((lo + binWidth * i + binWidth / 2) * 100).toFixed(0) + '%',
                rangeStart: lo + binWidth * i,
                rangeEnd: lo + binWidth * (i + 1),
                count: 0,
              }));
              // We don't have individual sim returns on frontend, so approximate from distribution
              // Instead show the distribution stats as a visual
              return bins;
            })();
            void histogramData;

            const lossColor = mcResult.probability_of_loss < 0.2 ? '#22c55e'
              : mcResult.probability_of_loss < 0.4 ? '#eab308' : '#ef4444';

            return (
              <div className="space-y-4">
                {/* Fan Chart */}
                <div className="bg-[#0f172a] rounded-lg p-4">
                  <h3 className="text-sm font-medium text-[#94a3b8] mb-3">模拟路径 (P5-P95 扇形图)</h3>
                  <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={fanData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="day" stroke="#64748b" tick={{ fontSize: 11 }} label={{ value: '交易日', position: 'insideBottomRight', offset: -5, fill: '#64748b', fontSize: 11 }} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} label={{ value: '净值', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                        formatter={(v: number, name: string) => [`${v?.toFixed(4)}`, name]}
                        labelFormatter={(l: number) => `第 ${l} 天`}
                      />
                      <Area type="monotone" dataKey="p95" stroke="none" fill="#a855f7" fillOpacity={0.1} name="P95" />
                      <Area type="monotone" dataKey="p75" stroke="none" fill="#a855f7" fillOpacity={0.15} name="P75" />
                      <Area type="monotone" dataKey="p50" stroke="#a855f7" strokeWidth={2} fill="#a855f7" fillOpacity={0.25} name="P50 (中位)" />
                      <Area type="monotone" dataKey="p25" stroke="none" fill="#0f172a" fillOpacity={0.6} name="P25" />
                      <Area type="monotone" dataKey="p5" stroke="none" fill="#0f172a" fillOpacity={0.7} name="P5" />
                      <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* Distribution Summary Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-[#0f172a] rounded-lg p-4 border border-[#334155]">
                    <div className="text-xs text-[#94a3b8] mb-1">预期收益 (中位)</div>
                    <div className="text-xl font-bold" style={{ color: rd.percentile_50 >= 0 ? '#22c55e' : '#ef4444' }}>
                      {(rd.percentile_50 * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-[#64748b] mt-1">
                      ± {((rd.std ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-[#0f172a] rounded-lg p-4 border border-[#334155]">
                    <div className="text-xs text-[#94a3b8] mb-1">最大回撤范围</div>
                    <div className="text-xl font-bold text-[#ef4444]">
                      {(dd.percentile_50 * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-[#64748b] mt-1">
                      P5: {(dd.percentile_5 * 100).toFixed(1)}% ~ P95: {(dd.percentile_95 * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-[#0f172a] rounded-lg p-4 border border-[#334155]">
                    <div className="text-xs text-[#94a3b8] mb-1">亏损概率</div>
                    <div className="text-xl font-bold" style={{ color: lossColor }}>
                      {(mcResult.probability_of_loss * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-[#64748b] mt-1">
                      最终收益 &lt; 0 的比例
                    </div>
                  </div>
                  <div className="bg-[#0f172a] rounded-lg p-4 border border-[#334155]">
                    <div className="text-xs text-[#94a3b8] mb-1">爆仓概率</div>
                    <div className="text-xl font-bold" style={{ color: mcResult.probability_of_ruin > 0.05 ? '#ef4444' : '#22c55e' }}>
                      {(mcResult.probability_of_ruin * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-[#64748b] mt-1">
                      最大回撤 &gt; 50% 的比例
                    </div>
                  </div>
                </div>

                {/* Detailed percentiles */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div className="bg-[#0f172a] rounded-lg p-4 border border-[#334155]">
                    <h3 className="text-sm font-medium text-[#94a3b8] mb-2">收益分布</h3>
                    <div className="space-y-1 text-sm font-mono">
                      {[
                        { label: 'P5 (悲观)', value: rd.percentile_5 },
                        { label: 'P25', value: rd.percentile_25 },
                        { label: 'P50 (中位)', value: rd.percentile_50 },
                        { label: 'P75', value: rd.percentile_75 },
                        { label: 'P95 (乐观)', value: rd.percentile_95 },
                        { label: '均值', value: rd.mean },
                      ].map(({ label, value }) => (
                        <div key={label} className="flex justify-between">
                          <span className="text-[#94a3b8]">{label}</span>
                          <span style={{ color: value >= 0 ? '#22c55e' : '#ef4444' }}>
                            {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="bg-[#0f172a] rounded-lg p-4 border border-[#334155]">
                    <h3 className="text-sm font-medium text-[#94a3b8] mb-2">回撤分布</h3>
                    <div className="space-y-1 text-sm font-mono">
                      {[
                        { label: 'P5 (最差)', value: dd.percentile_5 },
                        { label: 'P25', value: dd.percentile_25 },
                        { label: 'P50 (中位)', value: dd.percentile_50 },
                        { label: 'P75', value: dd.percentile_75 },
                        { label: 'P95 (最优)', value: dd.percentile_95 },
                        { label: '均值', value: dd.mean },
                      ].map(({ label, value }) => (
                        <div key={label} className="flex justify-between">
                          <span className="text-[#94a3b8]">{label}</span>
                          <span className="text-[#ef4444]">{(value * 100).toFixed(2)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="text-xs text-[#64748b] text-center">
                  基于 {mcResult.simulations} 次 Bootstrap 重采样模拟 · {mcResult.trading_days} 个交易日
                </div>
              </div>
            );
          })()}
        </div>
      )}
      </>)}

      {/* ── Performance Attribution ──────────────────────────────────── */}
      {result && taskId && (
        <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-[#f8fafc] flex items-center gap-2">
              <BarChart3 size={20} className="text-[#f59e0b]" /> 📊 绩效归因
            </h2>
            <button
              onClick={handleRunAttribution}
              disabled={attrRunning}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-[#f59e0b] text-white hover:bg-[#d97706] disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
            >
              {attrRunning ? (
                <><Loader2 size={16} className="animate-spin" /> 分析中…</>
              ) : (
                <><Target size={16} /> Brinson 归因分析</>
              )}
            </button>
          </div>

          {attrError && (
            <div className="text-sm text-red-400 bg-red-400/10 px-3 py-2 rounded-lg">
              {attrError}
            </div>
          )}

          {attrResult && (() => {
            const attr = attrResult;

            // Waterfall data
            const waterfallData = [
              { name: '选股', value: attr.selection_contribution, fill: '#22c55e' },
              { name: '择时', value: attr.timing_contribution, fill: attr.timing_contribution >= 0 ? '#22c55e' : '#ef4444' },
              { name: '交互', value: attr.interaction_contribution, fill: '#3b82f6' },
              { name: '费用', value: -attr.commission_drag, fill: '#ef4444' },
              { name: '总收益', value: attr.total_return, fill: '#8b5cf6' },
            ];

            // Pie data (absolute proportions)
            const total = Math.abs(attr.selection_contribution) + Math.abs(attr.timing_contribution) + Math.abs(attr.interaction_contribution) + Math.abs(attr.commission_drag);
            const pieData = total > 0 ? [
              { name: '选股', value: Math.abs(attr.selection_contribution), color: '#22c55e' },
              { name: '择时', value: Math.abs(attr.timing_contribution), color: '#3b82f6' },
              { name: '交互', value: Math.abs(attr.interaction_contribution), color: '#8b5cf6' },
              { name: '费用', value: Math.abs(attr.commission_drag), color: '#ef4444' },
            ] : [];

            // Top/Bottom trades
            const sorted = [...attr.trade_attribution].sort((a, b) => b.contribution - a.contribution);
            const topTrades = sorted.slice(0, 5);
            const bottomTrades = sorted.slice(-5).reverse();

            return (
              <div className="space-y-6">
                {/* Waterfall Chart */}
                <div>
                  <h3 className="text-sm font-medium text-[#94a3b8] mb-3">收益归因瀑布图</h3>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={waterfallData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                      <YAxis tickFormatter={(v: number) => `${v.toFixed(1)}%`} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                      <Tooltip
                        formatter={(value: number) => [`${value.toFixed(2)}%`, '贡献']}
                        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                        labelStyle={{ color: '#f8fafc' }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {waterfallData.map((d, i) => (
                          <Cell key={i} fill={d.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Contribution Breakdown Cards + Pie */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: '选股贡献', value: attr.selection_contribution, color: '#22c55e', icon: '🎯' },
                      { label: '择时贡献', value: attr.timing_contribution, color: '#3b82f6', icon: '⏱️' },
                      { label: '交互效应', value: attr.interaction_contribution, color: '#8b5cf6', icon: '🔄' },
                      { label: '费用拖累', value: -attr.commission_drag, color: '#ef4444', icon: '💰' },
                    ].map((item) => (
                      <div key={item.label} className="bg-[#0f172a] rounded-lg p-3 border border-[#334155]">
                        <div className="text-xs text-[#64748b] mb-1">{item.icon} {item.label}</div>
                        <div className="text-lg font-bold" style={{ color: item.value >= 0 ? '#22c55e' : '#ef4444' }}>
                          {item.value >= 0 ? '+' : ''}{item.value.toFixed(2)}%
                        </div>
                      </div>
                    ))}
                  </div>
                  {pieData.length > 0 && (
                    <div className="flex items-center justify-center">
                      <ResponsiveContainer width="100%" height={200}>
                        <PieChart>
                          <Pie
                            data={pieData}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            outerRadius={75}
                            innerRadius={40}
                            label={({ name, percent }: { name: string; percent: number }) => `${name} ${(percent * 100).toFixed(0)}%`}
                            labelLine={false}
                          >
                            {pieData.map((d, i) => (
                              <Cell key={i} fill={d.color} />
                            ))}
                          </Pie>
                          <Tooltip
                            formatter={(value: number) => [`${value.toFixed(2)}%`, '绝对贡献']}
                            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {/* Top / Bottom Trades */}
                {attr.trade_attribution.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Top Trades */}
                    <div>
                      <h3 className="text-sm font-medium text-[#22c55e] mb-2">🏆 最佳交易 Top 5</h3>
                      <div className="overflow-x-auto rounded-lg border border-[#334155]">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="bg-[#0f172a]">
                              <th className="py-2 px-3 text-left text-[#64748b]">标的</th>
                              <th className="py-2 px-3 text-left text-[#64748b]">日期</th>
                              <th className="py-2 px-3 text-right text-[#64748b]">收益%</th>
                              <th className="py-2 px-3 text-right text-[#64748b]">贡献%</th>
                            </tr>
                          </thead>
                          <tbody>
                            {topTrades.map((t, i) => (
                              <tr key={i} className="border-t border-[#1e293b] hover:bg-[#0f172a]/50">
                                <td className="py-1.5 px-3 text-[#f8fafc]">{t.symbol}</td>
                                <td className="py-1.5 px-3 text-[#94a3b8]">{t.entry_date?.slice(0, 10)} → {t.exit_date?.slice(0, 10)}</td>
                                <td className="py-1.5 px-3 text-right text-[#22c55e] font-mono">{t.return_pct >= 0 ? '+' : ''}{t.return_pct.toFixed(2)}%</td>
                                <td className="py-1.5 px-3 text-right text-[#22c55e] font-mono">{t.contribution >= 0 ? '+' : ''}{t.contribution.toFixed(2)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    {/* Bottom Trades */}
                    <div>
                      <h3 className="text-sm font-medium text-[#ef4444] mb-2">📉 最差交易 Bottom 5</h3>
                      <div className="overflow-x-auto rounded-lg border border-[#334155]">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="bg-[#0f172a]">
                              <th className="py-2 px-3 text-left text-[#64748b]">标的</th>
                              <th className="py-2 px-3 text-left text-[#64748b]">日期</th>
                              <th className="py-2 px-3 text-right text-[#64748b]">收益%</th>
                              <th className="py-2 px-3 text-right text-[#64748b]">贡献%</th>
                            </tr>
                          </thead>
                          <tbody>
                            {bottomTrades.map((t, i) => (
                              <tr key={i} className="border-t border-[#1e293b] hover:bg-[#0f172a]/50">
                                <td className="py-1.5 px-3 text-[#f8fafc]">{t.symbol}</td>
                                <td className="py-1.5 px-3 text-[#94a3b8]">{t.entry_date?.slice(0, 10)} → {t.exit_date?.slice(0, 10)}</td>
                                <td className="py-1.5 px-3 text-right text-[#ef4444] font-mono">{t.return_pct >= 0 ? '+' : ''}{t.return_pct.toFixed(2)}%</td>
                                <td className="py-1.5 px-3 text-right text-[#ef4444] font-mono">{t.contribution >= 0 ? '+' : ''}{t.contribution.toFixed(2)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {/* Monthly Attribution Table */}
                {attr.monthly_attribution.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-[#94a3b8] mb-2">📅 月度归因</h3>
                    <div className="overflow-x-auto rounded-lg border border-[#334155]">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-[#0f172a]">
                            <th className="py-2 px-3 text-left text-[#64748b]">月份</th>
                            <th className="py-2 px-3 text-right text-[#64748b]">收益%</th>
                            <th className="py-2 px-3 text-right text-[#64748b]">交易数</th>
                            <th className="py-2 px-3 text-left text-[#64748b]">最佳交易</th>
                            <th className="py-2 px-3 text-left text-[#64748b]">最差交易</th>
                          </tr>
                        </thead>
                        <tbody>
                          {attr.monthly_attribution.map((m) => (
                            <tr key={m.month} className="border-t border-[#1e293b] hover:bg-[#0f172a]/50">
                              <td className="py-1.5 px-3 text-[#f8fafc] font-mono">{m.month}</td>
                              <td className="py-1.5 px-3 text-right font-mono" style={{ color: m.return_pct >= 0 ? '#22c55e' : '#ef4444' }}>
                                {m.return_pct >= 0 ? '+' : ''}{m.return_pct.toFixed(2)}%
                              </td>
                              <td className="py-1.5 px-3 text-right text-[#94a3b8]">{m.trade_count}</td>
                              <td className="py-1.5 px-3 text-[#22c55e] text-xs">{m.best_trade ?? '—'}</td>
                              <td className="py-1.5 px-3 text-[#ef4444] text-xs">{m.worst_trade ?? '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                <div className="text-xs text-[#64748b] text-center">
                  Brinson 绩效归因模型 · 选股 + 择时 + 交互 = 总收益
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* ── Parameter Optimization Tab ────────────────────────────────── */}
      {activeTab === 'optimize' && (
        <>
          <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6 space-y-4">
            <h2 className="text-lg font-semibold text-[#f8fafc] flex items-center gap-2">
              <Grid3X3 size={20} className="text-[#8b5cf6]" /> 参数优化 (Grid Search)
            </h2>
            <p className="text-sm text-[#94a3b8]">
              选择两个策略参数，输入候选值范围，系统将遍历所有组合并生成灵敏度热力图。
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">策略</label>
                <select
                  value={config.strategy}
                  onChange={(e) => updateConfig('strategy', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#8b5cf6]"
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
                  placeholder="600519.SH"
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#8b5cf6]" />
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">K线周期</label>
                <select
                  value={config.period}
                  onChange={(e) => updateConfig('period', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#8b5cf6]"
                >
                  {PERIODS.map((p) => (
                    <option key={p.value} value={p.value}>{p.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">开始日期</label>
                <input type="date" value={config.start}
                  onChange={(e) => updateConfig('start', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#8b5cf6]" />
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">结束日期</label>
                <input type="date" value={config.end}
                  onChange={(e) => updateConfig('end', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#8b5cf6]" />
              </div>
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">初始资金 (¥)</label>
                <input type="number" value={config.capital} min={1000} step={10000}
                  onChange={(e) => updateConfig('capital', Number(e.target.value))}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#8b5cf6]" />
              </div>
            </div>

            <div className="border-t border-[#334155] pt-4 mt-4">
              <h3 className="text-sm font-semibold text-[#f8fafc] mb-3">参数网格配置</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="block text-sm text-[#94a3b8]">参数1 名称</label>
                  <input type="text" value={optParam1Name}
                    onChange={(e) => setOptParam1Name(e.target.value)}
                    placeholder="fast_period"
                    className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#8b5cf6]" />
                  <label className="block text-sm text-[#94a3b8]">参数1 候选值 (逗号分隔)</label>
                  <input type="text" value={optParam1Values}
                    onChange={(e) => setOptParam1Values(e.target.value)}
                    placeholder="3,5,8,10,13,15,20"
                    className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] font-mono outline-none focus:border-[#8b5cf6]" />
                </div>
                <div className="space-y-2">
                  <label className="block text-sm text-[#94a3b8]">参数2 名称</label>
                  <input type="text" value={optParam2Name}
                    onChange={(e) => setOptParam2Name(e.target.value)}
                    placeholder="slow_period"
                    className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#8b5cf6]" />
                  <label className="block text-sm text-[#94a3b8]">参数2 候选值 (逗号分隔)</label>
                  <input type="text" value={optParam2Values}
                    onChange={(e) => setOptParam2Values(e.target.value)}
                    placeholder="10,15,20,25,30,40,50,60"
                    className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] font-mono outline-none focus:border-[#8b5cf6]" />
                </div>
              </div>
            </div>

            {optError && (
              <div className="text-sm text-[#ef4444] bg-[#ef4444]/10 border border-[#ef4444]/20 rounded-lg px-4 py-2">{optError}</div>
            )}

            {optRunning && optProgress && (
              <div className="flex items-center gap-3 text-sm text-[#a78bfa] bg-[#8b5cf6]/10 border border-[#8b5cf6]/20 rounded-lg px-4 py-2.5">
                <Loader2 size={16} className="animate-spin flex-shrink-0" />
                <span>{optProgress}</span>
              </div>
            )}

            <button onClick={() => void handleOptimize()} disabled={optRunning}
              className="flex items-center gap-2 px-6 py-2.5 bg-[#8b5cf6] hover:bg-[#7c3aed] text-white rounded-lg text-sm font-medium disabled:opacity-50 transition-colors cursor-pointer">
              {optRunning ? <Loader2 size={16} className="animate-spin" /> : <Grid3X3 size={16} />}
              {optRunning ? '优化中…' : '开始优化'}
            </button>
          </div>

          {/* Optimization Results */}
          {optResult && optResult.grid && (
            <OptimizationResults result={optResult} metric={optMetric} onMetricChange={setOptMetric} />
          )}
        </>
      )}

      {/* ── Comparison Tab ────────────────────────────────────── */}
      {activeTab === 'compare' && (
        <ComparisonTab
          history={compareHistory}
          selected={compareSelected}
          onToggle={toggleCompareSelect}
          onCompare={() => void handleCompare()}
          compareResult={compareResult}
          loading={compareLoading}
          error={compareError}
        />
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

// ── Heatmap color helpers ────────────────────────────────────────────

function interpolateColor(t: number): string {
  // 0 = red (#ef4444), 0.5 = yellow (#eab308), 1 = green (#22c55e)
  const clamp = Math.max(0, Math.min(1, t));
  let r: number, g: number, b: number;
  if (clamp < 0.5) {
    const p = clamp / 0.5;
    r = Math.round(239 + (234 - 239) * p);
    g = Math.round(68 + (179 - 68) * p);
    b = Math.round(68 + (8 - 68) * p);
  } else {
    const p = (clamp - 0.5) / 0.5;
    r = Math.round(234 + (34 - 234) * p);
    g = Math.round(179 + (197 - 179) * p);
    b = Math.round(8 + (94 - 8) * p);
  }
  return `rgb(${r},${g},${b})`;
}

const METRIC_LABELS: Record<string, string> = {
  sharpe: '夏普比率',
  total_return: '总收益率 (%)',
  max_drawdown: '最大回撤 (%)',
  win_rate: '胜率 (%)',
};

function OptimizationResults({
  result,
  metric,
  onMetricChange,
}: {
  result: OptimizeResultData;
  metric: 'sharpe' | 'total_return' | 'max_drawdown' | 'win_rate';
  onMetricChange: (m: 'sharpe' | 'total_return' | 'max_drawdown' | 'win_rate') => void;
}) {
  const { grid, best, param1_name, param2_name } = result;

  // Extract unique sorted param values
  const p1Set = [...new Set(grid.map(g => g.param1))].sort((a, b) => a - b);
  const p2Set = [...new Set(grid.map(g => g.param2))].sort((a, b) => a - b);

  // Build lookup
  const lookup = new Map<string, OptGridEntry>();
  for (const g of grid) {
    lookup.set(`${g.param1}_${g.param2}`, g);
  }

  // Collect valid metric values for normalization
  const values: number[] = [];
  for (const g of grid) {
    const v = g[metric];
    if (v != null && !g.skipped) values.push(v);
  }
  const minVal = values.length > 0 ? Math.min(...values) : 0;
  const maxVal = values.length > 0 ? Math.max(...values) : 1;
  const range = maxVal - minVal || 1;

  const normalize = (v: number | null, entry: OptGridEntry): number | null => {
    if (v == null || entry.skipped) return null;
    const t = (v - minVal) / range;
    // For max_drawdown, invert: less negative (closer to 0) = better = green
    return metric === 'max_drawdown' ? 1 - t : t;
  };

  const formatCell = (v: number | null, entry: OptGridEntry): string => {
    if (entry.skipped) return '—';
    if (v == null) return 'ERR';
    return v.toFixed(2);
  };

  return (
    <>
      {/* Best Parameters Banner */}
      {best && (
        <div className="bg-[#1e293b] rounded-xl border border-[#8b5cf6]/40 p-4">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🏆</span>
            <div>
              <p className="text-sm text-[#94a3b8]">最优参数</p>
              <p className="text-lg font-bold text-[#f8fafc]">
                {param1_name}={best.param1}, {param2_name}={best.param2}
              </p>
              <p className="text-sm text-[#94a3b8]">
                收益率: <span className={best.total_return >= 0 ? 'text-green-400' : 'text-red-400'}>{best.total_return >= 0 ? '+' : ''}{best.total_return.toFixed(2)}%</span>
                {' · '}夏普: <span className={best.sharpe >= 1 ? 'text-green-400' : 'text-yellow-400'}>{best.sharpe.toFixed(2)}</span>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Heatmap */}
      <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-[#f8fafc]">🔥 灵敏度热力图</h3>
          <div className="flex gap-1">
            {(['sharpe', 'total_return', 'max_drawdown', 'win_rate'] as const).map((m) => (
              <button
                key={m}
                onClick={() => onMetricChange(m)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
                  metric === m
                    ? 'bg-[#8b5cf6] text-white'
                    : 'bg-[#334155] text-[#94a3b8] hover:text-[#f8fafc]'
                }`}
              >
                {METRIC_LABELS[m]}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="text-sm border-collapse">
            <thead>
              <tr>
                <th className="px-3 py-2 text-xs text-[#94a3b8] font-medium text-left border-b border-r border-[#334155]">
                  {param1_name} ↓ \ {param2_name} →
                </th>
                {p2Set.map((p2) => (
                  <th key={p2} className="px-3 py-2 text-xs text-[#f8fafc] font-mono font-medium text-center border-b border-[#334155]">
                    {p2}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {p1Set.map((p1) => (
                <tr key={p1}>
                  <td className="px-3 py-2 text-xs text-[#f8fafc] font-mono font-medium border-r border-[#334155]">
                    {p1}
                  </td>
                  {p2Set.map((p2) => {
                    const entry = lookup.get(`${p1}_${p2}`);
                    if (!entry) return <td key={p2} className="px-3 py-2 text-center text-xs text-[#475569]">—</td>;
                    const val = entry[metric];
                    const t = normalize(val, entry);
                    const isBest = best && p1 === best.param1 && p2 === best.param2;
                    return (
                      <td
                        key={p2}
                        className={`px-3 py-2 text-center text-xs font-mono font-medium ${
                          isBest ? 'ring-2 ring-[#f8fafc] ring-offset-1 ring-offset-[#1e293b]' : ''
                        }`}
                        style={{
                          backgroundColor: t != null ? interpolateColor(t) + '33' : 'transparent',
                          color: t != null ? interpolateColor(t) : '#475569',
                        }}
                        title={`${param1_name}=${p1}, ${param2_name}=${p2}`}
                      >
                        {formatCell(val, entry)}
                        {isBest && ' ★'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Color scale legend */}
        <div className="flex items-center gap-2 mt-4 text-xs text-[#94a3b8]">
          <span>{metric === 'max_drawdown' ? '差' : '差'}</span>
          <div className="flex h-3 rounded overflow-hidden flex-1 max-w-[200px]">
            {Array.from({ length: 20 }, (_, i) => (
              <div
                key={i}
                className="flex-1"
                style={{ backgroundColor: interpolateColor(i / 19) }}
              />
            ))}
          </div>
          <span>{metric === 'max_drawdown' ? '好' : '好'}</span>
          <span className="ml-2 text-[#475569]">({METRIC_LABELS[metric]})</span>
        </div>
      </div>

      {/* Full Grid Data Table */}
      <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
        <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📊 完整网格数据</h3>
        <div className="overflow-x-auto max-h-96 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-[#1e293b]">
              <tr className="border-b border-[#334155]">
                {[param1_name, param2_name, '收益率 (%)', '夏普比率', '最大回撤 (%)', '胜率 (%)'].map((h) => (
                  <th key={h} className="text-left py-3 px-4 text-xs font-medium text-[#94a3b8]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {grid
                .filter((g) => !g.skipped)
                .sort((a, b) => (b.sharpe ?? 0) - (a.sharpe ?? 0))
                .map((g, i) => {
                  const isBest = best && g.param1 === best.param1 && g.param2 === best.param2;
                  return (
                    <tr key={i} className={`border-b border-[#334155]/50 hover:bg-[#334155]/30 ${isBest ? 'bg-[#8b5cf6]/10' : ''}`}>
                      <td className="py-2 px-4 text-[#f8fafc] font-mono">{g.param1}</td>
                      <td className="py-2 px-4 text-[#f8fafc] font-mono">{g.param2}</td>
                      <td className={`py-2 px-4 font-mono font-bold ${(g.total_return ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {g.total_return != null ? g.total_return.toFixed(2) : '—'}
                      </td>
                      <td className={`py-2 px-4 font-mono ${(g.sharpe ?? 0) >= 1 ? 'text-green-400' : (g.sharpe ?? 0) >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {g.sharpe != null ? g.sharpe.toFixed(2) : '—'}
                      </td>
                      <td className="py-2 px-4 text-red-400 font-mono">
                        {g.max_drawdown != null ? `-${Math.abs(g.max_drawdown).toFixed(2)}` : '—'}
                      </td>
                      <td className={`py-2 px-4 font-mono ${(g.win_rate ?? 0) >= 50 ? 'text-green-400' : 'text-[#94a3b8]'}`}>
                        {g.win_rate != null ? g.win_rate.toFixed(1) : '—'}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

// ── Comparison Tab Component ────────────────────────────────────────

const COMPARE_COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#a855f7'];

function ComparisonTab({
  history,
  selected,
  onToggle,
  onCompare,
  compareResult,
  loading,
  error,
}: {
  history: BacktestRunSummary[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onCompare: () => void;
  compareResult: CompareRunData[] | null;
  loading: boolean;
  error: string | null;
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left Panel: History list with checkboxes */}
      <div className="lg:col-span-1 space-y-4">
        <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-semibold text-[#f8fafc]">📋 回测历史</h3>
            <span className="text-xs text-[#94a3b8]">{selected.size}/5 已选</span>
          </div>

          {history.length === 0 && !error && (
            <p className="text-sm text-[#94a3b8] text-center py-8">暂无回测记录。运行回测后，结果会自动保存。</p>
          )}

          <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
            {history.map((run) => {
              const isSelected = selected.has(run.id);
              const retPct = run.total_return != null ? (run.total_return * 100).toFixed(2) : '—';
              const sharpeFmt = run.sharpe != null ? run.sharpe.toFixed(2) : '—';
              return (
                <button
                  key={run.id}
                  onClick={() => onToggle(run.id)}
                  className={`w-full text-left p-3 rounded-lg border transition-colors cursor-pointer ${
                    isSelected
                      ? 'border-[#06b6d4] bg-[#06b6d4]/10'
                      : 'border-[#334155] bg-[#0f172a] hover:border-[#475569]'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <CheckSquare
                      size={16}
                      className={`mt-0.5 flex-shrink-0 ${isSelected ? 'text-[#06b6d4]' : 'text-[#475569]'}`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-[#f8fafc] truncate">{run.strategy}</span>
                        <span className="text-xs text-[#94a3b8] truncate">{run.symbols}</span>
                      </div>
                      <div className="text-xs text-[#64748b] mt-0.5">
                        {run.start_date} ~ {run.end_date}
                      </div>
                      <div className="flex gap-3 mt-1 text-xs">
                        <span className={Number(retPct) >= 0 ? 'text-green-400' : 'text-red-400'}>
                          收益: {retPct}%
                        </span>
                        <span className={Number(sharpeFmt) >= 1 ? 'text-green-400' : 'text-yellow-400'}>
                          夏普: {sharpeFmt}
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {error && (
            <div className="text-sm text-[#ef4444] bg-[#ef4444]/10 border border-[#ef4444]/20 rounded-lg px-4 py-2 mt-3">
              {error}
            </div>
          )}

          <button
            onClick={onCompare}
            disabled={selected.size < 2 || loading}
            className="w-full mt-4 flex items-center justify-center gap-2 px-4 py-2.5 bg-[#06b6d4] hover:bg-[#0891b2] text-white rounded-lg text-sm font-medium disabled:opacity-50 transition-colors cursor-pointer"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Layers size={16} />}
            {loading ? '对比中…' : `对比 (${selected.size})`}
          </button>
        </div>
      </div>

      {/* Right Panel: Comparison results */}
      <div className="lg:col-span-2 space-y-6">
        {!compareResult && !loading && (
          <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-12 text-center">
            <Layers size={48} className="mx-auto text-[#334155] mb-4" />
            <p className="text-[#94a3b8]">选择 2-5 个回测记录，点击「对比」查看结果</p>
          </div>
        )}

        {compareResult && compareResult.length > 0 && (
          <>
            {/* Overlaid Equity Curves */}
            <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
              <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📈 收益曲线对比</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="date"
                    stroke="#64748b"
                    tick={{ fontSize: 11 }}
                    allowDuplicatedCategory={false}
                  />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickFormatter={(v: number) => `${(v / 10000).toFixed(0)}万`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Legend />
                  {compareResult.map((run, i) => (
                    <Line
                      key={run.id}
                      data={run.equity_curve}
                      dataKey="value"
                      name={`${run.strategy} (${run.symbols})`}
                      stroke={COMPARE_COLORS[i % COMPARE_COLORS.length]}
                      dot={false}
                      strokeWidth={2}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Metrics Comparison Table */}
            <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
              <h3 className="text-base font-semibold text-[#f8fafc] mb-4">📊 指标对比</h3>
              <div className="overflow-x-auto">
                <MetricsComparisonTable runs={compareResult} />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Metrics Comparison Table ────────────────────────────────────────

const COMPARE_METRICS: { key: string; label: string; format: (v: number | null) => string; higherBetter: boolean }[] = [
  { key: 'total_return', label: '总收益率', format: (v) => v != null ? `${(v * 100).toFixed(2)}%` : '—', higherBetter: true },
  { key: 'annual_return', label: '年化收益率', format: (v) => v != null ? `${(v * 100).toFixed(2)}%` : '—', higherBetter: true },
  { key: 'sharpe_ratio', label: '夏普比率', format: (v) => v != null ? v.toFixed(2) : '—', higherBetter: true },
  { key: 'max_drawdown', label: '最大回撤', format: (v) => v != null ? `-${(v * 100).toFixed(2)}%` : '—', higherBetter: false },
  { key: 'win_rate', label: '胜率', format: (v) => v != null ? `${(v * 100).toFixed(1)}%` : '—', higherBetter: true },
  { key: 'profit_factor', label: '盈亏比', format: (v) => v != null ? v.toFixed(2) : '—', higherBetter: true },
  { key: 'total_trades', label: '交易次数', format: (v) => v != null ? String(v) : '—', higherBetter: true },
];

function MetricsComparisonTable({ runs }: { runs: CompareRunData[] }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-[#334155]">
          <th className="text-left py-3 px-4 text-xs font-medium text-[#94a3b8]">指标</th>
          {runs.map((run, i) => (
            <th key={run.id} className="text-center py-3 px-4 text-xs font-medium" style={{ color: COMPARE_COLORS[i % COMPARE_COLORS.length] }}>
              {run.strategy}
              <br />
              <span className="text-[#64748b] font-normal">{run.symbols}</span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {COMPARE_METRICS.map((metric) => {
          const values = runs.map((r) => {
            const m = r.metrics as Record<string, number | null>;
            return m[metric.key] ?? null;
          });
          const validValues = values.filter((v): v is number => v != null);
          const bestVal = validValues.length > 0
            ? (metric.higherBetter ? Math.max(...validValues) : Math.min(...validValues))
            : null;
          // For max_drawdown (lower better), compare absolute values
          const isBest = (v: number | null) => {
            if (v == null || bestVal == null) return false;
            if (metric.key === 'max_drawdown') return v === bestVal;
            return v === bestVal;
          };

          return (
            <tr key={metric.key} className="border-b border-[#334155]/50 hover:bg-[#334155]/20">
              <td className="py-2.5 px-4 text-[#94a3b8] font-medium">{metric.label}</td>
              {values.map((v, i) => (
                <td
                  key={runs[i].id}
                  className={`py-2.5 px-4 text-center font-mono ${isBest(v) ? 'text-green-400 font-bold' : 'text-[#f8fafc]'}`}
                >
                  {metric.format(v)}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
