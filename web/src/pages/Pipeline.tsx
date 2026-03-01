import { useState, useEffect, useCallback, useRef } from 'react';
import { useTaskManager } from '../hooks/useTaskManager';
import { TaskOutput } from '../components/TaskPipeline';
import {
  factorMineParametric,
  factorMineGP,
  mlRetrain,
  runBacktest,
  getBacktestResults,
  getCacheStatus,
  syncData,
} from '../api/client';

/* ── Types ─────────────────────────────────────────────────────────── */

type StepStatus = 'idle' | 'running' | 'done' | 'error' | 'skipped';
type MiningMethod = 'parametric' | 'gp';

interface SharedConfig {
  symbols: string;
  start_date: string;
  end_date: string;
  data_source: 'akshare' | 'synthetic';
}

interface MiningConfig {
  method: MiningMethod;
  n_bars: number;
  horizon: number;
  ic_threshold: number;
  top_n: number;
  pop_size: number;
  generations: number;
  max_depth: number;
}

interface TrainConfig {
  algorithms: string;
  threshold: number;
}

interface BacktestConfig {
  strategy: string;
  capital: number;
  period: string;
}

/* ── Constants ─────────────────────────────────────────────────────── */

const STEP_LABELS = ['数据准备', '因子挖掘', 'ML 训练', '策略回测'] as const;
const STEP_ICONS = ['📡', '🔬', '🧠', '📊'] as const;

const STRATEGIES = [
  { value: 'ml_factor', label: 'ML因子 (推荐)' },
  { value: 'multi_factor', label: '多因子' },
  { value: 'rsi_reversal', label: 'RSI反转' },
  { value: 'sma_cross', label: 'SMA交叉' },
  { value: 'macd_trend', label: 'MACD趋势' },
  { value: 'sentiment_aware', label: '舆情感知' },
];

const SYMBOL_PRESETS = [
  { label: '茅台', value: '600519' },
  { label: '平安', value: '601318' },
  { label: '宁德', value: '300750' },
  { label: '招行', value: '600036' },
  { label: '比亚迪', value: '002594' },
];

/* ── Shared input component ────────────────────────────────────────── */

function InputField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="text-xs text-[#94a3b8] block mb-1">{label}</label>
      {children}
    </div>
  );
}

const inputCls = 'w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none';
const selectCls = inputCls;
const btnPrimary = 'px-4 py-2 rounded-lg text-sm font-medium transition-colors';

/* ── Step indicator ────────────────────────────────────────────────── */

function StepIndicator({ steps }: { steps: StepStatus[] }) {
  const colors: Record<StepStatus, string> = {
    idle: 'border-[#475569] text-[#64748b]',
    running: 'border-[#3b82f6] text-[#3b82f6] animate-pulse',
    done: 'border-[#22c55e] text-[#22c55e]',
    error: 'border-[#ef4444] text-[#ef4444]',
    skipped: 'border-[#94a3b8] text-[#94a3b8]',
  };
  const bgColors: Record<StepStatus, string> = {
    idle: 'bg-transparent',
    running: 'bg-[#3b82f6]/10',
    done: 'bg-[#22c55e]/10',
    error: 'bg-[#ef4444]/10',
    skipped: 'bg-[#94a3b8]/10',
  };

  return (
    <div className="flex items-center gap-2 mb-6">
      {steps.map((status, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border-2 ${colors[status]} ${bgColors[status]}`}>
            <span className="text-base">{STEP_ICONS[i]}</span>
            <span className="text-sm font-medium">{STEP_LABELS[i]}</span>
            {status === 'done' && <span className="text-xs">✓</span>}
            {status === 'error' && <span className="text-xs">✗</span>}
            {status === 'skipped' && <span className="text-xs">⏭</span>}
          </div>
          {i < steps.length - 1 && (
            <div className={`w-8 h-0.5 ${status === 'done' ? 'bg-[#22c55e]' : 'bg-[#334155]'}`} />
          )}
        </div>
      ))}
    </div>
  );
}

/* ── Main Pipeline Page ────────────────────────────────────────────── */

export default function Pipeline() {
  // Shared config
  const [shared, setShared] = useState<SharedConfig>({
    symbols: '600519',
    start_date: '2023-01-01',
    end_date: '2024-12-31',
    data_source: 'akshare',
  });

  // Per-step configs
  const [mining, setMining] = useState<MiningConfig>({
    method: 'parametric',
    n_bars: 3000,
    horizon: 5,
    ic_threshold: 0.02,
    top_n: 30,
    pop_size: 200,
    generations: 30,
    max_depth: 6,
  });

  const [train, setTrain] = useState<TrainConfig>({
    algorithms: 'lgb',
    threshold: 0.01,
  });

  const [bt, setBt] = useState<BacktestConfig>({
    strategy: 'ml_factor',
    capital: 1000000,
    period: 'daily',
  });

  // Step statuses (4 steps now)
  const [stepStatus, setStepStatus] = useState<StepStatus[]>(['idle', 'idle', 'idle', 'idle']);

  // Task managers: step 0 (data sync), step 1 (mining), step 2 (training)
  const tmSync = useTaskManager('pipeline_sync');
  const tmMine = useTaskManager('pipeline_mine');
  const tmTrain = useTaskManager('pipeline_train');

  // Cache status for Step 0
  interface CacheInfo { symbol: string; bar_count: number; min_date: string; max_date: string }
  const [cacheInfo, setCacheInfo] = useState<CacheInfo[]>([]);
  const [cacheLoading, setCacheLoading] = useState(false);

  // Step 3 uses backtest polling
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [btResult, setBtResult] = useState<any>(null);
  const [btError, setBtError] = useState('');
  const btPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const autoRef = useRef(false);

  // Which steps to include in pipeline
  const [enableStep, setEnableStep] = useState([true, true, true, true]);

  // Logs for each step
  const [stepLogs, setStepLogs] = useState<string[]>(['', '', '', '']);

  const updateStatus = (idx: number, s: StepStatus) => {
    setStepStatus(prev => { const n = [...prev]; n[idx] = s; return n; });
  };
  const appendLog = (idx: number, msg: string) => {
    setStepLogs(prev => { const n = [...prev]; n[idx] = (n[idx] ? n[idx] + '\n' : '') + msg; return n; });
  };

  /* ── Step 0: Data Fetch / Cache Check ──────────────────────────── */

  const checkCache = useCallback(async () => {
    setCacheLoading(true);
    try {
      const res = await getCacheStatus();
      setCacheInfo(res.symbols || []);
    } catch { /* ignore */ }
    setCacheLoading(false);
  }, []);

  // Check cache on mount and when symbols change
  useEffect(() => { checkCache(); }, [checkCache]);

  const runDataSync = useCallback(async () => {
    updateStatus(0, 'running');
    setStepLogs(prev => { const n = [...prev]; n[0] = ''; return n; });

    const syms = shared.symbols.split(',').map(s => s.trim()).filter(Boolean);
    await tmSync.submit(() => syncData(syms, shared.start_date, shared.end_date));
  }, [shared, tmSync]);

  // Watch sync task completion
  useEffect(() => {
    if (tmSync.task?.status === 'Completed') {
      updateStatus(0, 'done');
      appendLog(0, '✅ 数据准备完成');
      checkCache(); // Refresh cache info
    } else if (tmSync.task?.status === 'Failed') {
      updateStatus(0, 'error');
      appendLog(0, '❌ 数据同步失败: ' + (tmSync.error || ''));
      autoRef.current = false;
    }
  }, [tmSync.task?.status, tmSync.error, checkCache]);

  /* ── Step 1: Factor Mining ──────────────────────────────────────── */

  const runMining = useCallback(async () => {
    updateStatus(1, 'running');
    setStepLogs(prev => { const n = [...prev]; n[1] = ''; return n; });

    const params = {
      data_source: shared.data_source,
      symbols: shared.symbols,
      start_date: shared.start_date,
      end_date: shared.end_date,
      n_bars: mining.n_bars,
      horizon: mining.horizon,
    };

    if (mining.method === 'parametric') {
      await tmMine.submit(() => factorMineParametric({
        ...params,
        ic_threshold: mining.ic_threshold,
        top_n: mining.top_n,
      }));
    } else {
      await tmMine.submit(() => factorMineGP({
        ...params,
        pop_size: mining.pop_size,
        generations: mining.generations,
        max_depth: mining.max_depth,
      }));
    }
  }, [shared, mining, tmMine]);

  // Watch mining task completion
  useEffect(() => {
    if (tmMine.task?.status === 'Completed') {
      updateStatus(1, 'done');
      appendLog(1, '✅ 因子挖掘完成');
    } else if (tmMine.task?.status === 'Failed') {
      updateStatus(1, 'error');
      appendLog(1, '❌ 因子挖掘失败: ' + (tmMine.error || ''));
      autoRef.current = false;
    }
  }, [tmMine.task?.status, tmMine.error]);

  /* ── Step 2: ML Training ────────────────────────────────────────── */

  const runTraining = useCallback(async () => {
    updateStatus(2, 'running');
    setStepLogs(prev => { const n = [...prev]; n[2] = ''; return n; });

    await tmTrain.submit(() => mlRetrain({
      algorithms: train.algorithms,
      data_source: shared.data_source,
      symbols: shared.symbols,
      start_date: shared.start_date,
      end_date: shared.end_date,
      horizon: mining.horizon,
      threshold: train.threshold,
    }));
  }, [shared, mining.horizon, train, tmTrain]);

  // Watch training task completion
  useEffect(() => {
    if (tmTrain.task?.status === 'Completed') {
      updateStatus(2, 'done');
      appendLog(2, '✅ 模型训练完成');
    } else if (tmTrain.task?.status === 'Failed') {
      updateStatus(2, 'error');
      appendLog(2, '❌ 模型训练失败: ' + (tmTrain.error || ''));
      autoRef.current = false;
    }
  }, [tmTrain.task?.status, tmTrain.error]);

  /* ── Step 3: Backtest ───────────────────────────────────────────── */

  const runBacktestStep = useCallback(async () => {
    updateStatus(3, 'running');
    setBtResult(null);
    setBtError('');
    setStepLogs(prev => { const n = [...prev]; n[3] = ''; return n; });

    try {
      const syms = shared.symbols.split(',').map(s => s.trim()).filter(Boolean);
      const primary = syms[0];
      const extra = syms.length > 1 ? syms.slice(1) : undefined;

      const res = await runBacktest({
        strategy: bt.strategy,
        symbol: primary,
        start: shared.start_date,
        end: shared.end_date,
        capital: bt.capital,
        period: bt.period,
        symbols: extra,
      });

      // Start polling
      if (btPollRef.current) clearInterval(btPollRef.current);
      btPollRef.current = setInterval(async () => {
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const r: any = await getBacktestResults(res.task_id);
          if (r.status === 'completed' || r.status === 'Completed') {
            if (btPollRef.current) clearInterval(btPollRef.current);
            setBtResult(r);
            updateStatus(3, 'done');
            appendLog(3, '✅ 回测完成');
          } else if (r.status === 'failed' || r.status === 'Failed') {
            if (btPollRef.current) clearInterval(btPollRef.current);
            setBtError(r.error || '回测失败');
            updateStatus(3, 'error');
            autoRef.current = false;
          }
        } catch { /* continue polling */ }
      }, 1500);
    } catch (e: any) {
      setBtError(e.message || '回测请求失败');
      updateStatus(3, 'error');
      autoRef.current = false;
    }
  }, [shared, bt]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => { if (btPollRef.current) clearInterval(btPollRef.current); };
  }, []);

  /* ── Auto pipeline: chain steps ─────────────────────────────────── */

  useEffect(() => {
    if (!autoRef.current) return;

    // Step 0 done → start step 1 (or skip)
    if (stepStatus[0] === 'done' && stepStatus[1] === 'idle') {
      if (enableStep[1]) {
        runMining();
      } else {
        updateStatus(1, 'skipped');
      }
    }

    // Step 0 skipped → start step 1
    if (stepStatus[0] === 'skipped' && stepStatus[1] === 'idle') {
      if (enableStep[1]) {
        runMining();
      } else {
        updateStatus(1, 'skipped');
      }
    }

    // Step 1 done → start step 2 (or skip)
    if (stepStatus[1] === 'done' && stepStatus[2] === 'idle') {
      if (enableStep[2]) {
        runTraining();
      } else {
        updateStatus(2, 'skipped');
      }
    }

    // Step 1 skipped → start step 2
    if (stepStatus[1] === 'skipped' && stepStatus[2] === 'idle') {
      if (enableStep[2]) {
        runTraining();
      } else {
        updateStatus(2, 'skipped');
      }
    }

    // Step 2 done/skipped → start step 3 (or skip)
    if ((stepStatus[2] === 'done' || stepStatus[2] === 'skipped') && stepStatus[3] === 'idle') {
      if (enableStep[3]) {
        runBacktestStep();
      } else {
        updateStatus(3, 'skipped');
        autoRef.current = false;
      }
    }

    // All done
    if (stepStatus[3] === 'done' || stepStatus[3] === 'skipped') {
      autoRef.current = false;
    }
  }, [stepStatus, enableStep, runMining, runTraining, runBacktestStep]);

  const runFullPipeline = useCallback(() => {
    setStepStatus(['idle', 'idle', 'idle', 'idle']);
    setStepLogs(['', '', '', '']);
    setBtResult(null);
    setBtError('');
    autoRef.current = true;

    if (enableStep[0]) {
      runDataSync();
    } else {
      updateStatus(0, 'skipped');
    }
  }, [enableStep, runDataSync]);

  const isAnyRunning = stepStatus.some(s => s === 'running');

  /* ── Backtest result mini display ───────────────────────────────── */

  const BtSummary = () => {
    if (!btResult) return null;
    const r = btResult;
    const metrics = [
      { label: '总收益', value: `${r.total_return_percent ?? 0}%`, color: (r.total_return_percent ?? 0) >= 0 ? '#22c55e' : '#ef4444' },
      { label: '年化', value: `${r.annual_return_percent ?? 0}%`, color: (r.annual_return_percent ?? 0) >= 0 ? '#22c55e' : '#ef4444' },
      { label: '最大回撤', value: `${r.max_drawdown_percent ?? 0}%`, color: '#f59e0b' },
      { label: '胜率', value: `${r.win_rate_percent ?? 0}%`, color: '#3b82f6' },
      { label: '交易次数', value: `${r.total_trades ?? 0}`, color: '#94a3b8' },
      { label: '夏普', value: `${r.sharpe_ratio ?? 0}`, color: '#a78bfa' },
    ];

    return (
      <div className="mt-3 rounded-xl border border-[#334155] bg-[#0f172a] p-4">
        <h4 className="text-sm font-bold text-[#f8fafc] mb-3">📊 回测结果</h4>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
          {metrics.map(m => (
            <div key={m.label} className="text-center">
              <div className="text-xs text-[#94a3b8]">{m.label}</div>
              <div className="text-lg font-bold" style={{ color: m.color }}>{m.value}</div>
            </div>
          ))}
        </div>
        {r.per_symbol_results && r.per_symbol_results.length > 0 && (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-xs text-[#cbd5e1]">
              <thead>
                <tr className="border-b border-[#334155]">
                  <th className="text-left py-1">股票</th>
                  <th className="text-right py-1">收益%</th>
                  <th className="text-right py-1">夏普</th>
                  <th className="text-right py-1">回撤%</th>
                  <th className="text-right py-1">交易</th>
                </tr>
              </thead>
              <tbody>
                {r.per_symbol_results.filter((s: any) => s.status === 'completed').map((s: any) => (
                  <tr key={s.symbol} className="border-b border-[#1e293b]">
                    <td className="py-1 font-medium">{s.symbol}</td>
                    <td className="py-1 text-right" style={{ color: (s.total_return_percent ?? 0) >= 0 ? '#22c55e' : '#ef4444' }}>
                      {s.total_return_percent}%
                    </td>
                    <td className="py-1 text-right">{s.sharpe_ratio}</td>
                    <td className="py-1 text-right text-[#f59e0b]">{s.max_drawdown_percent}%</td>
                    <td className="py-1 text-right">{s.total_trades}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#f8fafc]">🚀 量化流水线</h1>
          <p className="text-sm text-[#94a3b8] mt-1">数据准备 → 因子挖掘 → ML训练 → 策略回测，一键运行</p>
        </div>
        <button
          onClick={runFullPipeline}
          disabled={isAnyRunning}
          className={`${btnPrimary} ${isAnyRunning
            ? 'bg-[#475569] text-[#94a3b8] cursor-not-allowed'
            : 'bg-[#3b82f6] text-white hover:bg-[#2563eb]'
          }`}
        >
          {isAnyRunning ? '⏳ 运行中...' : '▶ 一键运行'}
        </button>
      </div>

      {/* Step indicator */}
      <StepIndicator steps={stepStatus} />

      {/* Shared Config Panel */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h3 className="text-sm font-bold text-[#f8fafc] mb-3">📋 共享配置</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <InputField label="股票代码 (逗号分隔)">
            <input className={inputCls} value={shared.symbols}
              onChange={e => setShared(p => ({ ...p, symbols: e.target.value }))} />
          </InputField>
          <InputField label="开始日期">
            <input type="date" className={inputCls} value={shared.start_date}
              onChange={e => setShared(p => ({ ...p, start_date: e.target.value }))} />
          </InputField>
          <InputField label="结束日期">
            <input type="date" className={inputCls} value={shared.end_date}
              onChange={e => setShared(p => ({ ...p, end_date: e.target.value }))} />
          </InputField>
          <InputField label="数据源">
            <select className={selectCls} value={shared.data_source}
              onChange={e => setShared(p => ({ ...p, data_source: e.target.value as any }))}>
              <option value="akshare">akshare (真实数据)</option>
              <option value="synthetic">synthetic (模拟)</option>
            </select>
          </InputField>
        </div>
        <div className="flex gap-2 mt-3">
          {SYMBOL_PRESETS.map(p => (
            <button key={p.value} onClick={() => {
              const syms = shared.symbols.split(',').map(s => s.trim()).filter(Boolean);
              if (!syms.includes(p.value)) {
                setShared(prev => ({ ...prev, symbols: [...syms, p.value].join(',') }));
              }
            }}
              className="px-2 py-1 rounded text-xs bg-[#334155] text-[#94a3b8] hover:bg-[#475569] hover:text-[#f8fafc]">
              +{p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Four step panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-4">

        {/* ── Step 0: Data Fetch ─────────────────────────────────── */}
        <div className={`rounded-xl border p-4 ${stepStatus[0] === 'running' ? 'border-[#3b82f6]' : 'border-[#334155]'} bg-[#1e293b]`}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-bold text-[#f8fafc]">📡 Step 0: 数据准备</h3>
            <label className="flex items-center gap-1.5 text-xs text-[#94a3b8]">
              <input type="checkbox" checked={enableStep[0]}
                onChange={e => setEnableStep(p => { const n = [...p]; n[0] = e.target.checked; return n; })} />
              启用
            </label>
          </div>
          <div className="space-y-3">
            {/* Cache status display */}
            <div className="text-xs text-[#94a3b8]">
              {cacheLoading ? (
                <span>⏳ 检查缓存...</span>
              ) : cacheInfo.length > 0 ? (
                <div className="space-y-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-[#f8fafc]">缓存状态</span>
                    <button onClick={checkCache} className="text-[#3b82f6] hover:underline text-xs">🔄 刷新</button>
                  </div>
                  {(() => {
                    const syms = shared.symbols.split(',').map(s => s.trim()).filter(Boolean);
                    return syms.map(sym => {
                      const info = cacheInfo.find(c => c.symbol === sym || c.symbol === sym.split('.')[0]);
                      return (
                        <div key={sym} className="flex items-center justify-between py-0.5">
                          <span className="font-mono">{sym}</span>
                          {info ? (
                            <span className="text-[#22c55e]">✓ {info.bar_count}条 {info.min_date}~{info.max_date}</span>
                          ) : (
                            <span className="text-[#f59e0b]">⚠ 无缓存</span>
                          )}
                        </div>
                      );
                    });
                  })()}
                </div>
              ) : (
                <span className="text-[#f59e0b]">⚠ 无缓存数据，请先同步</span>
              )}
            </div>
            <button onClick={runDataSync} disabled={isAnyRunning || !enableStep[0]}
              className={`w-full ${btnPrimary} ${isAnyRunning || !enableStep[0] ? 'bg-[#334155] text-[#64748b]' : 'bg-[#0891b2] text-white hover:bg-[#0e7490]'}`}>
              📡 同步数据
            </button>
          </div>
          <TaskOutput running={tmSync.running} error={tmSync.error} output={stepLogs[0] || tmSync.output}
            progress={tmSync.progress} runningText="数据同步中..." />
        </div>

        {/* ── Step 1: Factor Mining ──────────────────────────────── */}
        <div className={`rounded-xl border p-4 ${stepStatus[1] === 'running' ? 'border-[#3b82f6]' : 'border-[#334155]'} bg-[#1e293b]`}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-bold text-[#f8fafc]">🔬 Step 1: 因子挖掘</h3>
            <label className="flex items-center gap-1.5 text-xs text-[#94a3b8]">
              <input type="checkbox" checked={enableStep[1]}
                onChange={e => setEnableStep(p => { const n = [...p]; n[1] = e.target.checked; return n; })} />
              启用
            </label>
          </div>
          <div className="space-y-3">
            <InputField label="挖掘方法">
              <select className={selectCls} value={mining.method}
                onChange={e => setMining(p => ({ ...p, method: e.target.value as MiningMethod }))}>
                <option value="parametric">参数化搜索</option>
                <option value="gp">遗传编程 (GP)</option>
              </select>
            </InputField>
            <div className="grid grid-cols-2 gap-2">
              <InputField label="K线数">
                <input type="number" className={inputCls} value={mining.n_bars}
                  onChange={e => setMining(p => ({ ...p, n_bars: +e.target.value }))} />
              </InputField>
              <InputField label="预测周期">
                <input type="number" className={inputCls} value={mining.horizon}
                  onChange={e => setMining(p => ({ ...p, horizon: +e.target.value }))} />
              </InputField>
            </div>
            {mining.method === 'parametric' ? (
              <div className="grid grid-cols-2 gap-2">
                <InputField label="IC阈值">
                  <input type="number" step="0.01" className={inputCls} value={mining.ic_threshold}
                    onChange={e => setMining(p => ({ ...p, ic_threshold: +e.target.value }))} />
                </InputField>
                <InputField label="Top N">
                  <input type="number" className={inputCls} value={mining.top_n}
                    onChange={e => setMining(p => ({ ...p, top_n: +e.target.value }))} />
                </InputField>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-2">
                <InputField label="种群">
                  <input type="number" className={inputCls} value={mining.pop_size}
                    onChange={e => setMining(p => ({ ...p, pop_size: +e.target.value }))} />
                </InputField>
                <InputField label="代数">
                  <input type="number" className={inputCls} value={mining.generations}
                    onChange={e => setMining(p => ({ ...p, generations: +e.target.value }))} />
                </InputField>
                <InputField label="深度">
                  <input type="number" className={inputCls} value={mining.max_depth}
                    onChange={e => setMining(p => ({ ...p, max_depth: +e.target.value }))} />
                </InputField>
              </div>
            )}
            <button onClick={runMining} disabled={isAnyRunning || !enableStep[1]}
              className={`w-full ${btnPrimary} ${isAnyRunning || !enableStep[1] ? 'bg-[#334155] text-[#64748b]' : 'bg-[#0f766e] text-white hover:bg-[#0d9488]'}`}>
              🔬 运行因子挖掘
            </button>
          </div>
          <TaskOutput running={tmMine.running} error={tmMine.error} output={stepLogs[1] || tmMine.output}
            progress={tmMine.progress} runningText="因子挖掘中..." />
        </div>

        {/* ── Step 2: ML Training ───────────────────────────────── */}
        <div className={`rounded-xl border p-4 ${stepStatus[2] === 'running' ? 'border-[#3b82f6]' : 'border-[#334155]'} bg-[#1e293b]`}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-bold text-[#f8fafc]">🧠 Step 2: ML 训练</h3>
            <label className="flex items-center gap-1.5 text-xs text-[#94a3b8]">
              <input type="checkbox" checked={enableStep[2]}
                onChange={e => setEnableStep(p => { const n = [...p]; n[2] = e.target.checked; return n; })} />
              启用
            </label>
          </div>
          <div className="space-y-3">
            <InputField label="算法">
              <select className={selectCls} value={train.algorithms}
                onChange={e => setTrain(p => ({ ...p, algorithms: e.target.value }))}>
                <option value="lgb">LightGBM</option>
                <option value="xgb">XGBoost</option>
                <option value="catboost">CatBoost</option>
                <option value="lstm">LSTM</option>
                <option value="transformer">Transformer</option>
              </select>
            </InputField>
            <InputField label="IC阈值">
              <input type="number" step="0.01" className={inputCls} value={train.threshold}
                onChange={e => setTrain(p => ({ ...p, threshold: +e.target.value }))} />
            </InputField>
            <button onClick={runTraining} disabled={isAnyRunning || !enableStep[2]}
              className={`w-full ${btnPrimary} ${isAnyRunning || !enableStep[2] ? 'bg-[#334155] text-[#64748b]' : 'bg-[#7c3aed] text-white hover:bg-[#6d28d9]'}`}>
              🧠 运行训练
            </button>
          </div>
          <TaskOutput running={tmTrain.running} error={tmTrain.error} output={stepLogs[2] || tmTrain.output}
            progress={tmTrain.progress} runningText="模型训练中..." />
        </div>

        {/* ── Step 3: Backtest ──────────────────────────────────── */}
        <div className={`rounded-xl border p-4 ${stepStatus[3] === 'running' ? 'border-[#3b82f6]' : 'border-[#334155]'} bg-[#1e293b]`}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-bold text-[#f8fafc]">📊 Step 3: 策略回测</h3>
            <label className="flex items-center gap-1.5 text-xs text-[#94a3b8]">
              <input type="checkbox" checked={enableStep[3]}
                onChange={e => setEnableStep(p => { const n = [...p]; n[3] = e.target.checked; return n; })} />
              启用
            </label>
          </div>
          <div className="space-y-3">
            <InputField label="策略">
              <select className={selectCls} value={bt.strategy}
                onChange={e => setBt(p => ({ ...p, strategy: e.target.value }))}>
                {STRATEGIES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </InputField>
            <div className="grid grid-cols-2 gap-2">
              <InputField label="初始资金">
                <input type="number" className={inputCls} value={bt.capital}
                  onChange={e => setBt(p => ({ ...p, capital: +e.target.value }))} />
              </InputField>
              <InputField label="K线周期">
                <select className={selectCls} value={bt.period}
                  onChange={e => setBt(p => ({ ...p, period: e.target.value }))}>
                  <option value="daily">日线</option>
                  <option value="60">60分钟</option>
                  <option value="30">30分钟</option>
                  <option value="15">15分钟</option>
                </select>
              </InputField>
            </div>
            <button onClick={runBacktestStep} disabled={isAnyRunning || !enableStep[3]}
              className={`w-full ${btnPrimary} ${isAnyRunning || !enableStep[3] ? 'bg-[#334155] text-[#64748b]' : 'bg-[#ea580c] text-white hover:bg-[#c2410c]'}`}>
              📊 运行回测
            </button>
          </div>
          {stepStatus[3] === 'running' && (
            <div className="text-xs text-[#94a3b8] mt-3">⏳ 回测运行中...</div>
          )}
          {btError && (
            <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-sm text-red-400 mt-3">{btError}</div>
          )}
          {stepLogs[3] && <div className="text-xs text-[#22c55e] mt-2">{stepLogs[3]}</div>}
        </div>
      </div>

      {/* Backtest Results */}
      <BtSummary />

      {/* Pipeline summary when all done */}
      {stepStatus.every(s => s === 'done' || s === 'skipped') && stepStatus.some(s => s === 'done') && (
        <div className="rounded-xl border border-[#22c55e]/30 bg-[#22c55e]/5 p-4 text-center">
          <span className="text-lg">🎉</span>
          <span className="text-sm text-[#22c55e] ml-2 font-medium">
            流水线完成！{enableStep.filter((e, i) => e && stepStatus[i] === 'done').length} 个步骤已执行
          </span>
        </div>
      )}
    </div>
  );
}
