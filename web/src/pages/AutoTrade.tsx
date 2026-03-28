import { useState, useEffect, useRef } from 'react';
import { Play, Square, RefreshCw, Activity, TrendingUp, TrendingDown, AlertCircle, Loader2, Radio, FileText, BarChart3 } from 'lucide-react';
import { tradeStart, tradeStop, tradeStatus, qmtBridgeStatus, getJournal, type TradeStatus, type QmtBridgeStatus, type JournalEntry } from '../api/client';

const STRATEGIES = [
  { value: 'sma_cross', label: 'SMA 交叉 (5/20)', desc: '双均线金叉/死叉' },
  { value: 'rsi_reversal', label: 'RSI 反转 (14)', desc: '超买超卖反转' },
  { value: 'macd_trend', label: 'MACD 动量 (12/26/9)', desc: 'MACD柱状图交叉' },
  { value: 'multi_factor', label: '多因子模型', desc: '6因子综合评分 (趋势+动量+波动+KDJ+量价+价格行为)' },
  { value: 'sentiment_aware', label: '舆情增强策略', desc: '多因子+舆情数据联合决策，根据市场情绪调整信号' },
  { value: 'ml_factor', label: 'ML因子模型', desc: '24维特征工程 + GPU机器学习推理 (需启动Python推理服务)' },
];

const STOCK_PRESETS = [
  { symbols: ['600519.SH'], label: '贵州茅台' },
  { symbols: ['000858.SZ'], label: '五粮液' },
  { symbols: ['000001.SZ', '600036.SH'], label: '银行股组合' },
  { symbols: ['600519.SH', '000858.SZ'], label: '白酒组合' },
  { symbols: ['300750.SZ', '002594.SZ'], label: '新能源组合' },
  { symbols: ['000001.SZ', '600036.SH', '601318.SH'], label: '金融三巨头' },
  { symbols: ['AAPL', 'MSFT', 'GOOGL'], label: 'US Big Tech' },
  { symbols: ['NVDA', 'TSLA', 'AMZN'], label: 'US Growth' },
  { symbols: ['0700.HK', '9988.HK', '3690.HK'], label: 'HK Tech' },
];

export default function AutoTrade() {
  const [status, setStatus] = useState<TradeStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [strategy, setStrategy] = useState('sma_cross');
  const [symbolsInput, setSymbolsInput] = useState('000001.SZ,600036.SH');
  const [interval, setInterval_] = useState(5);
  const [positionSize, setPositionSize] = useState(0.15);
  const [mode, setMode] = useState<'paper' | 'live' | 'qmt' | 'replay' | 'l2'>('paper');
  const [qmtStatus, setQmtStatus] = useState<QmtBridgeStatus | null>(null);
  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([]);
  const [journalTotal, setJournalTotal] = useState(0);
  const [showJournal, setShowJournal] = useState(false);
  const [replayStart, setReplayStart] = useState('2024-01-01');
  const [replayEnd, setReplayEnd] = useState('2024-12-31');
  const [replaySpeed, setReplaySpeed] = useState(0);
  const [replayPeriod, setReplayPeriod] = useState('daily');
  const [slippageBps, setSlippageBps] = useState(5);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [deployNotice, setDeployNotice] = useState('');

  // ── Pre-fill from Backtest deploy ──────────────────────────────────
  useEffect(() => {
    const deployStr = localStorage.getItem('quant-deploy-config');
    if (deployStr) {
      try {
        const deploy = JSON.parse(deployStr);
        if (Date.now() - deploy.timestamp < 300000) {
          if (deploy.strategy) setStrategy(deploy.strategy);
          if (deploy.symbols) setSymbolsInput(deploy.symbols);
          setDeployNotice(`已从回测结果加载: ${deploy.strategy} / ${deploy.symbols}`);
        }
      } catch { /* ignore */ }
      localStorage.removeItem('quant-deploy-config');
    }
  }, []);

  // Fetch initial status
  useEffect(() => {
    fetchStatus();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Check QMT bridge status when mode changes to qmt
  useEffect(() => {
    if (mode === 'qmt') {
      qmtBridgeStatus().then(setQmtStatus).catch(() => setQmtStatus({ status: 'offline', message: 'Cannot reach bridge' }));
    }
  }, [mode]);

  const fetchStatus = async () => {
    try {
      const s = await tradeStatus();
      setStatus(s);
      return s;
    } catch {
      return null;
    }
  };

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(fetchStatus, 3000);
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const handleStart = async () => {
    setLoading(true);
    setError('');
    try {
      const symbols = symbolsInput.split(',').map(s => s.trim()).filter(Boolean);
      await tradeStart({
        strategy, symbols, interval: interval, position_size: positionSize, mode,
        ...(mode === 'live' || mode === 'paper' ? { slippage_bps: slippageBps } : {}),
        ...(mode === 'replay' ? { replay_start: replayStart, replay_end: replayEnd, replay_speed: replaySpeed, replay_period: replayPeriod } : {}),
      });
      await fetchStatus();
      startPolling();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    setError('');
    try {
      await tradeStop();
      stopPolling();
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to stop');
    } finally {
      setLoading(false);
    }
  };

  const isRunning = status?.running ?? false;

  const fetchJournal = async () => {
    try {
      const j = await getJournal({ limit: 50 });
      setJournalEntries(j?.entries ?? []);
      setJournalTotal(j?.total ?? 0);
    } catch { /* ignore */ }
  };

  const entryTypeLabel = (t: string) => {
    const map: Record<string, string> = {
      signal: '📡 信号', order_submitted: '📤 下单', order_filled: '✅ 成交',
      risk_rejected: '🚫 风控拒绝', engine_started: '🚀 启动', engine_stopped: '🛑 停止',
    };
    return map[t] || t;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#f8fafc]">{mode === 'qmt' ? '🔴' : mode === 'live' ? '📊' : '🤖'} 自动交易</h1>
          <p className="text-sm text-[#94a3b8] mt-1">
            {mode === 'qmt' ? 'QMT实盘引擎 — 数据→策略→风控→QMT下单' : mode === 'live' ? '模拟实盘 — 实时行情→策略→风控→模拟下单(含滑点)' : 'Actor模型引擎 — 数据→策略→风控→下单'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${
            isRunning ? 'bg-green-500/20 text-green-400' : 'bg-[#334155] text-[#94a3b8]'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-[#94a3b8]'}`} />
            {isRunning ? '运行中' : '已停止'}
          </span>
          {isRunning && <span className={`text-xs px-2 py-0.5 rounded ${
            mode === 'replay' ? 'bg-purple-500/20 text-purple-300' : 'bg-green-500/20 text-green-300'
          }`}>{mode === 'replay' ? '📂 历史回放' : '📡 真实行情'}</span>}
          <button onClick={fetchStatus}
            className="p-2 rounded-lg bg-[#334155] hover:bg-[#475569] text-[#94a3b8] transition-colors">
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm flex items-center gap-2">
          <AlertCircle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      {deployNotice && (
        <div className="mb-4 p-3 bg-green-900/50 border border-green-700 rounded-lg text-green-300 flex items-center justify-between">
          <span>✅ {deployNotice}</span>
          <button onClick={() => setDeployNotice('')} className="text-green-400 hover:text-white">✕</button>
        </div>
      )}

      {/* Config & Control Panel */}
      <div className="grid grid-cols-3 gap-6">
        {/* Left: Configuration */}
        <div className="col-span-2 bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
          <h2 className="text-sm font-semibold text-[#f8fafc] mb-4">交易配置</h2>

          {/* Broker Mode Selector */}
          <div className="mb-4">
            <label className="block text-xs text-[#94a3b8] mb-2">交易模式</label>
            <div className="flex gap-3">
              <button onClick={() => setMode('paper')} disabled={isRunning}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                  mode === 'paper'
                    ? 'bg-blue-600/20 border-blue-500 text-blue-400'
                    : 'bg-[#0f172a] border-[#334155] text-[#94a3b8] hover:border-[#475569]'
                } disabled:opacity-50`}>
                <Activity className="h-4 w-4" /> 模拟交易
              </button>
              <button onClick={() => setMode('live')} disabled={isRunning}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                  mode === 'live'
                    ? 'bg-green-600/20 border-green-500 text-green-400'
                    : 'bg-[#0f172a] border-[#334155] text-[#94a3b8] hover:border-[#475569]'
                } disabled:opacity-50`}>
                <TrendingUp className="h-4 w-4" /> 模拟实盘
              </button>
              <button onClick={() => setMode('replay')} disabled={isRunning}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                  mode === 'replay'
                    ? 'bg-purple-600/20 border-purple-500 text-purple-400'
                    : 'bg-[#0f172a] border-[#334155] text-[#94a3b8] hover:border-[#475569]'
                } disabled:opacity-50`}>
                <FileText className="h-4 w-4" /> 历史回放
              </button>
              <button onClick={() => setMode('qmt')} disabled={isRunning}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                  mode === 'qmt'
                    ? 'bg-red-600/20 border-red-500 text-red-400'
                    : 'bg-[#0f172a] border-[#334155] text-[#94a3b8] hover:border-[#475569]'
                } disabled:opacity-50`}>
                <Radio className="h-4 w-4" /> QMT 实盘
              </button>
              <button onClick={() => setMode('l2')} disabled={isRunning}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${
                  mode === 'l2'
                    ? 'bg-cyan-600/20 border-cyan-500 text-cyan-400'
                    : 'bg-[#0f172a] border-[#334155] text-[#94a3b8] hover:border-[#475569]'
                } disabled:opacity-50`}>
                <BarChart3 className="h-4 w-4" /> L2 逐笔
              </button>
            </div>

            {mode === 'live' && (
              <div className="mt-3 p-3 rounded-lg bg-green-500/5 border border-green-500/20">
                <div className="text-xs text-green-300 mb-2">📊 模拟实盘 — 使用AKShare实时行情，PaperBroker模拟下单</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-[#94a3b8] mb-1">滑点 (基点, 1bp=0.01%)</label>
                    <input type="number" min={0} max={50} value={slippageBps}
                      onChange={e => setSlippageBps(Number(e.target.value))}
                      disabled={isRunning}
                      className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-1.5 text-sm text-[#f8fafc] disabled:opacity-50" />
                  </div>
                  <div className="flex items-end">
                    <p className="text-xs text-[#64748b]">
                      买入价格上浮 {(slippageBps / 100).toFixed(2)}%，卖出价格下浮 {(slippageBps / 100).toFixed(2)}%，模拟真实交易摩擦
                    </p>
                  </div>
                </div>
              </div>
            )}

            {mode === 'replay' && (
              <div className="mt-3 p-3 rounded-lg bg-purple-500/5 border border-purple-500/20">
                <div className="text-xs text-purple-300 mb-2">📂 历史数据回放配置</div>
                <div className="grid grid-cols-4 gap-3">
                  <div>
                    <label className="block text-xs text-[#94a3b8] mb-1">开始日期</label>
                    <input type="date" value={replayStart} onChange={e => setReplayStart(e.target.value)}
                      disabled={isRunning}
                      className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-2 py-1.5 text-sm text-[#f8fafc] disabled:opacity-50" />
                  </div>
                  <div>
                    <label className="block text-xs text-[#94a3b8] mb-1">结束日期</label>
                    <input type="date" value={replayEnd} onChange={e => setReplayEnd(e.target.value)}
                      disabled={isRunning}
                      className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-2 py-1.5 text-sm text-[#f8fafc] disabled:opacity-50" />
                  </div>
                  <div>
                    <label className="block text-xs text-[#94a3b8] mb-1">K线周期</label>
                    <select value={replayPeriod} onChange={e => setReplayPeriod(e.target.value)}
                      disabled={isRunning}
                      className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-2 py-1.5 text-sm text-[#f8fafc] disabled:opacity-50">
                      <option value="daily">日线</option>
                      <option value="60">60分钟</option>
                      <option value="30">30分钟</option>
                      <option value="15">15分钟</option>
                      <option value="5">5分钟</option>
                      <option value="1">1分钟</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-[#94a3b8] mb-1">回放速度</label>
                    <select value={replaySpeed} onChange={e => setReplaySpeed(Number(e.target.value))}
                      disabled={isRunning}
                      className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-2 py-1.5 text-sm text-[#f8fafc] disabled:opacity-50">
                      <option value={0}>极速 (无延迟)</option>
                      <option value={100}>100x 快速</option>
                      <option value={10}>10x 正常</option>
                      <option value={1}>1x 实时</option>
                    </select>
                  </div>
                </div>
                <div className="mt-2 text-xs text-[#64748b]">
                  使用真实历史K线数据回放，可重复验证策略表现。分钟数据仅支持近期（约5个交易日）。
                </div>
              </div>
            )}

            {mode === 'qmt' && (
              <div className={`mt-2 px-3 py-2 rounded-lg text-xs ${
                qmtStatus?.connected
                  ? 'bg-green-500/10 border border-green-500/30 text-green-400'
                  : 'bg-yellow-500/10 border border-yellow-500/30 text-yellow-400'
              }`}>
                {qmtStatus?.connected
                  ? `✅ QMT Bridge 已连接 (${qmtStatus.account || 'N/A'})`
                  : `⚠️ QMT Bridge ${qmtStatus?.status === 'offline' ? '离线' : '未连接'} — ${qmtStatus?.message || '请启动 qmt_bridge.py'}`
                }
              </div>
            )}

            {mode === 'qmt' && (
              <div className="mt-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-xs">
                ⚠️ 实盘模式将通过QMT发送真实委托，请确认风险！
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1.5">策略</label>
              <select value={strategy} onChange={e => setStrategy(e.target.value)} disabled={isRunning}
                className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] disabled:opacity-50">
                {STRATEGIES.map(s => (
                  <option key={s.value} value={s.value}>{s.label} — {s.desc}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1.5">数据间隔 (秒)</label>
              <input type="number" min={1} max={60} value={interval} disabled={isRunning}
                onChange={e => setInterval_(Number(e.target.value))}
                className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] disabled:opacity-50" />
            </div>
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1.5">股票代码 (逗号分隔)</label>
              <input type="text" value={symbolsInput} disabled={isRunning}
                onChange={e => setSymbolsInput(e.target.value)}
                className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] disabled:opacity-50" />
            </div>
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1.5">仓位比例</label>
              <div className="flex items-center gap-3">
                <input type="range" min={0.05} max={0.5} step={0.05} value={positionSize} disabled={isRunning}
                  onChange={e => setPositionSize(Number(e.target.value))}
                  className="flex-1" />
                <span className="text-sm text-[#f8fafc] font-mono w-12 text-right">{(positionSize * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          {/* Stock Presets */}
          <div className="mt-4">
            <label className="block text-xs text-[#94a3b8] mb-2">快速选股</label>
            <div className="flex flex-wrap gap-2">
              {STOCK_PRESETS.map(p => (
                <button key={p.label} onClick={() => setSymbolsInput(p.symbols.join(','))} disabled={isRunning}
                  className="px-3 py-1 rounded-lg bg-[#0f172a] border border-[#334155] text-xs text-[#94a3b8] hover:text-[#f8fafc] hover:border-[#3b82f6] transition-colors disabled:opacity-50">
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Start/Stop Buttons */}
          <div className="mt-5 flex gap-3">
            {!isRunning ? (
              <button onClick={handleStart} disabled={loading}
                className="flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:bg-[#334155] text-white font-medium rounded-lg px-6 py-2.5 text-sm transition-colors">
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                启动交易
              </button>
            ) : (
              <button onClick={handleStop} disabled={loading}
                className="flex items-center gap-2 bg-red-600 hover:bg-red-700 disabled:bg-[#334155] text-white font-medium rounded-lg px-6 py-2.5 text-sm transition-colors">
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
                停止交易
              </button>
            )}
          </div>
        </div>

        {/* Right: Live Stats */}
        <div className="space-y-4">
          <div className="bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
            <h2 className="text-sm font-semibold text-[#f8fafc] mb-4 flex items-center gap-1.5">
              <Activity className="h-4 w-4 text-[#3b82f6]" /> 实时统计
            </h2>
            <div className="space-y-3">
              <MetricRow label="信号" value={status?.total_signals ?? 0} />
              <MetricRow label="订单" value={status?.total_orders ?? 0} />
              <MetricRow label="成交" value={status?.total_fills ?? 0} />
              <MetricRow label="拒绝" value={status?.total_rejected ?? 0} color={
                (status?.total_rejected ?? 0) > 0 ? 'text-red-400' : undefined
              } />
              <div className="pt-2 border-t border-[#334155]">
                <div className="flex justify-between items-center">
                  <span className="text-[#94a3b8] text-sm">盈亏 (PnL)</span>
                  <span className={`text-lg font-bold ${
                    (status?.pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    ¥{(status?.pnl ?? 0).toLocaleString('zh-CN', { minimumFractionDigits: 2 })}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {isRunning && status && (
            <div className="bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
              <h3 className="text-xs text-[#94a3b8] mb-2">当前配置</h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-[#94a3b8]">策略</span>
                  <span className="text-[#f8fafc]">{status.strategy}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#94a3b8]">标的</span>
                  <span className="text-[#3b82f6] text-xs">{status.symbols?.join(', ')}</span>
                </div>
              </div>
            </div>
          )}

          {/* Pipeline Latency */}
          {isRunning && status?.latency && (
            <div className="bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
              <h3 className="text-xs text-[#94a3b8] mb-2">⏱️ 管道延迟</h3>
              <div className="space-y-1.5 text-sm font-mono">
                <div className="flex justify-between">
                  <span className="text-[#94a3b8]">因子计算</span>
                  <span className={`${(status.latency.last_factor_compute_us) > 1000 ? 'text-yellow-400' : 'text-green-400'}`}>
                    {status.latency.last_factor_compute_us}μs
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#94a3b8]">平均因子</span>
                  <span className="text-[#f8fafc]">{status.latency.avg_factor_compute_us}μs</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#94a3b8]">下单延迟</span>
                  <span className="text-[#f8fafc]">{status.latency.last_order_submit_us}μs</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#94a3b8]">已处理K线</span>
                  <span className="text-[#3b82f6]">{status.latency.total_bars_processed}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Trades */}
      {status && status.recent_trades && status.recent_trades.length > 0 && (
        <div className="bg-[#1e293b] rounded-xl border border-[#334155] overflow-hidden">
          <div className="px-5 py-3 border-b border-[#334155]">
            <h2 className="text-sm font-semibold text-[#f8fafc]">📋 最近成交</h2>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#334155] text-[#94a3b8]">
                <th className="px-4 py-2 text-left font-medium">方向</th>
                <th className="px-4 py-2 text-left font-medium">代码</th>
                <th className="px-4 py-2 text-right font-medium">数量</th>
                <th className="px-4 py-2 text-right font-medium">价格</th>
                <th className="px-4 py-2 text-right font-medium">手续费</th>
              </tr>
            </thead>
            <tbody>
              {status.recent_trades.map((t, i) => (
                <tr key={i} className="border-b border-[#334155]/50">
                  <td className="px-4 py-2">
                    <span className={`flex items-center gap-1 ${
                      t.side === 'Buy' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {t.side === 'Buy' ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
                      {t.side === 'Buy' ? '买入' : '卖出'}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-[#3b82f6] font-mono">{t.symbol}</td>
                  <td className="px-4 py-2 text-right text-[#f8fafc]">{t.quantity}</td>
                  <td className="px-4 py-2 text-right text-[#f8fafc]">¥{t.price.toFixed(2)}</td>
                  <td className="px-4 py-2 text-right text-[#94a3b8]">¥{t.commission.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Journal Toggle */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-4">
        <button
          onClick={() => { setShowJournal(!showJournal); if (!showJournal) fetchJournal(); }}
          className="flex items-center gap-2 text-sm font-medium text-[#f8fafc] hover:text-[#3b82f6]"
        >
          <FileText className="h-4 w-4" />
          📝 交易日志 {journalTotal > 0 && `(${journalTotal}条)`}
        </button>

        {showJournal && (
          <div className="mt-4">
            <div className="flex justify-between items-center mb-3">
              <span className="text-xs text-[#94a3b8]">最近50条记录</span>
              <button onClick={fetchJournal} className="text-xs text-[#3b82f6] hover:underline">刷新</button>
            </div>
            {journalEntries.length === 0 ? (
              <p className="text-sm text-[#64748b]">暂无交易日志</p>
            ) : (
              <div className="overflow-x-auto max-h-80 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-[#1e293b]">
                    <tr className="border-b border-[#334155]">
                      <th className="text-left py-1.5 px-2 text-[#94a3b8]">时间</th>
                      <th className="text-left py-1.5 px-2 text-[#94a3b8]">类型</th>
                      <th className="text-left py-1.5 px-2 text-[#94a3b8]">股票</th>
                      <th className="text-left py-1.5 px-2 text-[#94a3b8]">方向</th>
                      <th className="text-right py-1.5 px-2 text-[#94a3b8]">数量</th>
                      <th className="text-right py-1.5 px-2 text-[#94a3b8]">价格</th>
                      <th className="text-left py-1.5 px-2 text-[#94a3b8]">详情</th>
                    </tr>
                  </thead>
                  <tbody>
                    {journalEntries.map((e) => (
                      <tr key={e.id} className="border-b border-[#334155]/30 hover:bg-[#334155]/20">
                        <td className="py-1.5 px-2 text-[#94a3b8] font-mono">{e.timestamp}</td>
                        <td className="py-1.5 px-2">{entryTypeLabel(e.entry_type)}</td>
                        <td className="py-1.5 px-2 text-[#3b82f6]">{e.symbol || '—'}</td>
                        <td className={`py-1.5 px-2 ${e.side === 'BUY' ? 'text-green-400' : e.side === 'SELL' ? 'text-red-400' : 'text-[#94a3b8]'}`}>
                          {e.side || '—'}
                        </td>
                        <td className="py-1.5 px-2 text-right text-[#f8fafc]">{e.quantity?.toFixed(0) ?? '—'}</td>
                        <td className="py-1.5 px-2 text-right text-[#f8fafc]">{e.price ? `¥${e.price.toFixed(2)}` : '—'}</td>
                        <td className="py-1.5 px-2 text-[#64748b] max-w-[200px] truncate">{e.reason || e.details || e.status || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function MetricRow({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[#94a3b8] text-sm">{label}</span>
      <span className={`font-mono font-semibold ${color ?? 'text-[#f8fafc]'}`}>{value}</span>
    </div>
  );
}
