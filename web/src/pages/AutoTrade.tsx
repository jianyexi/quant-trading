import { useState, useEffect, useRef } from 'react';
import { Play, Square, RefreshCw, Activity, TrendingUp, TrendingDown, AlertCircle, Loader2 } from 'lucide-react';
import { tradeStart, tradeStop, tradeStatus, type TradeStatus } from '../api/client';

const STRATEGIES = [
  { value: 'sma_cross', label: 'SMA 交叉 (5/20)', desc: '双均线金叉/死叉' },
  { value: 'rsi_reversal', label: 'RSI 反转 (14)', desc: '超买超卖反转' },
  { value: 'macd_trend', label: 'MACD 动量 (12/26/9)', desc: 'MACD柱状图交叉' },
];

const STOCK_PRESETS = [
  { symbols: ['600519.SH'], label: '贵州茅台' },
  { symbols: ['000858.SZ'], label: '五粮液' },
  { symbols: ['000001.SZ', '600036.SH'], label: '银行股组合' },
  { symbols: ['600519.SH', '000858.SZ'], label: '白酒组合' },
  { symbols: ['300750.SZ', '002594.SZ'], label: '新能源组合' },
  { symbols: ['000001.SZ', '600036.SH', '601318.SH'], label: '金融三巨头' },
];

export default function AutoTrade() {
  const [status, setStatus] = useState<TradeStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [strategy, setStrategy] = useState('sma_cross');
  const [symbolsInput, setSymbolsInput] = useState('000001.SZ,600036.SH');
  const [interval, setInterval_] = useState(5);
  const [positionSize, setPositionSize] = useState(0.15);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch initial status
  useEffect(() => {
    fetchStatus();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

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
      await tradeStart({ strategy, symbols, interval: interval, position_size: positionSize });
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#f8fafc]">🤖 自动交易</h1>
          <p className="text-sm text-[#94a3b8] mt-1">Actor模型引擎 — 数据→策略→风控→下单</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${
            isRunning ? 'bg-green-500/20 text-green-400' : 'bg-[#334155] text-[#94a3b8]'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-400 animate-pulse' : 'bg-[#94a3b8]'}`} />
            {isRunning ? '运行中' : '已停止'}
          </span>
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

      {/* Config & Control Panel */}
      <div className="grid grid-cols-3 gap-6">
        {/* Left: Configuration */}
        <div className="col-span-2 bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
          <h2 className="text-sm font-semibold text-[#f8fafc] mb-4">交易配置</h2>
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
