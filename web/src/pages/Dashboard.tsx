import { useState, useEffect, useCallback } from 'react';
import {
  DollarSign, TrendingUp, TrendingDown, BarChart3, Target, Loader2,
  Activity, ShieldCheck, AlertTriangle, Zap, RefreshCw,
} from 'lucide-react';
import { getDashboard } from '../api/client';

interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  unrealized_pnl: number;
  pnl_pct: number;
}

interface Trade {
  time: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  status: string;
}

interface Snapshot {
  date: string;
  value: number;
  pnl: number;
}

interface JournalEntry {
  time: string;
  type: string;
  symbol: string;
  side: string;
  price: number;
  quantity: number;
  reason: string;
}

interface DashboardData {
  portfolio_value: number;
  initial_capital: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  open_positions: number;
  win_rate: number;
  total_return_pct: number;
  drawdown_pct: number;
  max_drawdown_pct: number;
  profit_factor: number;
  avg_trade_pnl: number;
  wins: number;
  losses: number;
  engine_running: boolean;
  strategy: string;
  symbols: string[];
  total_signals: number;
  total_orders: number;
  total_fills: number;
  total_rejected: number;
  pnl: number;
  risk_daily_paused: boolean;
  risk_circuit_open: boolean;
  risk_drawdown_halted: boolean;
  pipeline_latency_us: number;
  recent_trades: Trade[];
  positions: Position[];
  snapshots: Snapshot[];
  journal: JournalEntry[];
}

function fmt(v: number | undefined | null): string {
  return (v ?? 0).toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function EquityCurve({ snapshots }: { snapshots: Snapshot[] }) {
  if (snapshots.length < 2) return <div className="text-[#64748b] text-sm py-8 text-center">暂无净值数据（引擎运行后每日自动记录）</div>;
  const values = snapshots.map(s => s.value);
  const min = Math.min(...values) * 0.998;
  const max = Math.max(...values) * 1.002;
  const range = max - min || 1;
  const w = 500, h = 120;
  const points = values.map((v, i) => `${(i / (values.length - 1)) * w},${h - ((v - min) / range) * h}`).join(' ');
  const fillPoints = `0,${h} ${points} ${w},${h}`;
  const lastVal = values[values.length - 1];
  const firstVal = values[0];
  const color = lastVal >= firstVal ? '#22c55e' : '#ef4444';

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height: 120 }}>
        <defs>
          <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <polygon points={fillPoints} fill="url(#eqGrad)" />
        <polyline points={points} fill="none" stroke={color} strokeWidth="2" />
      </svg>
      <div className="flex justify-between text-xs text-[#64748b] mt-1 px-1">
        <span>{snapshots[0]?.date}</span>
        <span>{snapshots[snapshots.length - 1]?.date}</span>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setRefreshing(true);
      const d = await getDashboard();
      setData(d as DashboardData);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  if (error) return <div className="text-red-400 p-6">Error: {error}</div>;
  if (!data) return <div className="flex items-center justify-center h-64 text-slate-400"><Loader2 className="animate-spin mr-2" size={20} /> Loading...</div>;

  const riskAlert = data.risk_daily_paused || data.risk_circuit_open || data.risk_drawdown_halted;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-[#f8fafc]">Dashboard</h1>
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
            data.engine_running
              ? 'bg-[#22c55e]/15 text-[#22c55e]'
              : 'bg-[#64748b]/15 text-[#64748b]'
          }`}>
            <span className={`w-2 h-2 rounded-full ${data.engine_running ? 'bg-[#22c55e] animate-pulse' : 'bg-[#64748b]'}`} />
            {data.engine_running ? `运行中 · ${data.strategy}` : '引擎未启动'}
          </span>
        </div>
        <button onClick={refresh} disabled={refreshing} className="p-2 rounded-lg hover:bg-[#334155] text-[#94a3b8] transition-colors">
          <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Risk Alerts */}
      {riskAlert && (
        <div className="flex items-center gap-2 rounded-lg bg-[#ef4444]/10 border border-[#ef4444]/30 px-4 py-3 text-sm text-[#ef4444]">
          <AlertTriangle size={16} />
          {data.risk_daily_paused && <span>每日亏损已暂停交易</span>}
          {data.risk_circuit_open && <span>熔断器已触发</span>}
          {data.risk_drawdown_halted && <span>最大回撤已停止引擎</span>}
        </div>
      )}

      {/* Primary Stats */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {[
          {
            label: '总资产', icon: DollarSign, color: '#3b82f6',
            value: `¥${fmt(data.portfolio_value)}`,
            sub: `初始 ¥${fmt(data.initial_capital)}`,
          },
          {
            label: '日盈亏', icon: data.daily_pnl >= 0 ? TrendingUp : TrendingDown,
            color: data.daily_pnl >= 0 ? '#22c55e' : '#ef4444',
            value: `${data.daily_pnl >= 0 ? '+' : ''}¥${fmt(data.daily_pnl)}`,
            sub: `${data.daily_pnl >= 0 ? '+' : ''}${data.daily_pnl_percent.toFixed(2)}%`,
          },
          {
            label: '累计收益', icon: BarChart3,
            color: data.total_return_pct >= 0 ? '#22c55e' : '#ef4444',
            value: `${data.total_return_pct >= 0 ? '+' : ''}${data.total_return_pct.toFixed(2)}%`,
            sub: `PnL ¥${fmt(data.pnl)}`,
          },
          {
            label: '持仓数', icon: Target, color: '#8b5cf6',
            value: String(data.open_positions),
            sub: data.symbols?.length ? `监控 ${data.symbols.length} 只` : '',
          },
        ].map(({ label, icon: Icon, color, value, sub }) => (
          <div key={label} className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
            <div className="flex items-center justify-between">
              <span className="text-sm text-[#94a3b8]">{label}</span>
              <Icon className="h-5 w-5" style={{ color }} />
            </div>
            <p className="mt-2 text-2xl font-bold text-[#f8fafc]">{value}</p>
            <span className="text-xs text-[#64748b]">{sub}</span>
          </div>
        ))}
      </div>

      {/* Secondary Metrics + Equity Curve */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Metrics Grid */}
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5 lg:col-span-1">
          <h2 className="text-sm font-semibold text-[#94a3b8] mb-4 flex items-center gap-2">
            <Activity size={14} /> 绩效指标
          </h2>
          <div className="space-y-3">
            {[
              { label: '胜率', value: `${data.win_rate.toFixed(1)}%`, good: data.win_rate >= 50 },
              { label: '盈亏比', value: data.profit_factor === Infinity ? '∞' : data.profit_factor.toFixed(2), good: data.profit_factor >= 1.5 },
              { label: '最大回撤', value: `${data.max_drawdown_pct.toFixed(2)}%`, good: data.max_drawdown_pct < 10 },
              { label: '当前回撤', value: `${data.drawdown_pct.toFixed(2)}%`, good: data.drawdown_pct < 5 },
              { label: '平均盈亏', value: `¥${fmt(data.avg_trade_pnl)}`, good: data.avg_trade_pnl >= 0 },
              { label: '胜/负', value: `${data.wins} / ${data.losses}`, good: data.wins >= data.losses },
            ].map(({ label, value, good }) => (
              <div key={label} className="flex justify-between items-center">
                <span className="text-sm text-[#94a3b8]">{label}</span>
                <span className={`text-sm font-medium ${good ? 'text-[#22c55e]' : 'text-[#f59e0b]'}`}>{value}</span>
              </div>
            ))}
          </div>
          {/* Engine pipeline stats */}
          {data.engine_running && (
            <div className="mt-4 pt-4 border-t border-[#334155]">
              <h3 className="text-xs text-[#64748b] mb-2 flex items-center gap-1"><Zap size={12} /> 流水线</h3>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div><span className="text-[#64748b]">信号</span> <span className="text-[#f8fafc] font-mono">{data.total_signals}</span></div>
                <div><span className="text-[#64748b]">订单</span> <span className="text-[#f8fafc] font-mono">{data.total_orders}</span></div>
                <div><span className="text-[#64748b]">成交</span> <span className="text-[#22c55e] font-mono">{data.total_fills}</span></div>
                <div><span className="text-[#64748b]">拒绝</span> <span className={`font-mono ${data.total_rejected > 0 ? 'text-[#ef4444]' : 'text-[#f8fafc]'}`}>{data.total_rejected}</span></div>
              </div>
              {data.pipeline_latency_us > 0 && (
                <div className="mt-2 text-xs text-[#64748b]">
                  延迟: <span className="text-[#f8fafc] font-mono">{data.pipeline_latency_us < 1000 ? `${data.pipeline_latency_us}µs` : `${(data.pipeline_latency_us / 1000).toFixed(1)}ms`}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Equity Curve */}
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5 lg:col-span-2">
          <h2 className="text-sm font-semibold text-[#94a3b8] mb-3">📈 净值曲线</h2>
          <EquityCurve snapshots={data.snapshots || []} />
        </div>
      </div>

      {/* Positions & Risk */}
      {data.positions && data.positions.length > 0 && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h2 className="mb-4 text-sm font-semibold text-[#94a3b8] flex items-center gap-2">
            <ShieldCheck size={14} /> 当前持仓
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-[#334155] text-[#64748b]">
                  <th className="pb-2 pr-4 font-medium">代码</th>
                  <th className="pb-2 pr-4 font-medium text-right">数量</th>
                  <th className="pb-2 pr-4 font-medium text-right">成本</th>
                  <th className="pb-2 pr-4 font-medium text-right">现价</th>
                  <th className="pb-2 pr-4 font-medium text-right">浮动盈亏</th>
                  <th className="pb-2 font-medium text-right">收益率</th>
                </tr>
              </thead>
              <tbody>
                {data.positions.map((p) => (
                  <tr key={p.symbol} className="border-b border-[#334155]/30 text-[#f8fafc]">
                    <td className="py-2.5 pr-4 font-mono text-[#3b82f6]">{p.symbol}</td>
                    <td className="py-2.5 pr-4 text-right">{p.quantity}</td>
                    <td className="py-2.5 pr-4 text-right">¥{p.avg_cost.toFixed(2)}</td>
                    <td className="py-2.5 pr-4 text-right">¥{p.current_price.toFixed(2)}</td>
                    <td className={`py-2.5 pr-4 text-right font-medium ${p.unrealized_pnl >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                      {p.unrealized_pnl >= 0 ? '+' : ''}¥{fmt(p.unrealized_pnl)}
                    </td>
                    <td className={`py-2.5 text-right font-medium ${p.pnl_pct >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                      {p.pnl_pct >= 0 ? '+' : ''}{p.pnl_pct.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent Trades & Journal */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Recent Trades */}
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h2 className="mb-4 text-sm font-semibold text-[#94a3b8]">🔄 最近成交</h2>
          {data.recent_trades.length === 0 ? (
            <div className="text-[#64748b] text-sm py-4 text-center">暂无成交记录</div>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {data.recent_trades.slice(0, 10).map((t, i) => (
                <div key={i} className="flex items-center justify-between text-sm py-1.5 border-b border-[#334155]/30 last:border-0">
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-[#64748b] font-mono w-16">{t.time}</span>
                    <span className="font-mono text-[#3b82f6]">{t.symbol}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      t.side === 'BUY' ? 'bg-[#22c55e]/15 text-[#22c55e]' : 'bg-[#ef4444]/15 text-[#ef4444]'
                    }`}>{t.side === 'BUY' ? '买' : '卖'}</span>
                  </div>
                  <div className="text-[#94a3b8] text-xs">
                    {t.quantity}股 × ¥{t.price.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Activity Feed */}
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h2 className="mb-4 text-sm font-semibold text-[#94a3b8]">📋 交易日志</h2>
          {(!data.journal || data.journal.length === 0) ? (
            <div className="text-[#64748b] text-sm py-4 text-center">暂无日志记录</div>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {data.journal.slice(0, 12).map((j, i) => (
                <div key={i} className="flex items-center justify-between text-sm py-1.5 border-b border-[#334155]/30 last:border-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[#64748b] font-mono w-16">{j.time}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      j.type === 'fill' ? 'bg-[#22c55e]/15 text-[#22c55e]' :
                      j.type === 'reject' ? 'bg-[#ef4444]/15 text-[#ef4444]' :
                      j.type === 'signal' ? 'bg-[#3b82f6]/15 text-[#3b82f6]' :
                      'bg-[#64748b]/15 text-[#64748b]'
                    }`}>{j.type}</span>
                    <span className="font-mono text-[#94a3b8] text-xs">{j.symbol}</span>
                  </div>
                  {j.reason && <span className="text-xs text-[#64748b] truncate max-w-32">{j.reason}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
