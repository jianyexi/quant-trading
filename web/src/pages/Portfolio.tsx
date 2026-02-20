import { useState, useEffect, useCallback } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Wallet, PiggyBank, BarChart3, Percent, Loader2, X, Archive } from 'lucide-react';
import { getPortfolio, closePosition, getClosedPositions } from '../api/client';

// --- Types ---

interface PositionRow {
  symbol: string;
  name: string;
  shares: number;
  avg_cost: number;
  current_price: number;
  pnl: number;
  pnl_pct?: number;
  entry_time?: string;
  holding_days?: number;
  scale_level?: number;
}

interface ClosedPositionRow {
  symbol: string;
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  realized_pnl: number;
  holding_days: number;
}

interface PortfolioData {
  total_value: number;
  cash: number;
  total_pnl: number;
  positions: PositionRow[];
}

// --- Helpers ---

function fmt(v: number): string {
  return v.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function computePositionStats(p: PositionRow) {
  const marketValue = p.shares * p.current_price;
  const cost = p.shares * p.avg_cost;
  const pnl = p.pnl;
  const pnlPercent = p.pnl_pct ?? (cost !== 0 ? (pnl / cost) * 100 : 0);
  return { ...p, marketValue, pnl, pnlPercent };
}

// --- Custom Pie Label ---

const PIE_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#64748b'];

interface LabelProps {
  cx: number;
  cy: number;
  midAngle: number;
  outerRadius: number;
  name: string;
  percent: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function renderLabel(props: any) {
  const { cx, cy, midAngle, outerRadius, name, percent } = props as LabelProps;
  const RADIAN = Math.PI / 180;
  const radius = outerRadius + 24;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  return (
    <text x={x} y={y} fill="#cbd5e1" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central" fontSize={12}>
      {name} {(percent * 100).toFixed(1)}%
    </text>
  );
}

// --- Pie Tooltip ---

interface PieTooltipPayload {
  name: string;
  value: number;
}

function PieTooltipContent({ active, payload }: { active?: boolean; payload?: { payload: PieTooltipPayload }[] }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-xs shadow-lg">
      <p className="font-medium text-slate-200">{d.name}</p>
      <p>¥{fmt(d.value)}</p>
    </div>
  );
}

// --- Summary Card ---

function SummaryCard({
  icon: Icon,
  label,
  value,
  colored,
}: {
  icon: typeof Wallet;
  label: string;
  value: string;
  colored?: 'green' | 'red';
}) {
  const colorClass = colored === 'green' ? 'text-[#22c55e]' : colored === 'red' ? 'text-[#ef4444]' : 'text-slate-100';
  return (
    <div className="flex-1 rounded-xl border border-slate-700 bg-[#1e293b] p-5">
      <div className="mb-2 flex items-center gap-2 text-sm text-slate-400">
        <Icon className="h-4 w-4" />
        {label}
      </div>
      <p className={`text-xl font-bold ${colorClass}`}>{value}</p>
    </div>
  );
}

// --- Main Component ---

export default function Portfolio() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [closedPositions, setClosedPositions] = useState<ClosedPositionRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'open' | 'closed'>('open');
  const [closingSymbol, setClosingSymbol] = useState<string | null>(null);

  const loadData = useCallback(() => {
    getPortfolio()
      .then((d) => setData(d as PortfolioData))
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load'));
    getClosedPositions()
      .then((d) => {
        const result = d as { closed_positions: ClosedPositionRow[] };
        setClosedPositions(result.closed_positions || []);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleClose = async (symbol: string) => {
    if (!confirm(`确认平仓 ${symbol}？`)) return;
    setClosingSymbol(symbol);
    try {
      await closePosition(symbol);
      loadData();
    } catch (e) {
      alert(e instanceof Error ? e.message : '平仓失败');
    } finally {
      setClosingSymbol(null);
    }
  };

  if (error) {
    return <div className="text-red-400 p-6">Error: {error}</div>;
  }
  if (!data) {
    return <div className="flex items-center justify-center h-64 text-slate-400"><Loader2 className="animate-spin mr-2" size={20} /> Loading...</div>;
  }

  const positionsWithStats = data.positions.map(computePositionStats);
  const totalCost = positionsWithStats.reduce((s, p) => s + p.shares * p.avg_cost, 0);
  const totalPnl = data.total_pnl;
  const totalPnlPercent = totalCost !== 0 ? (totalPnl / totalCost) * 100 : 0;

  const allocationData = [
    ...positionsWithStats.map((p) => ({ name: p.name || p.symbol, value: p.marketValue })),
    { name: '现金', value: data.cash },
  ];

  const pnlColor = totalPnl >= 0 ? 'green' as const : 'red' as const;

  return (
    <div className="text-slate-200">
      <h1 className="mb-6 text-2xl font-bold tracking-tight">投资组合</h1>

      {/* Summary Cards */}
      <div className="mb-6 flex gap-4">
        <SummaryCard icon={Wallet} label="总资产" value={`¥${fmt(data.total_value)}`} />
        <SummaryCard icon={PiggyBank} label="可用现金" value={`¥${fmt(data.cash)}`} />
        <SummaryCard
          icon={totalPnl >= 0 ? TrendingUp : TrendingDown}
          label="总盈亏"
          value={`${totalPnl >= 0 ? '+' : ''}¥${fmt(totalPnl)}`}
          colored={pnlColor}
        />
        <SummaryCard
          icon={totalPnlPercent >= 0 ? BarChart3 : Percent}
          label="盈亏比例"
          value={`${totalPnlPercent >= 0 ? '+' : ''}${totalPnlPercent.toFixed(2)}%`}
          colored={pnlColor}
        />
      </div>

      {/* Allocation Pie Chart */}
      <div className="mb-6 rounded-xl border border-slate-700 bg-[#1e293b] p-5">
        <p className="mb-3 text-sm font-medium text-slate-400">资产配置</p>
        <ResponsiveContainer width="100%" height={320}>
          <PieChart>
            <Pie
              data={allocationData}
              cx="50%"
              cy="50%"
              outerRadius={110}
              dataKey="value"
              label={renderLabel}
              isAnimationActive={false}
            >
              {allocationData.map((_entry, idx) => (
                <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip content={<PieTooltipContent />} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Tab Switcher */}
      <div className="mb-4 flex gap-2">
        <button
          onClick={() => setTab('open')}
          className={`flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium transition ${tab === 'open' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
        >
          <BarChart3 size={14} /> 持仓明细 ({positionsWithStats.length})
        </button>
        <button
          onClick={() => setTab('closed')}
          className={`flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium transition ${tab === 'closed' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
        >
          <Archive size={14} /> 已平仓 ({closedPositions.length})
        </button>
      </div>

      {/* Open Positions Table */}
      {tab === 'open' && (
        <div className="mb-6 rounded-xl border border-slate-700 bg-[#1e293b]">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-left text-xs text-slate-400">
                  <th className="px-4 py-3">代码</th>
                  <th className="px-4 py-3">名称</th>
                  <th className="px-4 py-3 text-right">持仓</th>
                  <th className="px-4 py-3 text-right">成本价</th>
                  <th className="px-4 py-3 text-right">现价</th>
                  <th className="px-4 py-3 text-right">市值</th>
                  <th className="px-4 py-3 text-right">盈亏</th>
                  <th className="px-4 py-3 text-right">盈亏%</th>
                  <th className="px-4 py-3 text-right">持仓天数</th>
                  <th className="px-4 py-3 text-right">加仓次数</th>
                  <th className="px-4 py-3 text-center">操作</th>
                </tr>
              </thead>
              <tbody>
                {positionsWithStats.length === 0 ? (
                  <tr><td colSpan={11} className="px-4 py-8 text-center text-slate-500">暂无持仓</td></tr>
                ) : positionsWithStats.map((p) => {
                  const color = p.pnl >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]';
                  return (
                    <tr key={p.symbol} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                      <td className="px-4 py-2.5 font-mono text-[#3b82f6]">{p.symbol}</td>
                      <td className="px-4 py-2.5">{p.name}</td>
                      <td className="px-4 py-2.5 text-right">{p.shares.toLocaleString()}</td>
                      <td className="px-4 py-2.5 text-right">{p.avg_cost.toFixed(2)}</td>
                      <td className="px-4 py-2.5 text-right">{p.current_price.toFixed(2)}</td>
                      <td className="px-4 py-2.5 text-right">¥{fmt(p.marketValue)}</td>
                      <td className={`px-4 py-2.5 text-right font-medium ${color}`}>
                        {p.pnl >= 0 ? '+' : ''}¥{fmt(p.pnl)}
                      </td>
                      <td className={`px-4 py-2.5 text-right font-medium ${color}`}>
                        {p.pnlPercent >= 0 ? '+' : ''}{p.pnlPercent.toFixed(2)}%
                      </td>
                      <td className="px-4 py-2.5 text-right text-slate-300">{p.holding_days ?? '-'}</td>
                      <td className="px-4 py-2.5 text-right text-slate-300">{p.scale_level ?? '-'}</td>
                      <td className="px-4 py-2.5 text-center">
                        <button
                          onClick={() => handleClose(p.symbol)}
                          disabled={closingSymbol === p.symbol}
                          className="inline-flex items-center gap-1 rounded bg-red-600/80 px-2.5 py-1 text-xs font-medium text-white hover:bg-red-500 disabled:opacity-50"
                        >
                          {closingSymbol === p.symbol ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />}
                          平仓
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Closed Positions Table */}
      {tab === 'closed' && (
        <div className="mb-6 rounded-xl border border-slate-700 bg-[#1e293b]">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-left text-xs text-slate-400">
                  <th className="px-4 py-3">代码</th>
                  <th className="px-4 py-3 text-right">买入价</th>
                  <th className="px-4 py-3 text-right">卖出价</th>
                  <th className="px-4 py-3 text-right">数量</th>
                  <th className="px-4 py-3 text-right">已实现盈亏</th>
                  <th className="px-4 py-3 text-right">持仓天数</th>
                  <th className="px-4 py-3">开仓时间</th>
                  <th className="px-4 py-3">平仓时间</th>
                </tr>
              </thead>
              <tbody>
                {closedPositions.length === 0 ? (
                  <tr><td colSpan={8} className="px-4 py-8 text-center text-slate-500">暂无已平仓记录</td></tr>
                ) : closedPositions.map((c, i) => {
                  const color = c.realized_pnl >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]';
                  return (
                    <tr key={`${c.symbol}-${i}`} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                      <td className="px-4 py-2.5 font-mono text-[#3b82f6]">{c.symbol}</td>
                      <td className="px-4 py-2.5 text-right">{c.entry_price.toFixed(2)}</td>
                      <td className="px-4 py-2.5 text-right">{c.exit_price.toFixed(2)}</td>
                      <td className="px-4 py-2.5 text-right">{c.quantity.toLocaleString()}</td>
                      <td className={`px-4 py-2.5 text-right font-medium ${color}`}>
                        {c.realized_pnl >= 0 ? '+' : ''}¥{fmt(c.realized_pnl)}
                      </td>
                      <td className="px-4 py-2.5 text-right">{c.holding_days}</td>
                      <td className="px-4 py-2.5 text-slate-400 text-xs">{c.entry_time}</td>
                      <td className="px-4 py-2.5 text-slate-400 text-xs">{c.exit_time}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
