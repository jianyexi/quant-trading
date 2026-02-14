import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Wallet, PiggyBank, BarChart3, Percent } from 'lucide-react';

// --- Mock Data ---

interface PositionRow {
  symbol: string;
  name: string;
  shares: number;
  avgCost: number;
  currentPrice: number;
}

interface OrderRow {
  id: string;
  time: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  status: '已成交' | '已撤单' | '委托中';
}

const POSITIONS: PositionRow[] = [
  { symbol: '600519.SH', name: '贵州茅台', shares: 100, avgCost: 1620.00, currentPrice: 1688.50 },
  { symbol: '000858.SZ', name: '五粮液', shares: 500, avgCost: 148.30, currentPrice: 142.85 },
  { symbol: '601318.SH', name: '中国平安', shares: 1000, avgCost: 49.80, currentPrice: 52.36 },
  { symbol: '000001.SZ', name: '平安银行', shares: 2000, avgCost: 13.10, currentPrice: 12.58 },
  { symbol: '600036.SH', name: '招商银行', shares: 800, avgCost: 32.50, currentPrice: 35.72 },
];

const CASH = 245_680.00;

function computePositionStats(p: PositionRow) {
  const marketValue = p.shares * p.currentPrice;
  const cost = p.shares * p.avgCost;
  const pnl = marketValue - cost;
  const pnlPercent = (pnl / cost) * 100;
  return { ...p, marketValue, pnl, pnlPercent };
}

const positionsWithStats = POSITIONS.map(computePositionStats);
const totalMarketValue = positionsWithStats.reduce((s, p) => s + p.marketValue, 0);
const totalValue = totalMarketValue + CASH;
const totalCost = positionsWithStats.reduce((s, p) => s + p.shares * p.avgCost, 0);
const totalPnl = totalMarketValue - totalCost;
const totalPnlPercent = (totalPnl / totalCost) * 100;

// Pie chart data
const PIE_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#64748b'];

const allocationData = [
  ...positionsWithStats.map((p) => ({ name: p.name, value: p.marketValue })),
  { name: '现金', value: CASH },
];

const ORDERS: OrderRow[] = [];

// --- Helpers ---

function fmt(v: number): string {
  return v.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// --- Custom Pie Label ---

interface LabelProps {
  cx: number;
  cy: number;
  midAngle: number;
  outerRadius: number;
  name: string;
  percent: number;
}

function renderLabel({ cx, cy, midAngle, outerRadius, name, percent }: LabelProps) {
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
  const pnlColor = totalPnl >= 0 ? 'green' : 'red';

  return (
    <div className="text-slate-200">
      <h1 className="mb-6 text-2xl font-bold tracking-tight">投资组合</h1>

      {/* Summary Cards */}
      <div className="mb-6 flex gap-4">
        <SummaryCard icon={Wallet} label="总资产" value={`¥${fmt(totalValue)}`} />
        <SummaryCard icon={PiggyBank} label="可用现金" value={`¥${fmt(CASH)}`} />
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

      {/* Positions Table */}
      <div className="mb-6 rounded-xl border border-slate-700 bg-[#1e293b]">
        <p className="border-b border-slate-700 px-5 py-3 text-sm font-medium text-slate-400">持仓明细</p>
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
              </tr>
            </thead>
            <tbody>
              {positionsWithStats.map((p) => {
                const color = p.pnl >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]';
                return (
                  <tr key={p.symbol} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="px-4 py-2.5 font-mono text-[#3b82f6]">{p.symbol}</td>
                    <td className="px-4 py-2.5">{p.name}</td>
                    <td className="px-4 py-2.5 text-right">{p.shares.toLocaleString()}</td>
                    <td className="px-4 py-2.5 text-right">{p.avgCost.toFixed(2)}</td>
                    <td className="px-4 py-2.5 text-right">{p.currentPrice.toFixed(2)}</td>
                    <td className="px-4 py-2.5 text-right">¥{fmt(p.marketValue)}</td>
                    <td className={`px-4 py-2.5 text-right font-medium ${color}`}>
                      {p.pnl >= 0 ? '+' : ''}¥{fmt(p.pnl)}
                    </td>
                    <td className={`px-4 py-2.5 text-right font-medium ${color}`}>
                      {p.pnlPercent >= 0 ? '+' : ''}{p.pnlPercent.toFixed(2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Orders Table */}
      <div className="rounded-xl border border-slate-700 bg-[#1e293b]">
        <p className="border-b border-slate-700 px-5 py-3 text-sm font-medium text-slate-400">最近委托</p>
        {ORDERS.length === 0 ? (
          <div className="flex h-32 items-center justify-center text-sm text-slate-500">No recent orders</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-left text-xs text-slate-400">
                  <th className="px-4 py-3">时间</th>
                  <th className="px-4 py-3">代码</th>
                  <th className="px-4 py-3">方向</th>
                  <th className="px-4 py-3 text-right">价格</th>
                  <th className="px-4 py-3 text-right">数量</th>
                  <th className="px-4 py-3">状态</th>
                </tr>
              </thead>
              <tbody>
                {ORDERS.map((o) => (
                  <tr key={o.id} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                    <td className="px-4 py-2.5 font-mono text-xs">{o.time}</td>
                    <td className="px-4 py-2.5 font-mono">{o.symbol}</td>
                    <td className={`px-4 py-2.5 font-medium ${o.side === 'buy' ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                      {o.side === 'buy' ? '买入' : '卖出'}
                    </td>
                    <td className="px-4 py-2.5 text-right">{o.price.toFixed(2)}</td>
                    <td className="px-4 py-2.5 text-right">{o.quantity.toLocaleString()}</td>
                    <td className="px-4 py-2.5">{o.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
