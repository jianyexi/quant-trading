import { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, BarChart3, Target, Loader2 } from 'lucide-react';
import { getDashboard } from '../api/client';

interface DashboardData {
  portfolio_value: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  open_positions: number;
  win_rate: number;
  recent_trades: {
    time: string;
    symbol: string;
    name: string;
    side: string;
    quantity: number;
    price: number;
    pnl: number;
  }[];
}

function fmt(v: number | undefined | null): string {
  return (v ?? 0).toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDashboard()
      .then((d) => setData(d as DashboardData))
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load'));
  }, []);

  if (error) {
    return <div className="text-red-400 p-6">Error: {error}</div>;
  }
  if (!data) {
    return <div className="flex items-center justify-center h-64 text-slate-400"><Loader2 className="animate-spin mr-2" size={20} /> Loading...</div>;
  }

  const stats = [
    { label: '总资产', value: `¥${fmt(data.portfolio_value)}`, icon: DollarSign, change: `${data.daily_pnl >= 0 ? '+' : ''}${data.daily_pnl_percent.toFixed(1)}%`, positive: data.daily_pnl >= 0 },
    { label: '日盈亏', value: `${data.daily_pnl >= 0 ? '+' : ''}¥${fmt(data.daily_pnl)}`, icon: TrendingUp, change: `${data.daily_pnl >= 0 ? '+' : ''}${data.daily_pnl_percent.toFixed(2)}%`, positive: data.daily_pnl >= 0 },
    { label: '持仓数', value: String(data.open_positions), icon: BarChart3, change: 'active', positive: true },
    { label: '胜率', value: `${data.win_rate.toFixed(1)}%`, icon: Target, change: data.win_rate >= 50 ? 'above avg' : 'below avg', positive: data.win_rate >= 50 },
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-[#f8fafc]">Dashboard</h1>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map(({ label, value, icon: Icon, change, positive }) => (
          <div
            key={label}
            className="rounded-xl border border-[#334155] bg-[#1e293b] p-5"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm text-[#94a3b8]">{label}</span>
              <Icon className="h-5 w-5 text-[#3b82f6]" />
            </div>
            <p className="mt-2 text-2xl font-semibold text-[#f8fafc]">{value}</p>
            <span className={`mt-1 text-xs ${positive ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
              {change}
            </span>
          </div>
        ))}
      </div>

      {/* Recent Trades */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h2 className="mb-4 text-lg font-semibold text-[#f8fafc]">最近交易</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[#334155] text-[#94a3b8]">
                <th className="pb-3 pr-4 font-medium">时间</th>
                <th className="pb-3 pr-4 font-medium">代码</th>
                <th className="pb-3 pr-4 font-medium">名称</th>
                <th className="pb-3 pr-4 font-medium">方向</th>
                <th className="pb-3 pr-4 font-medium">数量</th>
                <th className="pb-3 pr-4 font-medium">价格</th>
                <th className="pb-3 font-medium">盈亏</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_trades.map((trade, i) => (
                <tr key={i} className="border-b border-[#334155]/50 text-[#f8fafc]">
                  <td className="py-3 pr-4 font-mono text-xs text-[#94a3b8]">{trade.time}</td>
                  <td className="py-3 pr-4 font-mono text-[#3b82f6]">{trade.symbol}</td>
                  <td className="py-3 pr-4">{trade.name}</td>
                  <td className="py-3 pr-4">
                    <span
                      className={`rounded px-2 py-0.5 text-xs font-medium ${
                        trade.side === 'BUY'
                          ? 'bg-[#22c55e]/15 text-[#22c55e]'
                          : 'bg-[#ef4444]/15 text-[#ef4444]'
                      }`}
                    >
                      {trade.side === 'BUY' ? '买入' : '卖出'}
                    </span>
                  </td>
                  <td className="py-3 pr-4">{trade.quantity ?? 0}</td>
                  <td className="py-3 pr-4">¥{(trade.price ?? 0).toFixed(2)}</td>
                  <td
                    className={`py-3 font-medium ${
                      (trade.pnl ?? 0) >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'
                    }`}
                  >
                    {(trade.pnl ?? 0) >= 0 ? '+' : ''}¥{fmt(trade.pnl)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
