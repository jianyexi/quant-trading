import { DollarSign, TrendingUp, BarChart3, Target } from 'lucide-react';

const stats = [
  { label: 'Portfolio Value', value: '$128,430.50', icon: DollarSign, change: '+2.4%', positive: true },
  { label: 'Daily PnL', value: '+$3,241.12', icon: TrendingUp, change: '+1.8%', positive: true },
  { label: 'Open Positions', value: '12', icon: BarChart3, change: '3 new', positive: true },
  { label: 'Win Rate', value: '68.5%', icon: Target, change: '+0.7%', positive: true },
];

const recentTrades = [
  { id: 1, symbol: 'AAPL', side: 'BUY', qty: 50, price: 189.25, pnl: '+$312.50', time: '14:32:01' },
  { id: 2, symbol: 'TSLA', side: 'SELL', qty: 30, price: 248.10, pnl: '-$87.30', time: '13:45:22' },
  { id: 3, symbol: 'NVDA', side: 'BUY', qty: 20, price: 875.60, pnl: '+$1,420.00', time: '12:18:45' },
  { id: 4, symbol: 'MSFT', side: 'SELL', qty: 40, price: 415.30, pnl: '+$560.00', time: '11:05:33' },
  { id: 5, symbol: 'AMZN', side: 'BUY', qty: 25, price: 178.90, pnl: '-$225.00', time: '10:22:17' },
];

export default function Dashboard() {
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
        <h2 className="mb-4 text-lg font-semibold text-[#f8fafc]">Recent Trades</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[#334155] text-[#94a3b8]">
                <th className="pb-3 pr-4 font-medium">Time</th>
                <th className="pb-3 pr-4 font-medium">Symbol</th>
                <th className="pb-3 pr-4 font-medium">Side</th>
                <th className="pb-3 pr-4 font-medium">Qty</th>
                <th className="pb-3 pr-4 font-medium">Price</th>
                <th className="pb-3 font-medium">PnL</th>
              </tr>
            </thead>
            <tbody>
              {recentTrades.map((trade) => (
                <tr key={trade.id} className="border-b border-[#334155]/50 text-[#f8fafc]">
                  <td className="py-3 pr-4 font-mono text-xs text-[#94a3b8]">{trade.time}</td>
                  <td className="py-3 pr-4 font-medium">{trade.symbol}</td>
                  <td className="py-3 pr-4">
                    <span
                      className={`rounded px-2 py-0.5 text-xs font-medium ${
                        trade.side === 'BUY'
                          ? 'bg-[#22c55e]/15 text-[#22c55e]'
                          : 'bg-[#ef4444]/15 text-[#ef4444]'
                      }`}
                    >
                      {trade.side}
                    </span>
                  </td>
                  <td className="py-3 pr-4">{trade.qty}</td>
                  <td className="py-3 pr-4">${trade.price.toFixed(2)}</td>
                  <td
                    className={`py-3 font-medium ${
                      trade.pnl.startsWith('+') ? 'text-[#22c55e]' : 'text-[#ef4444]'
                    }`}
                  >
                    {trade.pnl}
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
