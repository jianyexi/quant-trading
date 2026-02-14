import { useState, useMemo } from 'react';
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
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { runBacktest } from '../api/client';
import type { BacktestResult } from '../types';

interface BacktestConfig {
  strategy: string;
  symbol: string;
  start: string;
  end: string;
  capital: number;
}

interface EquityPoint {
  date: string;
  value: number;
}

interface Trade {
  date: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  pnl: number;
}

const STRATEGIES = [
  { value: 'sma_cross', label: 'SMA Crossover' },
  { value: 'rsi_reversal', label: 'RSI Reversal' },
  { value: 'macd_trend', label: 'MACD Trend Following' },
  { value: 'bollinger_bands', label: 'Bollinger Bands' },
  { value: 'dual_momentum', label: 'Dual Momentum' },
];

function generateEquityCurve(capital: number): EquityPoint[] {
  const points: EquityPoint[] = [];
  let value = capital;
  const startDate = new Date('2024-01-02');
  for (let i = 0; i < 252; i++) {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + Math.floor(i * 1.4));
    const dailyReturn = (Math.random() - 0.47) * 0.025;
    value *= 1 + dailyReturn;
    points.push({
      date: date.toISOString().split('T')[0],
      value: Math.round(value * 100) / 100,
    });
  }
  return points;
}

function generateTrades(symbol: string): Trade[] {
  const trades: Trade[] = [];
  const baseDate = new Date('2024-01-15');
  for (let i = 0; i < 20; i++) {
    const date = new Date(baseDate);
    date.setDate(baseDate.getDate() + i * 12 + Math.floor(Math.random() * 5));
    const side: 'BUY' | 'SELL' = i % 2 === 0 ? 'BUY' : 'SELL';
    const price = 25 + Math.random() * 15;
    const quantity = Math.floor(Math.random() * 9 + 1) * 100;
    const pnl = side === 'SELL' ? (Math.random() - 0.4) * 5000 : 0;
    trades.push({
      date: date.toISOString().split('T')[0],
      symbol,
      side,
      price: Math.round(price * 100) / 100,
      quantity,
      pnl: Math.round(pnl * 100) / 100,
    });
  }
  return trades;
}

function loadSavedConfig(): Partial<BacktestConfig> {
  try {
    const raw = localStorage.getItem('quant-strategy-config');
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
  capital: 100000,
};

export default function Backtest() {
  const saved = useMemo(() => loadSavedConfig(), []);
  const [config, setConfig] = useState<BacktestConfig>({ ...defaultConfig, ...saved });
  const [configOpen, setConfigOpen] = useState(true);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = (await runBacktest({
        strategy: config.strategy,
        symbol: config.symbol,
        start: config.start,
        end: config.end,
        capital: config.capital,
      })) as BacktestResult;
      setResult(res);
      setEquityCurve(generateEquityCurve(config.capital));
      setTrades(generateTrades(config.symbol));
      setConfigOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Backtest failed');
    } finally {
      setRunning(false);
    }
  };

  const updateConfig = (key: keyof BacktestConfig, value: string | number) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const metrics = result
    ? [
        {
          label: 'Total Return',
          value: `${result.total_return_percent >= 0 ? '+' : ''}${result.total_return_percent.toFixed(2)}%`,
          color: result.total_return_percent >= 0 ? '#22c55e' : '#ef4444',
          icon: result.total_return_percent >= 0 ? TrendingUp : TrendingDown,
        },
        {
          label: 'Sharpe Ratio',
          value: result.sharpe_ratio.toFixed(2),
          color: result.sharpe_ratio >= 1 ? '#22c55e' : '#eab308',
          icon: BarChart3,
        },
        {
          label: 'Max Drawdown',
          value: `-${result.max_drawdown_percent.toFixed(2)}%`,
          color: '#ef4444',
          icon: TrendingDown,
        },
        {
          label: 'Win Rate',
          value: `${result.win_rate_percent.toFixed(1)}%`,
          color: result.win_rate_percent >= 50 ? '#22c55e' : '#eab308',
          icon: Target,
        },
        {
          label: 'Total Trades',
          value: result.total_trades.toString(),
          color: '#3b82f6',
          icon: Activity,
        },
        {
          label: 'Profit Factor',
          value: result.profit_factor.toFixed(2),
          color: result.profit_factor >= 1 ? '#22c55e' : '#ef4444',
          icon: DollarSign,
        },
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
          <h2 className="text-lg font-semibold text-[#f8fafc]">Backtest Configuration</h2>
          {configOpen ? (
            <ChevronUp size={20} className="text-[#94a3b8]" />
          ) : (
            <ChevronDown size={20} className="text-[#94a3b8]" />
          )}
        </button>

        {configOpen && (
          <div className="px-6 pb-6 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Strategy */}
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">Strategy</label>
                <select
                  value={config.strategy}
                  onChange={(e) => updateConfig('strategy', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6] transition-colors"
                >
                  {STRATEGIES.map((s) => (
                    <option key={s.value} value={s.value}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Symbol */}
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">Symbol</label>
                <input
                  type="text"
                  value={config.symbol}
                  onChange={(e) => updateConfig('symbol', e.target.value)}
                  placeholder="e.g. 600519.SH"
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6] transition-colors"
                />
              </div>

              {/* Start Date */}
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">Start Date</label>
                <input
                  type="date"
                  value={config.start}
                  onChange={(e) => updateConfig('start', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6] transition-colors"
                />
              </div>

              {/* End Date */}
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">End Date</label>
                <input
                  type="date"
                  value={config.end}
                  onChange={(e) => updateConfig('end', e.target.value)}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6] transition-colors"
                />
              </div>

              {/* Initial Capital */}
              <div>
                <label className="block text-sm text-[#94a3b8] mb-1.5">Initial Capital (¥)</label>
                <input
                  type="number"
                  value={config.capital}
                  onChange={(e) => updateConfig('capital', Number(e.target.value))}
                  min={1000}
                  step={10000}
                  className="w-full bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc] outline-none focus:border-[#3b82f6] transition-colors"
                />
              </div>
            </div>

            {error && (
              <div className="text-sm text-[#ef4444] bg-[#ef4444]/10 border border-[#ef4444]/20 rounded-lg px-4 py-2">
                {error}
              </div>
            )}

            <button
              onClick={() => void handleRun()}
              disabled={running}
              className="flex items-center gap-2 px-6 py-2.5 bg-[#3b82f6] hover:bg-[#2563eb] text-white rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
            >
              <Play size={16} />
              {running ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {metrics.map((m) => {
              const Icon = m.icon;
              return (
                <div
                  key={m.label}
                  className="bg-[#1e293b] rounded-xl border border-[#334155] p-4"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Icon size={16} style={{ color: m.color }} />
                    <span className="text-xs text-[#94a3b8]">{m.label}</span>
                  </div>
                  <p className="text-xl font-bold" style={{ color: m.color }}>
                    {m.value}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Equity Curve */}
          <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
            <h3 className="text-base font-semibold text-[#f8fafc] mb-4">Equity Curve</h3>
            <ResponsiveContainer width="100%" height={350}>
              <AreaChart data={equityCurve}>
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  tickLine={{ stroke: '#334155' }}
                  axisLine={{ stroke: '#334155' }}
                  interval={Math.floor(equityCurve.length / 6)}
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  tickLine={{ stroke: '#334155' }}
                  axisLine={{ stroke: '#334155' }}
                  tickFormatter={(v: number) => `¥${(v / 1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                  formatter={(value: number | undefined) => [`¥${(value ?? 0).toLocaleString()}`, 'Portfolio']}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fill="url(#equityGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Trade List */}
          <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-6">
            <h3 className="text-base font-semibold text-[#f8fafc] mb-4">Trade History</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[#334155]">
                    {['Date', 'Symbol', 'Side', 'Price', 'Quantity', 'PnL'].map((h) => (
                      <th
                        key={h}
                        className="text-left py-3 px-4 text-xs font-medium text-[#94a3b8] uppercase tracking-wider"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {trades.map((t, i) => (
                    <tr
                      key={i}
                      className="border-b border-[#334155]/50 hover:bg-[#334155]/30 transition-colors"
                    >
                      <td className="py-3 px-4 text-[#f8fafc]">{t.date}</td>
                      <td className="py-3 px-4 text-[#f8fafc] font-mono">{t.symbol}</td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-medium ${
                            t.side === 'BUY'
                              ? 'bg-[#22c55e]/15 text-[#22c55e]'
                              : 'bg-[#ef4444]/15 text-[#ef4444]'
                          }`}
                        >
                          {t.side}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-[#f8fafc] font-mono">
                        ¥{t.price.toFixed(2)}
                      </td>
                      <td className="py-3 px-4 text-[#f8fafc] font-mono">
                        {t.quantity.toLocaleString()}
                      </td>
                      <td
                        className="py-3 px-4 font-mono font-medium"
                        style={{
                          color:
                            t.pnl > 0 ? '#22c55e' : t.pnl < 0 ? '#ef4444' : '#94a3b8',
                        }}
                      >
                        {t.pnl !== 0
                          ? `${t.pnl > 0 ? '+' : ''}¥${t.pnl.toFixed(2)}`
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
