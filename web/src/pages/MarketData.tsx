import { useState, useMemo } from 'react';
import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
  BarChart,
} from 'recharts';
import { Search, TrendingUp, TrendingDown } from 'lucide-react';
import type { KlineData } from '../types';

// --- Mock data ---

interface StockInfo {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: string;
}

const POPULAR_STOCKS: { symbol: string; name: string }[] = [
  { symbol: '600519.SH', name: '贵州茅台' },
  { symbol: '000858.SZ', name: '五粮液' },
  { symbol: '601318.SH', name: '中国平安' },
  { symbol: '000001.SZ', name: '平安银行' },
];

const STOCK_INFO_MAP: Record<string, StockInfo> = {
  '600519.SH': { symbol: '600519.SH', name: '贵州茅台', price: 1688.50, change: 12.30, changePercent: 0.73, volume: 2_834_500, marketCap: '2.12万亿' },
  '000858.SZ': { symbol: '000858.SZ', name: '五粮液', price: 142.85, change: -1.65, changePercent: -1.14, volume: 5_612_300, marketCap: '5526亿' },
  '601318.SH': { symbol: '601318.SH', name: '中国平安', price: 52.36, change: 0.48, changePercent: 0.92, volume: 18_923_400, marketCap: '9568亿' },
  '000001.SZ': { symbol: '000001.SZ', name: '平安银行', price: 12.58, change: -0.12, changePercent: -0.95, volume: 32_145_600, marketCap: '2440亿' },
};

function generateKlineData(basePrice: number, days: number): KlineData[] {
  const data: KlineData[] = [];
  let price = basePrice;
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    if (date.getDay() === 0 || date.getDay() === 6) continue;

    const changePercent = (Math.random() - 0.48) * 0.04;
    const open = price;
    const close = +(price * (1 + changePercent)).toFixed(2);
    const high = +(Math.max(open, close) * (1 + Math.random() * 0.015)).toFixed(2);
    const low = +(Math.min(open, close) * (1 - Math.random() * 0.015)).toFixed(2);
    const volume = Math.floor(1_000_000 + Math.random() * 5_000_000);

    data.push({ date: date.toISOString().slice(0, 10), open, high, low, close, volume });
    price = close;
  }
  return data;
}

const TIME_RANGES = [
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
];

type SortKey = keyof KlineData;
type SortDir = 'asc' | 'desc';
const PAGE_SIZE = 10;

// --- Custom Tooltip ---

interface TooltipPayloadEntry {
  payload: KlineData & { isUp: boolean };
}

function KlineTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-xs shadow-lg">
      <p className="mb-1 font-medium text-slate-200">{d.date}</p>
      <p>开 {d.open.toFixed(2)}</p>
      <p>高 {d.high.toFixed(2)}</p>
      <p>低 {d.low.toFixed(2)}</p>
      <p>收 {d.close.toFixed(2)}</p>
      <p>量 {(d.volume / 10000).toFixed(0)}万</p>
    </div>
  );
}

// --- Main Component ---

export default function MarketData() {
  const [searchInput, setSearchInput] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState(60);
  const [sortKey, setSortKey] = useState<SortKey>('date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [page, setPage] = useState(0);

  const stockInfo = selectedSymbol ? STOCK_INFO_MAP[selectedSymbol] ?? null : null;

  const klineData = useMemo(
    () => (stockInfo ? generateKlineData(stockInfo.price * 0.92, timeRange + 20).slice(-timeRange) : []),
    [stockInfo, timeRange],
  );

  const chartData = useMemo(
    () =>
      klineData.map((d) => ({
        ...d,
        isUp: d.close >= d.open,
      })),
    [klineData],
  );

  const [priceMin, priceMax] = useMemo(() => {
    if (klineData.length === 0) return [0, 100];
    let lo = Infinity;
    let hi = -Infinity;
    for (const d of klineData) {
      if (d.low < lo) lo = d.low;
      if (d.high > hi) hi = d.high;
    }
    const pad = (hi - lo) * 0.05 || 1;
    return [Math.floor(lo - pad), Math.ceil(hi + pad)];
  }, [klineData]);

  const sortedData = useMemo(() => {
    const copy = [...klineData];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av < bv) return sortDir === 'asc' ? -1 : 1;
      if (av > bv) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
    return copy;
  }, [klineData, sortKey, sortDir]);

  const totalPages = Math.ceil(sortedData.length / PAGE_SIZE);
  const pagedData = sortedData.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
    setPage(0);
  }

  function selectStock(symbol: string) {
    setSelectedSymbol(symbol);
    setSearchInput('');
    setPage(0);
  }

  function handleSearch() {
    const s = searchInput.trim().toUpperCase();
    if (STOCK_INFO_MAP[s]) selectStock(s);
  }

  const isPositive = (v: number) => v >= 0;

  return (
    <div className="text-slate-200">
      <h1 className="mb-6 text-2xl font-bold tracking-tight">行情数据</h1>

      {/* Search Bar */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="输入股票代码 (e.g. 600519.SH)"
            className="h-10 w-64 rounded-lg border border-slate-600 bg-[#1e293b] pl-10 pr-3 text-sm text-slate-200 placeholder-slate-500 outline-none focus:border-[#3b82f6]"
          />
        </div>
        {POPULAR_STOCKS.map((s) => (
          <button
            key={s.symbol}
            onClick={() => selectStock(s.symbol)}
            className={`rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
              selectedSymbol === s.symbol
                ? 'border-[#3b82f6] bg-[#3b82f6]/20 text-[#3b82f6]'
                : 'border-slate-600 bg-[#1e293b] text-slate-300 hover:border-slate-500'
            }`}
          >
            {s.symbol} {s.name}
          </button>
        ))}
      </div>

      {!stockInfo && (
        <div className="flex h-64 items-center justify-center rounded-xl border border-slate-700 bg-[#1e293b] text-slate-500">
          请选择或搜索一支股票查看行情
        </div>
      )}

      {stockInfo && (
        <>
          {/* Stock Info Card */}
          <div className="mb-6 rounded-xl border border-slate-700 bg-[#1e293b] p-5">
            <div className="flex flex-wrap items-center gap-8">
              <div>
                <p className="text-sm text-slate-400">{stockInfo.symbol}</p>
                <p className="text-xl font-bold">{stockInfo.name}</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-3xl font-bold">{stockInfo.price.toFixed(2)}</span>
                {isPositive(stockInfo.change) ? (
                  <TrendingUp className="h-5 w-5 text-[#22c55e]" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-[#ef4444]" />
                )}
              </div>
              <div className="flex gap-6 text-sm">
                <div>
                  <span className="text-slate-400">涨跌额 </span>
                  <span className={isPositive(stockInfo.change) ? 'text-[#22c55e]' : 'text-[#ef4444]'}>
                    {stockInfo.change > 0 ? '+' : ''}{stockInfo.change.toFixed(2)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">涨跌幅 </span>
                  <span className={isPositive(stockInfo.changePercent) ? 'text-[#22c55e]' : 'text-[#ef4444]'}>
                    {stockInfo.changePercent > 0 ? '+' : ''}{stockInfo.changePercent.toFixed(2)}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">成交量 </span>
                  <span>{(stockInfo.volume / 10000).toFixed(0)}万</span>
                </div>
                <div>
                  <span className="text-slate-400">总市值 </span>
                  <span>{stockInfo.marketCap}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Time Range Selector */}
          <div className="mb-4 flex gap-2">
            {TIME_RANGES.map((r) => (
              <button
                key={r.label}
                onClick={() => { setTimeRange(r.days); setPage(0); }}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  timeRange === r.days
                    ? 'bg-[#3b82f6] text-white'
                    : 'bg-[#1e293b] text-slate-400 hover:text-slate-200'
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>

          {/* K-line Chart */}
          <div className="mb-2 rounded-xl border border-slate-700 bg-[#1e293b] p-4">
            <p className="mb-3 text-sm font-medium text-slate-400">K线图</p>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={chartData} barCategoryGap="20%">
                <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  tickFormatter={(v: string) => v.slice(5)}
                  minTickGap={30}
                />
                <YAxis
                  domain={[priceMin, priceMax]}
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  tickFormatter={(v: number) => v.toFixed(0)}
                  width={60}
                />
                <Tooltip content={<KlineTooltip />} />
                <Bar
                  dataKey="high"
                  isAnimationActive={false}
                  shape={(props: Record<string, unknown>) => {
                    const { x, y, width, height, index } = props as {
                      x: number; y: number; width: number; height: number; index: number;
                    };
                    const d = chartData[index];
                    if (!d || !height) return null;
                    // recharts maps "high" → y(top) and bar bottom is y + height (== axis baseline or domain min)
                    // We can derive the pixel-per-unit scale from the bar that recharts gave us for "high"
                    const pxPerUnit = height / (d.high - priceMin);
                    const toY = (v: number) => y + (d.high - v) * pxPerUnit;

                    const color = d.isUp ? '#22c55e' : '#ef4444';
                    const bodyTop = toY(Math.max(d.open, d.close));
                    const bodyBot = toY(Math.min(d.open, d.close));
                    const wickTop = toY(d.high);
                    const wickBot = toY(d.low);
                    const cx = x + width / 2;
                    return (
                      <g>
                        <line x1={cx} x2={cx} y1={wickTop} y2={wickBot} stroke={color} strokeWidth={1} />
                        <rect
                          x={x}
                          y={bodyTop}
                          width={width}
                          height={Math.max(bodyBot - bodyTop, 1)}
                          fill={color}
                        />
                      </g>
                    );
                  }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Volume Chart */}
          <div className="mb-6 rounded-xl border border-slate-700 bg-[#1e293b] p-4">
            <p className="mb-3 text-sm font-medium text-slate-400">成交量</p>
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={chartData} barCategoryGap="20%">
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  tickFormatter={(v: string) => v.slice(5)}
                  minTickGap={30}
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  tickFormatter={(v: number) => `${(v / 10000).toFixed(0)}万`}
                  width={60}
                />
                <Bar dataKey="volume" isAnimationActive={false}>
                  {chartData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.isUp ? '#22c55e44' : '#ef444444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Data Table */}
          <div className="rounded-xl border border-slate-700 bg-[#1e293b]">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700 text-left text-xs text-slate-400">
                    {(
                      [
                        ['date', '日期'],
                        ['open', '开盘'],
                        ['high', '最高'],
                        ['low', '最低'],
                        ['close', '收盘'],
                        ['volume', '成交量'],
                      ] as const
                    ).map(([key, label]) => (
                      <th
                        key={key}
                        onClick={() => handleSort(key)}
                        className="cursor-pointer px-4 py-3 hover:text-slate-200"
                      >
                        {label} {sortKey === key ? (sortDir === 'asc' ? '↑' : '↓') : ''}
                      </th>
                    ))}
                    <th className="px-4 py-3">涨跌%</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedData.map((d) => {
                    const changePct = ((d.close - d.open) / d.open) * 100;
                    return (
                      <tr key={d.date} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                        <td className="px-4 py-2.5 font-mono">{d.date}</td>
                        <td className="px-4 py-2.5">{d.open.toFixed(2)}</td>
                        <td className="px-4 py-2.5">{d.high.toFixed(2)}</td>
                        <td className="px-4 py-2.5">{d.low.toFixed(2)}</td>
                        <td className="px-4 py-2.5">{d.close.toFixed(2)}</td>
                        <td className="px-4 py-2.5">{(d.volume / 10000).toFixed(0)}万</td>
                        <td className={`px-4 py-2.5 font-medium ${changePct >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                          {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between border-t border-slate-700 px-4 py-3 text-xs text-slate-400">
              <span>共 {sortedData.length} 条 · 第 {page + 1}/{totalPages} 页</span>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="rounded bg-slate-700 px-2 py-1 disabled:opacity-40"
                >
                  上一页
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="rounded bg-slate-700 px-2 py-1 disabled:opacity-40"
                >
                  下一页
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
