import { useState, useMemo, useEffect, useCallback } from 'react';
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
import { Search, TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import { getQuote, getKline } from '../api/client';
import type { KlineData } from '../types';

interface StockInfo {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  turnover: number;
}

const POPULAR_STOCKS: { symbol: string; name: string }[] = [
  { symbol: '600519.SH', name: '贵州茅台' },
  { symbol: '000858.SZ', name: '五粮液' },
  { symbol: '601318.SH', name: '中国平安' },
  { symbol: '300750.SZ', name: '宁德时代' },
  { symbol: '002594.SZ', name: '比亚迪' },
  { symbol: '600036.SH', name: '招商银行' },
];

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
  const [stockInfo, setStockInfo] = useState<StockInfo | null>(null);
  const [klineData, setKlineData] = useState<KlineData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState(60);
  const [sortKey, setSortKey] = useState<SortKey>('date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [page, setPage] = useState(0);

  const fetchStockData = useCallback(async (symbol: string, days: number) => {
    setLoading(true);
    setError(null);
    try {
      // Fetch quote and klines in parallel
      const end = new Date().toISOString().slice(0, 10);
      const start = new Date(Date.now() - days * 24 * 3600 * 1000).toISOString().slice(0, 10);

      const [quoteRes, klineRes] = await Promise.all([
        getQuote(symbol).catch(() => null),
        getKline(symbol, start, end, days).catch(() => null),
      ]);

      // Process quote
      if (quoteRes && typeof quoteRes === 'object') {
        const q = quoteRes as Record<string, unknown>;
        setStockInfo({
          symbol: String(q.symbol ?? symbol),
          name: String(q.name ?? symbol),
          price: Number(q.price ?? 0),
          change: Number(q.change ?? 0),
          changePercent: Number(q.change_percent ?? 0),
          volume: Number(q.volume ?? 0),
          turnover: Number(q.turnover ?? 0),
        });
      }

      // Process klines
      if (klineRes && typeof klineRes === 'object') {
        const kr = klineRes as { data?: Array<Record<string, unknown>> };
        if (Array.isArray(kr.data)) {
          const mapped: KlineData[] = kr.data.map((d) => ({
            date: String(d.datetime ?? d.date ?? '').slice(0, 10),
            open: Number(d.open ?? 0),
            high: Number(d.high ?? 0),
            low: Number(d.low ?? 0),
            close: Number(d.close ?? 0),
            volume: Number(d.volume ?? 0),
          }));
          setKlineData(mapped);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedSymbol) {
      fetchStockData(selectedSymbol, timeRange + 20);
    }
  }, [selectedSymbol, timeRange, fetchStockData]);

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
    if (s) selectStock(s);
  }

  const isPositive = (v: number) => v >= 0;

  return (
    <div className="text-slate-200">
      <h1 className="mb-6 text-2xl font-bold tracking-tight">行情数据 <span className="text-sm font-normal text-green-400">📡 实时数据</span></h1>

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

      {!stockInfo && !loading && (
        <div className="flex h-64 items-center justify-center rounded-xl border border-slate-700 bg-[#1e293b] text-slate-500">
          请选择或搜索一支股票查看行情
        </div>
      )}

      {loading && (
        <div className="flex h-64 items-center justify-center rounded-xl border border-slate-700 bg-[#1e293b] text-slate-400">
          <Loader2 className="mr-2 h-5 w-5 animate-spin" /> 加载真实行情数据中...
        </div>
      )}

      {error && (
        <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {stockInfo && !loading && (
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
                  <span className="text-slate-400">成交额 </span>
                  <span>{stockInfo.turnover >= 1e8 ? (stockInfo.turnover / 1e8).toFixed(1) + '亿' : (stockInfo.turnover / 1e4).toFixed(0) + '万'}</span>
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
                  shape={(props: unknown) => {
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
