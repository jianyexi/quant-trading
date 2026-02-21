
const DEFAULT_SYMBOLS= '600519,000858,300750,600036,601318,002415,000651,600276';

export default function DataSourceConfig({
  dataSource, setDataSource,
  symbols, setSymbols,
  startDate, setStartDate,
  endDate, setEndDate,
  nBars, setNBars,
}: {
  dataSource: string; setDataSource: (v: string) => void;
  symbols: string; setSymbols: (v: string) => void;
  startDate: string; setStartDate: (v: string) => void;
  endDate: string; setEndDate: (v: string) => void;
  nBars: number; setNBars: (v: number) => void;
}) {
  return (
    <div className="rounded-lg border border-[#334155] bg-[#0f172a] p-4 mb-4">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-xs text-[#94a3b8] font-semibold">数据来源</span>
        {['synthetic', 'akshare'].map((src) => (
          <button key={src} onClick={() => setDataSource(src)}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
              dataSource === src
                ? 'bg-[#3b82f6] text-white'
                : 'bg-[#334155] text-[#94a3b8] hover:bg-[#475569]'
            }`}>
            {src === 'synthetic' ? '📊 模拟数据' : '📡 真实行情 (akshare)'}
          </button>
        ))}
      </div>

      {dataSource === 'synthetic' ? (
        <div className="grid grid-cols-4 gap-3">
          <div>
            <label className="text-xs text-[#64748b] block mb-1">数据量 (bars)</label>
            <input type="number" value={nBars} onChange={(e) => setNBars(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#1e293b] px-3 py-1.5 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div>
            <label className="text-xs text-[#64748b] block mb-1">
              股票代码 <span className="text-[#475569]">(逗号分隔，留空用默认20只)</span>
            </label>
            <input type="text" value={symbols} onChange={(e) => setSymbols(e.target.value)}
              placeholder={DEFAULT_SYMBOLS}
              className="w-full rounded-lg border border-[#334155] bg-[#1e293b] px-3 py-1.5 text-sm text-[#f8fafc] placeholder-[#475569] focus:border-[#3b82f6] focus:outline-none font-mono" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-[#64748b] block mb-1">开始日期</label>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)}
                className="w-full rounded-lg border border-[#334155] bg-[#1e293b] px-3 py-1.5 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
            </div>
            <div>
              <label className="text-xs text-[#64748b] block mb-1">结束日期</label>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)}
                className="w-full rounded-lg border border-[#334155] bg-[#1e293b] px-3 py-1.5 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
