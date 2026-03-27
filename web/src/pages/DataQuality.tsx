import { useState, useCallback } from 'react';
import { ClipboardCheck, Loader2, ChevronDown, ChevronRight, AlertTriangle } from 'lucide-react';
import { checkDataQuality, type SymbolQualityResult, type DataQualityIssue } from '../api/client';

const ISSUE_LABELS: Record<string, string> = {
  missing_bars: '缺失K线',
  price_anomaly: '价格异常',
  zero_volume: '零成交量',
  stale_price: '停牌/僵尸价',
  large_gap: '大幅跳空',
};

function scoreColor(score: number): string {
  if (score >= 90) return 'text-emerald-400';
  if (score >= 70) return 'text-yellow-400';
  return 'text-red-400';
}

function scoreBg(score: number): string {
  if (score >= 90) return 'bg-emerald-500/15 border-emerald-500/30';
  if (score >= 70) return 'bg-yellow-500/15 border-yellow-500/30';
  return 'bg-red-500/15 border-red-500/30';
}

function IssueRow({ issue }: { issue: DataQualityIssue }) {
  const hasIssues = (issue.count ?? 0) > 0;
  return (
    <div className={`flex items-start gap-3 px-3 py-1.5 text-sm rounded ${hasIssues ? 'bg-[#1e293b]' : ''}`}>
      <span className={`w-28 shrink-0 font-medium ${hasIssues ? 'text-yellow-400' : 'text-[#64748b]'}`}>
        {ISSUE_LABELS[issue.type] ?? issue.type}
      </span>
      <span className="text-[#94a3b8] w-12 text-right shrink-0">
        {issue.count ?? 0}
        {issue.expected !== undefined && issue.expected !== null && (
          <span className="text-[#64748b]">/{issue.expected}</span>
        )}
      </span>
      <div className="flex-1 text-xs text-[#64748b] truncate">
        {issue.details && <span className="text-yellow-300/80 mr-2">{issue.details}</span>}
        {issue.dates && issue.dates.length > 0 && (
          <span>{issue.dates.slice(0, 5).join(', ')}{issue.dates.length > 5 ? ` +${issue.dates.length - 5}` : ''}</span>
        )}
      </div>
    </div>
  );
}

function ResultCard({ result }: { result: SymbolQualityResult }) {
  const [open, setOpen] = useState(false);
  const issueCount = result.issues.reduce((n, i) => n + (i.count ?? 0), 0);

  return (
    <div className={`border rounded-lg ${scoreBg(result.quality_score)}`}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-4 px-4 py-3 text-left"
      >
        <span className="font-mono font-bold text-[#f8fafc] w-24">{result.symbol}</span>
        <span className={`text-2xl font-bold ${scoreColor(result.quality_score)}`}>
          {result.quality_score.toFixed(1)}
        </span>
        <span className="text-xs text-[#64748b]">/ 100</span>
        <span className="flex-1 text-xs text-[#94a3b8]">
          {result.total_bars} bars · {result.date_range.start} ~ {result.date_range.end}
        </span>
        {issueCount > 0 && (
          <span className="flex items-center gap-1 text-xs text-yellow-400">
            <AlertTriangle className="h-3.5 w-3.5" />
            {issueCount} issue{issueCount > 1 ? 's' : ''}
          </span>
        )}
        {open ? <ChevronDown className="h-4 w-4 text-[#64748b]" /> : <ChevronRight className="h-4 w-4 text-[#64748b]" />}
      </button>
      {open && (
        <div className="px-4 pb-3 flex flex-col gap-1">
          {result.issues.map(issue => (
            <IssueRow key={issue.type} issue={issue} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function DataQuality() {
  const [symbolsInput, setSymbolsInput] = useState('000001,000002');
  const [startDate, setStartDate] = useState('20230101');
  const [endDate, setEndDate] = useState('20240101');
  const [results, setResults] = useState<SymbolQualityResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    const symbols = symbolsInput
      .split(/[,\s]+/)
      .map(s => s.trim())
      .filter(Boolean);
    if (symbols.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await checkDataQuality(symbols, startDate, endDate);
      setResults(resp.results);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [symbolsInput, startDate, endDate]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <ClipboardCheck className="h-6 w-6 text-[#3b82f6]" />
        <h1 className="text-2xl font-bold text-[#f8fafc]">数据质量检查</h1>
      </div>

      {/* Input Form */}
      <div className="rounded-xl bg-[#1e293b] border border-[#334155] p-5">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-[#94a3b8] mb-1">股票代码 (逗号分隔)</label>
            <input
              type="text"
              value={symbolsInput}
              onChange={e => setSymbolsInput(e.target.value)}
              placeholder="000001, 000002, 600519"
              className="w-full rounded-lg bg-[#0f172a] border border-[#334155] px-3 py-2 text-sm text-[#f8fafc] placeholder:text-[#475569] focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/50"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1">开始日期</label>
            <input
              type="text"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              placeholder="YYYYMMDD"
              className="w-full rounded-lg bg-[#0f172a] border border-[#334155] px-3 py-2 text-sm text-[#f8fafc] placeholder:text-[#475569] focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/50"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1">结束日期</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={endDate}
                onChange={e => setEndDate(e.target.value)}
                placeholder="YYYYMMDD"
                className="flex-1 rounded-lg bg-[#0f172a] border border-[#334155] px-3 py-2 text-sm text-[#f8fafc] placeholder:text-[#475569] focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/50"
              />
              <button
                onClick={run}
                disabled={loading}
                className="rounded-lg bg-[#3b82f6] px-4 py-2 text-sm font-medium text-white hover:bg-[#2563eb] disabled:opacity-50 flex items-center gap-2"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                检查
              </button>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-lg bg-red-500/15 border border-red-500/30 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-3">
          {results.map(r => (
            <ResultCard key={r.symbol} result={r} />
          ))}
          {results.length === 0 && (
            <div className="text-center text-[#64748b] py-8">没有返回结果</div>
          )}
        </div>
      )}
    </div>
  );
}
