import { useState } from 'react';
import { factorCorrelation, type FactorCorrelation } from '../../api/client';

/* ── Colour helpers ────────────────────────────────────────────────── */

function getCorrelationColor(value: number): string {
  if (value > 0) return `rgba(239, 68, 68, ${Math.abs(value)})`;
  if (value < 0) return `rgba(59, 130, 246, ${Math.abs(value)})`;
  return 'transparent';
}

function getTextColor(value: number): string {
  return Math.abs(value) > 0.6 ? '#f8fafc' : '#cbd5e1';
}

/* ── Component ─────────────────────────────────────────────────────── */

export default function CorrelationTab() {
  const [data, setData] = useState<FactorCorrelation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const compute = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await factorCorrelation();
      if (res.error) {
        setError(res.error);
        setData(null);
      } else {
        setData(res);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  /* ── High-correlation pairs ──────────────────────────────────────── */
  const highPairs: { a: string; b: string; corr: number }[] = [];
  if (data) {
    const { factors, matrix } = data;
    for (let i = 0; i < factors.length; i++) {
      for (let j = i + 1; j < factors.length; j++) {
        const v = matrix[i][j];
        if (Math.abs(v) > 0.7) {
          highPairs.push({ a: factors[i], b: factors[j], corr: v });
        }
      }
    }
    highPairs.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));
  }

  return (
    <div className="space-y-4">
      {/* Action bar */}
      <div className="flex items-center gap-4">
        <button
          onClick={compute}
          disabled={loading}
          className="px-4 py-2 rounded-lg bg-[#3b82f6] text-white text-sm font-medium hover:bg-[#2563eb] disabled:opacity-50 transition-colors"
        >
          {loading ? '计算中…' : '计算相关矩阵'}
        </button>
        {data && (
          <span className="text-xs text-[#94a3b8]">
            {data.factors.length} 个因子
          </span>
        )}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Heatmap matrix */}
      {data && (
        <div className="overflow-auto rounded-xl border border-[#334155] bg-[#1e293b]">
          <table className="border-collapse text-xs">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 bg-[#1e293b] px-2 py-1" />
                {data.factors.map((f) => (
                  <th
                    key={f}
                    className="px-1 py-2 font-medium text-[#94a3b8] whitespace-nowrap"
                    style={{ writingMode: 'vertical-rl', maxWidth: 32 }}
                  >
                    {f}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.factors.map((rowName, i) => (
                <tr key={rowName}>
                  <td className="sticky left-0 z-10 bg-[#1e293b] px-2 py-1 text-right font-medium text-[#94a3b8] whitespace-nowrap">
                    {rowName}
                  </td>
                  {data.matrix[i].map((val, j) => {
                    const highlight = i !== j && Math.abs(val) > 0.7;
                    return (
                      <td
                        key={j}
                        className="px-1 py-1 text-center tabular-nums"
                        style={{
                          backgroundColor: getCorrelationColor(val),
                          color: getTextColor(val),
                          border: highlight
                            ? '2px solid #facc15'
                            : '1px solid #334155',
                          minWidth: 40,
                        }}
                        title={`${rowName} × ${data.factors[j]}: ${val.toFixed(4)}`}
                      >
                        {val.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Legend */}
      {data && (
        <div className="flex items-center gap-4 text-xs text-[#94a3b8]">
          <span className="flex items-center gap-1">
            <span
              className="inline-block w-4 h-4 rounded"
              style={{ backgroundColor: 'rgba(59, 130, 246, 0.8)' }}
            />
            负相关
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-4 h-4 rounded bg-[#1e293b] border border-[#334155]" />
            无相关
          </span>
          <span className="flex items-center gap-1">
            <span
              className="inline-block w-4 h-4 rounded"
              style={{ backgroundColor: 'rgba(239, 68, 68, 0.8)' }}
            />
            正相关
          </span>
          <span className="flex items-center gap-1">
            <span
              className="inline-block w-4 h-4 rounded border-2 border-yellow-400"
              style={{ backgroundColor: 'rgba(239, 68, 68, 0.5)' }}
            />
            |corr| &gt; 0.7
          </span>
        </div>
      )}

      {/* High-correlation pairs */}
      {highPairs.length > 0 && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-4">
          <h3 className="text-sm font-semibold text-[#f8fafc] mb-3">
            ⚠️ 高相关因子对 (|corr| &gt; 0.7)
          </h3>
          <div className="space-y-1">
            {highPairs.map(({ a, b, corr }, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 text-sm text-[#cbd5e1]"
              >
                <span className="font-mono text-[#f8fafc]">{a}</span>
                <span className="text-[#64748b]">↔</span>
                <span className="font-mono text-[#f8fafc]">{b}</span>
                <span
                  className={`ml-auto font-mono font-semibold ${
                    corr > 0 ? 'text-red-400' : 'text-blue-400'
                  }`}
                >
                  {corr > 0 ? '+' : ''}
                  {corr.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
