import { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { factorIcDecay, type IcDecayResult } from '../../api/client';

/* ── Colour palette for lines ──────────────────────────────────────── */

const LINE_COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#a855f7',
  '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#6366f1',
];

function halfLifeColor(hl: number): string {
  if (hl > 20) return '#22c55e';
  if (hl >= 10) return '#f59e0b';
  return '#ef4444';
}

/* ── Component ─────────────────────────────────────────────────────── */

export default function IcDecayTab() {
  const [data, setData] = useState<IcDecayResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const compute = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await factorIcDecay();
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

  /* Build chart data: one row per horizon */
  const chartData = data
    ? data.horizons.map((h, i) => {
        const row: Record<string, number | string> = { horizon: `${h}d` };
        data.factors.slice(0, 10).forEach((f) => {
          row[f.name] = f.ic_values[i];
        });
        return row;
      })
    : [];

  /* Sort factors by half-life descending for the table */
  const sortedFactors = data
    ? [...data.factors].sort((a, b) => b.half_life - a.half_life)
    : [];

  return (
    <div className="space-y-4">
      {/* Action bar */}
      <div className="flex items-center gap-4">
        <button
          onClick={compute}
          disabled={loading}
          className="px-4 py-2 rounded-lg bg-[#3b82f6] text-white text-sm font-medium hover:bg-[#2563eb] disabled:opacity-50 transition-colors"
        >
          {loading ? '计算中…' : '计算IC衰减'}
        </button>
        {data && (
          <span className="text-xs text-[#94a3b8]">
            {data.factors.length} 个因子 · {data.horizons.length} 个前瞻期
          </span>
        )}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Multi-line chart */}
      {data && chartData.length > 0 && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-4">
          <h3 className="text-sm font-semibold text-[#f8fafc] mb-3">IC 衰减曲线</h3>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="horizon" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: 8,
                  color: '#f8fafc',
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
              {data.factors.slice(0, 10).map((f, idx) => (
                <Line
                  key={f.name}
                  type="monotone"
                  dataKey={f.name}
                  stroke={LINE_COLORS[idx % LINE_COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Half-life table */}
      {data && sortedFactors.length > 0 && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] overflow-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#334155]">
                <th className="text-left px-4 py-3 text-[#94a3b8] font-medium">因子</th>
                <th className="text-right px-4 py-3 text-[#94a3b8] font-medium">Base IC</th>
                <th className="text-right px-4 py-3 text-[#94a3b8] font-medium">半衰期(天)</th>
                {data.horizons.map((h) => (
                  <th key={h} className="text-right px-4 py-3 text-[#94a3b8] font-medium">{h}d</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedFactors.map((f) => (
                <tr key={f.name} className="border-b border-[#334155]/50 hover:bg-[#334155]/30">
                  <td className="px-4 py-2 font-mono text-[#f8fafc] whitespace-nowrap">{f.name}</td>
                  <td className="text-right px-4 py-2 tabular-nums text-[#cbd5e1]">
                    {f.base_ic.toFixed(4)}
                  </td>
                  <td className="text-right px-4 py-2 tabular-nums font-semibold" style={{ color: halfLifeColor(f.half_life) }}>
                    {f.half_life.toFixed(1)}
                  </td>
                  {f.ic_values.map((v, i) => (
                    <td key={i} className="text-right px-4 py-2 tabular-nums text-[#cbd5e1]">
                      {v.toFixed(4)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Key insight */}
      {data && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b]/60 px-4 py-3 text-sm text-[#94a3b8]">
          💡 因子半衰期 &gt; 20 天的因子适合低频策略，&lt; 5 天的适合高频策略
        </div>
      )}
    </div>
  );
}
