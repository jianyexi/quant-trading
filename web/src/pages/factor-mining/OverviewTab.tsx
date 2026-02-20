import type { FactorRegistry, FactorResults } from '../../api/client';

const STATE_COLORS: Record<string, string> = {
  candidate: '#f59e0b',
  validated: '#3b82f6',
  promoted: '#10b981',
  retired: '#6b7280',
};

const STATE_LABELS: Record<string, string> = {
  candidate: '候选',
  validated: '验证中',
  promoted: '已晋升',
  retired: '已退役',
};

export default function OverviewTab({
  registry,
  results,
}: {
  registry: FactorRegistry | null;
  results: FactorResults | null;
}) {
  const stats = registry?.stats ?? { total_discovered: 0, total_promoted: 0, total_retired: 0 };
  const factorsByState = { candidate: 0, validated: 0, promoted: 0, retired: 0 };
  const factors = Object.entries(registry?.factors ?? {});
  factors.forEach(([, f]) => {
    factorsByState[f.state] = (factorsByState[f.state] || 0) + 1;
  });
  const totalFactors = factors.length;

  // Compute aggregate metrics from promoted factors
  const promoted = factors.filter(([, f]) => f.state === 'promoted');
  const avgIC = promoted.length
    ? promoted.reduce((s, [, f]) => s + f.ic_mean, 0) / promoted.length
    : 0;
  const avgIR = promoted.length
    ? promoted.reduce((s, [, f]) => s + f.ir, 0) / promoted.length
    : 0;
  const avgICPos = promoted.length
    ? promoted.reduce((s, [, f]) => s + f.ic_pos_rate, 0) / promoted.length
    : 0;
  const avgTurnover = promoted.length
    ? promoted.reduce((s, [, f]) => s + f.turnover, 0) / promoted.length
    : 0;

  // All factors sorted by IC descending
  const allSorted = [...factors].sort(([, a], [, b]) => Math.abs(b.ic_mean) - Math.abs(a.ic_mean));

  // Latest report from parametric mining
  const report = results?.parametric?.latest_report as {
    n_selected?: number;
    factors?: Array<{ factor_name: string; ic_mean: number; ir: number; ic_pos_rate: number; turnover: number; decay: number; n_obs: number }>;
  } | null;

  return (
    <div className="space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
        {[
          { label: '注册因子', value: totalFactors, color: '#3b82f6' },
          { label: '已晋升', value: factorsByState.promoted, color: '#10b981' },
          { label: '候选中', value: factorsByState.candidate, color: '#f59e0b' },
          { label: '验证中', value: factorsByState.validated, color: '#6366f1' },
          { label: '已退役', value: factorsByState.retired, color: '#6b7280' },
          { label: '累计发现', value: stats.total_discovered, color: '#8b5cf6' },
          { label: '平均IC', value: avgIC.toFixed(4), color: avgIC > 0.05 ? '#10b981' : '#f59e0b' },
          { label: '平均IR', value: avgIR.toFixed(3), color: avgIR > 0.5 ? '#10b981' : '#f59e0b' },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-[#334155] bg-[#1e293b] p-3">
            <div className="text-xs text-[#94a3b8]">{s.label}</div>
            <div className="text-lg font-bold mt-0.5" style={{ color: s.color }}>
              {s.value}
            </div>
          </div>
        ))}
      </div>

      {/* Pipeline status */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h3 className="text-base font-bold text-[#f8fafc] mb-4">📈 因子生命周期</h3>
        <div className="flex items-center justify-between">
          {(['candidate', 'validated', 'promoted', 'retired'] as const).map((state, i) => (
            <div key={state} className="flex items-center">
              <div className="text-center">
                <div
                  className="w-16 h-16 rounded-full flex items-center justify-center text-xl font-bold mx-auto"
                  style={{ backgroundColor: STATE_COLORS[state] + '20', color: STATE_COLORS[state] }}
                >
                  {factorsByState[state]}
                </div>
                <div className="text-xs text-[#94a3b8] mt-2">{STATE_LABELS[state]}</div>
              </div>
              {i < 3 && (
                <div className="text-[#475569] text-2xl mx-4">→</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Promoted factor quality summary */}
      {promoted.length > 0 && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h3 className="text-base font-bold text-[#f8fafc] mb-3">🏆 已晋升因子质量概览</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
            <div className="rounded-lg bg-[#0f172a] p-3">
              <div className="text-xs text-[#64748b]">平均 |IC|</div>
              <div className={`text-lg font-bold ${avgIC > 0.05 ? 'text-green-400' : 'text-yellow-400'}`}>{avgIC.toFixed(4)}</div>
              <div className="text-xs text-[#475569]">{avgIC > 0.1 ? '优秀' : avgIC > 0.05 ? '良好' : '一般'}</div>
            </div>
            <div className="rounded-lg bg-[#0f172a] p-3">
              <div className="text-xs text-[#64748b]">平均 IR</div>
              <div className={`text-lg font-bold ${avgIR > 0.5 ? 'text-green-400' : 'text-yellow-400'}`}>{avgIR.toFixed(3)}</div>
              <div className="text-xs text-[#475569]">{avgIR > 1.0 ? '优秀' : avgIR > 0.5 ? '良好' : '一般'}</div>
            </div>
            <div className="rounded-lg bg-[#0f172a] p-3">
              <div className="text-xs text-[#64748b]">平均 IC正率</div>
              <div className={`text-lg font-bold ${avgICPos > 0.6 ? 'text-green-400' : 'text-yellow-400'}`}>{(avgICPos * 100).toFixed(1)}%</div>
              <div className="text-xs text-[#475569]">{avgICPos > 0.7 ? '优秀' : avgICPos > 0.55 ? '良好' : '一般'}</div>
            </div>
            <div className="rounded-lg bg-[#0f172a] p-3">
              <div className="text-xs text-[#64748b]">平均换手率</div>
              <div className="text-lg font-bold text-[#cbd5e1]">{avgTurnover.toFixed(3)}</div>
              <div className="text-xs text-[#475569]">{avgTurnover < 0.3 ? '稳定' : '较高'}</div>
            </div>
          </div>
        </div>
      )}

      {/* All factors ranking table */}
      {allSorted.length > 0 && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h3 className="text-base font-bold text-[#f8fafc] mb-3">📊 全部因子排行（按|IC|排序）</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#334155] bg-[#0f172a]">
                  <th className="text-left py-2 px-2 text-[#94a3b8] font-medium text-xs w-8">#</th>
                  <th className="text-left py-2 px-2 text-[#94a3b8] font-medium text-xs">因子ID</th>
                  <th className="text-left py-2 px-2 text-[#94a3b8] font-medium text-xs">状态</th>
                  <th className="text-left py-2 px-2 text-[#94a3b8] font-medium text-xs">来源</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">|IC|</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">IR</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">IC正率</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">换手率</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">衰减</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">验证</th>
                  <th className="text-left py-2 px-2 text-[#94a3b8] font-medium text-xs">表达式</th>
                </tr>
              </thead>
              <tbody>
                {allSorted.map(([id, f], idx) => {
                  const icColor = Math.abs(f.ic_mean) > 0.1 ? 'text-green-400' : Math.abs(f.ic_mean) > 0.05 ? 'text-yellow-400' : 'text-[#cbd5e1]';
                  return (
                    <tr key={id} className="border-b border-[#334155]/30 hover:bg-[#334155]/20">
                      <td className="py-1.5 px-2 text-[#475569] text-xs">{idx + 1}</td>
                      <td className="py-1.5 px-2 text-[#f8fafc] font-mono text-xs">{id}</td>
                      <td className="py-1.5 px-2">
                        <span className="px-1.5 py-0.5 rounded-full text-xs"
                          style={{ backgroundColor: STATE_COLORS[f.state] + '20', color: STATE_COLORS[f.state] }}>
                          {STATE_LABELS[f.state]}
                        </span>
                      </td>
                      <td className="py-1.5 px-2 text-[#94a3b8] text-xs">{f.source === 'gp' ? '🧬 GP' : '🔍 参数'}</td>
                      <td className={`py-1.5 px-2 text-right font-mono text-xs ${icColor}`}>{Math.abs(f.ic_mean).toFixed(4)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-xs text-[#cbd5e1]">{f.ir.toFixed(3)}</td>
                      <td className="py-1.5 px-2 text-right text-xs text-[#cbd5e1]">{(f.ic_pos_rate * 100).toFixed(0)}%</td>
                      <td className="py-1.5 px-2 text-right font-mono text-xs text-[#cbd5e1]">{f.turnover.toFixed(3)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-xs text-[#cbd5e1]">{f.decay.toFixed(1)}</td>
                      <td className="py-1.5 px-2 text-right text-xs text-[#cbd5e1]">{f.validation_count}</td>
                      <td className="py-1.5 px-2 text-[#94a3b8] text-xs truncate max-w-[250px]" title={f.expression}>{f.expression}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Latest mining report */}
      {report?.factors && report.factors.length > 0 && (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h3 className="text-base font-bold text-[#f8fafc] mb-3">📋 最近一次参数化搜索报告</h3>
          <div className="text-xs text-[#94a3b8] mb-3">发现 {report.n_selected ?? report.factors.length} 个通过统计检验的因子</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#334155] bg-[#0f172a]">
                  <th className="text-left py-2 px-2 text-[#94a3b8] font-medium text-xs">因子名</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">IC均值</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">IR</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">IC正率</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">换手率</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">衰减半衰期</th>
                  <th className="text-right py-2 px-2 text-[#94a3b8] font-medium text-xs">样本数</th>
                </tr>
              </thead>
              <tbody>
                {report.factors.map((f) => (
                  <tr key={f.factor_name} className="border-b border-[#334155]/30 hover:bg-[#334155]/20">
                    <td className="py-1.5 px-2 text-[#f8fafc] font-mono text-xs">{f.factor_name}</td>
                    <td className={`py-1.5 px-2 text-right font-mono text-xs ${Math.abs(f.ic_mean) > 0.1 ? 'text-green-400' : 'text-[#cbd5e1]'}`}>
                      {f.ic_mean.toFixed(4)}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-xs text-[#cbd5e1]">{f.ir.toFixed(3)}</td>
                    <td className="py-1.5 px-2 text-right text-xs text-[#cbd5e1]">{(f.ic_pos_rate * 100).toFixed(0)}%</td>
                    <td className="py-1.5 px-2 text-right font-mono text-xs text-[#cbd5e1]">{f.turnover.toFixed(3)}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-xs text-[#cbd5e1]">{f.decay.toFixed(1)}</td>
                    <td className="py-1.5 px-2 text-right text-xs text-[#cbd5e1]">{f.n_obs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Discovered factors summary (Phase 1 + Phase 2) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h3 className="text-base font-bold text-[#f8fafc] mb-3">🔍 Phase 1: 参数化因子</h3>
          <div className="text-sm text-[#94a3b8] mb-2">
            模板×参数网格搜索 · {results?.parametric.features.length ?? 0} 个特征
          </div>
          {results?.parametric.features.length ? (
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {results.parametric.features.map((f) => (
                <div key={f} className="text-xs text-[#cbd5e1] px-2 py-1 bg-[#0f172a] rounded font-mono">
                  {f}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-[#475569]">尚未运行参数化搜索</div>
          )}
        </div>

        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h3 className="text-base font-bold text-[#f8fafc] mb-3">🧬 Phase 2: GP进化因子</h3>
          <div className="text-sm text-[#94a3b8] mb-2">
            遗传编程表达式进化 · {results?.gp.features.length ?? 0} 个因子
          </div>
          {results?.gp.features.length ? (
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {results.gp.features.map((f) => (
                <div key={f.id} className="text-xs text-[#cbd5e1] px-2 py-1 bg-[#0f172a] rounded">
                  <span className="text-[#3b82f6] font-mono">{f.id}</span>
                  <span className="text-[#64748b] ml-2">{f.expression}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-[#475569]">尚未运行GP进化</div>
          )}
        </div>
      </div>
    </div>
  );
}
