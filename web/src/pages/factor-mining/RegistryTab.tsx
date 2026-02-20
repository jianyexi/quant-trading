import { useState } from 'react';
import {
  factorRegistryManage,
  type FactorRegistry,
  type FactorRegistryEntry,
} from '../../api/client';

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

function FactorRow({ id, factor: f }: { id: string; factor: FactorRegistryEntry }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <tr className="border-b border-[#334155]/50 hover:bg-[#334155]/20 cursor-pointer"
        onClick={() => setExpanded(!expanded)}>
        <td className="py-2 px-3 text-[#f8fafc] font-mono text-xs">{id}</td>
        <td className="py-2 px-3">
          <span className="px-2 py-0.5 rounded-full text-xs font-medium"
            style={{ backgroundColor: STATE_COLORS[f.state] + '20', color: STATE_COLORS[f.state] }}>
            {STATE_LABELS[f.state]}
          </span>
        </td>
        <td className="py-2 px-3 text-right text-[#cbd5e1] font-mono text-xs">{f.ic_mean.toFixed(4)}</td>
        <td className="py-2 px-3 text-right text-[#cbd5e1] font-mono text-xs">{f.ir.toFixed(3)}</td>
        <td className="py-2 px-3 text-right text-[#cbd5e1] text-xs">{f.validation_count}</td>
        <td className="py-2 px-3 text-[#94a3b8] text-xs truncate max-w-[300px]">{f.expression}</td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={6} className="bg-[#0f172a] px-4 py-3">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs mb-3">
              <div>
                <span className="text-[#64748b]">来源：</span>
                <span className="text-[#cbd5e1] ml-1">{f.source}</span>
              </div>
              <div>
                <span className="text-[#64748b]">树大小：</span>
                <span className="text-[#cbd5e1] ml-1">{f.tree_size}</span>
              </div>
              <div>
                <span className="text-[#64748b]">IC正率：</span>
                <span className="text-[#cbd5e1] ml-1">{(f.ic_pos_rate * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span className="text-[#64748b]">换手率：</span>
                <span className="text-[#cbd5e1] ml-1">{f.turnover.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-[#64748b]">衰减：</span>
                <span className="text-[#cbd5e1] ml-1">{f.decay.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[#64748b]">失败次数：</span>
                <span className="text-[#cbd5e1] ml-1">{f.fail_count}</span>
              </div>
              <div>
                <span className="text-[#64748b]">创建：</span>
                <span className="text-[#cbd5e1] ml-1">{f.created?.slice(0, 16)}</span>
              </div>
              <div>
                <span className="text-[#64748b]">最后验证：</span>
                <span className="text-[#cbd5e1] ml-1">{f.last_validated?.slice(0, 16) || '-'}</span>
              </div>
            </div>
            <div className="text-xs">
              <span className="text-[#64748b]">完整表达式：</span>
              <code className="text-[#3b82f6] bg-[#1e293b] px-2 py-0.5 rounded ml-1">{f.expression}</code>
            </div>
            {f.ic_history.length > 1 && (
              <div className="mt-3">
                <span className="text-[#64748b] text-xs">IC历史：</span>
                <div className="flex gap-1 mt-1 flex-wrap">
                  {f.ic_history.slice(-10).map((h, i) => (
                    <span key={i}
                      className={`px-1.5 py-0.5 rounded text-xs font-mono ${
                        Math.abs(h.ic) >= 0.03 ? 'bg-green-500/20 text-green-400' : 'bg-[#334155] text-[#94a3b8]'
                      }`}>
                      {h.ic.toFixed(4)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

export default function RegistryTab({
  registry,
  onRefresh,
}: {
  registry: FactorRegistry | null;
  onRefresh: () => void;
}) {
  const [managing, setManaging] = useState(false);
  const [manageOutput, setManageOutput] = useState('');
  const [filter, setFilter] = useState<string>('all');

  const handleManage = async () => {
    setManaging(true);
    setManageOutput('');
    try {
      const result = await factorRegistryManage({ n_bars: 3000 });
      setManageOutput(result.stdout || '完成');
      onRefresh();
    } catch (e: unknown) {
      setManageOutput(e instanceof Error ? e.message : '失败');
    } finally {
      setManaging(false);
    }
  };

  const factors = Object.entries(registry?.factors ?? {});
  const filtered = filter === 'all'
    ? factors
    : factors.filter(([, f]) => f.state === filter);

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          {['all', 'candidate', 'validated', 'promoted', 'retired'].map((s) => (
            <button key={s} onClick={() => setFilter(s)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                filter === s
                  ? 'bg-[#3b82f6] text-white'
                  : 'bg-[#334155] text-[#94a3b8] hover:bg-[#475569]'
              }`}>
              {s === 'all' ? '全部' : STATE_LABELS[s]}
              {s !== 'all' && (
                <span className="ml-1 opacity-70">
                  ({factors.filter(([, f]) => f.state === s).length})
                </span>
              )}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <button onClick={onRefresh}
            className="px-3 py-1.5 rounded-lg bg-[#334155] text-[#94a3b8] text-xs hover:bg-[#475569]">
            🔄 刷新
          </button>
          <button onClick={handleManage} disabled={managing}
            className="px-3 py-1.5 rounded-lg bg-[#10b981] text-white text-xs font-medium hover:bg-[#059669] disabled:opacity-50">
            {managing ? '⏳ 管理中...' : '⚙️ 运行生命周期管理'}
          </button>
        </div>
      </div>

      {manageOutput && (
        <div className="rounded-xl border border-[#334155] bg-[#0f172a] p-4">
          <pre className="text-xs text-[#cbd5e1] whitespace-pre-wrap max-h-48 overflow-y-auto font-mono">{manageOutput}</pre>
        </div>
      )}

      {/* Factor table */}
      {filtered.length === 0 ? (
        <div className="text-center py-12 text-[#475569]">
          <div className="text-4xl mb-2">📋</div>
          <div className="text-sm">暂无因子记录</div>
          <div className="text-xs mt-1">运行GP进化或参数化搜索来发现因子</div>
        </div>
      ) : (
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#334155] bg-[#0f172a]">
                <th className="text-left py-2.5 px-3 text-[#94a3b8] font-medium text-xs">ID</th>
                <th className="text-left py-2.5 px-3 text-[#94a3b8] font-medium text-xs">状态</th>
                <th className="text-right py-2.5 px-3 text-[#94a3b8] font-medium text-xs">IC</th>
                <th className="text-right py-2.5 px-3 text-[#94a3b8] font-medium text-xs">IR</th>
                <th className="text-right py-2.5 px-3 text-[#94a3b8] font-medium text-xs">验证次数</th>
                <th className="text-left py-2.5 px-3 text-[#94a3b8] font-medium text-xs">表达式</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(([id, f]) => (
                <FactorRow key={id} id={id} factor={f} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
