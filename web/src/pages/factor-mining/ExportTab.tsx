import { useState } from 'react';
import { factorExportPromoted, type FactorResults } from '../../api/client';
import { useTaskManager } from '../../hooks/useTaskManager';
import { TaskOutput } from '../../components/TaskPipeline';

export default function ExportTab({ results }: { results: FactorResults | null }) {
  const [exportRetrain, setExportRetrain] = useState(true);
  const [showRust, setShowRust] = useState<'p1' | 'gp' | null>(null);

  const tm = useTaskManager('task_export');

  const handleExport = () => tm.submit(() => factorExportPromoted({ retrain: exportRetrain }));

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h3 className="text-base font-bold text-[#f8fafc] mb-1">📦 导出已晋升因子</h3>
        <p className="text-xs text-[#94a3b8] mb-4">
          将所有状态为"已晋升"的因子导出为特征列表、Rust代码片段和因子数据，可选重训练ML模型
        </p>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-[#cbd5e1]">
            <input type="checkbox" checked={exportRetrain} onChange={(e) => setExportRetrain(e.target.checked)}
              className="rounded border-[#334155]" />
            导出后重训练模型
          </label>
          <button onClick={handleExport} disabled={tm.running}
            className="rounded-lg bg-[#10b981] px-5 py-2 text-sm font-medium text-white hover:bg-[#059669] disabled:opacity-50">
            {tm.running ? '⏳ 导出中...' : '📦 导出晋升因子'}
          </button>
        </div>

        <TaskOutput {...tm} />
      </div>

      {/* Rust code snippets */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h3 className="text-base font-bold text-[#f8fafc] mb-3">🦀 Rust集成代码</h3>
        <p className="text-xs text-[#94a3b8] mb-4">
          自动生成的Rust代码片段，可集成到fast_factors.rs中进行增量计算
        </p>
        <div className="flex gap-2 mb-3">
          <button onClick={() => setShowRust(showRust === 'p1' ? null : 'p1')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium ${
              showRust === 'p1' ? 'bg-[#3b82f6] text-white' : 'bg-[#334155] text-[#94a3b8]'
            }`}>
            Phase 1 参数化
          </button>
          <button onClick={() => setShowRust(showRust === 'gp' ? null : 'gp')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium ${
              showRust === 'gp' ? 'bg-[#8b5cf6] text-white' : 'bg-[#334155] text-[#94a3b8]'
            }`}>
            Phase 2 GP
          </button>
        </div>
        {showRust && (
          <pre className="text-xs text-[#cbd5e1] bg-[#0f172a] rounded-lg p-4 max-h-64 overflow-y-auto font-mono whitespace-pre-wrap">
            {showRust === 'p1' ? (results?.parametric.rust_snippet || '尚未生成') : (results?.gp.rust_snippet || '尚未生成')}
          </pre>
        )}
      </div>

      {/* Backtest with mined factors */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h3 className="text-base font-bold text-[#f8fafc] mb-1">🧪 因子回测</h3>
        <p className="text-xs text-[#94a3b8] mb-4">
          使用已挖掘的因子进行策略回测验证
        </p>
        <button
          onClick={() => {
            localStorage.setItem('quant-backtest-strategy', 'ml_factor');
            window.location.href = '/backtest';
          }}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg flex items-center gap-2 text-sm font-medium"
        >
          🧪 用挖掘因子回测
        </button>
      </div>
    </div>
  );
}
