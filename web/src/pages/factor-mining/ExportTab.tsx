import { useState, useEffect } from 'react';
import { factorExportPromoted, type FactorResults } from '../../api/client';
import { useTaskPoller } from '../../hooks/useTaskPoller';

const STORAGE_KEY = 'task_export';

export default function ExportTab({ results }: { results: FactorResults | null }) {
  const [exportRetrain, setExportRetrain] = useState(true);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [showRust, setShowRust] = useState<'p1' | 'gp' | null>(null);

  const { task, startPolling } = useTaskPoller();
  const exporting = task?.status === 'Running';

  useEffect(() => {
    const savedId = sessionStorage.getItem(STORAGE_KEY);
    if (savedId) startPolling(savedId);
  }, [startPolling]);

  useEffect(() => {
    if (!task) return;
    if (task.status === 'Completed') {
      sessionStorage.removeItem(STORAGE_KEY);
      try {
        const parsed = task.result ? JSON.parse(task.result) : null;
        setOutput(parsed?.stdout || task.result || '完成');
      } catch {
        setOutput(task.result || '完成');
      }
    } else if (task.status === 'Failed') {
      sessionStorage.removeItem(STORAGE_KEY);
      setError(task.error || '导出失败');
    }
  }, [task?.status]);

  const handleExport = async () => {
    setError('');
    setOutput('');
    try {
      const result = await factorExportPromoted({ retrain: exportRetrain });
      sessionStorage.setItem(STORAGE_KEY, result.task_id);
      startPolling(result.task_id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '导出失败');
    }
  };

  return (
    <div className="space-y-6">
      {/* Export controls */}
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
          <button onClick={handleExport} disabled={exporting}
            className="rounded-lg bg-[#10b981] px-5 py-2 text-sm font-medium text-white hover:bg-[#059669] disabled:opacity-50">
            {exporting ? '⏳ 导出中...' : '📦 导出晋升因子'}
          </button>
        </div>
      </div>

      {error && <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-sm text-red-400">{error}</div>}

      {output && (
        <div className="rounded-xl border border-[#334155] bg-[#0f172a] p-4">
          <pre className="text-xs text-[#cbd5e1] whitespace-pre-wrap max-h-48 overflow-y-auto font-mono">{output}</pre>
        </div>
      )}

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
    </div>
  );
}
