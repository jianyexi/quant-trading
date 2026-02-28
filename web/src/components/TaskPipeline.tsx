/**
 * Shared UI component for displaying task status, error, and output.
 * Eliminates repeated error/output/running blocks across all task-based pages.
 */
export function TaskOutput({
  running,
  error,
  output,
  progress,
  runningText,
}: {
  running: boolean;
  error: string;
  output: string;
  progress?: string | null;
  runningText?: string;
}) {
  return (
    <>
      {running && (
        <div className="text-xs text-[#94a3b8] mt-3">
          ⏳ {progress || runningText || '任务运行中，请稍候...'}
        </div>
      )}
      {error && (
        <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-sm text-red-400 mt-3">
          {error}
        </div>
      )}
      {output && (
        <div className="rounded-xl border border-[#334155] bg-[#0f172a] p-4 mt-3">
          <h4 className="text-sm font-bold text-[#f8fafc] mb-2">输出</h4>
          <pre className="text-xs text-[#cbd5e1] whitespace-pre-wrap max-h-96 overflow-y-auto font-mono leading-relaxed">
            {output}
          </pre>
        </div>
      )}
    </>
  );
}

/**
 * Shared numeric parameter grid. Renders labeled number inputs in a responsive grid.
 */
export interface ParamField {
  key: string;
  label: string;
  value: number;
  step?: number;
  min?: number;
  max?: number;
}

export function ParamGrid({
  fields,
  onChange,
  columns = 3,
}: {
  fields: ParamField[];
  onChange: (key: string, value: number) => void;
  columns?: 2 | 3 | 4;
}) {
  const colClass =
    columns === 2 ? 'grid-cols-2' :
    columns === 4 ? 'grid-cols-2 sm:grid-cols-4' :
    'grid-cols-2 sm:grid-cols-3';

  return (
    <div className={`grid ${colClass} gap-4 mb-4`}>
      {fields.map((f) => (
        <div key={f.key}>
          <label className="text-xs text-[#94a3b8] block mb-1">{f.label}</label>
          <input
            type="number"
            value={f.value}
            step={f.step}
            min={f.min}
            max={f.max}
            onChange={(e) => onChange(f.key, Number(e.target.value))}
            className="w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none"
          />
        </div>
      ))}
    </div>
  );
}
