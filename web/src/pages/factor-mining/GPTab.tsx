import { useState } from 'react';
import { factorMineGP } from '../../api/client';
import DataSourceConfig from './DataSourceConfig';

export default function GPTab() {
  const [nBars, setNBars] = useState(3000);
  const [popSize, setPopSize] = useState(200);
  const [generations, setGenerations] = useState(30);
  const [maxDepth, setMaxDepth] = useState(6);
  const [horizon, setHorizon] = useState(5);
  const [retrain, setRetrain] = useState(false);
  const [dataSource, setDataSource] = useState('synthetic');
  const [symbols, setSymbols] = useState('');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');

  const handleRun = async () => {
    setRunning(true);
    setError('');
    setOutput('');
    try {
      const result = await factorMineGP({
        n_bars: nBars,
        pop_size: popSize,
        generations,
        max_depth: maxDepth,
        horizon,
        retrain,
        data_source: dataSource,
        symbols: symbols || undefined,
        start_date: startDate,
        end_date: endDate,
      });
      setOutput(result.stdout || '完成');
      if (result.stderr) setOutput((prev) => prev + '\n\n--- stderr ---\n' + result.stderr);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '请求失败');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h3 className="text-base font-bold text-[#f8fafc] mb-1">🧬 遗传编程因子进化</h3>
        <p className="text-xs text-[#94a3b8] mb-4">
          进化表达式树发现新因子：随机生成→交叉/变异→IC适应度选择→自动注册到因子注册表
        </p>

        <DataSourceConfig
          dataSource={dataSource} setDataSource={setDataSource}
          symbols={symbols} setSymbols={setSymbols}
          startDate={startDate} setStartDate={setStartDate}
          endDate={endDate} setEndDate={setEndDate}
          nBars={nBars} setNBars={setNBars}
        />

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">种群大小</label>
            <input type="number" value={popSize} onChange={(e) => setPopSize(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">迭代代数</label>
            <input type="number" value={generations} onChange={(e) => setGenerations(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">最大树深</label>
            <input type="number" value={maxDepth} onChange={(e) => setMaxDepth(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">预测窗口</label>
            <input type="number" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
        </div>

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-[#cbd5e1]">
            <input type="checkbox" checked={retrain} onChange={(e) => setRetrain(e.target.checked)}
              className="rounded border-[#334155]" />
            发现后重训练模型
          </label>
          <button onClick={handleRun} disabled={running}
            className="rounded-lg bg-[#8b5cf6] px-5 py-2 text-sm font-medium text-white hover:bg-[#7c3aed] disabled:opacity-50">
            {running ? '⏳ 进化中...' : '🧬 开始进化'}
          </button>
        </div>

        {running && (
          <div className="mt-3 text-xs text-[#94a3b8]">
            ⏱️ {dataSource === 'akshare' ? '正在从akshare拉取真实行情数据并进化，可能需要较长时间...' : 'GP进化可能需要数分钟，取决于种群大小和代数...'}
          </div>
        )}
      </div>

      {error && <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3 text-sm text-red-400">{error}</div>}

      {output && (
        <div className="rounded-xl border border-[#334155] bg-[#0f172a] p-4">
          <h4 className="text-sm font-bold text-[#f8fafc] mb-2">输出</h4>
          <pre className="text-xs text-[#cbd5e1] whitespace-pre-wrap max-h-96 overflow-y-auto font-mono leading-relaxed">{output}</pre>
        </div>
      )}
    </div>
  );
}
