import { useState } from 'react';
import { factorMineParametric } from '../../api/client';
import DataSourceConfig from './DataSourceConfig';

export default function ParametricTab() {
  const [nBars, setNBars] = useState(3000);
  const [horizon, setHorizon] = useState(5);
  const [icThreshold, setIcThreshold] = useState(0.02);
  const [topN, setTopN] = useState(30);
  const [retrain, setRetrain] = useState(false);
  const [crossStock, setCrossStock] = useState(false);
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
      const result = await factorMineParametric({
        n_bars: nBars,
        horizon,
        ic_threshold: icThreshold,
        top_n: topN,
        retrain,
        cross_stock: crossStock,
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
        <h3 className="text-base font-bold text-[#f8fafc] mb-1">🔍 参数化因子搜索</h3>
        <p className="text-xs text-[#94a3b8] mb-4">
          在预定义模板（MA、RSI、MACD、Bollinger等）上遍历参数网格，评估IC/IR，Bonferroni校正后去相关
        </p>

        <DataSourceConfig
          dataSource={dataSource} setDataSource={setDataSource}
          symbols={symbols} setSymbols={setSymbols}
          startDate={startDate} setStartDate={setStartDate}
          endDate={endDate} setEndDate={setEndDate}
          nBars={nBars} setNBars={setNBars}
        />

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">预测窗口</label>
            <input type="number" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">IC阈值</label>
            <input type="number" step="0.01" value={icThreshold} onChange={(e) => setIcThreshold(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
          <div>
            <label className="text-xs text-[#94a3b8] block mb-1">Top N</label>
            <input type="number" value={topN} onChange={(e) => setTopN(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
        </div>

        <div className="flex items-center gap-4 flex-wrap">
          <label className="flex items-center gap-2 text-sm text-[#cbd5e1]">
            <input type="checkbox" checked={retrain} onChange={(e) => setRetrain(e.target.checked)}
              className="rounded border-[#334155]" />
            发现后自动重训练模型
          </label>
          {dataSource === 'akshare' && (
            <label className="flex items-center gap-2 text-sm text-[#cbd5e1]">
              <input type="checkbox" checked={crossStock} onChange={(e) => setCrossStock(e.target.checked)}
                className="rounded border-[#334155]" />
              跨股票筛选
            </label>
          )}
          <button onClick={handleRun} disabled={running}
            className="rounded-lg bg-[#3b82f6] px-5 py-2 text-sm font-medium text-white hover:bg-[#2563eb] disabled:opacity-50">
            {running ? '⏳ 搜索中...' : '🚀 开始搜索'}
          </button>
        </div>

        {running && (
          <div className="mt-3 text-xs text-[#94a3b8]">
            ⏱️ {dataSource === 'akshare' ? '正在从akshare拉取真实行情数据，首次可能需要几分钟...' : '搜索中...'}
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
