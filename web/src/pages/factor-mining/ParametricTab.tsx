import { useState } from 'react';
import { factorMineParametric } from '../../api/client';
import { useTaskManager } from '../../hooks/useTaskManager';
import { TaskOutput, ParamGrid } from '../../components/TaskPipeline';
import DataSourceConfig from './DataSourceConfig';

export default function ParametricTab() {
  const [params, setParams] = useState({
    nBars: 3000, horizon: 5, icThreshold: 0.02, topN: 30,
  });
  const [retrain, setRetrain] = useState(false);
  const [crossStock, setCrossStock] = useState(false);
  const [dataSource, setDataSource] = useState('akshare');
  const [symbols, setSymbols] = useState('');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');

  const tm = useTaskManager('task_parametric');
  const setP = (k: string, v: number) => setParams(p => ({ ...p, [k]: v }));

  const handleRun = () => tm.submit(() => factorMineParametric({
    n_bars: params.nBars, horizon: params.horizon,
    ic_threshold: params.icThreshold, top_n: params.topN,
    retrain, cross_stock: crossStock,
    data_source: dataSource, symbols: symbols || undefined, start_date: startDate, end_date: endDate,
  }));

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
          nBars={params.nBars} setNBars={(v) => setP('nBars', v)}
        />

        <ParamGrid fields={[
          { key: 'horizon', label: '预测窗口', value: params.horizon },
          { key: 'icThreshold', label: 'IC阈值', value: params.icThreshold, step: 0.01 },
          { key: 'topN', label: 'Top N', value: params.topN },
        ]} onChange={setP} columns={4} />

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
          <button onClick={handleRun} disabled={tm.running}
            className="rounded-lg bg-[#3b82f6] px-5 py-2 text-sm font-medium text-white hover:bg-[#2563eb] disabled:opacity-50">
            {tm.running ? '⏳ 搜索中...' : '🚀 开始搜索'}
          </button>
        </div>

        <TaskOutput {...tm} runningText={dataSource === 'akshare' ? '正在从akshare拉取真实行情数据，首次可能需要几分钟...' : '搜索中...'} />
      </div>
    </div>
  );
}
