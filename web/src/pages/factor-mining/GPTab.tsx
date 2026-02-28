import { useState } from 'react';
import { factorMineGP } from '../../api/client';
import { useTaskManager } from '../../hooks/useTaskManager';
import { TaskOutput, ParamGrid } from '../../components/TaskPipeline';
import DataSourceConfig from './DataSourceConfig';

export default function GPTab() {
  const [params, setParams] = useState({
    nBars: 3000, popSize: 200, generations: 30, maxDepth: 6, horizon: 5,
  });
  const [retrain, setRetrain] = useState(false);
  const [dataSource, setDataSource] = useState('akshare');
  const [symbols, setSymbols] = useState('');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');

  const tm = useTaskManager('task_gp');
  const setP = (k: string, v: number) => setParams(p => ({ ...p, [k]: v }));

  const handleRun = () => tm.submit(() => factorMineGP({
    n_bars: params.nBars, pop_size: params.popSize, generations: params.generations,
    max_depth: params.maxDepth, horizon: params.horizon, retrain,
    data_source: dataSource, symbols: symbols || undefined, start_date: startDate, end_date: endDate,
  }));

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
          nBars={params.nBars} setNBars={(v) => setP('nBars', v)}
        />

        <ParamGrid fields={[
          { key: 'popSize', label: '种群大小', value: params.popSize },
          { key: 'generations', label: '迭代代数', value: params.generations },
          { key: 'maxDepth', label: '最大树深', value: params.maxDepth },
          { key: 'horizon', label: '预测窗口', value: params.horizon },
        ]} onChange={setP} />

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-[#cbd5e1]">
            <input type="checkbox" checked={retrain} onChange={(e) => setRetrain(e.target.checked)}
              className="rounded border-[#334155]" />
            发现后重训练模型
          </label>
          <button onClick={handleRun} disabled={tm.running}
            className="rounded-lg bg-[#8b5cf6] px-5 py-2 text-sm font-medium text-white hover:bg-[#7c3aed] disabled:opacity-50">
            {tm.running ? '⏳ 进化中...' : '🧬 开始进化'}
          </button>
        </div>

        <TaskOutput {...tm} runningText={dataSource === 'akshare' ? '正在从akshare拉取真实行情数据并进化，可能需要较长时间...' : 'GP进化可能需要数分钟...'} />
      </div>
    </div>
  );
}
