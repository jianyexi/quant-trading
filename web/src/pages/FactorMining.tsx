import { useState, useEffect, useCallback } from 'react';
import {
  factorMineParametric,
  factorMineGP,
  factorRegistryGet,
  factorRegistryManage,
  factorExportPromoted,
  factorResults,
  type FactorRegistry,
  type FactorRegistryEntry,
  type FactorResults,
} from '../api/client';

/* ── Tab types ───────────────────────────────────────────────────── */
type Tab = 'overview' | 'parametric' | 'gp' | 'registry' | 'export';

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'overview', label: '总览', icon: '📊' },
  { id: 'parametric', label: '参数化搜索', icon: '🔍' },
  { id: 'gp', label: 'GP进化', icon: '🧬' },
  { id: 'registry', label: '因子注册表', icon: '📋' },
  { id: 'export', label: '导出集成', icon: '📦' },
];

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

const DEFAULT_SYMBOLS = '600519,000858,300750,600036,601318,002415,000651,600276';

/* ── Data Source Config (shared) ─────────────────────────────────── */
function DataSourceConfig({
  dataSource, setDataSource,
  symbols, setSymbols,
  startDate, setStartDate,
  endDate, setEndDate,
  nBars, setNBars,
}: {
  dataSource: string; setDataSource: (v: string) => void;
  symbols: string; setSymbols: (v: string) => void;
  startDate: string; setStartDate: (v: string) => void;
  endDate: string; setEndDate: (v: string) => void;
  nBars: number; setNBars: (v: number) => void;
}) {
  return (
    <div className="rounded-lg border border-[#334155] bg-[#0f172a] p-4 mb-4">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-xs text-[#94a3b8] font-semibold">数据来源</span>
        {['synthetic', 'akshare'].map((src) => (
          <button key={src} onClick={() => setDataSource(src)}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
              dataSource === src
                ? 'bg-[#3b82f6] text-white'
                : 'bg-[#334155] text-[#94a3b8] hover:bg-[#475569]'
            }`}>
            {src === 'synthetic' ? '📊 模拟数据' : '📡 真实行情 (akshare)'}
          </button>
        ))}
      </div>

      {dataSource === 'synthetic' ? (
        <div className="grid grid-cols-4 gap-3">
          <div>
            <label className="text-xs text-[#64748b] block mb-1">数据量 (bars)</label>
            <input type="number" value={nBars} onChange={(e) => setNBars(Number(e.target.value))}
              className="w-full rounded-lg border border-[#334155] bg-[#1e293b] px-3 py-1.5 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div>
            <label className="text-xs text-[#64748b] block mb-1">
              股票代码 <span className="text-[#475569]">(逗号分隔，留空用默认20只)</span>
            </label>
            <input type="text" value={symbols} onChange={(e) => setSymbols(e.target.value)}
              placeholder={DEFAULT_SYMBOLS}
              className="w-full rounded-lg border border-[#334155] bg-[#1e293b] px-3 py-1.5 text-sm text-[#f8fafc] placeholder-[#475569] focus:border-[#3b82f6] focus:outline-none font-mono" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-[#64748b] block mb-1">开始日期</label>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)}
                className="w-full rounded-lg border border-[#334155] bg-[#1e293b] px-3 py-1.5 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
            </div>
            <div>
              <label className="text-xs text-[#64748b] block mb-1">结束日期</label>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)}
                className="w-full rounded-lg border border-[#334155] bg-[#1e293b] px-3 py-1.5 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Overview Tab ────────────────────────────────────────────────── */
function OverviewTab({
  registry,
  results,
}: {
  registry: FactorRegistry | null;
  results: FactorResults | null;
}) {
  const stats = registry?.stats ?? { total_discovered: 0, total_promoted: 0, total_retired: 0 };
  const factorsByState = { candidate: 0, validated: 0, promoted: 0, retired: 0 };
  if (registry) {
    Object.values(registry.factors).forEach((f) => {
      factorsByState[f.state] = (factorsByState[f.state] || 0) + 1;
    });
  }
  const totalFactors = Object.keys(registry?.factors ?? {}).length;

  return (
    <div className="space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: '注册因子', value: totalFactors, color: '#3b82f6' },
          { label: '已晋升', value: factorsByState.promoted, color: '#10b981' },
          { label: '累计发现', value: stats.total_discovered, color: '#8b5cf6' },
          { label: '已退役', value: stats.total_retired, color: '#6b7280' },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-[#334155] bg-[#1e293b] p-4">
            <div className="text-xs text-[#94a3b8]">{s.label}</div>
            <div className="text-2xl font-bold mt-1" style={{ color: s.color }}>
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

      {/* Discovered factors summary */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
          <h3 className="text-base font-bold text-[#f8fafc] mb-3">🔍 Phase 1: 参数化因子</h3>
          <div className="text-sm text-[#94a3b8] mb-2">
            通过模板×参数网格搜索发现的因子
          </div>
          {results?.parametric.features.length ? (
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {results.parametric.features.map((f) => (
                <div key={f} className="text-xs text-[#cbd5e1] px-2 py-1 bg-[#0f172a] rounded">
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
            通过遗传编程进化发现的因子
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

/* ── Parametric Mining Tab ───────────────────────────────────────── */
function ParametricTab() {
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

/* ── GP Evolution Tab ────────────────────────────────────────────── */
function GPTab() {
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

/* ── Registry Tab ────────────────────────────────────────────────── */
function RegistryTab({
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

/* ── Export Tab ───────────────────────────────────────────────────── */
function ExportTab({ results }: { results: FactorResults | null }) {
  const [exporting, setExporting] = useState(false);
  const [exportRetrain, setExportRetrain] = useState(true);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [showRust, setShowRust] = useState<'p1' | 'gp' | null>(null);

  const handleExport = async () => {
    setExporting(true);
    setError('');
    setOutput('');
    try {
      const result = await factorExportPromoted({ retrain: exportRetrain });
      setOutput(result.stdout || '完成');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '导出失败');
    } finally {
      setExporting(false);
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

/* ── Main Page ───────────────────────────────────────────────────── */
export default function FactorMining() {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [registry, setRegistry] = useState<FactorRegistry | null>(null);
  const [results, setResults] = useState<FactorResults | null>(null);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    try {
      const [reg, res] = await Promise.all([factorRegistryGet(), factorResults()]);
      setRegistry(reg);
      setResults(res);
    } catch {
      // Registry may not exist yet
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#3b82f6]" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-[#f8fafc]">🧬 因子挖掘</h1>
        <p className="text-sm text-[#94a3b8] mt-1">
          自动发现、进化、验证和管理交易因子 · 参数化搜索 + 遗传编程 + 生命周期管理
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-[#1e293b] rounded-xl p-1 border border-[#334155]">
        {TABS.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-[#3b82f6] text-white'
                : 'text-[#94a3b8] hover:bg-[#334155] hover:text-[#f8fafc]'
            }`}>
            <span>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'overview' && <OverviewTab registry={registry} results={results} />}
      {activeTab === 'parametric' && <ParametricTab />}
      {activeTab === 'gp' && <GPTab />}
      {activeTab === 'registry' && <RegistryTab registry={registry} onRefresh={loadData} />}
      {activeTab === 'export' && <ExportTab results={results} />}
    </div>
  );
}
