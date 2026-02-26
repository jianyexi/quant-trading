import { useState, useCallback, useEffect, useRef } from 'react';
import type { StrategyConfig, StrategyParam } from '../types';
import { runBacktest, saveStrategyConfig, loadStrategyConfig, mlRetrain, mlModelInfo, getTask, type ModelInfo, type RetrainOptions } from '../api/client';
import { Save, Upload, Play, RotateCcw, Brain, Loader2, CheckCircle, AlertCircle, Zap, Database } from 'lucide-react';

const STRATEGIES: StrategyConfig[] = [
  {
    name: 'DualMaCrossover',
    displayName: '双均线交叉 (SMA Cross)',
    description: '快速均线上穿慢速均线时买入，死叉时卖出。经典趋势跟踪策略。',
    parameters: [
      { key: 'fast_period', label: '快线周期', type: 'number', default: 5, min: 2, max: 60, step: 1 },
      { key: 'slow_period', label: '慢线周期', type: 'number', default: 20, min: 5, max: 240, step: 1 },
    ],
  },
  {
    name: 'RsiMeanReversion',
    displayName: 'RSI 均值回归',
    description: 'RSI低于超卖阈值时买入，高于超买阈值时卖出。适合震荡行情。',
    parameters: [
      { key: 'period', label: 'RSI周期', type: 'number', default: 14, min: 2, max: 50, step: 1 },
      { key: 'oversold', label: '超卖阈值', type: 'number', default: 30, min: 10, max: 40, step: 1 },
      { key: 'overbought', label: '超买阈值', type: 'number', default: 70, min: 60, max: 90, step: 1 },
    ],
  },
  {
    name: 'MacdMomentum',
    displayName: 'MACD 动量策略',
    description: 'MACD柱状图上穿零轴时买入，下穿时卖出。动量驱动策略。',
    parameters: [
      { key: 'fast_period', label: '快线EMA', type: 'number', default: 12, min: 2, max: 50, step: 1 },
      { key: 'slow_period', label: '慢线EMA', type: 'number', default: 26, min: 5, max: 100, step: 1 },
      { key: 'signal_period', label: '信号线', type: 'number', default: 9, min: 2, max: 30, step: 1 },
    ],
  },
  {
    name: 'MultiFactorModel',
    displayName: '多因子模型',
    description: '6因子综合评分: 趋势+动量+波动率+KDJ+量价+价格行为。动态因子权重自适应调整。',
    parameters: [
      { key: 'buy_threshold', label: '买入阈值', type: 'number', default: 0.30, min: 0.1, max: 0.6, step: 0.05 },
      { key: 'sell_threshold', label: '卖出阈值 (负值)', type: 'number', default: -0.30, min: -0.6, max: -0.1, step: 0.05 },
    ],
  },
  {
    name: 'SentimentAware',
    displayName: '舆情增强策略',
    description: '基于舆情数据增强的多因子策略，根据市场情绪调整交易信号强度。利好放大买入，利空抑制买入。',
    parameters: [
      { key: 'sentiment_weight', label: '舆情权重', type: 'number', default: 0.20, min: 0.05, max: 0.50, step: 0.05 },
      { key: 'min_items', label: '最少舆情条数', type: 'number', default: 3, min: 1, max: 20, step: 1 },
    ],
  },
  {
    name: 'MlFactor',
    displayName: 'ML因子模型',
    description: '24维特征工程 + GPU机器学习推理。支持LightGBM/XGBoost/CatBoost/LSTM/Transformer多算法竞争训练。',
    parameters: [
      { key: 'buy_threshold', label: 'ML买入阈值', type: 'number', default: 0.60, min: 0.50, max: 0.80, step: 0.05 },
      { key: 'sell_threshold', label: 'ML卖出阈值', type: 'number', default: 0.35, min: 0.20, max: 0.50, step: 0.05 },
    ],
  },
];

interface TradingConfig {
  initialCapital: number;
  commissionRate: number;
  symbol: string;
  startDate: string;
  endDate: string;
}

interface SavedConfig {
  selectedStrategy: string;
  paramValues: Record<string, Record<string, number>>;
  tradingConfig: TradingConfig;
}

function defaultParamValues(): Record<string, Record<string, number>> {
  const values: Record<string, Record<string, number>> = {};
  for (const s of STRATEGIES) {
    values[s.name] = {};
    for (const p of s.parameters) {
      values[s.name][p.key] = p.default as number;
    }
  }
  return values;
}

function defaultTradingConfig(): TradingConfig {
  return {
    initialCapital: 1000000,
    commissionRate: 0.025,
    symbol: '600519.SH',
    startDate: '2023-01-01',
    endDate: '2024-01-01',
  };
}

const ALGO_LABELS: Record<string, { label: string; desc: string; badge: string }> = {
  lgb: { label: 'LightGBM', desc: '梯度提升树 — 速度快，精度高', badge: 'bg-green-600/20 text-green-400' },
  xgb: { label: 'XGBoost', desc: '极端梯度提升 — 竞赛冠军模型', badge: 'bg-blue-600/20 text-blue-400' },
  catboost: { label: 'CatBoost', desc: '类别特征自动处理 — 鲁棒性强', badge: 'bg-purple-600/20 text-purple-400' },
  lstm: { label: 'LSTM', desc: '长短期记忆网络 — 序列建模', badge: 'bg-orange-600/20 text-orange-400' },
  transformer: { label: 'Transformer', desc: '注意力机制 — 捕捉长距离依赖', badge: 'bg-red-600/20 text-red-400' },
};

export default function StrategyConfigPage() {
  const [selectedStrategy, setSelectedStrategy] = useState<string>(STRATEGIES[0].name);
  const [paramValues, setParamValues] = useState<Record<string, Record<string, number>>>(defaultParamValues);
  const [tradingConfig, setTradingConfig] = useState<TradingConfig>(defaultTradingConfig);
  const [status, setStatus] = useState<{ text: string; type: 'info' | 'success' | 'error' } | null>(null);
  const [saving, setSaving] = useState(false);
  const [modelInfoData, setModelInfoData] = useState<ModelInfo | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [selectedAlgos, setSelectedAlgos] = useState<string[]>(['lgb', 'xgb', 'catboost']);
  const [trainDataSource, setTrainDataSource] = useState<'synthetic' | 'akshare'>('akshare');
  const [trainSymbols, setTrainSymbols] = useState('600519,000858,000001,600036,300750,002594,601318,600276,000333,601888,600030,601166,600900,000568,600809,601899,600031,600309,300059,600887,000651,002415,300760,601398,601288,600438,002460,603259,600690,601669');
  const [trainStartDate, setTrainStartDate] = useState('2020-01-01');
  const [trainEndDate, setTrainEndDate] = useState('2024-12-31');
  const [trainHorizon, setTrainHorizon] = useState(5);
  const [trainThreshold, setTrainThreshold] = useState(0.01);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const activeStrategy = STRATEGIES.find((s) => s.name === selectedStrategy)!;

  const stopPolling = useCallback(() => {
    if (pollTimer.current) { clearInterval(pollTimer.current); pollTimer.current = null; }
  }, []);

  const pollTask = useCallback(async (taskId: string) => {
    try {
      const t = await getTask(taskId);
      if (t.status === 'Completed') {
        sessionStorage.removeItem('task_retrain');
        setRetraining(false);
        stopPolling();
        showStatus('模型训练完成！', 'success');
        fetchModelInfo();
      } else if (t.status === 'Failed') {
        sessionStorage.removeItem('task_retrain');
        setRetraining(false);
        stopPolling();
        showStatus(t.error || '训练失败，请检查日志', 'error');
      } else if (t.progress) {
        showStatus(t.progress, 'info');
      }
    } catch { /* transient error, keep polling */ }
  }, []);

  const startTaskPolling = useCallback((taskId: string) => {
    setRetraining(true);
    stopPolling();
    pollTask(taskId);
    pollTimer.current = setInterval(() => pollTask(taskId), 2000);
  }, [pollTask, stopPolling]);

  // Load config from server on mount + restore active task
  useEffect(() => {
    loadFromServer();
    fetchModelInfo();
    const savedTaskId = sessionStorage.getItem('task_retrain');
    if (savedTaskId) startTaskPolling(savedTaskId);
    return stopPolling;
  }, []);

  const showStatus = (text: string, type: 'info' | 'success' | 'error' = 'info') => {
    setStatus({ text, type });
    setTimeout(() => setStatus(null), 3000);
  };

  const fetchModelInfo = async () => {
    try {
      const info = await mlModelInfo();
      setModelInfoData(info);
    } catch { /* ignore */ }
  };

  const setParam = useCallback(
    (key: string, value: number) => {
      setParamValues((prev) => ({
        ...prev,
        [selectedStrategy]: { ...prev[selectedStrategy], [key]: value },
      }));
    },
    [selectedStrategy],
  );

  const resetDefaults = () => {
    const defaults: Record<string, number> = {};
    for (const p of activeStrategy.parameters) {
      defaults[p.key] = p.default as number;
    }
    setParamValues((prev) => ({ ...prev, [selectedStrategy]: defaults }));
  };

  const saveToServer = async () => {
    setSaving(true);
    try {
      const config: SavedConfig = { selectedStrategy, paramValues, tradingConfig };
      await saveStrategyConfig(config as unknown as Record<string, unknown>);
      // Also save to localStorage as fallback
      localStorage.setItem('quant-strategy-config', JSON.stringify(config));
      showStatus('配置已保存到服务器', 'success');
    } catch {
      // Fallback to localStorage only
      const config: SavedConfig = { selectedStrategy, paramValues, tradingConfig };
      localStorage.setItem('quant-strategy-config', JSON.stringify(config));
      showStatus('服务器不可用，已保存到本地', 'info');
    } finally {
      setSaving(false);
    }
  };

  const loadFromServer = async () => {
    try {
      const result = await loadStrategyConfig();
      if (result.exists && result.config) {
        const config = result.config as unknown as SavedConfig;
        if (config.selectedStrategy) setSelectedStrategy(config.selectedStrategy);
        if (config.paramValues) setParamValues(config.paramValues);
        if (config.tradingConfig) setTradingConfig(config.tradingConfig);
        showStatus('已从服务器加载配置', 'success');
        return;
      }
    } catch { /* fallback to localStorage */ }

    // Fallback: localStorage
    const raw = localStorage.getItem('quant-strategy-config');
    if (raw) {
      try {
        const config: SavedConfig = JSON.parse(raw);
        if (config.selectedStrategy) setSelectedStrategy(config.selectedStrategy);
        if (config.paramValues) setParamValues(config.paramValues);
        if (config.tradingConfig) setTradingConfig(config.tradingConfig);
      } catch { /* ignore */ }
    }
  };

  const handleRunBacktest = async () => {
    showStatus('正在运行回测…', 'info');
    try {
      await runBacktest({
        strategy: selectedStrategy,
        symbol: tradingConfig.symbol,
        start: tradingConfig.startDate,
        end: tradingConfig.endDate,
        capital: tradingConfig.initialCapital,
      });
      window.location.href = '/backtest';
    } catch {
      showStatus('回测请求失败', 'error');
    }
  };

  const toggleAlgo = (algo: string) => {
    setSelectedAlgos(prev =>
      prev.includes(algo) ? prev.filter(a => a !== algo) : [...prev, algo]
    );
  };

  const handleRetrain = async () => {
    if (selectedAlgos.length === 0) {
      showStatus('请至少选择一个算法', 'error');
      return;
    }
    setRetraining(true);
    const srcLabel = trainDataSource === 'akshare' ? '真实行情' : '模拟数据';
    showStatus(`模型训练中 (${srcLabel})，请耐心等待…`, 'info');
    try {
      const opts: RetrainOptions = {
        algorithms: selectedAlgos.join(','),
        data_source: trainDataSource,
        horizon: trainHorizon,
        threshold: trainThreshold,
      };
      if (trainDataSource === 'akshare') {
        opts.symbols = trainSymbols;
        opts.start_date = trainStartDate;
        opts.end_date = trainEndDate;
      }
      const result = await mlRetrain(opts);
      sessionStorage.setItem('task_retrain', result.task_id);
      startTaskPolling(result.task_id);
    } catch {
      showStatus('训练请求失败', 'error');
      setRetraining(false);
    }
  };

  const currentParams = paramValues[selectedStrategy] ?? {};

  const report = modelInfoData?.latest_report as Record<string, unknown> | null;

  return (
    <div className="text-[#f8fafc] space-y-6">
      <h1 className="text-2xl font-bold">⚙️ 策略配置</h1>

      {/* Strategy Selector */}
      <section>
        <h2 className="text-lg font-semibold text-[#94a3b8] mb-3">选择策略</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {STRATEGIES.map((s) => (
            <button
              key={s.name}
              onClick={() => setSelectedStrategy(s.name)}
              className={`text-left rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                selectedStrategy === s.name
                  ? 'border-[#3b82f6] bg-[#1e293b]'
                  : 'border-[#334155] bg-[#1e293b] hover:border-[#475569]'
              }`}
            >
              <h3 className="font-semibold text-sm mb-1">{s.displayName}</h3>
              <p className="text-xs text-[#94a3b8] leading-relaxed">{s.description}</p>
            </button>
          ))}
        </div>
      </section>

      {/* Parameter Configuration */}
      <section className="bg-[#1e293b] rounded-lg border border-[#334155] p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold">
            {activeStrategy.displayName} — 参数
          </h2>
          <button
            onClick={resetDefaults}
            className="flex items-center gap-1 text-sm text-[#94a3b8] hover:text-[#f8fafc] transition-colors cursor-pointer"
          >
            <RotateCcw className="h-3.5 w-3.5" /> 重置默认
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">
          {activeStrategy.parameters.map((p: StrategyParam) => (
            <div key={p.key}>
              <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">
                {p.label}
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={currentParams[p.key] ?? p.default}
                  onChange={(e) => setParam(p.key, Number(e.target.value))}
                  className="flex-1 accent-[#3b82f6] h-2 cursor-pointer"
                />
                <input
                  type="number"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={currentParams[p.key] ?? p.default}
                  onChange={(e) => setParam(p.key, Number(e.target.value))}
                  className="w-20 bg-[#0f172a] border border-[#334155] rounded px-2 py-1 text-sm text-center focus:outline-none focus:border-[#3b82f6]"
                />
              </div>
              <div className="flex justify-between text-xs text-[#64748b] mt-1">
                <span>{p.min}</span>
                <span>{p.max}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Trading Configuration */}
      <section className="bg-[#1e293b] rounded-lg border border-[#334155] p-6">
        <h2 className="text-lg font-semibold mb-5">交易配置</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-5">
          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">初始资金</label>
            <input
              type="number"
              value={tradingConfig.initialCapital}
              onChange={(e) => setTradingConfig((c) => ({ ...c, initialCapital: Number(e.target.value) }))}
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">佣金率 (%)</label>
            <input
              type="number"
              step={0.001}
              value={tradingConfig.commissionRate}
              onChange={(e) => setTradingConfig((c) => ({ ...c, commissionRate: Number(e.target.value) }))}
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">股票代码</label>
            <input
              type="text"
              value={tradingConfig.symbol}
              onChange={(e) => setTradingConfig((c) => ({ ...c, symbol: e.target.value }))}
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">开始日期</label>
            <input
              type="date"
              value={tradingConfig.startDate}
              onChange={(e) => setTradingConfig((c) => ({ ...c, startDate: e.target.value }))}
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">结束日期</label>
            <input
              type="date"
              value={tradingConfig.endDate}
              onChange={(e) => setTradingConfig((c) => ({ ...c, endDate: e.target.value }))}
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>
        </div>
      </section>

      {/* ML Model Training Section */}
      <section className="bg-[#1e293b] rounded-lg border border-[#334155] p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Brain className="h-5 w-5 text-purple-400" /> ML模型训练
        </h2>

        {/* Algorithm Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-[#94a3b8] mb-2">选择训练算法 (多选竞争，最优AUC获胜)</label>
          <div className="flex flex-wrap gap-2">
            {Object.entries(ALGO_LABELS).map(([key, { label, desc, badge }]) => (
              <button
                key={key}
                onClick={() => toggleAlgo(key)}
                className={`px-3 py-2 rounded-lg text-sm border transition-colors ${
                  selectedAlgos.includes(key)
                    ? `${badge} border-current`
                    : 'bg-[#0f172a] border-[#334155] text-[#64748b] hover:border-[#475569]'
                }`}
              >
                <span className="font-medium">{label}</span>
                <span className="block text-xs opacity-75 mt-0.5">{desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Data Source Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-[#94a3b8] mb-2 flex items-center gap-1">
            <Database className="h-4 w-4" /> 训练数据源
          </label>
          <div className="flex gap-2 mb-3">
            <button
              onClick={() => setTrainDataSource('akshare')}
              className={`px-4 py-2 rounded-lg text-sm border transition-colors ${
                trainDataSource === 'akshare'
                  ? 'bg-cyan-600/20 text-cyan-400 border-cyan-500/50'
                  : 'bg-[#0f172a] border-[#334155] text-[#64748b] hover:border-[#475569]'
              }`}
            >
              📡 真实行情 (akshare)
            </button>
            <button
              onClick={() => setTrainDataSource('synthetic')}
              className={`px-4 py-2 rounded-lg text-sm border transition-colors ${
                trainDataSource === 'synthetic'
                  ? 'bg-yellow-600/20 text-yellow-400 border-yellow-500/50'
                  : 'bg-[#0f172a] border-[#334155] text-[#64748b] hover:border-[#475569]'
              }`}
            >
              🧪 模拟数据 (合成GBM)
            </button>
          </div>

          {trainDataSource === 'akshare' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 bg-[#0f172a] rounded-lg border border-[#334155] p-4">
              <div className="md:col-span-2">
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-[#64748b]">股票列表 (逗号分隔代码)</label>
                  <span className="text-xs text-cyan-400">{trainSymbols.split(',').filter(s => s.trim()).length} 只股票</span>
                </div>
                <textarea
                  value={trainSymbols}
                  onChange={(e) => setTrainSymbols(e.target.value)}
                  rows={2}
                  className="w-full bg-[#1e293b] border border-[#334155] rounded px-3 py-1.5 text-sm text-[#f8fafc] focus:border-cyan-500 outline-none resize-none"
                  placeholder="600519,000858,000001,..."
                />
                <div className="flex flex-wrap gap-1.5 mt-1.5">
                  <button onClick={() => setTrainSymbols('600519,000858,600036,601318,300750,002594,000333,601888,600900,601398')} className="px-2 py-0.5 rounded text-xs bg-[#1e293b] border border-[#334155] text-[#94a3b8] hover:border-cyan-500/50 hover:text-cyan-400 transition-colors">蓝筹10只</button>
                  <button onClick={() => setTrainSymbols('300750,300760,300059,300122,300782,300015,300274,300498')} className="px-2 py-0.5 rounded text-xs bg-[#1e293b] border border-[#334155] text-[#94a3b8] hover:border-green-500/50 hover:text-green-400 transition-colors">创业板8只</button>
                  <button onClick={() => setTrainSymbols('688981,688111,688036,688561,688005,688012,688185,688599')} className="px-2 py-0.5 rounded text-xs bg-[#1e293b] border border-[#334155] text-[#94a3b8] hover:border-purple-500/50 hover:text-purple-400 transition-colors">科创板8只</button>
                  <button onClick={() => setTrainSymbols('600519,000858,000001,600036,300750,002594,601318,600276,000333,601888,600030,601166,600900,000568,600809,601899,600031,600309,300059,600887,000651,002415,300760,601398,601288,600438,002460,603259,600690,601669')} className="px-2 py-0.5 rounded text-xs bg-[#1e293b] border border-[#334155] text-[#94a3b8] hover:border-cyan-500/50 hover:text-cyan-400 transition-colors">多行业30只</button>
                  <button onClick={() => setTrainSymbols('600519,000858,000568,600809,600887,002304,603288,600036,601318,601166,600030,601398,601288,300750,002594,600438,601012,002460,600276,000333,300760,603259,300122,002415,603501,300782,688981,688111,688036,688561,002049,000651,600690,002032,601888,601899,600031,600309,601225,600585,600900,601669,600048,601800,300059,002230,603444,600760,002179,600893')} className="px-2 py-0.5 rounded text-xs bg-[#1e293b] border border-[#334155] text-[#94a3b8] hover:border-cyan-500/50 hover:text-cyan-400 transition-colors">全行业50只</button>
                </div>
              </div>
              <div>
                <label className="block text-xs text-[#64748b] mb-1">训练年数</label>
                <div className="flex gap-1.5">
                  {[
                    { years: 2, start: '2023-01-01' },
                    { years: 3, start: '2022-01-01' },
                    { years: 5, start: '2020-01-01' },
                    { years: 7, start: '2018-01-01' },
                  ].map(({ years, start }) => (
                    <button
                      key={years}
                      onClick={() => { setTrainStartDate(start); setTrainEndDate('2024-12-31'); }}
                      className={`flex-1 px-2 py-1.5 rounded text-xs border transition-colors ${
                        trainStartDate === start
                          ? 'bg-cyan-600/20 text-cyan-400 border-cyan-500/50'
                          : 'bg-[#1e293b] border-[#334155] text-[#94a3b8] hover:border-[#475569]'
                      }`}
                    >
                      {years}年
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-xs text-[#64748b] mb-1">自定义日期范围</label>
                <div className="flex gap-2">
                  <input type="date" value={trainStartDate} onChange={(e) => setTrainStartDate(e.target.value)}
                    className="flex-1 bg-[#1e293b] border border-[#334155] rounded px-2 py-1.5 text-xs text-[#f8fafc] focus:border-cyan-500 outline-none" />
                  <input type="date" value={trainEndDate} onChange={(e) => setTrainEndDate(e.target.value)}
                    className="flex-1 bg-[#1e293b] border border-[#334155] rounded px-2 py-1.5 text-xs text-[#f8fafc] focus:border-cyan-500 outline-none" />
                </div>
              </div>
              <div>
                <label className="block text-xs text-[#64748b] mb-1">预测周期 (前瞻天数)</label>
                <select
                  value={trainHorizon}
                  onChange={(e) => setTrainHorizon(Number(e.target.value))}
                  className="w-full bg-[#1e293b] border border-[#334155] rounded px-3 py-1.5 text-sm text-[#f8fafc] focus:border-cyan-500 outline-none"
                >
                  <option value={3}>3天</option>
                  <option value={5}>5天</option>
                  <option value={10}>10天</option>
                  <option value={20}>20天</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-[#64748b] mb-1">正样本阈值 (收益率)</label>
                <select
                  value={trainThreshold}
                  onChange={(e) => setTrainThreshold(Number(e.target.value))}
                  className="w-full bg-[#1e293b] border border-[#334155] rounded px-3 py-1.5 text-sm text-[#f8fafc] focus:border-cyan-500 outline-none"
                >
                  <option value={0.005}>0.5%</option>
                  <option value={0.01}>1.0%</option>
                  <option value={0.02}>2.0%</option>
                  <option value={0.03}>3.0%</option>
                  <option value={0.05}>5.0%</option>
                </select>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 mb-4">
          <button
            onClick={handleRetrain}
            disabled={retraining || selectedAlgos.length === 0}
            className="flex items-center gap-2 px-5 py-2.5 bg-purple-600 hover:bg-purple-700 disabled:bg-[#334155] disabled:text-[#64748b] text-white font-medium rounded-lg text-sm transition-colors"
          >
            {retraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
            {retraining ? '训练中…' : '开始训练'}
          </button>
          <span className="text-xs text-[#64748b]">
            已选: {selectedAlgos.map(a => ALGO_LABELS[a]?.label).join(', ') || '无'}
          </span>
        </div>

        {/* Latest Model Report */}
        {report && (
          <div className="bg-[#0f172a] rounded-lg border border-[#334155] p-4">
            <h3 className="text-sm font-semibold text-[#94a3b8] mb-3">📊 最新模型报告</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              {report.best_algorithm != null && (
                <div>
                  <span className="text-[#64748b]">最优算法</span>
                  <p className="font-bold text-purple-400">{String(report.best_algorithm).toUpperCase()}</p>
                </div>
              )}
              {(report.final_model as Record<string, unknown> | undefined)?.auc != null && (
                <div>
                  <span className="text-[#64748b]">AUC</span>
                  <p className="font-bold text-[#f8fafc]">{String((report.final_model as Record<string, unknown>).auc)}</p>
                </div>
              )}
              {(report.final_model as Record<string, unknown> | undefined)?.accuracy != null && (
                <div>
                  <span className="text-[#64748b]">准确率</span>
                  <p className="font-bold text-[#f8fafc]">{String((report.final_model as Record<string, unknown>).accuracy)}</p>
                </div>
              )}
              {report.n_samples != null && (
                <div>
                  <span className="text-[#64748b]">样本数</span>
                  <p className="font-bold text-[#f8fafc]">{String(report.n_samples)}</p>
                </div>
              )}
            </div>
            {Array.isArray(report.feature_importance) && (report.feature_importance as Array<{feature: string; importance: number}>).length > 0 && (
              <div className="mt-3">
                <h4 className="text-xs text-[#64748b] mb-1">Top 5 特征重要性</h4>
                <div className="space-y-1">
                  {(report.feature_importance as Array<{feature: string; importance: number}>).slice(0, 5).map((f, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="text-[#94a3b8] w-32 truncate">{f.feature}</span>
                      <div className="flex-1 bg-[#334155] rounded-full h-1.5">
                        <div
                          className="bg-purple-500 rounded-full h-1.5"
                          style={{
                            width: `${Math.min(100, (f.importance / (report.feature_importance as Array<{feature: string; importance: number}>)[0].importance) * 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-[#64748b] w-14 text-right">{f.importance.toFixed(0)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {report.timestamp != null && (
              <p className="text-xs text-[#64748b] mt-2">{'训练时间: ' + String(report.timestamp)}</p>
            )}
          </div>
        )}
      </section>

      {/* Status message */}
      {status && (
        <div className={`flex items-center gap-2 text-sm p-3 rounded-lg border ${
          status.type === 'success' ? 'bg-green-500/10 border-green-500/30 text-green-400' :
          status.type === 'error' ? 'bg-red-500/10 border-red-500/30 text-red-400' :
          'bg-blue-500/10 border-blue-500/30 text-blue-400'
        }`}>
          {status.type === 'success' ? <CheckCircle className="h-4 w-4" /> :
           status.type === 'error' ? <AlertCircle className="h-4 w-4" /> :
           <Loader2 className="h-4 w-4 animate-spin" />}
          {status.text}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={saveToServer}
          disabled={saving}
          className="flex items-center gap-2 px-5 py-2.5 bg-[#3b82f6] hover:bg-[#2563eb] disabled:bg-[#334155] text-white font-medium rounded-lg transition-colors cursor-pointer text-sm"
        >
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
          保存配置
        </button>
        <button
          onClick={handleRunBacktest}
          className="flex items-center gap-2 px-5 py-2.5 bg-[#22c55e] hover:bg-[#16a34a] text-white font-medium rounded-lg transition-colors cursor-pointer text-sm"
        >
          <Play className="h-4 w-4" /> 运行回测
        </button>
        <button
          onClick={loadFromServer}
          className="flex items-center gap-2 px-5 py-2.5 bg-[#334155] hover:bg-[#475569] text-white font-medium rounded-lg transition-colors cursor-pointer text-sm"
        >
          <Upload className="h-4 w-4" /> 加载配置
        </button>
      </div>
    </div>
  );
}
