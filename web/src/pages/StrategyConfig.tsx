import { useState, useCallback, useEffect, useRef } from 'react';
import type { StrategyConfig, StrategyParam } from '../types';
import { runBacktest, saveStrategyConfig, loadStrategyConfig, mlModelInfo, type ModelInfo } from '../api/client';
import { Save, Upload, Play, RotateCcw, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { useMarket } from '../contexts/MarketContext';

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

const DEFAULT_SYMBOLS: Record<string, string> = {
  CN: '600519.SH',
  US: 'AAPL',
  HK: '0700.HK',
  ALL: '600519.SH',
};

function defaultTradingConfig(market?: string): TradingConfig {
  return {
    initialCapital: 1000000,
    commissionRate: 0.025,
    symbol: DEFAULT_SYMBOLS[market || 'ALL'] || '600519.SH',
    startDate: '2023-01-01',
    endDate: '2024-01-01',
  };
}

export default function StrategyConfigPage() {
  const { market } = useMarket();
  const [selectedStrategy, setSelectedStrategy] = useState<string>(STRATEGIES[0].name);
  const [paramValues, setParamValues] = useState<Record<string, Record<string, number>>>(defaultParamValues);
  const [tradingConfig, setTradingConfig] = useState<TradingConfig>(() => defaultTradingConfig(market));
  const [status, setStatus] = useState<{ text: string; type: 'info' | 'success' | 'error' } | null>(null);
  const [saving, setSaving] = useState(false);
  const [modelInfoData, setModelInfoData] = useState<ModelInfo | null>(null);

  const activeStrategy = STRATEGIES.find((s) => s.name === selectedStrategy)!;

  // Load config from server on mount
  const mountedRef = useRef(false);
  useEffect(() => {
    if (mountedRef.current) return;
    mountedRef.current = true;
    loadFromServer();
    fetchModelInfo();
  });

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

      {/* Model Info (read-only) */}
      {report && (
        <section className="bg-[#1e293b] rounded-lg border border-[#334155] p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">📊 当前模型状态</h2>
            <a href="/pipeline" className="text-sm text-[#3b82f6] hover:underline">前往流水线训练 →</a>
          </div>
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
          {report.timestamp != null && (
            <p className="text-xs text-[#64748b] mt-2">{'训练时间: ' + String(report.timestamp)}</p>
          )}
        </section>
      )}

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
