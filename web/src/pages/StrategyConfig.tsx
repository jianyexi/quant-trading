import { useState, useCallback } from 'react';
import type { StrategyConfig, StrategyParam } from '../types';
import { runBacktest } from '../api/client';

const STRATEGIES: StrategyConfig[] = [
  {
    name: 'DualMaCrossover',
    displayName: 'Dual MA Crossover',
    description:
      'Buy when fast MA crosses above slow MA, sell on death cross. Classic trend-following strategy.',
    parameters: [
      { key: 'fast_period', label: 'Fast MA Period', type: 'number', default: 5, min: 2, max: 60, step: 1 },
      { key: 'slow_period', label: 'Slow MA Period', type: 'number', default: 20, min: 5, max: 240, step: 1 },
    ],
  },
  {
    name: 'RsiMeanReversion',
    displayName: 'RSI Mean Reversion',
    description:
      'Buy when RSI drops below oversold threshold, sell when RSI rises above overbought threshold.',
    parameters: [
      { key: 'period', label: 'RSI Period', type: 'number', default: 14, min: 2, max: 50, step: 1 },
      { key: 'oversold', label: 'Oversold Threshold', type: 'number', default: 30, min: 10, max: 40, step: 1 },
      { key: 'overbought', label: 'Overbought Threshold', type: 'number', default: 70, min: 60, max: 90, step: 1 },
    ],
  },
  {
    name: 'MacdMomentum',
    displayName: 'MACD Momentum',
    description:
      'Buy when MACD histogram crosses above zero, sell when it crosses below. Momentum-based strategy.',
    parameters: [
      { key: 'fast_period', label: 'Fast EMA Period', type: 'number', default: 12, min: 2, max: 50, step: 1 },
      { key: 'slow_period', label: 'Slow EMA Period', type: 'number', default: 26, min: 5, max: 100, step: 1 },
      { key: 'signal_period', label: 'Signal Period', type: 'number', default: 9, min: 2, max: 30, step: 1 },
    ],
  },
];

const STORAGE_KEY = 'quant-strategy-config';

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

export default function StrategyConfigPage() {
  const [selectedStrategy, setSelectedStrategy] = useState<string>(STRATEGIES[0].name);
  const [paramValues, setParamValues] = useState<Record<string, Record<string, number>>>(defaultParamValues);
  const [tradingConfig, setTradingConfig] = useState<TradingConfig>(defaultTradingConfig);
  const [status, setStatus] = useState<string | null>(null);

  const activeStrategy = STRATEGIES.find((s) => s.name === selectedStrategy)!;

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

  const saveConfig = () => {
    const config: SavedConfig = { selectedStrategy, paramValues, tradingConfig };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
    setStatus('Configuration saved.');
    setTimeout(() => setStatus(null), 2000);
  };

  const loadConfig = () => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      setStatus('No saved configuration found.');
      setTimeout(() => setStatus(null), 2000);
      return;
    }
    try {
      const config: SavedConfig = JSON.parse(raw);
      setSelectedStrategy(config.selectedStrategy);
      setParamValues(config.paramValues);
      setTradingConfig(config.tradingConfig);
      setStatus('Configuration loaded.');
      setTimeout(() => setStatus(null), 2000);
    } catch {
      setStatus('Failed to load configuration.');
      setTimeout(() => setStatus(null), 2000);
    }
  };

  const handleRunBacktest = async () => {
    setStatus('Running backtest…');
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
      setStatus('Backtest request failed.');
      setTimeout(() => setStatus(null), 3000);
    }
  };

  const currentParams = paramValues[selectedStrategy] ?? {};

  return (
    <div className="text-[#f8fafc]">
      <h1 className="text-2xl font-bold mb-6">Strategy Configuration</h1>

      {/* Strategy Selector */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold text-[#94a3b8] mb-3">Select Strategy</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {STRATEGIES.map((s) => (
            <button
              key={s.name}
              onClick={() => setSelectedStrategy(s.name)}
              className={`text-left rounded-lg p-5 border-2 transition-colors cursor-pointer ${
                selectedStrategy === s.name
                  ? 'border-[#3b82f6] bg-[#1e293b]'
                  : 'border-[#334155] bg-[#1e293b] hover:border-[#475569]'
              }`}
            >
              <h3 className="font-semibold text-base mb-1">{s.displayName}</h3>
              <p className="text-sm text-[#94a3b8] leading-relaxed">{s.description}</p>
            </button>
          ))}
        </div>
      </section>

      {/* Parameter Configuration */}
      <section className="mb-8 bg-[#1e293b] rounded-lg border border-[#334155] p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold">
            {activeStrategy.displayName} — Parameters
          </h2>
          <button
            onClick={resetDefaults}
            className="text-sm text-[#94a3b8] hover:text-[#f8fafc] transition-colors cursor-pointer"
          >
            Reset to Defaults
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
      <section className="mb-8 bg-[#1e293b] rounded-lg border border-[#334155] p-6">
        <h2 className="text-lg font-semibold mb-5">Trading Configuration</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-5">
          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">
              Initial Capital
            </label>
            <input
              type="number"
              value={tradingConfig.initialCapital}
              onChange={(e) =>
                setTradingConfig((c) => ({ ...c, initialCapital: Number(e.target.value) }))
              }
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">
              Commission Rate (%)
            </label>
            <input
              type="number"
              step={0.001}
              value={tradingConfig.commissionRate}
              onChange={(e) =>
                setTradingConfig((c) => ({ ...c, commissionRate: Number(e.target.value) }))
              }
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">Symbol</label>
            <input
              type="text"
              value={tradingConfig.symbol}
              onChange={(e) =>
                setTradingConfig((c) => ({ ...c, symbol: e.target.value }))
              }
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">Start Date</label>
            <input
              type="date"
              value={tradingConfig.startDate}
              onChange={(e) =>
                setTradingConfig((c) => ({ ...c, startDate: e.target.value }))
              }
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#94a3b8] mb-1.5">End Date</label>
            <input
              type="date"
              value={tradingConfig.endDate}
              onChange={(e) =>
                setTradingConfig((c) => ({ ...c, endDate: e.target.value }))
              }
              className="w-full bg-[#0f172a] border border-[#334155] rounded px-3 py-2 text-sm focus:outline-none focus:border-[#3b82f6]"
            />
          </div>
        </div>
      </section>

      {/* Status message */}
      {status && (
        <div className="mb-4 text-sm text-[#94a3b8]">{status}</div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={saveConfig}
          className="px-5 py-2.5 bg-[#3b82f6] hover:bg-[#2563eb] text-white font-medium rounded transition-colors cursor-pointer"
        >
          Save Configuration
        </button>
        <button
          onClick={handleRunBacktest}
          className="px-5 py-2.5 bg-[#22c55e] hover:bg-[#16a34a] text-white font-medium rounded transition-colors cursor-pointer"
        >
          Run Backtest
        </button>
        <button
          onClick={loadConfig}
          className="px-5 py-2.5 bg-[#334155] hover:bg-[#475569] text-white font-medium rounded transition-colors cursor-pointer"
        >
          Load Saved
        </button>
      </div>
    </div>
  );
}
