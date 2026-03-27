import { useState, useEffect, useCallback } from 'react';
import {
  factorRegistryGet,
  factorResults,
  type FactorRegistry,
  type FactorResults,
} from '../../api/client';
import OverviewTab from './OverviewTab';
import ParametricTab from './ParametricTab';
import GPTab from './GPTab';
import ManualTab from './ManualTab';
import RegistryTab from './RegistryTab';
import ExportTab from './ExportTab';
import CorrelationTab from './CorrelationTab';
import IcDecayTab from './IcDecayTab';

/* ── Tab types ───────────────────────────────────────────────────── */
type Tab = 'overview' | 'parametric' | 'gp' | 'manual' | 'registry' | 'export' | 'correlation' | 'ic-decay';

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'overview', label: '总览', icon: '📊' },
  { id: 'parametric', label: '参数化搜索', icon: '🔍' },
  { id: 'gp', label: 'GP进化', icon: '🧬' },
  { id: 'manual', label: '手动探索', icon: '✏️' },
  { id: 'registry', label: '因子注册表', icon: '📋' },
  { id: 'export', label: '导出集成', icon: '📦' },
  { id: 'correlation', label: '因子相关性', icon: '🔗' },
  { id: 'ic-decay', label: 'IC衰减', icon: '📈' },
];

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

      {/* Tab content — all tabs stay mounted to preserve state */}
      <div className={activeTab === 'overview' ? '' : 'hidden'}><OverviewTab registry={registry} results={results} /></div>
      <div className={activeTab === 'parametric' ? '' : 'hidden'}><ParametricTab /></div>
      <div className={activeTab === 'gp' ? '' : 'hidden'}><GPTab /></div>
      <div className={activeTab === 'manual' ? '' : 'hidden'}><ManualTab /></div>
      <div className={activeTab === 'registry' ? '' : 'hidden'}><RegistryTab registry={registry} onRefresh={loadData} /></div>
      <div className={activeTab === 'export' ? '' : 'hidden'}><ExportTab results={results} /></div>
      <div className={activeTab === 'correlation' ? '' : 'hidden'}><CorrelationTab /></div>
      <div className={activeTab === 'ic-decay' ? '' : 'hidden'}><IcDecayTab /></div>
    </div>
  );
}
