import { useState, useEffect, useCallback } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from 'recharts';
import {
  getResearchDlModels,
  collectResearch,
  getMlFeatureImportance,
  type ResearchKnowledgeBase,
  type DlModelEntry,
  type CollectedResearch,
  type FeatureImportanceItem,
  type FeatureImportanceModelInfo,
} from '../api/client';

/* ── category color mapping ─────────────────────────────────────── */
const CAT_COLORS: Record<string, string> = {
  'Transformer系列': '#3b82f6',
  'VAE/生成式模型': '#8b5cf6',
  '图神经网络 (GNN)': '#10b981',
  '表格数据模型': '#f59e0b',
  '强化学习+注意力': '#ef4444',
  '多模态融合': '#ec4899',
};

function getCatColor(name: string) {
  return CAT_COLORS[name] || '#6b7280';
}

/* ── Model Card ─────────────────────────────────────────────────── */
function ModelCard({ model }: { model: DlModelEntry }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5 hover:border-[#3b82f6]/40 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-bold text-[#f8fafc]">{model.name}</h3>
          <div className="flex gap-2 mt-1">
            <span className="px-2 py-0.5 text-xs rounded-full bg-[#3b82f6]/20 text-[#3b82f6]">
              {model.category}
            </span>
            <span className="px-2 py-0.5 text-xs rounded-full bg-[#334155] text-[#94a3b8]">
              {model.year}
            </span>
          </div>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-[#3b82f6] hover:underline"
        >
          {expanded ? '收起' : '展开详情'}
        </button>
      </div>

      <p className="text-sm text-[#cbd5e1] mb-3 leading-relaxed">{model.description}</p>

      <div className="mb-3">
        <span className="text-xs text-[#94a3b8] font-semibold">🔬 核心创新：</span>
        <span className="text-xs text-[#e2e8f0] ml-1">{model.key_innovation}</span>
      </div>

      {expanded && (
        <div className="space-y-3 mt-4 pt-4 border-t border-[#334155]">
          <div>
            <span className="text-xs text-[#94a3b8] font-semibold block mb-1">🏗️ 架构：</span>
            <span className="text-xs text-[#e2e8f0]">{model.architecture}</span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <span className="text-xs text-[#94a3b8] font-semibold block mb-1">📥 输入数据：</span>
              <div className="flex flex-wrap gap-1">
                {model.input_data.map((d, i) => (
                  <span key={i} className="px-2 py-0.5 text-xs rounded bg-[#334155] text-[#cbd5e1]">{d}</span>
                ))}
              </div>
            </div>
            <div>
              <span className="text-xs text-[#94a3b8] font-semibold block mb-1">📤 输出：</span>
              <span className="text-xs text-[#e2e8f0]">{model.output}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <span className="text-xs text-green-400 font-semibold block mb-1">✅ 优势：</span>
              <ul className="text-xs text-[#cbd5e1] space-y-1">
                {model.strengths.map((s, i) => (
                  <li key={i} className="flex items-start gap-1">
                    <span className="text-green-400 mt-0.5">•</span> {s}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <span className="text-xs text-yellow-400 font-semibold block mb-1">⚠️ 局限：</span>
              <ul className="text-xs text-[#cbd5e1] space-y-1">
                {model.limitations.map((l, i) => (
                  <li key={i} className="flex items-start gap-1">
                    <span className="text-yellow-400 mt-0.5">•</span> {l}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="pt-2">
            <a
              href={model.reference_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-[#3b82f6] hover:underline"
            >
              📄 {model.reference} →
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Collected Research Card ─────────────────────────────────────── */
function CollectedCard({ item }: { item: CollectedResearch }) {
  const relevanceColor = item.relevance === '高' ? 'text-green-400' : item.relevance === '中' ? 'text-yellow-400' : 'text-[#94a3b8]';
  return (
    <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-4">
      <div className="flex items-start justify-between mb-2">
        <h4 className="text-sm font-semibold text-[#f8fafc]">{item.title}</h4>
        <span className={`text-xs font-semibold ${relevanceColor}`}>相关度: {item.relevance}</span>
      </div>
      <p className="text-xs text-[#cbd5e1] leading-relaxed mb-2">{item.summary}</p>
      <div className="flex justify-between text-xs text-[#64748b]">
        <span>来源: {item.source}</span>
        <span>{item.collected_at}</span>
      </div>
    </div>
  );
}

/* ── Feature Importance Chart ─────────────────────────────────────── */
function FeatureImportanceSection() {
  const [features, setFeatures] = useState<FeatureImportanceItem[]>([]);
  const [modelInfo, setModelInfo] = useState<FeatureImportanceModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await getMlFeatureImportance();
      setFeatures(data.features.slice(0, 20));
      setModelInfo(data.model_info);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '加载失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // Color gradient: most important = dark blue, least = light blue
  const getBarColor = (index: number, total: number) => {
    const ratio = total > 1 ? index / (total - 1) : 0;
    const r = Math.round(30 + ratio * 117);   // 30 → 147
    const g = Math.round(64 + ratio * 133);   // 64 → 197
    const b = Math.round(175 + ratio * 68);   // 175 → 243
    return `rgb(${r}, ${g}, ${b})`;
  };

  // Reverse data so highest importance is at top of vertical bar chart
  const chartData = [...features].reverse();

  return (
    <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-bold text-[#f8fafc]">📊 ML 特征重要性</h2>
          <p className="text-xs text-[#94a3b8] mt-1">
            LightGBM 模型 Top 20 特征（按 gain 排序）
          </p>
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          className="rounded-lg bg-[#334155] px-3 py-1.5 text-xs font-medium text-[#f8fafc] hover:bg-[#475569] disabled:opacity-50"
        >
          {loading ? '加载中...' : '🔄 刷新'}
        </button>
      </div>

      {error && <div className="text-xs text-red-400 mb-3">{error}</div>}

      {/* Model Info Card */}
      {modelInfo && (modelInfo.auc !== null || modelInfo.accuracy !== null) && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          {modelInfo.auc !== null && (
            <div className="rounded-lg bg-[#0f172a] border border-[#334155] p-3 text-center">
              <div className="text-lg font-bold text-[#3b82f6]">{modelInfo.auc.toFixed(3)}</div>
              <div className="text-xs text-[#94a3b8]">AUC</div>
            </div>
          )}
          {modelInfo.accuracy !== null && (
            <div className="rounded-lg bg-[#0f172a] border border-[#334155] p-3 text-center">
              <div className="text-lg font-bold text-[#10b981]">{(modelInfo.accuracy * 100).toFixed(1)}%</div>
              <div className="text-xs text-[#94a3b8]">准确率</div>
            </div>
          )}
          <div className="rounded-lg bg-[#0f172a] border border-[#334155] p-3 text-center">
            <div className="text-lg font-bold text-[#f59e0b]">{modelInfo.n_features}</div>
            <div className="text-xs text-[#94a3b8]">特征数</div>
          </div>
          {modelInfo.timestamp && (
            <div className="rounded-lg bg-[#0f172a] border border-[#334155] p-3 text-center">
              <div className="text-lg font-bold text-[#94a3b8]">{modelInfo.timestamp}</div>
              <div className="text-xs text-[#94a3b8]">训练日期</div>
            </div>
          )}
        </div>
      )}

      {/* Chart */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#3b82f6]" />
        </div>
      ) : features.length > 0 ? (
        <ResponsiveContainer width="100%" height={Math.max(400, chartData.length * 28)}>
          <BarChart layout="vertical" data={chartData} margin={{ top: 5, right: 60, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#334155' }} />
            <YAxis
              type="category"
              dataKey="name"
              width={150}
              tick={{ fill: '#cbd5e1', fontSize: 11 }}
              axisLine={{ stroke: '#334155' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '8px',
                color: '#f8fafc',
                fontSize: '12px',
              }}
              formatter={((value: number | null) => [
                value !== null ? value.toFixed(1) : 'N/A',
                '重要性',
              ]) as any}
            />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]} label={{ position: 'right', fill: '#94a3b8', fontSize: 10, formatter: ((v: number | null) => v !== null ? v.toFixed(1) : '') as any }}>
              {chartData.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getBarColor(chartData.length - 1 - index, chartData.length)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className="text-sm text-[#94a3b8] text-center py-8">暂无特征重要性数据</div>
      )}
    </div>
  );
}

/* ── Main Page ──────────────────────────────────────────────────── */
export default function DLModels() {
  const [kb, setKb] = useState<ResearchKnowledgeBase | null>(null);
  const [collected, setCollected] = useState<CollectedResearch[]>([]);
  const [loading, setLoading] = useState(true);
  const [collecting, setCollecting] = useState(false);
  const [collectError, setCollectError] = useState('');
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [collectTopic, setCollectTopic] = useState('');

  useEffect(() => {
    getResearchDlModels()
      .then((data) => {
        setKb(data);
        setCollected(data.collected || []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleCollect = async () => {
    setCollecting(true);
    setCollectError('');
    try {
      const result = await collectResearch(collectTopic || undefined);
      if (result.status === 'ok') {
        setCollected((prev) => [...result.collected, ...prev]);
      } else {
        setCollectError(result.message || '收集失败');
      }
    } catch (e: unknown) {
      setCollectError(e instanceof Error ? e.message : '请求失败');
    } finally {
      setCollecting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#3b82f6]" />
      </div>
    );
  }

  if (!kb) return <div className="text-[#94a3b8] p-8">加载失败</div>;

  const totalModels = kb.categories.reduce((sum, c) => sum + c.models.length, 0);
  const filteredCategories = activeCategory
    ? kb.categories.filter((c) => c.name === activeCategory)
    : kb.categories;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#f8fafc]">🧠 DL因子模型研究</h1>
          <p className="text-sm text-[#94a3b8] mt-1">
            最新深度学习多因子量化模型知识库 · {totalModels} 个模型 · {kb.categories.length} 个类别
          </p>
        </div>
        <span className="text-xs text-[#64748b]">更新: {kb.last_updated}</span>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
        {kb.categories.map((cat) => (
          <button
            key={cat.name}
            onClick={() => setActiveCategory(activeCategory === cat.name ? null : cat.name)}
            className={`rounded-xl border p-3 text-center transition-colors ${
              activeCategory === cat.name
                ? 'border-[#3b82f6] bg-[#3b82f6]/10'
                : 'border-[#334155] bg-[#1e293b] hover:border-[#475569]'
            }`}
          >
            <div className="text-xl font-bold" style={{ color: getCatColor(cat.name) }}>
              {cat.models.length}
            </div>
            <div className="text-xs text-[#94a3b8] mt-1 truncate">{cat.name}</div>
          </button>
        ))}
      </div>

      {/* ML Feature Importance */}
      <FeatureImportanceSection />

      {/* Auto-Collect */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5">
        <h2 className="text-base font-bold text-[#f8fafc] mb-3">🔍 自动收集最新研究</h2>
        <p className="text-xs text-[#94a3b8] mb-3">
          使用LLM自动搜集并概括最新的深度学习因子模型研究进展。需要配置LLM API。
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={collectTopic}
            onChange={(e) => setCollectTopic(e.target.value)}
            placeholder="输入研究主题(可选，默认: 量化多因子深度学习模型最新进展)"
            className="flex-1 rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] placeholder-[#64748b] focus:border-[#3b82f6] focus:outline-none"
          />
          <button
            onClick={handleCollect}
            disabled={collecting}
            className="rounded-lg bg-[#3b82f6] px-4 py-2 text-sm font-medium text-white hover:bg-[#2563eb] disabled:opacity-50"
          >
            {collecting ? '收集中...' : '🤖 开始收集'}
          </button>
        </div>
        {collectError && (
          <div className="mt-2 text-xs text-red-400">{collectError}</div>
        )}
      </div>

      {/* Collected items */}
      {collected.length > 0 && (
        <div>
          <h2 className="text-base font-bold text-[#f8fafc] mb-3">📰 已收集的研究 ({collected.length})</h2>
          <div className="space-y-3">
            {collected.map((item, i) => (
              <CollectedCard key={i} item={item} />
            ))}
          </div>
        </div>
      )}

      {/* Model categories */}
      {filteredCategories.map((cat) => (
        <div key={cat.name}>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1 h-6 rounded-full" style={{ backgroundColor: getCatColor(cat.name) }} />
            <h2 className="text-lg font-bold text-[#f8fafc]">{cat.name}</h2>
            <span className="text-xs text-[#94a3b8]">— {cat.description}</span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {cat.models.map((model) => (
              <ModelCard key={model.id} model={model} />
            ))}
          </div>
        </div>
      ))}

      {/* Comparison table */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] p-5 overflow-x-auto">
        <h2 className="text-base font-bold text-[#f8fafc] mb-4">📊 模型对比总览</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[#334155]">
              <th className="text-left py-2 px-3 text-[#94a3b8] font-medium">模型</th>
              <th className="text-left py-2 px-3 text-[#94a3b8] font-medium">类别</th>
              <th className="text-left py-2 px-3 text-[#94a3b8] font-medium">年份</th>
              <th className="text-left py-2 px-3 text-[#94a3b8] font-medium">核心架构</th>
              <th className="text-left py-2 px-3 text-[#94a3b8] font-medium">输出</th>
            </tr>
          </thead>
          <tbody>
            {kb.categories.flatMap((cat) =>
              cat.models.map((m) => (
                <tr key={m.id} className="border-b border-[#334155]/50 hover:bg-[#334155]/20">
                  <td className="py-2 px-3 text-[#f8fafc] font-medium">{m.name}</td>
                  <td className="py-2 px-3">
                    <span
                      className="px-2 py-0.5 text-xs rounded-full"
                      style={{ backgroundColor: getCatColor(cat.name) + '20', color: getCatColor(cat.name) }}
                    >
                      {m.category}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-[#94a3b8]">{m.year}</td>
                  <td className="py-2 px-3 text-[#cbd5e1] text-xs max-w-[200px] truncate">{m.architecture}</td>
                  <td className="py-2 px-3 text-[#cbd5e1] text-xs">{m.output}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
