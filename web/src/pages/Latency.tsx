import { useState, useEffect, useCallback } from 'react';
import { getLatency } from '../api/client';

interface ModuleInfo {
  name: string;
  last_us: number;
  avg_us: number;
  total_calls: number;
  total_us: number;
  pct_of_pipeline: number;
}

interface BottleneckInfo {
  module: string;
  avg_us: number;
  pct: number;
  suggestion: string;
}

interface LatencyData {
  engine_running: boolean;
  timestamp: string;
  modules: ModuleInfo[];
  bottleneck: BottleneckInfo | null;
  pipeline_total_us: number;
  throughput: {
    total_bars: number;
    total_signals: number;
    total_orders: number;
    total_fills: number;
    total_rejected: number;
    signal_ratio: number;
    fill_ratio: number;
  };
  health_score: number;
  thresholds: Record<string, number>;
}

function formatUs(us: number): string {
  if (us === 0) return '—';
  if (us < 1_000) return `${us}µs`;
  if (us < 1_000_000) return `${(us / 1_000).toFixed(1)}ms`;
  return `${(us / 1_000_000).toFixed(2)}s`;
}

function statusColor(avg_us: number, name: string): string {
  const thresholds: Record<string, [number, number]> = {
    'Data Fetch': [500_000, 2_000_000],
    'Strategy': [10_000, 100_000],
    'Risk': [1_000, 10_000],
    'Order': [5_000, 50_000],
  };
  const key = Object.keys(thresholds).find(k => name.includes(k)) || 'Strategy';
  const [warn, crit] = thresholds[key] || [10_000, 100_000];
  if (avg_us === 0) return '#6b7280';
  if (avg_us >= crit) return '#ef4444';
  if (avg_us >= warn) return '#f59e0b';
  return '#22c55e';
}

function HealthGauge({ score }: { score: number }) {
  const color = score >= 80 ? '#22c55e' : score >= 50 ? '#f59e0b' : '#ef4444';
  const label = score >= 80 ? '健康' : score >= 50 ? '一般' : '需优化';
  const angle = (score / 100) * 180;
  return (
    <div style={{ textAlign: 'center' }}>
      <svg viewBox="0 0 200 120" width="200" height="120">
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#334155" strokeWidth="12" strokeLinecap="round" />
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${(angle / 180) * 251.2} 251.2`}
        />
        <text x="100" y="85" textAnchor="middle" fontSize="28" fontWeight="bold" fill={color}>{score}</text>
        <text x="100" y="110" textAnchor="middle" fontSize="14" fill="#94a3b8">{label}</text>
      </svg>
    </div>
  );
}

function PipelineBar({ modules }: { modules: ModuleInfo[] }) {
  const total = modules.reduce((s, m) => s + m.avg_us, 0);
  if (total === 0) return <div style={{ color: '#64748b', fontSize: 14 }}>引擎未运行或无数据</div>;
  const colors = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'];
  return (
    <div>
      <div style={{ display: 'flex', height: 32, borderRadius: 6, overflow: 'hidden', marginBottom: 8 }}>
        {modules.map((m, i) => (
          <div
            key={m.name}
            title={`${m.name}: ${formatUs(m.avg_us)} (${m.pct_of_pipeline.toFixed(1)}%)`}
            style={{
              width: `${m.pct_of_pipeline}%`,
              minWidth: m.avg_us > 0 ? 4 : 0,
              background: colors[i % colors.length],
              transition: 'width 0.5s ease',
            }}
          />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#94a3b8', flexWrap: 'wrap' }}>
        {modules.map((m, i) => (
          <span key={m.name} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: colors[i % colors.length], display: 'inline-block' }} />
            {m.name.split('(')[0].trim()} {m.pct_of_pipeline.toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Latency() {
  const [data, setData] = useState<LatencyData | null>(null);
  const [refreshInterval, setRefreshInterval] = useState(3);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      const d = await getLatency();
      setData(d);
      setError('');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to fetch latency data');
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, refreshInterval * 1000);
    return () => clearInterval(timer);
  }, [refresh, refreshInterval]);

  return (
    <div className="p-6 max-w-[1100px] mx-auto">
      <div className="flex justify-between items-center mb-5">
        <h1 className="text-[22px] font-bold text-[#f8fafc]">⏱ 延迟分析 & 瓶颈检测</h1>
        <div className="flex gap-2 items-center">
          <span className="text-[13px] text-[#64748b]">刷新间隔:</span>
          {[2, 3, 5, 10].map(s => (
            <button
              key={s}
              onClick={() => setRefreshInterval(s)}
              className={`px-2.5 py-1 text-xs rounded ${
                refreshInterval === s
                  ? 'bg-[#3b82f6] text-white'
                  : 'bg-[#334155] text-[#94a3b8] hover:bg-[#475569]'
              }`}
            >{s}s</button>
          ))}
          <button onClick={refresh} className="px-3 py-1 text-xs rounded bg-[#334155] text-[#94a3b8] hover:bg-[#475569]">
            🔄
          </button>
        </div>
      </div>

      {error && <div className="bg-red-500/10 text-red-400 p-3 rounded-lg mb-4 text-sm">{error}</div>}

      {!data ? (
        <div className="text-[#64748b]">加载中...</div>
      ) : (
        <>
          {/* Health + Pipeline Overview */}
          <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 20 }} className="mb-6">
            <div className="bg-[#1e293b] rounded-lg p-4 border border-[#334155]">
              <div className="text-xs text-[#64748b] mb-1">系统健康度</div>
              <HealthGauge score={data.health_score} />
              <div className="text-center text-xs text-[#64748b] mt-1">
                Pipeline: {formatUs(data.pipeline_total_us)}
              </div>
            </div>
            <div className="bg-[#1e293b] rounded-lg p-4 border border-[#334155]">
              <div className="text-xs text-[#64748b] mb-3">流水线延迟占比</div>
              <PipelineBar modules={data.modules} />
              <div className="mt-4 text-xs text-[#64748b]">
                总延迟 = 数据获取 + 策略计算 + 风控检查 + 订单提交（单次 Pass）
              </div>
            </div>
          </div>

          {/* Bottleneck Alert */}
          {data.bottleneck && (
            <div className={`rounded-lg p-4 mb-6 border ${
              data.bottleneck.avg_us > 100_000
                ? 'bg-red-500/10 border-red-500/30'
                : 'bg-yellow-500/10 border-yellow-500/30'
            }`}>
              <div className="font-semibold text-[#f8fafc] mb-1">
                🔍 瓶颈模块: {data.bottleneck.module} — {formatUs(data.bottleneck.avg_us)} ({data.bottleneck.pct.toFixed(1)}%)
              </div>
              <div className="text-sm text-[#94a3b8]">{data.bottleneck.suggestion}</div>
            </div>
          )}

          {/* Per-module cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }} className="mb-6">
            {data.modules.map(m => {
              const color = statusColor(m.avg_us, m.name);
              const isBottleneck = data.bottleneck?.module === m.name;
              return (
                <div key={m.name} className={`bg-[#1e293b] rounded-lg p-4 border border-[#334155] ${isBottleneck ? 'ring-2 ring-yellow-500/50' : ''}`}
                  style={{ borderLeft: `4px solid ${color}` }}>
                  <div className="text-sm font-semibold text-[#f8fafc] mb-2 flex justify-between">
                    <span>{m.name.split('(')[0].trim()}</span>
                    {isBottleneck && <span className="text-[11px] bg-yellow-500/20 text-yellow-400 px-1.5 py-0.5 rounded">瓶颈</span>}
                  </div>
                  <div className="text-xs text-[#64748b]">{m.name.match(/\((.+)\)/)?.[1]}</div>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <div>
                      <div className="text-[11px] text-[#64748b]">最近</div>
                      <div className="text-base font-bold" style={{ color }}>{formatUs(m.last_us)}</div>
                    </div>
                    <div>
                      <div className="text-[11px] text-[#64748b]">平均</div>
                      <div className="text-base font-bold" style={{ color }}>{formatUs(m.avg_us)}</div>
                    </div>
                    <div>
                      <div className="text-[11px] text-[#64748b]">总调用</div>
                      <div className="text-sm font-semibold text-[#cbd5e1]">{m.total_calls.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-[11px] text-[#64748b]">占比</div>
                      <div className="text-sm font-semibold text-[#cbd5e1]">{m.pct_of_pipeline.toFixed(1)}%</div>
                    </div>
                  </div>
                  <div className="mt-2.5 h-1.5 bg-[#334155] rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.min(m.pct_of_pipeline, 100)}%`, background: color }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Throughput Summary */}
          <div className="bg-[#1e293b] rounded-lg p-4 border border-[#334155] mb-6">
            <div className="text-[15px] font-semibold text-[#f8fafc] mb-3">📊 吞吐量统计</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 12 }}>
              {[
                { label: '处理K线', value: data.throughput.total_bars.toLocaleString(), color: undefined },
                { label: '产生信号', value: data.throughput.total_signals.toLocaleString(), color: undefined },
                { label: '提交订单', value: data.throughput.total_orders.toLocaleString(), color: undefined },
                { label: '成交', value: data.throughput.total_fills.toLocaleString(), color: undefined },
                { label: '拒绝', value: data.throughput.total_rejected.toLocaleString(), color: data.throughput.total_rejected > 0 ? '#ef4444' : undefined },
                { label: '信号率', value: `${(data.throughput.signal_ratio * 100).toFixed(1)}%`, color: undefined },
                { label: '成交率', value: `${(data.throughput.fill_ratio * 100).toFixed(1)}%`, color: undefined },
              ].map(item => (
                <div key={item.label} className="text-center">
                  <div className="text-[11px] text-[#64748b]">{item.label}</div>
                  <div className="text-lg font-bold" style={{ color: item.color || '#f8fafc' }}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Thresholds Reference */}
          <div className="bg-[#0f172a] rounded-lg p-4 text-xs text-[#64748b] border border-[#334155]">
            <div className="font-semibold mb-2 text-[#94a3b8]">阈值参考</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 4 }}>
              <div>🟢 数据获取: &lt;500ms正常 | 🟡 500ms~2s警告 | 🔴 &gt;2s严重</div>
              <div>🟢 策略计算: &lt;10ms正常 | 🟡 10ms~100ms警告 | 🔴 &gt;100ms严重</div>
              <div>🟢 风控检查: &lt;1ms正常 | 🟡 1ms~10ms警告 | 🔴 &gt;10ms严重</div>
              <div>🟢 订单提交: &lt;5ms正常 | 🟡 5ms~50ms警告 | 🔴 &gt;50ms严重</div>
            </div>
          </div>

          {/* Engine status */}
          {!data.engine_running && (
            <div className="mt-4 text-center text-[#64748b] text-sm">
              ⚠️ 引擎未运行 — 启动自动交易后将显示实时延迟数据
            </div>
          )}
        </>
      )}
    </div>
  );
}
