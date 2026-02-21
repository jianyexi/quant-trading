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
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e5e7eb" strokeWidth="12" strokeLinecap="round" />
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${(angle / 180) * 251.2} 251.2`}
        />
        <text x="100" y="85" textAnchor="middle" fontSize="28" fontWeight="bold" fill={color}>{score}</text>
        <text x="100" y="110" textAnchor="middle" fontSize="14" fill="#6b7280">{label}</text>
      </svg>
    </div>
  );
}

function PipelineBar({ modules }: { modules: ModuleInfo[] }) {
  const total = modules.reduce((s, m) => s + m.avg_us, 0);
  if (total === 0) return <div style={{ color: '#9ca3af', fontSize: 14 }}>引擎未运行或无数据</div>;
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
      <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#6b7280', flexWrap: 'wrap' }}>
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
    } catch (e: any) {
      setError(e.message || 'Failed to fetch latency data');
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, refreshInterval * 1000);
    return () => clearInterval(timer);
  }, [refresh, refreshInterval]);

  return (
    <div style={{ padding: 24, maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, margin: 0 }}>⏱ 延迟分析 & 瓶颈检测</h1>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ fontSize: 13, color: '#6b7280' }}>刷新间隔:</span>
          {[2, 3, 5, 10].map(s => (
            <button
              key={s}
              onClick={() => setRefreshInterval(s)}
              style={{
                padding: '4px 10px', fontSize: 12, borderRadius: 4, cursor: 'pointer',
                background: refreshInterval === s ? '#3b82f6' : '#f3f4f6',
                color: refreshInterval === s ? '#fff' : '#374151',
                border: 'none',
              }}
            >{s}s</button>
          ))}
          <button onClick={refresh} style={{ padding: '4px 12px', fontSize: 12, borderRadius: 4, cursor: 'pointer', background: '#e0e7ff', border: 'none', color: '#4338ca' }}>
            🔄
          </button>
        </div>
      </div>

      {error && <div style={{ background: '#fef2f2', color: '#dc2626', padding: 12, borderRadius: 6, marginBottom: 16 }}>{error}</div>}

      {!data ? (
        <div style={{ color: '#9ca3af' }}>加载中...</div>
      ) : (
        <>
          {/* Health + Pipeline Overview */}
          <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 20, marginBottom: 24 }}>
            <div style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 4 }}>系统健康度</div>
              <HealthGauge score={data.health_score} />
              <div style={{ textAlign: 'center', fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
                Pipeline: {formatUs(data.pipeline_total_us)}
              </div>
            </div>
            <div style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 12 }}>流水线延迟占比</div>
              <PipelineBar modules={data.modules} />
              <div style={{ marginTop: 16, fontSize: 12, color: '#9ca3af' }}>
                总延迟 = 数据获取 + 策略计算 + 风控检查 + 订单提交（单次 Pass）
              </div>
            </div>
          </div>

          {/* Bottleneck Alert */}
          {data.bottleneck && (
            <div style={{
              background: data.bottleneck.avg_us > 100_000 ? '#fef2f2' : '#fffbeb',
              border: `1px solid ${data.bottleneck.avg_us > 100_000 ? '#fecaca' : '#fde68a'}`,
              borderRadius: 8, padding: 16, marginBottom: 24,
            }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                🔍 瓶颈模块: {data.bottleneck.module} — {formatUs(data.bottleneck.avg_us)} ({data.bottleneck.pct.toFixed(1)}%)
              </div>
              <div style={{ fontSize: 13, color: '#6b7280' }}>{data.bottleneck.suggestion}</div>
            </div>
          )}

          {/* Per-module cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16, marginBottom: 24 }}>
            {data.modules.map(m => {
              const color = statusColor(m.avg_us, m.name);
              const isBottleneck = data.bottleneck?.module === m.name;
              return (
                <div key={m.name} style={{
                  background: '#fff', borderRadius: 8, padding: 16,
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                  borderLeft: `4px solid ${color}`,
                  ...(isBottleneck ? { outline: '2px solid #f59e0b' } : {}),
                }}>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
                    <span>{m.name.split('(')[0].trim()}</span>
                    {isBottleneck && <span style={{ fontSize: 11, background: '#fef3c7', color: '#92400e', padding: '2px 6px', borderRadius: 4 }}>瓶颈</span>}
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{m.name.match(/\((.+)\)/)?.[1]}</div>
                  <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    <div>
                      <div style={{ fontSize: 11, color: '#9ca3af' }}>最近</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color }}>{formatUs(m.last_us)}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: '#9ca3af' }}>平均</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color }}>{formatUs(m.avg_us)}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: '#9ca3af' }}>总调用</div>
                      <div style={{ fontSize: 14, fontWeight: 600 }}>{m.total_calls.toLocaleString()}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: '#9ca3af' }}>占比</div>
                      <div style={{ fontSize: 14, fontWeight: 600 }}>{m.pct_of_pipeline.toFixed(1)}%</div>
                    </div>
                  </div>
                  {/* Mini bar */}
                  <div style={{ marginTop: 10, height: 6, background: '#f3f4f6', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${Math.min(m.pct_of_pipeline, 100)}%`, background: color, borderRadius: 3, transition: 'width 0.5s ease' }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Throughput Summary */}
          <div style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.1)', marginBottom: 24 }}>
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>📊 吞吐量统计</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 12 }}>
              {[
                { label: '处理K线', value: data.throughput.total_bars.toLocaleString() },
                { label: '产生信号', value: data.throughput.total_signals.toLocaleString() },
                { label: '提交订单', value: data.throughput.total_orders.toLocaleString() },
                { label: '成交', value: data.throughput.total_fills.toLocaleString() },
                { label: '拒绝', value: data.throughput.total_rejected.toLocaleString(), color: data.throughput.total_rejected > 0 ? '#ef4444' : undefined },
                { label: '信号率', value: `${(data.throughput.signal_ratio * 100).toFixed(1)}%` },
                { label: '成交率', value: `${(data.throughput.fill_ratio * 100).toFixed(1)}%` },
              ].map(item => (
                <div key={item.label} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#9ca3af' }}>{item.label}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: (item as any).color || '#111' }}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Thresholds Reference */}
          <div style={{ background: '#f9fafb', borderRadius: 8, padding: 16, fontSize: 12, color: '#6b7280' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, color: '#374151' }}>阈值参考</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 4 }}>
              <div>🟢 数据获取: &lt;500ms正常 | 🟡 500ms~2s警告 | 🔴 &gt;2s严重</div>
              <div>🟢 策略计算: &lt;10ms正常 | 🟡 10ms~100ms警告 | 🔴 &gt;100ms严重</div>
              <div>🟢 风控检查: &lt;1ms正常 | 🟡 1ms~10ms警告 | 🔴 &gt;10ms严重</div>
              <div>🟢 订单提交: &lt;5ms正常 | 🟡 5ms~50ms警告 | 🔴 &gt;50ms严重</div>
            </div>
          </div>

          {/* Engine status */}
          {!data.engine_running && (
            <div style={{ marginTop: 16, textAlign: 'center', color: '#9ca3af', fontSize: 13 }}>
              ⚠️ 引擎未运行 — 启动自动交易后将显示实时延迟数据
            </div>
          )}
        </>
      )}
    </div>
  );
}
