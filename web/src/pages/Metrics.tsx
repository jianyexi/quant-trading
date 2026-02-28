import { useEffect, useState } from 'react';
import { getMetrics } from '../api/client';

interface Metrics {
  timestamp: string;
  uptime_secs: number;
  engine: {
    running: boolean;
    strategy?: string;
    symbols?: string[];
    throughput?: {
      total_bars: number;
      total_signals: number;
      total_orders: number;
      total_fills: number;
      total_rejected: number;
      signal_rate: number;
      fill_rate: number;
      reject_rate: number;
    };
    latency?: {
      last_factor_us: number;
      avg_factor_us: number;
      last_risk_us: number;
      last_order_us: number;
    };
    performance?: {
      portfolio_value: number;
      pnl: number;
      total_return_pct: number;
      max_drawdown_pct: number;
      win_rate: number;
      profit_factor: number;
      wins: number;
      losses: number;
    };
    risk?: {
      daily_pnl: number;
      daily_paused: boolean;
      circuit_open: boolean;
      drawdown_halted: boolean;
      peak_value: number;
      consecutive_failures: number;
    };
  };
  api: {
    total_requests: number;
    info: number;
    warnings: number;
    errors: number;
    avg_duration_ms: number;
    p99_duration_ms: number;
    endpoints: Array<{ path: string; calls: number; avg_ms: number }>;
  };
  database: {
    backend: string;
    pool_size?: number;
    idle_connections?: number;
    active_connections?: number;
  };
}

function formatUptime(secs: number): string {
  const d = Math.floor(secs / 86400);
  const h = Math.floor((secs % 86400) / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  const parts: string[] = [];
  if (d > 0) parts.push(`${d}d`);
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  parts.push(`${s}s`);
  return parts.join(' ');
}

function LatencyBar({ label, value, max }: { label: string; value: number; max: number }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  const color = value > 1000 ? '#ef4444' : value > 500 ? '#f59e0b' : '#22c55e';
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
        <span>{label}</span>
        <span style={{ fontFamily: 'monospace', color }}>{value}μs</span>
      </div>
      <div style={{ height: 6, background: '#374151', borderRadius: 3 }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 3, transition: 'width 0.3s' }} />
      </div>
    </div>
  );
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState('');
  const [history, setHistory] = useState<Array<{ t: number; factor: number; bars: number; pnl: number }>>([]);
  const [refreshInterval, setRefreshInterval] = useState(5);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    const load = async () => {
      try {
        const m = await getMetrics();
        setMetrics(m);
        setError('');
        setHistory(prev => {
          const next = [...prev, {
            t: Date.now(),
            factor: m.engine?.latency?.avg_factor_us ?? 0,
            bars: m.engine?.throughput?.total_bars ?? 0,
            pnl: m.engine?.performance?.pnl ?? 0,
          }];
          return next.slice(-60); // keep last 60 data points
        });
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Failed to fetch metrics');
      }
    };
    load();
    timer = setInterval(load, refreshInterval * 1000);
    return () => clearInterval(timer);
  }, [refreshInterval]);

  if (!metrics) return <div style={{ padding: 24, color: '#9ca3af' }}>Loading metrics...</div>;

  const e = metrics.engine;
  const a = metrics.api;
  const db = metrics.database;
  const latencyMax = Math.max(
    e.latency?.last_factor_us ?? 0,
    e.latency?.last_risk_us ?? 0,
    e.latency?.last_order_us ?? 0,
    100,
  );

  const cardStyle: React.CSSProperties = {
    background: '#1f2937', borderRadius: 8, padding: 16, minWidth: 280,
  };
  const headStyle: React.CSSProperties = {
    fontSize: 14, fontWeight: 600, color: '#9ca3af', marginBottom: 12, textTransform: 'uppercase' as const, letterSpacing: 1,
  };
  const statStyle: React.CSSProperties = {
    display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 13, borderBottom: '1px solid #374151',
  };
  const valStyle: React.CSSProperties = { fontFamily: 'monospace', color: '#e5e7eb' };

  // Sparkline: simple SVG mini chart
  const Sparkline = ({ data, color }: { data: number[]; color: string }) => {
    if (data.length < 2) return null;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const w = 200;
    const h = 30;
    const points = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`).join(' ');
    return (
      <svg width={w} height={h} style={{ display: 'block', marginTop: 4 }}>
        <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} />
      </svg>
    );
  };

  // Sort endpoints by call count
  const sortedEndpoints = [...(a.endpoints || [])].sort((x, y) => y.calls - x.calls).slice(0, 10);

  return (
    <div style={{ padding: 24, color: '#e5e7eb', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22 }}>📊 System Metrics</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 13 }}>
          <span style={{ color: '#6b7280' }}>⏱ {formatUptime(metrics.uptime_secs)}</span>
          <select
            value={refreshInterval}
            onChange={e => setRefreshInterval(Number(e.target.value))}
            style={{ background: '#374151', color: '#e5e7eb', border: 'none', borderRadius: 4, padding: '4px 8px', fontSize: 13 }}
          >
            <option value={2}>2s</option>
            <option value={5}>5s</option>
            <option value={10}>10s</option>
            <option value={30}>30s</option>
          </select>
          {error && <span style={{ color: '#ef4444' }}>⚠ {error}</span>}
        </div>
      </div>

      {/* Status bar */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ ...cardStyle, display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px' }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: e.running ? '#22c55e' : '#6b7280' }} />
          <span style={{ fontSize: 14 }}>{e.running ? `Running: ${e.strategy}` : 'Engine Stopped'}</span>
        </div>
        <div style={{ ...cardStyle, display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px' }}>
          <span style={{ fontSize: 14 }}>💾 {db.backend.toUpperCase()}</span>
          {db.pool_size !== undefined && (
            <span style={{ fontSize: 12, color: '#9ca3af' }}>
              {db.active_connections}/{db.pool_size} active
            </span>
          )}
        </div>
        {e.risk && (
          <div style={{ ...cardStyle, display: 'flex', alignItems: 'center', gap: 8, padding: '10px 16px' }}>
            {e.risk.circuit_open && <span style={{ color: '#ef4444' }}>🔴 Circuit Open</span>}
            {e.risk.daily_paused && <span style={{ color: '#f59e0b' }}>⏸ Daily Paused</span>}
            {e.risk.drawdown_halted && <span style={{ color: '#ef4444' }}>🛑 Drawdown Halted</span>}
            {!e.risk.circuit_open && !e.risk.daily_paused && !e.risk.drawdown_halted && (
              <span style={{ color: '#22c55e' }}>✅ Risk OK</span>
            )}
          </div>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        {/* Pipeline Latency */}
        <div style={cardStyle}>
          <div style={headStyle}>⚡ Pipeline Latency</div>
          {e.latency ? (
            <>
              <LatencyBar label="Factor Compute" value={e.latency.last_factor_us} max={latencyMax} />
              <LatencyBar label="Risk Check" value={e.latency.last_risk_us} max={latencyMax} />
              <LatencyBar label="Order Submit" value={e.latency.last_order_us} max={latencyMax} />
              <div style={{ ...statStyle, marginTop: 8 }}>
                <span>Avg Factor</span><span style={valStyle}>{e.latency.avg_factor_us}μs</span>
              </div>
              <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
                Trend: <Sparkline data={history.map(h => h.factor)} color="#60a5fa" />
              </div>
            </>
          ) : (
            <div style={{ color: '#6b7280', fontSize: 13 }}>Engine not running</div>
          )}
        </div>

        {/* Throughput */}
        <div style={cardStyle}>
          <div style={headStyle}>📈 Throughput</div>
          {e.throughput ? (
            <>
              <div style={statStyle}><span>Bars Processed</span><span style={valStyle}>{e.throughput.total_bars.toLocaleString()}</span></div>
              <div style={statStyle}><span>Signals Generated</span><span style={valStyle}>{e.throughput.total_signals.toLocaleString()}</span></div>
              <div style={statStyle}><span>Orders Sent</span><span style={valStyle}>{e.throughput.total_orders.toLocaleString()}</span></div>
              <div style={statStyle}><span>Fills</span><span style={valStyle}>{e.throughput.total_fills.toLocaleString()}</span></div>
              <div style={statStyle}><span>Rejected</span><span style={{ ...valStyle, color: e.throughput.total_rejected > 0 ? '#ef4444' : '#e5e7eb' }}>{e.throughput.total_rejected}</span></div>
              <div style={statStyle}><span>Signal Rate</span><span style={valStyle}>{(e.throughput.signal_rate * 100).toFixed(1)}%</span></div>
              <div style={statStyle}><span>Fill Rate</span><span style={valStyle}>{e.throughput.fill_rate.toFixed(1)}%</span></div>
            </>
          ) : (
            <div style={{ color: '#6b7280', fontSize: 13 }}>Engine not running</div>
          )}
        </div>

        {/* Performance */}
        <div style={cardStyle}>
          <div style={headStyle}>💰 Performance</div>
          {e.performance ? (
            <>
              <div style={statStyle}><span>Portfolio Value</span><span style={valStyle}>¥{e.performance.portfolio_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span></div>
              <div style={statStyle}><span>PnL</span><span style={{ ...valStyle, color: e.performance.pnl >= 0 ? '#22c55e' : '#ef4444' }}>¥{e.performance.pnl.toFixed(2)}</span></div>
              <div style={statStyle}><span>Return</span><span style={{ ...valStyle, color: e.performance.total_return_pct >= 0 ? '#22c55e' : '#ef4444' }}>{e.performance.total_return_pct.toFixed(2)}%</span></div>
              <div style={statStyle}><span>Max Drawdown</span><span style={{ ...valStyle, color: e.performance.max_drawdown_pct > 5 ? '#ef4444' : '#e5e7eb' }}>{e.performance.max_drawdown_pct.toFixed(2)}%</span></div>
              <div style={statStyle}><span>Win Rate</span><span style={valStyle}>{e.performance.win_rate.toFixed(1)}%</span></div>
              <div style={statStyle}><span>Profit Factor</span><span style={valStyle}>{e.performance.profit_factor === Infinity ? '∞' : e.performance.profit_factor.toFixed(2)}</span></div>
              <div style={statStyle}><span>W/L</span><span style={valStyle}>{e.performance.wins}/{e.performance.losses}</span></div>
              <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
                PnL trend: <Sparkline data={history.map(h => h.pnl)} color={history.length > 0 && history[history.length - 1].pnl >= 0 ? '#22c55e' : '#ef4444'} />
              </div>
            </>
          ) : (
            <div style={{ color: '#6b7280', fontSize: 13 }}>Engine not running</div>
          )}
        </div>

        {/* API Requests */}
        <div style={cardStyle}>
          <div style={headStyle}>🌐 API Requests</div>
          <div style={statStyle}><span>Total Requests</span><span style={valStyle}>{a.total_requests.toLocaleString()}</span></div>
          <div style={statStyle}><span>Success (2xx/3xx)</span><span style={{ ...valStyle, color: '#22c55e' }}>{a.info}</span></div>
          <div style={statStyle}><span>Client Errors (4xx)</span><span style={{ ...valStyle, color: a.warnings > 0 ? '#f59e0b' : '#e5e7eb' }}>{a.warnings}</span></div>
          <div style={statStyle}><span>Server Errors (5xx)</span><span style={{ ...valStyle, color: a.errors > 0 ? '#ef4444' : '#e5e7eb' }}>{a.errors}</span></div>
          <div style={statStyle}><span>Avg Latency</span><span style={valStyle}>{a.avg_duration_ms}ms</span></div>
          <div style={statStyle}><span>P99 Latency</span><span style={valStyle}>{a.p99_duration_ms}ms</span></div>
        </div>

        {/* Database */}
        <div style={cardStyle}>
          <div style={headStyle}>💾 Database</div>
          <div style={statStyle}><span>Backend</span><span style={valStyle}>{db.backend.toUpperCase()}</span></div>
          {db.pool_size !== undefined && (
            <>
              <div style={statStyle}><span>Pool Size</span><span style={valStyle}>{db.pool_size}</span></div>
              <div style={statStyle}><span>Idle</span><span style={valStyle}>{db.idle_connections}</span></div>
              <div style={statStyle}><span>Active</span><span style={valStyle}>{db.active_connections}</span></div>
            </>
          )}
        </div>

        {/* Top Endpoints */}
        <div style={cardStyle}>
          <div style={headStyle}>🔗 Top Endpoints</div>
          {sortedEndpoints.length > 0 ? (
            <div style={{ fontSize: 12 }}>
              {sortedEndpoints.map((ep, i) => (
                <div key={i} style={{ ...statStyle, fontSize: 12 }}>
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ep.path}</span>
                  <span style={{ ...valStyle, marginLeft: 8, fontSize: 12 }}>{ep.calls}× {ep.avg_ms.toFixed(0)}ms</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: '#6b7280', fontSize: 13 }}>No requests yet</div>
          )}
        </div>
      </div>

      {/* Debug: Raw RUST_LOG hint */}
      <div style={{ marginTop: 24, padding: 12, background: '#111827', borderRadius: 8, fontSize: 12, color: '#6b7280' }}>
        <strong>💡 Debug Logging:</strong> Set <code style={{ color: '#60a5fa' }}>RUST_LOG=debug</code> for detailed pipeline logs,
        or <code style={{ color: '#60a5fa' }}>RUST_LOG=quant=trace</code> for per-tick tracing.
        Example: <code style={{ color: '#f59e0b' }}>$env:RUST_LOG="quant=debug,quant_broker=trace"; .\target\release\quant.exe serve</code>
      </div>
    </div>
  );
}
