import { useEffect, useState } from 'react';
import { getReports } from '../api/client';

// ── Helpers ──

function fmt(v: number | undefined | null, d = 2): string {
  if (v == null || isNaN(v)) return '-';
  if (!isFinite(v)) return '∞';
  return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}
function pnlColor(v: number): string { return v > 0 ? '#22c55e' : v < 0 ? '#ef4444' : '#9ca3af'; }

// ── Sub-components ──

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min(Math.abs(value) / max * 100, 100) : 0;
  return <div style={{ height: 6, background: '#374151', borderRadius: 3, width: 60, display: 'inline-block', verticalAlign: 'middle', marginLeft: 6 }}>
    <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 3 }} />
  </div>;
}

function SimpleSvgChart({ data, width = 600, height = 160, yKey, color = '#3b82f6', xKey = 'date', fill = false }: {
  data: any[]; width?: number; height?: number; yKey: string; color?: string; xKey?: string; fill?: boolean;
}) {
  if (!data || data.length < 2) return <div style={{ color: '#6b7280', fontSize: 13 }}>No data available</div>;
  const values = data.map(d => d[yKey] as number);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pad = 20;
  const chartW = width - pad * 2;
  const chartH = height - pad * 2;

  const points = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * chartW;
    const y = pad + chartH - ((v - min) / range) * chartH;
    return `${x},${y}`;
  });

  const zeroY = pad + chartH - ((0 - min) / range) * chartH;

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      {min < 0 && max > 0 && (
        <line x1={pad} y1={zeroY} x2={width - pad} y2={zeroY} stroke="#4b5563" strokeDasharray="4" />
      )}
      {fill && (
        <polygon
          points={`${pad},${pad + chartH} ${points.join(' ')} ${width - pad},${pad + chartH}`}
          fill={color} fillOpacity={0.1}
        />
      )}
      <polyline points={points.join(' ')} fill="none" stroke={color} strokeWidth={2} />
      {/* Y axis labels */}
      <text x={pad - 4} y={pad + 4} fill="#6b7280" fontSize={10} textAnchor="end">{fmt(max, 0)}</text>
      <text x={pad - 4} y={pad + chartH + 4} fill="#6b7280" fontSize={10} textAnchor="end">{fmt(min, 0)}</text>
      {/* X axis labels */}
      {data.length > 0 && (
        <>
          <text x={pad} y={height - 2} fill="#6b7280" fontSize={9}>{data[0][xKey]?.slice(0, 10)}</text>
          <text x={width - pad} y={height - 2} fill="#6b7280" fontSize={9} textAnchor="end">{data[data.length - 1][xKey]?.slice(0, 10)}</text>
        </>
      )}
    </svg>
  );
}

function BarChart({ data, width = 500, height = 120, barKey, labelKey, color = '#3b82f6' }: {
  data: any[]; width?: number; height?: number; barKey: string; labelKey: string; color?: string;
}) {
  if (!data || data.length === 0) return null;
  const maxVal = Math.max(...data.map(d => d[barKey] as number), 1);
  const barW = Math.max(Math.floor((width - 40) / data.length) - 2, 4);

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      {data.map((d, i) => {
        const val = d[barKey] as number;
        const barH = (val / maxVal) * (height - 30);
        const x = 20 + i * (barW + 2);
        return (
          <g key={i}>
            <rect x={x} y={height - 20 - barH} width={barW} height={barH} fill={color} rx={2} />
            {barW >= 12 && <text x={x + barW / 2} y={height - 4} fill="#6b7280" fontSize={8} textAnchor="middle">{d[labelKey]}</text>}
          </g>
        );
      })}
    </svg>
  );
}

// ── Main Component ──

export default function Reports() {
  const [data, setData] = useState<any>(null);
  const [tab, setTab] = useState<'overview' | 'symbols' | 'daily' | 'risk' | 'orders' | 'hourly'>('overview');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getReports()
      .then(d => { setData(d); setError(''); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const refresh = () => {
    setLoading(true);
    getReports()
      .then(d => { setData(d); setError(''); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  const cardStyle: React.CSSProperties = { background: '#1f2937', borderRadius: 8, padding: 16, marginBottom: 16 };
  const headStyle: React.CSSProperties = { fontSize: 14, fontWeight: 600, color: '#9ca3af', marginBottom: 12, textTransform: 'uppercase' as const, letterSpacing: 1 };
  const thStyle: React.CSSProperties = { padding: '6px 10px', textAlign: 'left' as const, borderBottom: '1px solid #374151', fontSize: 12, color: '#9ca3af', fontWeight: 600 };
  const tdStyle: React.CSSProperties = { padding: '6px 10px', borderBottom: '1px solid #1f2937', fontSize: 13, fontFamily: 'monospace' };

  const tabs = [
    { id: 'overview' as const, label: '📊 总览' },
    { id: 'symbols' as const, label: '📈 个股分析' },
    { id: 'daily' as const, label: '📅 每日损益' },
    { id: 'risk' as const, label: '🛡 风控事件' },
    { id: 'orders' as const, label: '💹 订单分析' },
    { id: 'hourly' as const, label: '⏰ 时段分布' },
  ];

  if (loading && !data) return <div style={{ padding: 24, color: '#9ca3af' }}>Loading reports...</div>;

  return (
    <div style={{ padding: 24, color: '#e5e7eb', maxWidth: 1200 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22 }}>📋 统计报表</h2>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          {data && <span style={{ fontSize: 12, color: '#6b7280' }}>生成于 {data.generated_at?.slice(0, 19).replace('T', ' ')}</span>}
          <button onClick={refresh} disabled={loading} style={{ background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}>
            {loading ? '加载中...' : '🔄 刷新'}
          </button>
        </div>
      </div>
      {error && <div style={{ color: '#ef4444', marginBottom: 12 }}>⚠ {error}</div>}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{ padding: '8px 16px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === t.id ? '#3b82f6' : '#374151', color: tab === t.id ? '#fff' : '#9ca3af' }}>
            {t.label}
          </button>
        ))}
      </div>

      {!data ? null : tab === 'overview' ? <OverviewTab data={data} cardStyle={cardStyle} headStyle={headStyle} /> :
        tab === 'symbols' ? <SymbolsTab data={data} cardStyle={cardStyle} headStyle={headStyle} thStyle={thStyle} tdStyle={tdStyle} /> :
        tab === 'daily' ? <DailyTab data={data} cardStyle={cardStyle} headStyle={headStyle} thStyle={thStyle} tdStyle={tdStyle} /> :
        tab === 'risk' ? <RiskTab data={data} cardStyle={cardStyle} headStyle={headStyle} thStyle={thStyle} tdStyle={tdStyle} /> :
        tab === 'orders' ? <OrdersTab data={data} cardStyle={cardStyle} headStyle={headStyle} /> :
        tab === 'hourly' ? <HourlyTab data={data} cardStyle={cardStyle} headStyle={headStyle} /> : null
      }
    </div>
  );
}

// ── Tab Components ──

function OverviewTab({ data, cardStyle, headStyle }: { data: any; cardStyle: React.CSSProperties; headStyle: React.CSSProperties }) {
  const s = data.summary;
  const p = data.performance;
  const o = data.order_analysis;
  const statRow = (label: string, value: string | number, color?: string) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid #374151', fontSize: 13 }}>
      <span>{label}</span>
      <span style={{ fontFamily: 'monospace', color: color || '#e5e7eb' }}>{value}</span>
    </div>
  );

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
      {/* Trading Summary */}
      <div style={cardStyle}>
        <div style={headStyle}>🔢 交易概况</div>
        {statRow('总事件', s.total_events.toLocaleString())}
        {statRow('信号数', s.total_signals.toLocaleString())}
        {statRow('已提交', s.total_submitted.toLocaleString())}
        {statRow('已成交', s.total_filled.toLocaleString())}
        {statRow('被拒绝', s.total_rejected.toLocaleString(), s.total_rejected > 0 ? '#f59e0b' : undefined)}
        {statRow('成交率', `${fmt(s.fill_rate_pct)}%`)}
        <div style={{ marginTop: 12, fontSize: 12, color: '#6b7280' }}>
          事件类型: {Object.entries(s.event_breakdown || {}).map(([k, v]) => `${k}:${v}`).join(', ')}
        </div>
      </div>

      {/* Performance */}
      <div style={cardStyle}>
        <div style={headStyle}>💰 绩效指标</div>
        {p.source === 'live_engine' ? (
          <>
            {statRow('组合价值', `¥${fmt(p.portfolio_value)}`, '#60a5fa')}
            {statRow('初始资金', `¥${fmt(p.initial_capital)}`)}
            {statRow('总收益率', `${fmt(p.total_return_pct)}%`, pnlColor(p.total_return_pct))}
            {statRow('最大回撤', `${fmt(p.max_drawdown_pct)}%`, p.max_drawdown_pct > 5 ? '#ef4444' : '#e5e7eb')}
            {statRow('胜率', `${fmt(p.win_rate)}%`)}
            {statRow('盈亏比', fmt(p.profit_factor))}
            {statRow('盈/亏', `${p.wins} / ${p.losses}`)}
          </>
        ) : (
          <>
            {statRow('总盈亏', `¥${fmt(p.total_pnl)}`, pnlColor(p.total_pnl))}
            {statRow('总利润', `¥${fmt(p.gross_profit)}`, '#22c55e')}
            {statRow('总亏损', `¥${fmt(p.gross_loss)}`, '#ef4444')}
            {statRow('胜率', `${fmt(p.win_rate)}%`)}
            {statRow('盈亏比', fmt(p.profit_factor))}
            {statRow('盈/亏', `${p.wins} / ${p.losses}`)}
          </>
        )}
      </div>

      {/* Order Analysis */}
      <div style={cardStyle}>
        <div style={headStyle}>📊 订单统计</div>
        {statRow('完整交易', o.total_round_trips)}
        {statRow('总盈亏', `¥${fmt(o.total_pnl)}`, pnlColor(o.total_pnl))}
        {statRow('平均盈亏', `¥${fmt(o.avg_pnl)}`, pnlColor(o.avg_pnl))}
        {statRow('平均盈利', `¥${fmt(o.avg_win)}`, '#22c55e')}
        {statRow('平均亏损', `¥${fmt(o.avg_loss)}`, '#ef4444')}
        {statRow('期望收益', `¥${fmt(o.expectancy)}`, pnlColor(o.expectancy))}
        {statRow('最大连胜', o.max_consecutive_wins)}
        {statRow('最大连亏', o.max_consecutive_losses, o.max_consecutive_losses > 3 ? '#ef4444' : undefined)}
        {statRow('单笔最大盈利', `¥${fmt(o.best_trade)}`, '#22c55e')}
        {statRow('单笔最大亏损', `¥${fmt(o.worst_trade)}`, '#ef4444')}
      </div>

      {/* Reject Summary */}
      <div style={cardStyle}>
        <div style={headStyle}>🚫 拒绝原因分布</div>
        {(data.reject_summary || []).length === 0 ? (
          <div style={{ color: '#6b7280', fontSize: 13 }}>暂无拒绝记录</div>
        ) : (
          (data.reject_summary as Array<{ reason: string; count: number }>).map((r, i) => {
            const maxCount = Math.max(...data.reject_summary.map((x: any) => x.count), 1);
            return (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0', fontSize: 13 }}>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.reason}</span>
                <span style={{ fontFamily: 'monospace', marginLeft: 8 }}>{r.count}</span>
                <MiniBar value={r.count} max={maxCount} color="#f59e0b" />
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function SymbolsTab({ data, cardStyle, headStyle, thStyle, tdStyle }: any) {
  const symbols = data.symbols || [];
  if (symbols.length === 0) return <div style={cardStyle}><div style={{ color: '#6b7280' }}>暂无个股数据</div></div>;
  const maxPnl = Math.max(...symbols.map((s: any) => Math.abs(s.total_pnl)), 1);

  return (
    <div style={cardStyle}>
      <div style={headStyle}>📈 个股表现排名</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['代码', '信号', '成交', '拒绝', '买/卖', '盈/亏', '胜率', '盈亏比', '总盈亏', '成交量'].map(h => (
                <th key={h} style={thStyle}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {symbols.map((s: any, i: number) => (
              <tr key={i} style={{ background: i % 2 === 0 ? '#111827' : 'transparent' }}>
                <td style={{ ...tdStyle, color: '#60a5fa', fontWeight: 600 }}>{s.symbol}</td>
                <td style={tdStyle}>{s.signals}</td>
                <td style={tdStyle}>{s.fills}</td>
                <td style={{ ...tdStyle, color: s.rejected > 0 ? '#f59e0b' : '#e5e7eb' }}>{s.rejected}</td>
                <td style={tdStyle}>{s.buy_count}/{s.sell_count}</td>
                <td style={tdStyle}>{s.wins}/{s.losses}</td>
                <td style={tdStyle}>{fmt(s.win_rate)}%</td>
                <td style={tdStyle}>{s.profit_factor === Infinity ? '∞' : fmt(s.profit_factor)}</td>
                <td style={{ ...tdStyle, color: pnlColor(s.total_pnl) }}>
                  ¥{fmt(s.total_pnl)}
                  <MiniBar value={s.total_pnl} max={maxPnl} color={pnlColor(s.total_pnl)} />
                </td>
                <td style={tdStyle}>{fmt(s.total_volume, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DailyTab({ data, cardStyle, headStyle, thStyle, tdStyle }: any) {
  const daily = data.daily_pnl || [];

  return (
    <>
      {/* Cumulative PnL Chart */}
      <div style={cardStyle}>
        <div style={headStyle}>📈 累计损益曲线</div>
        <SimpleSvgChart data={daily} yKey="cumulative_pnl" color="#3b82f6" fill width={900} height={200} />
      </div>

      {/* Daily PnL Bar Chart */}
      <div style={cardStyle}>
        <div style={headStyle}>📊 每日损益</div>
        {daily.length > 0 ? (
          <svg width={900} height={140}>
            {daily.map((d: any, i: number) => {
              const maxAbs = Math.max(...daily.map((x: any) => Math.abs(x.daily_pnl)), 1);
              const barH = Math.abs(d.daily_pnl) / maxAbs * 50;
              const x = 30 + i * Math.max(Math.floor(870 / daily.length), 3);
              const isPos = d.daily_pnl >= 0;
              return <rect key={i} x={x} y={isPos ? 70 - barH : 70} width={Math.max(Math.floor(870 / daily.length) - 1, 2)} height={barH}
                fill={isPos ? '#22c55e' : '#ef4444'} rx={1} />;
            })}
            <line x1={30} y1={70} x2={900} y2={70} stroke="#4b5563" strokeDasharray="4" />
          </svg>
        ) : <div style={{ color: '#6b7280', fontSize: 13 }}>暂无每日数据</div>}
      </div>

      {/* Daily Table */}
      <div style={cardStyle}>
        <div style={headStyle}>📅 每日明细</div>
        {daily.length === 0 ? <div style={{ color: '#6b7280', fontSize: 13 }}>暂无数据</div> : (
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                {['日期', '组合价值', '现金', '持仓数', '日盈亏', '累计盈亏', '交易数'].map(h => <th key={h} style={thStyle}>{h}</th>)}
              </tr></thead>
              <tbody>
                {[...daily].reverse().map((d: any, i: number) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#111827' : 'transparent' }}>
                    <td style={tdStyle}>{d.date}</td>
                    <td style={tdStyle}>¥{fmt(d.portfolio_value)}</td>
                    <td style={tdStyle}>¥{fmt(d.cash)}</td>
                    <td style={tdStyle}>{d.positions}</td>
                    <td style={{ ...tdStyle, color: pnlColor(d.daily_pnl) }}>¥{fmt(d.daily_pnl)}</td>
                    <td style={{ ...tdStyle, color: pnlColor(d.cumulative_pnl) }}>¥{fmt(d.cumulative_pnl)}</td>
                    <td style={tdStyle}>{d.total_trades}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}

function RiskTab({ data, cardStyle, headStyle, thStyle, tdStyle }: any) {
  const events = data.risk_events || [];
  return (
    <div style={cardStyle}>
      <div style={headStyle}>🛡 最近100条风控事件</div>
      {events.length === 0 ? <div style={{ color: '#6b7280', fontSize: 13 }}>暂无风控事件</div> : (
        <div style={{ maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              {['时间', '代码', '方向', '数量', '价格', '原因'].map(h => <th key={h} style={thStyle}>{h}</th>)}
            </tr></thead>
            <tbody>
              {events.map((e: any, i: number) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#111827' : 'transparent' }}>
                  <td style={{ ...tdStyle, fontSize: 11 }}>{e.timestamp?.slice(0, 19)}</td>
                  <td style={{ ...tdStyle, color: '#60a5fa' }}>{e.symbol}</td>
                  <td style={tdStyle}>{e.side || '-'}</td>
                  <td style={tdStyle}>{e.quantity != null ? fmt(e.quantity, 0) : '-'}</td>
                  <td style={tdStyle}>{e.price != null ? fmt(e.price) : '-'}</td>
                  <td style={{ ...tdStyle, color: '#f59e0b', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.reason || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function OrdersTab({ data, cardStyle, headStyle }: any) {
  const o = data.order_analysis;
  const statRow = (label: string, value: string | number, color?: string) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid #374151', fontSize: 13 }}>
      <span>{label}</span><span style={{ fontFamily: 'monospace', color: color || '#e5e7eb' }}>{value}</span>
    </div>
  );

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
      <div style={cardStyle}>
        <div style={headStyle}>💹 盈亏分析</div>
        {statRow('完整交易轮次', o.total_round_trips)}
        {statRow('总盈亏', `¥${fmt(o.total_pnl)}`, pnlColor(o.total_pnl))}
        {statRow('平均每笔盈亏', `¥${fmt(o.avg_pnl)}`, pnlColor(o.avg_pnl))}
        {statRow('数学期望', `¥${fmt(o.expectancy)}`, pnlColor(o.expectancy))}
      </div>
      <div style={cardStyle}>
        <div style={headStyle}>🏆 极值统计</div>
        {statRow('平均盈利', `¥${fmt(o.avg_win)}`, '#22c55e')}
        {statRow('平均亏损', `¥${fmt(o.avg_loss)}`, '#ef4444')}
        {statRow('单笔最大盈利', `¥${fmt(o.best_trade)}`, '#22c55e')}
        {statRow('单笔最大亏损', `¥${fmt(o.worst_trade)}`, '#ef4444')}
      </div>
      <div style={cardStyle}>
        <div style={headStyle}>📊 连续统计</div>
        {statRow('最大连胜', o.max_consecutive_wins)}
        {statRow('最大连亏', o.max_consecutive_losses, o.max_consecutive_losses > 3 ? '#ef4444' : undefined)}
        {statRow('盈亏比 (avg_win/avg_loss)', o.avg_loss > 0 ? fmt(o.avg_win / o.avg_loss) : '∞')}
      </div>
      {/* Latency */}
      {data.latency && (
        <div style={cardStyle}>
          <div style={headStyle}>⚡ 延迟统计</div>
          {statRow('因子计算 (avg)', `${data.latency.avg_factor_us}μs`)}
          {statRow('因子计算 (last)', `${data.latency.last_factor_us}μs`)}
          {statRow('风控检查 (last)', `${data.latency.last_risk_us}μs`)}
          {statRow('订单提交 (last)', `${data.latency.last_order_us}μs`)}
          {statRow('已处理K线', data.latency.total_bars.toLocaleString())}
        </div>
      )}
    </div>
  );
}

function HourlyTab({ data, cardStyle, headStyle }: any) {
  const hourly = data.hourly_distribution || [];
  const tradingHours = hourly.filter((h: any) => h.signals > 0 || h.fills > 0);
  const maxSignals = Math.max(...hourly.map((h: any) => h.signals), 1);
  const maxFills = Math.max(...hourly.map((h: any) => h.fills), 1);

  return (
    <>
      <div style={cardStyle}>
        <div style={headStyle}>📊 信号时段分布</div>
        <BarChart data={hourly} barKey="signals" labelKey="hour" color="#3b82f6" width={900} height={140} />
        <div style={{ fontSize: 12, color: '#6b7280', marginTop: 8 }}>X轴: 小时 (0-23), Y轴: 信号数量</div>
      </div>
      <div style={cardStyle}>
        <div style={headStyle}>📊 成交时段分布</div>
        <BarChart data={hourly} barKey="fills" labelKey="hour" color="#22c55e" width={900} height={140} />
        <div style={{ fontSize: 12, color: '#6b7280', marginTop: 8 }}>X轴: 小时 (0-23), Y轴: 成交数量</div>
      </div>
      {tradingHours.length > 0 && (
        <div style={cardStyle}>
          <div style={headStyle}>⏰ 活跃时段明细</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 8 }}>
            {tradingHours.map((h: any) => (
              <div key={h.hour} style={{ background: '#111827', borderRadius: 6, padding: 10, textAlign: 'center' }}>
                <div style={{ fontSize: 16, fontWeight: 600 }}>{String(h.hour).padStart(2, '0')}:00</div>
                <div style={{ fontSize: 11, color: '#3b82f6' }}>信号 {h.signals}
                  <MiniBar value={h.signals} max={maxSignals} color="#3b82f6" />
                </div>
                <div style={{ fontSize: 11, color: '#22c55e' }}>成交 {h.fills}
                  <MiniBar value={h.fills} max={maxFills} color="#22c55e" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
