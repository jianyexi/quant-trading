import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Shield, ShieldAlert, ShieldCheck, RefreshCw, RotateCcw, AlertTriangle,
  TrendingDown, Zap, Activity, Ban, CheckCircle, XCircle, Loader2,
  Radio, Gauge, GitBranch, Clock,
} from 'lucide-react';
import {
  getRiskSignals, getPerformance, resetCircuitBreaker, resetDailyLoss,
  createMonitorWebSocket,
  type RiskSignalsSnapshot, type RiskEvent, type TailRiskData,
} from '../api/client';

interface PerformanceData {
  portfolio_value: number;
  initial_capital: number;
  total_return_pct: number;
  peak_value: number;
  drawdown_pct: number;
  max_drawdown_pct: number;
  win_rate: number;
  wins: number;
  losses: number;
  profit_factor: number;
  avg_trade_pnl: number;
  risk_daily_pnl: number;
  risk_daily_paused: boolean;
  risk_circuit_open: boolean;
  risk_drawdown_halted: boolean;
}

export default function RiskManagement() {
  const [snapshot, setSnapshot] = useState<RiskSignalsSnapshot | null>(null);
  const [perf, setPerf] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const [actionMsg, setActionMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchAll = useCallback(async () => {
    try {
      const [s, p] = await Promise.all([getRiskSignals(), getPerformance()]);
      setSnapshot(s);
      setPerf(p as PerformanceData);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  // WebSocket for real-time updates
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      try {
        ws = createMonitorWebSocket();
        wsRef.current = ws;

        ws.onopen = () => setWsConnected(true);
        ws.onclose = () => {
          setWsConnected(false);
          reconnectTimer = setTimeout(connect, 5000);
        };
        ws.onerror = () => ws?.close();
        ws.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data);
            if (data.type === 'risk_update' && data.risk) {
              setSnapshot(data.risk);
              if (data.performance) {
                setPerf(prev => prev ? { ...prev, ...data.performance } : prev);
              }
            }
          } catch { /* ignore parse errors */ }
        };
      } catch { /* ignore */ }
    };

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  // Fallback polling when WS is not connected
  useEffect(() => {
    fetchAll();
    pollRef.current = setInterval(fetchAll, wsConnected ? 30000 : 5000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [wsConnected, fetchAll]);

  const showAction = (text: string, type: 'success' | 'error') => {
    setActionMsg({ text, type });
    setTimeout(() => setActionMsg(null), 3000);
  };

  const handleResetCircuit = async () => {
    try {
      await resetCircuitBreaker();
      showAction('熔断器已重置', 'success');
      fetchAll();
    } catch {
      showAction('重置失败', 'error');
    }
  };

  const handleResetDaily = async () => {
    try {
      await resetDailyLoss();
      showAction('日内亏损已重置', 'success');
      fetchAll();
    } catch {
      showAction('重置失败', 'error');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-[#3b82f6]" />
      </div>
    );
  }

  const risk = snapshot?.status;
  const config = risk?.config;
  const hasAlerts = risk?.daily_paused || risk?.drawdown_halted || risk?.circuit_open || risk?.vol_spike_active;

  return (
    <div className="space-y-6 text-[#f8fafc]">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            {hasAlerts ? <ShieldAlert className="h-7 w-7 text-red-400" /> : <ShieldCheck className="h-7 w-7 text-green-400" />}
            风控管理
          </h1>
          <p className="text-sm text-[#94a3b8] mt-1">实时风险监控 · 止损管理 · 熔断控制</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
            wsConnected ? 'bg-green-500/10 text-green-400 border border-green-500/30' : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/30'
          }`}>
            <Radio className={`h-3 w-3 ${wsConnected ? 'animate-pulse' : ''}`} />
            {wsConnected ? '实时连接' : '轮询模式'}
          </div>
          <button onClick={fetchAll} className="p-2 rounded-lg bg-[#334155] hover:bg-[#475569] text-[#94a3b8]">
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Action message */}
      {actionMsg && (
        <div className={`flex items-center gap-2 text-sm p-3 rounded-lg border ${
          actionMsg.type === 'success' ? 'bg-green-500/10 border-green-500/30 text-green-400' : 'bg-red-500/10 border-red-500/30 text-red-400'
        }`}>
          {actionMsg.type === 'success' ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
          {actionMsg.text}
        </div>
      )}

      {/* Active Alerts */}
      {hasAlerts && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5">
          <h2 className="text-sm font-bold text-red-400 mb-3 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5" /> 风控警报
          </h2>
          <div className="space-y-2">
            {risk?.vol_spike_active && (
              <div className="flex items-center gap-2 bg-orange-500/10 rounded-lg px-4 py-3 text-orange-300">
                <Activity className="h-4 w-4" />
                <span className="text-sm font-medium">波动率突变 — 买入已暂停</span>
                <span className="text-xs text-orange-400">自动减仓中</span>
              </div>
            )}
            {risk?.daily_paused && (
              <div className="flex items-center justify-between bg-red-500/10 rounded-lg px-4 py-3">
                <div className="flex items-center gap-2 text-red-300">
                  <Ban className="h-4 w-4" />
                  <span className="text-sm font-medium">日内亏损触发暂停</span>
                  <span className="text-xs text-red-400">日亏损: ¥{risk.daily_pnl.toFixed(2)}</span>
                </div>
                <button onClick={handleResetDaily}
                  className="flex items-center gap-1 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded-lg">
                  <RotateCcw className="h-3 w-3" /> 重置
                </button>
              </div>
            )}
            {risk?.circuit_open && (
              <div className="flex items-center justify-between bg-orange-500/10 rounded-lg px-4 py-3">
                <div className="flex items-center gap-2 text-orange-300">
                  <Zap className="h-4 w-4" />
                  <span className="text-sm font-medium">熔断器触发</span>
                  <span className="text-xs text-orange-400">连续失败: {risk.consecutive_failures}次</span>
                </div>
                <button onClick={handleResetCircuit}
                  className="flex items-center gap-1 px-3 py-1.5 bg-orange-600 hover:bg-orange-700 text-white text-xs font-medium rounded-lg">
                  <RotateCcw className="h-3 w-3" /> 重置
                </button>
              </div>
            )}
            {risk?.drawdown_halted && (
              <div className="flex items-center gap-2 bg-red-500/10 rounded-lg px-4 py-3 text-red-300">
                <TrendingDown className="h-4 w-4" />
                <span className="text-sm font-medium">最大回撤触发停止</span>
                <span className="text-xs text-red-400">需重启交易引擎</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Risk Status Cards — 6 columns */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatusCard
          label="止损"
          icon={<Shield className="h-5 w-5" />}
          value={`${((config?.stop_loss_pct ?? 0) * 100).toFixed(1)}%`}
          sub="每仓位最大亏损"
          ok={true}
        />
        <StatusCard
          label="日内亏损"
          icon={<TrendingDown className="h-5 w-5" />}
          value={`${((config?.max_daily_loss_pct ?? 0) * 100).toFixed(1)}%`}
          sub={`当前: ¥${(risk?.daily_pnl ?? 0).toFixed(0)}`}
          ok={!risk?.daily_paused}
        />
        <StatusCard
          label="最大回撤"
          icon={<AlertTriangle className="h-5 w-5" />}
          value={`${((config?.max_drawdown_pct ?? 0) * 100).toFixed(0)}%`}
          sub={`当前: ${((perf?.drawdown_pct ?? 0) * 100).toFixed(2)}%`}
          ok={!risk?.drawdown_halted}
        />
        <StatusCard
          label="熔断器"
          icon={<Zap className="h-5 w-5" />}
          value={`${risk?.consecutive_failures ?? 0} / ${config?.circuit_breaker_failures ?? 5}`}
          sub="连续失败次数"
          ok={!risk?.circuit_open}
        />
        <StatusCard
          label="波动率"
          icon={<Activity className="h-5 w-5" />}
          value={risk?.vol_spike_active ? '⚡ 突变' : '正常'}
          sub={`阈值: ${((config?.vol_spike_ratio ?? 2) ).toFixed(1)}x`}
          ok={!risk?.vol_spike_active}
        />
        <StatusCard
          label="VaR"
          icon={<Gauge className="h-5 w-5" />}
          value={snapshot?.tail_risk ? `${(snapshot.tail_risk.var_pct * 100).toFixed(2)}%` : 'N/A'}
          sub={snapshot?.tail_risk ? `CVaR: ${(snapshot.tail_risk.cvar_pct * 100).toFixed(2)}%` : `需${60 - (snapshot?.return_history_len ?? 0)}天数据`}
          ok={!snapshot?.tail_risk?.breach}
        />
      </div>

      {/* VaR / Tail Risk Detail + Event Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tail Risk Panel */}
        <TailRiskPanel tailRisk={snapshot?.tail_risk ?? null} config={config} historyLen={snapshot?.return_history_len ?? 0} />

        {/* Event Timeline */}
        <EventTimeline events={snapshot?.recent_events ?? []} />
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Portfolio Overview */}
        <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-5">
          <h2 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Activity className="h-4 w-4 text-[#3b82f6]" /> 组合概况
          </h2>
          <div className="space-y-3">
            <MetricRow label="组合净值" value={`¥${(perf?.portfolio_value ?? 0).toLocaleString('zh-CN', { minimumFractionDigits: 0 })}`} />
            <MetricRow label="初始资金" value={`¥${(perf?.initial_capital ?? 0).toLocaleString('zh-CN', { minimumFractionDigits: 0 })}`} />
            <MetricRow
              label="总收益率"
              value={`${(perf?.total_return_pct ?? 0).toFixed(2)}%`}
              color={(perf?.total_return_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <MetricRow label="峰值净值" value={`¥${(perf?.peak_value ?? 0).toLocaleString('zh-CN', { minimumFractionDigits: 0 })}`} />
            <MetricRow
              label="当前回撤"
              value={`${((perf?.drawdown_pct ?? 0) * 100).toFixed(2)}%`}
              color={(perf?.drawdown_pct ?? 0) > 0.05 ? 'text-red-400' : 'text-[#f8fafc]'}
            />
            <MetricRow
              label="最大回撤"
              value={`${((perf?.max_drawdown_pct ?? 0) * 100).toFixed(2)}%`}
              color={(perf?.max_drawdown_pct ?? 0) > 0.05 ? 'text-orange-400' : 'text-[#f8fafc]'}
            />
          </div>
        </div>

        {/* Trade Statistics */}
        <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-5">
          <h2 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Activity className="h-4 w-4 text-green-400" /> 交易统计
          </h2>
          <div className="space-y-3">
            <MetricRow
              label="胜率"
              value={`${((perf?.win_rate ?? 0) * 100).toFixed(1)}%`}
              color={(perf?.win_rate ?? 0) >= 0.5 ? 'text-green-400' : 'text-orange-400'}
            />
            <MetricRow label="盈利次数" value={String(perf?.wins ?? 0)} color="text-green-400" />
            <MetricRow label="亏损次数" value={String(perf?.losses ?? 0)} color="text-red-400" />
            <MetricRow
              label="盈亏比"
              value={(perf?.profit_factor ?? 0).toFixed(2)}
              color={(perf?.profit_factor ?? 0) >= 1.5 ? 'text-green-400' : (perf?.profit_factor ?? 0) >= 1.0 ? 'text-[#f8fafc]' : 'text-red-400'}
            />
            <MetricRow
              label="平均交易盈亏"
              value={`¥${(perf?.avg_trade_pnl ?? 0).toFixed(2)}`}
              color={(perf?.avg_trade_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <MetricRow
              label="日内盈亏"
              value={`¥${(perf?.risk_daily_pnl ?? risk?.daily_pnl ?? 0).toFixed(2)}`}
              color={(perf?.risk_daily_pnl ?? risk?.daily_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}
            />
          </div>
        </div>
      </div>

      {/* Risk Rules Configuration (read-only display) */}
      <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-5">
        <h2 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <Shield className="h-4 w-4 text-purple-400" /> 风控规则配置
        </h2>
        <p className="text-xs text-[#64748b] mb-4">规则在 config/default.toml 中配置，修改后需重启服务生效</p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <RuleCard
            title="止损比例"
            value={`${((config?.stop_loss_pct ?? 0.05) * 100).toFixed(1)}%`}
            desc="单个仓位未实现亏损超过此比例时自动卖出"
            field="stop_loss_pct"
          />
          <RuleCard
            title="日内亏损上限"
            value={`${((config?.max_daily_loss_pct ?? 0.03) * 100).toFixed(1)}%`}
            desc="日内已实现亏损超过初始资金此比例时暂停买入"
            field="max_daily_loss_pct"
          />
          <RuleCard
            title="最大回撤"
            value={`${((config?.max_drawdown_pct ?? 0.10) * 100).toFixed(0)}%`}
            desc="组合净值从峰值回撤超过此比例时停止交易"
            field="max_drawdown_pct"
          />
          <RuleCard
            title="波动率突变阈值"
            value={`${(config?.vol_spike_ratio ?? 2.0).toFixed(1)}x`}
            desc="短期波动率/长期波动率超过此倍数时自动减仓"
            field="vol_spike_ratio"
          />
          <RuleCard
            title="减仓因子"
            value={`${((config?.vol_deleverage_factor ?? 0.5) * 100).toFixed(0)}%`}
            desc="波动率突变时保留的仓位比例"
            field="vol_deleverage_factor"
          />
          <RuleCard
            title="VaR上限"
            value={`${((config?.max_var_pct ?? 0.05) * 100).toFixed(1)}%`}
            desc={`${((config?.var_confidence ?? 0.95) * 100).toFixed(0)}%置信度下的最大可接受损失`}
            field="max_var_pct"
          />
        </div>
      </div>

      {/* Manual Reset Controls */}
      <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-5">
        <h2 className="text-sm font-semibold mb-4">手动操作</h2>
        <div className="flex flex-wrap gap-3">
          <button onClick={handleResetDaily}
            className="flex items-center gap-2 px-4 py-2.5 bg-[#334155] hover:bg-[#475569] text-[#f8fafc] text-sm font-medium rounded-lg transition-colors">
            <RotateCcw className="h-4 w-4" /> 重置日内亏损
          </button>
          <button onClick={handleResetCircuit}
            className="flex items-center gap-2 px-4 py-2.5 bg-[#334155] hover:bg-[#475569] text-[#f8fafc] text-sm font-medium rounded-lg transition-colors">
            <Zap className="h-4 w-4" /> 重置熔断器
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Tail Risk Panel ─────────────────────────────────────────────────

function TailRiskPanel({ tailRisk, config, historyLen }: {
  tailRisk: TailRiskData | null;
  config: RiskSignalsSnapshot['status']['config'] | undefined;
  historyLen: number;
}) {
  const maxVar = (config?.max_var_pct ?? 0.05) * 100;

  return (
    <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-5">
      <h2 className="text-sm font-semibold mb-4 flex items-center gap-2">
        <Gauge className="h-4 w-4 text-amber-400" /> 尾部风险 (VaR / CVaR)
      </h2>
      {tailRisk ? (
        <div className="space-y-4">
          {/* VaR Gauge */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs text-[#94a3b8]">{(tailRisk.confidence * 100).toFixed(0)}% VaR</span>
              <span className={`text-sm font-bold ${tailRisk.breach ? 'text-red-400' : 'text-green-400'}`}>
                {(tailRisk.var_pct * 100).toFixed(2)}%
              </span>
            </div>
            <div className="w-full bg-[#0f172a] rounded-full h-3 relative overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${tailRisk.breach ? 'bg-red-500' : 'bg-green-500'}`}
                style={{ width: `${Math.min((tailRisk.var_pct / (config?.max_var_pct ?? 0.05)) * 100, 100)}%` }}
              />
              {/* Threshold marker */}
              <div className="absolute top-0 h-full w-0.5 bg-yellow-400" style={{ left: '100%' }} />
            </div>
            <div className="flex justify-between text-xs text-[#64748b] mt-1">
              <span>0%</span>
              <span>上限: {maxVar.toFixed(1)}%</span>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-[#0f172a] rounded-lg p-3">
              <p className="text-xs text-[#94a3b8]">VaR (风险价值)</p>
              <p className={`text-lg font-bold ${tailRisk.breach ? 'text-red-400' : 'text-[#f8fafc]'}`}>
                {(tailRisk.var_pct * 100).toFixed(2)}%
              </p>
              <p className="text-xs text-[#64748b]">{(tailRisk.confidence * 100).toFixed(0)}%置信度日损失</p>
            </div>
            <div className="bg-[#0f172a] rounded-lg p-3">
              <p className="text-xs text-[#94a3b8]">CVaR (条件VaR)</p>
              <p className="text-lg font-bold text-[#f8fafc]">
                {(tailRisk.cvar_pct * 100).toFixed(2)}%
              </p>
              <p className="text-xs text-[#64748b]">极端情况期望损失</p>
            </div>
          </div>

          <p className="text-xs text-[#64748b]">
            基于 {historyLen} 天历史收益率模拟
            {tailRisk.breach && <span className="text-red-400 ml-1">⚠ VaR超过阈值</span>}
          </p>
        </div>
      ) : (
        <div className="text-center py-8 text-[#64748b]">
          <Gauge className="h-10 w-10 mx-auto mb-2 opacity-30" />
          <p className="text-sm">需要至少 30 天收益率数据才能估算</p>
          <p className="text-xs mt-1">当前: {historyLen} 天</p>
        </div>
      )}
    </div>
  );
}

// ── Event Timeline ──────────────────────────────────────────────────

function EventTimeline({ events }: { events: RiskEvent[] }) {
  const reversedEvents = [...events].reverse(); // newest first

  const severityConfig = {
    Critical: { bg: 'bg-red-500/10', border: 'border-red-500/30', dot: 'bg-red-500', text: 'text-red-400' },
    Warning: { bg: 'bg-orange-500/10', border: 'border-orange-500/30', dot: 'bg-orange-500', text: 'text-orange-400' },
    Info: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', dot: 'bg-blue-500', text: 'text-blue-400' },
  };

  return (
    <div className="bg-[#1e293b] rounded-xl border border-[#334155] p-5">
      <h2 className="text-sm font-semibold mb-4 flex items-center gap-2">
        <Clock className="h-4 w-4 text-[#3b82f6]" /> 风控事件时间线
      </h2>
      {reversedEvents.length > 0 ? (
        <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
          {reversedEvents.map((event, i) => {
            const cfg = severityConfig[event.severity] || severityConfig.Info;
            const time = new Date(event.timestamp).toLocaleTimeString('zh-CN', { hour12: false });
            const date = new Date(event.timestamp).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
            return (
              <div key={i} className={`flex items-start gap-3 ${cfg.bg} ${cfg.border} border rounded-lg px-3 py-2`}>
                <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${cfg.dot}`} />
                <div className="min-w-0 flex-1">
                  <p className={`text-sm font-medium ${cfg.text}`}>{event.message}</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-xs text-[#64748b] font-mono">{date} {time}</span>
                    <span className="text-xs text-[#475569]">{event.event_type}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-8 text-[#64748b]">
          <GitBranch className="h-10 w-10 mx-auto mb-2 opacity-30" />
          <p className="text-sm">暂无风控事件</p>
          <p className="text-xs mt-1">系统正常运行中</p>
        </div>
      )}
    </div>
  );
}

// ── Shared Components ───────────────────────────────────────────────

function StatusCard({ label, icon, value, sub, ok }: {
  label: string; icon: React.ReactNode; value: string; sub: string; ok: boolean;
}) {
  return (
    <div className={`rounded-xl border p-4 ${
      ok ? 'bg-[#1e293b] border-[#334155]' : 'bg-red-500/10 border-red-500/30'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <span className={ok ? 'text-[#94a3b8]' : 'text-red-400'}>{icon}</span>
        {ok
          ? <CheckCircle className="h-4 w-4 text-green-400" />
          : <XCircle className="h-4 w-4 text-red-400" />
        }
      </div>
      <p className="text-xs text-[#94a3b8]">{label}</p>
      <p className={`text-lg font-bold ${ok ? 'text-[#f8fafc]' : 'text-red-400'}`}>{value}</p>
      <p className="text-xs text-[#64748b] mt-0.5">{sub}</p>
    </div>
  );
}

function MetricRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-sm text-[#94a3b8]">{label}</span>
      <span className={`font-mono font-semibold text-sm ${color ?? 'text-[#f8fafc]'}`}>{value}</span>
    </div>
  );
}

function RuleCard({ title, value, desc, field }: {
  title: string; value: string; desc: string; field: string;
}) {
  return (
    <div className="bg-[#0f172a] rounded-lg border border-[#334155] p-4">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-[#f8fafc]">{title}</span>
        <span className="text-sm font-bold text-purple-400">{value}</span>
      </div>
      <p className="text-xs text-[#64748b] leading-relaxed">{desc}</p>
      <p className="text-xs text-[#475569] mt-1 font-mono">{field}</p>
    </div>
  );
}
