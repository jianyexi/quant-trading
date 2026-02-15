import { useState, useEffect, useRef } from 'react';
import {
  Shield, ShieldAlert, ShieldCheck, RefreshCw, RotateCcw, AlertTriangle,
  TrendingDown, Zap, Activity, Ban, CheckCircle, XCircle, Loader2,
} from 'lucide-react';
import { getRiskStatus, getPerformance, resetCircuitBreaker, resetDailyLoss } from '../api/client';

interface RiskStatusData {
  daily_pnl: number;
  daily_paused: boolean;
  drawdown_halted: boolean;
  circuit_open: boolean;
  consecutive_failures: number;
  peak_value: number;
  config: {
    stop_loss_pct: number;
    max_daily_loss_pct: number;
    max_drawdown_pct: number;
    circuit_breaker_failures: number;
    halt_on_drawdown: boolean;
  };
}

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
  const [risk, setRisk] = useState<RiskStatusData | null>(null);
  const [perf, setPerf] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionMsg, setActionMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchAll = async () => {
    try {
      const [r, p] = await Promise.all([getRiskStatus(), getPerformance()]);
      setRisk(r as RiskStatusData);
      setPerf(p as PerformanceData);
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => {
    fetchAll();
    pollRef.current = setInterval(fetchAll, 5000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

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

  const config = risk?.config;
  const hasAlerts = risk?.daily_paused || risk?.drawdown_halted || risk?.circuit_open;

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
        <button onClick={fetchAll} className="p-2 rounded-lg bg-[#334155] hover:bg-[#475569] text-[#94a3b8]">
          <RefreshCw className="h-4 w-4" />
        </button>
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
            {risk?.daily_paused && (
              <div className="flex items-center justify-between bg-red-500/10 rounded-lg px-4 py-3">
                <div className="flex items-center gap-2 text-red-300">
                  <Ban className="h-4 w-4" />
                  <span className="text-sm font-medium">日内亏损触发暂停</span>
                  <span className="text-xs text-red-400">日亏损: ¥{risk.daily_pnl.toFixed(2)}</span>
                </div>
                <button onClick={handleResetDaily}
                  className="flex items-center gap-1 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded-lg">
                  <RotateCcw className="h-3 w-3" /> 重置日亏损
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
                  <RotateCcw className="h-3 w-3" /> 重置熔断器
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

      {/* Risk Status Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatusCard
          label="止损"
          icon={<Shield className="h-5 w-5" />}
          value={`${((config?.stop_loss_pct ?? 0) * 100).toFixed(1)}%`}
          sub="每仓位最大亏损"
          ok={true}
        />
        <StatusCard
          label="日内亏损限制"
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
            title="熔断阈值"
            value={`${config?.circuit_breaker_failures ?? 5} 次`}
            desc="连续订单失败超过此次数后触发熔断，暂停所有下单"
            field="circuit_breaker_failures"
          />
          <RuleCard
            title="回撤自动停止"
            value={config?.halt_on_drawdown ? '启用' : '禁用'}
            desc="最大回撤触发时是否自动停止交易引擎"
            field="halt_on_drawdown"
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
