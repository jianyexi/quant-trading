import { useState, useEffect, useCallback } from 'react';
import {
  Play,
  Square,
  RefreshCw,
  Server,
  Cpu,
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
} from 'lucide-react';

interface MlServeStatus {
  service: string;
  managed: boolean;
  process_info: {
    process: string;
    pid?: number;
    started_at?: string;
    uptime_secs?: number;
    exit_code?: number;
    error?: string;
  };
  health: {
    reachable: boolean;
    data?: {
      status?: string;
      model_type?: string;
      device?: string;
      num_features?: number;
    };
    status?: number;
  };
}

function formatUptime(secs: number): string {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return `${h}h ${m}m`;
}

export default function Services() {
  const [status, setStatus] = useState<MlServeStatus | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Config form
  const [modelPath, setModelPath] = useState('ml_models/factor_model.lgb.txt');
  const [httpPort, setHttpPort] = useState(18091);
  const [tcpPort, setTcpPort] = useState(18094);
  const [device, setDevice] = useState('auto');

  const fetchStatus = useCallback(async () => {
    try {
      const resp = await fetch('/api/services/ml-serve/status');
      if (resp.ok) {
        setStatus(await resp.json());
        setError(null);
      }
    } catch {
      // Server not reachable
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleStart = async () => {
    setActionLoading('start');
    setError(null);
    try {
      const resp = await fetch('/api/services/ml-serve/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: modelPath,
          http_port: httpPort,
          tcp_port: tcpPort,
          device,
        }),
      });
      const data = await resp.json();
      if (data.error) setError(data.error);
      // Wait a moment for process to start, then refresh
      setTimeout(fetchStatus, 2000);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start');
    } finally {
      setActionLoading(null);
    }
  };

  const handleStop = async () => {
    setActionLoading('stop');
    setError(null);
    try {
      const resp = await fetch('/api/services/ml-serve/stop', { method: 'POST' });
      const data = await resp.json();
      if (data.error) setError(data.error);
      setTimeout(fetchStatus, 1000);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to stop');
    } finally {
      setActionLoading(null);
    }
  };

  const isRunning = status?.process_info?.process === 'running';
  const isHealthy = status?.health?.reachable === true;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[#f8fafc]">服务管理</h1>
        <button
          onClick={fetchStatus}
          className="flex items-center gap-2 rounded-lg bg-[#334155] px-3 py-2 text-sm text-[#94a3b8] hover:bg-[#475569] hover:text-[#f8fafc] transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
          刷新
        </button>
      </div>

      {error && (
        <div className="rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* ML Serve Card */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#334155] px-6 py-4">
          <div className="flex items-center gap-3">
            <div className={`rounded-lg p-2 ${isRunning && isHealthy ? 'bg-green-500/15' : isRunning ? 'bg-yellow-500/15' : 'bg-[#334155]'}`}>
              <Server className={`h-5 w-5 ${isRunning && isHealthy ? 'text-green-400' : isRunning ? 'text-yellow-400' : 'text-[#64748b]'}`} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-[#f8fafc]">ML Inference Server</h2>
              <p className="text-sm text-[#94a3b8]">ml_serve.py — 模型推理服务 (TcpMq / HTTP)</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isRunning ? (
              <button
                onClick={handleStop}
                disabled={actionLoading !== null}
                className="flex items-center gap-2 rounded-lg bg-red-500/15 px-4 py-2 text-sm font-medium text-red-400 hover:bg-red-500/25 transition-colors disabled:opacity-50"
              >
                {actionLoading === 'stop' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
                停止
              </button>
            ) : (
              <button
                onClick={handleStart}
                disabled={actionLoading !== null}
                className="flex items-center gap-2 rounded-lg bg-green-500/15 px-4 py-2 text-sm font-medium text-green-400 hover:bg-green-500/25 transition-colors disabled:opacity-50"
              >
                {actionLoading === 'start' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                启动
              </button>
            )}
          </div>
        </div>

        {/* Status indicators */}
        <div className="grid grid-cols-3 gap-4 px-6 py-4 border-b border-[#334155]">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              {isRunning
                ? <CheckCircle2 className="h-4 w-4 text-green-400" />
                : <XCircle className="h-4 w-4 text-[#64748b]" />}
              <span className="text-sm text-[#94a3b8]">进程</span>
            </div>
            <span className={`text-sm font-medium ${isRunning ? 'text-green-400' : 'text-[#64748b]'}`}>
              {isRunning ? `Running (PID ${status?.process_info?.pid})` : status?.process_info?.process === 'exited' ? `Exited (code ${status?.process_info?.exit_code})` : 'Stopped'}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              {isHealthy
                ? <Activity className="h-4 w-4 text-green-400" />
                : <Activity className="h-4 w-4 text-[#64748b]" />}
              <span className="text-sm text-[#94a3b8]">Health</span>
            </div>
            <span className={`text-sm font-medium ${isHealthy ? 'text-green-400' : 'text-[#64748b]'}`}>
              {isHealthy ? 'Healthy' : 'Unreachable'}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-[#94a3b8]" />
              <span className="text-sm text-[#94a3b8]">Uptime</span>
            </div>
            <span className="text-sm font-medium text-[#f8fafc]">
              {status?.process_info?.uptime_secs != null
                ? formatUptime(status.process_info.uptime_secs)
                : '—'}
            </span>
          </div>
        </div>

        {/* Health details (when healthy) */}
        {isHealthy && status?.health?.data && (
          <div className="grid grid-cols-4 gap-4 px-6 py-4 border-b border-[#334155] bg-[#0f172a]/50">
            <div>
              <span className="text-xs text-[#64748b] uppercase">Model Type</span>
              <p className="text-sm font-medium text-[#f8fafc] mt-1">{status.health.data.model_type || '—'}</p>
            </div>
            <div>
              <span className="text-xs text-[#64748b] uppercase">Device</span>
              <p className="text-sm font-medium text-[#f8fafc] mt-1">
                <span className="inline-flex items-center gap-1">
                  <Cpu className="h-3 w-3" />
                  {status.health.data.device || '—'}
                </span>
              </p>
            </div>
            <div>
              <span className="text-xs text-[#64748b] uppercase">Features</span>
              <p className="text-sm font-medium text-[#f8fafc] mt-1">{status.health.data.num_features ?? '—'}</p>
            </div>
            <div>
              <span className="text-xs text-[#64748b] uppercase">Status</span>
              <p className="text-sm font-medium text-green-400 mt-1">{status.health.data.status || 'ok'}</p>
            </div>
          </div>
        )}

        {/* Configuration (when stopped) */}
        {!isRunning && (
          <div className="px-6 py-4">
            <h3 className="text-sm font-medium text-[#cbd5e1] mb-3">启动配置</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-[#64748b] mb-1">模型路径</label>
                <input
                  type="text"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  className="w-full rounded-lg bg-[#0f172a] border border-[#334155] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs text-[#64748b] mb-1">设备</label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="w-full rounded-lg bg-[#0f172a] border border-[#334155] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none"
                >
                  <option value="auto">Auto Detect</option>
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA GPU</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-[#64748b] mb-1">HTTP 端口</label>
                <input
                  type="number"
                  value={httpPort}
                  onChange={(e) => setHttpPort(Number(e.target.value))}
                  className="w-full rounded-lg bg-[#0f172a] border border-[#334155] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs text-[#64748b] mb-1">TCP MQ 端口</label>
                <input
                  type="number"
                  value={tcpPort}
                  onChange={(e) => setTcpPort(Number(e.target.value))}
                  className="w-full rounded-lg bg-[#0f172a] border border-[#334155] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Info card */}
      <div className="rounded-xl border border-[#334155] bg-[#1e293b] px-6 py-4">
        <h3 className="text-sm font-medium text-[#cbd5e1] mb-2">推理模式说明</h3>
        <div className="grid grid-cols-3 gap-4 text-xs text-[#94a3b8]">
          <div className="rounded-lg bg-[#0f172a] p-3">
            <span className="font-medium text-[#f8fafc]">Embedded (默认)</span>
            <p className="mt-1">LightGBM Rust 内嵌推理，~0.01ms，无需启动 ml_serve</p>
          </div>
          <div className="rounded-lg bg-[#0f172a] p-3">
            <span className="font-medium text-[#f8fafc]">ONNX</span>
            <p className="mt-1">ONNX Runtime 推理，~0.05ms，支持 GPU，需编译 --features onnx</p>
          </div>
          <div className="rounded-lg bg-[#0f172a] p-3 border border-[#3b82f6]/30">
            <span className="font-medium text-[#3b82f6]">TcpMq (ml_serve)</span>
            <p className="mt-1">TCP 协议连接 ml_serve.py，~0.3ms，支持 Ensemble/深度学习/热重载</p>
          </div>
        </div>
      </div>
    </div>
  );
}
