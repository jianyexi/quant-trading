import { useState, useEffect, useCallback } from 'react';
import { useTaskManager } from '../hooks/useTaskManager';
import {
  llmExportDataset,
  llmTrain,
  llmListModels,
  llmActivateModel,
  llmSignalServeStart,
  llmSignalServeStop,
  llmSignalServeStatus,
  type LlmModelsResponse,
} from '../api/client';

const inputCls = 'w-full rounded-lg border border-[#334155] bg-[#0f172a] px-3 py-2 text-sm text-[#f8fafc] focus:border-[#3b82f6] focus:outline-none';
const btnCls = 'px-4 py-2 rounded-lg text-sm font-medium transition-all';
const cardCls = 'rounded-xl border border-[#334155] bg-[#1e293b] p-5';

export default function LLMTraining() {
  const [data, setData] = useState<LlmModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  // Training config
  const [trainType, setTrainType] = useState<'sft' | 'dpo'>('sft');
  const [baseModel, setBaseModel] = useState('Qwen/Qwen2.5-7B-Instruct');
  const [loraRank, setLoraRank] = useState(16);
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [learningRate, setLearningRate] = useState(0.0002);
  const [beta, setBeta] = useState(0.1);

  // Task managers
  const tmExport = useTaskManager();
  const tmTrain = useTaskManager();

  // LLM Signal Server state
  const [signalServerStatus, setSignalServerStatus] = useState<{
    managed: boolean;
    process: string;
    pid?: number;
    uptime_secs?: number;
    reachable: boolean;
    model?: string;
    device?: string;
  } | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await llmListModels();
      setData(res);
    } catch { /* ignore */ }
    // Also refresh signal server status
    try {
      const ss = await llmSignalServeStatus();
      setSignalServerStatus({
        managed: ss.managed,
        process: ss.process_info.process,
        pid: ss.process_info.pid,
        uptime_secs: ss.process_info.uptime_secs,
        reachable: ss.health.reachable,
        model: (ss.health.data as Record<string, unknown>)?.base_model as string | undefined,
        device: (ss.health.data as Record<string, unknown>)?.device as string | undefined,
      });
    } catch {
      setSignalServerStatus(null);
    }
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Watch export completion
  useEffect(() => {
    if (tmExport.task?.status === 'Completed') refresh();
  }, [tmExport.task?.status, refresh]);

  // Watch train completion
  useEffect(() => {
    if (tmTrain.task?.status === 'Completed') refresh();
  }, [tmTrain.task?.status, refresh]);

  const handleExport = async () => {
    await tmExport.submit(() => llmExportDataset());
  };

  const handleTrain = async () => {
    await tmTrain.submit(() => llmTrain({
      train_type: trainType,
      base_model: baseModel,
      lora_rank: loraRank,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      beta: trainType === 'dpo' ? beta : undefined,
      sft_adapter: trainType === 'dpo' ? 'ml_models/llm_adapters/sft/adapter' : undefined,
    }));
  };

  const handleActivate = async (name: string) => {
    try {
      await llmActivateModel(name);
      refresh();
    } catch { /* ignore */ }
  };

  const handleSignalServerStart = async () => {
    try {
      await llmSignalServeStart({ base_model: baseModel });
      setTimeout(refresh, 3000); // Give the server time to load the model
    } catch { /* ignore */ }
  };

  const handleSignalServerStop = async () => {
    try {
      await llmSignalServeStop();
      refresh();
    } catch { /* ignore */ }
  };

  const ds = data?.dataset;
  const totalSamples = (ds?.sft_chat ?? 0) + (ds?.sft_sentiment ?? 0) + (ds?.dpo_trades ?? 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-[#f8fafc]">🧠 LLM Post-Training</h1>
        <button onClick={refresh} className={`${btnCls} bg-[#334155] text-[#94a3b8] hover:bg-[#475569]`}>
          🔄 刷新
        </button>
      </div>

      {/* Dataset Stats */}
      <div className={cardCls}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-[#f8fafc]">📊 训练数据集</h2>
          <button
            onClick={handleExport}
            disabled={tmExport.running}
            className={`${btnCls} ${tmExport.running ? 'bg-[#475569] text-[#64748b] cursor-not-allowed' : 'bg-[#3b82f6] text-white hover:bg-[#2563eb]'}`}
          >
            {tmExport.running ? '⏳ 导出中...' : '📤 导出数据集'}
          </button>
        </div>

        {loading ? (
          <div className="text-[#64748b] text-sm">加载中...</div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="SFT 对话" value={ds?.sft_chat ?? 0} icon="💬" />
            <StatCard label="SFT 舆情" value={ds?.sft_sentiment ?? 0} icon="📰" />
            <StatCard label="DPO 交易" value={ds?.dpo_trades ?? 0} icon="📈" />
            <StatCard label="总样本" value={totalSamples} icon="∑" highlight />
          </div>
        )}

        {tmExport.task && (
          <div className={`mt-3 text-xs p-2 rounded ${tmExport.task.status === 'Completed' ? 'bg-[#064e3b] text-[#6ee7b7]' : tmExport.task.status === 'Failed' ? 'bg-[#7f1d1d] text-[#fca5a5]' : 'bg-[#1e3a5f] text-[#93c5fd]'}`}>
            {tmExport.task.status}: {tmExport.output?.slice(0, 200) || tmExport.task.progress || '...'}
          </div>
        )}
      </div>

      {/* Training Config */}
      <div className={cardCls}>
        <h2 className="text-lg font-semibold text-[#f8fafc] mb-4">🎯 训练配置</h2>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="text-xs text-[#64748b] block mb-1">训练类型</label>
            <select className={inputCls} value={trainType} onChange={e => setTrainType(e.target.value as 'sft' | 'dpo')}>
              <option value="sft">SFT (指令微调)</option>
              <option value="dpo">DPO (偏好优化)</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-[#64748b] block mb-1">基座模型</label>
            <select className={inputCls} value={baseModel} onChange={e => setBaseModel(e.target.value)}>
              <option value="Qwen/Qwen2.5-7B-Instruct">Qwen2.5-7B-Instruct</option>
              <option value="Qwen/Qwen2.5-3B-Instruct">Qwen2.5-3B-Instruct</option>
              <option value="meta-llama/Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</option>
              <option value="mistralai/Mistral-7B-Instruct-v0.3">Mistral-7B-v0.3</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-[#64748b] block mb-1">LoRA Rank</label>
            <select className={inputCls} value={loraRank} onChange={e => setLoraRank(Number(e.target.value))}>
              <option value={8}>8 (轻量)</option>
              <option value={16}>16 (推荐)</option>
              <option value={32}>32 (高精度)</option>
              <option value={64}>64 (大容量)</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-[#64748b] block mb-1">训练轮次</label>
            <input type="number" className={inputCls} value={epochs} min={1} max={20}
              onChange={e => setEpochs(Number(e.target.value))} />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="text-xs text-[#64748b] block mb-1">Batch Size</label>
            <input type="number" className={inputCls} value={batchSize} min={1} max={32}
              onChange={e => setBatchSize(Number(e.target.value))} />
          </div>
          <div>
            <label className="text-xs text-[#64748b] block mb-1">学习率</label>
            <input type="number" className={inputCls} value={learningRate} step={0.00001}
              onChange={e => setLearningRate(Number(e.target.value))} />
          </div>
          {trainType === 'dpo' && (
            <div>
              <label className="text-xs text-[#64748b] block mb-1">DPO Beta</label>
              <input type="number" className={inputCls} value={beta} step={0.01} min={0.01} max={1.0}
                onChange={e => setBeta(Number(e.target.value))} />
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleTrain}
            disabled={tmTrain.running || totalSamples === 0}
            className={`${btnCls} ${tmTrain.running || totalSamples === 0 ? 'bg-[#475569] text-[#64748b] cursor-not-allowed' : 'bg-[#8b5cf6] text-white hover:bg-[#7c3aed]'}`}
          >
            {tmTrain.running ? '⏳ 训练中...' : `🚀 开始 ${trainType.toUpperCase()} 训练`}
          </button>
          {totalSamples === 0 && (
            <span className="text-xs text-[#f59e0b]">⚠️ 请先导出数据集</span>
          )}
        </div>

        {tmTrain.task && (
          <div className={`mt-3 text-xs p-2 rounded font-mono ${tmTrain.task.status === 'Completed' ? 'bg-[#064e3b] text-[#6ee7b7]' : tmTrain.task.status === 'Failed' ? 'bg-[#7f1d1d] text-[#fca5a5]' : 'bg-[#1e3a5f] text-[#93c5fd]'}`}>
            {tmTrain.task.status}: {tmTrain.output?.slice(0, 300) || tmTrain.task.progress || '训练进行中...'}
          </div>
        )}
      </div>

      {/* Models List */}
      <div className={cardCls}>
        <h2 className="text-lg font-semibold text-[#f8fafc] mb-4">🗃️ 可用适配器</h2>

        {!data?.models?.length ? (
          <div className="text-[#64748b] text-sm">暂无训练好的适配器。完成训练后将显示在此。</div>
        ) : (
          <div className="space-y-3">
            {data.models.map(m => (
              <div key={m.name} className="flex items-center justify-between p-3 rounded-lg bg-[#0f172a] border border-[#334155]">
                <div>
                  <div className="text-sm font-medium text-[#f8fafc]">
                    {m.name === 'sft' ? '🎓 SFT' : m.name === 'dpo' ? '🏆 DPO' : `📦 ${m.name}`}
                    <span className={`ml-2 text-xs px-2 py-0.5 rounded ${m.has_adapter ? 'bg-[#064e3b] text-[#6ee7b7]' : 'bg-[#7f1d1d] text-[#fca5a5]'}`}>
                      {m.has_adapter ? '✓ 就绪' : '✗ 未完成'}
                    </span>
                  </div>
                  {m.report && (
                    <div className="text-xs text-[#64748b] mt-1">
                      {(m.report as Record<string, unknown>).base_model as string} · LoRA r={String((m.report as Record<string, unknown>).lora_rank)} · Loss: {Number((m.report as Record<string, unknown>).train_loss).toFixed(4)} · {Number((m.report as Record<string, unknown>).elapsed_seconds)}s
                    </div>
                  )}
                </div>
                {m.has_adapter && (
                  <button
                    onClick={() => handleActivate(m.name)}
                    className={`${btnCls} bg-[#059669] text-white hover:bg-[#047857] text-xs`}
                  >
                    激活
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* LLM Signal Server */}
      <div className={cardCls}>
        <h2 className="text-lg font-semibold text-[#f8fafc] mb-3">🤖 LLM 信号服务</h2>
        <p className="text-xs text-[#64748b] mb-4">
          启动本地 LLM 信号服务后，可在回测和实盘中使用 "LLM信号" 策略。模型将分析行情数据并生成买入/卖出/持有信号。
        </p>
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <span className={`inline-block w-2.5 h-2.5 rounded-full ${
              signalServerStatus?.reachable ? 'bg-green-500' :
              signalServerStatus?.process === 'running' ? 'bg-yellow-500 animate-pulse' :
              'bg-red-500'
            }`} />
            <span className="text-sm text-[#94a3b8]">
              {signalServerStatus?.reachable ? '运行中' :
               signalServerStatus?.process === 'running' ? '启动中...' :
               '未运行'}
            </span>
          </div>
          {signalServerStatus?.pid && (
            <span className="text-xs text-[#64748b]">PID: {signalServerStatus.pid}</span>
          )}
          {signalServerStatus?.uptime_secs != null && signalServerStatus.uptime_secs > 0 && (
            <span className="text-xs text-[#64748b]">
              运行 {Math.floor(signalServerStatus.uptime_secs / 60)}分钟
            </span>
          )}
          {signalServerStatus?.model && (
            <span className="text-xs text-[#64748b]">模型: {signalServerStatus.model}</span>
          )}
          {signalServerStatus?.device && (
            <span className="text-xs text-[#64748b]">设备: {signalServerStatus.device}</span>
          )}
        </div>
        <div className="flex gap-3">
          {signalServerStatus?.process !== 'running' ? (
            <button onClick={handleSignalServerStart}
              className={`${btnCls} bg-[#3b82f6] text-white hover:bg-[#2563eb]`}>
              🚀 启动信号服务
            </button>
          ) : (
            <button onClick={handleSignalServerStop}
              className={`${btnCls} bg-[#dc2626] text-white hover:bg-[#b91c1c]`}>
              ⏹ 停止服务
            </button>
          )}
          <button onClick={refresh}
            className={`${btnCls} bg-[#334155] text-[#94a3b8] hover:bg-[#475569]`}>
            🔄 刷新状态
          </button>
        </div>
      </div>

      {/* How it works */}
      <div className={cardCls}>
        <h2 className="text-lg font-semibold text-[#f8fafc] mb-3">ℹ️ 工作原理</h2>
        <div className="text-xs text-[#94a3b8] space-y-2">
          <p><strong>1. 导出数据集</strong> — 从聊天记录(SFT)、舆情分析(SFT)、交易日志(DPO)中提取训练数据</p>
          <p><strong>2. SFT 微调</strong> — 使用 LoRA 在对话和舆情数据上进行指令微调，让模型更懂A股</p>
          <p><strong>3. DPO 对齐</strong> — 用盈利/亏损交易作为偏好对，让模型的交易建议更倾向盈利策略</p>
          <p><strong>4. 激活使用</strong> — 将训练好的 LoRA 适配器加载到推理服务中</p>
          <p><strong>5. 信号服务</strong> — 启动 LLM 信号服务后，在回测/实盘中选择 "LLM信号" 策略即可使用微调模型生成交易信号</p>
          <p className="text-[#64748b] mt-2">推荐流程：导出 → SFT → DPO → 激活 → 启动信号服务 → 使用 LLM信号 策略。需要 GPU (≥16GB VRAM)。</p>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon, highlight }: { label: string; value: number; icon: string; highlight?: boolean }) {
  return (
    <div className={`rounded-lg p-3 ${highlight ? 'bg-[#3b82f6]/10 border border-[#3b82f6]/30' : 'bg-[#0f172a] border border-[#334155]'}`}>
      <div className="flex items-center gap-2 mb-1">
        <span>{icon}</span>
        <span className="text-xs text-[#64748b]">{label}</span>
      </div>
      <div className={`text-xl font-bold ${highlight ? 'text-[#3b82f6]' : 'text-[#f8fafc]'}`}>
        {value.toLocaleString()}
      </div>
    </div>
  );
}
