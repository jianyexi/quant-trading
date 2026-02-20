import { useState } from 'react';
import { Search, Filter, TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp, Loader2, AlertTriangle } from 'lucide-react';
import { screenScan, screenFactors, type StockCandidate, type ScreenerResult, type VoteResult } from '../api/client';

function formatVote(vote: VoteResult): { text: string; color: string } {
  if (vote && typeof vote === 'object') {
    if ('Buy' in vote && vote.Buy !== undefined) return { text: `买入(${(vote.Buy as number).toFixed(3)})`, color: 'text-green-400' };
    if ('Sell' in vote && vote.Sell !== undefined) return { text: `卖出(${(vote.Sell as number).toFixed(3)})`, color: 'text-red-400' };
  }
  return { text: '中性', color: 'text-gray-400' };
}

function recColor(rec: string) {
  if (rec === '强烈推荐') return 'bg-green-500/20 text-green-400 border-green-500/30';
  if (rec === '推荐') return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
  if (rec === '观望') return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
  return 'bg-red-500/20 text-red-400 border-red-500/30';
}

export default function Screener() {
  const [result, setResult] = useState<ScreenerResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [topN, setTopN] = useState(10);
  const [minVotes, setMinVotes] = useState(1);
  const [pool, setPool] = useState('custom');
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [factorSymbol, setFactorSymbol] = useState('');
  const [factorData, setFactorData] = useState<StockCandidate | null>(null);
  const [factorLoading, setFactorLoading] = useState(false);

  const handleScan = async () => {
    setLoading(true);
    setError('');
    setExpandedIdx(null);
    try {
      const data = await screenScan(topN, minVotes, pool);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Scan failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFactorLookup = async () => {
    if (!factorSymbol.trim()) return;
    setFactorLoading(true);
    try {
      const data = await screenFactors(factorSymbol.trim());
      setFactorData(data);
    } catch (e) {
      setFactorData(null);
      setError(e instanceof Error ? e.message : 'Factor lookup failed');
    } finally {
      setFactorLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-[#f8fafc]">🔍 智能选股</h1>
        <p className="text-sm text-[#94a3b8] mt-1">多因子评分 + 策略信号聚合 + LLM分析</p>
      </div>

      {/* Controls */}
      <div className="bg-[#1e293b] rounded-xl p-5 border border-[#334155]">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="block text-xs text-[#94a3b8] mb-1">股票池</label>
            <select value={pool} onChange={e => setPool(e.target.value)}
              className="bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc]">
              <option value="custom">自选 (~40)</option>
              <option value="csi300">沪深300</option>
              <option value="csi500">中证500</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-[#94a3b8] mb-1">推荐数量</label>
            <input type="number" min={1} max={30} value={topN}
              onChange={e => setTopN(Number(e.target.value))}
              className="w-20 bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc]" />
          </div>
          <div>
            <label className="block text-xs text-[#94a3b8] mb-1">最低策略共识</label>
            <select value={minVotes} onChange={e => setMinVotes(Number(e.target.value))}
              className="bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc]">
              <option value={1}>≥1 策略买入</option>
              <option value={2}>≥2 策略买入</option>
              <option value={3}>3 策略一致</option>
            </select>
          </div>
          <button onClick={handleScan} disabled={loading}
            className="flex items-center gap-2 bg-[#3b82f6] hover:bg-[#2563eb] disabled:bg-[#334155] text-white font-medium rounded-lg px-5 py-2 text-sm transition-colors">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            {loading ? '扫描中...' : '全市场扫描'}
          </button>

          <div className="ml-auto flex items-end gap-2">
            <div>
              <label className="block text-xs text-[#94a3b8] mb-1">单股因子查询</label>
              <input type="text" value={factorSymbol} onChange={e => setFactorSymbol(e.target.value)}
                placeholder="如 600519.SH"
                onKeyDown={e => e.key === 'Enter' && handleFactorLookup()}
                className="w-36 bg-[#0f172a] border border-[#334155] rounded-lg px-3 py-2 text-sm text-[#f8fafc]" />
            </div>
            <button onClick={handleFactorLookup} disabled={factorLoading}
              className="flex items-center gap-1.5 bg-[#334155] hover:bg-[#475569] text-[#f8fafc] rounded-lg px-4 py-2 text-sm transition-colors">
              <Filter className="h-4 w-4" />
              查询
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Scan Stats */}
      {result && (
        <div className="space-y-3">
          {result.regime && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-[#94a3b8]">市场状态:</span>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                result.regime === 'Trending' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                result.regime === 'Volatile' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                'bg-blue-500/20 text-blue-400 border border-blue-500/30'
              }`}>
                {result.regime === 'Trending' ? '📈 趋势行情' : result.regime === 'Volatile' ? '⚡ 高波动' : '🔄 震荡行情'}
              </span>
            </div>
          )}
          <div className="grid grid-cols-4 gap-4">
            <StatCard label="扫描股票" value={result.total_scanned} />
            <StatCard label="因子通过" value={result.phase1_passed} />
            <StatCard label="策略通过" value={result.phase2_passed} />
            <StatCard label="最终推荐" value={result.candidates.length} />
          </div>
        </div>
      )}

      {/* Results Table */}
      {result && result.candidates.length > 0 && (
        <div className="bg-[#1e293b] rounded-xl border border-[#334155] overflow-hidden">
          <div className="px-5 py-3 border-b border-[#334155]">
            <h2 className="text-lg font-semibold text-[#f8fafc]">🏆 推荐股票</h2>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#334155] text-[#94a3b8]">
                <th className="px-4 py-3 text-left font-medium">排名</th>
                <th className="px-4 py-3 text-left font-medium">代码</th>
                <th className="px-4 py-3 text-left font-medium">名称</th>
                <th className="px-4 py-3 text-left font-medium">行业</th>
                <th className="px-4 py-3 text-right font-medium">价格</th>
                <th className="px-4 py-3 text-right font-medium">因子分</th>
                <th className="px-4 py-3 text-right font-medium">综合分</th>
                <th className="px-4 py-3 text-center font-medium">投票</th>
                <th className="px-4 py-3 text-right font-medium">成交额</th>
                <th className="px-4 py-3 text-right font-medium">RSI</th>
                <th className="px-4 py-3 text-center font-medium">推荐</th>
                <th className="px-4 py-3 text-center font-medium">详情</th>
              </tr>
            </thead>
            <tbody>
              {result.candidates.map((c, i) => (
                <>
                  <tr key={c.symbol} className="border-b border-[#334155]/50 hover:bg-[#334155]/30 transition-colors">
                    <td className="px-4 py-3 text-[#f8fafc] font-bold">{i + 1}</td>
                    <td className="px-4 py-3 text-[#3b82f6] font-mono">{c.symbol}</td>
                    <td className="px-4 py-3 text-[#f8fafc]">{c.name}</td>
                    <td className="px-4 py-3 text-[#94a3b8] text-sm">{c.sector}</td>
                    <td className="px-4 py-3 text-right text-[#f8fafc]">¥{c.price.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right text-[#f8fafc]">{c.factor_score.toFixed(1)}</td>
                    <td className="px-4 py-3 text-right font-semibold text-[#f8fafc]">{c.composite_score.toFixed(1)}</td>
                    <td className="px-4 py-3 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        c.strategy_vote.consensus_count >= 3 ? 'bg-green-500/20 text-green-400' :
                        c.strategy_vote.consensus_count >= 2 ? 'bg-blue-500/20 text-blue-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {c.strategy_vote.consensus_count}/3
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right text-[#94a3b8] text-sm">
                      {(c.avg_turnover / 100000000).toFixed(1)}亿
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className={c.factors.rsi_14 < 30 ? 'text-green-400' : c.factors.rsi_14 > 70 ? 'text-red-400' : 'text-[#f8fafc]'}>
                        {c.factors.rsi_14.toFixed(1)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`px-2 py-0.5 rounded border text-xs font-medium ${recColor(c.recommendation)}`}>
                        {c.recommendation}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                        className="text-[#94a3b8] hover:text-[#f8fafc] transition-colors">
                        {expandedIdx === i ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                      </button>
                    </td>
                  </tr>
                  {expandedIdx === i && (
                    <tr key={`${c.symbol}-detail`}>
                      <td colSpan={12} className="px-4 py-4 bg-[#0f172a]/50">
                        <CandidateDetail candidate={c} />
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {result && result.candidates.length === 0 && (
        <div className="bg-[#1e293b] rounded-xl p-8 border border-[#334155] text-center text-[#94a3b8]">
          没有找到满足条件的股票，请降低策略共识要求。
        </div>
      )}

      {/* Factor Detail Panel */}
      {factorData && (
        <div className="bg-[#1e293b] rounded-xl border border-[#334155] overflow-hidden">
          <div className="px-5 py-3 border-b border-[#334155] flex items-center justify-between">
            <h2 className="text-lg font-semibold text-[#f8fafc]">
              📊 {factorData.symbol} {factorData.name} 因子分析
            </h2>
            <button onClick={() => setFactorData(null)} className="text-[#94a3b8] hover:text-[#f8fafc] text-sm">关闭</button>
          </div>
          <div className="p-5">
            <CandidateDetail candidate={factorData} />
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-[#1e293b] rounded-xl p-4 border border-[#334155]">
      <div className="text-xs text-[#94a3b8]">{label}</div>
      <div className="text-2xl font-bold text-[#f8fafc] mt-1">{value}</div>
    </div>
  );
}

function CandidateDetail({ candidate: c }: { candidate: StockCandidate }) {
  return (
    <div className="grid grid-cols-4 gap-6">
      {/* Factors */}
      <div>
        <h3 className="text-sm font-semibold text-[#f8fafc] mb-3 flex items-center gap-1.5">
          <TrendingUp className="h-4 w-4 text-[#3b82f6]" /> 技术因子
        </h3>
        <div className="space-y-2 text-sm">
          <FactorRow label="5日涨幅" value={`${(c.factors.momentum_5d * 100).toFixed(2)}%`}
            color={c.factors.momentum_5d > 0 ? 'text-green-400' : 'text-red-400'} />
          <FactorRow label="20日涨幅" value={`${(c.factors.momentum_20d * 100).toFixed(2)}%`}
            color={c.factors.momentum_20d > 0 ? 'text-green-400' : 'text-red-400'} />
          <FactorRow label="RSI(14)" value={c.factors.rsi_14.toFixed(1)}
            color={c.factors.rsi_14 < 30 ? 'text-green-400' : c.factors.rsi_14 > 70 ? 'text-red-400' : 'text-[#f8fafc]'} />
          <FactorRow label="MACD柱" value={c.factors.macd_histogram.toFixed(4)}
            color={c.factors.macd_histogram > 0 ? 'text-green-400' : 'text-red-400'} />
          <FactorRow label="布林位置" value={c.factors.bollinger_position.toFixed(2)} color="text-[#f8fafc]" />
          <FactorRow label="MA趋势" value={c.factors.ma_trend.toFixed(4)}
            color={c.factors.ma_trend > 0 ? 'text-green-400' : 'text-red-400'} />
          <FactorRow label="KDJ J值" value={c.factors.kdj_j.toFixed(1)} color="text-[#f8fafc]" />
          <FactorRow label="成交量比" value={`${c.factors.volume_ratio.toFixed(2)}x`} color="text-[#f8fafc]" />
          <FactorRow label="波动率" value={`${(c.factors.volatility_20d * 100).toFixed(1)}%`} color="text-[#f8fafc]" />
        </div>
      </div>

      {/* Strategy Votes */}
      <div>
        <h3 className="text-sm font-semibold text-[#f8fafc] mb-3 flex items-center gap-1.5">
          <Filter className="h-4 w-4 text-[#8b5cf6]" /> 策略投票
        </h3>
        <div className="space-y-3">
          <VoteRow label="SMA交叉(5/20)" vote={c.strategy_vote.sma_cross} />
          <VoteRow label="RSI反转(14)" vote={c.strategy_vote.rsi_reversal} />
          <VoteRow label="MACD动量(12/26)" vote={c.strategy_vote.macd_trend} />
          <div className="pt-2 border-t border-[#334155]">
            <div className="flex justify-between">
              <span className="text-[#94a3b8]">策略共识</span>
              <span className={`font-semibold ${
                c.strategy_vote.consensus_count >= 3 ? 'text-green-400' :
                c.strategy_vote.consensus_count >= 2 ? 'text-blue-400' : 'text-gray-400'
              }`}>
                {c.strategy_vote.consensus_count}/3
              </span>
            </div>
            <div className="flex justify-between mt-1">
              <span className="text-[#94a3b8]">平均置信度</span>
              <span className="text-[#f8fafc]">{(c.strategy_vote.avg_confidence * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Metrics */}
      <div>
        <h3 className="text-sm font-semibold text-[#f8fafc] mb-3 flex items-center gap-1.5">
          <AlertTriangle className="h-4 w-4 text-[#f59e0b]" /> 风险指标
        </h3>
        <div className="space-y-2 text-sm">
          <FactorRow label="20日最大回撤" value={`${(c.risk.max_drawdown_20d * 100).toFixed(1)}%`}
            color={c.risk.max_drawdown_20d > 0.15 ? 'text-red-400' : 'text-[#f8fafc]'} />
          <FactorRow label="连续下跌天数" value={`${c.risk.consecutive_down_days}`}
            color={c.risk.consecutive_down_days > 5 ? 'text-red-400' : 'text-[#f8fafc]'} />
          <FactorRow label="距20日高点" value={`${(c.risk.distance_from_high_20d * 100).toFixed(1)}%`}
            color="text-[#f8fafc]" />
          <FactorRow label="ATR比率" value={`${(c.risk.atr_ratio * 100).toFixed(2)}%`}
            color="text-[#f8fafc]" />
          <div className="pt-2 border-t border-[#334155]">
            <FactorRow label="行业" value={c.sector} color="text-[#94a3b8]" />
            <FactorRow label="日均成交额" value={`${(c.avg_turnover / 100000000).toFixed(2)}亿`} color="text-[#94a3b8]" />
          </div>
        </div>
      </div>

      {/* Scores & Recommendation */}
      <div>
        <h3 className="text-sm font-semibold text-[#f8fafc] mb-3 flex items-center gap-1.5">
          <TrendingDown className="h-4 w-4 text-[#f59e0b]" /> 评分与建议
        </h3>
        <div className="space-y-3">
          <div className="bg-[#0f172a] rounded-lg p-3">
            <div className="text-xs text-[#94a3b8]">因子评分</div>
            <div className="text-xl font-bold text-[#f8fafc]">{c.factor_score.toFixed(1)} <span className="text-sm font-normal text-[#94a3b8]">/ 100</span></div>
            <div className="w-full bg-[#334155] rounded-full h-1.5 mt-2">
              <div className="bg-[#3b82f6] h-1.5 rounded-full" style={{ width: `${Math.min(c.factor_score, 100)}%` }} />
            </div>
          </div>
          <div className="bg-[#0f172a] rounded-lg p-3">
            <div className="text-xs text-[#94a3b8]">综合评分</div>
            <div className="text-xl font-bold text-[#f8fafc]">{c.composite_score.toFixed(1)}</div>
            <div className="w-full bg-[#334155] rounded-full h-1.5 mt-2">
              <div className="bg-[#8b5cf6] h-1.5 rounded-full" style={{ width: `${Math.min(c.composite_score, 100)}%` }} />
            </div>
          </div>
          <div className={`rounded-lg p-3 border ${recColor(c.recommendation)}`}>
            <div className="text-xs opacity-70">推荐</div>
            <div className="text-lg font-bold">{c.recommendation}</div>
          </div>
          {c.reasons.length > 0 && (
            <div className="space-y-1.5 mt-2">
              <div className="text-xs text-[#94a3b8] font-medium">推荐理由</div>
              {c.reasons.map((r, i) => (
                <div key={i} className="text-xs text-[#cbd5e1] flex items-start gap-1.5">
                  <span className="text-[#3b82f6] mt-0.5">✦</span> {r}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FactorRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-[#94a3b8]">{label}</span>
      <span className={`font-mono ${color}`}>{value}</span>
    </div>
  );
}

function VoteRow({ label, vote }: { label: string; vote: VoteResult }) {
  const { text, color } = formatVote(vote);
  const Icon = color.includes('green') ? TrendingUp : color.includes('red') ? TrendingDown : Minus;
  return (
    <div className="flex items-center justify-between">
      <span className="text-[#94a3b8] text-sm">{label}</span>
      <span className={`flex items-center gap-1 text-sm ${color}`}>
        <Icon className="h-3.5 w-3.5" /> {text}
      </span>
    </div>
  );
}
