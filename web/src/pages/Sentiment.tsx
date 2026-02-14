import { useState, useEffect, useCallback } from 'react';
import {
  sentimentSubmit,
  sentimentQuery,
  sentimentSummary,
  type SentimentOverview,
  type SentimentQueryResult,
} from '../api/client';

function scoreColor(score: number): string {
  if (score > 0.3) return 'text-green-400';
  if (score > 0) return 'text-green-300';
  if (score > -0.3) return 'text-gray-400';
  if (score > -0.6) return 'text-red-300';
  return 'text-red-400';
}

function scoreBg(score: number): string {
  if (score > 0.3) return 'bg-green-900/30';
  if (score > 0) return 'bg-green-900/15';
  if (score > -0.3) return 'bg-gray-800';
  if (score > -0.6) return 'bg-red-900/15';
  return 'bg-red-900/30';
}

function ScoreBar({ score }: { score: number }) {
  const pct = ((score + 1) / 2) * 100;
  const barColor = score > 0.2 ? 'bg-green-500' : score < -0.2 ? 'bg-red-500' : 'bg-gray-500';
  return (
    <div className="w-full bg-gray-700 rounded h-2 relative">
      <div
        className={`h-2 rounded ${barColor}`}
        style={{ width: `${Math.max(2, pct)}%` }}
      />
      <div
        className="absolute top-0 w-0.5 h-2 bg-white/50"
        style={{ left: '50%' }}
      />
    </div>
  );
}

export default function Sentiment() {
  const [overview, setOverview] = useState<SentimentOverview | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [queryResult, setQueryResult] = useState<SentimentQueryResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Submit form state
  const [symbol, setSymbol] = useState('600519.SH');
  const [source, setSource] = useState('新闻');
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [score, setScore] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [submitMsg, setSubmitMsg] = useState('');

  const refreshOverview = useCallback(async () => {
    try {
      const data = await sentimentSummary();
      setOverview(data);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refreshOverview();
  }, [refreshOverview]);

  const handleQuery = async (sym: string) => {
    setSelectedSymbol(sym);
    setLoading(true);
    try {
      const data = await sentimentQuery(sym, 20);
      setQueryResult(data);
    } catch {
      setQueryResult(null);
    }
    setLoading(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol || !title) return;
    setSubmitting(true);
    setSubmitMsg('');
    try {
      await sentimentSubmit({
        symbol,
        source,
        title,
        content,
        sentiment_score: score,
      });
      setSubmitMsg('✅ 提交成功');
      setTitle('');
      setContent('');
      setScore(0);
      refreshOverview();
      if (selectedSymbol === symbol) handleQuery(symbol);
    } catch (err) {
      setSubmitMsg(`❌ 提交失败: ${err}`);
    }
    setSubmitting(false);
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">📰 舆情数据 Sentiment</h1>

      {/* Submit Form */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3">提交舆情数据</h2>
        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="block text-sm text-gray-400 mb-1">股票代码</label>
            <input
              type="text"
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
              placeholder="600519.SH"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">来源</label>
            <select
              value={source}
              onChange={e => setSource(e.target.value)}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
            >
              <option value="新闻">新闻</option>
              <option value="研报">研报</option>
              <option value="公告">公告</option>
              <option value="社交媒体">社交媒体</option>
              <option value="论坛">论坛</option>
              <option value="LLM分析">LLM分析</option>
              <option value="其他">其他</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              情感分数: <span className={scoreColor(score)}>{score.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.05"
              value={score}
              onChange={e => setScore(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-0.5">
              <span>看空 -1.0</span>
              <span>中性 0</span>
              <span>看多 +1.0</span>
            </div>
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm text-gray-400 mb-1">标题</label>
            <input
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
              placeholder="舆情标题..."
              required
            />
          </div>
          <div className="flex items-end">
            <button
              type="submit"
              disabled={submitting || !title}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded px-4 py-2 text-sm font-medium"
            >
              {submitting ? '提交中...' : '提交'}
            </button>
          </div>
          <div className="md:col-span-3">
            <label className="block text-sm text-gray-400 mb-1">内容 (可选)</label>
            <textarea
              value={content}
              onChange={e => setContent(e.target.value)}
              className="w-full bg-gray-700 rounded px-3 py-2 text-sm h-16 resize-none"
              placeholder="详细内容..."
            />
          </div>
          {submitMsg && (
            <div className="md:col-span-3 text-sm">{submitMsg}</div>
          )}
        </form>
      </div>

      {/* Overview */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold">舆情概览</h2>
          <div className="text-sm text-gray-400">
            共 {overview?.total_items ?? 0} 条数据，{overview?.symbols?.length ?? 0} 只股票
          </div>
        </div>

        {(!overview || overview.symbols.length === 0) ? (
          <div className="text-center text-gray-500 py-8">
            <p className="text-lg mb-2">暂无舆情数据</p>
            <p className="text-sm">通过上方表单提交，或使用 API 批量导入</p>
            <div className="mt-3 text-xs text-gray-600">
              <code>POST /api/sentiment/submit</code> 或 <code>POST /api/sentiment/batch</code>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="text-left py-2 px-2">股票</th>
                  <th className="text-center py-2 px-2">条数</th>
                  <th className="text-center py-2 px-2">情感分</th>
                  <th className="text-center py-2 px-2">情感方向</th>
                  <th className="text-center py-2 px-2">看多</th>
                  <th className="text-center py-2 px-2">看空</th>
                  <th className="text-center py-2 px-2">中性</th>
                  <th className="text-left py-2 px-2">最新标题</th>
                  <th className="text-center py-2 px-2">操作</th>
                </tr>
              </thead>
              <tbody>
                {overview.symbols.map(s => (
                  <tr
                    key={s.symbol}
                    className={`border-b border-gray-700/50 hover:bg-gray-700/30 cursor-pointer ${
                      selectedSymbol === s.symbol ? 'bg-gray-700/50' : ''
                    }`}
                    onClick={() => handleQuery(s.symbol)}
                  >
                    <td className="py-2 px-2 font-mono">{s.symbol}</td>
                    <td className="py-2 px-2 text-center">{s.count}</td>
                    <td className={`py-2 px-2 text-center font-bold ${scoreColor(s.avg_score)}`}>
                      {s.avg_score.toFixed(2)}
                    </td>
                    <td className="py-2 px-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs ${scoreBg(s.avg_score)} ${scoreColor(s.avg_score)}`}>
                        {s.level}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-center text-green-400">{s.bullish_count}</td>
                    <td className="py-2 px-2 text-center text-red-400">{s.bearish_count}</td>
                    <td className="py-2 px-2 text-center text-gray-400">{s.neutral_count}</td>
                    <td className="py-2 px-2 truncate max-w-[200px]">{s.latest_title}</td>
                    <td className="py-2 px-2 text-center">
                      <button
                        onClick={e => { e.stopPropagation(); handleQuery(s.symbol); }}
                        className="text-blue-400 hover:text-blue-300 text-xs"
                      >
                        详情
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail view */}
      {selectedSymbol && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-3">
            {selectedSymbol} 舆情详情
          </h2>

          {loading ? (
            <div className="text-center text-gray-500 py-4">加载中...</div>
          ) : queryResult ? (
            <>
              {/* Summary gauge */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                <div className={`rounded-lg p-3 ${scoreBg(queryResult.summary.avg_score)}`}>
                  <div className="text-xs text-gray-400 mb-1">综合情感</div>
                  <div className={`text-xl font-bold ${scoreColor(queryResult.summary.avg_score)}`}>
                    {queryResult.summary.avg_score.toFixed(2)}
                  </div>
                  <div className={`text-sm ${scoreColor(queryResult.summary.avg_score)}`}>
                    {queryResult.summary.level}
                  </div>
                  <ScoreBar score={queryResult.summary.avg_score} />
                </div>
                <div className="rounded-lg p-3 bg-gray-700/50">
                  <div className="text-xs text-gray-400 mb-1">数据量</div>
                  <div className="text-xl font-bold">{queryResult.summary.count}</div>
                  <div className="text-sm text-gray-400">条舆情</div>
                </div>
                <div className="rounded-lg p-3 bg-green-900/20">
                  <div className="text-xs text-gray-400 mb-1">看多</div>
                  <div className="text-xl font-bold text-green-400">
                    {queryResult.summary.bullish_count}
                  </div>
                  <div className="text-sm text-gray-500">
                    {queryResult.summary.count > 0
                      ? `${((queryResult.summary.bullish_count / queryResult.summary.count) * 100).toFixed(0)}%`
                      : '0%'}
                  </div>
                </div>
                <div className="rounded-lg p-3 bg-red-900/20">
                  <div className="text-xs text-gray-400 mb-1">看空</div>
                  <div className="text-xl font-bold text-red-400">
                    {queryResult.summary.bearish_count}
                  </div>
                  <div className="text-sm text-gray-500">
                    {queryResult.summary.count > 0
                      ? `${((queryResult.summary.bearish_count / queryResult.summary.count) * 100).toFixed(0)}%`
                      : '0%'}
                  </div>
                </div>
              </div>

              {/* Items table */}
              {queryResult.items.length > 0 && (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-700">
                      <th className="text-left py-2 px-2">时间</th>
                      <th className="text-left py-2 px-2">来源</th>
                      <th className="text-left py-2 px-2">标题</th>
                      <th className="text-center py-2 px-2">分数</th>
                      <th className="text-center py-2 px-2">方向</th>
                    </tr>
                  </thead>
                  <tbody>
                    {queryResult.items.map(item => (
                      <tr key={item.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                        <td className="py-2 px-2 text-xs text-gray-400 whitespace-nowrap">
                          {item.published_at}
                        </td>
                        <td className="py-2 px-2">
                          <span className="bg-gray-700 px-2 py-0.5 rounded text-xs">{item.source}</span>
                        </td>
                        <td className="py-2 px-2">{item.title}</td>
                        <td className={`py-2 px-2 text-center font-mono font-bold ${scoreColor(item.sentiment_score)}`}>
                          {item.sentiment_score > 0 ? '+' : ''}{item.sentiment_score.toFixed(2)}
                        </td>
                        <td className="py-2 px-2 text-center">
                          <span className={`px-2 py-0.5 rounded text-xs ${scoreBg(item.sentiment_score)} ${scoreColor(item.sentiment_score)}`}>
                            {item.level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          ) : (
            <div className="text-center text-gray-500 py-4">无数据</div>
          )}
        </div>
      )}

      {/* API guide */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-3">💡 API 接入指南</h2>
        <div className="text-sm text-gray-400 space-y-2">
          <p>舆情数据可从任何外部系统推送，支持以下接口：</p>
          <div className="bg-gray-900 rounded p-3 font-mono text-xs space-y-1">
            <div><span className="text-green-400">POST</span> /api/sentiment/submit — 提交单条舆情</div>
            <div><span className="text-green-400">POST</span> /api/sentiment/batch — 批量提交</div>
            <div><span className="text-blue-400">GET</span>  /api/sentiment/{'{'}<span className="text-yellow-300">symbol</span>{'}'} — 查询股票舆情</div>
            <div><span className="text-blue-400">GET</span>  /api/sentiment/summary — 全局概览</div>
          </div>
          <p className="mt-2">提交格式示例：</p>
          <pre className="bg-gray-900 rounded p-3 text-xs overflow-x-auto">{`{
  "symbol": "600519.SH",
  "source": "新闻",
  "title": "贵州茅台业绩超预期",
  "content": "2024年报显示营收增长15%...",
  "sentiment_score": 0.7,
  "published_at": "2024-03-15"
}`}</pre>
          <p className="mt-2">
            <span className="text-yellow-400">舆情增强策略</span>：在自动交易中选择「舆情增强策略」，
            系统会自动结合舆情数据调整交易信号强度——看多舆情增强买入信号，看空舆情增强卖出信号。
          </p>
        </div>
      </div>
    </div>
  );
}
