//! Stock Screener — three-phase screening pipeline:
//!   Phase 1: Multi-factor scoring (momentum, volatility, trend, volume)
//!   Phase 2: Strategy signal aggregation (vote across built-in strategies)
//!   Phase 3: LLM-enhanced analysis (optional, requires API key)

use std::collections::HashMap;

use quant_core::models::Kline;
use quant_core::traits::Strategy;
use serde::Serialize;

use crate::builtin::{DualMaCrossover, MacdMomentum, RsiMeanReversion};
use crate::indicators::{BollingerBands, KDJ, MACD, RSI, SMA};

// ── Types ───────────────────────────────────────────────────────────

/// Per-stock factor scores from Phase 1
#[derive(Debug, Clone, Serialize)]
pub struct FactorScores {
    pub momentum_5d: f64,
    pub momentum_20d: f64,
    pub rsi_14: f64,
    pub macd_histogram: f64,
    pub bollinger_position: f64, // 0=lower band, 0.5=middle, 1=upper band
    pub volume_ratio: f64,      // current vol / 20-day avg vol
    pub ma_trend: f64,          // (MA5 - MA20) / MA20
    pub kdj_j: f64,
    pub volatility_20d: f64,    // annualized
}

/// Strategy voting result from Phase 2
#[derive(Debug, Clone, Serialize)]
pub struct StrategyVote {
    pub sma_cross: VoteResult,
    pub rsi_reversal: VoteResult,
    pub macd_trend: VoteResult,
    pub consensus_count: u32,  // how many strategies agree on BUY
    pub avg_confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub enum VoteResult {
    Buy(f64),  // confidence
    Sell(f64),
    Neutral,
}

/// A screened stock candidate
#[derive(Debug, Clone, Serialize)]
pub struct StockCandidate {
    pub symbol: String,
    pub name: String,
    pub price: f64,
    pub factor_score: f64,      // Phase 1 composite score (0-100)
    pub factors: FactorScores,
    pub strategy_vote: StrategyVote, // Phase 2
    pub composite_score: f64,   // Final weighted score
    pub recommendation: String, // "强烈推荐" / "推荐" / "观望" / "回避"
    pub reasons: Vec<String>,
}

/// Full screening result
#[derive(Debug, Clone, Serialize)]
pub struct ScreenerResult {
    pub candidates: Vec<StockCandidate>,
    pub total_scanned: usize,
    pub phase1_passed: usize,
    pub phase2_passed: usize,
}

/// Screening configuration
#[derive(Debug, Clone)]
pub struct ScreenerConfig {
    pub top_n: usize,              // final number of recommendations
    pub phase1_cutoff: usize,      // how many pass Phase 1 to Phase 2
    pub factor_weights: FactorWeights,
    pub min_consensus: u32,        // min strategy votes for Phase 2 pass
}

#[derive(Debug, Clone)]
pub struct FactorWeights {
    pub momentum: f64,
    pub trend: f64,
    pub mean_reversion: f64,
    pub volume: f64,
    pub volatility: f64,
}

impl Default for ScreenerConfig {
    fn default() -> Self {
        Self {
            top_n: 10,
            phase1_cutoff: 30,
            min_consensus: 2,
            factor_weights: FactorWeights {
                momentum: 0.25,
                trend: 0.30,
                mean_reversion: 0.20,
                volume: 0.15,
                volatility: 0.10,
            },
        }
    }
}

// ── Screener ────────────────────────────────────────────────────────

pub struct StockScreener {
    config: ScreenerConfig,
}

impl StockScreener {
    pub fn new(config: ScreenerConfig) -> Self {
        Self { config }
    }

    /// Run full 3-phase screening pipeline.
    /// Input: map of symbol -> (name, kline_data)
    /// Returns scored and ranked candidates.
    pub fn screen(
        &self,
        stock_data: &HashMap<String, (String, Vec<Kline>)>,
    ) -> ScreenerResult {
        let total_scanned = stock_data.len();

        // Phase 1: Multi-factor scoring
        let mut scored: Vec<(String, String, f64, FactorScores, f64)> = stock_data
            .iter()
            .filter_map(|(symbol, (name, klines))| {
                if klines.len() < 30 {
                    return None; // need at least 30 bars
                }
                let factors = self.compute_factors(klines);
                let score = self.composite_factor_score(&factors);
                let price = klines.last().unwrap().close;
                Some((symbol.clone(), name.clone(), price, factors, score))
            })
            .collect();

        // Sort by factor score descending
        scored.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.phase1_cutoff);
        let phase1_passed = scored.len();

        // Phase 2: Strategy signal aggregation
        let mut candidates: Vec<StockCandidate> = scored
            .into_iter()
            .filter_map(|(symbol, name, price, factors, factor_score)| {
                let klines = &stock_data[&symbol].1;
                let vote = self.run_strategy_votes(klines);

                // Filter: need >= min_consensus strategies agreeing BUY
                if vote.consensus_count < self.config.min_consensus {
                    return None;
                }

                // Composite: 60% factor score + 40% strategy confidence
                let composite = factor_score * 0.6 + vote.avg_confidence * 100.0 * 0.4;

                let (recommendation, reasons) =
                    self.generate_recommendation(&factors, &vote, composite);

                Some(StockCandidate {
                    symbol,
                    name,
                    price,
                    factor_score,
                    factors,
                    strategy_vote: vote,
                    composite_score: composite,
                    recommendation,
                    reasons,
                })
            })
            .collect();

        let phase2_passed = candidates.len();

        // Sort by composite score descending, take top N
        candidates
            .sort_by(|a, b| b.composite_score.partial_cmp(&a.composite_score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.top_n);

        ScreenerResult {
            candidates,
            total_scanned,
            phase1_passed,
            phase2_passed,
        }
    }

    /// Phase 1: Compute all technical factors from kline data
    fn compute_factors(&self, klines: &[Kline]) -> FactorScores {
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let n = closes.len();

        // Momentum: 5-day and 20-day return
        let momentum_5d = if n >= 6 {
            (closes[n - 1] - closes[n - 6]) / closes[n - 6]
        } else {
            0.0
        };
        let momentum_20d = if n >= 21 {
            (closes[n - 1] - closes[n - 21]) / closes[n - 21]
        } else {
            0.0
        };

        // RSI(14)
        let mut rsi_ind = RSI::new(14);
        let mut rsi_val = 50.0;
        for &c in &closes {
            if let Some(v) = rsi_ind.update(c) {
                rsi_val = v;
            }
        }

        // MACD histogram
        let mut macd_ind = MACD::new(12, 26, 9);
        for &c in &closes {
            macd_ind.update(c);
        }
        let macd_histogram = macd_ind.histogram().unwrap_or(0.0);

        // Bollinger Bands position
        let mut bb = BollingerBands::new(20, 2.0);
        for &c in &closes {
            bb.update(c);
        }
        let bollinger_position = match (bb.upper(), bb.lower()) {
            (Some(upper), Some(lower)) if upper > lower => {
                (closes[n - 1] - lower) / (upper - lower)
            }
            _ => 0.5,
        };

        // Volume ratio: last 5-day avg / 20-day avg
        let vol_5 = if n >= 5 {
            volumes[n - 5..].iter().sum::<f64>() / 5.0
        } else {
            volumes.iter().sum::<f64>() / n as f64
        };
        let vol_20 = if n >= 20 {
            volumes[n - 20..].iter().sum::<f64>() / 20.0
        } else {
            volumes.iter().sum::<f64>() / n as f64
        };
        let volume_ratio = if vol_20 > 0.0 { vol_5 / vol_20 } else { 1.0 };

        // MA trend: (MA5 - MA20) / MA20
        let mut sma5 = SMA::new(5);
        let mut sma20 = SMA::new(20);
        let mut last_sma5 = None;
        let mut last_sma20 = None;
        for &c in &closes {
            last_sma5 = sma5.update(c);
            last_sma20 = sma20.update(c);
        }
        let ma_trend = match (last_sma5, last_sma20) {
            (Some(f), Some(s)) if s > 0.0 => (f - s) / s,
            _ => 0.0,
        };

        // KDJ
        let mut kdj = KDJ::new(9, 3, 3);
        for k in klines {
            kdj.update(k.high, k.low, k.close);
        }
        let kdj_j = kdj.j().unwrap_or(50.0);

        // 20-day annualized volatility
        let volatility_20d = if n >= 21 {
            let daily_returns: Vec<f64> = closes
                .windows(2)
                .skip(n - 21)
                .map(|w| (w[1] / w[0]).ln())
                .collect();
            let mean = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let var = daily_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (daily_returns.len() - 1) as f64;
            var.sqrt() * (252.0_f64).sqrt()
        } else {
            0.2 // default 20%
        };

        FactorScores {
            momentum_5d,
            momentum_20d,
            rsi_14: rsi_val,
            macd_histogram,
            bollinger_position,
            volume_ratio,
            ma_trend,
            kdj_j,
            volatility_20d,
        }
    }

    /// Composite factor score (0-100 scale)
    fn composite_factor_score(&self, f: &FactorScores) -> f64 {
        let w = &self.config.factor_weights;

        // Momentum score: positive momentum is good (0-100)
        let mom_score = normalize_score(f.momentum_5d * 0.4 + f.momentum_20d * 0.6, -0.10, 0.15);

        // Trend score: MA5 > MA20 and MACD positive (0-100)
        let trend_ma = normalize_score(f.ma_trend, -0.05, 0.05);
        let trend_macd = normalize_score(f.macd_histogram, -2.0, 2.0);
        let trend_score = trend_ma * 0.6 + trend_macd * 0.4;

        // Mean reversion: RSI in sweet spot (40-60 = high score, extremes = medium)
        let mr_rsi = if f.rsi_14 < 30.0 {
            80.0 // oversold — good entry
        } else if f.rsi_14 > 70.0 {
            20.0 // overbought — risky
        } else {
            50.0 + (50.0 - f.rsi_14).abs() // moderate area
        };
        let mr_bb = if f.bollinger_position < 0.2 {
            80.0 // near lower band
        } else if f.bollinger_position > 0.8 {
            30.0
        } else {
            50.0
        };
        let mr_score = mr_rsi * 0.6 + mr_bb * 0.4;

        // Volume score: moderate increase is good
        let vol_score = if f.volume_ratio > 1.5 && f.volume_ratio < 3.0 {
            85.0 // healthy volume increase
        } else if f.volume_ratio > 1.0 {
            70.0
        } else if f.volume_ratio > 0.5 {
            50.0
        } else {
            30.0 // very low volume
        };

        // Volatility score: moderate is best (inverse U)
        let vol_pct = f.volatility_20d;
        let volatility_score = if vol_pct < 0.15 {
            60.0 // too quiet
        } else if vol_pct < 0.30 {
            85.0 // sweet spot
        } else if vol_pct < 0.50 {
            65.0 // getting risky
        } else {
            40.0 // too volatile
        };

        w.momentum * mom_score
            + w.trend * trend_score
            + w.mean_reversion * mr_score
            + w.volume * vol_score
            + w.volatility * volatility_score
    }

    /// Phase 2: Run all strategies and tally votes
    fn run_strategy_votes(&self, klines: &[Kline]) -> StrategyVote {
        let sma_vote = self.run_single_strategy(&mut DualMaCrossover::new(5, 20), klines);
        let rsi_vote = self.run_single_strategy(&mut RsiMeanReversion::new(14, 70.0, 30.0), klines);
        let macd_vote = self.run_single_strategy(&mut MacdMomentum::new(12, 26, 9), klines);

        let mut consensus = 0u32;
        let mut total_conf = 0.0f64;
        let mut conf_count = 0u32;

        for v in [&sma_vote, &rsi_vote, &macd_vote] {
            if let VoteResult::Buy(c) = v {
                consensus += 1;
                total_conf += c;
                conf_count += 1;
            }
        }

        let avg_confidence = if conf_count > 0 {
            total_conf / conf_count as f64
        } else {
            0.0
        };

        StrategyVote {
            sma_cross: sma_vote,
            rsi_reversal: rsi_vote,
            macd_trend: macd_vote,
            consensus_count: consensus,
            avg_confidence,
        }
    }

    /// Run a single strategy across all klines, return latest signal direction
    fn run_single_strategy(&self, strategy: &mut dyn Strategy, klines: &[Kline]) -> VoteResult {
        strategy.on_init();
        let mut last_signal = VoteResult::Neutral;

        // Only look at the last few signals (last 5 bars)
        let start_check = if klines.len() > 5 { klines.len() - 5 } else { 0 };

        for (i, kline) in klines.iter().enumerate() {
            if let Some(signal) = strategy.on_bar(kline) {
                if i >= start_check {
                    match signal.action {
                        quant_core::types::SignalAction::Buy => {
                            last_signal = VoteResult::Buy(signal.confidence);
                        }
                        quant_core::types::SignalAction::Sell => {
                            last_signal = VoteResult::Sell(signal.confidence);
                        }
                        quant_core::types::SignalAction::Hold => {}
                    }
                }
            }
        }

        strategy.on_stop();
        last_signal
    }

    /// Generate recommendation text and reasons
    fn generate_recommendation(
        &self,
        factors: &FactorScores,
        vote: &StrategyVote,
        composite: f64,
    ) -> (String, Vec<String>) {
        let mut reasons = Vec::new();

        // Momentum reasons
        if factors.momentum_5d > 0.03 {
            reasons.push(format!("5日涨幅 {:.1}%，短期动量强劲", factors.momentum_5d * 100.0));
        }
        if factors.momentum_20d > 0.05 {
            reasons.push(format!("20日涨幅 {:.1}%，中期趋势向好", factors.momentum_20d * 100.0));
        }

        // Trend reasons
        if factors.ma_trend > 0.01 {
            reasons.push("均线多头排列 (MA5 > MA20)".to_string());
        }
        if factors.macd_histogram > 0.0 {
            reasons.push("MACD柱状图为正，动能向上".to_string());
        }

        // RSI reasons
        if factors.rsi_14 < 35.0 {
            reasons.push(format!("RSI={:.1}，处于超卖区，反弹概率大", factors.rsi_14));
        } else if factors.rsi_14 > 50.0 && factors.rsi_14 < 65.0 {
            reasons.push(format!("RSI={:.1}，处于强势区间", factors.rsi_14));
        }

        // Volume reasons
        if factors.volume_ratio > 1.3 {
            reasons.push(format!("成交量放大 {:.1}倍，资金关注度高", factors.volume_ratio));
        }

        // Strategy consensus
        if vote.consensus_count == 3 {
            reasons.push("三大策略(SMA/RSI/MACD)同时发出买入信号".to_string());
        } else if vote.consensus_count == 2 {
            reasons.push(format!("{}个策略发出买入信号", vote.consensus_count));
        }

        // KDJ
        if factors.kdj_j < 20.0 {
            reasons.push(format!("KDJ J值={:.1}，超卖金叉可期", factors.kdj_j));
        }

        let recommendation = if composite >= 75.0 && vote.consensus_count >= 3 {
            "强烈推荐".to_string()
        } else if composite >= 60.0 && vote.consensus_count >= 2 {
            "推荐".to_string()
        } else if composite >= 45.0 {
            "观望".to_string()
        } else {
            "回避".to_string()
        };

        if reasons.is_empty() {
            reasons.push("综合因子评分较高".to_string());
        }

        (recommendation, reasons)
    }

    /// Generate a prompt for LLM analysis (Phase 3)
    pub fn generate_llm_prompt(&self, candidates: &[StockCandidate]) -> String {
        let mut prompt = String::from(
            "你是一位专业的A股量化分析师。请基于以下技术面数据分析这些股票，\
             给出你的投资建议。请用JSON格式返回，包含 recommendations 数组，\
             每个元素含 symbol, score(1-10), analysis, risks 字段。\n\n",
        );

        prompt.push_str("候选股票技术面数据：\n");
        for c in candidates {
            prompt.push_str(&format!(
                "\n{}（{}）当前价: ¥{:.2}\n\
                 - 5日涨幅: {:.1}%, 20日涨幅: {:.1}%\n\
                 - RSI(14): {:.1}, MACD柱: {:.4}\n\
                 - 布林带位置: {:.2}, 成交量倍数: {:.1}x\n\
                 - MA趋势: {:.3}, KDJ J值: {:.1}\n\
                 - 20日波动率: {:.1}%\n\
                 - 策略投票: {}/3个买入, 综合评分: {:.1}\n",
                c.symbol,
                c.name,
                c.price,
                c.factors.momentum_5d * 100.0,
                c.factors.momentum_20d * 100.0,
                c.factors.rsi_14,
                c.factors.macd_histogram,
                c.factors.bollinger_position,
                c.factors.volume_ratio,
                c.factors.ma_trend,
                c.factors.kdj_j,
                c.factors.volatility_20d * 100.0,
                c.strategy_vote.consensus_count,
                c.composite_score,
            ));
        }

        prompt
    }
}

/// Normalize a value from [min, max] to [0, 100]
fn normalize_score(value: f64, min: f64, max: f64) -> f64 {
    ((value - min) / (max - min) * 100.0).clamp(0.0, 100.0)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn make_klines(symbol: &str, base_price: f64, count: usize) -> Vec<Kline> {
        let mut klines = Vec::new();
        let mut price = base_price;
        let start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();

        // Seeded pseudo-random
        let seed: u64 = symbol.bytes().fold(42u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let mut rng = seed;

        for i in 0..count {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((rng >> 33) as f64) / (u32::MAX as f64) - 0.5;
            let cycle = (i as f64 * std::f64::consts::PI * 2.0 / 30.0).sin() * 0.01;
            let ret = 0.005 * r + cycle;
            let open = price;
            price *= 1.0 + ret;
            let close = price;
            let high = open.max(close) * 1.005;
            let low = open.min(close) * 0.995;
            let volume = 10_000_000.0 * (1.0 + r.abs());

            let date = start + chrono::Duration::days(i as i64);
            klines.push(Kline {
                symbol: symbol.to_string(),
                datetime: date.and_hms_opt(15, 0, 0).unwrap(),
                open,
                high,
                low,
                close,
                volume,
            });
        }
        klines
    }

    #[test]
    fn test_factor_computation() {
        let screener = StockScreener::new(ScreenerConfig::default());
        let klines = make_klines("600519.SH", 1700.0, 60);
        let factors = screener.compute_factors(&klines);

        assert!(factors.rsi_14 > 0.0 && factors.rsi_14 < 100.0);
        assert!(factors.volatility_20d > 0.0);
        assert!(factors.volume_ratio > 0.0);
    }

    #[test]
    fn test_screening_pipeline() {
        let screener = StockScreener::new(ScreenerConfig {
            top_n: 5,
            phase1_cutoff: 10,
            min_consensus: 1,
            ..ScreenerConfig::default()
        });

        let mut data = HashMap::new();
        let stocks = vec![
            ("600519.SH", "贵州茅台", 1700.0),
            ("000858.SZ", "五粮液", 150.0),
            ("601318.SH", "中国平安", 48.0),
            ("000001.SZ", "平安银行", 11.5),
            ("600036.SH", "招商银行", 34.0),
        ];

        for (sym, name, price) in stocks {
            data.insert(sym.to_string(), (name.to_string(), make_klines(sym, price, 60)));
        }

        let result = screener.screen(&data);
        assert_eq!(result.total_scanned, 5);
        assert!(result.phase1_passed <= 5);
        // Candidates should have composite scores and recommendations
        for c in &result.candidates {
            assert!(!c.recommendation.is_empty());
            assert!(c.composite_score > 0.0);
        }
    }

    #[test]
    fn test_llm_prompt_generation() {
        let screener = StockScreener::new(ScreenerConfig::default());
        let klines = make_klines("600519.SH", 1700.0, 60);
        let factors = screener.compute_factors(&klines);
        let vote = screener.run_strategy_votes(&klines);
        let candidate = StockCandidate {
            symbol: "600519.SH".to_string(),
            name: "贵州茅台".to_string(),
            price: 1700.0,
            factor_score: 65.0,
            factors,
            strategy_vote: vote,
            composite_score: 72.0,
            recommendation: "推荐".to_string(),
            reasons: vec!["测试".to_string()],
        };

        let prompt = screener.generate_llm_prompt(&[candidate]);
        assert!(prompt.contains("600519.SH"));
        assert!(prompt.contains("贵州茅台"));
        assert!(prompt.contains("RSI"));
    }
}
