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

/// Market regime detected from index data
#[derive(Debug, Clone, Serialize)]
pub enum MarketRegime {
    Trending,
    MeanReverting,
    Volatile,
}

/// Risk metrics for a stock candidate
#[derive(Debug, Clone, Serialize)]
pub struct RiskMetrics {
    pub max_drawdown_20d: f64,
    pub consecutive_down_days: u32,
    pub distance_from_high_20d: f64,
    pub atr_ratio: f64,
}

/// A mined factor loaded from the factor registry
#[derive(Debug, Clone)]
pub struct MinedFactor {
    pub name: String,
    pub expression: String,
    pub ic: f64,
    pub ir: f64,
    pub weight: f64,
}

/// Input entry for each stock (name, klines, sector)
#[derive(Debug, Clone)]
pub struct StockEntry {
    pub name: String,
    pub klines: Vec<Kline>,
    pub sector: String,
}

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
    pub sector: String,
    pub avg_turnover: f64,
    pub mined_factor_bonus: f64,
    pub risk: RiskMetrics,
}

/// Full screening result
#[derive(Debug, Clone, Serialize)]
pub struct ScreenerResult {
    pub candidates: Vec<StockCandidate>,
    pub total_scanned: usize,
    pub phase1_passed: usize,
    pub phase2_passed: usize,
    pub regime: Option<MarketRegime>,
}

/// Screening configuration
#[derive(Debug, Clone)]
pub struct ScreenerConfig {
    pub top_n: usize,              // final number of recommendations
    pub phase1_cutoff: usize,      // how many pass Phase 1 to Phase 2
    pub factor_weights: FactorWeights,
    pub min_consensus: u32,        // min strategy votes for Phase 2 pass
    pub buy_threshold: f64,        // composite score >= this for "推荐"
    pub strong_buy_threshold: f64, // for "强烈推荐"
    pub lookback_days: u32,        // data lookback
    pub min_turnover: f64,         // minimum avg daily turnover (volume * price)
    pub max_per_sector: usize,     // max stocks from same sector
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
            buy_threshold: 60.0,
            strong_buy_threshold: 75.0,
            lookback_days: 150,
            min_turnover: 50_000_000.0,
            max_per_sector: 3,
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

    /// Detect market regime from index kline data
    pub fn detect_regime(index_klines: &[Kline]) -> MarketRegime {
        if index_klines.len() < 60 {
            return MarketRegime::MeanReverting;
        }
        let closes: Vec<f64> = index_klines.iter().map(|k| k.close).collect();
        let n = closes.len();

        // Compute MA20 and MA60
        let ma20: f64 = closes[n - 20..].iter().sum::<f64>() / 20.0;
        let ma60: f64 = closes[n - 60..].iter().sum::<f64>() / 60.0;

        // 20-day volatility
        let daily_returns: Vec<f64> = closes.windows(2).skip(n - 21)
            .map(|w| (w[1] / w[0]).ln()).collect();
        let mean_ret = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let var = daily_returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / (daily_returns.len() - 1) as f64;
        let ann_vol = var.sqrt() * (252.0_f64).sqrt();

        let ma_diff_pct = (ma20 - ma60).abs() / ma60;
        if ann_vol > 0.35 {
            MarketRegime::Volatile
        } else if ma20 > ma60 && ma_diff_pct > 0.01 {
            MarketRegime::Trending
        } else {
            MarketRegime::MeanReverting
        }
    }

    /// Adjust factor weights based on market regime
    pub fn regime_adjusted_weights(regime: &MarketRegime) -> FactorWeights {
        match regime {
            MarketRegime::Trending => FactorWeights {
                momentum: 0.35, trend: 0.35, mean_reversion: 0.10, volume: 0.10, volatility: 0.10,
            },
            MarketRegime::MeanReverting => FactorWeights {
                momentum: 0.15, trend: 0.20, mean_reversion: 0.35, volume: 0.15, volatility: 0.15,
            },
            MarketRegime::Volatile => FactorWeights {
                momentum: 0.15, trend: 0.25, mean_reversion: 0.20, volume: 0.15, volatility: 0.25,
            },
        }
    }

    /// Load mined factors from factor registry SQLite DB
    pub fn load_mined_factors() -> Vec<MinedFactor> {
        let db_path = std::path::Path::new("data/factor_registry.db");
        if !db_path.exists() {
            return Vec::new();
        }
        let conn = match rusqlite::Connection::open(db_path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let mut stmt = match conn.prepare(
            "SELECT expression, expression, ic_mean, ir, ic_mean \
             FROM factors WHERE state = 'promoted' ORDER BY ir DESC LIMIT 5"
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map([], |row| {
            Ok(MinedFactor {
                name: row.get::<_, String>(0)?,
                expression: row.get::<_, String>(1)?,
                ic: row.get::<_, f64>(2)?,
                ir: row.get::<_, f64>(3)?,
                weight: row.get::<_, f64>(4)?.abs().min(1.0),
            })
        });
        match rows {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Compute mined factor bonus (0-20)
    pub(crate) fn compute_mined_bonus(factors: &[MinedFactor]) -> f64 {
        if factors.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = factors.iter().map(|f| f.weight).sum();
        if total_weight <= 0.0 {
            return 0.0;
        }
        let weighted_ic: f64 = factors.iter().map(|f| f.ic.abs() * f.weight).sum::<f64>() / total_weight;
        // Scale IC (0..0.1) to bonus (0..20)
        (weighted_ic / 0.1 * 20.0).clamp(0.0, 20.0)
    }

    /// Compute risk metrics from kline data
    pub(crate) fn compute_risk(klines: &[Kline]) -> RiskMetrics {
        let n = klines.len();
        let window = n.min(20);
        let recent = &klines[n - window..];
        let closes: Vec<f64> = recent.iter().map(|k| k.close).collect();

        // Max drawdown over last 20 days
        let mut peak = closes[0];
        let mut max_dd = 0.0_f64;
        for &c in &closes {
            if c > peak { peak = c; }
            let dd = (peak - c) / peak;
            if dd > max_dd { max_dd = dd; }
        }

        // Consecutive down days (from end)
        let mut consecutive_down = 0u32;
        for w in closes.windows(2).rev() {
            if w[1] < w[0] { consecutive_down += 1; } else { break; }
        }

        // Distance from 20-day high
        let high_20d = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let last = *closes.last().unwrap_or(&1.0);
        let dist = if high_20d > 0.0 { (high_20d - last) / high_20d } else { 0.0 };

        // ATR ratio (ATR / price)
        let atr_vals: Vec<f64> = recent.windows(2).map(|w| {
            let tr = (w[1].high - w[1].low)
                .max((w[1].high - w[0].close).abs())
                .max((w[1].low - w[0].close).abs());
            tr
        }).collect();
        let atr = if atr_vals.is_empty() { 0.0 } else { atr_vals.iter().sum::<f64>() / atr_vals.len() as f64 };
        let atr_ratio = if last > 0.0 { atr / last } else { 0.0 };

        RiskMetrics {
            max_drawdown_20d: max_dd,
            consecutive_down_days: consecutive_down,
            distance_from_high_20d: dist,
            atr_ratio,
        }
    }

    /// Compute average daily turnover (volume * close) over last 20 days
    pub(crate) fn compute_avg_turnover(klines: &[Kline]) -> f64 {
        let n = klines.len();
        let window = n.min(20);
        let recent = &klines[n - window..];
        let sum: f64 = recent.iter().map(|k| k.volume * k.close).sum();
        sum / window as f64
    }

    /// Run full 3-phase screening pipeline.
    /// Input: map of symbol -> StockEntry (name, klines, sector)
    /// Returns scored and ranked candidates.
    pub fn screen(
        &self,
        stock_data: &HashMap<String, StockEntry>,
    ) -> ScreenerResult {
        self.screen_with_regime(stock_data, None)
    }

    /// Screen with optional regime detection
    pub fn screen_with_regime(
        &self,
        stock_data: &HashMap<String, StockEntry>,
        regime: Option<MarketRegime>,
    ) -> ScreenerResult {
        let total_scanned = stock_data.len();
        let mined_factors = Self::load_mined_factors();
        let mined_bonus = Self::compute_mined_bonus(&mined_factors);

        // Phase 1: Multi-factor scoring + liquidity filtering
        let mut scored: Vec<(String, String, String, f64, FactorScores, f64, f64, RiskMetrics)> = stock_data
            .iter()
            .filter_map(|(symbol, entry)| {
                if entry.klines.len() < 30 {
                    return None;
                }
                let avg_turnover = Self::compute_avg_turnover(&entry.klines);
                if avg_turnover < self.config.min_turnover {
                    return None;
                }
                let factors = self.compute_factors(&entry.klines);
                let score = self.composite_factor_score(&factors);
                let price = entry.klines.last().unwrap().close;
                let risk = Self::compute_risk(&entry.klines);
                Some((symbol.clone(), entry.name.clone(), entry.sector.clone(), price, factors, score, avg_turnover, risk))
            })
            .collect();

        // Sort by factor score descending
        scored.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));

        // Sector neutralization: limit max_per_sector
        let mut sector_counts: HashMap<String, usize> = HashMap::new();
        scored.retain(|item| {
            let count = sector_counts.entry(item.2.clone()).or_insert(0);
            if *count < self.config.max_per_sector {
                *count += 1;
                true
            } else {
                false
            }
        });

        scored.truncate(self.config.phase1_cutoff);
        let phase1_passed = scored.len();

        // Phase 2: Strategy signal aggregation
        let mut candidates: Vec<StockCandidate> = scored
            .into_iter()
            .filter_map(|(symbol, name, sector, price, factors, factor_score, avg_turnover, risk)| {
                let klines = &stock_data[&symbol].klines;
                let vote = self.run_strategy_votes(klines);

                if vote.consensus_count < self.config.min_consensus {
                    return None;
                }

                // Composite: factor_score * 0.5 + strategy * 0.3 + mined_bonus * 0.2
                let composite = factor_score * 0.5
                    + vote.avg_confidence * 100.0 * 0.3
                    + mined_bonus * 0.2;

                let (recommendation, reasons) =
                    self.generate_recommendation(&factors, &vote, composite, &risk);

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
                    sector,
                    avg_turnover,
                    mined_factor_bonus: mined_bonus,
                    risk,
                })
            })
            .collect();

        let phase2_passed = candidates.len();

        candidates
            .sort_by(|a, b| b.composite_score.partial_cmp(&a.composite_score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.top_n);

        ScreenerResult {
            candidates,
            total_scanned,
            phase1_passed,
            phase2_passed,
            regime,
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
        risk: &RiskMetrics,
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

        // Risk warnings
        if risk.max_drawdown_20d > 0.15 {
            reasons.push(format!("⚠️ 20日最大回撤 {:.1}%，风险较高", risk.max_drawdown_20d * 100.0));
        }
        if risk.consecutive_down_days > 5 {
            reasons.push(format!("⚠️ 连续下跌{}天，注意风险", risk.consecutive_down_days));
        }

        let recommendation = if composite >= self.config.strong_buy_threshold && vote.consensus_count >= 3 {
            "强烈推荐".to_string()
        } else if composite >= self.config.buy_threshold && vote.consensus_count >= 2 {
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

    fn make_entry(name: &str, klines: Vec<Kline>) -> StockEntry {
        StockEntry { name: name.to_string(), klines, sector: "测试".to_string() }
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
            min_turnover: 0.0,
            max_per_sector: 100,
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
            data.insert(sym.to_string(), make_entry(name, make_klines(sym, price, 60)));
        }

        let result = screener.screen(&data);
        assert_eq!(result.total_scanned, 5);
        assert!(result.phase1_passed <= 5);
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
        let risk = StockScreener::compute_risk(&klines);
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
            sector: "白酒".to_string(),
            avg_turnover: 100_000_000.0,
            mined_factor_bonus: 0.0,
            risk,
        };

        let prompt = screener.generate_llm_prompt(&[candidate]);
        assert!(prompt.contains("600519.SH"));
        assert!(prompt.contains("贵州茅台"));
        assert!(prompt.contains("RSI"));
    }

    #[test]
    fn test_market_regime_detection() {
        let klines = make_klines("000001.SH", 3000.0, 120);
        let regime = StockScreener::detect_regime(&klines);
        // Just check it returns a valid regime
        match regime {
            MarketRegime::Trending | MarketRegime::MeanReverting | MarketRegime::Volatile => {}
        }
    }

    #[test]
    fn test_risk_metrics() {
        let klines = make_klines("600519.SH", 1700.0, 60);
        let risk = StockScreener::compute_risk(&klines);
        assert!(risk.max_drawdown_20d >= 0.0);
        assert!(risk.atr_ratio >= 0.0);
    }

    #[test]
    fn test_sector_neutralization() {
        let screener = StockScreener::new(ScreenerConfig {
            top_n: 10,
            phase1_cutoff: 10,
            min_consensus: 0,
            min_turnover: 0.0,
            max_per_sector: 1,
            ..ScreenerConfig::default()
        });

        let mut data = HashMap::new();
        // All same sector
        for (sym, name, price) in &[
            ("600519.SH", "贵州茅台", 1700.0),
            ("000858.SZ", "五粮液", 150.0),
            ("000568.SZ", "泸州老窖", 200.0),
        ] {
            data.insert(sym.to_string(), StockEntry {
                name: name.to_string(),
                klines: make_klines(sym, *price, 60),
                sector: "白酒".to_string(),
            });
        }

        let result = screener.screen(&data);
        // max_per_sector=1, so at most 1 白酒 stock should pass phase 1
        assert!(result.phase1_passed <= 1);
    }
}
