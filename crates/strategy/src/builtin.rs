// Built-in strategies

use crate::indicators::{SMA, MACD, RSI};
use quant_core::models::Kline;
use quant_core::traits::Strategy;
use quant_core::types::Signal;


/// Dual Moving Average Crossover Strategy
pub struct DualMaCrossover {
    fast_period: usize,
    slow_period: usize,
    fast_ma: SMA,
    slow_ma: SMA,
    prev_fast: Option<f64>,
    prev_slow: Option<f64>,
}

impl DualMaCrossover {
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            fast_ma: SMA::new(fast_period),
            slow_ma: SMA::new(slow_period),
            prev_fast: None,
            prev_slow: None,
        }
    }
}

impl Strategy for DualMaCrossover {
    fn name(&self) -> &str {
        "DualMaCrossover"
    }

    fn on_init(&mut self) {
        self.fast_ma = SMA::new(self.fast_period);
        self.slow_ma = SMA::new(self.slow_period);
        self.prev_fast = None;
        self.prev_slow = None;
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        let fast_val = self.fast_ma.update(kline.close);
        let slow_val = self.slow_ma.update(kline.close);

        let result = match (fast_val, slow_val, self.prev_fast, self.prev_slow) {
            (Some(fast), Some(slow), Some(pf), Some(ps)) => {
                if pf <= ps && fast > slow {
                    // Golden cross: fast crosses above slow
                    let confidence = (fast - slow) / slow;
                    Some(Signal::buy(&kline.symbol, confidence, kline.datetime))
                } else if pf >= ps && fast < slow {
                    // Death cross: fast crosses below slow
                    let confidence = (slow - fast) / slow;
                    Some(Signal::sell(&kline.symbol, confidence, kline.datetime))
                } else {
                    None
                }
            }
            _ => None,
        };

        self.prev_fast = fast_val;
        self.prev_slow = slow_val;
        result
    }

    fn on_stop(&mut self) {}
}

/// RSI Mean Reversion Strategy
pub struct RsiMeanReversion {
    period: usize,
    overbought: f64,
    oversold: f64,
    rsi: RSI,
}

impl RsiMeanReversion {
    pub fn new(period: usize, overbought: f64, oversold: f64) -> Self {
        Self {
            period,
            overbought,
            oversold,
            rsi: RSI::new(period),
        }
    }
}

impl Strategy for RsiMeanReversion {
    fn name(&self) -> &str {
        "RsiMeanReversion"
    }

    fn on_init(&mut self) {
        self.rsi = RSI::new(self.period);
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        let rsi_val = self.rsi.update(kline.close)?;

        if rsi_val < self.oversold {
            // Oversold — expect reversion upward
            let confidence = (self.oversold - rsi_val) / self.oversold;
            Some(Signal::buy(&kline.symbol, confidence, kline.datetime))
        } else if rsi_val > self.overbought {
            // Overbought — expect reversion downward
            let confidence = (rsi_val - self.overbought) / (100.0 - self.overbought);
            Some(Signal::sell(&kline.symbol, confidence, kline.datetime))
        } else {
            None
        }
    }

    fn on_stop(&mut self) {}
}

/// Momentum Strategy using MACD
pub struct MacdMomentum {
    fast: usize,
    slow: usize,
    signal: usize,
    macd: MACD,
    prev_histogram: Option<f64>,
}

impl MacdMomentum {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast,
            slow,
            signal,
            macd: MACD::new(fast, slow, signal),
            prev_histogram: None,
        }
    }
}

impl Strategy for MacdMomentum {
    fn name(&self) -> &str {
        "MacdMomentum"
    }

    fn on_init(&mut self) {
        self.macd = MACD::new(self.fast, self.slow, self.signal);
        self.prev_histogram = None;
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        self.macd.update(kline.close);
        let histogram = self.macd.histogram();

        let result = match (histogram, self.prev_histogram) {
            (Some(curr), Some(prev)) => {
                if prev <= 0.0 && curr > 0.0 {
                    // Histogram crosses above zero — bullish momentum
                    Some(Signal::buy(&kline.symbol, curr.abs(), kline.datetime))
                } else if prev >= 0.0 && curr < 0.0 {
                    // Histogram crosses below zero — bearish momentum
                    Some(Signal::sell(&kline.symbol, curr.abs(), kline.datetime))
                } else {
                    None
                }
            }
            _ => None,
        };

        self.prev_histogram = histogram;
        result
    }

    fn on_stop(&mut self) {}
}
