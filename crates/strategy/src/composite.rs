// Composite strategy: combines multiple sub-strategies with weighted voting.

use quant_core::models::Kline;
use quant_core::traits::Strategy;
use quant_core::types::{Signal, SignalAction};

/// A strategy that combines multiple sub-strategies with weighted voting.
///
/// Each sub-strategy produces a signal on every bar. The composite strategy
/// converts BUY → +weight, SELL → −weight, HOLD → 0 and sums all weighted
/// votes into a single score. If the score exceeds `buy_threshold` a BUY
/// signal is emitted; if it falls below `sell_threshold` a SELL is emitted.
pub struct CompositeStrategy {
    strategies: Vec<(Box<dyn Strategy>, f64)>,
    name: String,
    buy_threshold: f64,
    sell_threshold: f64,
}

impl CompositeStrategy {
    pub fn new(
        strategies: Vec<(Box<dyn Strategy>, f64)>,
        buy_threshold: f64,
        sell_threshold: f64,
    ) -> Self {
        let names: Vec<String> = strategies
            .iter()
            .map(|(s, w)| format!("{}({:.0}%)", s.name(), w * 100.0))
            .collect();
        Self {
            strategies,
            name: format!("Composite[{}]", names.join("+")),
            buy_threshold,
            sell_threshold,
        }
    }

    pub fn with_defaults(strategies: Vec<(Box<dyn Strategy>, f64)>) -> Self {
        Self::new(strategies, 0.3, -0.3)
    }
}

impl Strategy for CompositeStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_init(&mut self) {
        for (strategy, _) in &mut self.strategies {
            strategy.on_init();
        }
    }

    fn on_bar(&mut self, kline: &Kline) -> Option<Signal> {
        let mut score = 0.0_f64;
        let mut total_weight = 0.0_f64;
        for (strategy, weight) in &mut self.strategies {
            let signal = strategy.on_bar(kline);
            total_weight += *weight;
            match signal {
                Some(ref s) => match s.action {
                    SignalAction::Buy => score += *weight * s.confidence.max(0.5),
                    SignalAction::Sell => score -= *weight * s.confidence.max(0.5),
                    SignalAction::Hold => {}
                },
                None => {}
            }
        }
        // Normalize score by total weight so it stays in roughly [-1, +1]
        if total_weight > f64::EPSILON {
            score /= total_weight;
        }

        if score > self.buy_threshold {
            Some(Signal::buy(
                &kline.symbol,
                score.abs().min(1.0),
                kline.datetime,
            ))
        } else if score < self.sell_threshold {
            Some(Signal::sell(
                &kline.symbol,
                score.abs().min(1.0),
                kline.datetime,
            ))
        } else {
            None
        }
    }

    fn on_stop(&mut self) {
        for (strategy, _) in &mut self.strategies {
            strategy.on_stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::DualMaCrossover;
    use crate::builtin::RsiMeanReversion;

    #[test]
    fn composite_creates_name() {
        let strategies: Vec<(Box<dyn Strategy>, f64)> = vec![
            (Box::new(DualMaCrossover::new(5, 20)), 0.5),
            (Box::new(RsiMeanReversion::new(14, 70.0, 30.0)), 0.5),
        ];
        let comp = CompositeStrategy::with_defaults(strategies);
        assert!(comp.name().contains("Composite"));
        assert!(comp.name().contains("DualMaCrossover"));
        assert!(comp.name().contains("RsiMeanReversion"));
    }
}
