// Strategy engine

use quant_core::models::Kline;
use quant_core::traits::Strategy;
use quant_core::types::Signal;

pub struct StrategyRunner {
    strategy: Box<dyn Strategy>,
    signals: Vec<Signal>,
}

impl StrategyRunner {
    pub fn new(strategy: Box<dyn Strategy>) -> Self {
        Self {
            strategy,
            signals: Vec::new(),
        }
    }

    pub fn run(&mut self, data: &[Kline]) -> Vec<Signal> {
        self.strategy.on_init();
        self.signals.clear();

        for kline in data {
            if let Some(signal) = self.strategy.on_bar(kline) {
                self.signals.push(signal);
            }
        }

        self.strategy.on_stop();
        self.signals.clone()
    }

    pub fn get_signals(&self) -> &[Signal] {
        &self.signals
    }
}
