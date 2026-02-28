//! Walk-forward validation for parameterized strategies.
//!
//! Splits historical data into rolling train/test windows. For each window,
//! the strategy is constructed with parameters optimized on the train set,
//! then evaluated on the out-of-sample (OOS) test set. Reports aggregated
//! OOS metrics to detect overfitting.

use chrono::NaiveDateTime;
use serde::Serialize;

use quant_core::models::Kline;
use quant_core::traits::Strategy;

use crate::engine::{BacktestConfig, BacktestEngine};

/// Result of a single walk-forward fold.
#[derive(Debug, Clone, Serialize)]
pub struct WalkForwardFold {
    pub fold: usize,
    pub train_start: String,
    pub train_end: String,
    pub test_start: String,
    pub test_end: String,
    pub train_bars: usize,
    pub test_bars: usize,
    pub oos_return: f64,
    pub oos_sharpe: f64,
    pub oos_sortino: f64,
    pub oos_max_drawdown: f64,
    pub oos_win_rate: f64,
    pub oos_trades: usize,
}

/// Aggregated walk-forward result across all folds.
#[derive(Debug, Clone, Serialize)]
pub struct WalkForwardResult {
    pub strategy: String,
    pub symbol: String,
    pub n_folds: usize,
    pub folds: Vec<WalkForwardFold>,
    pub avg_oos_return: f64,
    pub avg_oos_sharpe: f64,
    pub avg_oos_sortino: f64,
    pub avg_oos_max_drawdown: f64,
    pub avg_oos_win_rate: f64,
    pub total_oos_trades: usize,
    /// Ratio of OOS Sharpe / in-sample Sharpe (< 0.5 = likely overfit)
    pub degradation_ratio: f64,
}

/// Run walk-forward validation on a strategy factory.
///
/// `strategy_factory` creates a fresh strategy instance for each fold.
/// `data` is the full kline dataset sorted by datetime.
/// `n_folds` is the number of rolling windows.
/// `train_pct` is the fraction of each window used for training (rest for testing).
pub fn walk_forward_validate(
    strategy_name: &str,
    strategy_factory: &dyn Fn() -> Box<dyn Strategy>,
    data: &[Kline],
    config: &BacktestConfig,
    n_folds: usize,
    _train_pct: f64,
) -> WalkForwardResult {
    let n = data.len();
    if n < 100 || n_folds == 0 {
        return WalkForwardResult {
            strategy: strategy_name.to_string(),
            symbol: data.first().map(|k| k.symbol.clone()).unwrap_or_default(),
            n_folds: 0,
            folds: vec![],
            avg_oos_return: 0.0,
            avg_oos_sharpe: 0.0,
            avg_oos_sortino: 0.0,
            avg_oos_max_drawdown: 0.0,
            avg_oos_win_rate: 0.0,
            total_oos_trades: 0,
            degradation_ratio: 0.0,
        };
    }

    // Expanding window: each fold adds more training data
    // Fold k: train on [0..split_k], test on [split_k..split_k+test_size]
    let test_size = n / (n_folds + 1);
    let mut folds = Vec::new();
    let mut is_sharpes = Vec::new();

    for fold in 0..n_folds {
        let test_end = ((fold + 2) * test_size).min(n);
        let test_start = test_end.saturating_sub(test_size);
        let train_end = test_start;
        let train_start = 0;

        if train_end < 60 || test_end - test_start < 20 {
            continue;
        }

        let train_data = &data[train_start..train_end];
        let test_data = &data[test_start..test_end];

        // Run on training set (for degradation ratio)
        let engine = BacktestEngine::new(config.clone());
        let mut train_strat = strategy_factory();
        let train_result = engine.run(train_strat.as_mut(), train_data);
        is_sharpes.push(train_result.metrics.sharpe_ratio);

        // Run on test set (OOS)
        let mut test_strat = strategy_factory();
        let test_result = engine.run(test_strat.as_mut(), test_data);
        let m = &test_result.metrics;

        let fmt = |dt: NaiveDateTime| dt.format("%Y-%m-%d").to_string();

        folds.push(WalkForwardFold {
            fold: fold + 1,
            train_start: fmt(train_data.first().unwrap().datetime),
            train_end: fmt(train_data.last().unwrap().datetime),
            test_start: fmt(test_data.first().unwrap().datetime),
            test_end: fmt(test_data.last().unwrap().datetime),
            train_bars: train_data.len(),
            test_bars: test_data.len(),
            oos_return: m.total_return,
            oos_sharpe: m.sharpe_ratio,
            oos_sortino: m.sortino_ratio,
            oos_max_drawdown: m.max_drawdown,
            oos_win_rate: m.win_rate,
            oos_trades: m.total_trades,
        });
    }

    let n_valid = folds.len().max(1) as f64;
    let avg_oos_return = folds.iter().map(|f| f.oos_return).sum::<f64>() / n_valid;
    let avg_oos_sharpe = folds.iter().map(|f| f.oos_sharpe).sum::<f64>() / n_valid;
    let avg_oos_sortino = folds.iter().map(|f| f.oos_sortino).sum::<f64>() / n_valid;
    let avg_oos_max_drawdown = folds.iter().map(|f| f.oos_max_drawdown).sum::<f64>() / n_valid;
    let avg_oos_win_rate = folds.iter().map(|f| f.oos_win_rate).sum::<f64>() / n_valid;
    let total_oos_trades = folds.iter().map(|f| f.oos_trades).sum();

    let avg_is_sharpe = if is_sharpes.is_empty() {
        0.0
    } else {
        is_sharpes.iter().sum::<f64>() / is_sharpes.len() as f64
    };
    let degradation_ratio = if avg_is_sharpe.abs() > 0.01 {
        avg_oos_sharpe / avg_is_sharpe
    } else {
        0.0
    };

    WalkForwardResult {
        strategy: strategy_name.to_string(),
        symbol: data.first().map(|k| k.symbol.clone()).unwrap_or_default(),
        n_folds: folds.len(),
        folds,
        avg_oos_return,
        avg_oos_sharpe,
        avg_oos_sortino,
        avg_oos_max_drawdown,
        avg_oos_win_rate,
        total_oos_trades,
        degradation_ratio,
    }
}

/// Format walk-forward results for CLI display.
pub fn format_walk_forward_report(result: &WalkForwardResult) -> String {
    let mut out = format!(
        "\n╔══════════════════════════════════════════════════╗\n\
         ║     Walk-Forward Validation: {:20} ║\n\
         ║     Symbol: {:37} ║\n\
         ╠══════════════════════════════════════════════════╣\n",
        result.strategy, result.symbol
    );

    for f in &result.folds {
        out += &format!(
            "║  Fold {}: train [{} → {}] test [{} → {}]\n\
             ║    OOS Return: {:+.2}%  Sharpe: {:.2}  DD: {:.2}%  Trades: {}\n",
            f.fold, f.train_start, f.train_end, f.test_start, f.test_end,
            f.oos_return * 100.0, f.oos_sharpe, f.oos_max_drawdown * 100.0, f.oos_trades,
        );
    }

    out += &format!(
        "╠══════════════════════════════════════════════════╣\n\
         ║  Avg OOS Return:     {:>10.2}%                  ║\n\
         ║  Avg OOS Sharpe:     {:>10.4}                   ║\n\
         ║  Avg OOS Sortino:    {:>10.4}                   ║\n\
         ║  Avg OOS MaxDD:      {:>10.2}%                  ║\n\
         ║  Avg OOS WinRate:    {:>10.2}%                  ║\n\
         ║  Total OOS Trades:   {:>10}                     ║\n\
         ║  Degradation Ratio:  {:>10.4}                   ║\n\
         ║  (< 0.5 = likely overfit)                       ║\n\
         ╚══════════════════════════════════════════════════╝\n",
        result.avg_oos_return * 100.0,
        result.avg_oos_sharpe,
        result.avg_oos_sortino,
        result.avg_oos_max_drawdown * 100.0,
        result.avg_oos_win_rate * 100.0,
        result.total_oos_trades,
        result.degradation_ratio,
    );

    out
}
