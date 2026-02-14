use crate::metrics::PerformanceMetrics;

pub fn format_report(metrics: &PerformanceMetrics) -> String {
    format!(
        "\
╔══════════════════════════════════════════╗
║          Backtest Performance            ║
╠══════════════════════════════════════════╣
║  Total Return:       {:>10.2}%          ║
║  Annual Return:      {:>10.2}%          ║
║  Sharpe Ratio:       {:>10.4}           ║
║  Max Drawdown:       {:>10.2}%          ║
║  Max DD Duration:    {:>7} days         ║
╠══════════════════════════════════════════╣
║  Total Trades:       {:>10}             ║
║  Winning Trades:     {:>10}             ║
║  Losing Trades:      {:>10}             ║
║  Win Rate:           {:>10.2}%          ║
║  Profit Factor:      {:>10.4}           ║
║  Avg Win:            {:>10.2}           ║
║  Avg Loss:           {:>10.2}           ║
╚══════════════════════════════════════════╝",
        metrics.total_return * 100.0,
        metrics.annual_return * 100.0,
        metrics.sharpe_ratio,
        metrics.max_drawdown * 100.0,
        metrics.max_drawdown_duration,
        metrics.total_trades,
        metrics.winning_trades,
        metrics.losing_trades,
        metrics.win_rate * 100.0,
        metrics.profit_factor,
        metrics.avg_win,
        metrics.avg_loss,
    )
}
