use chrono::NaiveDateTime;
use quant_core::models::Trade;
use quant_core::types::OrderSide;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annual_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: i64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub avg_win: f64,
    pub avg_loss: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from equity curve and trades.
    pub fn calculate(
        equity_curve: &[(NaiveDateTime, f64)],
        trades: &[Trade],
        initial_capital: f64,
    ) -> Self {
        let final_equity = equity_curve.last().map(|(_, v)| *v).unwrap_or(initial_capital);
        let total_return = (final_equity - initial_capital) / initial_capital;

        // Annualized return
        let trading_days = if equity_curve.len() > 1 {
            let first = equity_curve.first().unwrap().0;
            let last = equity_curve.last().unwrap().0;
            (last - first).num_days().max(1) as f64
        } else {
            1.0
        };
        let years = trading_days / 365.0;
        let annual_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        // Daily returns for Sharpe ratio
        let daily_returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1].1 - w[0].1) / w[0].1)
            .collect();

        let sharpe_ratio = if daily_returns.len() > 1 {
            let mean = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
            let variance = daily_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / (daily_returns.len() - 1) as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.0 {
                (mean / std_dev) * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Max drawdown & duration
        let mut peak = initial_capital;
        let mut max_drawdown = 0.0_f64;
        let mut max_dd_duration: i64 = 0;
        let mut dd_start: Option<NaiveDateTime> = None;

        for (dt, equity) in equity_curve {
            if *equity >= peak {
                peak = *equity;
                dd_start = None;
            } else {
                let dd = (peak - equity) / peak;
                if dd > max_drawdown {
                    max_drawdown = dd;
                }
                if dd_start.is_none() {
                    dd_start = Some(*dt);
                }
                if let Some(start) = dd_start {
                    let dur = (*dt - start).num_days();
                    if dur > max_dd_duration {
                        max_dd_duration = dur;
                    }
                }
            }
        }

        // Round-trip PnL: pair buy/sell trades per symbol
        let mut pnls: Vec<f64> = Vec::new();
        let mut open_buys: std::collections::HashMap<String, Vec<(f64, f64)>> =
            std::collections::HashMap::new();

        for trade in trades {
            match trade.side {
                OrderSide::Buy => {
                    open_buys
                        .entry(trade.symbol.clone())
                        .or_default()
                        .push((trade.price, trade.quantity));
                }
                OrderSide::Sell => {
                    if let Some(buys) = open_buys.get_mut(&trade.symbol) {
                        let mut remaining = trade.quantity;
                        while remaining > 0.0 && !buys.is_empty() {
                            let (buy_price, buy_qty) = &mut buys[0];
                            let matched = remaining.min(*buy_qty);
                            let pnl = (trade.price - *buy_price) * matched
                                - trade.commission * (matched / trade.quantity);
                            pnls.push(pnl);
                            remaining -= matched;
                            *buy_qty -= matched;
                            if *buy_qty <= 1e-10 {
                                buys.remove(0);
                            }
                        }
                    }
                }
            }
        }

        let winning_trades = pnls.iter().filter(|p| **p > 0.0).count();
        let losing_trades = pnls.iter().filter(|p| **p < 0.0).count();
        let total_trades = pnls.len();
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = pnls.iter().filter(|p| **p > 0.0).sum();
        let gross_loss: f64 = pnls.iter().filter(|p| **p < 0.0).map(|p| p.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_win = if winning_trades > 0 {
            gross_profit / winning_trades as f64
        } else {
            0.0
        };
        let avg_loss = if losing_trades > 0 {
            gross_loss / losing_trades as f64
        } else {
            0.0
        };

        Self {
            total_return,
            annual_return,
            sharpe_ratio,
            max_drawdown,
            max_drawdown_duration: max_dd_duration,
            win_rate,
            profit_factor,
            total_trades,
            winning_trades,
            losing_trades,
            avg_win,
            avg_loss,
        }
    }
}
