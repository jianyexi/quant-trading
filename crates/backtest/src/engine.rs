use std::collections::HashMap;

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use quant_core::models::{Kline, Order, Portfolio, Position, Trade};
use quant_core::traits::Strategy;
use quant_core::types::{OrderSide, OrderStatus, OrderType, SignalAction};

use crate::matching::MatchingEngine;
use crate::metrics::PerformanceMetrics;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub stamp_tax_rate: f64,
    pub slippage_ticks: u32,
    /// Fraction of portfolio value per buy order (e.g. 0.10 = 10%). Default 1.0 = all-in.
    #[serde(default = "default_position_size_pct")]
    pub position_size_pct: f64,
    /// Max fraction of portfolio in a single stock (e.g. 0.25 = 25%). Default 1.0 = no limit.
    #[serde(default = "default_max_concentration")]
    pub max_concentration: f64,
    /// Stop-loss threshold per position (e.g. 0.05 = 5%). Default 0.0 = disabled.
    #[serde(default)]
    pub stop_loss_pct: f64,
    /// Max holding days before forced exit (0 = disabled).
    #[serde(default)]
    pub max_holding_days: u32,
}

fn default_position_size_pct() -> f64 { 1.0 }
fn default_max_concentration() -> f64 { 1.0 }

pub struct BacktestResult {
    pub config: BacktestConfig,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<(NaiveDateTime, f64)>,
    pub metrics: PerformanceMetrics,
    pub final_portfolio: Portfolio,
}

pub struct BacktestEngine {
    config: BacktestConfig,
    matching: MatchingEngine,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        let matching = MatchingEngine::new(config.slippage_ticks);
        Self { config, matching }
    }

    /// Execute a sell order and update portfolio. Returns the trade if filled.
    fn execute_sell(
        &self,
        symbol: &str,
        quantity: f64,
        kline: &Kline,
        portfolio: &mut Portfolio,
    ) -> Option<Trade> {
        if quantity <= 0.0 {
            return None;
        }
        let order = Order {
            id: Uuid::new_v4(),
            symbol: symbol.to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Market,
            price: kline.close,
            quantity,
            filled_qty: 0.0,
            status: OrderStatus::Pending,
            created_at: kline.datetime,
            updated_at: kline.datetime,
        };

        let mut trade = self.matching.try_match(&order, kline)?;
        let turnover = trade.price * trade.quantity;
        trade.commission =
            turnover * self.config.commission_rate + turnover * self.config.stamp_tax_rate;

        let proceeds = turnover - trade.commission;
        portfolio.cash += proceeds;

        if let Some(pos) = portfolio.positions.get_mut(symbol) {
            pos.realized_pnl += (trade.price - pos.avg_cost) * trade.quantity;
            pos.quantity -= trade.quantity;
            if pos.quantity <= 1e-10 {
                portfolio.positions.remove(symbol);
            }
        }
        Some(trade)
    }

    /// Run backtest with a strategy on kline data.
    pub fn run(&self, strategy: &mut dyn Strategy, data: &[Kline]) -> BacktestResult {
        let mut portfolio = Portfolio {
            positions: HashMap::new(),
            cash: self.config.initial_capital,
            total_value: self.config.initial_capital,
        };

        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<(NaiveDateTime, f64)> = Vec::new();

        strategy.on_init();

        for kline in data {
            // 0. Risk checks: stop-loss & holding timeout on existing positions
            let mut forced_sells: Vec<(String, f64)> = Vec::new();
            for pos in portfolio.positions.values() {
                if pos.symbol != kline.symbol {
                    continue;
                }
                // Stop-loss check
                if self.config.stop_loss_pct > 0.0 && pos.avg_cost > 0.0 {
                    let loss_pct = (pos.current_price - pos.avg_cost) / pos.avg_cost;
                    if loss_pct < -self.config.stop_loss_pct {
                        forced_sells.push((pos.symbol.clone(), pos.quantity));
                        continue;
                    }
                }
                // Holding timeout check
                if self.config.max_holding_days > 0 {
                    let days = (kline.datetime - pos.entry_time).num_days();
                    if days > self.config.max_holding_days as i64 {
                        forced_sells.push((pos.symbol.clone(), pos.quantity));
                    }
                }
            }
            for (sym, qty) in &forced_sells {
                if let Some(trade) = self.execute_sell(sym, *qty, kline, &mut portfolio) {
                    trades.push(trade);
                }
            }

            // 1. Generate signal
            let signal = strategy.on_bar(kline);

            // 2. Convert signal to order and attempt fill
            if let Some(sig) = signal {
                let (side, should_trade) = match sig.action {
                    SignalAction::Buy => (OrderSide::Buy, true),
                    SignalAction::Sell => (OrderSide::Sell, true),
                    SignalAction::Hold => (OrderSide::Buy, false),
                };

                if should_trade {
                    let quantity = match side {
                        OrderSide::Buy => {
                            // Position sizing: use configured percentage of portfolio
                            let allocation = portfolio.total_value * self.config.position_size_pct;
                            let budget = allocation.min(portfolio.cash);

                            // Concentration check: cap by max allowed for this symbol
                            let existing_value = portfolio
                                .positions
                                .get(&sig.symbol)
                                .map(|p| p.current_price * p.quantity)
                                .unwrap_or(0.0);
                            let max_allowed =
                                portfolio.total_value * self.config.max_concentration - existing_value;
                            let budget = budget.min(max_allowed.max(0.0));

                            let affordable =
                                (budget / (kline.close * (1.0 + self.config.commission_rate)))
                                    .floor();
                            // Round to lot size of 100 for A-shares
                            (affordable / 100.0).floor() * 100.0
                        }
                        OrderSide::Sell => {
                            // Sell entire position
                            portfolio
                                .positions
                                .get(&sig.symbol)
                                .map(|p| p.quantity)
                                .unwrap_or(0.0)
                        }
                    };

                    if quantity > 0.0 {
                        if side == OrderSide::Sell {
                            if let Some(trade) =
                                self.execute_sell(&sig.symbol, quantity, kline, &mut portfolio)
                            {
                                trades.push(trade);
                            }
                        } else {
                            let order = Order {
                                id: Uuid::new_v4(),
                                symbol: sig.symbol.clone(),
                                side,
                                order_type: OrderType::Market,
                                price: kline.close,
                                quantity,
                                filled_qty: 0.0,
                                status: OrderStatus::Pending,
                                created_at: kline.datetime,
                                updated_at: kline.datetime,
                            };

                            if let Some(mut trade) = self.matching.try_match(&order, kline) {
                                let turnover = trade.price * trade.quantity;
                                trade.commission = turnover * self.config.commission_rate;

                                let cost = turnover + trade.commission;
                                portfolio.cash -= cost;

                                let pos = portfolio
                                    .positions
                                    .entry(trade.symbol.clone())
                                    .or_insert(Position {
                                        symbol: trade.symbol.clone(),
                                        quantity: 0.0,
                                        avg_cost: 0.0,
                                        current_price: 0.0,
                                        unrealized_pnl: 0.0,
                                        realized_pnl: 0.0,
                                        entry_time: kline.datetime,
                                        scale_level: 1,
                                        target_weight: 0.0,
                                    });
                                let total_cost =
                                    pos.avg_cost * pos.quantity + trade.price * trade.quantity;
                                pos.quantity += trade.quantity;
                                pos.avg_cost = if pos.quantity > 0.0 {
                                    total_cost / pos.quantity
                                } else {
                                    0.0
                                };

                                trades.push(trade);
                            }
                        }
                    }
                }
            }

            // 3. Update current prices & portfolio value
            for pos in portfolio.positions.values_mut() {
                if pos.symbol == kline.symbol {
                    pos.current_price = kline.close;
                    pos.unrealized_pnl = (kline.close - pos.avg_cost) * pos.quantity;
                }
            }

            let positions_value: f64 = portfolio
                .positions
                .values()
                .map(|p| p.current_price * p.quantity)
                .sum();
            portfolio.total_value = portfolio.cash + positions_value;

            // 4. Record equity curve
            equity_curve.push((kline.datetime, portfolio.total_value));
        }

        strategy.on_stop();

        let metrics =
            PerformanceMetrics::calculate(&equity_curve, &trades, self.config.initial_capital);

        BacktestResult {
            config: self.config.clone(),
            trades,
            equity_curve,
            metrics,
            final_portfolio: portfolio,
        }
    }
}
