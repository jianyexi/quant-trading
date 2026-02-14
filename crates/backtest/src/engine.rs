use std::collections::HashMap;

use chrono::NaiveDateTime;
use serde::Serialize;
use uuid::Uuid;

use quant_core::models::{Kline, Order, Portfolio, Position, Trade};
use quant_core::traits::Strategy;
use quant_core::types::{OrderSide, OrderStatus, OrderType, SignalAction};

use crate::matching::MatchingEngine;
use crate::metrics::PerformanceMetrics;

#[derive(Debug, Clone, Serialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub stamp_tax_rate: f64,
    pub slippage_ticks: u32,
}

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
                            // Use all available cash
                            let affordable =
                                (portfolio.cash / (kline.close * (1.0 + self.config.commission_rate)))
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
                            // Calculate commission
                            let turnover = trade.price * trade.quantity;
                            let mut commission = turnover * self.config.commission_rate;

                            // Stamp tax on sell side only (Chinese A-share market)
                            if trade.side == OrderSide::Sell {
                                commission += turnover * self.config.stamp_tax_rate;
                            }
                            trade.commission = commission;

                            // Update portfolio
                            match trade.side {
                                OrderSide::Buy => {
                                    let cost = turnover + commission;
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
                                        });
                                    let total_cost =
                                        pos.avg_cost * pos.quantity + trade.price * trade.quantity;
                                    pos.quantity += trade.quantity;
                                    pos.avg_cost = if pos.quantity > 0.0 {
                                        total_cost / pos.quantity
                                    } else {
                                        0.0
                                    };
                                }
                                OrderSide::Sell => {
                                    let proceeds = turnover - commission;
                                    portfolio.cash += proceeds;

                                    if let Some(pos) = portfolio.positions.get_mut(&trade.symbol) {
                                        let realized =
                                            (trade.price - pos.avg_cost) * trade.quantity;
                                        pos.realized_pnl += realized;
                                        pos.quantity -= trade.quantity;
                                        if pos.quantity <= 1e-10 {
                                            portfolio.positions.remove(&trade.symbol);
                                        }
                                    }
                                }
                            }

                            trades.push(trade);
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
