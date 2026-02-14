/// Paper trading broker for simulation and backtesting.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use quant_core::error::{QuantError, Result};
use quant_core::models::*;
use quant_core::traits::Broker;
use quant_core::types::*;

use crate::orders::OrderStateMachine;

pub struct PaperBroker {
    inner: Mutex<PaperBrokerInner>,
    /// When true, submit_order immediately fills the order (for engine use)
    auto_fill: std::sync::atomic::AtomicBool,
}

struct PaperBrokerInner {
    account: Account,
    pending_orders: Vec<Order>,
    filled_trades: Vec<Trade>,
    commission_rate: f64,
    stamp_tax_rate: f64,
}

impl PaperBroker {
    pub fn new(initial_capital: f64, commission_rate: f64, stamp_tax_rate: f64) -> Self {
        let account = Account {
            id: Uuid::new_v4(),
            name: "Paper Trading".to_string(),
            portfolio: Portfolio {
                positions: HashMap::new(),
                cash: initial_capital,
                total_value: initial_capital,
            },
            initial_capital,
        };
        Self {
            inner: Mutex::new(PaperBrokerInner {
                account,
                pending_orders: Vec::new(),
                filled_trades: Vec::new(),
                commission_rate,
                stamp_tax_rate,
            }),
            auto_fill: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Set whether submit_order should auto-fill (default: true).
    pub fn set_auto_fill(&self, auto_fill: bool) {
        self.auto_fill.store(auto_fill, std::sync::atomic::Ordering::SeqCst);
    }

    /// Simulate filling an order at the given price.
    pub fn fill_order(&self, order_id: Uuid, fill_price: f64) -> std::result::Result<Trade, String> {
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;

        let order_idx = inner
            .pending_orders
            .iter()
            .position(|o| o.id == order_id)
            .ok_or_else(|| format!("Order {} not found in pending orders", order_id))?;

        // Remove the order from pending so we own it outright
        let mut order = inner.pending_orders.remove(order_idx);

        // Transition to Filled
        order.status = OrderStateMachine::transition(&order.status, OrderStatus::Filled)?;
        order.filled_qty = order.quantity;
        order.updated_at = Utc::now().naive_utc();

        let trade_value = fill_price * order.quantity;
        let commission = {
            let c = trade_value * inner.commission_rate;
            if c < 5.0 { 5.0 } else { c }
        };
        let stamp_tax = if order.side == OrderSide::Sell {
            trade_value * inner.stamp_tax_rate
        } else {
            0.0
        };
        let total_cost = commission + stamp_tax;

        let trade = Trade {
            id: Uuid::new_v4(),
            order_id,
            symbol: order.symbol.clone(),
            side: order.side,
            price: fill_price,
            quantity: order.quantity,
            commission: total_cost,
            timestamp: Utc::now().naive_utc(),
        };

        // Update portfolio
        let portfolio = &mut inner.account.portfolio;
        match order.side {
            OrderSide::Buy => {
                portfolio.cash -= trade_value + total_cost;
                let pos = portfolio
                    .positions
                    .entry(order.symbol.clone())
                    .or_insert(Position {
                        symbol: order.symbol.clone(),
                        quantity: 0.0,
                        avg_cost: 0.0,
                        current_price: fill_price,
                        unrealized_pnl: 0.0,
                        realized_pnl: 0.0,
                    });
                let total_qty = pos.quantity + order.quantity;
                pos.avg_cost =
                    (pos.avg_cost * pos.quantity + fill_price * order.quantity) / total_qty;
                pos.quantity = total_qty;
                pos.current_price = fill_price;
            }
            OrderSide::Sell => {
                portfolio.cash += trade_value - total_cost;
                if let Some(pos) = portfolio.positions.get_mut(&order.symbol) {
                    let realized = (fill_price - pos.avg_cost) * order.quantity;
                    pos.realized_pnl += realized;
                    pos.quantity -= order.quantity;
                    pos.current_price = fill_price;
                    if pos.quantity <= 0.0 {
                        portfolio.positions.remove(&order.symbol);
                    }
                }
            }
        }

        // Recalculate total value
        let positions_value: f64 = portfolio
            .positions
            .values()
            .map(|p| p.quantity * p.current_price)
            .sum();
        portfolio.total_value = portfolio.cash + positions_value;

        inner.filled_trades.push(trade.clone());

        Ok(trade)
    }
}

#[async_trait]
impl Broker for PaperBroker {
    async fn submit_order(&self, order: &Order) -> Result<Order> {
        let mut inner = self.inner.lock().map_err(|e| QuantError::BrokerError(e.to_string()))?;

        let mut new_order = order.clone();
        new_order.status =
            OrderStateMachine::transition(&new_order.status, OrderStatus::Submitted)
                .map_err(QuantError::BrokerError)?;
        new_order.updated_at = Utc::now().naive_utc();

        if self.auto_fill.load(std::sync::atomic::Ordering::SeqCst) {
            // Auto-fill: immediately execute at order price
            new_order.status =
                OrderStateMachine::transition(&new_order.status, OrderStatus::Filled)
                    .map_err(QuantError::BrokerError)?;
            new_order.filled_qty = new_order.quantity;

            let fill_price = new_order.price;
            let trade_value = fill_price * new_order.quantity;
            let commission = {
                let c = trade_value * inner.commission_rate;
                if c < 5.0 { 5.0 } else { c }
            };
            let stamp_tax = if new_order.side == OrderSide::Sell {
                trade_value * inner.stamp_tax_rate
            } else {
                0.0
            };
            let total_cost = commission + stamp_tax;

            let trade = Trade {
                id: Uuid::new_v4(),
                order_id: new_order.id,
                symbol: new_order.symbol.clone(),
                side: new_order.side,
                price: fill_price,
                quantity: new_order.quantity,
                commission: total_cost,
                timestamp: Utc::now().naive_utc(),
            };

            // Update portfolio
            let portfolio = &mut inner.account.portfolio;
            match new_order.side {
                OrderSide::Buy => {
                    portfolio.cash -= trade_value + total_cost;
                    let pos = portfolio
                        .positions
                        .entry(new_order.symbol.clone())
                        .or_insert(Position {
                            symbol: new_order.symbol.clone(),
                            quantity: 0.0,
                            avg_cost: 0.0,
                            current_price: fill_price,
                            unrealized_pnl: 0.0,
                            realized_pnl: 0.0,
                        });
                    let total_qty = pos.quantity + new_order.quantity;
                    pos.avg_cost =
                        (pos.avg_cost * pos.quantity + fill_price * new_order.quantity) / total_qty;
                    pos.quantity = total_qty;
                    pos.current_price = fill_price;
                }
                OrderSide::Sell => {
                    portfolio.cash += trade_value - total_cost;
                    if let Some(pos) = portfolio.positions.get_mut(&new_order.symbol) {
                        let realized = (fill_price - pos.avg_cost) * new_order.quantity;
                        pos.realized_pnl += realized;
                        pos.quantity -= new_order.quantity;
                        pos.current_price = fill_price;
                        if pos.quantity <= 0.0 {
                            portfolio.positions.remove(&new_order.symbol);
                        }
                    }
                }
            }

            // Recalculate total value
            let positions_value: f64 = portfolio
                .positions
                .values()
                .map(|p| p.quantity * p.current_price)
                .sum();
            portfolio.total_value = portfolio.cash + positions_value;

            inner.filled_trades.push(trade);
        } else {
            inner.pending_orders.push(new_order.clone());
        }

        Ok(new_order)
    }

    async fn cancel_order(&self, order_id: Uuid) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|e| QuantError::BrokerError(e.to_string()))?;

        let order = inner
            .pending_orders
            .iter_mut()
            .find(|o| o.id == order_id)
            .ok_or_else(|| {
                QuantError::BrokerError(format!("Order {} not found", order_id))
            })?;

        order.status = OrderStateMachine::transition(&order.status, OrderStatus::Cancelled)
            .map_err(QuantError::BrokerError)?;
        order.updated_at = Utc::now().naive_utc();

        Ok(())
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        let inner = self.inner.lock().map_err(|e| QuantError::BrokerError(e.to_string()))?;
        Ok(inner.account.portfolio.positions.values().cloned().collect())
    }

    async fn get_account(&self) -> Result<Account> {
        let inner = self.inner.lock().map_err(|e| QuantError::BrokerError(e.to_string()))?;
        Ok(inner.account.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_submit_and_fill() {
        let broker = PaperBroker::new(1_000_000.0, 0.0003, 0.001);

        let order = Order {
            id: Uuid::new_v4(),
            symbol: "600519.SH".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            price: 1800.0,
            quantity: 100.0,
            filled_qty: 0.0,
            status: OrderStatus::Pending,
            created_at: Utc::now().naive_utc(),
            updated_at: Utc::now().naive_utc(),
        };

        // With auto_fill=true (default), submit_order fills immediately
        let submitted = broker.submit_order(&order).await.unwrap();
        assert_eq!(submitted.status, OrderStatus::Filled);
        assert_eq!(submitted.filled_qty, 100.0);

        let account = broker.get_account().await.unwrap();
        assert!(account.portfolio.cash < 1_000_000.0);
        assert!(account.portfolio.positions.contains_key("600519.SH"));
    }

    #[tokio::test]
    async fn test_manual_fill() {
        let broker = PaperBroker::new(1_000_000.0, 0.0003, 0.001);
        broker.set_auto_fill(false);

        let order = Order {
            id: Uuid::new_v4(),
            symbol: "600519.SH".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            price: 1800.0,
            quantity: 100.0,
            filled_qty: 0.0,
            status: OrderStatus::Pending,
            created_at: Utc::now().naive_utc(),
            updated_at: Utc::now().naive_utc(),
        };

        let submitted = broker.submit_order(&order).await.unwrap();
        assert_eq!(submitted.status, OrderStatus::Submitted);

        let trade = broker.fill_order(submitted.id, 1800.0).unwrap();
        assert_eq!(trade.quantity, 100.0);

        let account = broker.get_account().await.unwrap();
        assert!(account.portfolio.cash < 1_000_000.0);
        assert!(account.portfolio.positions.contains_key("600519.SH"));
    }

    #[tokio::test]
    async fn test_cancel_order() {
        let broker = PaperBroker::new(1_000_000.0, 0.0003, 0.001);
        broker.set_auto_fill(false);

        let order = Order {
            id: Uuid::new_v4(),
            symbol: "000001.SZ".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            price: 15.0,
            quantity: 1000.0,
            filled_qty: 0.0,
            status: OrderStatus::Pending,
            created_at: Utc::now().naive_utc(),
            updated_at: Utc::now().naive_utc(),
        };

        let submitted = broker.submit_order(&order).await.unwrap();
        let result = broker.cancel_order(submitted.id).await;
        assert!(result.is_ok());
    }
}
