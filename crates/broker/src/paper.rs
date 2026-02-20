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
    /// Slippage in basis points (e.g., 5 = 0.05%), applied as adverse price impact
    slippage_bps: std::sync::atomic::AtomicU32,
}

struct PaperBrokerInner {
    account: Account,
    pending_orders: Vec<Order>,
    filled_trades: Vec<Trade>,
    closed_positions: Vec<ClosedPosition>,
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
                closed_positions: Vec::new(),
                commission_rate,
                stamp_tax_rate,
            }),
            auto_fill: std::sync::atomic::AtomicBool::new(true),
            slippage_bps: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Set slippage in basis points (1 bp = 0.01%). Buy fills higher, sell fills lower.
    pub fn set_slippage_bps(&self, bps: u32) {
        self.slippage_bps.store(bps, std::sync::atomic::Ordering::SeqCst);
    }

    /// Apply slippage: buy at higher price, sell at lower price
    fn apply_slippage(&self, price: f64, side: OrderSide) -> f64 {
        let bps = self.slippage_bps.load(std::sync::atomic::Ordering::SeqCst) as f64;
        if bps == 0.0 { return price; }
        let factor = bps / 10_000.0;
        match side {
            OrderSide::Buy => price * (1.0 + factor),
            OrderSide::Sell => price * (1.0 - factor),
        }
    }

    /// Set whether submit_order should auto-fill (default: true).
    pub fn set_auto_fill(&self, auto_fill: bool) {
        self.auto_fill.store(auto_fill, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get archived closed positions.
    pub fn closed_positions(&self) -> Vec<ClosedPosition> {
        self.inner.lock().map(|inner| inner.closed_positions.clone()).unwrap_or_default()
    }

    /// Force close a position at current price. Returns the realized PnL.
    pub fn force_close_position(&self, symbol: &str, price: f64) -> std::result::Result<ClosedPosition, String> {
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        let commission_rate = inner.commission_rate;
        let stamp_tax_rate = inner.stamp_tax_rate;

        let pos = inner.account.portfolio.positions.get(symbol)
            .ok_or_else(|| format!("No position in {}", symbol))?
            .clone();

        let trade_value = price * pos.quantity;
        let commission = (trade_value * commission_rate).max(5.0);
        let stamp_tax = trade_value * stamp_tax_rate;
        let total_cost = commission + stamp_tax;

        let realized = (price - pos.avg_cost) * pos.quantity;
        inner.account.portfolio.cash += trade_value - total_cost;
        inner.account.portfolio.positions.remove(symbol);

        let positions_value: f64 = inner.account.portfolio.positions.values()
            .map(|p| p.quantity * p.current_price).sum();
        inner.account.portfolio.total_value = inner.account.portfolio.cash + positions_value;

        let now = Utc::now().naive_utc();
        let closed = ClosedPosition {
            symbol: symbol.to_string(),
            entry_time: pos.entry_time,
            exit_time: now,
            entry_price: pos.avg_cost,
            exit_price: price,
            quantity: pos.quantity,
            realized_pnl: realized,
            holding_days: (now - pos.entry_time).num_days(),
        };
        inner.closed_positions.push(closed.clone());
        Ok(closed)
    }

    /// Simulate filling an order at the given price.
    pub fn fill_order(&self, order_id: Uuid, fill_price: f64) -> std::result::Result<Trade, String> {
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;

        let order_idx = inner
            .pending_orders
            .iter()
            .position(|o| o.id == order_id)
            .ok_or_else(|| format!("Order {} not found in pending orders", order_id))?;

        let mut order = inner.pending_orders.remove(order_idx);
        order.status = OrderStateMachine::transition(&order.status, OrderStatus::Filled)?;
        order.filled_qty = order.quantity;
        order.updated_at = Utc::now().naive_utc();

        let commission_rate = inner.commission_rate;
        let stamp_tax_rate = inner.stamp_tax_rate;
        let trade_value = fill_price * order.quantity;
        let commission = (trade_value * commission_rate).max(5.0);
        let stamp_tax = if order.side == OrderSide::Sell { trade_value * stamp_tax_rate } else { 0.0 };
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

        let mut should_archive = None;
        match order.side {
            OrderSide::Buy => {
                inner.account.portfolio.cash -= trade_value + total_cost;
                let pos = inner.account.portfolio
                    .positions
                    .entry(order.symbol.clone())
                    .or_insert(Position {
                        symbol: order.symbol.clone(),
                        quantity: 0.0,
                        avg_cost: 0.0,
                        current_price: fill_price,
                        unrealized_pnl: 0.0,
                        realized_pnl: 0.0,
                        entry_time: Utc::now().naive_utc(),
                        scale_level: 0,
                        target_weight: 0.0,
                    });
                let total_qty = pos.quantity + order.quantity;
                pos.avg_cost =
                    (pos.avg_cost * pos.quantity + fill_price * order.quantity) / total_qty;
                pos.quantity = total_qty;
                pos.current_price = fill_price;
                pos.scale_level += 1;
            }
            OrderSide::Sell => {
                inner.account.portfolio.cash += trade_value - total_cost;
                if let Some(pos) = inner.account.portfolio.positions.get_mut(&order.symbol) {
                    let realized = (fill_price - pos.avg_cost) * order.quantity;
                    pos.realized_pnl += realized;
                    pos.quantity -= order.quantity;
                    pos.current_price = fill_price;
                    if pos.quantity <= 0.0 {
                        let now = Utc::now().naive_utc();
                        should_archive = Some(ClosedPosition {
                            symbol: order.symbol.clone(),
                            entry_time: pos.entry_time,
                            exit_time: now,
                            entry_price: pos.avg_cost,
                            exit_price: fill_price,
                            quantity: order.quantity,
                            realized_pnl: pos.realized_pnl,
                            holding_days: (now - pos.entry_time).num_days(),
                        });
                    }
                }
                if should_archive.is_some() {
                    inner.account.portfolio.positions.remove(&order.symbol);
                }
            }
        }

        let positions_value: f64 = inner.account.portfolio.positions.values()
            .map(|p| p.quantity * p.current_price).sum();
        inner.account.portfolio.total_value = inner.account.portfolio.cash + positions_value;

        if let Some(closed) = should_archive {
            inner.closed_positions.push(closed);
        }
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

            let fill_price = self.apply_slippage(new_order.price, new_order.side);
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
            let mut should_archive = None;
            match new_order.side {
                OrderSide::Buy => {
                    inner.account.portfolio.cash -= trade_value + total_cost;
                    let pos = inner.account.portfolio
                        .positions
                        .entry(new_order.symbol.clone())
                        .or_insert(Position {
                            symbol: new_order.symbol.clone(),
                            quantity: 0.0,
                            avg_cost: 0.0,
                            current_price: fill_price,
                            unrealized_pnl: 0.0,
                            realized_pnl: 0.0,
                            entry_time: Utc::now().naive_utc(),
                            scale_level: 0,
                            target_weight: 0.0,
                        });
                    let total_qty = pos.quantity + new_order.quantity;
                    pos.avg_cost =
                        (pos.avg_cost * pos.quantity + fill_price * new_order.quantity) / total_qty;
                    pos.quantity = total_qty;
                    pos.current_price = fill_price;
                    pos.scale_level += 1;
                }
                OrderSide::Sell => {
                    inner.account.portfolio.cash += trade_value - total_cost;
                    if let Some(pos) = inner.account.portfolio.positions.get_mut(&new_order.symbol) {
                        let realized = (fill_price - pos.avg_cost) * new_order.quantity;
                        pos.realized_pnl += realized;
                        pos.quantity -= new_order.quantity;
                        pos.current_price = fill_price;
                        if pos.quantity <= 0.0 {
                            let now = Utc::now().naive_utc();
                            should_archive = Some(ClosedPosition {
                                symbol: new_order.symbol.clone(),
                                entry_time: pos.entry_time,
                                exit_time: now,
                                entry_price: pos.avg_cost,
                                exit_price: fill_price,
                                quantity: new_order.quantity,
                                realized_pnl: pos.realized_pnl,
                                holding_days: (now - pos.entry_time).num_days(),
                            });
                        }
                    }
                    if should_archive.is_some() {
                        inner.account.portfolio.positions.remove(&new_order.symbol);
                    }
                }
            }

            let positions_value: f64 = inner.account.portfolio.positions.values()
                .map(|p| p.quantity * p.current_price).sum();
            inner.account.portfolio.total_value = inner.account.portfolio.cash + positions_value;

            if let Some(closed) = should_archive {
                inner.closed_positions.push(closed);
            }
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
