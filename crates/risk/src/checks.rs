/// Pre-trade risk checks for order validation.

use quant_core::models::{Order, Portfolio};
use quant_core::types::OrderSide;

pub struct RiskChecker {
    /// Maximum percentage of portfolio in a single stock (e.g. 0.25 = 25%).
    pub max_concentration: f64,
    /// Maximum daily loss as a fraction of portfolio (e.g. 0.05 = 5%).
    pub max_daily_loss: f64,
    /// Maximum drawdown from peak as a fraction (e.g. 0.10 = 10%).
    pub max_drawdown: f64,
}

impl RiskChecker {
    pub fn new(max_concentration: f64, max_daily_loss: f64, max_drawdown: f64) -> Self {
        Self {
            max_concentration,
            max_daily_loss,
            max_drawdown,
        }
    }

    /// Run all pre-trade checks for an order against the current portfolio.
    pub fn check_order(&self, order: &Order, portfolio: &Portfolio) -> Result<(), String> {
        let order_value = order.price * order.quantity;
        let total_value = portfolio.total_value;

        if total_value <= 0.0 {
            return Err("Portfolio value is zero or negative".to_string());
        }

        // Only check concentration for buy orders
        if order.side == OrderSide::Buy {
            self.check_concentration(&order.symbol, order_value, total_value)?;

            // Check sufficient cash
            if order_value > portfolio.cash {
                return Err(format!(
                    "Insufficient cash: need {:.2}, available {:.2}",
                    order_value, portfolio.cash
                ));
            }
        }

        // For sell orders, check that we hold enough shares
        if order.side == OrderSide::Sell {
            if let Some(pos) = portfolio.positions.get(&order.symbol) {
                if order.quantity > pos.quantity {
                    return Err(format!(
                        "Insufficient position: want to sell {}, hold {}",
                        order.quantity, pos.quantity
                    ));
                }
            } else {
                return Err(format!("No position in {}", order.symbol));
            }
        }

        Ok(())
    }

    /// Check that a single position does not exceed the concentration limit.
    pub fn check_concentration(
        &self,
        symbol: &str,
        order_value: f64,
        total_value: f64,
    ) -> Result<(), String> {
        if total_value <= 0.0 {
            return Err("Portfolio total value is zero or negative".to_string());
        }
        let concentration = order_value / total_value;
        if concentration > self.max_concentration {
            return Err(format!(
                "Concentration limit exceeded for {}: {:.2}% > {:.2}%",
                symbol,
                concentration * 100.0,
                self.max_concentration * 100.0,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quant_core::types::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn make_portfolio(cash: f64, total_value: f64) -> Portfolio {
        Portfolio {
            positions: HashMap::new(),
            cash,
            total_value,
        }
    }

    fn make_order(side: OrderSide, price: f64, quantity: f64) -> Order {
        Order {
            id: Uuid::new_v4(),
            symbol: "600519.SH".to_string(),
            side,
            order_type: OrderType::Limit,
            price,
            quantity,
            filled_qty: 0.0,
            status: OrderStatus::Pending,
            created_at: chrono::Utc::now().naive_utc(),
            updated_at: chrono::Utc::now().naive_utc(),
        }
    }

    #[test]
    fn test_check_order_buy_pass() {
        let checker = RiskChecker::new(0.25, 0.05, 0.10);
        let order = make_order(OrderSide::Buy, 100.0, 100.0); // 10000 value
        let portfolio = make_portfolio(50000.0, 100000.0);
        assert!(checker.check_order(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_check_order_insufficient_cash() {
        let checker = RiskChecker::new(0.25, 0.05, 0.10);
        let order = make_order(OrderSide::Buy, 100.0, 100.0); // 10000 value
        let portfolio = make_portfolio(5000.0, 100000.0);
        assert!(checker.check_order(&order, &portfolio).is_err());
    }

    #[test]
    fn test_concentration_exceeded() {
        let checker = RiskChecker::new(0.10, 0.05, 0.10);
        let result = checker.check_concentration("600519.SH", 20000.0, 100000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sell_no_position() {
        let checker = RiskChecker::new(0.25, 0.05, 0.10);
        let order = make_order(OrderSide::Sell, 100.0, 100.0);
        let portfolio = make_portfolio(50000.0, 100000.0);
        assert!(checker.check_order(&order, &portfolio).is_err());
    }
}
