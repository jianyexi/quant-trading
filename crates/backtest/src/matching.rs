use quant_core::models::{Kline, Order, Trade};
use quant_core::types::{OrderSide, OrderType};
use uuid::Uuid;

pub struct MatchingEngine {
    pub slippage_ticks: u32,
    pub tick_size: f64,
}

impl MatchingEngine {
    pub fn new(slippage_ticks: u32) -> Self {
        Self {
            slippage_ticks,
            tick_size: 0.01,
        }
    }

    /// Try to match an order against a kline bar.
    /// Returns a Trade if the order can be filled.
    pub fn try_match(&self, order: &Order, kline: &Kline) -> Option<Trade> {
        let slippage = self.slippage_ticks as f64 * self.tick_size;

        let fill_price = match order.order_type {
            OrderType::Market => match order.side {
                OrderSide::Buy => kline.open + slippage,
                OrderSide::Sell => kline.open - slippage,
            },
            OrderType::Limit => match order.side {
                OrderSide::Buy => {
                    if order.price >= kline.low {
                        order.price.min(kline.open + slippage)
                    } else {
                        return None;
                    }
                }
                OrderSide::Sell => {
                    if order.price <= kline.high {
                        order.price.max(kline.open - slippage)
                    } else {
                        return None;
                    }
                }
            },
        };

        Some(Trade {
            id: Uuid::new_v4(),
            order_id: order.id,
            symbol: order.symbol.clone(),
            side: order.side,
            price: fill_price,
            quantity: order.quantity,
            commission: 0.0, // calculated by the backtest engine
            timestamp: kline.datetime,
        })
    }
}
