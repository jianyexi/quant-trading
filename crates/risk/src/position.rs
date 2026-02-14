/// Position sizing methods for portfolio allocation.

#[derive(Debug, Clone)]
pub enum PositionSizeMethod {
    /// Allocate a fixed dollar amount per trade.
    FixedAmount(f64),
    /// Allocate a fixed percentage of portfolio value per trade.
    FixedPercentage(f64),
    /// Kelly Criterion based on win rate and win/loss ratio.
    KellyCriterion { win_rate: f64, win_loss_ratio: f64 },
}

/// Calculate the number of shares to buy given a sizing method, portfolio value, and price.
pub fn calculate_position_size(method: &PositionSizeMethod, portfolio_value: f64, price: f64) -> f64 {
    if price <= 0.0 || portfolio_value <= 0.0 {
        return 0.0;
    }

    let allocation = match method {
        PositionSizeMethod::FixedAmount(amount) => *amount,
        PositionSizeMethod::FixedPercentage(pct) => portfolio_value * pct,
        PositionSizeMethod::KellyCriterion { win_rate, win_loss_ratio } => {
            // Kelly fraction: f* = W - (1 - W) / R
            let kelly_fraction = win_rate - (1.0 - win_rate) / win_loss_ratio;
            let kelly_fraction = kelly_fraction.max(0.0).min(1.0);
            portfolio_value * kelly_fraction
        }
    };

    (allocation / price).floor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_amount() {
        let size = calculate_position_size(&PositionSizeMethod::FixedAmount(10000.0), 100000.0, 25.0);
        assert_eq!(size, 400.0);
    }

    #[test]
    fn test_fixed_percentage() {
        let size = calculate_position_size(&PositionSizeMethod::FixedPercentage(0.1), 100000.0, 50.0);
        assert_eq!(size, 200.0);
    }

    #[test]
    fn test_kelly_criterion() {
        let size = calculate_position_size(
            &PositionSizeMethod::KellyCriterion { win_rate: 0.6, win_loss_ratio: 2.0 },
            100000.0,
            10.0,
        );
        // Kelly fraction = 0.6 - 0.4/2.0 = 0.4, allocation = 40000, shares = 4000
        assert_eq!(size, 4000.0);
    }

    #[test]
    fn test_zero_price() {
        let size = calculate_position_size(&PositionSizeMethod::FixedAmount(10000.0), 100000.0, 0.0);
        assert_eq!(size, 0.0);
    }
}
