/// Position sizing methods for portfolio allocation.

#[derive(Debug, Clone)]
pub enum PositionSizeMethod {
    /// Allocate a fixed dollar amount per trade.
    FixedAmount(f64),
    /// Allocate a fixed percentage of portfolio value per trade.
    FixedPercentage(f64),
    /// Kelly Criterion based on win rate and win/loss ratio.
    KellyCriterion { win_rate: f64, win_loss_ratio: f64 },
    /// ATR-based: risk_budget / ATR per share. High-vol stocks get smaller positions.
    AtrBased {
        /// Fraction of portfolio to risk per trade (e.g. 0.01 = 1%)
        risk_per_trade: f64,
        /// ATR value (average true range) of the stock
        atr: f64,
        /// Stop-loss multiplier of ATR (e.g. 2.0 = stop at 2x ATR)
        atr_multiplier: f64,
    },
}

/// Scaling configuration for gradual position building
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// Initial entry fraction (e.g. 0.3 = 30% of target position)
    pub initial_fraction: f64,
    /// Add-on fraction per scale-up (e.g. 0.3 = add 30%)
    pub add_fraction: f64,
    /// Max scale level (e.g. 3 = initial + 2 adds = 100%)
    pub max_level: u32,
    /// Min profit % required to scale up
    pub scale_up_threshold: f64,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            initial_fraction: 0.3,
            add_fraction: 0.35,
            max_level: 3,
            scale_up_threshold: 0.02, // 2% profit to add
        }
    }
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
            let kelly_fraction = win_rate - (1.0 - win_rate) / win_loss_ratio;
            let kelly_fraction = kelly_fraction.max(0.0).min(1.0);
            portfolio_value * kelly_fraction
        }
        PositionSizeMethod::AtrBased { risk_per_trade, atr, atr_multiplier } => {
            if *atr <= 0.0 || *atr_multiplier <= 0.0 {
                return 0.0;
            }
            // Risk budget = portfolio * risk_per_trade
            // Risk per share = ATR * multiplier
            // Shares = risk_budget / risk_per_share
            let risk_budget = portfolio_value * risk_per_trade;
            let risk_per_share = atr * atr_multiplier;
            return (risk_budget / risk_per_share).floor();
        }
    };

    (allocation / price).floor()
}

/// Calculate scaled position size based on current scale level
pub fn calculate_scaled_size(
    full_shares: f64,
    config: &ScalingConfig,
    current_level: u32,
) -> f64 {
    if current_level == 0 {
        // Initial entry
        (full_shares * config.initial_fraction).floor()
    } else if current_level < config.max_level {
        // Scale-up
        (full_shares * config.add_fraction).floor()
    } else {
        0.0 // Already at max
    }
}

/// Check if a position should be scaled up (added to)
pub fn should_scale_up(
    current_price: f64,
    avg_cost: f64,
    current_level: u32,
    config: &ScalingConfig,
) -> bool {
    if current_level >= config.max_level {
        return false;
    }
    let profit_pct = (current_price - avg_cost) / avg_cost;
    profit_pct >= config.scale_up_threshold
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
        assert_eq!(size, 4000.0);
    }

    #[test]
    fn test_zero_price() {
        let size = calculate_position_size(&PositionSizeMethod::FixedAmount(10000.0), 100000.0, 0.0);
        assert_eq!(size, 0.0);
    }

    #[test]
    fn test_atr_based_sizing() {
        // Portfolio 100k, risk 1% per trade = 1000 risk budget
        // ATR = 2.0, multiplier = 2.0 → risk per share = 4.0
        // Shares = 1000 / 4.0 = 250
        let size = calculate_position_size(
            &PositionSizeMethod::AtrBased { risk_per_trade: 0.01, atr: 2.0, atr_multiplier: 2.0 },
            100000.0,
            50.0,
        );
        assert_eq!(size, 250.0);
    }

    #[test]
    fn test_atr_high_volatility_smaller_position() {
        // Same setup but ATR = 5.0 → risk per share = 10.0 → shares = 100
        let size = calculate_position_size(
            &PositionSizeMethod::AtrBased { risk_per_trade: 0.01, atr: 5.0, atr_multiplier: 2.0 },
            100000.0,
            50.0,
        );
        assert_eq!(size, 100.0);
    }

    #[test]
    fn test_scaling_initial_entry() {
        let config = ScalingConfig::default();
        let full = 1000.0;
        let initial = calculate_scaled_size(full, &config, 0);
        assert_eq!(initial, 300.0); // 30% of 1000
    }

    #[test]
    fn test_scaling_add_on() {
        let config = ScalingConfig::default();
        let full = 1000.0;
        let add = calculate_scaled_size(full, &config, 1);
        assert_eq!(add, 350.0); // 35% of 1000
    }

    #[test]
    fn test_scaling_max_level() {
        let config = ScalingConfig::default();
        let add = calculate_scaled_size(1000.0, &config, 3);
        assert_eq!(add, 0.0); // at max
    }

    #[test]
    fn test_should_scale_up() {
        let config = ScalingConfig::default();
        assert!(should_scale_up(102.0, 100.0, 1, &config)); // +2% >= threshold
        assert!(!should_scale_up(101.0, 100.0, 1, &config)); // +1% < threshold
        assert!(!should_scale_up(105.0, 100.0, 3, &config)); // at max level
    }
}
