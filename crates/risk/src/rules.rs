/// Chinese A-share market rules and fee calculations.

use chrono::NaiveDate;

pub struct ChineseMarketRules;

impl ChineseMarketRules {
    /// Check T+1 rule: returns true if the sell is allowed (buy_date < sell_date).
    pub fn check_t_plus_1(buy_date: NaiveDate, sell_date: NaiveDate) -> bool {
        sell_date > buy_date
    }

    /// Get price limit bounds for a stock.
    /// Main board: ±10%, ChiNext/STAR Market: ±20%.
    /// Returns (lower_limit, upper_limit).
    pub fn get_price_limit(prev_close: f64, is_chinext_or_star: bool) -> (f64, f64) {
        let pct = if is_chinext_or_star { 0.20 } else { 0.10 };
        let lower = (prev_close * (1.0 - pct) * 100.0).round() / 100.0;
        let upper = (prev_close * (1.0 + pct) * 100.0).round() / 100.0;
        (lower, upper)
    }

    /// Check if a price is within the daily price limit.
    pub fn is_price_valid(price: f64, prev_close: f64, is_chinext_or_star: bool) -> bool {
        let (lower, upper) = Self::get_price_limit(prev_close, is_chinext_or_star);
        price >= lower && price <= upper
    }

    /// Calculate stamp tax (sell side only, 0.1%).
    pub fn stamp_tax(amount: f64) -> f64 {
        (amount * 0.001 * 100.0).round() / 100.0
    }

    /// Calculate broker commission. Minimum ¥5 per trade.
    pub fn commission(amount: f64, rate: f64) -> f64 {
        let fee = amount * rate;
        if fee < 5.0 { 5.0 } else { (fee * 100.0).round() / 100.0 }
    }

    /// Round quantity down to the nearest lot (100 shares).
    pub fn round_to_lot(quantity: f64) -> f64 {
        (quantity / 100.0).floor() * 100.0
    }

    /// Detect if a symbol is on ChiNext (创业板 300xxx) or STAR Market (科创板 688xxx).
    pub fn is_chinext_or_star(symbol: &str) -> bool {
        let code = symbol.split('.').next().unwrap_or(symbol);
        code.starts_with("300") || code.starts_with("688")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_plus_1() {
        let buy = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let same_day = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let next_day = NaiveDate::from_ymd_opt(2024, 1, 16).unwrap();

        assert!(!ChineseMarketRules::check_t_plus_1(buy, same_day));
        assert!(ChineseMarketRules::check_t_plus_1(buy, next_day));
    }

    #[test]
    fn test_price_limits_main_board() {
        let (lower, upper) = ChineseMarketRules::get_price_limit(10.0, false);
        assert_eq!(lower, 9.0);
        assert_eq!(upper, 11.0);
    }

    #[test]
    fn test_price_limits_chinext() {
        let (lower, upper) = ChineseMarketRules::get_price_limit(10.0, true);
        assert_eq!(lower, 8.0);
        assert_eq!(upper, 12.0);
    }

    #[test]
    fn test_price_valid() {
        assert!(ChineseMarketRules::is_price_valid(10.5, 10.0, false));
        assert!(!ChineseMarketRules::is_price_valid(12.0, 10.0, false));
    }

    #[test]
    fn test_stamp_tax() {
        assert_eq!(ChineseMarketRules::stamp_tax(100000.0), 100.0);
    }

    #[test]
    fn test_commission_minimum() {
        // Very small trade should still pay ¥5 minimum
        assert_eq!(ChineseMarketRules::commission(100.0, 0.0003), 5.0);
    }

    #[test]
    fn test_round_to_lot() {
        assert_eq!(ChineseMarketRules::round_to_lot(350.0), 300.0);
        assert_eq!(ChineseMarketRules::round_to_lot(99.0), 0.0);
    }
}
