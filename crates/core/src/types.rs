use std::fmt;

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Market {
    SH,
    SZ,
    /// US equities (NASDAQ, NYSE, AMEX)
    US,
    /// Hong Kong (HKEX)
    HK,
}

impl fmt::Display for Market {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Market::SH => write!(f, "SH"),
            Market::SZ => write!(f, "SZ"),
            Market::US => write!(f, "US"),
            Market::HK => write!(f, "HK"),
        }
    }
}

impl Market {
    /// Detect market from symbol string.
    /// - Contains ".HK" → HK
    /// - All alphabetic (or alpha with dots like BRK.B) → US
    /// - Otherwise → CN (SH/SZ based on first digit)
    pub fn from_symbol(symbol: &str) -> Self {
        let upper = symbol.to_uppercase();
        if upper.ends_with(".HK") {
            return Market::HK;
        }
        if upper.ends_with(".SH") {
            return Market::SH;
        }
        if upper.ends_with(".SZ") {
            return Market::SZ;
        }
        if upper.ends_with(".US") {
            return Market::US;
        }
        // Pure alphabetic ticker → US (AAPL, GOOGL, BRK.B)
        let base = upper.split('.').next().unwrap_or(&upper);
        if !base.is_empty() && base.chars().all(|c| c.is_ascii_alphabetic()) {
            return Market::US;
        }
        // Numeric codes → CN
        if base.starts_with(['6', '5', '9']) {
            Market::SH
        } else {
            Market::SZ
        }
    }

    /// Return market region string: "CN", "US", "HK"
    pub fn region(&self) -> &'static str {
        match self {
            Market::SH | Market::SZ => "CN",
            Market::US => "US",
            Market::HK => "HK",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeFrame {
    Min1,
    Min5,
    Min15,
    Min30,
    Hour1,
    Daily,
    Weekly,
    Monthly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalAction {
    Buy,
    Sell,
    Hold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub action: SignalAction,
    pub confidence: f64,
    pub symbol: String,
    pub timestamp: NaiveDateTime,
}

impl Signal {
    pub fn buy(symbol: &str, confidence: f64, timestamp: NaiveDateTime) -> Self {
        Self {
            action: SignalAction::Buy,
            confidence,
            symbol: symbol.to_string(),
            timestamp,
        }
    }

    pub fn sell(symbol: &str, confidence: f64, timestamp: NaiveDateTime) -> Self {
        Self {
            action: SignalAction::Sell,
            confidence,
            symbol: symbol.to_string(),
            timestamp,
        }
    }

    pub fn hold(symbol: &str, timestamp: NaiveDateTime) -> Self {
        Self {
            action: SignalAction::Hold,
            confidence: 0.0,
            symbol: symbol.to_string(),
            timestamp,
        }
    }

    pub fn is_buy(&self) -> bool {
        self.action == SignalAction::Buy
    }

    pub fn is_sell(&self) -> bool {
        self.action == SignalAction::Sell
    }
}
