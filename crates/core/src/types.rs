use std::fmt;

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Market {
    SH,
    SZ,
}

impl fmt::Display for Market {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Market::SH => write!(f, "SH"),
            Market::SZ => write!(f, "SZ"),
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
