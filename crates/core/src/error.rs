use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantError {
    #[error("Data error: {0}")]
    DataError(String),
    #[error("Strategy error: {0}")]
    StrategyError(String),
    #[error("Broker error: {0}")]
    BrokerError(String),
    #[error("Risk error: {0}")]
    RiskError(String),
    #[error("Config error: {0}")]
    ConfigError(String),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("LLM error: {0}")]
    LlmError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
}

pub type Result<T> = std::result::Result<T, QuantError>;

impl From<sqlx::Error> for QuantError {
    fn from(err: sqlx::Error) -> Self {
        QuantError::DatabaseError(err.to_string())
    }
}

impl From<reqwest::Error> for QuantError {
    fn from(err: reqwest::Error) -> Self {
        QuantError::NetworkError(err.to_string())
    }
}

impl From<serde_json::Error> for QuantError {
    fn from(err: serde_json::Error) -> Self {
        QuantError::DataError(err.to_string())
    }
}
