use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    pub tushare: TushareConfig,
    pub akshare: AkshareConfig,
    pub llm: LlmConfig,
    pub trading: TradingConfig,
    pub risk: RiskConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TushareConfig {
    pub token: String,
    pub base_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AkshareConfig {
    pub base_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    pub api_url: String,
    pub api_key: String,
    pub model: String,
    pub temperature: f64,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TradingConfig {
    pub commission_rate: f64,
    pub stamp_tax_rate: f64,
    pub slippage_ticks: u32,
    pub initial_capital: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RiskConfig {
    pub max_concentration: f64,
    pub max_daily_loss: f64,
    pub max_drawdown: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl AppConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        toml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    pub fn from_default() -> Result<Self, ConfigError> {
        Self::from_file("config/default.toml")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
}
