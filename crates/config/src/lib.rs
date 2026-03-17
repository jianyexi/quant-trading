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
    pub qmt: QmtConfig,
    pub server: ServerConfig,
    #[serde(default)]
    pub sentiment: SentimentConfig,
    #[serde(default)]
    pub data_source: DataSourceConfig,
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
    #[serde(default)]
    pub fees: std::collections::HashMap<String, MarketFees>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarketFees {
    pub commission_rate: f64,
    pub stamp_tax_rate: f64,
    #[serde(default = "default_slippage")]
    pub slippage_ticks: u32,
}

fn default_slippage() -> u32 { 1 }

impl TradingConfig {
    /// Get fees for a specific market region (CN/US/HK), falling back to global defaults.
    pub fn fees_for_market(&self, market: &str) -> MarketFees {
        self.fees.get(market).cloned().unwrap_or(MarketFees {
            commission_rate: self.commission_rate,
            stamp_tax_rate: self.stamp_tax_rate,
            slippage_ticks: self.slippage_ticks,
        })
    }

    /// Detect market from symbol and return appropriate fees.
    pub fn fees_for_symbol(&self, symbol: &str) -> MarketFees {
        let market = detect_market_region(symbol);
        self.fees_for_market(market)
    }
}

/// Detect market region from symbol string.
pub fn detect_market_region(symbol: &str) -> &'static str {
    let s = symbol.to_uppercase();
    if s.ends_with(".SH") || s.ends_with(".SZ") { return "CN"; }
    if s.ends_with(".HK") { return "HK"; }
    if s.chars().all(|c| c.is_ascii_digit()) && s.len() == 6 { return "CN"; }
    "US"
}

#[derive(Debug, Clone, Deserialize)]
pub struct RiskConfig {
    pub max_concentration: f64,
    pub max_daily_loss: f64,
    pub max_drawdown: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QmtConfig {
    pub bridge_url: String,
    pub account: String,
    pub qmt_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SentimentConfig {
    /// Interval in seconds between news collection cycles
    #[serde(default = "default_collect_interval")]
    pub collect_interval_secs: u64,
    /// Symbols to track for sentiment (e.g., ["600519.SH", "000858.SZ"])
    #[serde(default)]
    pub watch_symbols: Vec<String>,
    /// Max news items to fetch per symbol per cycle
    #[serde(default = "default_max_news_per_symbol")]
    pub max_news_per_symbol: usize,
    /// Max news items to analyze with LLM per cycle (controls API cost)
    #[serde(default = "default_max_llm_calls")]
    pub max_llm_calls_per_cycle: usize,
    /// News sources to use
    #[serde(default = "default_news_sources")]
    pub news_sources: Vec<String>,
}

fn default_collect_interval() -> u64 { 3600 }
fn default_max_news_per_symbol() -> usize { 10 }
fn default_max_llm_calls() -> usize { 50 }
fn default_news_sources() -> Vec<String> { vec!["eastmoney".to_string(), "sina".to_string()] }

fn default_cn_providers() -> Vec<String> { vec!["tushare".into(), "akshare".into()] }
fn default_us_providers() -> Vec<String> { vec!["yfinance".into()] }
fn default_hk_providers() -> Vec<String> { vec!["yfinance".into()] }

#[derive(Debug, Clone, Deserialize)]
pub struct DataSourceConfig {
    /// Provider priority for CN stocks (first available wins)
    #[serde(default = "default_cn_providers")]
    pub cn_providers: Vec<String>,
    /// Provider priority for US stocks
    #[serde(default = "default_us_providers")]
    pub us_providers: Vec<String>,
    /// Provider priority for HK stocks
    #[serde(default = "default_hk_providers")]
    pub hk_providers: Vec<String>,
    /// If true, never fetch from network
    #[serde(default)]
    pub cache_only: bool,
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self {
            cn_providers: default_cn_providers(),
            us_providers: default_us_providers(),
            hk_providers: default_hk_providers(),
            cache_only: false,
        }
    }
}

impl DataSourceConfig {
    /// Get provider list for a given market region.
    pub fn providers_for_market(&self, market: &str) -> &[String] {
        match market {
            "CN" => &self.cn_providers,
            "US" => &self.us_providers,
            "HK" => &self.hk_providers,
            _ => &self.cn_providers,
        }
    }
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            collect_interval_secs: default_collect_interval(),
            watch_symbols: vec![],
            max_news_per_symbol: default_max_news_per_symbol(),
            max_llm_calls_per_cycle: default_max_llm_calls(),
            news_sources: default_news_sources(),
        }
    }
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
