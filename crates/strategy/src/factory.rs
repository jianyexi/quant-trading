/// Single source of truth for strategy instantiation.
///
/// Both CLI and API handlers should use `create_strategy()` instead of
/// duplicating the strategy-name → implementation match block.

use quant_core::traits::Strategy;

use serde_json::Value;

use crate::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum, MultiFactorStrategy, MultiFactorConfig};
use crate::ml_factor::{MlFactorStrategy, MlFactorConfig, MlInferenceMode};
use crate::llm_strategy::{LlmSignalStrategy, LlmSignalConfig};
use crate::sentiment::{SentimentAwareStrategy, SentimentStore};

/// Strategy creation options.
#[derive(Debug, Clone, Default)]
pub struct StrategyOptions {
    /// ML inference mode (e.g. "embedded", "tcp", "onnx"). Only used for ml_factor.
    pub inference_mode: Option<String>,
    /// Pre-existing sentiment store to share. Only used for sentiment_aware.
    pub sentiment_store: Option<SentimentStore>,
}

/// Result of strategy creation, including metadata.
pub struct CreatedStrategy {
    pub strategy: Box<dyn Strategy>,
    /// Active inference mode description (non-empty only for ml_factor).
    pub active_inference_mode: String,
}

impl std::fmt::Debug for CreatedStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreatedStrategy")
            .field("strategy", &self.strategy.name())
            .field("active_inference_mode", &self.active_inference_mode)
            .finish()
    }
}

/// Known strategy names for listing/validation.
pub const STRATEGY_NAMES: &[&str] = &[
    "sma_cross",
    "rsi_reversal",
    "macd_trend",
    "multi_factor",
    "sentiment_aware",
    "ml_factor",
    "llm_signal",
];

/// Create a strategy by name. Returns `Err` if the name is unknown.
///
/// Canonical aliases are supported (e.g. "DualMaCrossover" maps to "sma_cross").
pub fn create_strategy(name: &str, opts: StrategyOptions) -> Result<CreatedStrategy, String> {
    match name {
        "sma_cross" | "DualMaCrossover" => Ok(CreatedStrategy {
            strategy: Box::new(DualMaCrossover::new(5, 20)),
            active_inference_mode: String::new(),
        }),
        "rsi_reversal" | "RsiMeanReversion" => Ok(CreatedStrategy {
            strategy: Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
            active_inference_mode: String::new(),
        }),
        "macd_trend" | "MacdMomentum" => Ok(CreatedStrategy {
            strategy: Box::new(MacdMomentum::new(12, 26, 9)),
            active_inference_mode: String::new(),
        }),
        "multi_factor" | "MultiFactorModel" => Ok(CreatedStrategy {
            strategy: Box::new(MultiFactorStrategy::new(MultiFactorConfig::default())),
            active_inference_mode: String::new(),
        }),
        "sentiment_aware" | "SentimentAware" => {
            let store = opts.sentiment_store.unwrap_or_else(SentimentStore::new);
            Ok(CreatedStrategy {
                strategy: Box::new(SentimentAwareStrategy::with_defaults(
                    Box::new(MultiFactorStrategy::with_defaults()),
                    store,
                )),
                active_inference_mode: String::new(),
            })
        }
        "ml_factor" | "MlFactor" => {
            let mode_str = opts.inference_mode.as_deref().unwrap_or("embedded");
            let ml_cfg = MlFactorConfig {
                inference_mode: MlInferenceMode::from_str(mode_str),
                ..Default::default()
            };
            let ml_strategy = MlFactorStrategy::new(ml_cfg);
            let active = format!("{}", ml_strategy.active_mode());
            Ok(CreatedStrategy {
                strategy: Box::new(ml_strategy),
                active_inference_mode: active,
            })
        }
        "llm_signal" | "LlmSignal" => Ok(CreatedStrategy {
            strategy: Box::new(LlmSignalStrategy::new(LlmSignalConfig::default())),
            active_inference_mode: "llm_http".to_string(),
        }),
        _ => Err(format!(
            "Unknown strategy: '{}'. Available: {}",
            name,
            STRATEGY_NAMES.join(", ")
        )),
    }
}

/// Create a strategy by name with custom parameters from a JSON map.
///
/// This is the entry point for parameter-optimization workflows where each
/// iteration may supply different numeric parameters.  Parameters that are
/// absent or non-numeric fall back to sensible defaults.
///
/// For strategies that do not support custom parameters the call is forwarded
/// to [`create_strategy`] with default options.
pub fn create_strategy_with_params(
    name: &str,
    params: &Value,
) -> Result<CreatedStrategy, String> {
    /// Read an integer param, trying `key` first.
    fn int_param(params: &Value, key: &str, default: usize) -> usize {
        params
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    /// Read an integer param, trying `primary` then `fallback`.
    fn int_param2(params: &Value, primary: &str, fallback: &str, default: usize) -> usize {
        params
            .get(primary)
            .or_else(|| params.get(fallback))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    /// Read a float param, trying `primary` then `fallback`.
    fn float_param2(params: &Value, primary: &str, fallback: &str, default: f64) -> f64 {
        params
            .get(primary)
            .or_else(|| params.get(fallback))
            .and_then(|v| v.as_f64())
            .unwrap_or(default)
    }

    match name {
        "sma_cross" | "DualMaCrossover" => {
            let fast = int_param(params, "fast_period", 5);
            let slow = int_param(params, "slow_period", 20);
            Ok(CreatedStrategy {
                strategy: Box::new(DualMaCrossover::new(fast, slow)),
                active_inference_mode: String::new(),
            })
        }
        "rsi_reversal" | "RsiMeanReversion" => {
            let period = int_param2(params, "rsi_period", "period", 14);
            let oversold = float_param2(params, "oversold_threshold", "oversold", 30.0);
            let overbought = float_param2(params, "overbought_threshold", "overbought", 70.0);
            Ok(CreatedStrategy {
                strategy: Box::new(RsiMeanReversion::new(period, overbought, oversold)),
                active_inference_mode: String::new(),
            })
        }
        "macd_trend" | "MacdMomentum" => {
            let fast = int_param(params, "fast_period", 12);
            let slow = int_param(params, "slow_period", 26);
            let signal = int_param(params, "signal_period", 9);
            Ok(CreatedStrategy {
                strategy: Box::new(MacdMomentum::new(fast, slow, signal)),
                active_inference_mode: String::new(),
            })
        }
        _ => create_strategy(name, StrategyOptions::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_all_strategies() {
        for name in STRATEGY_NAMES {
            let result = create_strategy(name, StrategyOptions::default());
            assert!(result.is_ok(), "Failed to create strategy: {}", name);
        }
    }

    #[test]
    fn create_with_alias() {
        assert!(create_strategy("DualMaCrossover", StrategyOptions::default()).is_ok());
        assert!(create_strategy("RsiMeanReversion", StrategyOptions::default()).is_ok());
        assert!(create_strategy("MacdMomentum", StrategyOptions::default()).is_ok());
    }

    #[test]
    fn unknown_strategy_error() {
        let result = create_strategy("nonexistent", StrategyOptions::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown strategy"));
    }
}
