/// Single source of truth for strategy instantiation.
///
/// Both CLI and API handlers should use `create_strategy()` instead of
/// duplicating the strategy-name → implementation match block.

use quant_core::traits::Strategy;

use serde_json::Value;

use crate::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum, MultiFactorStrategy, MultiFactorConfig, BollingerBandsStrategy};
use crate::composite::CompositeStrategy;
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
    "bollinger_bands",
    "multi_factor",
    "sentiment_aware",
    "ml_factor",
    "llm_signal",
    "composite",
];

/// Create a strategy by name. Returns `Err` if the name is unknown.
///
/// Canonical aliases are supported (e.g. "DualMaCrossover" maps to "sma_cross").
pub fn create_strategy(name: &str, opts: StrategyOptions) -> Result<CreatedStrategy, String> {
    match name {
        "sma_cross" | "dual_ma" | "DualMaCrossover" => Ok(CreatedStrategy {
            strategy: Box::new(DualMaCrossover::new(5, 20)),
            active_inference_mode: String::new(),
        }),
        "rsi_reversal" | "rsi" | "RsiMeanReversion" => Ok(CreatedStrategy {
            strategy: Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
            active_inference_mode: String::new(),
        }),
        "macd_trend" | "macd" | "MacdMomentum" => Ok(CreatedStrategy {
            strategy: Box::new(MacdMomentum::new(12, 26, 9)),
            active_inference_mode: String::new(),
        }),
        "bollinger_bands" | "bollinger" | "BollingerBands" => Ok(CreatedStrategy {
            strategy: Box::new(BollingerBandsStrategy::new(20, 2.0)),
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

    /// Read a float param, trying `key` first.
    fn float_param(params: &Value, key: &str, default: f64) -> f64 {
        params
            .get(key)
            .and_then(|v| v.as_f64())
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

    /// Read a string param.
    fn str_param<'a>(params: &'a Value, key: &str, default: &'a str) -> &'a str {
        params
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or(default)
    }

    match name {
        "sma_cross" | "dual_ma" | "DualMaCrossover" => {
            let fast = int_param(params, "fast_period", 5);
            let slow = int_param(params, "slow_period", 20);
            Ok(CreatedStrategy {
                strategy: Box::new(DualMaCrossover::new(fast, slow)),
                active_inference_mode: String::new(),
            })
        }
        "rsi_reversal" | "rsi" | "RsiMeanReversion" => {
            let period = int_param2(params, "rsi_period", "period", 14);
            let oversold = float_param2(params, "oversold_threshold", "oversold", 30.0);
            let overbought = float_param2(params, "overbought_threshold", "overbought", 70.0);
            Ok(CreatedStrategy {
                strategy: Box::new(RsiMeanReversion::new(period, overbought, oversold)),
                active_inference_mode: String::new(),
            })
        }
        "macd_trend" | "macd" | "MacdMomentum" => {
            let fast = int_param(params, "fast_period", 12);
            let slow = int_param(params, "slow_period", 26);
            let signal = int_param(params, "signal_period", 9);
            Ok(CreatedStrategy {
                strategy: Box::new(MacdMomentum::new(fast, slow, signal)),
                active_inference_mode: String::new(),
            })
        }
        "bollinger_bands" | "bollinger" | "BollingerBands" => {
            let period = int_param(params, "period", 20);
            let std_dev = float_param2(params, "std_dev", "std_deviation", 2.0);
            Ok(CreatedStrategy {
                strategy: Box::new(BollingerBandsStrategy::new(period, std_dev)),
                active_inference_mode: String::new(),
            })
        }
        "multi_factor" | "MultiFactorModel" => {
            let buy_threshold = float_param(params, "buy_threshold", 0.15);
            let sell_threshold = float_param(params, "sell_threshold", -0.15);
            let cfg = MultiFactorConfig {
                buy_threshold,
                sell_threshold,
                ..Default::default()
            };
            Ok(CreatedStrategy {
                strategy: Box::new(MultiFactorStrategy::new(cfg)),
                active_inference_mode: String::new(),
            })
        }
        "sentiment_aware" | "SentimentAware" => {
            let sentiment_weight = float_param(params, "sentiment_weight", 0.2);
            let min_items = int_param(params, "min_items", 3);
            // Build the base strategy (defaults to multi_factor)
            let base_name = str_param(params, "base_strategy", "multi_factor");
            let base = create_strategy_with_params(
                base_name,
                params.get("base_params").unwrap_or(&Value::Object(Default::default())),
            )
            .unwrap_or_else(|_| CreatedStrategy {
                strategy: Box::new(MultiFactorStrategy::with_defaults()),
                active_inference_mode: String::new(),
            });
            let store = SentimentStore::new();
            Ok(CreatedStrategy {
                strategy: Box::new(SentimentAwareStrategy::new(
                    base.strategy,
                    store,
                    sentiment_weight,
                    min_items,
                )),
                active_inference_mode: String::new(),
            })
        }
        "ml_factor" | "MlFactor" => {
            let buy_threshold = float_param(params, "buy_threshold", 0.60);
            let sell_threshold = float_param(params, "sell_threshold", 0.35);
            let inference_mode_str = str_param(params, "inference_mode", "embedded");
            let ml_cfg = MlFactorConfig {
                inference_mode: MlInferenceMode::from_str(inference_mode_str),
                buy_threshold,
                sell_threshold,
                ..Default::default()
            };
            let ml_strategy = MlFactorStrategy::new(ml_cfg);
            let active = format!("{}", ml_strategy.active_mode());
            Ok(CreatedStrategy {
                strategy: Box::new(ml_strategy),
                active_inference_mode: active,
            })
        }
        "composite" | "Composite" => {
            create_composite_from_params(params)
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

/// Build a CompositeStrategy from a JSON config.
///
/// Expected format:
/// ```json
/// {
///   "strategies": [
///     { "name": "sma_cross", "weight": 0.4, "params": { "fast_period": 5 } },
///     { "name": "rsi_reversal", "weight": 0.3 }
///   ],
///   "buy_threshold": 0.3,
///   "sell_threshold": -0.3
/// }
/// ```
pub fn create_composite_from_params(params: &Value) -> Result<CreatedStrategy, String> {
    let strategies_arr = params
        .get("strategies")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "composite requires a 'strategies' array".to_string())?;

    if strategies_arr.is_empty() {
        return Err("composite requires at least one sub-strategy".to_string());
    }

    let mut sub_strategies: Vec<(Box<dyn Strategy>, f64)> = Vec::new();
    for item in strategies_arr {
        let sub_name = item
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "each sub-strategy must have a 'name'".to_string())?;
        let weight = item
            .get("weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let sub_params = item.get("params").cloned().unwrap_or(Value::Object(Default::default()));
        let created = create_strategy_with_params(sub_name, &sub_params)?;
        sub_strategies.push((created.strategy, weight));
    }

    let buy_threshold = params
        .get("buy_threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.3);
    let sell_threshold = params
        .get("sell_threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(-0.3);

    Ok(CreatedStrategy {
        strategy: Box::new(CompositeStrategy::new(sub_strategies, buy_threshold, sell_threshold)),
        active_inference_mode: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn create_all_strategies() {
        for name in STRATEGY_NAMES {
            if *name == "composite" {
                continue; // composite needs params
            }
            let result = create_strategy(name, StrategyOptions::default());
            assert!(result.is_ok(), "Failed to create strategy: {}", name);
        }
    }

    #[test]
    fn create_with_alias() {
        assert!(create_strategy("DualMaCrossover", StrategyOptions::default()).is_ok());
        assert!(create_strategy("dual_ma", StrategyOptions::default()).is_ok());
        assert!(create_strategy("RsiMeanReversion", StrategyOptions::default()).is_ok());
        assert!(create_strategy("rsi", StrategyOptions::default()).is_ok());
        assert!(create_strategy("MacdMomentum", StrategyOptions::default()).is_ok());
        assert!(create_strategy("macd", StrategyOptions::default()).is_ok());
        assert!(create_strategy("BollingerBands", StrategyOptions::default()).is_ok());
        assert!(create_strategy("bollinger", StrategyOptions::default()).is_ok());
    }

    #[test]
    fn unknown_strategy_error() {
        let result = create_strategy("nonexistent", StrategyOptions::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown strategy"));
    }

    #[test]
    fn params_bollinger() {
        let params = json!({"period": 30, "std_dev": 2.5});
        let result = create_strategy_with_params("bollinger_bands", &params);
        assert!(result.is_ok());
    }

    #[test]
    fn params_multi_factor() {
        let params = json!({"buy_threshold": 0.3, "sell_threshold": -0.3});
        let result = create_strategy_with_params("multi_factor", &params);
        assert!(result.is_ok());
    }

    #[test]
    fn params_sentiment_aware() {
        let params = json!({"sentiment_weight": 0.3, "min_items": 5});
        let result = create_strategy_with_params("sentiment_aware", &params);
        assert!(result.is_ok());
    }

    #[test]
    fn params_ml_factor() {
        let params = json!({"buy_threshold": 0.7, "sell_threshold": 0.4, "inference_mode": "embedded"});
        let result = create_strategy_with_params("ml_factor", &params);
        assert!(result.is_ok());
    }

    #[test]
    fn params_composite() {
        let params = json!({
            "strategies": [
                {"name": "sma_cross", "weight": 0.5, "params": {"fast_period": 5, "slow_period": 20}},
                {"name": "rsi_reversal", "weight": 0.5}
            ],
            "buy_threshold": 0.3,
            "sell_threshold": -0.3
        });
        let result = create_strategy_with_params("composite", &params);
        assert!(result.is_ok());
        assert!(result.unwrap().strategy.name().contains("Composite"));
    }
}
