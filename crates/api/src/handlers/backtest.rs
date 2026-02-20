use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde_json::{json, Value};

use crate::state::AppState;
use super::BacktestRequest;
use super::market::{fetch_real_klines_with_period, generate_backtest_klines};

pub async fn run_backtest(
    State(_state): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> (StatusCode, Json<Value>) {
    use quant_backtest::engine::{BacktestConfig, BacktestEngine};
    use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum, MultiFactorStrategy, MultiFactorConfig};
    use quant_strategy::ml_factor::{MlFactorStrategy, MlFactorConfig};

    let capital = req.capital.unwrap_or(1_000_000.0);
    let period = req.period.as_deref().unwrap_or("daily");

    let (klines, data_source) = match fetch_real_klines_with_period(&req.symbol, &req.start, &req.end, period) {
        Ok(k) if !k.is_empty() => {
            let n = k.len();
            let label = if period == "daily" { "日线" } else { &format!("{}分钟线", period) };
            (k, format!("akshare ({}条真实{})", n, label))
        }
        Ok(_) | Err(_) if period != "daily" => {
            return (StatusCode::BAD_REQUEST, Json(json!({
                "error": format!("无法获取{}分钟级数据。分钟K线仅支持近5个交易日，请缩短日期范围或使用日线(daily)。", period)
            })));
        }
        Ok(_) => {
            let k = generate_backtest_klines(&req.symbol, &req.start, &req.end);
            (k, "synthetic (akshare返回空数据)".to_string())
        }
        Err(reason) => {
            let k = generate_backtest_klines(&req.symbol, &req.start, &req.end);
            (k, format!("synthetic ({})", reason))
        }
    };

    if klines.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "No kline data for date range"})));
    }

    let bt_config = BacktestConfig {
        initial_capital: capital,
        commission_rate: 0.001,
        stamp_tax_rate: 0.001,
        slippage_ticks: 1,
    };

    let engine = BacktestEngine::new(bt_config);

    let mut strategy: Box<dyn quant_core::traits::Strategy> = match req.strategy.as_str() {
        "sma_cross" | "DualMaCrossover" => Box::new(DualMaCrossover::new(5, 20)),
        "rsi_reversal" | "RsiMeanReversion" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
        "macd_trend" | "MacdMomentum" => Box::new(MacdMomentum::new(12, 26, 9)),
        "multi_factor" | "MultiFactorModel" => Box::new(MultiFactorStrategy::new(MultiFactorConfig::default())),
        "ml_factor" | "MlFactor" => {
            let mode = req.inference_mode.as_deref().unwrap_or("embedded");
            let ml_cfg = MlFactorConfig {
                inference_mode: quant_strategy::ml_factor::MlInferenceMode::from_str(mode),
                ..Default::default()
            };
            Box::new(MlFactorStrategy::new(ml_cfg))
        }
        _ => Box::new(DualMaCrossover::new(5, 20)),
    };

    let result = engine.run(strategy.as_mut(), &klines);

    let eq_fmt = if period == "daily" { "%Y-%m-%d" } else { "%Y-%m-%d %H:%M" };
    let equity_curve: Vec<Value> = result.equity_curve.iter().map(|(dt, val)| {
        json!({ "date": dt.format(eq_fmt).to_string(), "value": (*val * 100.0).round() / 100.0 })
    }).collect();

    let trades: Vec<Value> = result.trades.iter().map(|t| {
        json!({
            "date": t.timestamp.format("%Y-%m-%d %H:%M").to_string(),
            "symbol": t.symbol,
            "side": if t.side == quant_core::types::OrderSide::Buy { "BUY" } else { "SELL" },
            "price": (t.price * 100.0).round() / 100.0,
            "quantity": t.quantity as i64,
            "commission": (t.commission * 100.0).round() / 100.0,
        })
    }).collect();

    let m = &result.metrics;
    let id = format!("bt-{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap());

    (StatusCode::OK, Json(json!({
        "id": id,
        "strategy": req.strategy,
        "symbol": req.symbol,
        "start": req.start,
        "end": req.end,
        "initial_capital": capital,
        "final_value": (result.final_portfolio.total_value * 100.0).round() / 100.0,
        "total_return_percent": (m.total_return * 10000.0).round() / 100.0,
        "annual_return_percent": (m.annual_return * 10000.0).round() / 100.0,
        "sharpe_ratio": (m.sharpe_ratio * 100.0).round() / 100.0,
        "max_drawdown_percent": (m.max_drawdown * 10000.0).round() / 100.0,
        "max_drawdown_duration_days": m.max_drawdown_duration,
        "win_rate_percent": (m.win_rate * 10000.0).round() / 100.0,
        "total_trades": m.total_trades,
        "winning_trades": m.winning_trades,
        "losing_trades": m.losing_trades,
        "profit_factor": (m.profit_factor * 100.0).round() / 100.0,
        "avg_win": (m.avg_win * 100.0).round() / 100.0,
        "avg_loss": (m.avg_loss * 100.0).round() / 100.0,
        "equity_curve": equity_curve,
        "trades": trades,
        "data_source": data_source,
        "period": period,
        "status": "completed"
    })))
}

pub async fn get_backtest_results(
    Path(id): Path<String>,
    State(_state): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "id": id,
        "status": "not_found",
        "error": "Backtest results are not persisted. Please run a new backtest."
    }))
}
