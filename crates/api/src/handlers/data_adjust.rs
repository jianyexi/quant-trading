use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use chrono::NaiveDateTime;
use serde::Deserialize;
use serde_json::{json, Value};

use quant_core::models::Kline;
use crate::state::AppState;
use super::market::fetch_real_klines_with_period;

// ── Request ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AdjustRequest {
    pub symbols: Vec<String>,
    pub start: String,
    pub end: String,
}

// ── Split event ─────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct SplitEvent {
    pub date: String,
    pub ratio: f64,
    pub split_type: String,
}

// ── Detection ───────────────────────────────────────────────────────

/// Detect probable stock splits from consecutive kline price/volume changes.
///
/// A split is flagged when the close-to-close ratio is extreme AND either
/// the inverse ratio is a clean integer (2:1, 3:1, 10:1 …) or there is a
/// volume spike that typically accompanies a split.
fn detect_splits(klines: &[Kline]) -> Vec<SplitEvent> {
    let mut events = Vec::new();
    if klines.len() < 2 {
        return events;
    }
    for i in 1..klines.len() {
        let prev_close = klines[i - 1].close;
        if prev_close == 0.0 {
            continue;
        }
        let ratio = klines[i].close / prev_close;
        if ratio < 0.65 || ratio > 1.55 {
            let inv_ratio = if ratio < 1.0 { 1.0 / ratio } else { ratio };
            let is_clean = (inv_ratio - inv_ratio.round()).abs() < 0.05;

            let prev_vol = klines[i - 1].volume;
            let vol_spike = prev_vol > 0.0 && klines[i].volume > prev_vol * 1.5;

            if is_clean || vol_spike {
                events.push(SplitEvent {
                    date: klines[i].datetime.format("%Y-%m-%d").to_string(),
                    ratio: if ratio < 1.0 {
                        (1.0 / ratio).round()
                    } else {
                        ratio.round()
                    },
                    split_type: if ratio < 1.0 {
                        "forward_split".to_string()
                    } else {
                        "reverse_split".to_string()
                    },
                });
            }
        }
    }
    events
}

// ── Adjustment ──────────────────────────────────────────────────────

/// Apply backward price adjustment: for each detected split, scale all
/// bars **before** the split date so the series becomes continuous.
fn adjust_prices(klines: &mut [Kline], splits: &[SplitEvent]) {
    for split in splits.iter().rev() {
        let split_dt = match NaiveDateTime::parse_from_str(
            &format!("{} 00:00:00", split.date),
            "%Y-%m-%d %H:%M:%S",
        ) {
            Ok(dt) => dt,
            Err(_) => continue,
        };

        let factor = if split.split_type == "forward_split" {
            1.0 / split.ratio
        } else {
            split.ratio
        };

        for kline in klines.iter_mut() {
            if kline.datetime < split_dt {
                kline.open *= factor;
                kline.high *= factor;
                kline.low *= factor;
                kline.close *= factor;
                // Volume scales inversely to price
                kline.volume *= split.ratio;
            }
        }
    }
}

// ── Public helper for backtest integration ──────────────────────────

/// Detect splits in kline data and apply backward adjustment in-place.
/// Returns the list of detected split events.
pub fn detect_and_adjust(klines: &mut Vec<Kline>) -> Vec<SplitEvent> {
    let splits = detect_splits(klines);
    if !splits.is_empty() {
        adjust_prices(klines, &splits);
    }
    splits
}

// ── Handler ─────────────────────────────────────────────────────────

pub async fn data_adjust(
    State(_state): State<AppState>,
    Json(req): Json<AdjustRequest>,
) -> (StatusCode, Json<Value>) {
    let mut results = Vec::new();

    for symbol in &req.symbols {
        match fetch_real_klines_with_period(symbol, &req.start, &req.end, "daily") {
            Ok(mut klines) => {
                let original_bars = klines.len();
                let splits = detect_and_adjust(&mut klines);
                results.push(json!({
                    "symbol": symbol,
                    "splits_detected": splits,
                    "adjustments_applied": !splits.is_empty(),
                    "original_bars": original_bars,
                    "adjusted_bars": klines.len(),
                }));
            }
            Err(e) => {
                results.push(json!({
                    "symbol": symbol,
                    "error": e,
                    "splits_detected": [],
                    "adjustments_applied": false,
                    "original_bars": 0,
                    "adjusted_bars": 0,
                }));
            }
        }
    }

    (StatusCode::OK, Json(json!({ "results": results })))
}
