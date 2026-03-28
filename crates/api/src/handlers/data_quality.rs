use axum::{extract::State, http::StatusCode, Json};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct DataQualityRequest {
    pub symbols: Vec<String>,
    pub start: String,
    pub end: String,
}

#[derive(Debug, Serialize)]
pub struct DateRange {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Serialize)]
pub struct QualityIssue {
    #[serde(rename = "type")]
    pub issue_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dates: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SymbolQualityResult {
    pub symbol: String,
    pub total_bars: i64,
    pub date_range: DateRange,
    pub issues: Vec<QualityIssue>,
    pub quality_score: f64,
}

#[derive(Debug, Serialize)]
pub struct DataQualityResponse {
    pub results: Vec<SymbolQualityResult>,
}

/// POST /api/data/quality — Check data quality for symbols
pub async fn check_data_quality(
    State(state): State<AppState>,
    Json(req): Json<DataQualityRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let start_date = NaiveDate::parse_from_str(&req.start, "%Y%m%d").map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid start date format, expected YYYYMMDD" })),
        )
    })?;
    let end_date = NaiveDate::parse_from_str(&req.end, "%Y%m%d").map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid end date format, expected YYYYMMDD" })),
        )
    })?;

    let start_dt = start_date.and_hms_opt(0, 0, 0).unwrap();
    let end_dt = end_date.and_hms_opt(23, 59, 59).unwrap();

    // ~245 trading days per year for CN market
    let total_calendar_days = (end_date - start_date).num_days().max(1);
    let expected_bars = (total_calendar_days as f64 / 365.0 * 245.0).round() as i64;

    let mut results = Vec::new();

    for symbol in &req.symbols {
        // Try database first, fallback to generated data
        let rows: Vec<(chrono::NaiveDateTime, f64, f64, f64, f64, f64)> = if let Some(pool) = state.db.as_ref() {
            sqlx::query_as::<_, (chrono::NaiveDateTime, f64, f64, f64, f64, f64)>(
                "SELECT datetime, open, high, low, close, volume \
                 FROM kline_daily \
                 WHERE symbol = $1 AND datetime >= $2 AND datetime <= $3 \
                 ORDER BY datetime ASC",
            )
            .bind(symbol)
            .bind(start_dt)
            .bind(end_dt)
            .fetch_all(pool)
            .await
            .unwrap_or_default()
        } else {
            // Fallback: use generated klines for quality analysis demo
            let klines = super::market::generate_backtest_klines(symbol, &req.start, &req.end);
            klines.iter().map(|k| (k.datetime, k.open, k.high, k.low, k.close, k.volume as f64)).collect()
        };

        let total_bars = rows.len() as i64;

        let date_range = if rows.is_empty() {
            DateRange {
                start: start_date.format("%Y-%m-%d").to_string(),
                end: end_date.format("%Y-%m-%d").to_string(),
            }
        } else {
            DateRange {
                start: rows.first().unwrap().0.date().format("%Y-%m-%d").to_string(),
                end: rows.last().unwrap().0.date().format("%Y-%m-%d").to_string(),
            }
        };

        let mut anomaly_dates = Vec::new();
        let mut anomaly_details = Vec::new();
        let mut zero_vol_dates = Vec::new();
        let mut stale_dates = Vec::new();
        let mut stale_run = 0i64;
        let mut gap_dates = Vec::new();
        let mut gap_details = Vec::new();

        for i in 0..rows.len() {
            let (_dt, _open, _high, _low, close, volume) = &rows[i];
            let date_str = rows[i].0.date().format("%Y-%m-%d").to_string();

            // Zero volume check
            if *volume == 0.0 {
                zero_vol_dates.push(date_str.clone());
            }

            if i > 0 {
                let prev_close = rows[i - 1].4;

                // Price anomaly: |return| > 20%
                if prev_close > 0.0 {
                    let ret = (close - prev_close) / prev_close * 100.0;
                    if ret.abs() > 20.0 {
                        anomaly_dates.push(date_str.clone());
                        anomaly_details.push(format!("return {ret:+.1}%"));
                    }
                }

                // Stale price: consecutive identical closes
                if (close - prev_close).abs() < 1e-9 {
                    stale_run += 1;
                    if stale_run >= 1 {
                        stale_dates.push(date_str.clone());
                    }
                } else {
                    stale_run = 0;
                }

                // Gap detection: overnight gap > 10%
                if prev_close > 0.0 {
                    let gap = (rows[i].1 - prev_close) / prev_close * 100.0; // open vs prev close
                    if gap.abs() > 10.0 {
                        gap_dates.push(date_str.clone());
                        gap_details.push(format!("gap {gap:+.1}%"));
                    }
                }
            }
        }

        // Compute quality score
        let bars_f = total_bars.max(1) as f64;
        let expected_f = expected_bars.max(1) as f64;
        let missing_pct = if total_bars < expected_bars {
            (expected_bars - total_bars) as f64 / expected_f * 100.0
        } else {
            0.0
        };
        let anomaly_pct = anomaly_dates.len() as f64 / bars_f * 100.0;
        let zero_vol_pct = zero_vol_dates.len() as f64 / bars_f * 100.0;
        let stale_pct = stale_dates.len() as f64 / bars_f * 100.0;
        let gap_pct = gap_dates.len() as f64 / bars_f * 100.0;

        let quality_score = (100.0
            - (missing_pct * 0.30
                + anomaly_pct * 0.25
                + zero_vol_pct * 0.20
                + stale_pct * 0.15
                + gap_pct * 0.10))
            .clamp(0.0, 100.0);

        // Round to 1 decimal
        let quality_score = (quality_score * 10.0).round() / 10.0;

        let mut issues = Vec::new();

        issues.push(QualityIssue {
            issue_type: "missing_bars".to_string(),
            count: Some((expected_bars - total_bars).max(0)),
            expected: Some(expected_bars),
            dates: None,
            details: None,
        });

        issues.push(QualityIssue {
            issue_type: "price_anomaly".to_string(),
            count: Some(anomaly_dates.len() as i64),
            expected: None,
            dates: Some(anomaly_dates.clone()),
            details: if anomaly_details.is_empty() {
                None
            } else {
                Some(anomaly_details.join(", "))
            },
        });

        issues.push(QualityIssue {
            issue_type: "zero_volume".to_string(),
            count: Some(zero_vol_dates.len() as i64),
            expected: None,
            dates: Some(zero_vol_dates),
            details: None,
        });

        issues.push(QualityIssue {
            issue_type: "stale_price".to_string(),
            count: Some(stale_dates.len() as i64),
            expected: None,
            dates: Some(stale_dates),
            details: None,
        });

        issues.push(QualityIssue {
            issue_type: "large_gap".to_string(),
            count: Some(gap_dates.len() as i64),
            expected: None,
            dates: Some(gap_dates.clone()),
            details: if gap_details.is_empty() {
                None
            } else {
                Some(gap_details.join(", "))
            },
        });

        results.push(SymbolQualityResult {
            symbol: symbol.clone(),
            total_bars,
            date_range,
            issues,
            quality_score,
        });
    }

    Ok(Json(json!(DataQualityResponse { results })))
}
