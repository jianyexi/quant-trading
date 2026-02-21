use axum::{extract::State, Json};
use serde_json::{json, Value};
use crate::state::AppState;

/// Per-module latency detail with bottleneck identification.
#[derive(serde::Serialize)]
struct ModuleLatency {
    name: String,
    last_us: u64,
    avg_us: u64,
    total_calls: u64,
    total_us: u64,
    pct_of_pipeline: f64,
}

/// GET /api/latency — per-module latency breakdown with bottleneck detection
pub async fn get_latency(State(state): State<AppState>) -> Json<Value> {
    let engine_guard = state.engine.lock().await;

    let Some(ref engine) = *engine_guard else {
        return Json(json!({
            "engine_running": false,
            "modules": [],
            "bottleneck": null,
            "pipeline_total_us": 0,
        }));
    };

    let status = engine.status().await;
    let lat = &status.latency;

    // Build per-module stats
    let mut modules: Vec<ModuleLatency> = Vec::new();

    // 1. Data Fetch
    modules.push(ModuleLatency {
        name: "数据获取 (Data Fetch)".into(),
        last_us: lat.last_data_fetch_us,
        avg_us: lat.avg_data_fetch_us,
        total_calls: lat.total_data_fetches,
        total_us: lat.avg_data_fetch_us * lat.total_data_fetches,
        pct_of_pipeline: 0.0, // computed below
    });

    // 2. Strategy / Factor Compute
    modules.push(ModuleLatency {
        name: "策略计算 (Strategy/Factor)".into(),
        last_us: lat.last_factor_compute_us,
        avg_us: lat.avg_factor_compute_us,
        total_calls: lat.total_bars_processed,
        total_us: lat.avg_factor_compute_us * lat.total_bars_processed,
        pct_of_pipeline: 0.0,
    });

    // 3. Risk Check
    modules.push(ModuleLatency {
        name: "风控检查 (Risk Check)".into(),
        last_us: lat.last_risk_check_us,
        avg_us: lat.avg_risk_check_us,
        total_calls: lat.total_risk_checks,
        total_us: lat.avg_risk_check_us * lat.total_risk_checks,
        pct_of_pipeline: 0.0,
    });

    // 4. Order Submit
    modules.push(ModuleLatency {
        name: "订单提交 (Order Submit)".into(),
        last_us: lat.last_order_submit_us,
        avg_us: lat.avg_order_submit_us,
        total_calls: lat.total_orders_submitted,
        total_us: lat.avg_order_submit_us * lat.total_orders_submitted,
        pct_of_pipeline: 0.0,
    });

    // Compute pipeline total (sum of avg latencies for one full pass)
    let pipeline_one_pass_us = lat.avg_data_fetch_us
        + lat.avg_factor_compute_us
        + lat.avg_risk_check_us
        + lat.avg_order_submit_us;

    // Compute percentages
    if pipeline_one_pass_us > 0 {
        for m in &mut modules {
            m.pct_of_pipeline = m.avg_us as f64 / pipeline_one_pass_us as f64 * 100.0;
        }
    }

    // Identify bottleneck (module with highest avg_us)
    let bottleneck = modules.iter()
        .filter(|m| m.avg_us > 0)
        .max_by_key(|m| m.avg_us)
        .map(|m| json!({
            "module": m.name,
            "avg_us": m.avg_us,
            "pct": m.pct_of_pipeline,
            "suggestion": bottleneck_suggestion(&m.name, m.avg_us),
        }));

    // Error rate from data actor
    let _data_errors = {
        0u64
    };

    // Throughput analysis
    let bars = lat.total_bars_processed;
    let signals = status.total_signals;
    let orders = status.total_orders;
    let fills = status.total_fills;
    let rejects = status.total_rejected;

    // Historical latency trend (last N samples not available with atomics,
    // but we can show current snapshot)
    let throughput = json!({
        "total_bars": bars,
        "total_signals": signals,
        "total_orders": orders,
        "total_fills": fills,
        "total_rejected": rejects,
        "signal_ratio": if bars > 0 { signals as f64 / bars as f64 } else { 0.0 },
        "fill_ratio": if orders > 0 { fills as f64 / orders as f64 } else { 0.0 },
    });

    // Performance scoring (0-100)
    let score = compute_health_score(&lat, pipeline_one_pass_us);

    Json(json!({
        "engine_running": status.running,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "modules": modules.iter().map(|m| json!({
            "name": m.name,
            "last_us": m.last_us,
            "avg_us": m.avg_us,
            "total_calls": m.total_calls,
            "total_us": m.total_us,
            "pct_of_pipeline": (m.pct_of_pipeline * 10.0).round() / 10.0,
        })).collect::<Vec<_>>(),
        "bottleneck": bottleneck,
        "pipeline_total_us": pipeline_one_pass_us,
        "throughput": throughput,
        "health_score": score,
        "thresholds": {
            "data_fetch_warn_us": 500_000u64,
            "data_fetch_critical_us": 2_000_000u64,
            "strategy_warn_us": 10_000u64,
            "strategy_critical_us": 100_000u64,
            "risk_warn_us": 1_000u64,
            "risk_critical_us": 10_000u64,
            "order_warn_us": 5_000u64,
            "order_critical_us": 50_000u64,
        },
    }))
}

fn bottleneck_suggestion(module: &str, avg_us: u64) -> String {
    if module.contains("Data Fetch") {
        if avg_us > 2_000_000 {
            "数据源响应极慢，建议切换到低延迟模式(LowLatency)或本地数据缓存".into()
        } else if avg_us > 500_000 {
            "数据获取延迟较高，检查网络连接或考虑使用市场数据服务器".into()
        } else {
            "数据获取延迟正常".into()
        }
    } else if module.contains("Strategy") || module.contains("Factor") {
        if avg_us > 100_000 {
            "策略计算极慢，考虑减少因子数量或使用预编译ONNX模型".into()
        } else if avg_us > 10_000 {
            "策略计算延迟较高，检查指标计算复杂度".into()
        } else {
            "策略计算延迟正常".into()
        }
    } else if module.contains("Risk") {
        if avg_us > 10_000 {
            "风控检查异常缓慢，检查是否有锁竞争或数据库查询".into()
        } else {
            "风控检查延迟正常".into()
        }
    } else if module.contains("Order") {
        if avg_us > 50_000 {
            "订单提交延迟极高，检查Broker连接或QMT Bridge状态".into()
        } else if avg_us > 5_000 {
            "订单提交延迟较高，考虑批量下单或异步提交".into()
        } else {
            "订单提交延迟正常".into()
        }
    } else {
        "延迟正常".into()
    }
}

fn compute_health_score(
    _lat: &quant_broker::engine::PipelineLatency,
    pipeline_total_us: u64,
) -> u64 {
    // Score 0-100 based on pipeline latency
    // < 1ms total = 100, < 10ms = 90, < 100ms = 70, < 1s = 50, > 1s = 30, > 5s = 10
    if pipeline_total_us == 0 { return 100; }
    match pipeline_total_us {
        0..=1_000 => 100,
        1_001..=10_000 => 95,
        10_001..=50_000 => 85,
        50_001..=100_000 => 70,
        100_001..=500_000 => 55,
        500_001..=1_000_000 => 40,
        1_000_001..=5_000_000 => 25,
        _ => 10,
    }
}
