use axum::{
    extract::Request,
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::auth::api_key_auth;
use crate::handlers;
use crate::log_store::LogLevel;
use crate::state::AppState;
use crate::ws;

fn market_routes() -> Router<AppState> {
    Router::new()
        .route("/kline/:symbol", get(handlers::get_kline))
        .route("/quote/:symbol", get(handlers::get_quote))
        .route("/stocks", get(handlers::list_stocks))
        .route("/data-source", get(handlers::data_source_status))
        .route("/cache-status", get(handlers::cache_status))
        .route("/sync-data", post(handlers::sync_data))
}

fn backtest_routes() -> Router<AppState> {
    Router::new()
        .route("/run", post(handlers::run_backtest))
        .route("/walk-forward", post(handlers::walk_forward))
        .route("/results/:id", get(handlers::get_backtest_results))
}

fn order_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::list_orders))
}

fn portfolio_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::get_portfolio))
        .route("/close", post(handlers::close_position))
        .route("/closed", get(handlers::get_closed_positions))
}

fn chat_routes() -> Router<AppState> {
    Router::new()
        .route("/", post(handlers::chat))
        .route("/history", get(handlers::chat_history))
        .route("/stream", get(ws::ws_chat))
}

fn monitor_routes() -> Router<AppState> {
    Router::new()
        .route("/ws", get(ws::ws_monitor))
}

fn trade_routes() -> Router<AppState> {
    Router::new()
        .route("/start", post(handlers::trade_start))
        .route("/stop", post(handlers::trade_stop))
        .route("/status", get(handlers::trade_status))
        .route("/performance", get(handlers::trade_performance))
        .route("/risk", get(handlers::risk_status))
        .route("/risk/signals", get(handlers::risk_signals))
        .route("/risk/reset-circuit", post(handlers::risk_reset_circuit))
        .route("/risk/reset-daily", post(handlers::risk_reset_daily))
        .route("/retrain", post(handlers::ml_retrain))
        .route("/model-info", get(handlers::ml_model_info))
        .route("/qmt/status", get(handlers::qmt_bridge_status))
        .route("/ticks", get(handlers::get_recorded_ticks))
}

fn ml_routes() -> Router<AppState> {
    Router::new()
        .route("/training-history", get(handlers::ml_training_history))
        .route("/training-history/:id", get(handlers::ml_training_run_detail))
}

fn screen_routes() -> Router<AppState> {
    Router::new()
        .route("/scan", post(handlers::screen_scan))
        .route("/factors/:symbol", get(handlers::screen_factors))
}

fn sentiment_routes() -> Router<AppState> {
    Router::new()
        .route("/submit", post(handlers::sentiment_submit))
        .route("/batch", post(handlers::sentiment_batch_submit))
        .route("/summary", get(handlers::sentiment_summary))
        .route("/collector/start", post(handlers::collector_start))
        .route("/collector/stop", post(handlers::collector_stop))
        .route("/collector/status", get(handlers::collector_status))
        .route("/:symbol", get(handlers::sentiment_query))
}

fn research_routes() -> Router<AppState> {
    Router::new()
        .route("/dl-models", get(handlers::research_dl_models))
        .route("/dl-models/summary", get(handlers::research_dl_models_summary))
        .route("/dl-models/collect", post(handlers::research_dl_collect))
}

fn factor_routes() -> Router<AppState> {
    Router::new()
        .route("/mine/parametric", post(handlers::factor_mine_parametric))
        .route("/mine/gp", post(handlers::factor_mine_gp))
        .route("/registry", get(handlers::factor_registry_get))
        .route("/manage", post(handlers::factor_registry_manage))
        .route("/export", post(handlers::factor_export_promoted))
        .route("/results", get(handlers::factor_results))
        .route("/mining-history", get(handlers::factor_mining_history))
        .route("/mining-history/:id", get(handlers::factor_mining_run_detail))
}

fn task_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::list_tasks))
        .route("/running", get(handlers::list_running_tasks))
        .route("/:id", get(handlers::get_task).delete(handlers::cancel_task))
}

fn journal_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::get_journal))
        .route("/snapshots", get(handlers::get_journal_snapshots))
}

fn notification_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::list_notifications))
        .route("/unread-count", get(handlers::notification_unread_count))
        .route("/read-all", post(handlers::notification_mark_all_read))
        .route("/config", get(handlers::notification_config_get).post(handlers::notification_config_save))
        .route("/test", post(handlers::notification_test))
        .route("/:id/read", post(handlers::notification_mark_read))
}

fn log_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::get_logs).delete(handlers::clear_logs))
}

/// Middleware that logs every API request to the in-memory LogStore.
async fn request_logger(
    axum::extract::State(state): axum::extract::State<AppState>,
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    // Skip logging the logs endpoint itself to avoid recursion
    if path == "/api/logs" || path == "/api/metrics" || path == "/api/latency" {
        return next.run(request).await;
    }
    let start = std::time::Instant::now();
    let response = next.run(request).await;
    let duration_ms = start.elapsed().as_millis() as u64;
    let status = response.status().as_u16();

    let (level, message) = if status >= 500 {
        (LogLevel::Error, format!("{} {} → {} ({}ms)", method, path, status, duration_ms))
    } else if status >= 400 {
        (LogLevel::Warn, format!("{} {} → {} ({}ms)", method, path, status, duration_ms))
    } else {
        (LogLevel::Info, format!("{} {} → {} ({}ms)", method, path, status, duration_ms))
    };

    state.log_store.push(level, &method, &path, status, duration_ms, &message, None);
    response
}

pub fn create_router(state: AppState, web_dist: &str) -> Router {
    let api_routes = Router::new()
        .route("/api/health", get(handlers::health))
        .route("/api/dashboard", get(handlers::get_dashboard))
        .route("/api/metrics", get(handlers::get_metrics))
        .route("/api/reports", get(handlers::get_reports))
        .route("/api/latency", get(handlers::get_latency))
        .route("/api/strategies", get(handlers::list_strategies))
        .route("/api/strategy/config", get(handlers::load_strategy_config).post(handlers::save_strategy_config))
        .nest("/api/market", market_routes())
        .nest("/api/backtest", backtest_routes())
        .nest("/api/orders", order_routes())
        .nest("/api/portfolio", portfolio_routes())
        .nest("/api/chat", chat_routes())
        .nest("/api/monitor", monitor_routes())
        .nest("/api/trade", trade_routes())
        .nest("/api/ml", ml_routes())
        .nest("/api/screen", screen_routes())
        .nest("/api/sentiment", sentiment_routes())
        .nest("/api/research", research_routes())
        .nest("/api/factor", factor_routes())
        .nest("/api/tasks", task_routes())
        .nest("/api/journal", journal_routes())
        .nest("/api/notifications", notification_routes())
        .nest("/api/logs", log_routes())
        .layer(middleware::from_fn_with_state(state.clone(), request_logger))
        .layer(middleware::from_fn(api_key_auth))
        .with_state(state);

    // SPA fallback: serve index.html for any non-API, non-static path (returns 200)
    let index_path = std::path::PathBuf::from(web_dist).join("index.html");
    let spa_handler = move || {
        let path = index_path.clone();
        async move {
            match tokio::fs::read(&path).await {
                Ok(bytes) => axum::response::Html(bytes).into_response(),
                Err(_) => (axum::http::StatusCode::NOT_FOUND, "index.html not found").into_response(),
            }
        }
    };

    api_routes
        .fallback_service(
            ServeDir::new(web_dist).fallback(get(spa_handler))
        )
        .layer(CorsLayer::permissive())
}
