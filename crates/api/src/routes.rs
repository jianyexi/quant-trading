use axum::{
    middleware,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::auth::api_key_auth;
use crate::handlers;
use crate::state::AppState;
use crate::ws;

fn market_routes() -> Router<AppState> {
    Router::new()
        .route("/kline/:symbol", get(handlers::get_kline))
        .route("/quote/:symbol", get(handlers::get_quote))
        .route("/stocks", get(handlers::list_stocks))
}

fn backtest_routes() -> Router<AppState> {
    Router::new()
        .route("/run", post(handlers::run_backtest))
        .route("/results/:id", get(handlers::get_backtest_results))
}

fn order_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::list_orders))
}

fn portfolio_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::get_portfolio))
}

fn chat_routes() -> Router<AppState> {
    Router::new()
        .route("/", post(handlers::chat))
        .route("/history", get(handlers::chat_history))
        .route("/stream", get(ws::ws_chat))
}

fn trade_routes() -> Router<AppState> {
    Router::new()
        .route("/start", post(handlers::trade_start))
        .route("/stop", post(handlers::trade_stop))
        .route("/status", get(handlers::trade_status))
        .route("/qmt/status", get(handlers::qmt_bridge_status))
}

fn screen_routes() -> Router<AppState> {
    Router::new()
        .route("/scan", post(handlers::screen_scan))
        .route("/factors/:symbol", get(handlers::screen_factors))
}

pub fn create_router(state: AppState, web_dist: &str) -> Router {
    let api_routes = Router::new()
        .route("/api/health", get(handlers::health))
        .route("/api/dashboard", get(handlers::get_dashboard))
        .route("/api/strategies", get(handlers::list_strategies))
        .nest("/api/market", market_routes())
        .nest("/api/backtest", backtest_routes())
        .nest("/api/orders", order_routes())
        .nest("/api/portfolio", portfolio_routes())
        .nest("/api/chat", chat_routes())
        .nest("/api/trade", trade_routes())
        .nest("/api/screen", screen_routes())
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
