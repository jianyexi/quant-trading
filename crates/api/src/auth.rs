use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};

/// API key authentication middleware.
///
/// Validates `X-API-Key` header against the `API_KEY` env var.
/// If `API_KEY` is not set or empty, authentication is disabled (dev mode).
pub async fn api_key_auth(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let configured_key = std::env::var("API_KEY").unwrap_or_default();

    // If no API key is configured, allow all requests (dev mode)
    if configured_key.is_empty() {
        return Ok(next.run(request).await);
    }

    // Health endpoint is always accessible (for load balancers / k8s probes)
    if request.uri().path() == "/api/health" {
        return Ok(next.run(request).await);
    }

    // Static file requests (no /api prefix) don't require auth
    if !request.uri().path().starts_with("/api/") {
        return Ok(next.run(request).await);
    }

    let api_key = request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok());

    match api_key {
        Some(key) if key == configured_key => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
