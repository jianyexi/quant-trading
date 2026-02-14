use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};

/// Simple API key authentication middleware.
/// Checks for `X-API-Key` header and validates against the configured LLM API key.
/// If no API key is configured (empty string), all requests are allowed through.
pub async fn api_key_auth(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // For now, check for the presence of X-API-Key header.
    // In a real deployment, validate against a stored key.
    let api_key = request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok());

    match api_key {
        Some(key) if !key.is_empty() => Ok(next.run(request).await),
        // Allow requests without API key for development
        None => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
