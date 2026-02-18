// Financial news fetcher for sentiment analysis
//
// Supports multiple Chinese financial news sources:
// - East Money (东方财富) via AKShare proxy
// - Sina Finance (新浪财经) via public API

use chrono::{NaiveDateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use quant_core::error::{QuantError, Result};

/// A single news article from any source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsArticle {
    pub title: String,
    pub content: String,
    pub source: String,
    pub symbol: String,
    pub published_at: NaiveDateTime,
    pub url: String,
}

/// Unified news fetcher that aggregates from multiple sources.
pub struct NewsFetcher {
    client: Client,
    akshare_base: String,
}

impl NewsFetcher {
    pub fn new(akshare_base: &str) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            akshare_base: akshare_base.trim_end_matches('/').to_string(),
        }
    }

    /// Fetch news for a symbol from all enabled sources.
    pub async fn fetch_news(
        &self,
        symbol: &str,
        sources: &[String],
        max_per_source: usize,
    ) -> Vec<NewsArticle> {
        let mut articles = Vec::new();

        for source in sources {
            let result = match source.as_str() {
                "eastmoney" => self.fetch_eastmoney(symbol, max_per_source).await,
                "sina" => self.fetch_sina(symbol, max_per_source).await,
                _ => {
                    warn!("Unknown news source: {}", source);
                    continue;
                }
            };

            match result {
                Ok(mut news) => {
                    debug!("Fetched {} articles from {} for {}", news.len(), source, symbol);
                    articles.append(&mut news);
                }
                Err(e) => {
                    warn!("Failed to fetch news from {} for {}: {}", source, symbol, e);
                }
            }
        }

        articles
    }

    /// Fetch stock news from East Money (东方财富) via AKShare.
    async fn fetch_eastmoney(&self, symbol: &str, limit: usize) -> Result<Vec<NewsArticle>> {
        // AKShare endpoint: stock_news_em (东方财富个股新闻)
        // The symbol format for East Money is just the code part (e.g., "600519")
        let code = symbol.split('.').next().unwrap_or(symbol);

        let url = format!(
            "{}/api/public/stock_news_em?stock={}",
            self.akshare_base, code
        );

        debug!("Fetching East Money news: {}", url);

        let resp = self.client.get(&url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(QuantError::DataError(format!(
                "East Money news API error (HTTP {})", status
            )));
        }

        let items: Vec<EastMoneyNewsItem> = resp.json().await.map_err(|e| {
            QuantError::DataError(format!("Failed to parse East Money news: {}", e))
        })?;

        let articles: Vec<NewsArticle> = items
            .into_iter()
            .take(limit)
            .filter_map(|item| {
                let published_at = parse_datetime_flexible(&item.datetime).unwrap_or_else(|| Utc::now().naive_utc());
                Some(NewsArticle {
                    title: item.title,
                    content: item.content.unwrap_or_default(),
                    source: "东方财富".to_string(),
                    symbol: symbol.to_string(),
                    published_at,
                    url: item.url.unwrap_or_default(),
                })
            })
            .collect();

        Ok(articles)
    }

    /// Fetch stock news from Sina Finance (新浪财经).
    async fn fetch_sina(&self, symbol: &str, limit: usize) -> Result<Vec<NewsArticle>> {
        // Sina Finance has a public API for stock-related news
        let code = symbol.split('.').next().unwrap_or(symbol);
        let market = if symbol.contains(".SH") { "sh" } else { "sz" };

        let url = format!(
            "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol={}{}&Page=1",
            market, code
        );

        debug!("Fetching Sina news: {}", url);

        let resp = self.client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(QuantError::DataError(format!(
                "Sina news API error (HTTP {})", resp.status()
            )));
        }

        let html = resp.text().await.map_err(|e| {
            QuantError::DataError(format!("Failed to read Sina response: {}", e))
        })?;

        // Parse simple news items from the HTML response
        let articles = parse_sina_news_html(&html, symbol, limit);
        Ok(articles)
    }
}

// ── AKShare response types ──────────────────────────────────────

#[derive(Debug, Deserialize)]
struct EastMoneyNewsItem {
    #[serde(alias = "新闻标题", alias = "title")]
    title: String,
    #[serde(alias = "新闻内容", alias = "content")]
    content: Option<String>,
    #[serde(alias = "发布时间", alias = "datetime", alias = "publish_time")]
    datetime: String,
    #[serde(alias = "新闻链接", alias = "url")]
    url: Option<String>,
}

// ── Helpers ─────────────────────────────────────────────────────

fn parse_datetime_flexible(s: &str) -> Option<NaiveDateTime> {
    // Try common Chinese datetime formats
    let formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y%m%d%H%M%S",
        "%Y年%m月%d日 %H:%M",
    ];
    for fmt in &formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some(dt);
        }
    }
    // Try date-only
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(d.and_hms_opt(0, 0, 0)?);
    }
    None
}

/// Parse news titles from Sina Finance HTML page.
fn parse_sina_news_html(html: &str, symbol: &str, limit: usize) -> Vec<NewsArticle> {
    let mut articles = Vec::new();
    let now = Utc::now().naive_utc();

    // Sina news pages have links in <a> tags with titles
    // Pattern: <a ... title="..." href="...">...</a> inside datelist
    for line in html.lines() {
        if articles.len() >= limit {
            break;
        }

        let line = line.trim();
        if !line.contains("target=\"_blank\"") || !line.contains("href=") {
            continue;
        }

        // Extract title from tag content or title attribute
        if let Some(title) = extract_tag_content(line, "a") {
            if title.len() < 4 || title.contains("更多") {
                continue;
            }

            let url = extract_attr(line, "href").unwrap_or_default();
            let date_str = extract_nearby_date(line);
            let published_at = date_str
                .and_then(|d| parse_datetime_flexible(&d))
                .unwrap_or(now);

            articles.push(NewsArticle {
                title: title.to_string(),
                content: String::new(),
                source: "新浪财经".to_string(),
                symbol: symbol.to_string(),
                published_at,
                url: url.to_string(),
            });
        }
    }

    articles
}

fn extract_tag_content<'a>(html: &'a str, tag: &str) -> Option<&'a str> {
    let open_end = html.find(&format!("<{}", tag))
        .and_then(|start| html[start..].find('>').map(|i| start + i + 1))?;
    let close_start = html[open_end..].find(&format!("</{}>", tag))
        .map(|i| open_end + i)?;
    let content = html[open_end..close_start].trim();
    if content.is_empty() { None } else { Some(content) }
}

fn extract_attr<'a>(html: &'a str, attr: &str) -> Option<&'a str> {
    let pattern = format!("{}=\"", attr);
    let start = html.find(&pattern).map(|i| i + pattern.len())?;
    let end = html[start..].find('"').map(|i| start + i)?;
    Some(&html[start..end])
}

fn extract_nearby_date(html: &str) -> Option<String> {
    // Look for date patterns like "2024-01-15" or "01-15" in surrounding text
    let re_full = regex_lite::Regex::new(r"(\d{4}-\d{2}-\d{2})").ok()?;
    re_full.find(html).map(|m| m.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_datetime_flexible() {
        assert!(parse_datetime_flexible("2024-01-15 10:30:00").is_some());
        assert!(parse_datetime_flexible("2024-01-15 10:30").is_some());
        assert!(parse_datetime_flexible("2024-01-15").is_some());
        assert!(parse_datetime_flexible("garbage").is_none());
    }

    #[test]
    fn test_extract_attr() {
        let html = r#"<a href="https://example.com" target="_blank">Test</a>"#;
        assert_eq!(extract_attr(html, "href"), Some("https://example.com"));
    }

    #[test]
    fn test_extract_tag_content() {
        let html = r#"<a href="url">News Title</a>"#;
        assert_eq!(extract_tag_content(html, "a"), Some("News Title"));
    }
}
