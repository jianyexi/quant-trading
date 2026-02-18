// LLM-powered sentiment analysis for financial news
//
// Uses the existing LlmClient to analyze news articles and produce
// structured sentiment scores in [-1.0, +1.0].

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::client::{ChatMessage, LlmClient};

/// Result of LLM sentiment analysis on a single news article.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    /// Sentiment score in [-1.0, +1.0]
    pub score: f64,
    /// Brief reasoning from the LLM
    pub reasoning: String,
    /// Key entities or topics mentioned
    pub keywords: Vec<String>,
}

/// Analyze a batch of news articles for sentiment using the LLM.
pub struct SentimentAnalyzer {
    client: LlmClient,
}

impl SentimentAnalyzer {
    pub fn new(client: LlmClient) -> Self {
        Self { client }
    }

    /// Analyze a single news article and return a sentiment score.
    pub async fn analyze(&self, title: &str, content: &str, symbol: &str) -> anyhow::Result<SentimentAnalysis> {
        let text = if content.is_empty() {
            title.to_string()
        } else {
            format!("{}\n\n{}", title, &content[..content.len().min(500)])
        };

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(SENTIMENT_SYSTEM_PROMPT.to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(format!(
                    "请分析以下关于股票 {} 的新闻的市场情绪：\n\n{}\n\n请严格按照JSON格式输出。",
                    symbol, text
                )),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let resp = self.client.chat(&messages, None).await?;

        let reply = resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .ok_or_else(|| anyhow::anyhow!("No response from LLM"))?;

        debug!("LLM sentiment response: {}", reply);

        parse_sentiment_response(reply)
    }

    /// Analyze multiple articles efficiently by batching them into fewer LLM calls.
    pub async fn analyze_batch(
        &self,
        articles: &[(String, String, String)], // (title, content, symbol)
        max_calls: usize,
    ) -> Vec<(usize, anyhow::Result<SentimentAnalysis>)> {
        let mut results = Vec::new();

        // For cost control, analyze up to max_calls articles
        for (idx, (title, content, symbol)) in articles.iter().enumerate().take(max_calls) {
            let result = self.analyze(title, content, symbol).await;
            results.push((idx, result));
            // Small delay between calls to avoid rate limiting
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        results
    }
}

const SENTIMENT_SYSTEM_PROMPT: &str = r#"你是一个专业的A股市场情绪分析师。你的任务是分析给定的财经新闻，判断其对相关股票的情绪影响。

请严格按照以下JSON格式输出（不要添加markdown代码块标记）：
{
  "score": 0.5,
  "reasoning": "简短的分析理由",
  "keywords": ["关键词1", "关键词2"]
}

评分标准：
- score: [-1.0, 1.0] 的浮点数
  - 1.0: 极度利好（重大业绩超预期、重组成功、政策大利好）
  - 0.5~0.8: 利好（业绩增长、产品发布、合同中标）
  - 0.2~0.5: 轻微利好
  - -0.2~0.2: 中性（无明显方向性影响）
  - -0.5~-0.2: 轻微利空
  - -0.8~-0.5: 利空（业绩下滑、管理层变动、行业监管）
  - -1.0: 极度利空（重大违规、退市风险、黑天鹅事件）

关键判断维度：
1. 基本面影响（盈利、收入、市场份额）
2. 政策/监管影响
3. 行业趋势
4. 市场情绪/资金面
5. 技术面暗示

只输出JSON，不要输出其他内容。"#;

/// Parse the LLM response into a structured SentimentAnalysis.
fn parse_sentiment_response(response: &str) -> anyhow::Result<SentimentAnalysis> {
    // Try to find JSON in the response (LLM might wrap it in markdown code blocks)
    let json_str = extract_json(response);

    match serde_json::from_str::<SentimentAnalysis>(&json_str) {
        Ok(mut analysis) => {
            // Clamp score to valid range
            analysis.score = analysis.score.max(-1.0).min(1.0);
            Ok(analysis)
        }
        Err(e) => {
            warn!("Failed to parse LLM sentiment JSON: {} — response: {}", e, response);
            // Fallback: try to extract score from text
            fallback_parse(response)
        }
    }
}

/// Extract JSON from potentially markdown-wrapped response.
fn extract_json(text: &str) -> String {
    // Remove markdown code block markers
    let cleaned = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Find the first { and last }
    if let (Some(start), Some(end)) = (cleaned.find('{'), cleaned.rfind('}')) {
        cleaned[start..=end].to_string()
    } else {
        cleaned.to_string()
    }
}

/// Fallback parser when JSON parsing fails.
fn fallback_parse(text: &str) -> anyhow::Result<SentimentAnalysis> {
    // Try to find a number that looks like a score
    let score = text
        .split_whitespace()
        .filter_map(|w| w.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').parse::<f64>().ok())
        .find(|&v| (-1.0..=1.0).contains(&v))
        .unwrap_or(0.0);

    Ok(SentimentAnalysis {
        score,
        reasoning: format!("(fallback parse) {}", &text[..text.len().min(100)]),
        keywords: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sentiment_json() {
        let resp = r#"{"score": 0.7, "reasoning": "业绩超预期", "keywords": ["业绩", "增长"]}"#;
        let analysis = parse_sentiment_response(resp).unwrap();
        assert!((analysis.score - 0.7).abs() < 0.01);
        assert_eq!(analysis.keywords.len(), 2);
    }

    #[test]
    fn test_parse_markdown_wrapped() {
        let resp = "```json\n{\"score\": -0.5, \"reasoning\": \"利空\", \"keywords\": []}\n```";
        let analysis = parse_sentiment_response(resp).unwrap();
        assert!((analysis.score - -0.5).abs() < 0.01);
    }

    #[test]
    fn test_score_clamping() {
        let resp = r#"{"score": 1.5, "reasoning": "超出范围", "keywords": []}"#;
        let analysis = parse_sentiment_response(resp).unwrap();
        assert!((analysis.score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_json() {
        assert_eq!(extract_json("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(extract_json("{\"a\":1}"), "{\"a\":1}");
        assert_eq!(extract_json("some text {\"a\":1} more"), "{\"a\":1}");
    }
}
