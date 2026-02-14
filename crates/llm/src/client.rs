use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;
use tracing::debug;

// ── Request / Response types ────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming types ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ── Client ──────────────────────────────────────────────────────────

pub struct LlmClient {
    client: Client,
    api_url: String,
    api_key: String,
    model: String,
    temperature: f64,
    max_tokens: u32,
}

impl LlmClient {
    pub fn new(
        api_url: &str,
        api_key: &str,
        model: &str,
        temperature: f64,
        max_tokens: u32,
    ) -> Self {
        Self {
            client: Client::new(),
            api_url: api_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            temperature,
            max_tokens,
        }
    }

    fn build_request(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
        stream: bool,
    ) -> ChatRequest {
        ChatRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            tools: tools.map(|t| t.to_vec()),
            temperature: Some(self.temperature),
            max_tokens: Some(self.max_tokens),
            stream: if stream { Some(true) } else { None },
        }
    }

    /// Send a chat completion request (non-streaming).
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
    ) -> anyhow::Result<ChatResponse> {
        let body = self.build_request(messages, tools, false);
        let url = format!("{}/chat/completions", self.api_url);

        debug!("POST {url} model={}", self.model);

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM API error (HTTP {status}): {text}");
        }

        let chat_resp: ChatResponse = resp.json().await?;
        Ok(chat_resp)
    }

    /// Send a streaming chat completion request; returns an async stream of
    /// parsed SSE chunks.
    pub async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
    ) -> anyhow::Result<impl Stream<Item = anyhow::Result<StreamChunk>>> {
        let body = self.build_request(messages, tools, true);
        let url = format!("{}/chat/completions", self.api_url);

        debug!("POST (stream) {url} model={}", self.model);

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM API error (HTTP {status}): {text}");
        }

        let byte_stream = resp.bytes_stream();

        // Buffer incoming bytes and split on newlines to extract SSE `data:` lines.
        let stream = futures::stream::unfold(
            (byte_stream, String::new()),
            |(mut inner, mut buf)| async move {
                loop {
                    // Drain buffered lines first.
                    if let Some(newline_pos) = buf.find('\n') {
                        let line = buf[..newline_pos].trim().to_string();
                        buf = buf[newline_pos + 1..].to_string();

                        if line == "data: [DONE]" {
                            return None;
                        }
                        if let Some(json_str) = line.strip_prefix("data: ") {
                            match serde_json::from_str::<StreamChunk>(json_str) {
                                Ok(chunk) => return Some((Ok(chunk), (inner, buf))),
                                Err(e) => {
                                    return Some((
                                        Err(anyhow::anyhow!("Failed to parse SSE chunk: {e}")),
                                        (inner, buf),
                                    ))
                                }
                            }
                        }
                        // Skip non-data lines (comments, empty, etc.) and keep looping.
                        continue;
                    }

                    // Need more data from the network.
                    match inner.next().await {
                        Some(Ok(bytes)) => {
                            buf.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            return Some((Err(e.into()), (inner, buf)));
                        }
                        None => {
                            // Stream ended; try to process remaining buffer.
                            if buf.trim().is_empty() || buf.trim() == "data: [DONE]" {
                                return None;
                            }
                            if let Some(json_str) = buf.trim().strip_prefix("data: ") {
                                let result = serde_json::from_str::<StreamChunk>(json_str)
                                    .map_err(|e| {
                                        anyhow::anyhow!("Failed to parse final SSE chunk: {e}")
                                    });
                                return Some((result, (inner, String::new())));
                            }
                            return None;
                        }
                    }
                }
            },
        );

        Ok(stream)
    }
}
