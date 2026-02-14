use crate::client::{ChatMessage, ToolCall};

/// Manages the conversation context (message history) sent to the LLM.
pub struct ConversationContext {
    pub messages: Vec<ChatMessage>,
    pub system_prompt: String,
    max_history: usize,
}

const DEFAULT_SYSTEM_PROMPT: &str = "\
You are an expert quantitative trading assistant specializing in Chinese A-share markets. \
You have deep knowledge of:\n\
- A-share trading rules (T+1 settlement, ±10% daily price limits, ±20% for ChiNext/STAR Market)\n\
- Technical analysis indicators (MACD, KDJ, RSI, Bollinger Bands, moving averages)\n\
- Fundamental analysis (P/E, P/B, ROE, revenue growth, industry analysis)\n\
- Factor-based investing (value, momentum, quality, size, volatility factors)\n\
- Backtest methodology and performance metrics (Sharpe ratio, max drawdown, annualized return)\n\
- Portfolio construction and risk management\n\
- Common A-share data sources (Wind, Tushare, AKShare)\n\
- Stock screening and selection strategies\n\n\
When the user asks about stocks, always use the provided tools to fetch real data before answering. \
Present analysis in a clear, structured format. Use stock codes (e.g., 600519.SH, 000858.SZ) \
when referring to specific securities.\
";

impl ConversationContext {
    pub fn new(system_prompt: &str, max_history: usize) -> Self {
        let prompt = if system_prompt.is_empty() {
            DEFAULT_SYSTEM_PROMPT.to_string()
        } else {
            system_prompt.to_string()
        };
        Self {
            messages: Vec::new(),
            system_prompt: prompt,
            max_history,
        }
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
        self.trim_history();
    }

    pub fn add_assistant_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: Some(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
        self.trim_history();
    }

    pub fn add_tool_result(&mut self, tool_call_id: &str, result: &str) {
        self.messages.push(ChatMessage {
            role: "tool".to_string(),
            content: Some(result.to_string()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        });
        self.trim_history();
    }

    pub fn add_assistant_tool_calls(&mut self, tool_calls: Vec<ToolCall>) {
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        });
        self.trim_history();
    }

    /// Return the full message list including the system prompt at the front.
    pub fn get_messages(&self) -> Vec<ChatMessage> {
        let mut msgs = Vec::with_capacity(1 + self.messages.len());
        msgs.push(ChatMessage {
            role: "system".to_string(),
            content: Some(self.system_prompt.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
        msgs.extend(self.messages.clone());
        msgs
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    fn trim_history(&mut self) {
        if self.messages.len() > self.max_history {
            let excess = self.messages.len() - self.max_history;
            self.messages.drain(..excess);
        }
    }
}
