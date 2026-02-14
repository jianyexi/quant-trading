use crate::client::{FunctionDefinition, ToolCall, ToolDefinition};
use serde_json::json;

fn tool(name: &str, description: &str, parameters: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        },
    }
}

// ── Tool Executor ───────────────────────────────────────────────────

/// Dispatches tool calls to the appropriate handler and returns JSON results.
/// Currently returns mock/placeholder data; the real data layer will be
/// connected later.
pub struct ToolExecutor;

impl ToolExecutor {
    pub fn new() -> Self {
        Self
    }

    /// Execute a tool call and return the result as a JSON string.
    pub async fn execute(&self, tool_call: &ToolCall) -> anyhow::Result<String> {
        let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
            .unwrap_or(json!({}));

        match tool_call.function.name.as_str() {
            "get_kline" => {
                let symbol = args["symbol"].as_str().unwrap_or("unknown");
                let start = args["start_date"].as_str().unwrap_or("");
                let end = args["end_date"].as_str().unwrap_or("");
                let freq = args["frequency"].as_str().unwrap_or("daily");
                let result = json!({
                    "symbol": symbol,
                    "frequency": freq,
                    "start_date": start,
                    "end_date": end,
                    "data": [
                        {"date": "2024-01-02", "open": 100.0, "high": 102.5, "low": 99.5, "close": 101.8, "volume": 15000000},
                        {"date": "2024-01-03", "open": 101.8, "high": 103.2, "low": 101.0, "close": 102.5, "volume": 18000000},
                        {"date": "2024-01-04", "open": 102.5, "high": 104.0, "low": 102.0, "close": 103.6, "volume": 12000000},
                    ]
                });
                Ok(result.to_string())
            }
            "get_stock_info" => {
                let symbol = args["symbol"].as_str().unwrap_or("unknown");
                let result = json!({
                    "symbol": symbol,
                    "name": "贵州茅台",
                    "industry": "白酒",
                    "market_cap_billion": 2100.0,
                    "pe_ratio": 33.5,
                    "pb_ratio": 10.2,
                    "roe_percent": 30.5,
                    "revenue_growth_percent": 15.3,
                    "dividend_yield_percent": 1.8,
                    "total_shares_million": 1256.0,
                    "float_shares_million": 1256.0,
                });
                Ok(result.to_string())
            }
            "run_backtest" => {
                let strategy = args["strategy_name"].as_str().unwrap_or("unknown");
                let initial_capital = args["initial_capital"].as_f64().unwrap_or(1_000_000.0);
                let result = json!({
                    "strategy": strategy,
                    "initial_capital": initial_capital,
                    "final_value": initial_capital * 1.25,
                    "total_return_percent": 25.0,
                    "annualized_return_percent": 18.5,
                    "sharpe_ratio": 1.45,
                    "max_drawdown_percent": 12.3,
                    "win_rate_percent": 58.0,
                    "total_trades": 42,
                    "profit_factor": 1.85,
                });
                Ok(result.to_string())
            }
            "get_portfolio" => {
                let portfolio_id = args["portfolio_id"].as_str().unwrap_or("default");
                let result = json!({
                    "portfolio_id": portfolio_id,
                    "total_value": 1_250_000.0,
                    "cash": 350_000.0,
                    "total_pnl": 250_000.0,
                    "total_pnl_percent": 25.0,
                    "positions": [
                        {"symbol": "600519.SH", "name": "贵州茅台", "shares": 100, "avg_cost": 1800.0, "current_price": 1950.0, "pnl": 15000.0},
                        {"symbol": "000858.SZ", "name": "五粮液", "shares": 500, "avg_cost": 160.0, "current_price": 172.0, "pnl": 6000.0},
                    ]
                });
                Ok(result.to_string())
            }
            "screen_stocks" => {
                let market = args["market"].as_str().unwrap_or("ALL");
                let result = json!({
                    "market": market,
                    "total_matches": 3,
                    "results": [
                        {"symbol": "600519.SH", "name": "贵州茅台", "pe": 33.5, "pb": 10.2, "roe": 30.5, "market_cap_billion": 2100.0},
                        {"symbol": "000858.SZ", "name": "五粮液", "pe": 25.8, "pb": 7.1, "roe": 24.3, "market_cap_billion": 650.0},
                        {"symbol": "002304.SZ", "name": "洋河股份", "pe": 20.1, "pb": 4.5, "roe": 22.1, "market_cap_billion": 280.0},
                    ]
                });
                Ok(result.to_string())
            }
            other => anyhow::bail!("Unknown tool: {}", other),
        }
    }
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

pub fn get_kline_tool() -> ToolDefinition {
    tool(
        "get_kline",
        "Get K-line (candlestick) data for a given stock symbol over a date range and frequency.",
        json!({
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock code, e.g. 600519.SH or 000858.SZ"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "frequency": {
                    "type": "string",
                    "enum": ["1m", "5m", "15m", "30m", "60m", "daily", "weekly", "monthly"],
                    "description": "K-line frequency / period"
                }
            },
            "required": ["symbol", "start_date", "end_date", "frequency"]
        }),
    )
}

pub fn get_stock_info_tool() -> ToolDefinition {
    tool(
        "get_stock_info",
        "Get fundamental and descriptive information for a stock, including name, sector, market cap, P/E, P/B, and more.",
        json!({
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock code, e.g. 600519.SH"
                }
            },
            "required": ["symbol"]
        }),
    )
}

pub fn run_backtest_tool() -> ToolDefinition {
    tool(
        "run_backtest",
        "Run a backtest for a specified strategy with given parameters over a date range.",
        json!({
            "type": "object",
            "properties": {
                "strategy_name": {
                    "type": "string",
                    "description": "Name of the strategy to backtest"
                },
                "symbols": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of stock symbols to include"
                },
                "start_date": {
                    "type": "string",
                    "description": "Backtest start date in YYYY-MM-DD format"
                },
                "end_date": {
                    "type": "string",
                    "description": "Backtest end date in YYYY-MM-DD format"
                },
                "initial_capital": {
                    "type": "number",
                    "description": "Initial capital in CNY"
                },
                "parameters": {
                    "type": "object",
                    "description": "Strategy-specific parameters as key-value pairs"
                }
            },
            "required": ["strategy_name", "symbols", "start_date", "end_date", "initial_capital"]
        }),
    )
}

pub fn get_portfolio_tool() -> ToolDefinition {
    tool(
        "get_portfolio",
        "Get the current portfolio status including positions, P&L, and allocation.",
        json!({
            "type": "object",
            "properties": {
                "portfolio_id": {
                    "type": "string",
                    "description": "Portfolio identifier. Use 'default' for the main portfolio."
                }
            },
            "required": ["portfolio_id"]
        }),
    )
}

pub fn screen_stocks_tool() -> ToolDefinition {
    tool(
        "screen_stocks",
        "Screen A-share stocks by fundamental and technical criteria.",
        json!({
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "enum": ["SH", "SZ", "BJ", "ALL"],
                    "description": "Market to screen (Shanghai, Shenzhen, Beijing, or all)"
                },
                "min_pe": {
                    "type": "number",
                    "description": "Minimum P/E ratio"
                },
                "max_pe": {
                    "type": "number",
                    "description": "Maximum P/E ratio"
                },
                "min_pb": {
                    "type": "number",
                    "description": "Minimum P/B ratio"
                },
                "max_pb": {
                    "type": "number",
                    "description": "Maximum P/B ratio"
                },
                "min_market_cap": {
                    "type": "number",
                    "description": "Minimum market capitalization in billion CNY"
                },
                "min_roe": {
                    "type": "number",
                    "description": "Minimum ROE percentage"
                },
                "industry": {
                    "type": "string",
                    "description": "Filter by industry name (e.g. '白酒', '新能源', '半导体')"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["market_cap", "pe", "pb", "roe", "revenue_growth"],
                    "description": "Field to sort results by"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["market"]
        }),
    )
}

/// Returns all available tool definitions for LLM function calling.
pub fn get_all_tools() -> Vec<ToolDefinition> {
    vec![
        get_kline_tool(),
        get_stock_info_tool(),
        run_backtest_tool(),
        get_portfolio_tool(),
        screen_stocks_tool(),
    ]
}
