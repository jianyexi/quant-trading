use std::io::{self, Write};
use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use quant_api::{create_router, AppState};
use quant_config::AppConfig;
use quant_llm::{
    client::LlmClient,
    context::ConversationContext,
    tools::{get_all_tools, ToolExecutor},
};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "quant", about = "Quantitative Trading System for Chinese A-Shares")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to config file
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Market data operations
    Data {
        #[command(subcommand)]
        action: DataAction,
    },
    /// Backtesting operations
    Backtest {
        #[command(subcommand)]
        action: BacktestAction,
    },
    /// Trading operations
    Trade {
        #[command(subcommand)]
        action: TradeAction,
    },
    /// Portfolio management
    Portfolio {
        #[command(subcommand)]
        action: PortfolioAction,
    },
    /// Interactive LLM chat
    Chat,
    /// Start API server
    Serve,
}

#[derive(Subcommand)]
enum DataAction {
    /// Sync market data from remote sources
    Sync,
    /// Query local market data
    Query {
        /// Stock symbol (e.g. 600519.SH)
        #[arg(short, long)]
        symbol: String,
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        start: Option<String>,
        /// End date (YYYY-MM-DD)
        #[arg(long)]
        end: Option<String>,
    },
}

#[derive(Subcommand)]
enum BacktestAction {
    /// Run a backtest
    Run {
        /// Strategy name
        #[arg(short, long)]
        strategy: String,
        /// Stock symbol
        #[arg(long)]
        symbol: String,
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        start: String,
        /// End date (YYYY-MM-DD)
        #[arg(long)]
        end: String,
        /// Initial capital
        #[arg(long, default_value_t = 1_000_000.0)]
        capital: f64,
    },
    /// View backtest report
    Report {
        /// Backtest run ID
        #[arg(short, long)]
        id: String,
    },
}

#[derive(Subcommand)]
enum TradeAction {
    /// Start paper trading
    Paper {
        /// Strategy name
        #[arg(short, long)]
        strategy: String,
    },
    /// Start live trading
    Live {
        /// Strategy name
        #[arg(short, long)]
        strategy: String,
    },
}

#[derive(Subcommand)]
enum PortfolioAction {
    /// Show current portfolio
    Show,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Set up tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Load config
    let config = AppConfig::from_file(&cli.config)?;

    match cli.command {
        Commands::Data { action } => match action {
            DataAction::Sync => {
                println!("📦 Syncing market data...");
                println!("  Data sync not yet implemented. Connect a DataProvider to fetch from Tushare/AKShare.");
            }
            DataAction::Query { symbol, start, end } => {
                println!("🔍 Querying data for {symbol}");
                println!("  Start: {}", start.as_deref().unwrap_or("(all)"));
                println!("  End:   {}", end.as_deref().unwrap_or("(all)"));
                println!("  Query not yet implemented. Connect a DataProvider.");
            }
        },
        Commands::Backtest { action } => match action {
            BacktestAction::Run {
                strategy,
                symbol,
                start,
                end,
                capital,
            } => {
                println!("📊 Running backtest");
                println!("  Strategy: {strategy}");
                println!("  Symbol:   {symbol}");
                println!("  Period:   {start} → {end}");
                println!("  Capital:  ¥{capital:.2}");
                println!("  Backtest engine not yet connected.");
            }
            BacktestAction::Report { id } => {
                println!("📋 Backtest report for run: {id}");
                println!("  Report not yet implemented.");
            }
        },
        Commands::Trade { action } => match action {
            TradeAction::Paper { strategy } => {
                println!("📝 Starting paper trading with strategy: {strategy}");
                println!("  Paper trading not yet implemented.");
            }
            TradeAction::Live { strategy } => {
                println!("🚀 Starting live trading with strategy: {strategy}");
                println!("  ⚠️  Live trading not yet implemented.");
            }
        },
        Commands::Portfolio { action } => match action {
            PortfolioAction::Show => {
                println!("💼 Portfolio");
                println!("  Portfolio display not yet implemented.");
            }
        },
        Commands::Chat => {
            run_chat(&config).await;
        }
        Commands::Serve => {
            run_server(&config).await?;
        }
    }

    Ok(())
}

async fn run_chat(config: &AppConfig) {
    let client = LlmClient::new(
        &config.llm.api_url,
        &config.llm.api_key,
        &config.llm.model,
        config.llm.temperature,
        config.llm.max_tokens,
    );
    let mut context = ConversationContext::new("", 50);
    let tools = get_all_tools();
    let executor = ToolExecutor::new();

    println!("🤖 Quant Trading AI Assistant (type 'quit' to exit)");
    println!();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            println!("Goodbye! 👋");
            break;
        }

        context.add_user_message(input);

        // Chat loop with tool-call handling
        loop {
            let messages = context.get_messages();
            let response = match client.chat(&messages, Some(&tools)).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("❌ LLM error: {e}");
                    break;
                }
            };

            if let Some(choice) = response.choices.first() {
                let msg = &choice.message;

                // If the model wants to call tools
                if let Some(tool_calls) = &msg.tool_calls {
                    context.add_assistant_tool_calls(tool_calls.clone());

                    for tc in tool_calls {
                        println!("🔧 Calling tool: {} ...", tc.function.name);
                        match executor.execute(tc).await {
                            Ok(result) => {
                                context.add_tool_result(&tc.id, &result);
                            }
                            Err(e) => {
                                let err_msg = format!("Tool error: {e}");
                                context.add_tool_result(&tc.id, &err_msg);
                            }
                        }
                    }
                    // Loop back to send tool results to LLM
                    continue;
                }

                // Normal text response
                if let Some(content) = &msg.content {
                    println!("\n{content}\n");
                    context.add_assistant_message(content);
                }
            }
            break;
        }
    }
}

async fn run_server(config: &AppConfig) -> anyhow::Result<()> {
    let state = AppState {
        config: config.clone(),
    };

    let app = create_router(state);

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    println!("🚀 Starting API server on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
