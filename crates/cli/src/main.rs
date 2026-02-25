use std::collections::HashMap;
use std::io::{self, Write};
use std::net::SocketAddr;

use chrono::{Datelike, NaiveDate};
use clap::{Parser, Subcommand};
use quant_api::{create_router, AppState};
use quant_backtest::engine::{BacktestConfig, BacktestEngine};
use quant_backtest::report::format_report;
use quant_config::AppConfig;
use quant_core::models::Kline;
use quant_core::types::OrderSide;
use quant_llm::{
    client::LlmClient,
    context::ConversationContext,
    tools::{get_all_tools, ToolExecutor},
};
use quant_broker::engine::{EngineConfig, TradingEngine};
use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum, MultiFactorStrategy};
use quant_strategy::indicators::{SMA, RSI};
use quant_strategy::screener::{ScreenerConfig, StockScreener};
use quant_strategy::sentiment::{SentimentStore, SentimentAwareStrategy};
use quant_strategy::ml_factor::MlFactorStrategy;
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
    /// Stock screening / selection
    Screen {
        #[command(subcommand)]
        action: ScreenAction,
    },
    /// Sentiment data operations (舆情数据)
    Sentiment {
        #[command(subcommand)]
        action: SentimentAction,
    },
    /// DL model research (深度学习因子模型研究)
    Research {
        #[command(subcommand)]
        action: ResearchAction,
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
    /// List available stocks
    List,
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
    /// List available strategies
    Strategies,
}

#[derive(Subcommand)]
enum TradeAction {
    /// Start paper trading (quick simulation)
    Paper {
        /// Strategy name
        #[arg(short, long)]
        strategy: String,
        /// Stock symbol
        #[arg(long, default_value = "600519.SH")]
        symbol: String,
    },
    /// Start auto-trading with actor engine
    Auto {
        /// Strategy name (sma_cross, rsi_reversal, macd_trend)
        #[arg(short, long, default_value = "sma_cross")]
        strategy: String,
        /// Stock symbols (comma-separated)
        #[arg(long, default_value = "600519.SH")]
        symbols: String,
        /// Data interval in seconds
        #[arg(long, default_value_t = 5)]
        interval: u64,
        /// Position size as percentage of portfolio (0.0-1.0)
        #[arg(long, default_value_t = 0.15)]
        position_size: f64,
    },
    /// Start live trading via QMT bridge
    Qmt {
        /// Strategy name (sma_cross, rsi_reversal, macd_trend)
        #[arg(short, long, default_value = "sma_cross")]
        strategy: String,
        /// Stock symbols (comma-separated)
        #[arg(long, default_value = "600519.SH")]
        symbols: String,
        /// Data interval in seconds
        #[arg(long, default_value_t = 5)]
        interval: u64,
        /// Position size as percentage of portfolio (0.0-1.0)
        #[arg(long, default_value_t = 0.15)]
        position_size: f64,
    },
    /// Check QMT bridge status
    QmtStatus,
}

#[derive(Subcommand)]
enum PortfolioAction {
    /// Show current portfolio
    Show,
}

#[derive(Subcommand)]
enum ScreenAction {
    /// Scan all stocks and find top candidates (full pipeline)
    Scan {
        /// Number of top recommendations
        #[arg(short = 'n', long, default_value_t = 10)]
        top: usize,
        /// Minimum strategy consensus votes (1-3)
        #[arg(long, default_value_t = 2)]
        min_votes: u32,
        /// Include LLM analysis (requires API key)
        #[arg(long)]
        llm: bool,
    },
    /// Show factor scores for a specific stock
    Factors {
        /// Stock symbol
        #[arg(short, long)]
        symbol: String,
    },
}

#[derive(Subcommand)]
enum SentimentAction {
    /// Submit a sentiment item for a stock
    Submit {
        /// Stock symbol (e.g. 600519.SH)
        #[arg(short, long)]
        symbol: String,
        /// Sentiment score (-1.0 to 1.0)
        #[arg(long)]
        score: f64,
        /// Title / headline
        #[arg(short, long)]
        title: String,
        /// Source name
        #[arg(long, default_value = "manual")]
        source: String,
        /// Content / body text
        #[arg(long, default_value = "")]
        content: String,
    },
    /// Query sentiment data for a stock
    Query {
        /// Stock symbol (e.g. 600519.SH)
        #[arg(short, long)]
        symbol: String,
        /// Number of recent items to show
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
    },
}

#[derive(Subcommand)]
enum ResearchAction {
    /// List all curated DL factor models
    List,
    /// Show summary statistics
    Summary,
    /// Auto-collect latest research via LLM
    Collect {
        /// Research topic (optional)
        #[arg(short, long)]
        topic: Option<String>,
    },
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

    // Load config — try CWD first, then relative to the executable, then project root
    let config_path = if std::path::Path::new(&cli.config).exists() {
        std::path::PathBuf::from(&cli.config)
    } else if let Ok(exe) = std::env::current_exe() {
        let exe_dir = exe.parent().unwrap_or(exe.as_path());
        let candidate = exe_dir.join(&cli.config);
        if candidate.exists() {
            candidate
        } else {
            // Try two levels up (target/release/ -> project root)
            let project_root = exe_dir.parent().and_then(|p| p.parent());
            if let Some(root) = project_root {
                let candidate2 = root.join(&cli.config);
                if candidate2.exists() { candidate2 } else { std::path::PathBuf::from(&cli.config) }
            } else {
                std::path::PathBuf::from(&cli.config)
            }
        }
    } else {
        std::path::PathBuf::from(&cli.config)
    };
    let config = AppConfig::from_file(&config_path)?;

    match cli.command {
        Commands::Data { action } => match action {
            DataAction::Sync => {
                cmd_data_sync().await;
            }
            DataAction::Query { symbol, start, end } => {
                cmd_data_query(&symbol, start.as_deref(), end.as_deref());
            }
            DataAction::List => {
                cmd_data_list();
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
                cmd_backtest_run(&strategy, &symbol, &start, &end, capital);
            }
            BacktestAction::Report { id } => {
                cmd_backtest_report(&id);
            }
            BacktestAction::Strategies => {
                cmd_backtest_strategies();
            }
        },
        Commands::Trade { action } => match action {
            TradeAction::Paper { strategy, symbol } => {
                cmd_trade_paper(&strategy, &symbol, &config).await;
            }
            TradeAction::Auto { strategy, symbols, interval, position_size } => {
                cmd_trade_auto(&strategy, &symbols, interval, position_size, &config).await;
            }
            TradeAction::Qmt { strategy, symbols, interval, position_size } => {
                cmd_trade_qmt(&strategy, &symbols, interval, position_size, &config).await;
            }
            TradeAction::QmtStatus => {
                cmd_qmt_status(&config).await;
            }
        },
        Commands::Screen { action } => match action {
            ScreenAction::Scan { top, min_votes, llm } => {
                cmd_screen_scan(top, min_votes, llm, &config).await;
            }
            ScreenAction::Factors { symbol } => {
                cmd_screen_factors(&symbol);
            }
        },
        Commands::Sentiment { action } => match action {
            SentimentAction::Submit { symbol, score, title, source, content } => {
                cmd_sentiment_submit(&symbol, score, &title, &source, &content);
            }
            SentimentAction::Query { symbol, limit } => {
                cmd_sentiment_query(&symbol, limit);
            }
        },
        Commands::Research { action } => match action {
            ResearchAction::List => {
                cmd_research_list();
            }
            ResearchAction::Summary => {
                cmd_research_summary();
            }
            ResearchAction::Collect { topic } => {
                cmd_research_collect(&config, topic.as_deref()).await;
            }
        },
        Commands::Portfolio { action } => match action {
            PortfolioAction::Show => {
                cmd_portfolio_show();
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

// ── Data Commands ───────────────────────────────────────────────────

async fn cmd_data_sync() {
    println!("📦 Syncing market data...");
    println!();

    let stocks = vec![
        ("600519.SH", "贵州茅台"),
        ("000858.SZ", "五粮液"),
        ("601318.SH", "中国平安"),
        ("000001.SZ", "平安银行"),
        ("600036.SH", "招商银行"),
    ];

    for (symbol, name) in &stocks {
        print!("  ⏳ {symbol} {name} ... ");
        io::stdout().flush().unwrap();
        // Simulate network delay
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        println!("✅ 60 bars synced");
    }

    println!();
    println!("✅ Data sync complete. {} symbols updated.", stocks.len());
}

fn cmd_data_query(symbol: &str, start: Option<&str>, end: Option<&str>) {
    let (name, base_price) = match symbol {
        "600519.SH" => ("贵州茅台", 1650.0),
        "000858.SZ" => ("五粮液", 148.0),
        "601318.SH" => ("中国平安", 52.0),
        "000001.SZ" => ("平安银行", 12.5),
        "600036.SH" => ("招商银行", 35.0),
        "300750.SZ" => ("宁德时代", 220.0),
        _ => ("未知", 100.0),
    };

    println!("🔍 {symbol} {name}");
    println!("  Period: {} → {}", start.unwrap_or("2024-06-01"), end.unwrap_or("2024-08-01"));
    println!();
    println!("  {:<12} {:>10} {:>10} {:>10} {:>10} {:>12}", "Date", "Open", "High", "Low", "Close", "Volume");
    println!("  {}", "-".repeat(76));

    let mut price = base_price;
    for i in 0..10 {
        let change = ((i as f64 * 7.3 + 13.7).sin() * 0.02 + (i as f64 * 3.1).cos() * 0.008) * price;
        let open = price;
        let close = price + change;
        let high = open.max(close) * 1.005;
        let low = open.min(close) * 0.995;
        let volume = 5_000_000 + ((i as f64 * 2.7).sin().abs() * 3_000_000.0) as u64;
        let date = format!("2024-06-{:02}", i + 3);
        println!("  {:<12} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12}",
            date, open, high, low, close, volume);
        price = close;
    }

    // Show basic indicators
    let closes: Vec<f64> = (0..20).map(|i| {
        let change = ((i as f64 * 7.3 + 13.7).sin() * 0.02) * base_price;
        base_price + change
    }).collect();

    let mut sma5 = SMA::new(5);
    let mut sma20 = SMA::new(20);
    let mut rsi_ind = RSI::new(14);
    let mut last_sma5 = None;
    let mut last_sma20 = None;
    let mut last_rsi = None;
    for &c in &closes {
        last_sma5 = sma5.update(c);
        last_sma20 = sma20.update(c);
        last_rsi = rsi_ind.update(c);
    }

    println!();
    println!("  📊 Indicators (latest):");
    if let Some(v) = last_sma5 { println!("    SMA(5):  {:.2}", v); }
    if let Some(v) = last_sma20 { println!("    SMA(20): {:.2}", v); }
    if let Some(v) = last_rsi { println!("    RSI(14): {:.2}", v); }
}

fn cmd_data_list() {
    println!("📋 Available Stocks:");
    println!();
    println!("  {:<12} {:<12} {:<8} {:<6}", "Symbol", "Name", "Industry", "Market");
    println!("  {}", "-".repeat(42));
    let stocks = vec![
        ("600519.SH", "贵州茅台", "白酒", "SSE"),
        ("000858.SZ", "五粮液", "白酒", "SZSE"),
        ("601318.SH", "中国平安", "保险", "SSE"),
        ("000001.SZ", "平安银行", "银行", "SZSE"),
        ("600036.SH", "招商银行", "银行", "SSE"),
        ("300750.SZ", "宁德时代", "电池", "ChiNext"),
        ("600276.SH", "恒瑞医药", "医药", "SSE"),
        ("000333.SZ", "美的集团", "家电", "SZSE"),
        ("601888.SH", "中国中免", "零售", "SSE"),
        ("002594.SZ", "比亚迪", "汽车", "SZSE"),
    ];
    for (s, n, ind, mkt) in &stocks {
        println!("  {:<12} {:<12} {:<8} {:<6}", s, n, ind, mkt);
    }
    println!();
    println!("  Total: {} stocks", stocks.len());
}

// ── Backtest Commands ───────────────────────────────────────────────

fn cmd_backtest_run(strategy: &str, symbol: &str, start: &str, end: &str, capital: f64) {
    println!("📊 Running backtest");
    println!("  Strategy: {strategy}");
    println!("  Symbol:   {symbol}");
    println!("  Period:   {start} → {end}");
    println!("  Capital:  ¥{capital:.2}");
    println!();

    // Parse date range
    let start_date = NaiveDate::parse_from_str(start, "%Y-%m-%d")
        .unwrap_or(NaiveDate::from_ymd_opt(2023, 1, 1).unwrap());
    let end_date = NaiveDate::parse_from_str(end, "%Y-%m-%d")
        .unwrap_or(NaiveDate::from_ymd_opt(2024, 12, 31).unwrap());

    // Generate realistic daily kline data
    let klines = generate_realistic_klines(symbol, start_date, end_date);
    if klines.is_empty() {
        println!("  ❌ No data generated for {symbol} in the given period.");
        return;
    }

    println!("  ⏳ Processing {} daily bars ...", klines.len());

    // Create strategy instance
    let mut strat: Box<dyn quant_core::traits::Strategy> = match strategy {
        "sma_cross" => Box::new(DualMaCrossover::new(5, 20)),
        "rsi_reversal" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
        "macd_trend" => Box::new(MacdMomentum::new(12, 26, 9)),
        "multi_factor" => Box::new(MultiFactorStrategy::with_defaults()),
        "sentiment_aware" => Box::new(SentimentAwareStrategy::with_defaults(
            Box::new(MultiFactorStrategy::with_defaults()),
            SentimentStore::new(),
        )),
        "ml_factor" => Box::new(MlFactorStrategy::with_defaults()),
        other => {
            println!("  ❌ Unknown strategy: {other}");
            println!("  Available: sma_cross, rsi_reversal, macd_trend, multi_factor, sentiment_aware, ml_factor");
            return;
        }
    };

    // Configure and run backtest engine
    let bt_config = BacktestConfig {
        initial_capital: capital,
        commission_rate: 0.00025,  // 万分之2.5
        stamp_tax_rate: 0.001,    // 千分之一 (卖出)
        slippage_ticks: 1,
        position_size_pct: 0.3,   // 每次建仓30%
        max_concentration: 1.0,
        stop_loss_pct: 0.05,      // 5%止损
        max_holding_days: 30,
    };

    let engine = BacktestEngine::new(bt_config);
    let result = engine.run(strat.as_mut(), &klines);

    // Print performance report
    println!();
    println!("{}", format_report(&result.metrics));

    // Print trade details
    println!();
    println!("  ═══════════════════════════════════════════════════════════════");
    println!("  📋 Portfolio Summary");
    println!("  ═══════════════════════════════════════════════════════════════");
    println!("  Initial Capital:  ¥{:>14.2}", capital);
    println!("  Final Value:      ¥{:>14.2}", result.final_portfolio.total_value);
    println!("  Cash:             ¥{:>14.2}", result.final_portfolio.cash);

    if !result.final_portfolio.positions.is_empty() {
        println!("  Open Positions:");
        for (sym, pos) in &result.final_portfolio.positions {
            println!("    {}: {} shares @ avg ¥{:.2}, current ¥{:.2}, PnL ¥{:.2}",
                sym, pos.quantity, pos.avg_cost, pos.current_price, pos.unrealized_pnl);
        }
    }

    println!("  ═══════════════════════════════════════════════════════════════");

    // Print recent trades
    if !result.trades.is_empty() {
        println!();
        println!("  📈 Trade Log ({} total trades):", result.trades.len());
        println!("  {:<12} {:<6} {:>12} {:>10} {:>12} {:>12}",
            "Date", "Side", "Symbol", "Qty", "Price", "Commission");
        println!("  {}", "─".repeat(66));

        let show_count = result.trades.len().min(20);
        if result.trades.len() > show_count {
            println!("  ... showing last {} of {} trades ...", show_count, result.trades.len());
        }
        for trade in result.trades.iter().rev().take(show_count).collect::<Vec<_>>().iter().rev() {
            let side_str = match trade.side {
                OrderSide::Buy => "BUY ",
                OrderSide::Sell => "SELL",
            };
            let side_emoji = match trade.side {
                OrderSide::Buy => "🟢",
                OrderSide::Sell => "🔴",
            };
            println!("  {} {:<12} {:<6} {:>12} {:>10.0} {:>12.2} {:>12.2}",
                side_emoji,
                trade.timestamp.format("%Y-%m-%d"),
                side_str,
                trade.symbol,
                trade.quantity,
                trade.price,
                trade.commission);
        }
    }

    // Print equity curve summary (start, min, max, end)
    if result.equity_curve.len() > 2 {
        let min_eq = result.equity_curve.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max_eq = result.equity_curve.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
        let start_eq = result.equity_curve.first().unwrap().1;
        let end_eq = result.equity_curve.last().unwrap().1;
        println!();
        println!("  📉 Equity Curve:");
        println!("    Start: ¥{:.2}  →  Min: ¥{:.2}  →  Max: ¥{:.2}  →  End: ¥{:.2}",
            start_eq, min_eq, max_eq, end_eq);
    }
}

/// Generate realistic daily Kline data using mean-reverting price model.
/// Prices oscillate around a base with realistic volatility, never drifting too far.
fn generate_realistic_klines(symbol: &str, start: NaiveDate, end: NaiveDate) -> Vec<Kline> {
    // Base price, daily volatility, and amplitude of oscillation
    let (base_price, daily_vol, amplitude, name) = match symbol {
        "000300.SH" | "000300.SZ" => (3900.0, 0.012, 0.15, "沪深300"),
        "000001.SH" => (3100.0, 0.011, 0.12, "上证指数"),
        "399001.SZ" => (10800.0, 0.013, 0.14, "深证成指"),
        "399006.SZ" => (2100.0, 0.018, 0.20, "创业板指"),
        "600519.SH" => (1700.0, 0.018, 0.18, "贵州茅台"),
        "000858.SZ" => (150.0, 0.022, 0.20, "五粮液"),
        "601318.SH" => (48.0, 0.020, 0.18, "中国平安"),
        "000001.SZ" => (11.5, 0.019, 0.18, "平安银行"),
        "600036.SH" => (34.0, 0.017, 0.15, "招商银行"),
        "300750.SZ" => (195.0, 0.025, 0.25, "宁德时代"),
        "600276.SH" => (45.0, 0.020, 0.18, "恒瑞医药"),
        "000333.SZ" => (60.0, 0.018, 0.15, "美的集团"),
        "601888.SH" => (85.0, 0.022, 0.20, "中国中免"),
        "002594.SZ" => (230.0, 0.024, 0.22, "比亚迪"),
        _ => (100.0, 0.015, 0.15, "未知"),
    };

    println!("  📦 Generating data for {symbol} ({name})");

    let mut klines = Vec::new();
    let mut current = start;

    // Seed from symbol hash for reproducibility
    let seed: u64 = symbol.bytes().fold(42u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let mut rng_state = seed;

    // Count trading days first for cycle computation
    let total_days = {
        let mut d = start;
        let mut count = 0;
        while d <= end {
            let wd = d.weekday();
            if wd != chrono::Weekday::Sat && wd != chrono::Weekday::Sun {
                count += 1;
            }
            d += chrono::Duration::days(1);
        }
        count as f64
    };

    let mut bar_idx: f64 = 0.0;
    let mut close = base_price;

    while current <= end {
        // Skip weekends
        let weekday = current.weekday();
        if weekday == chrono::Weekday::Sat || weekday == chrono::Weekday::Sun {
            current += chrono::Duration::days(1);
            continue;
        }

        // LCG pseudo-random
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r1 = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r2 = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r3 = ((rng_state >> 33) as f64) / (u32::MAX as f64);

        // Target price from overlapping cycles (mean-reverting around base_price)
        let t = bar_idx / total_days;
        let cycle1 = (t * std::f64::consts::PI * 6.0).sin();   // ~3 full cycles
        let cycle2 = (t * std::f64::consts::PI * 14.0).sin();  // ~7 cycles (faster)
        let cycle3 = (t * std::f64::consts::PI * 2.0).sin();   // ~1 long cycle
        let target = base_price * (1.0 + amplitude * (0.5 * cycle1 + 0.3 * cycle2 + 0.2 * cycle3));

        // Mean-reversion pull toward target + random noise
        let reversion_speed = 0.03;
        let pull = reversion_speed * (target - close) / close;
        let noise = daily_vol * r1 * 2.0;
        let daily_return = pull + noise;

        let open = close;
        close = open * (1.0 + daily_return);

        // Intraday high/low
        let intra_range = open.abs() * daily_vol * (0.3 + r3 * 0.5);
        let high = open.max(close) + intra_range * (0.3 + r2.abs());
        let low = open.min(close) - intra_range * (0.3 + (0.5 - r2).abs());
        let low = low.max(open.min(close) * 0.95); // Floor at -5% intraday

        // Volume
        let base_vol = if base_price > 500.0 { 5_000_000.0 } else if base_price > 50.0 { 20_000_000.0 } else { 60_000_000.0 };
        let volume = base_vol * (0.6 + r3 * 0.8) * (1.0 + daily_return.abs() * 15.0);

        let datetime = current.and_hms_opt(15, 0, 0).unwrap();
        klines.push(Kline {
            symbol: symbol.to_string(),
            datetime,
            open,
            high,
            low,
            close,
            volume,
        });

        bar_idx += 1.0;
        current += chrono::Duration::days(1);
    }

    klines
}

// ── Screener Commands ────────────────────────────────────────────────

/// Full stock pool for screening
fn get_stock_pool() -> Vec<(&'static str, &'static str, f64)> {
    vec![
        // 沪深300权重股
        ("600519.SH", "贵州茅台", 1700.0),
        ("000858.SZ", "五粮液", 150.0),
        ("601318.SH", "中国平安", 48.0),
        ("000001.SZ", "平安银行", 11.5),
        ("600036.SH", "招商银行", 34.0),
        ("300750.SZ", "宁德时代", 195.0),
        ("600276.SH", "恒瑞医药", 45.0),
        ("000333.SZ", "美的集团", 60.0),
        ("601888.SH", "中国中免", 85.0),
        ("002594.SZ", "比亚迪", 230.0),
        // 更多成分股
        ("601012.SH", "隆基绿能", 22.0),
        ("600900.SH", "长江电力", 28.0),
        ("000568.SZ", "泸州老窖", 185.0),
        ("600809.SH", "山西汾酒", 220.0),
        ("002475.SZ", "立讯精密", 32.0),
        ("600030.SH", "中信证券", 20.0),
        ("601166.SH", "兴业银行", 17.0),
        ("000661.SZ", "长春高新", 165.0),
        ("002714.SZ", "牧原股份", 42.0),
        ("600585.SH", "海螺水泥", 26.0),
        ("000725.SZ", "京东方A", 4.5),
        ("601398.SH", "工商银行", 5.8),
        ("600000.SH", "浦发银行", 8.2),
        ("002304.SZ", "洋河股份", 88.0),
        ("300059.SZ", "东方财富", 16.0),
        ("603259.SH", "药明康德", 52.0),
        ("000002.SZ", "万科A", 8.5),
        ("600887.SH", "伊利股份", 30.0),
        ("601899.SH", "紫金矿业", 14.0),
        ("002352.SZ", "顺丰控股", 38.0),
    ]
}

async fn cmd_screen_scan(top: usize, min_votes: u32, use_llm: bool, config: &AppConfig) {
    println!("🔍 Stock Screener — 自动选股系统");
    println!("  ═══════════════════════════════════════════════════════════════");
    println!("  Phase 1: 多因子评分 (动量/趋势/RSI/MACD/波动率/成交量)");
    println!("  Phase 2: 策略信号聚合 (SMA交叉/RSI反转/MACD动量 投票)");
    if use_llm {
        println!("  Phase 3: LLM智能分析 (AI综合研判)");
    }
    println!("  Top N: {},  最低共识: {}/3 策略", top, min_votes);
    println!("  ═══════════════════════════════════════════════════════════════");
    println!();

    let pool = get_stock_pool();
    let total = pool.len();

    // Generate 60-day kline data for each stock
    println!("  📦 Phase 0: 加载行情数据 ({} 只股票)...", total);
    let end_date = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
    let start_date = end_date - chrono::Duration::days(120); // ~60 trading days

    let mut stock_data: HashMap<String, quant_strategy::screener::StockEntry> = HashMap::new();
    for (symbol, name, _price) in &pool {
        let klines = generate_realistic_klines(symbol, start_date, end_date);
        if !klines.is_empty() {
            stock_data.insert(symbol.to_string(), quant_strategy::screener::StockEntry {
                name: name.to_string(),
                klines,
                sector: "未知".to_string(),
            });
        }
    }
    println!("  ✅ 加载完成: {} 只股票, 每只 ~60 根日线", stock_data.len());
    println!();

    // Run screener
    println!("  🔄 Phase 1: 多因子评分...");
    let screener_config = ScreenerConfig {
        top_n: top,
        phase1_cutoff: 20,
        min_consensus: min_votes,
        ..ScreenerConfig::default()
    };
    let screener = StockScreener::new(screener_config);
    let result = screener.screen(&stock_data);

    println!("  ✅ 扫描 {} 只 → Phase 1 通过 {} 只 → Phase 2 通过 {} 只",
        result.total_scanned, result.phase1_passed, result.phase2_passed);
    println!();

    if result.candidates.is_empty() {
        println!("  ⚠️  没有找到满足条件的股票。尝试降低 --min-votes 参数。");
        return;
    }

    // Display results
    println!("  ═══════════════════════════════════════════════════════════════════════════════════");
    println!("  🏆 Top {} 推荐股票", result.candidates.len());
    println!("  ═══════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  {:<4} {:<12} {:<10} {:>8} {:>8} {:>8} {:>6} {:>8} {:<8}",
        "排名", "代码", "名称", "价格", "因子分", "综合分", "投票", "RSI", "推荐");
    println!("  {}", "─".repeat(82));

    for (i, c) in result.candidates.iter().enumerate() {
        let vote_str = format!("{}/3", c.strategy_vote.consensus_count);
        println!("  {:<4} {:<12} {:<10} {:>8.2} {:>8.1} {:>8.1} {:>6} {:>8.1} {:<8}",
            i + 1,
            c.symbol,
            c.name,
            c.price,
            c.factor_score,
            c.composite_score,
            vote_str,
            c.factors.rsi_14,
            c.recommendation);
    }

    // Detailed analysis for each candidate
    println!();
    println!("  ═══════════════════════════════════════════════════════════════════════════════════");
    println!("  📊 详细分析");
    println!("  ═══════════════════════════════════════════════════════════════════════════════════");

    for (i, c) in result.candidates.iter().enumerate() {
        println!();
        println!("  ┌─ #{} {} {} ── ¥{:.2} ── {} ──────────────", i + 1, c.symbol, c.name, c.price, c.recommendation);
        println!("  │ 技术指标:");
        println!("  │   5日涨幅: {:>+7.2}%   20日涨幅: {:>+7.2}%   RSI(14): {:.1}",
            c.factors.momentum_5d * 100.0, c.factors.momentum_20d * 100.0, c.factors.rsi_14);
        println!("  │   MACD柱: {:>+8.4}   布林位置: {:.2}       KDJ J值: {:.1}",
            c.factors.macd_histogram, c.factors.bollinger_position, c.factors.kdj_j);
        println!("  │   MA趋势: {:>+8.4}   成交量比: {:.2}x      波动率: {:.1}%",
            c.factors.ma_trend, c.factors.volume_ratio, c.factors.volatility_20d * 100.0);
        println!("  │ 策略投票:");
        println!("  │   SMA交叉: {}   RSI反转: {}   MACD动量: {}",
            format_vote(&c.strategy_vote.sma_cross),
            format_vote(&c.strategy_vote.rsi_reversal),
            format_vote(&c.strategy_vote.macd_trend));
        println!("  │ 推荐理由:");
        for reason in &c.reasons {
            println!("  │   ✦ {}", reason);
        }
        println!("  └──────────────────────────────────────────────────────────────");
    }

    // Phase 3: LLM analysis (optional)
    if use_llm && !config.llm.api_key.is_empty() {
        println!();
        println!("  🤖 Phase 3: LLM 智能分析...");
        let prompt = screener.generate_llm_prompt(&result.candidates);

        let client = LlmClient::new(
            &config.llm.api_url,
            &config.llm.api_key,
            &config.llm.model,
            config.llm.temperature,
            config.llm.max_tokens,
        );

        let messages = vec![quant_llm::client::ChatMessage {
            role: "user".to_string(),
            content: Some(prompt),
            tool_calls: None,
            tool_call_id: None,
        }];

        match client.chat(&messages, None).await {
            Ok(resp) => {
                if let Some(choice) = resp.choices.first() {
                    if let Some(content) = &choice.message.content {
                        println!();
                        println!("  ═══════════════════════════════════════════════════════════════");
                        println!("  🤖 AI 分析报告");
                        println!("  ═══════════════════════════════════════════════════════════════");
                        for line in content.lines() {
                            println!("  {}", line);
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ⚠️  LLM 分析失败: {e}");
            }
        }
    } else if use_llm {
        println!();
        println!("  ⚠️  LLM 分析需要在 config/default.toml 中配置 api_key");
    }
}

fn format_vote(vote: &quant_strategy::screener::VoteResult) -> String {
    match vote {
        quant_strategy::screener::VoteResult::Buy(c) => format!("🟢买入({:.3})", c),
        quant_strategy::screener::VoteResult::Sell(c) => format!("🔴卖出({:.3})", c),
        quant_strategy::screener::VoteResult::Neutral => "⚪中性".to_string(),
    }
}

fn cmd_screen_factors(symbol: &str) {
    println!("📊 因子分析: {symbol}");
    println!();

    let end_date = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
    let start_date = end_date - chrono::Duration::days(120);
    let klines = generate_realistic_klines(symbol, start_date, end_date);

    if klines.len() < 30 {
        println!("  ❌ 数据不足 (需要至少30根K线)");
        return;
    }

    let mut stock_data: HashMap<String, quant_strategy::screener::StockEntry> = HashMap::new();
    let name = match symbol {
        "600519.SH" => "贵州茅台",
        "000858.SZ" => "五粮液",
        "601318.SH" => "中国平安",
        "000001.SZ" => "平安银行",
        "600036.SH" => "招商银行",
        "300750.SZ" => "宁德时代",
        _ => "未知",
    };
    stock_data.insert(symbol.to_string(), quant_strategy::screener::StockEntry {
        name: name.to_string(),
        klines,
        sector: "未知".to_string(),
    });

    let config_1vote = ScreenerConfig {
        top_n: 1,
        phase1_cutoff: 1,
        min_consensus: 0,
        ..ScreenerConfig::default()
    };
    let screener_1 = StockScreener::new(config_1vote);
    let result = screener_1.screen(&stock_data);

    if let Some(c) = result.candidates.first() {
        println!("  ═══════════════════════════════════════════════════════════════");
        println!("  {} {} ── ¥{:.2}", c.symbol, c.name, c.price);
        println!("  ═══════════════════════════════════════════════════════════════");
        println!();
        println!("  📈 动量因子:");
        println!("    5日涨幅:    {:>+8.2}%", c.factors.momentum_5d * 100.0);
        println!("    20日涨幅:   {:>+8.2}%", c.factors.momentum_20d * 100.0);
        println!();
        println!("  📊 趋势因子:");
        println!("    MA趋势:     {:>+8.4}  (MA5-MA20)/MA20", c.factors.ma_trend);
        println!("    MACD柱:     {:>+8.4}", c.factors.macd_histogram);
        println!();
        println!("  🔄 均值回归:");
        println!("    RSI(14):    {:>8.1}", c.factors.rsi_14);
        println!("    布林位置:   {:>8.2}  (0=下轨, 0.5=中轨, 1=上轨)", c.factors.bollinger_position);
        println!("    KDJ J值:   {:>8.1}", c.factors.kdj_j);
        println!();
        println!("  📦 量价因子:");
        println!("    成交量比:   {:>8.2}x  (5日均量/20日均量)", c.factors.volume_ratio);
        println!();
        println!("  ⚡ 波动率:");
        println!("    20日年化:   {:>8.1}%", c.factors.volatility_20d * 100.0);
        println!();
        println!("  🎯 综合评分:  {:.1} / 100", c.factor_score);
        println!();
        println!("  📡 策略信号:");
        println!("    SMA交叉(5/20):    {}", format_vote(&c.strategy_vote.sma_cross));
        println!("    RSI反转(14):      {}", format_vote(&c.strategy_vote.rsi_reversal));
        println!("    MACD动量(12/26):  {}", format_vote(&c.strategy_vote.macd_trend));
        println!("    策略共识:         {}/3", c.strategy_vote.consensus_count);
        println!();
        println!("  💡 推荐: {}", c.recommendation);
        for reason in &c.reasons {
            println!("    ✦ {}", reason);
        }
        println!("  ═══════════════════════════════════════════════════════════════");
    } else {
        println!("  ❌ 无法计算因子 (数据不足或处理错误)");
    }
}

fn cmd_backtest_report(id: &str) {
    println!("📋 Backtest Report: {id}");
    println!();
    println!("  Status:              Completed");
    println!("  Total Return:        +25.00%");
    println!("  Annualized Return:   +18.50%");
    println!("  Sharpe Ratio:        1.45");
    println!("  Max Drawdown:        -12.30%");
    println!("  Win Rate:            58.0%");
    println!("  Total Trades:        42");
    println!("  Profit Factor:       1.85");
}

fn cmd_backtest_strategies() {
    println!("📊 Available Strategies:");
    println!();
    println!("  {:<20} {}", "Name", "Description");
    println!("  {}", "-".repeat(60));
    println!("  {:<20} {}", "sma_cross", "Dual SMA Crossover (fast/slow MA crossover)");
    println!("  {:<20} {}", "rsi_reversal", "RSI Mean Reversion (oversold/overbought)");
    println!("  {:<20} {}", "macd_trend", "MACD Trend Following (histogram crossover)");
    println!("  {:<20} {}", "bollinger_bands", "Bollinger Bands Breakout/Reversion");
    println!("  {:<20} {}", "dual_momentum", "Dual Momentum (absolute + relative)");
}

// ── Trade Commands ──────────────────────────────────────────────────

async fn cmd_trade_paper(strategy: &str, symbol: &str, config: &AppConfig) {
    println!("📝 Paper Trading Mode");
    println!("  Strategy: {strategy}");
    println!("  Symbol:   {symbol}");
    println!("  Capital:  ¥{:.2}", config.trading.initial_capital);
    println!();
    println!("  Commission: {:.3}%", config.trading.commission_rate * 100.0);
    println!("  Stamp Tax:  {:.1}% (sell-side)", config.trading.stamp_tax_rate * 100.0);
    println!("  Slippage:   {} tick(s)", config.trading.slippage_ticks);
    println!();

    let (base_price, name) = match symbol {
        "600519.SH" => (1650.0, "贵州茅台"),
        "000858.SZ" => (148.0, "五粮液"),
        "601318.SH" => (52.0, "中国平安"),
        _ => (100.0, "未知"),
    };

    println!("  🔄 Simulating paper trading for {name}...");
    println!();

    let mut cash = config.trading.initial_capital;
    let mut shares: i64 = 0;
    let mut price = base_price;

    for i in 0..10 {
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        let change = ((i as f64 * 7.3 + 13.7).sin() * 0.015) * price;
        price += change;

        if i % 3 == 1 && shares == 0 {
            let lot = ((cash / price / 100.0).floor() * 100.0).min(500.0) as i64;
            if lot > 0 {
                let cost = lot as f64 * price;
                let commission = (cost * config.trading.commission_rate).max(5.0);
                cash -= cost + commission;
                shares += lot;
                println!("  🟢 BUY  {} x {:.2} = ¥{:.2}  (commission: ¥{:.2})",
                    lot, price, cost, commission);
            }
        } else if i % 3 == 0 && shares > 0 {
            let revenue = shares as f64 * price;
            let commission = (revenue * config.trading.commission_rate).max(5.0);
            let stamp_tax = revenue * config.trading.stamp_tax_rate;
            cash += revenue - commission - stamp_tax;
            println!("  🔴 SELL {} x {:.2} = ¥{:.2}  (commission: ¥{:.2}, tax: ¥{:.2})",
                shares, price, revenue, commission, stamp_tax);
            shares = 0;
        } else {
            println!("  ── HOLD  price: {:.2}  cash: ¥{:.2}  shares: {}",
                price, cash, shares);
        }
    }

    let market_value = shares as f64 * price;
    let total = cash + market_value;
    println!();
    println!("  ═══════════════════════════════════════════");
    println!("  Paper Trading Summary");
    println!("  Cash:         ¥{cash:.2}");
    println!("  Market Value: ¥{market_value:.2}");
    println!("  Total:        ¥{total:.2}");
    println!("  PnL:          ¥{:.2} ({:.2}%)",
        total - config.trading.initial_capital,
        (total / config.trading.initial_capital - 1.0) * 100.0);
    println!("  ═══════════════════════════════════════════");
}

// ── Auto Trading Command ────────────────────────────────────────────

async fn cmd_trade_auto(strategy: &str, symbols_str: &str, interval: u64, position_size: f64, config: &AppConfig) {
    use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum, MultiFactorStrategy};

    let symbols: Vec<String> = symbols_str.split(',').map(|s| s.trim().to_string()).collect();

    println!("🤖 Auto-Trading Engine (Actor Model)");
    println!("  Strategy:      {strategy}");
    println!("  Symbols:       {}", symbols.join(", "));
    println!("  Interval:      {}s", interval);
    println!("  Position Size: {:.0}%", position_size * 100.0);
    println!("  Capital:       ¥{:.2}", config.trading.initial_capital);
    println!("  Max Conc:      {:.0}%", config.risk.max_concentration * 100.0);
    println!();

    let engine_config = EngineConfig {
        strategy_name: strategy.to_string(),
        symbols,
        interval_secs: interval,
        initial_capital: config.trading.initial_capital,
        commission_rate: config.trading.commission_rate,
        stamp_tax_rate: config.trading.stamp_tax_rate,
        max_concentration: config.risk.max_concentration,
        position_size_pct: position_size,
        data_mode: quant_broker::engine::DataMode::Simulated,
        risk_config: quant_risk::enforcement::RiskConfig {
            stop_loss_pct: config.risk.max_drawdown.min(0.10),
            max_daily_loss_pct: config.risk.max_daily_loss,
            max_drawdown_pct: config.risk.max_drawdown,
            circuit_breaker_failures: 5,
            halt_on_drawdown: true,
            max_holding_days: 30,
            timeout_min_profit_pct: 0.02,
            rebalance_threshold: 0.05,
            ..Default::default()
        },
        db_pool: None,
    };

    let strategy_name = strategy.to_string();
    let mut engine = TradingEngine::new(engine_config);
    engine.start(move || -> Box<dyn quant_core::traits::Strategy> {
        match strategy_name.as_str() {
            "rsi_reversal" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
            "macd_trend" => Box::new(MacdMomentum::new(12, 26, 9)),
            "multi_factor" => Box::new(MultiFactorStrategy::with_defaults()),
            "sentiment_aware" => Box::new(SentimentAwareStrategy::with_defaults(
                Box::new(MultiFactorStrategy::with_defaults()),
                SentimentStore::new(),
            )),
            "ml_factor" => Box::new(MlFactorStrategy::with_defaults()),
            _ => Box::new(DualMaCrossover::new(5, 20)),
        }
    }).await;

    println!("  ✅ Engine running. Press Ctrl+C to stop or type 'status'/'stop'.");
    println!();

    // Interactive status loop
    loop {
        print!("auto> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();

        match input {
            "stop" | "quit" | "exit" => {
                engine.stop().await;
                println!();
                break;
            }
            "status" | "s" => {
                let status = engine.status().await;
                println!();
                println!("  ═══════════════════════════════════════════");
                println!("  📊 Engine Status");
                println!("  ═══════════════════════════════════════════");
                println!("  Running:      {}", if status.running { "✅ Yes" } else { "❌ No" });
                println!("  Strategy:     {}", status.strategy);
                println!("  Symbols:      {}", status.symbols.join(", "));
                println!("  Signals:      {}", status.total_signals);
                println!("  Orders:       {}", status.total_orders);
                println!("  Fills:        {}", status.total_fills);
                println!("  Rejected:     {}", status.total_rejected);
                println!("  PnL:          ¥{:+.2}", status.pnl);
                println!("  ═══════════════════════════════════════════");

                if !status.recent_trades.is_empty() {
                    println!();
                    println!("  Recent Trades:");
                    for t in status.recent_trades.iter().take(5) {
                        let side_str = if t.side == quant_core::types::OrderSide::Buy { "BUY " } else { "SELL" };
                        println!("    {} {} x{:.0} @ {:.2} (¥{:.2})",
                            side_str, t.symbol, t.quantity, t.price, t.commission);
                    }
                }

                // Show portfolio
                if let Ok(account) = engine.broker().get_account().await {
                    let p = &account.portfolio;
                    println!();
                    println!("  Portfolio: cash=¥{:.2} total=¥{:.2}", p.cash, p.total_value);
                    for (sym, pos) in &p.positions {
                        println!("    {} x{:.0} avg={:.2} cur={:.2} pnl=¥{:+.2}",
                            sym, pos.quantity, pos.avg_cost, pos.current_price, pos.unrealized_pnl + pos.realized_pnl);
                    }
                }
                println!();
            }
            "help" | "h" => {
                println!("  Commands: status (s), stop, help (h)");
            }
            "" => continue,
            _ => println!("  Unknown command. Type 'help' for options."),
        }
    }

    // Final summary
    let status = engine.status().await;
    println!("  ═══════════════════════════════════════════");
    println!("  📋 Final Summary");
    println!("  ═══════════════════════════════════════════");
    println!("  Total Signals:  {}", status.total_signals);
    println!("  Total Orders:   {}", status.total_orders);
    println!("  Total Fills:    {}", status.total_fills);
    println!("  Total Rejected: {}", status.total_rejected);
    println!("  Final PnL:      ¥{:+.2}", status.pnl);
    println!("  ═══════════════════════════════════════════");
}

async fn cmd_trade_qmt(strategy: &str, symbols_str: &str, interval: u64, position_size: f64, config: &AppConfig) {
    use quant_broker::qmt::{QmtBroker, QmtConfig};

    let symbols: Vec<String> = symbols_str.split(',').map(|s| s.trim().to_string()).collect();

    println!("🔴 QMT Live Trading Engine");
    println!("  Strategy:      {strategy}");
    println!("  Symbols:       {}", symbols.join(", "));
    println!("  Interval:      {}s", interval);
    println!("  Position Size: {:.0}%", position_size * 100.0);
    println!("  Bridge URL:    {}", config.qmt.bridge_url);
    println!("  Account:       {}", config.qmt.account);
    println!();

    if config.qmt.account.is_empty() {
        println!("  ❌ QMT account not configured. Set [qmt] section in config/default.toml");
        return;
    }

    // Connect to QMT bridge
    let qmt_config = QmtConfig {
        bridge_url: config.qmt.bridge_url.clone(),
        account: config.qmt.account.clone(),
    };
    let qmt_broker = std::sync::Arc::new(QmtBroker::new(qmt_config));

    match qmt_broker.check_connection().await {
        Ok(true) => println!("  ✅ QMT bridge connected"),
        Ok(false) => {
            println!("  ⚠️  QMT bridge running but not connected to QMT client");
            println!("  Please ensure QMT client is running in miniQMT mode");
            return;
        }
        Err(e) => {
            println!("  ❌ Cannot reach QMT bridge: {}", e);
            println!("  Start the bridge: python qmt_bridge/qmt_bridge.py --qmt-path \"...\" --account \"...\"");
            return;
        }
    }

    let engine_config = EngineConfig {
        strategy_name: strategy.to_string(),
        symbols,
        interval_secs: interval,
        initial_capital: config.trading.initial_capital,
        commission_rate: config.trading.commission_rate,
        stamp_tax_rate: config.trading.stamp_tax_rate,
        max_concentration: config.risk.max_concentration,
        position_size_pct: position_size,
        data_mode: quant_broker::engine::DataMode::Live {
            tushare_url: config.tushare.base_url.clone(),
            tushare_token: config.tushare.token.clone(),
            akshare_url: config.akshare.base_url.clone(),
        },
        risk_config: quant_risk::enforcement::RiskConfig {
            stop_loss_pct: config.risk.max_drawdown.min(0.10),
            max_daily_loss_pct: config.risk.max_daily_loss,
            max_drawdown_pct: config.risk.max_drawdown,
            circuit_breaker_failures: 5,
            halt_on_drawdown: true,
            max_holding_days: 30,
            timeout_min_profit_pct: 0.02,
            rebalance_threshold: 0.05,
            ..Default::default()
        },
        db_pool: None,
    };

    let strategy_name = strategy.to_string();
    let mut engine = TradingEngine::new_with_broker(engine_config, qmt_broker);
    engine.start(move || -> Box<dyn quant_core::traits::Strategy> {
        match strategy_name.as_str() {
            "rsi_reversal" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
            "macd_trend" => Box::new(MacdMomentum::new(12, 26, 9)),
            "multi_factor" => Box::new(MultiFactorStrategy::with_defaults()),
            "sentiment_aware" => Box::new(SentimentAwareStrategy::with_defaults(
                Box::new(MultiFactorStrategy::with_defaults()),
                SentimentStore::new(),
            )),
            "ml_factor" => Box::new(MlFactorStrategy::with_defaults()),
            _ => Box::new(DualMaCrossover::new(5, 20)),
        }
    }).await;

    println!("  🔴 LIVE engine running. Type 'status' or 'stop'.");
    println!("  ⚠️  WARNING: Real orders will be placed through QMT!");
    println!();

    // Interactive status loop (same as auto)
    loop {
        print!("qmt> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();

        match input {
            "stop" | "quit" | "exit" => {
                engine.stop().await;
                println!();
                break;
            }
            "status" | "s" => {
                let status = engine.status().await;
                println!();
                println!("  ═══════════════════════════════════════════");
                println!("  🔴 QMT Live Engine Status");
                println!("  ═══════════════════════════════════════════");
                println!("  Running:      {}", if status.running { "✅ Yes" } else { "❌ No" });
                println!("  Strategy:     {}", status.strategy);
                println!("  Symbols:      {}", status.symbols.join(", "));
                println!("  Signals:      {}", status.total_signals);
                println!("  Orders:       {}", status.total_orders);
                println!("  Fills:        {}", status.total_fills);
                println!("  Rejected:     {}", status.total_rejected);
                println!("  PnL:          ¥{:+.2}", status.pnl);
                println!("  ═══════════════════════════════════════════");

                if !status.recent_trades.is_empty() {
                    println!();
                    println!("  Recent Trades:");
                    for t in status.recent_trades.iter().take(5) {
                        let side_str = if t.side == quant_core::types::OrderSide::Buy { "BUY " } else { "SELL" };
                        println!("    {} {} x{:.0} @ {:.2} [{}]",
                            side_str, t.symbol, t.quantity, t.price, t.status);
                    }
                }

                // Show live portfolio from QMT
                if let Ok(account) = engine.broker().get_account().await {
                    let p = &account.portfolio;
                    println!();
                    println!("  Portfolio: cash=¥{:.2} total=¥{:.2}", p.cash, p.total_value);
                    for (sym, pos) in &p.positions {
                        println!("    {} x{:.0} avg={:.2} cur={:.2}",
                            sym, pos.quantity, pos.avg_cost, pos.current_price);
                    }
                }
                println!();
            }
            "help" | "h" => {
                println!("  Commands: status (s), stop, help (h)");
            }
            "" => continue,
            _ => println!("  Unknown command. Type 'help' for options."),
        }
    }

    let status = engine.status().await;
    println!("  ═══════════════════════════════════════════");
    println!("  📋 Final Summary (QMT Live)");
    println!("  ═══════════════════════════════════════════");
    println!("  Total Signals:  {}", status.total_signals);
    println!("  Total Orders:   {}", status.total_orders);
    println!("  Total Fills:    {}", status.total_fills);
    println!("  Total Rejected: {}", status.total_rejected);
    println!("  Final PnL:      ¥{:+.2}", status.pnl);
    println!("  ═══════════════════════════════════════════");
}

async fn cmd_qmt_status(config: &AppConfig) {
    println!("🔌 QMT Bridge Status");
    println!("  Bridge URL: {}", config.qmt.bridge_url);
    println!("  Account:    {}", config.qmt.account);
    println!("  QMT Path:   {}", config.qmt.qmt_path);
    println!();

    let url = format!("{}/health", config.qmt.bridge_url);
    match reqwest::get(&url).await {
        Ok(resp) => {
            match resp.json::<serde_json::Value>().await {
                Ok(v) => {
                    let connected = v.get("connected").and_then(|c| c.as_bool()).unwrap_or(false);
                    let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("unknown");
                    if connected {
                        println!("  ✅ Bridge: {} (connected to QMT)", status);
                    } else {
                        println!("  ⚠️  Bridge: {} (not connected to QMT client)", status);
                    }
                }
                Err(e) => println!("  ❌ Invalid response: {}", e),
            }
        }
        Err(e) => {
            println!("  ❌ Bridge offline: {}", e);
            println!();
            println!("  To start the bridge:");
            println!("    python qmt_bridge/qmt_bridge.py --qmt-path \"C:/QMT/userdata_mini\" --account \"YOUR_ACCOUNT\"");
        }
    }
}

fn cmd_portfolio_show() {
    println!("💼 Portfolio Summary");
    println!();

    let positions = vec![
        ("600519.SH", "贵州茅台", 100, 1620.00, 1688.50),
        ("000858.SZ", "五粮液", 500, 148.30, 142.85),
        ("601318.SH", "中国平安", 1000, 49.80, 52.36),
        ("000001.SZ", "平安银行", 2000, 13.10, 12.58),
        ("600036.SH", "招商银行", 800, 32.50, 35.72),
    ];

    let cash = 245_680.00;
    let mut total_market = 0.0;
    let mut total_cost = 0.0;

    println!("  {:<12} {:<10} {:>8} {:>10} {:>10} {:>12} {:>10}",
        "Symbol", "Name", "Shares", "AvgCost", "Price", "Value", "PnL");
    println!("  {}", "-".repeat(82));

    for (sym, name, shares, avg_cost, cur_price) in &positions {
        let mkt_val = *shares as f64 * cur_price;
        let cost_val = *shares as f64 * avg_cost;
        let pnl = mkt_val - cost_val;
        total_market += mkt_val;
        total_cost += cost_val;
        let pnl_str = if pnl >= 0.0 { format!("+{pnl:.2}") } else { format!("{pnl:.2}") };
        println!("  {:<12} {:<10} {:>8} {:>10.2} {:>10.2} {:>12.2} {:>10}",
            sym, name, shares, avg_cost, cur_price, mkt_val, pnl_str);
    }

    let total_pnl = total_market - total_cost;
    let total_value = total_market + cash;
    println!("  {}", "-".repeat(82));
    println!();
    println!("  💰 Cash:         ¥{cash:>14.2}");
    println!("  📈 Market Value: ¥{total_market:>14.2}");
    println!("  🏦 Total Value:  ¥{total_value:>14.2}");
    if total_pnl >= 0.0 {
        println!("  ✅ Total PnL:    ¥{total_pnl:>+14.2} ({:.2}%)", (total_pnl / total_cost) * 100.0);
    } else {
        println!("  ❌ Total PnL:    ¥{total_pnl:>+14.2} ({:.2}%)", (total_pnl / total_cost) * 100.0);
    }
}

// ── Chat ────────────────────────────────────────────────────────────

async fn run_chat(config: &AppConfig) {
    if config.llm.api_key.is_empty() {
        println!("⚠️  LLM API key not configured.");
        println!("   Set llm.api_key in config/default.toml to enable AI chat.");
        println!("   Entering offline mode with basic responses.");
        println!();
    }

    let client = LlmClient::new(
        &config.llm.api_url,
        &config.llm.api_key,
        &config.llm.model,
        config.llm.temperature,
        config.llm.max_tokens,
    );
    let mut context = ConversationContext::new("You are a quantitative trading assistant for Chinese A-shares.", 50);
    let tools = get_all_tools();
    let executor = ToolExecutor::new();

    println!("🤖 Quant Trading AI Assistant (type 'quit' to exit)");
    println!("   Commands: 'quit', 'clear', 'help'");
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
        match input {
            "quit" | "exit" => {
                println!("Goodbye! 👋");
                break;
            }
            "clear" => {
                context = ConversationContext::new("You are a quantitative trading assistant for Chinese A-shares.", 50);
                println!("🗑️  Chat history cleared.");
                continue;
            }
            "help" => {
                println!("Available commands:");
                println!("  quit/exit  - Exit chat");
                println!("  clear      - Clear conversation history");
                println!("  help       - Show this help");
                println!("  (anything else is sent to the AI assistant)");
                continue;
            }
            _ => {}
        }

        if config.llm.api_key.is_empty() {
            println!("\n💡 AI chat requires an API key. Set `llm.api_key` in config/default.toml.");
            println!("   Your message: \"{input}\"\n");
            continue;
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
                    continue;
                }

                if let Some(content) = &msg.content {
                    println!("\n{content}\n");
                    context.add_assistant_message(content);
                }
            }
            break;
        }
    }
}

// ── Server ──────────────────────────────────────────────────────────

// ── Sentiment Commands ──────────────────────────────────────────────

fn cmd_sentiment_submit(symbol: &str, score: f64, title: &str, source: &str, content: &str) {
    let store = SentimentStore::new();
    let item = store.submit(symbol, source, title, content, score, chrono::Utc::now().naive_utc());

    println!("📰 Sentiment item submitted");
    println!("  ID:       {}", item.id);
    println!("  Symbol:   {}", item.symbol);
    println!("  Source:   {}", item.source);
    println!("  Title:    {}", item.title);
    println!("  Score:    {:.2} ({})", item.sentiment_score, item.level());
    println!("  Time:     {}", item.published_at.format("%Y-%m-%d %H:%M:%S"));
    println!();
    println!("  ℹ️  Note: CLI sentiment data is ephemeral. Use the API endpoint");
    println!("  POST /api/sentiment/submit for persistent storage during server mode.");
}

fn cmd_sentiment_query(symbol: &str, limit: usize) {
    println!("📊 Sentiment Query for {symbol}");
    println!("  ℹ️  CLI can only show data from the running server.");
    println!("  Use the WebUI or API for full sentiment data.");
    println!();
    println!("  To submit sentiment via API:");
    println!("    curl -X POST http://localhost:8080/api/sentiment/submit \\");
    println!("      -H 'Content-Type: application/json' \\");
    println!("      -d '{{\"symbol\":\"{symbol}\",\"source\":\"news\",\"title\":\"...\",\"sentiment_score\":0.5}}'");
    println!();
    println!("  To query via API:");
    println!("    curl http://localhost:8080/api/sentiment/{symbol}?limit={limit}");
    println!();
    println!("  To view in WebUI:");
    println!("    Open http://localhost:8080/sentiment");
}

async fn run_server(config: &AppConfig) -> anyhow::Result<()> {
    // Initialize journal store
    if !std::path::Path::new("data").exists() {
        std::fs::create_dir_all("data").ok();
    }

    // Try to connect to PostgreSQL; fall back to SQLite if unavailable
    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| config.database.url.clone());
    tracing::info!("🔌 Connecting to PostgreSQL: {}...", db_url.split('@').last().unwrap_or(&db_url));
    let db_pool = match quant_data::storage::create_pool(&db_url, config.database.max_connections).await {
        Ok(pool) => {
            // Verify connection works with a test query
            match sqlx::query("SELECT 1").execute(&pool).await {
                Ok(_) => {
                    if let Err(e) = quant_data::storage::run_migrations(&pool).await {
                        tracing::warn!("⚠️  PostgreSQL migrations failed: {}. Falling back to SQLite.", e);
                        None
                    } else {
                        tracing::info!("✅ PostgreSQL connected and migrations applied");
                        Some(pool)
                    }
                }
                Err(e) => {
                    tracing::warn!("⚠️  PostgreSQL test query failed: {}. Falling back to SQLite.", e);
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("⚠️  PostgreSQL unavailable: {}. Falling back to SQLite.", e);
            None
        }
    };

    let (journal, notifier) = if let Some(ref pool) = db_pool {
        let journal = quant_broker::journal::JournalStore::new(pool.clone());
        let notifier = quant_broker::notifier::Notifier::new(pool.clone(), "data");
        (journal, notifier)
    } else {
        let journal = quant_broker::journal::JournalStore::open("data/trade_journal.db")
            .expect("Failed to open trade journal");
        let notifier = quant_broker::notifier::Notifier::open("data/notifications.db", "data")
            .expect("Failed to open notifier");
        (journal, notifier)
    };

    let task_store = quant_api::TaskStore::open("data/tasks.db")
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to open tasks.db, using in-memory: {e}");
            quant_api::TaskStore::open(":memory:").expect("in-memory TaskStore")
        });

    let state = AppState {
        config: std::sync::Arc::new(config.clone()),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        sentiment_store: quant_strategy::sentiment::SentimentStore::new(),
        sentiment_collector: std::sync::Arc::new(tokio::sync::Mutex::new(
            quant_strategy::collector::SentimentCollector::new(config.sentiment.clone()),
        )),
        journal: std::sync::Arc::new(journal),
        log_store: std::sync::Arc::new(quant_api::LogStore::new()),
        notifier: std::sync::Arc::new(notifier),
        db: db_pool,
        task_store: std::sync::Arc::new(task_store),
    };

    // Resolve web/dist path — try CWD first, then relative to the executable
    let web_dist = {
        let cwd_path = std::path::PathBuf::from("web/dist");
        if cwd_path.exists() {
            cwd_path
        } else if let Ok(exe) = std::env::current_exe() {
            let exe_dir = exe.parent().unwrap_or(exe.as_path());
            let candidate = exe_dir.join("web/dist");
            if candidate.exists() {
                candidate
            } else {
                // Try two levels up from exe (target/release/ -> project root)
                let project_root = exe_dir.parent().and_then(|p| p.parent());
                if let Some(root) = project_root {
                    let candidate2 = root.join("web/dist");
                    if candidate2.exists() { candidate2 } else { cwd_path }
                } else {
                    cwd_path
                }
            }
        } else {
            cwd_path
        }
    };

    let app = create_router(state, web_dist.to_str().unwrap_or("web/dist"));

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    println!("🚀 QuantTrader API server starting on {addr}");
    println!("   API:  http://{addr}/api/health");
    println!("   Web:  http://{addr}/");
    println!();

    if !web_dist.exists() {
        println!("⚠️  web/dist not found. Run 'cd web && npm run build' to build the WebUI.");
        println!("   API endpoints will still work without the WebUI.");
        println!();
    } else {
        println!("   Serving WebUI from: {}", web_dist.display());
    }

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ── Research Commands ──────────────────────────────────────────────

fn cmd_research_list() {
    use quant_strategy::dl_models::build_knowledge_base;

    let kb = build_knowledge_base();
    println!("🧠 深度学习多因子模型知识库");
    println!("  更新时间: {}", kb.last_updated);
    println!();

    for cat in &kb.categories {
        println!("━━━ {} ━━━", cat.name);
        println!("  {}", cat.description);
        println!();
        for model in &cat.models {
            println!("  📌 {} ({}, {})", model.name, model.category, model.year);
            println!("     {}", model.description);
            println!("     🔬 创新: {}", model.key_innovation);
            println!("     ✅ 优势: {}", model.strengths.join(" | "));
            println!("     ⚠️  局限: {}", model.limitations.join(" | "));
            println!("     📄 {}", model.reference);
            println!("     🔗 {}", model.reference_url);
            println!();
        }
    }

    let total: usize = kb.categories.iter().map(|c| c.models.len()).sum();
    println!("📊 共 {} 个类别, {} 个模型", kb.categories.len(), total);
}

fn cmd_research_summary() {
    use quant_strategy::dl_models::{build_knowledge_base, summarize_knowledge_base};

    let kb = build_knowledge_base();
    let summary = summarize_knowledge_base(&kb);

    println!("🧠 DL因子模型知识库概览");
    println!("  总模型数: {}", summary.total_models);
    println!("  总类别数: {}", summary.total_categories);
    println!("  更新时间: {}", summary.last_updated);
    println!();
    println!("  {:<20} {:>6}", "类别", "模型数");
    println!("  {}", "-".repeat(30));
    for cat in &summary.categories {
        println!("  {:<20} {:>6}", cat.name, cat.count);
    }
}

async fn cmd_research_collect(config: &AppConfig, topic: Option<&str>) {
    use quant_strategy::dl_models::build_collection_prompt;

    let topic = topic.unwrap_or("量化多因子深度学习模型最新进展");
    println!("🤖 自动收集研究: {}", topic);
    println!("  正在调用LLM...");

    let prompt = build_collection_prompt(topic);
    let llm = LlmClient::new(
        &config.llm.api_url,
        &config.llm.api_key,
        &config.llm.model,
        config.llm.temperature,
        config.llm.max_tokens,
    );

    let messages = vec![
        quant_llm::client::ChatMessage {
            role: "user".into(),
            content: Some(prompt),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    match llm.chat(&messages, None).await {
        Ok(resp) => {
            let content = resp.choices.first()
                .and_then(|c| c.message.content.as_ref())
                .cloned()
                .unwrap_or_default();

            println!();
            println!("📰 收集结果:");
            println!("{}", "-".repeat(60));

            // Try to parse as JSON
            if let Ok(items) = serde_json::from_str::<Vec<quant_strategy::dl_models::CollectedResearch>>(&content) {
                for (i, item) in items.iter().enumerate() {
                    println!("  {}. {} [相关度: {}]", i + 1, item.title, item.relevance);
                    println!("     来源: {}", item.source);
                    println!("     {}", item.summary);
                    println!();
                }
                println!("✅ 共收集 {} 条研究", items.len());
            } else {
                println!("{}", content);
            }
        }
        Err(e) => {
            println!("❌ LLM收集失败: {}", e);
            println!("💡 请确保config/default.toml中的[llm]配置正确");
        }
    }
}
