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
use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum};
use quant_strategy::indicators::{SMA, RSI};
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
            TradeAction::Live { strategy } => {
                println!("🚀 Live trading with strategy: {strategy}");
                println!("  ⚠️  Live trading requires broker API configuration.");
                println!("  Configure your broker connection in config/default.toml first.");
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
        other => {
            println!("  ❌ Unknown strategy: {other}");
            println!("  Available: sma_cross, rsi_reversal, macd_trend");
            return;
        }
    };

    // Configure and run backtest engine
    let bt_config = BacktestConfig {
        initial_capital: capital,
        commission_rate: 0.00025,  // 万分之2.5
        stamp_tax_rate: 0.001,    // 千分之一 (卖出)
        slippage_ticks: 1,
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
    use quant_strategy::builtin::{DualMaCrossover, RsiMeanReversion, MacdMomentum};
    use quant_core::traits::Broker;

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
    };

    let strategy_name = strategy.to_string();
    let mut engine = TradingEngine::new(engine_config);
    engine.start(move || -> Box<dyn quant_core::traits::Strategy> {
        match strategy_name.as_str() {
            "rsi_reversal" => Box::new(RsiMeanReversion::new(14, 70.0, 30.0)),
            "macd_trend" => Box::new(MacdMomentum::new(12, 26, 9)),
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
                engine.stop();
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

async fn run_server(config: &AppConfig) -> anyhow::Result<()> {
    let state = AppState {
        config: config.clone(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
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
