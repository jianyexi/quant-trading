use std::collections::HashMap;

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use quant_core::models::{Kline, Order, Portfolio, Position, Trade};
use quant_core::traits::Strategy;
use quant_core::types::{OrderSide, OrderStatus, OrderType, Signal, SignalAction};

use crate::matching::MatchingEngine;
use crate::metrics::PerformanceMetrics;

// ── Backtest Events ─────────────────────────────────────────────────

/// Comprehensive event log for post-hoc analysis of backtest runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BacktestEvent {
    /// Strategy generated a signal (Buy/Sell/Hold)
    Signal {
        seq: u32,
        timestamp: NaiveDateTime,
        symbol: String,
        action: String,
        confidence: f64,
        price: f64,
    },
    /// An order was created from a signal
    OrderCreated {
        seq: u32,
        timestamp: NaiveDateTime,
        symbol: String,
        side: String,
        price: f64,
        quantity: f64,
    },
    /// Order was filled by the matching engine
    OrderFilled {
        seq: u32,
        timestamp: NaiveDateTime,
        symbol: String,
        side: String,
        price: f64,
        quantity: f64,
        commission: f64,
    },
    /// Order was rejected (insufficient cash, zero quantity, etc.)
    OrderRejected {
        seq: u32,
        timestamp: NaiveDateTime,
        symbol: String,
        side: String,
        reason: String,
    },
    /// Risk module triggered a forced action
    RiskTriggered {
        seq: u32,
        timestamp: NaiveDateTime,
        symbol: String,
        trigger: String,       // "stop_loss", "holding_timeout", "concentration_limit"
        detail: String,
    },
    /// A new position was opened
    PositionOpened {
        seq: u32,
        timestamp: NaiveDateTime,
        symbol: String,
        side: String,
        entry_price: f64,
        quantity: f64,
    },
    /// A position was fully closed
    PositionClosed {
        seq: u32,
        timestamp: NaiveDateTime,
        symbol: String,
        entry_price: f64,
        exit_price: f64,
        quantity: f64,
        realized_pnl: f64,
        holding_days: i64,
    },
    /// End-of-bar portfolio snapshot
    PortfolioSnapshot {
        seq: u32,
        timestamp: NaiveDateTime,
        cash: f64,
        total_value: f64,
        n_positions: usize,
        unrealized_pnl: f64,
        drawdown_pct: f64,
    },
}

/// Per-symbol performance breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMetrics {
    pub symbol: String,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_pnl: f64,
    pub win_rate: f64,
    pub avg_holding_days: f64,
    pub max_win: f64,
    pub max_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub stamp_tax_rate: f64,
    pub slippage_ticks: u32,
    /// Fraction of portfolio value per buy order (e.g. 0.10 = 10%). Default 0.10 = 10%.
    #[serde(default = "default_position_size_pct")]
    pub position_size_pct: f64,
    /// Max fraction of portfolio in a single stock (e.g. 0.25 = 25%). Default 0.30.
    #[serde(default = "default_max_concentration")]
    pub max_concentration: f64,
    /// Stop-loss threshold per position (e.g. 0.05 = 5%). Default 0.08 = 8%.
    #[serde(default = "default_stop_loss_pct")]
    pub stop_loss_pct: f64,
    /// Max holding days before forced exit (0 = disabled). Default 30.
    #[serde(default = "default_max_holding_days")]
    pub max_holding_days: u32,
    /// Daily loss limit as fraction of portfolio (e.g. 0.03 = 3%). 0 = disabled.
    #[serde(default)]
    pub daily_loss_limit: f64,
    /// Max portfolio drawdown from peak (e.g. 0.15 = 15%). 0 = disabled.
    #[serde(default)]
    pub max_drawdown_limit: f64,
    /// Enable ATR-based position sizing (inverse volatility). Default true.
    #[serde(default = "default_true")]
    pub use_atr_sizing: bool,
    /// ATR lookback period for volatility sizing. Default 14.
    #[serde(default = "default_atr_period")]
    pub atr_period: usize,
    /// Target risk per trade as fraction of portfolio (e.g. 0.02 = 2%). Default 0.02.
    #[serde(default = "default_risk_per_trade")]
    pub risk_per_trade: f64,
}

fn default_position_size_pct() -> f64 { 0.10 }
fn default_max_concentration() -> f64 { 0.30 }
fn default_stop_loss_pct() -> f64 { 0.08 }
fn default_max_holding_days() -> u32 { 30 }
fn default_true() -> bool { true }
fn default_atr_period() -> usize { 14 }
fn default_risk_per_trade() -> f64 { 0.02 }

pub struct BacktestResult {
    pub config: BacktestConfig,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<(NaiveDateTime, f64)>,
    pub metrics: PerformanceMetrics,
    pub final_portfolio: Portfolio,
    pub events: Vec<BacktestEvent>,
    pub symbol_metrics: Vec<SymbolMetrics>,
}

pub struct BacktestEngine {
    config: BacktestConfig,
    matching: MatchingEngine,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        let matching = MatchingEngine::new(config.slippage_ticks);
        Self { config, matching }
    }

    /// Execute a sell order and update portfolio. Returns the trade if filled.
    fn execute_sell(
        &self,
        symbol: &str,
        quantity: f64,
        kline: &Kline,
        portfolio: &mut Portfolio,
    ) -> Option<Trade> {
        if quantity <= 0.0 {
            return None;
        }
        let order = Order {
            id: Uuid::new_v4(),
            symbol: symbol.to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Market,
            price: kline.close,
            quantity,
            filled_qty: 0.0,
            status: OrderStatus::Pending,
            created_at: kline.datetime,
            updated_at: kline.datetime,
        };

        let mut trade = self.matching.try_match(&order, kline)?;
        let turnover = trade.price * trade.quantity;
        // Stamp tax (印花税) only applies to SELL side in China A-shares
        trade.commission =
            turnover * self.config.commission_rate + turnover * self.config.stamp_tax_rate;

        let proceeds = turnover - trade.commission;
        portfolio.cash += proceeds;

        if let Some(pos) = portfolio.positions.get_mut(symbol) {
            pos.realized_pnl += (trade.price - pos.avg_cost) * trade.quantity;
            pos.quantity -= trade.quantity;
            if pos.quantity <= 1e-10 {
                portfolio.positions.remove(symbol);
            }
        }
        Some(trade)
    }

    /// Run backtest with a strategy on kline data.
    pub fn run(&self, strategy: &mut dyn Strategy, data: &[Kline]) -> BacktestResult {
        let mut portfolio = Portfolio {
            positions: HashMap::new(),
            cash: self.config.initial_capital,
            total_value: self.config.initial_capital,
        };

        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<(NaiveDateTime, f64)> = Vec::new();
        let mut events: Vec<BacktestEvent> = Vec::new();
        let mut seq: u32 = 0;
        let mut peak_value: f64 = self.config.initial_capital;

        // Track position entry info for PositionClosed events
        let mut entry_info: HashMap<String, (NaiveDateTime, f64)> = HashMap::new();

        strategy.on_init();

        // Pending signal from previous bar (signal on bar N → fill at bar N+1 open)
        let mut pending_signal: Option<Signal> = None;
        // ATR tracking: rolling true range for volatility-based position sizing
        let mut true_ranges: Vec<f64> = Vec::new();
        let mut prev_close: f64 = 0.0;
        // Daily PnL tracking
        let mut day_start_value = self.config.initial_capital;
        let mut current_day: Option<chrono::NaiveDate> = None;
        let mut daily_paused = false;

        for kline in data {
            // Day boundary: reset daily PnL tracking
            let bar_date = kline.datetime.date();
            if current_day != Some(bar_date) {
                day_start_value = portfolio.total_value;
                current_day = Some(bar_date);
                daily_paused = false;
            }

            // Daily loss limit check
            if self.config.daily_loss_limit > 0.0 {
                let daily_pnl_pct = (portfolio.total_value - day_start_value) / day_start_value;
                if daily_pnl_pct < -self.config.daily_loss_limit {
                    if !daily_paused {
                        seq += 1;
                        events.push(BacktestEvent::RiskTriggered {
                            seq, timestamp: kline.datetime, symbol: kline.symbol.clone(),
                            trigger: "daily_loss_limit".into(),
                            detail: format!("daily loss {:.2}% > limit {:.2}%", daily_pnl_pct * 100.0, self.config.daily_loss_limit * 100.0),
                        });
                        daily_paused = true;
                    }
                    pending_signal = None; // Cancel any pending buy
                }
            }

            // Max drawdown limit check
            if self.config.max_drawdown_limit > 0.0 && peak_value > 0.0 {
                let drawdown = (peak_value - portfolio.total_value) / peak_value;
                if drawdown > self.config.max_drawdown_limit {
                    if !daily_paused {
                        seq += 1;
                        events.push(BacktestEvent::RiskTriggered {
                            seq, timestamp: kline.datetime, symbol: kline.symbol.clone(),
                            trigger: "max_drawdown".into(),
                            detail: format!("drawdown {:.2}% > limit {:.2}%", drawdown * 100.0, self.config.max_drawdown_limit * 100.0),
                        });
                        daily_paused = true;
                    }
                    pending_signal = None;
                }
            }
            // 0. Risk checks: stop-loss & holding timeout on existing positions
            let mut forced_sells: Vec<(String, f64)> = Vec::new();
            for pos in portfolio.positions.values() {
                if pos.symbol != kline.symbol {
                    continue;
                }
                // Stop-loss check: use kline.low as worst-case price during bar
                if self.config.stop_loss_pct > 0.0 && pos.avg_cost > 0.0 {
                    let loss_pct = (kline.low - pos.avg_cost) / pos.avg_cost;
                    if loss_pct < -self.config.stop_loss_pct {
                        seq += 1;
                        events.push(BacktestEvent::RiskTriggered {
                            seq, timestamp: kline.datetime, symbol: pos.symbol.clone(),
                            trigger: "stop_loss".into(),
                            detail: format!("loss {:.2}% > threshold {:.2}%", loss_pct * 100.0, self.config.stop_loss_pct * 100.0),
                        });
                        forced_sells.push((pos.symbol.clone(), pos.quantity));
                        continue;
                    }
                }
                // Holding timeout check
                if self.config.max_holding_days > 0 {
                    let days = (kline.datetime - pos.entry_time).num_days();
                    if days > self.config.max_holding_days as i64 {
                        seq += 1;
                        events.push(BacktestEvent::RiskTriggered {
                            seq, timestamp: kline.datetime, symbol: pos.symbol.clone(),
                            trigger: "holding_timeout".into(),
                            detail: format!("held {} days > max {}", days, self.config.max_holding_days),
                        });
                        forced_sells.push((pos.symbol.clone(), pos.quantity));
                    }
                }
            }
            for (sym, qty) in &forced_sells {
                let entry = entry_info.get(sym).copied();
                if let Some(trade) = self.execute_sell(sym, *qty, kline, &mut portfolio) {
                    seq += 1;
                    events.push(BacktestEvent::OrderFilled {
                        seq, timestamp: kline.datetime, symbol: sym.clone(),
                        side: "SELL".into(), price: trade.price, quantity: trade.quantity,
                        commission: trade.commission,
                    });
                    // PositionClosed event
                    if let Some((entry_time, entry_price)) = entry {
                        let holding_days = (kline.datetime - entry_time).num_days();
                        let pnl = (trade.price - entry_price) * trade.quantity - trade.commission;
                        seq += 1;
                        events.push(BacktestEvent::PositionClosed {
                            seq, timestamp: kline.datetime, symbol: sym.clone(),
                            entry_price, exit_price: trade.price, quantity: trade.quantity,
                            realized_pnl: pnl, holding_days,
                        });
                        entry_info.remove(sym);
                    }
                    trades.push(trade);
                }
            }

            // 1. Execute PENDING signal from previous bar (fill at current bar's open)
            //    This eliminates look-ahead bias: signal on bar N → fill at bar N+1 open
            if let Some(sig) = pending_signal.take() {
                if !daily_paused && sig.symbol == kline.symbol {
                    let (side, should_trade) = match sig.action {
                        SignalAction::Buy => (OrderSide::Buy, true),
                        SignalAction::Sell => (OrderSide::Sell, true),
                        SignalAction::Hold => (OrderSide::Buy, false),
                    };

                    if should_trade {
                        let quantity = match side {
                            OrderSide::Buy => {
                                // ATR-based position sizing: risk_per_trade / ATR determines shares
                                // Fallback to fixed % if ATR not available or disabled
                                let atr_pct = if self.config.use_atr_sizing && true_ranges.len() >= self.config.atr_period {
                                    let atr: f64 = true_ranges[true_ranges.len() - self.config.atr_period..]
                                        .iter().sum::<f64>() / self.config.atr_period as f64;
                                    let atr_ratio = atr / kline.open;
                                    // Scale: higher vol → smaller position
                                    // position_size = risk_per_trade / atr_ratio, capped by position_size_pct
                                    let vol_scaled = (self.config.risk_per_trade / atr_ratio.max(0.001))
                                        .min(self.config.position_size_pct);
                                    vol_scaled
                                } else {
                                    self.config.position_size_pct
                                };

                                let allocation = portfolio.total_value * atr_pct;
                                let budget = allocation.min(portfolio.cash);
                                let existing_value = portfolio
                                    .positions
                                    .get(&sig.symbol)
                                    .map(|p| p.current_price * p.quantity)
                                    .unwrap_or(0.0);
                                let max_allowed =
                                    portfolio.total_value * self.config.max_concentration - existing_value;
                                let capped_budget = budget.min(max_allowed.max(0.0));

                                if capped_budget < budget {
                                    seq += 1;
                                    events.push(BacktestEvent::RiskTriggered {
                                        seq, timestamp: kline.datetime, symbol: sig.symbol.clone(),
                                        trigger: "concentration_limit".into(),
                                        detail: format!("budget capped from {:.0} to {:.0}", budget, capped_budget),
                                    });
                                }

                                // Use kline.open for position sizing (what's actually available)
                                let price_est = kline.open;
                                let affordable =
                                    (capped_budget / (price_est * (1.0 + self.config.commission_rate)))
                                        .floor();
                                (affordable / 100.0).floor() * 100.0
                            }
                            OrderSide::Sell => {
                                portfolio
                                    .positions
                                    .get(&sig.symbol)
                                    .map(|p| p.quantity)
                                    .unwrap_or(0.0)
                            }
                        };

                        if quantity > 0.0 {
                            let side_str = if side == OrderSide::Sell { "SELL" } else { "BUY" };
                            seq += 1;
                            events.push(BacktestEvent::OrderCreated {
                                seq, timestamp: kline.datetime, symbol: sig.symbol.clone(),
                                side: side_str.into(), price: kline.open, quantity,
                            });

                            if side == OrderSide::Sell {
                                let entry = entry_info.get(&sig.symbol).copied();
                                if let Some(trade) =
                                    self.execute_sell(&sig.symbol, quantity, kline, &mut portfolio)
                                {
                                    seq += 1;
                                    events.push(BacktestEvent::OrderFilled {
                                        seq, timestamp: kline.datetime, symbol: sig.symbol.clone(),
                                        side: "SELL".into(), price: trade.price, quantity: trade.quantity,
                                        commission: trade.commission,
                                    });
                                    if let Some((entry_time, entry_price)) = entry {
                                        let holding_days = (kline.datetime - entry_time).num_days();
                                        let pnl = (trade.price - entry_price) * trade.quantity - trade.commission;
                                        seq += 1;
                                        events.push(BacktestEvent::PositionClosed {
                                            seq, timestamp: kline.datetime, symbol: sig.symbol.clone(),
                                            entry_price, exit_price: trade.price, quantity: trade.quantity,
                                            realized_pnl: pnl, holding_days,
                                        });
                                        entry_info.remove(&sig.symbol);
                                    }
                                    trades.push(trade);
                                }
                            } else {
                                let is_new_position = !portfolio.positions.contains_key(&sig.symbol);
                                let order = Order {
                                    id: Uuid::new_v4(),
                                    symbol: sig.symbol.clone(),
                                    side,
                                    order_type: OrderType::Market,
                                    price: kline.open,
                                    quantity,
                                    filled_qty: 0.0,
                                    status: OrderStatus::Pending,
                                    created_at: kline.datetime,
                                    updated_at: kline.datetime,
                                };

                                if let Some(mut trade) = self.matching.try_match(&order, kline) {
                                    let turnover = trade.price * trade.quantity;
                                    // Buy side: commission only, no stamp tax
                                    trade.commission = turnover * self.config.commission_rate;

                                    let cost = turnover + trade.commission;
                                    portfolio.cash -= cost;

                                    let pos = portfolio
                                        .positions
                                        .entry(trade.symbol.clone())
                                        .or_insert(Position {
                                            symbol: trade.symbol.clone(),
                                            quantity: 0.0,
                                            avg_cost: 0.0,
                                            current_price: trade.price,
                                            unrealized_pnl: 0.0,
                                            realized_pnl: 0.0,
                                            entry_time: kline.datetime,
                                            scale_level: 1,
                                            target_weight: 0.0,
                                        });
                                    let total_cost =
                                        pos.avg_cost * pos.quantity + trade.price * trade.quantity;
                                    pos.quantity += trade.quantity;
                                    pos.avg_cost = if pos.quantity > 0.0 {
                                        total_cost / pos.quantity
                                    } else {
                                        0.0
                                    };

                                    seq += 1;
                                    events.push(BacktestEvent::OrderFilled {
                                        seq, timestamp: kline.datetime, symbol: sig.symbol.clone(),
                                        side: "BUY".into(), price: trade.price, quantity: trade.quantity,
                                        commission: trade.commission,
                                    });

                                    if is_new_position {
                                        seq += 1;
                                        events.push(BacktestEvent::PositionOpened {
                                            seq, timestamp: kline.datetime, symbol: sig.symbol.clone(),
                                            side: "LONG".into(), entry_price: trade.price, quantity: trade.quantity,
                                        });
                                        entry_info.insert(sig.symbol.clone(), (kline.datetime, trade.price));
                                    }

                                    trades.push(trade);
                                }
                            }
                        } else {
                            let reason = if side == OrderSide::Buy {
                                "insufficient cash or lot size rounding"
                            } else {
                                "no position to sell"
                            };
                            seq += 1;
                            events.push(BacktestEvent::OrderRejected {
                                seq, timestamp: kline.datetime, symbol: sig.symbol.clone(),
                                side: if side == OrderSide::Sell { "SELL" } else { "BUY" }.into(),
                                reason: reason.into(),
                            });
                        }
                    }
                }
            }

            // 2. Generate signal for NEXT bar execution
            //    Strategy sees current bar (including close), but order fills at NEXT bar open
            let signal = strategy.on_bar(kline);

            if let Some(ref sig) = signal {
                seq += 1;
                events.push(BacktestEvent::Signal {
                    seq, timestamp: kline.datetime, symbol: sig.symbol.clone(),
                    action: format!("{:?}", sig.action), confidence: sig.confidence,
                    price: kline.close,
                });
            }

            // Store signal for execution at next bar
            pending_signal = signal;

            // 3. Update current prices & portfolio value
            let mut unrealized_pnl = 0.0;
            for pos in portfolio.positions.values_mut() {
                if pos.symbol == kline.symbol {
                    pos.current_price = kline.close;
                    pos.unrealized_pnl = (kline.close - pos.avg_cost) * pos.quantity;
                }
                unrealized_pnl += pos.unrealized_pnl;
            }

            let positions_value: f64 = portfolio
                .positions
                .values()
                .map(|p| p.current_price * p.quantity)
                .sum();
            portfolio.total_value = portfolio.cash + positions_value;

            // Update ATR (true range) ring buffer
            let tr = if prev_close > 0.0 {
                (kline.high - kline.low)
                    .max((kline.high - prev_close).abs())
                    .max((kline.low - prev_close).abs())
            } else {
                kline.high - kline.low
            };
            true_ranges.push(tr);
            if true_ranges.len() > self.config.atr_period + 10 {
                true_ranges.remove(0);
            }
            prev_close = kline.close;

            // Track drawdown
            if portfolio.total_value > peak_value {
                peak_value = portfolio.total_value;
            }
            let drawdown_pct = if peak_value > 0.0 {
                (peak_value - portfolio.total_value) / peak_value * 100.0
            } else {
                0.0
            };

            // 4. Record equity curve
            equity_curve.push((kline.datetime, portfolio.total_value));

            // 5. Portfolio snapshot event (every bar)
            seq += 1;
            events.push(BacktestEvent::PortfolioSnapshot {
                seq, timestamp: kline.datetime,
                cash: portfolio.cash,
                total_value: portfolio.total_value,
                n_positions: portfolio.positions.len(),
                unrealized_pnl,
                drawdown_pct,
            });
        }

        strategy.on_stop();

        let metrics =
            PerformanceMetrics::calculate(&equity_curve, &trades, self.config.initial_capital);
        let symbol_metrics = compute_symbol_metrics(&trades, &events);

        BacktestResult {
            config: self.config.clone(),
            trades,
            equity_curve,
            metrics,
            final_portfolio: portfolio,
            events,
            symbol_metrics,
        }
    }
}

/// Compute per-symbol performance breakdown from trades and events.
fn compute_symbol_metrics(trades: &[Trade], events: &[BacktestEvent]) -> Vec<SymbolMetrics> {
    let mut by_symbol: HashMap<String, Vec<&Trade>> = HashMap::new();
    for t in trades {
        by_symbol.entry(t.symbol.clone()).or_default().push(t);
    }

    // Collect holding days from PositionClosed events
    let mut holding_days_map: HashMap<String, Vec<i64>> = HashMap::new();
    let mut pnl_map: HashMap<String, Vec<f64>> = HashMap::new();
    for ev in events {
        if let BacktestEvent::PositionClosed { symbol, realized_pnl, holding_days, .. } = ev {
            holding_days_map.entry(symbol.clone()).or_default().push(*holding_days);
            pnl_map.entry(symbol.clone()).or_default().push(*realized_pnl);
        }
    }

    let mut result: Vec<SymbolMetrics> = Vec::new();
    for (symbol, sym_trades) in &by_symbol {
        let pnls = pnl_map.get(symbol).cloned().unwrap_or_default();
        let hold_days = holding_days_map.get(symbol).cloned().unwrap_or_default();

        let winning = pnls.iter().filter(|p| **p > 0.0).count();
        let losing = pnls.iter().filter(|p| **p < 0.0).count();
        let total_pnl: f64 = pnls.iter().sum();
        let n_round_trips = pnls.len();
        let win_rate = if n_round_trips > 0 { winning as f64 / n_round_trips as f64 } else { 0.0 };
        let avg_hold = if !hold_days.is_empty() {
            hold_days.iter().sum::<i64>() as f64 / hold_days.len() as f64
        } else {
            0.0
        };
        let max_win = pnls.iter().cloned().fold(0.0_f64, f64::max);
        let max_loss = pnls.iter().cloned().fold(0.0_f64, f64::min);

        result.push(SymbolMetrics {
            symbol: symbol.clone(),
            total_trades: sym_trades.len(),
            winning_trades: winning,
            losing_trades: losing,
            total_pnl,
            win_rate,
            avg_holding_days: avg_hold,
            max_win,
            max_loss,
        });
    }

    result.sort_by(|a, b| b.total_pnl.partial_cmp(&a.total_pnl).unwrap_or(std::cmp::Ordering::Equal));
    result
}
