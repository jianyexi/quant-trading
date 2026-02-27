-- Backtest event log for comprehensive post-hoc analysis
CREATE TABLE IF NOT EXISTS backtest_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    event_type VARCHAR(30) NOT NULL,  -- Signal, OrderCreated, OrderFilled, OrderRejected, RiskTriggered, PositionOpened, PositionClosed, PortfolioSnapshot
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10),
    data JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_backtest_events_run ON backtest_events(run_id);
CREATE INDEX idx_backtest_events_type ON backtest_events(run_id, event_type);

-- Per-symbol backtest metrics
CREATE TABLE IF NOT EXISTS backtest_symbol_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    total_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    win_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_holding_days DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_win DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_loss DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE INDEX idx_backtest_symbol_metrics_run ON backtest_symbol_metrics(run_id);

-- ML model training history
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    algorithm VARCHAR(50) NOT NULL,
    data_source VARCHAR(30) NOT NULL DEFAULT 'unknown',
    n_samples INTEGER NOT NULL DEFAULT 0,
    n_features INTEGER NOT NULL DEFAULT 0,
    auc DOUBLE PRECISION,
    accuracy DOUBLE PRECISION,
    cv_avg_auc DOUBLE PRECISION,
    cv_std_auc DOUBLE PRECISION,
    n_cv_folds INTEGER,
    model_path TEXT,
    feature_importance JSONB,
    cv_results JSONB,
    label_distribution JSONB,
    config JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'completed'
);

CREATE INDEX idx_training_runs_ts ON training_runs(timestamp DESC);

-- Factor mining history
CREATE TABLE IF NOT EXISTS factor_mining_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    method VARCHAR(30) NOT NULL,  -- 'parametric' or 'gp'
    n_candidates INTEGER NOT NULL DEFAULT 0,
    n_selected INTEGER NOT NULL DEFAULT 0,
    factors JSONB,           -- array of {name, ic_mean, ir, turnover, ...}
    config JSONB,            -- mining configuration used
    status VARCHAR(20) NOT NULL DEFAULT 'completed'
);

CREATE INDEX idx_factor_mining_runs_ts ON factor_mining_runs(timestamp DESC);
