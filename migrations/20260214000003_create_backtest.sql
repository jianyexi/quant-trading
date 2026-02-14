CREATE TABLE IF NOT EXISTS backtest_runs (
    id UUID PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbols TEXT NOT NULL, -- comma-separated
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DOUBLE PRECISION NOT NULL,
    final_capital DOUBLE PRECISION,
    total_return DOUBLE PRECISION,
    annual_return DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    total_trades INTEGER,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(id),
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION NOT NULL DEFAULT 0,
    executed_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES backtest_runs(id),
    datetime TIMESTAMP NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    drawdown DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE INDEX idx_backtest_trades_run ON backtest_trades(run_id);
CREATE INDEX idx_equity_curve_run ON equity_curve(run_id);
