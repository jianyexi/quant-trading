-- Journal entries (trade audit trail)
CREATE TABLE IF NOT EXISTS journal_entries (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    entry_type VARCHAR(30) NOT NULL,
    symbol VARCHAR(10) NOT NULL DEFAULT '',
    side VARCHAR(4),
    quantity DOUBLE PRECISION,
    price DOUBLE PRECISION,
    order_id TEXT,
    status VARCHAR(20),
    reason TEXT,
    pnl DOUBLE PRECISION,
    portfolio_value DOUBLE PRECISION,
    cash DOUBLE PRECISION,
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_journal_symbol ON journal_entries(symbol);
CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entries(entry_type);

-- Daily portfolio snapshots
CREATE TABLE IF NOT EXISTS daily_snapshots (
    date DATE PRIMARY KEY,
    portfolio_value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    positions_count INTEGER NOT NULL DEFAULT 0,
    daily_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    cumulative_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_trades INTEGER NOT NULL DEFAULT 0
);

-- Notifications
CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    symbol VARCHAR(10),
    side VARCHAR(4),
    quantity DOUBLE PRECISION,
    price DOUBLE PRECISION,
    read BOOLEAN NOT NULL DEFAULT FALSE,
    delivery VARCHAR(20) NOT NULL DEFAULT 'delivered',
    channels VARCHAR(50) NOT NULL DEFAULT 'in_app'
);

CREATE INDEX IF NOT EXISTS idx_notif_ts ON notifications(timestamp);
CREATE INDEX IF NOT EXISTS idx_notif_read ON notifications(read);

-- Market tick recording
CREATE TABLE IF NOT EXISTS market_ticks (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    datetime TIMESTAMP NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ticks_symbol_dt ON market_ticks(symbol, datetime);
CREATE INDEX IF NOT EXISTS idx_ticks_symbol ON market_ticks(symbol);
