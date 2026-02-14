CREATE TABLE IF NOT EXISTS stock_info (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    market VARCHAR(2) NOT NULL, -- SH or SZ
    industry VARCHAR(100),
    list_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kline_daily (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    datetime TIMESTAMP NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    amount DOUBLE PRECISION,
    UNIQUE(symbol, datetime)
);

CREATE INDEX idx_kline_daily_symbol ON kline_daily(symbol);
CREATE INDEX idx_kline_daily_datetime ON kline_daily(datetime);
CREATE INDEX idx_kline_daily_symbol_datetime ON kline_daily(symbol, datetime);
