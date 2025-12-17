-- Initial schema for Kraken Scalper

-- Trading positions (open and closed)
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    entry_price TEXT NOT NULL,
    exit_price TEXT,
    quantity TEXT NOT NULL,
    pnl TEXT,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Trade history
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    pair TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    price TEXT NOT NULL,
    quantity TEXT NOT NULL,
    fee TEXT,
    order_id TEXT,
    is_paper INTEGER NOT NULL DEFAULT 1,
    executed_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (position_id) REFERENCES positions(id)
);

-- Performance metrics (daily/hourly snapshots)
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period TEXT NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    total_pnl TEXT NOT NULL DEFAULT '0',
    max_drawdown TEXT,
    balance TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indices for faster queries
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_pair ON positions(pair);
CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair);
CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded_at ON metrics(recorded_at);
