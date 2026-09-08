import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 3

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS markets (
    condition_id TEXT PRIMARY KEY,
    event_id TEXT,
    market_slug TEXT,
    question TEXT NOT NULL,
    description TEXT,
    category TEXT,
    resolution_source TEXT,
    end_date TEXT,
    liquidity REAL,
    volume_24hr REAL,
    volume_1wk REAL,
    volume_1mo REAL,
    comment_count INTEGER,
    outcomes_json TEXT,
    token_ids_json TEXT,
    active INTEGER DEFAULT 1,
    closed INTEGER DEFAULT 0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_markets_end_date ON markets(end_date);
CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);
CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active, closed);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    yes_price REAL,
    no_price REAL,
    pair_cost REAL,
    liquidity REAL,
    volume_24hr REAL,
    FOREIGN KEY (condition_id) REFERENCES markets(condition_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market_time
    ON price_snapshots(condition_id, captured_at DESC);

CREATE TABLE IF NOT EXISTS edge_scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    markets_scanned INTEGER DEFAULT 0,
    flags_raised INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS edge_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    condition_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    score REAL NOT NULL,
    detail TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES edge_scan_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_flags_type_score ON edge_flags(edge_type, score DESC);
CREATE INDEX IF NOT EXISTS idx_flags_market ON edge_flags(condition_id, created_at DESC);

CREATE TABLE IF NOT EXISTS screenings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    run_id INTEGER,
    model TEXT NOT NULL,
    promote INTEGER NOT NULL,
    quick_probability REAL,
    confidence REAL,
    tractability REAL,
    rationale TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (condition_id) REFERENCES markets(condition_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_screenings_market ON screenings(condition_id, created_at DESC);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    screening_id INTEGER,
    model TEXT NOT NULL,
    raw_probability REAL NOT NULL,
    calibrated_probability REAL NOT NULL,
    confidence REAL NOT NULL,
    market_price REAL NOT NULL,
    edge REAL NOT NULL,
    side TEXT NOT NULL,
    reasoning TEXT,
    key_drivers_json TEXT,
    resolution_risk TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (condition_id) REFERENCES markets(condition_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_forecasts_market ON forecasts(condition_id, created_at DESC);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT NOT NULL,
    side TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    shares REAL NOT NULL,
    avg_price REAL NOT NULL,
    cost_basis REAL NOT NULL,
    forecast_id INTEGER,
    kelly_fraction REAL,
    edge_at_entry REAL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    realized_pnl REAL,
    FOREIGN KEY (condition_id) REFERENCES markets(condition_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status, condition_id);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    condition_id TEXT NOT NULL,
    token_id TEXT,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    notional REAL NOT NULL,
    fees REAL DEFAULT 0,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    external_order_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(condition_id, created_at DESC);

CREATE TABLE IF NOT EXISTS resolutions (
    condition_id TEXT PRIMARY KEY,
    outcome TEXT NOT NULL,
    resolved_value REAL NOT NULL,
    resolved_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (condition_id) REFERENCES markets(condition_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS calibration_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id INTEGER NOT NULL,
    condition_id TEXT NOT NULL,
    category TEXT,
    forecast_probability REAL NOT NULL,
    market_probability REAL NOT NULL,
    outcome_value REAL NOT NULL,
    model_brier REAL NOT NULL,
    market_brier REAL NOT NULL,
    scored_at TEXT NOT NULL,
    UNIQUE (forecast_id)
);

CREATE INDEX IF NOT EXISTS idx_calibration_category ON calibration_scores(category);

CREATE TABLE IF NOT EXISTS cycle_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    markets_fetched INTEGER DEFAULT 0,
    flags_raised INTEGER DEFAULT 0,
    screened INTEGER DEFAULT 0,
    promoted INTEGER DEFAULT 0,
    forecasts INTEGER DEFAULT 0,
    orders_placed INTEGER DEFAULT 0,
    capital_deployed REAL DEFAULT 0,
    status TEXT DEFAULT 'running',
    error TEXT
);

CREATE TABLE IF NOT EXISTS llm_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    model TEXT,
    condition_id TEXT,
    error_type TEXT NOT NULL,
    message TEXT,
    raw_output TEXT,
    created_at TEXT NOT NULL
);
"""


def utc_now():
    return datetime.now(timezone.utc)


def iso_now():
    return utc_now().isoformat()


def parse_iso(value):
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def hours_since(value):
    parsed = parse_iso(value)
    if parsed is None:
        return None
    return (utc_now() - parsed).total_seconds() / 3600.0


class Database:
    def __init__(self, path):
        self.path = str(path)
        parent = Path(self.path).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 30000")

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @contextmanager
    def transaction(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def initialize(self):
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )
        return self

    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)

    def query(self, sql, params=()):
        return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def query_one(self, sql, params=()):
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def scalar(self, sql, params=(), default=None):
        row = self._conn.execute(sql, params).fetchone()
        if row is None or row[0] is None:
            return default
        return row[0]

    def upsert_market(self, market):
        now = iso_now()
        self._conn.execute(
            """
            INSERT INTO markets (
                condition_id, event_id, market_slug, question, description, category,
                resolution_source, end_date, liquidity, volume_24hr, volume_1wk,
                volume_1mo, comment_count, outcomes_json, token_ids_json, active,
                closed, first_seen_at, last_seen_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(condition_id) DO UPDATE SET
                event_id = excluded.event_id,
                market_slug = excluded.market_slug,
                question = excluded.question,
                description = excluded.description,
                category = excluded.category,
                resolution_source = excluded.resolution_source,
                end_date = excluded.end_date,
                liquidity = excluded.liquidity,
                volume_24hr = excluded.volume_24hr,
                volume_1wk = excluded.volume_1wk,
                volume_1mo = excluded.volume_1mo,
                comment_count = excluded.comment_count,
                outcomes_json = excluded.outcomes_json,
                token_ids_json = excluded.token_ids_json,
                active = excluded.active,
                closed = excluded.closed,
                last_seen_at = excluded.last_seen_at
            """,
            (
                market.condition_id,
                market.event_id,
                market.market_slug,
                market.question,
                market.description,
                market.category,
                market.resolution_source,
                market.end_date,
                market.liquidity,
                market.volume_24hr,
                market.volume_1wk,
                market.volume_1mo,
                market.comment_count,
                json.dumps(market.outcomes),
                json.dumps(market.token_ids),
                1 if market.active else 0,
                1 if market.closed else 0,
                now,
                now,
            ),
        )

    def insert_price_snapshot(self, market):
        self._conn.execute(
            """
            INSERT INTO price_snapshots (
                condition_id, captured_at, yes_price, no_price, pair_cost,
                liquidity, volume_24hr
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                market.condition_id,
                iso_now(),
                market.yes_price,
                market.no_price,
                market.pair_cost,
                market.liquidity,
                market.volume_24hr,
            ),
        )

    def recent_snapshots(self, condition_id, limit=50):
        return self.query(
            "SELECT * FROM price_snapshots WHERE condition_id = ? "
            "ORDER BY captured_at DESC LIMIT ?",
            (condition_id, limit),
        )

    def start_scan_run(self):
        cursor = self._conn.execute(
            "INSERT INTO edge_scan_runs (started_at) VALUES (?)", (iso_now(),)
        )
        return cursor.lastrowid

    def finish_scan_run(self, run_id, markets_scanned, flags_raised, status="complete", notes=None):
        self._conn.execute(
            "UPDATE edge_scan_runs SET finished_at = ?, markets_scanned = ?, "
            "flags_raised = ?, status = ?, notes = ? WHERE id = ?",
            (iso_now(), markets_scanned, flags_raised, status, notes, run_id),
        )

    def insert_edge_flag(self, run_id, flag):
        self._conn.execute(
            """
            INSERT INTO edge_flags (
                run_id, condition_id, edge_type, score, detail, payload_json, created_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                run_id,
                flag.condition_id,
                flag.edge_type,
                flag.score,
                flag.detail,
                json.dumps(flag.payload),
                iso_now(),
            ),
        )

    def start_cycle(self):
        cursor = self._conn.execute(
            "INSERT INTO cycle_runs (started_at) VALUES (?)", (iso_now(),)
        )
        return cursor.lastrowid

    def finish_cycle(self, cycle_id, stats, status="complete", error=None):
        self._conn.execute(
            """
            UPDATE cycle_runs SET finished_at = ?, markets_fetched = ?, flags_raised = ?,
                screened = ?, promoted = ?, forecasts = ?, orders_placed = ?,
                capital_deployed = ?, status = ?, error = ?
            WHERE id = ?
            """,
            (
                iso_now(),
                stats.get("markets_fetched", 0),
                stats.get("flags_raised", 0),
                stats.get("screened", 0),
                stats.get("promoted", 0),
                stats.get("forecasts", 0),
                stats.get("orders_placed", 0),
                stats.get("capital_deployed", 0.0),
                status,
                error,
                cycle_id,
            ),
        )

    def insert_screening(self, condition_id, run_id, model, result, usage):
        cursor = self._conn.execute(
            """
            INSERT INTO screenings (
                condition_id, run_id, model, promote, quick_probability, confidence,
                tractability, rationale, input_tokens, output_tokens, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                condition_id,
                run_id,
                model,
                1 if result.get("promote") else 0,
                result.get("quick_probability"),
                result.get("confidence"),
                result.get("tractability"),
                result.get("rationale"),
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                iso_now(),
            ),
        )
        return cursor.lastrowid

    def insert_forecast(self, record):
        cursor = self._conn.execute(
            """
            INSERT INTO forecasts (
                condition_id, screening_id, model, raw_probability, calibrated_probability,
                confidence, market_price, edge, side, reasoning, key_drivers_json,
                resolution_risk, input_tokens, output_tokens, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record["condition_id"],
                record.get("screening_id"),
                record["model"],
                record["raw_probability"],
                record["calibrated_probability"],
                record["confidence"],
                record["market_price"],
                record["edge"],
                record["side"],
                record.get("reasoning"),
                json.dumps(record.get("key_drivers", [])),
                record.get("resolution_risk"),
                record.get("input_tokens"),
                record.get("output_tokens"),
                iso_now(),
            ),
        )
        return cursor.lastrowid

    def insert_position(self, record):
        cursor = self._conn.execute(
            """
            INSERT INTO positions (
                condition_id, side, status, shares, avg_price, cost_basis,
                forecast_id, kelly_fraction, edge_at_entry, opened_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record["condition_id"],
                record["side"],
                record.get("status", "open"),
                record["shares"],
                record["avg_price"],
                record["cost_basis"],
                record.get("forecast_id"),
                record.get("kelly_fraction"),
                record.get("edge_at_entry"),
                iso_now(),
            ),
        )
        return cursor.lastrowid

    def insert_trade(self, record):
        cursor = self._conn.execute(
            """
            INSERT INTO trades (
                position_id, condition_id, token_id, side, action, shares, price,
                notional, fees, mode, status, external_order_id, error, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.get("position_id"),
                record["condition_id"],
                record.get("token_id"),
                record["side"],
                record.get("action", "buy"),
                record["shares"],
                record["price"],
                record["notional"],
                record.get("fees", 0.0),
                record["mode"],
                record["status"],
                record.get("external_order_id"),
                record.get("error"),
                iso_now(),
            ),
        )
        return cursor.lastrowid

    def log_llm_error(self, stage, model, condition_id, error_type, message, raw_output=None):
        self._conn.execute(
            """
            INSERT INTO llm_errors (
                stage, model, condition_id, error_type, message, raw_output, created_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (stage, model, condition_id, error_type, str(message)[:2000],
             (raw_output or "")[:8000], iso_now()),
        )

    def open_positions(self):
        return self.query("SELECT * FROM positions WHERE status = 'open'")

    def open_position_count(self):
        return int(self.scalar("SELECT COUNT(*) FROM positions WHERE status = 'open'", (), 0))

    def market_exposure(self, condition_id):
        return float(
            self.scalar(
                "SELECT COALESCE(SUM(cost_basis), 0) FROM positions "
                "WHERE condition_id = ? AND status = 'open'",
                (condition_id,),
                0.0,
            )
        )

    def deployed_since(self, since_iso):
        return float(
            self.scalar(
                "SELECT COALESCE(SUM(cost_basis), 0) FROM positions WHERE opened_at >= ?",
                (since_iso,),
                0.0,
            )
        )

    def deployed_today(self):
        cutoff = (utc_now() - timedelta(hours=24)).isoformat()
        return self.deployed_since(cutoff)

    def latest_forecast(self, condition_id):
        return self.query_one(
            "SELECT * FROM forecasts WHERE condition_id = ? ORDER BY created_at DESC LIMIT 1",
            (condition_id,),
        )

    def record_resolution(self, condition_id, outcome, resolved_value, resolved_at):
        self._conn.execute(
            """
            INSERT INTO resolutions (condition_id, outcome, resolved_value, resolved_at, recorded_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(condition_id) DO UPDATE SET
                outcome = excluded.outcome,
                resolved_value = excluded.resolved_value,
                resolved_at = excluded.resolved_at
            """,
            (condition_id, outcome, resolved_value, resolved_at, iso_now()),
        )

    def unscored_forecasts(self):
        return self.query(
            """
            SELECT f.id AS forecast_id, f.condition_id, f.calibrated_probability,
                   f.market_price, f.side, m.category, r.resolved_value
            FROM forecasts f
            JOIN resolutions r ON r.condition_id = f.condition_id
            JOIN markets m ON m.condition_id = f.condition_id
            LEFT JOIN calibration_scores c ON c.forecast_id = f.id
            WHERE c.forecast_id IS NULL
            """
        )

    def insert_calibration_score(self, record):
        self._conn.execute(
            """
            INSERT OR IGNORE INTO calibration_scores (
                forecast_id, condition_id, category, forecast_probability,
                market_probability, outcome_value, model_brier, market_brier, scored_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                record["forecast_id"],
                record["condition_id"],
                record.get("category"),
                record["forecast_probability"],
                record["market_probability"],
                record["outcome_value"],
                record["model_brier"],
                record["market_brier"],
                iso_now(),
            ),
        )
