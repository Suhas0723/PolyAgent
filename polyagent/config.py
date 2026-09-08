import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "polyagent.db"


def load_env_file(path=None):
    path = Path(path) if path else DEFAULT_ENV_FILE
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
    return True


def _str(key, default):
    value = os.environ.get(key)
    return value.strip() if value and value.strip() else default


def _float(key, default):
    try:
        return float(os.environ[key])
    except (KeyError, TypeError, ValueError):
        return default


def _int(key, default):
    try:
        return int(os.environ[key])
    except (KeyError, TypeError, ValueError):
        return default


def _bool(key, default):
    value = os.environ.get(key)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _list(key, default):
    value = os.environ.get(key)
    if not value or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


class ConfigError(RuntimeError):
    pass


@dataclass
class Config:
    db_path: str = str(DEFAULT_DB_PATH)
    log_level: str = "INFO"

    anthropic_api_key: str = ""
    llm_stub: bool = False
    screen_model: str = "claude-haiku-4-5-20251001"
    forecast_model: str = "claude-sonnet-5"
    screen_max_tokens: int = 900
    forecast_max_tokens: int = 2000
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 4
    llm_batch_size: int = 8

    gamma_base_url: str = "https://gamma-api.polymarket.com"
    gamma_page_limit: int = 100
    gamma_max_pages: int = 8
    gamma_order: str = "volume24hr"
    gamma_ascending: bool = True
    gamma_timeout_seconds: float = 30.0
    gamma_max_retries: int = 4
    gamma_concurrency: int = 4
    book_annotation_limit: int = 150

    min_liquidity: float = 500.0
    max_volume_24hr: float = 250000.0
    min_days_to_resolution: float = 1.0
    max_days_to_resolution: float = 400.0
    min_price: float = 0.03
    max_price: float = 0.97

    fee_rate: float = 0.02
    slippage_buffer: float = 0.01
    min_edge: float = 0.10
    min_confidence: float = 0.55
    pair_accumulation_threshold: float = 0.985
    stale_price_hours: float = 48.0
    stale_price_tolerance: float = 0.005

    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    max_position_fraction: float = 0.05
    max_position_usd: float = 250.0
    min_position_usd: float = 5.0
    max_open_positions: int = 25
    max_market_exposure_usd: float = 400.0
    max_daily_deployment_usd: float = 500.0

    execution_mode: str = "paper"
    clob_base_url: str = "https://clob.polymarket.com"
    clob_private_key: str = ""
    clob_api_key: str = ""
    clob_api_secret: str = ""
    clob_api_passphrase: str = ""
    clob_funder_address: str = ""
    clob_chain_id: int = 137

    cycle_interval_seconds: int = 900
    poller_interval_seconds: int = 30
    screen_candidates_per_cycle: int = 60
    forecast_candidates_per_cycle: int = 12

    calibration_min_samples: int = 20
    calibration_shrinkage: float = 0.35
    excluded_categories: list = field(default_factory=list)

    @classmethod
    def load(cls, env_file=None):
        load_env_file(env_file)
        cfg = cls(
            db_path=_str("POLYAGENT_DB_PATH", str(DEFAULT_DB_PATH)),
            log_level=_str("POLYAGENT_LOG_LEVEL", "INFO").upper(),
            anthropic_api_key=_str("ANTHROPIC_API_KEY", ""),
            llm_stub=_bool("POLYAGENT_LLM_STUB", False),
            screen_model=_str("POLYAGENT_SCREEN_MODEL", "claude-haiku-4-5-20251001"),
            forecast_model=_str("POLYAGENT_FORECAST_MODEL", "claude-sonnet-5"),
            screen_max_tokens=_int("POLYAGENT_SCREEN_MAX_TOKENS", 900),
            forecast_max_tokens=_int("POLYAGENT_FORECAST_MAX_TOKENS", 2000),
            llm_timeout_seconds=_float("POLYAGENT_LLM_TIMEOUT", 90.0),
            llm_max_retries=_int("POLYAGENT_LLM_MAX_RETRIES", 4),
            llm_batch_size=_int("POLYAGENT_LLM_BATCH_SIZE", 8),
            gamma_base_url=_str("POLYAGENT_GAMMA_URL", "https://gamma-api.polymarket.com"),
            gamma_page_limit=_int("POLYAGENT_GAMMA_PAGE_LIMIT", 100),
            gamma_max_pages=_int("POLYAGENT_GAMMA_MAX_PAGES", 8),
            gamma_order=_str("POLYAGENT_GAMMA_ORDER", "volume24hr"),
            gamma_ascending=_bool("POLYAGENT_GAMMA_ASCENDING", True),
            gamma_timeout_seconds=_float("POLYAGENT_GAMMA_TIMEOUT", 30.0),
            gamma_max_retries=_int("POLYAGENT_GAMMA_MAX_RETRIES", 4),
            gamma_concurrency=_int("POLYAGENT_GAMMA_CONCURRENCY", 4),
            book_annotation_limit=_int("POLYAGENT_BOOK_ANNOTATION_LIMIT", 150),
            min_liquidity=_float("POLYAGENT_MIN_LIQUIDITY", 500.0),
            max_volume_24hr=_float("POLYAGENT_MAX_VOLUME_24HR", 250000.0),
            min_days_to_resolution=_float("POLYAGENT_MIN_DAYS", 1.0),
            max_days_to_resolution=_float("POLYAGENT_MAX_DAYS", 400.0),
            min_price=_float("POLYAGENT_MIN_PRICE", 0.03),
            max_price=_float("POLYAGENT_MAX_PRICE", 0.97),
            fee_rate=_float("POLYAGENT_FEE_RATE", 0.02),
            slippage_buffer=_float("POLYAGENT_SLIPPAGE_BUFFER", 0.01),
            min_edge=_float("POLYAGENT_MIN_EDGE", 0.10),
            min_confidence=_float("POLYAGENT_MIN_CONFIDENCE", 0.55),
            pair_accumulation_threshold=_float("POLYAGENT_PAIR_THRESHOLD", 0.985),
            stale_price_hours=_float("POLYAGENT_STALE_HOURS", 48.0),
            stale_price_tolerance=_float("POLYAGENT_STALE_TOLERANCE", 0.005),
            bankroll=_float("POLYAGENT_BANKROLL", 1000.0),
            kelly_fraction=_float("POLYAGENT_KELLY_FRACTION", 0.25),
            max_position_fraction=_float("POLYAGENT_MAX_POSITION_FRACTION", 0.05),
            max_position_usd=_float("POLYAGENT_MAX_POSITION_USD", 250.0),
            min_position_usd=_float("POLYAGENT_MIN_POSITION_USD", 5.0),
            max_open_positions=_int("POLYAGENT_MAX_OPEN_POSITIONS", 25),
            max_market_exposure_usd=_float("POLYAGENT_MAX_MARKET_EXPOSURE_USD", 400.0),
            max_daily_deployment_usd=_float("POLYAGENT_MAX_DAILY_DEPLOYMENT_USD", 500.0),
            execution_mode=_str("POLYAGENT_EXECUTION_MODE", "paper").lower(),
            clob_base_url=_str("POLYAGENT_CLOB_URL", "https://clob.polymarket.com"),
            clob_private_key=_str("POLYAGENT_CLOB_PRIVATE_KEY", ""),
            clob_api_key=_str("POLYAGENT_CLOB_API_KEY", ""),
            clob_api_secret=_str("POLYAGENT_CLOB_API_SECRET", ""),
            clob_api_passphrase=_str("POLYAGENT_CLOB_API_PASSPHRASE", ""),
            clob_funder_address=_str("POLYAGENT_CLOB_FUNDER", ""),
            clob_chain_id=_int("POLYAGENT_CLOB_CHAIN_ID", 137),
            cycle_interval_seconds=_int("POLYAGENT_CYCLE_INTERVAL", 900),
            poller_interval_seconds=_int("POLYAGENT_POLLER_INTERVAL", 30),
            screen_candidates_per_cycle=_int("POLYAGENT_SCREEN_CANDIDATES", 60),
            forecast_candidates_per_cycle=_int("POLYAGENT_FORECAST_CANDIDATES", 12),
            calibration_min_samples=_int("POLYAGENT_CALIBRATION_MIN_SAMPLES", 20),
            calibration_shrinkage=_float("POLYAGENT_CALIBRATION_SHRINKAGE", 0.35),
            excluded_categories=_list("POLYAGENT_EXCLUDED_CATEGORIES", []),
        )
        cfg.validate()
        return cfg

    def validate(self):
        if self.execution_mode not in ("paper", "live"):
            raise ConfigError("POLYAGENT_EXECUTION_MODE must be 'paper' or 'live'")
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ConfigError("POLYAGENT_KELLY_FRACTION must be in (0, 1]")
        if not 0.0 < self.max_position_fraction <= 1.0:
            raise ConfigError("POLYAGENT_MAX_POSITION_FRACTION must be in (0, 1]")
        if self.bankroll <= 0:
            raise ConfigError("POLYAGENT_BANKROLL must be positive")
        if self.min_price >= self.max_price:
            raise ConfigError("POLYAGENT_MIN_PRICE must be below POLYAGENT_MAX_PRICE")
        if not 0.0 <= self.calibration_shrinkage <= 1.0:
            raise ConfigError("POLYAGENT_CALIBRATION_SHRINKAGE must be in [0, 1]")
        if self.execution_mode == "live" and not self.clob_private_key:
            raise ConfigError("live execution requires POLYAGENT_CLOB_PRIVATE_KEY")
        if not self.llm_stub and not self.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is required unless POLYAGENT_LLM_STUB=1")
        return self

    @property
    def total_cost_drag(self):
        return self.fee_rate + self.slippage_buffer

    def redacted(self):
        data = dict(self.__dict__)
        for secret in (
            "anthropic_api_key",
            "clob_private_key",
            "clob_api_key",
            "clob_api_secret",
            "clob_api_passphrase",
        ):
            if data.get(secret):
                data[secret] = "***redacted***"
        return data
