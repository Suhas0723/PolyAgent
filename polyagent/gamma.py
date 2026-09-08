import asyncio
import json
import logging
from dataclasses import dataclass, field

import httpx

LOG = logging.getLogger("polyagent.gamma")

RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)


def parse_json_field(value, default=None):
    if value is None:
        return default if default is not None else []
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default if default is not None else []
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return default if default is not None else []
    return default if default is not None else []


def to_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


@dataclass
class Market:
    condition_id: str
    question: str
    event_id: str = ""
    event_title: str = ""
    market_slug: str = ""
    description: str = ""
    category: str = ""
    resolution_source: str = ""
    end_date: str = ""
    liquidity: float = 0.0
    volume_24hr: float = 0.0
    volume_1wk: float = 0.0
    volume_1mo: float = 0.0
    comment_count: int = 0
    outcomes: list = field(default_factory=list)
    outcome_prices: list = field(default_factory=list)
    token_ids: list = field(default_factory=list)
    active: bool = True
    closed: bool = False
    yes_ask: float = None
    no_ask: float = None

    @property
    def has_executable_prices(self):
        return self.yes_ask is not None and self.no_ask is not None

    @property
    def executable_pair_cost(self):
        if not self.has_executable_prices:
            return None
        return self.yes_ask + self.no_ask

    @property
    def is_binary(self):
        return len(self.outcomes) == 2 and len(self.outcome_prices) == 2

    @property
    def yes_index(self):
        for index, outcome in enumerate(self.outcomes):
            if str(outcome).strip().lower() == "yes":
                return index
        return 0

    @property
    def yes_price(self):
        if not self.outcome_prices:
            return None
        return to_float(self.outcome_prices[self.yes_index])

    @property
    def no_price(self):
        if not self.is_binary:
            return None
        return to_float(self.outcome_prices[1 - self.yes_index])

    @property
    def pair_cost(self):
        yes = self.yes_price
        no = self.no_price
        if yes is None or no is None:
            return None
        return yes + no

    @property
    def yes_token_id(self):
        if len(self.token_ids) > self.yes_index:
            return str(self.token_ids[self.yes_index])
        return None

    @property
    def no_token_id(self):
        if self.is_binary and len(self.token_ids) == 2:
            return str(self.token_ids[1 - self.yes_index])
        return None

    def token_for(self, side):
        return self.yes_token_id if side.upper() == "YES" else self.no_token_id

    def price_for(self, side):
        if side.upper() == "YES":
            return self.yes_ask if self.yes_ask is not None else self.yes_price
        return self.no_ask if self.no_ask is not None else self.no_price


def _resolve_category(raw, event):
    for candidate in (raw.get("category"), event.get("category")):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    tags = event.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                label = tag.get("label") or tag.get("slug")
                if label:
                    return str(label).strip()
            elif isinstance(tag, str) and tag.strip():
                return tag.strip()
    return ""


def market_from_payload(raw, event=None):
    event = event or {}
    outcomes = [str(item) for item in parse_json_field(raw.get("outcomes"), [])]
    prices_raw = parse_json_field(raw.get("outcomePrices"), [])
    outcome_prices = [to_float(item) for item in prices_raw]
    token_ids = [str(item) for item in parse_json_field(raw.get("clobTokenIds"), [])]

    condition_id = raw.get("conditionId") or raw.get("condition_id") or raw.get("id")
    if not condition_id:
        return None

    category = _resolve_category(raw, event)

    return Market(
        condition_id=str(condition_id),
        question=raw.get("question") or raw.get("title") or "",
        event_id=str(event.get("id", "")),
        event_title=event.get("title", ""),
        market_slug=raw.get("slug", ""),
        description=raw.get("description") or event.get("description") or "",
        category=str(category or ""),
        resolution_source=raw.get("resolutionSource") or "",
        end_date=raw.get("endDate") or event.get("endDate") or "",
        liquidity=to_float(raw.get("liquidity"), 0.0) or 0.0,
        volume_24hr=to_float(raw.get("volume24hr"), 0.0) or 0.0,
        volume_1wk=to_float(raw.get("volume1wk"), 0.0) or 0.0,
        volume_1mo=to_float(raw.get("volume1mo"), 0.0) or 0.0,
        comment_count=to_int(event.get("commentCount") or raw.get("commentCount"), 0),
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        token_ids=token_ids,
        active=to_bool(raw.get("active"), True),
        closed=to_bool(raw.get("closed"), False),
    )


class GammaClient:
    def __init__(self, config):
        self.config = config
        self.base_url = config.gamma_base_url.rstrip("/")
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.gamma_timeout_seconds),
            headers={"Accept": "application/json", "User-Agent": "polyagent/1.0"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path, params):
        if self._client is None:
            raise RuntimeError("GammaClient must be used as an async context manager")
        url = f"{self.base_url}{path}"
        delay = 1.0
        last_error = None
        for attempt in range(self.config.gamma_max_retries):
            try:
                response = await self._client.get(url, params=params)
                if response.status_code in RETRY_STATUS:
                    last_error = f"HTTP {response.status_code}"
                    LOG.warning("gamma retryable status %s on %s", response.status_code, url)
                else:
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = repr(exc)
                LOG.warning("gamma request failed (%s/%s): %s",
                            attempt + 1, self.config.gamma_max_retries, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError(f"Gamma request failed after retries: {url} ({last_error})")

    async def fetch_events(self, limit=None, max_pages=None, order=None, ascending=None):
        limit = limit or self.config.gamma_page_limit
        max_pages = max_pages or self.config.gamma_max_pages
        order = order or self.config.gamma_order
        ascending = self.config.gamma_ascending if ascending is None else ascending

        events = []
        for page in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": page * limit,
                "order": order,
                "ascending": "true" if ascending else "false",
            }
            payload = await self._get("/events", params)
            batch = payload if isinstance(payload, list) else payload.get("data", [])
            if not batch:
                break
            events.extend(batch)
            if len(batch) < limit:
                break
        return events

    async def fetch_markets(self):
        events = await self.fetch_events()
        markets = []
        seen = set()
        for event in events:
            for raw in event.get("markets", []) or []:
                market = market_from_payload(raw, event)
                if market is None or market.condition_id in seen:
                    continue
                seen.add(market.condition_id)
                markets.append(market)
        LOG.info("fetched %s markets across %s events", len(markets), len(events))
        return markets

    async def fetch_market(self, condition_id):
        payload = await self._get("/markets", {"condition_ids": condition_id, "limit": 1})
        batch = payload if isinstance(payload, list) else payload.get("data", [])
        if not batch:
            return None
        return market_from_payload(batch[0])

    async def fetch_closed_markets(self, condition_ids):
        results = {}
        semaphore = asyncio.Semaphore(self.config.gamma_concurrency)

        async def worker(condition_id):
            async with semaphore:
                try:
                    payload = await self._get(
                        "/markets", {"condition_ids": condition_id, "limit": 1}
                    )
                except RuntimeError as exc:
                    LOG.warning("resolution lookup failed for %s: %s", condition_id, exc)
                    return
                batch = payload if isinstance(payload, list) else payload.get("data", [])
                if batch:
                    results[condition_id] = batch[0]

        await asyncio.gather(*(worker(cid) for cid in condition_ids))
        return results


def filter_markets(markets, config):
    from .db import parse_iso, utc_now

    kept = []
    now = utc_now()
    excluded = {item.lower() for item in config.excluded_categories}
    for market in markets:
        if market.closed or not market.active:
            continue
        if not market.is_binary:
            continue
        yes = market.yes_price
        if yes is None or not config.min_price <= yes <= config.max_price:
            continue
        if market.liquidity < config.min_liquidity:
            continue
        if market.volume_24hr > config.max_volume_24hr:
            continue
        if market.category and market.category.lower() in excluded:
            continue
        end = parse_iso(market.end_date)
        if end is None:
            continue
        days = (end - now).total_seconds() / 86400.0
        if not config.min_days_to_resolution <= days <= config.max_days_to_resolution:
            continue
        kept.append(market)
    LOG.info("%s markets passed universe filters (from %s)", len(kept), len(markets))
    return kept
