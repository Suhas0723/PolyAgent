import logging
import re
from dataclasses import dataclass, field

from .db import hours_since, parse_iso, utc_now

LOG = logging.getLogger("polyagent.edge_scanner")

EDGE_PAIR_ACCUMULATION = "pair_accumulation"
EDGE_INFORMATIONAL = "informational"
EDGE_RESOLUTION_AMBIGUITY = "resolution_ambiguity"
EDGE_STALE_PRICE = "stale_price"

SOURCE_PATTERNS = [
    r"\baccording to\b",
    r"\bas reported by\b",
    r"\bofficial(?:ly)?\s+(?:results?|announcement|source|data|statement)\b",
    r"\bresolve[sd]?\s+(?:to|according|based)\b",
    r"\bconsensus of credible reporting\b",
    r"\bhttps?://",
    r"\b(?:reuters|associated press|\bap\b|bloomberg|espn|nasa|noaa|cdc|bls|fed|sec\.gov)\b",
]

AMBIGUITY_PATTERNS = [
    r"\bsubjective\b",
    r"\bat the discretion\b",
    r"\bmay be resolved\b",
    r"\bsubstantially\b",
    r"\bsignificant(?:ly)?\b",
    r"\bwidely (?:reported|considered|regarded)\b",
    r"\bgenerally\b",
    r"\breasonable\b",
    r"\bor similar\b",
    r"\betc\.?\b",
    r"\bany other\b",
    r"\bumbrella\b",
    r"\bin the sole judgment\b",
]

SPECIFICITY_PATTERNS = [
    r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\s*(?:et|est|edt|utc|gmt)\b",
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\bexactly\b",
    r"\bstrictly\b",
    r"\bgreater than or equal\b",
    r"\bat least \d",
    r"\bmore than \d",
    r"\bthreshold\b",
]

INFORMATIONAL_KEYWORDS = [
    "report", "release", "announce", "publish", "filing", "data", "index",
    "statistic", "record", "schedule", "deadline", "vote", "hearing",
    "earnings", "approval", "launch", "confirm", "certif",
]

SOURCE_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in SOURCE_PATTERNS]
AMBIGUITY_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in AMBIGUITY_PATTERNS]
SPECIFICITY_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in SPECIFICITY_PATTERNS]


@dataclass
class EdgeFlag:
    condition_id: str
    edge_type: str
    score: float
    detail: str
    payload: dict = field(default_factory=dict)


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _days_to_end(market):
    end = parse_iso(market.end_date)
    if end is None:
        return None
    return (end - utc_now()).total_seconds() / 86400.0


def _count_matches(regexes, text):
    return sum(1 for regex in regexes if regex.search(text))


def detect_pair_accumulation(market, config, snapshots):
    pair_cost = market.executable_pair_cost
    if pair_cost is None:
        return None

    threshold = min(config.pair_accumulation_threshold, 1.0 - config.fee_rate)
    best_yes = market.yes_ask
    best_no = market.no_ask
    window_hours = 0.0

    for snapshot in snapshots:
        snap_yes = snapshot.get("yes_price")
        snap_no = snapshot.get("no_price")
        if snap_yes is not None and snap_yes < best_yes:
            best_yes = snap_yes
        if snap_no is not None and snap_no < best_no:
            best_no = snap_no
        age = hours_since(snapshot.get("captured_at"))
        if age is not None:
            window_hours = max(window_hours, age)

    synthetic_cost = best_yes + best_no
    effective_cost = min(pair_cost, synthetic_cost)
    if effective_cost >= threshold:
        return None

    locked_profit = 1.0 - effective_cost - config.fee_rate
    if locked_profit <= 0:
        return None

    score = _clamp(locked_profit / max(1.0 - threshold, 1e-6))
    kind = "simultaneous" if pair_cost < threshold else "staged"
    return EdgeFlag(
        condition_id=market.condition_id,
        edge_type=EDGE_PAIR_ACCUMULATION,
        score=score,
        detail=(
            f"{kind} pair cost {effective_cost:.4f} vs threshold {threshold:.4f}, "
            f"locked profit {locked_profit:.4f}/share"
        ),
        payload={
            "kind": kind,
            "price_source": "clob_book_ask",
            "live_pair_cost": pair_cost,
            "synthetic_pair_cost": synthetic_cost,
            "best_yes": best_yes,
            "best_no": best_no,
            "threshold": threshold,
            "locked_profit_per_share": locked_profit,
            "observation_window_hours": round(window_hours, 2),
        },
    )


def detect_informational(market, config):
    days = _days_to_end(market)
    if days is None:
        return None

    text = f"{market.question} {market.description}".lower()
    keyword_hits = [word for word in INFORMATIONAL_KEYWORDS if word in text]
    if not keyword_hits:
        return None

    recent_flow = market.volume_24hr + market.volume_1wk
    if recent_flow <= 0:
        return None

    inattention = _clamp(1.0 - (market.volume_24hr / max(config.max_volume_24hr * 0.05, 1.0)))
    tradeability = _clamp(recent_flow / max(config.min_liquidity * 4.0, 1.0))
    attention_component = _clamp(1.0 - (market.comment_count / 40.0))
    horizon_component = _clamp(1.0 - abs(days - 21.0) / 60.0)
    liquidity_component = _clamp(market.liquidity / max(config.min_liquidity * 10.0, 1.0))
    keyword_component = _clamp(len(keyword_hits) / 5.0)

    score = (
        0.22 * inattention
        + 0.24 * tradeability
        + 0.12 * attention_component
        + 0.18 * horizon_component
        + 0.14 * liquidity_component
        + 0.10 * keyword_component
    )
    if score < 0.62:
        return None

    return EdgeFlag(
        condition_id=market.condition_id,
        edge_type=EDGE_INFORMATIONAL,
        score=round(score, 4),
        detail=(
            f"low-attention resolvable market: {market.volume_24hr:.0f} 24h volume, "
            f"{market.comment_count} comments, {days:.1f} days to resolution"
        ),
        payload={
            "days_to_resolution": round(days, 2),
            "volume_24hr": market.volume_24hr,
            "volume_1wk": market.volume_1wk,
            "liquidity": market.liquidity,
            "comment_count": market.comment_count,
            "keywords": keyword_hits[:8],
            "components": {
                "inattention": round(inattention, 4),
                "tradeability": round(tradeability, 4),
                "attention": round(attention_component, 4),
                "horizon": round(horizon_component, 4),
                "liquidity": round(liquidity_component, 4),
                "keywords": round(keyword_component, 4),
            },
        },
    )


def detect_resolution_ambiguity(market):
    description = (market.description or "").strip()
    if len(description) < 40:
        return EdgeFlag(
            condition_id=market.condition_id,
            edge_type=EDGE_RESOLUTION_AMBIGUITY,
            score=0.85,
            detail=f"resolution criteria absent: description is {len(description)} characters",
            payload={"description_length": len(description), "reason": "missing_description"},
        )

    haystack = f"{description} {market.question}"
    source_hits = _count_matches(SOURCE_REGEX, haystack)
    ambiguity_hits = _count_matches(AMBIGUITY_REGEX, haystack)
    specificity_hits = _count_matches(SPECIFICITY_REGEX, haystack)

    if source_hits == 0 and specificity_hits == 0:
        base = 0.60
    elif source_hits == 0:
        base = 0.35
    else:
        base = 0.15

    score = _clamp(base + 0.10 * ambiguity_hits - 0.07 * specificity_hits)
    if score < 0.45:
        return None

    return EdgeFlag(
        condition_id=market.condition_id,
        edge_type=EDGE_RESOLUTION_AMBIGUITY,
        score=round(score, 4),
        detail=(
            f"prose resolution criteria weak: {source_hits} source cues, "
            f"{ambiguity_hits} ambiguity cues, {specificity_hits} specificity cues"
        ),
        payload={
            "source_cues_in_prose": source_hits,
            "ambiguity_cues": ambiguity_hits,
            "specificity_cues": specificity_hits,
            "resolution_source_field": market.resolution_source or None,
            "description_length": len(description),
            "note": "resolutionSource field is not used as a signal; Polymarket routinely leaves it blank",
        },
    )


def detect_stale_price(market, config, snapshots):
    if len(snapshots) < 3:
        return None

    prices = []
    oldest_age = 0.0
    for snapshot in snapshots:
        price = snapshot.get("yes_price")
        age = hours_since(snapshot.get("captured_at"))
        if price is None or age is None:
            continue
        if age <= config.stale_price_hours:
            prices.append(price)
            oldest_age = max(oldest_age, age)

    if len(prices) < 3 or oldest_age < config.stale_price_hours * 0.5:
        return None

    spread = max(prices) - min(prices)
    if spread > config.stale_price_tolerance:
        return None

    days = _days_to_end(market)
    if days is None or days <= 0:
        return None

    current = market.yes_price
    decay_pressure = _clamp(1.0 - (days / 30.0))
    extremity = _clamp(1.0 - abs(current - 0.5) * 2.0)
    score = _clamp(0.4 + 0.35 * decay_pressure + 0.25 * extremity)

    return EdgeFlag(
        condition_id=market.condition_id,
        edge_type=EDGE_STALE_PRICE,
        score=round(score, 4),
        detail=(
            f"price pinned at {current:.3f} across {len(prices)} snapshots over "
            f"{oldest_age:.1f}h (spread {spread:.4f}) with {days:.1f} days remaining"
        ),
        payload={
            "snapshot_count": len(prices),
            "window_hours": round(oldest_age, 2),
            "price_spread": round(spread, 5),
            "current_price": current,
            "days_to_resolution": round(days, 2),
        },
    )


def scan_market(market, config, snapshots):
    flags = []
    for detector in (
        lambda: detect_pair_accumulation(market, config, snapshots),
        lambda: detect_informational(market, config),
        lambda: detect_resolution_ambiguity(market),
        lambda: detect_stale_price(market, config, snapshots),
    ):
        try:
            flag = detector()
        except Exception as exc:
            LOG.exception("detector failed for %s: %s", market.condition_id, exc)
            continue
        if flag is not None:
            flags.append(flag)
    return flags


def run_scan(db, config, markets):
    run_id = db.start_scan_run()
    total_flags = 0
    flags_by_market = {}
    try:
        for market in markets:
            snapshots = db.recent_snapshots(market.condition_id, limit=60)
            flags = scan_market(market, config, snapshots)
            if not flags:
                continue
            flags_by_market[market.condition_id] = flags
            for flag in flags:
                db.insert_edge_flag(run_id, flag)
                total_flags += 1
        db.finish_scan_run(run_id, len(markets), total_flags)
    except Exception as exc:
        db.finish_scan_run(run_id, len(markets), total_flags, status="failed", notes=str(exc)[:500])
        raise
    LOG.info("edge scan %s raised %s flags across %s markets", run_id, total_flags, len(markets))
    return run_id, flags_by_market
