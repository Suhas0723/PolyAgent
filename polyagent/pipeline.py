import asyncio
import logging

from . import calibration
from .db import Database, iso_now, utc_now
from .edge_scanner import (
    EDGE_INFORMATIONAL,
    EDGE_PAIR_ACCUMULATION,
    EDGE_RESOLUTION_AMBIGUITY,
    EDGE_STALE_PRICE,
    run_scan,
)
from .executor import build_executor
from .clob_prices import BookPriceClient
from .gamma import GammaClient, filter_markets, to_float
from .kelly import size_position
from .llm import TwoTierPipeline

LOG = logging.getLogger("polyagent.pipeline")

EDGE_PRIORITY = {
    EDGE_PAIR_ACCUMULATION: 4.0,
    EDGE_INFORMATIONAL: 2.0,
    EDGE_STALE_PRICE: 1.5,
    EDGE_RESOLUTION_AMBIGUITY: -1.0,
}


def candidate_score(flags):
    score = 0.0
    for flag in flags:
        score += EDGE_PRIORITY.get(flag.edge_type, 0.0) * flag.score
    return score


def rank_candidates(markets, flags_by_market, limit):
    scored = []
    for market in markets:
        flags = flags_by_market.get(market.condition_id, [])
        if not flags:
            continue
        scored.append((candidate_score(flags), market, flags))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(market, flags) for score, market, flags in scored[:limit] if score > 0]


class Engine:
    def __init__(self, config, db=None, llm=None, executor=None):
        self.config = config
        self.db = db or Database(config.db_path).initialize()
        self.llm = llm or TwoTierPipeline(config)
        self.executor = executor or build_executor(config)

    def close(self):
        self.executor.close()
        self.db.close()

    def _llm_error(self, stage, model, condition_id, error_type, message, raw_output):
        LOG.error("llm %s %s on %s: %s", stage, error_type, condition_id or "batch", message)
        self.db.log_llm_error(stage, model, condition_id, error_type, message, raw_output)

    async def fetch_universe(self):
        async with GammaClient(self.config) as client:
            markets = await client.fetch_markets()
        universe = filter_markets(markets, self.config)
        await self.annotate_books(universe)
        return universe

    async def annotate_books(self, markets):
        ranked = sorted(markets, key=lambda m: m.liquidity, reverse=True)
        subset = ranked[: self.config.book_annotation_limit]
        if not subset:
            return 0
        async with BookPriceClient(self.config) as client:
            return await client.annotate(subset)

    def persist_universe(self, markets):
        with self.db.transaction():
            for market in markets:
                self.db.upsert_market(market)
                self.db.insert_price_snapshot(market)

    def screen_stage(self, candidates, run_id):
        markets = [market for market, _ in candidates]
        if not markets:
            return {}
        results = self.llm.screen(markets, on_error=self._llm_error)
        with self.db.transaction():
            for condition_id, result in results.items():
                screening_id = self.db.insert_screening(
                    condition_id, run_id, result["model"], result, result.get("usage", {})
                )
                result["screening_id"] = screening_id
        return results

    def forecast_stage(self, promoted, calibrator):
        forecasts = []
        for market, flags, screening in promoted:
            result = self.llm.forecast(market, flags, screening, on_error=self._llm_error)
            if result is None:
                continue

            market_price = market.yes_price
            calibrated, note = calibrator.apply(
                result["probability"], market_price, market.category
            )
            side = "YES" if calibrated > market_price else "NO"
            edge = abs(calibrated - market_price)

            record = {
                "condition_id": market.condition_id,
                "screening_id": screening.get("screening_id") if screening else None,
                "model": result["model"],
                "raw_probability": result["probability"],
                "calibrated_probability": calibrated,
                "confidence": result["confidence"],
                "market_price": market_price,
                "edge": edge,
                "side": side,
                "reasoning": result["reasoning"],
                "key_drivers": result["key_drivers"],
                "resolution_risk": result["resolution_risk"],
                "input_tokens": result.get("usage", {}).get("input_tokens"),
                "output_tokens": result.get("usage", {}).get("output_tokens"),
            }
            with self.db.transaction():
                forecast_id = self.db.insert_forecast(record)
            record["forecast_id"] = forecast_id
            record["calibration_note"] = note
            forecasts.append((market, record))
            LOG.info(
                "forecast %s raw=%.3f cal=%.3f market=%.3f edge=%.3f side=%s (%s)",
                market.condition_id, result["probability"], calibrated,
                market_price, edge, side, note,
            )
        return forecasts

    def execute_stage(self, forecasts):
        placed = 0
        deployed = 0.0
        deployed_today = self.db.deployed_today()

        for market, record in forecasts:
            open_positions = self.db.open_position_count()
            exposure = self.db.market_exposure(market.condition_id)

            decision = size_position(
                self.config,
                probability=record["calibrated_probability"],
                confidence=record["confidence"],
                yes_price=market.yes_price,
                no_price=market.no_price,
                yes_cost=market.price_for("YES"),
                no_cost=market.price_for("NO"),
                current_market_exposure=exposure,
                open_positions=open_positions,
                deployed_today=deployed_today + deployed,
            )

            if not decision.approved:
                LOG.info("skip %s: %s", market.condition_id, decision.reason)
                continue

            fill = self.executor.buy(
                market, decision.side, decision.shares, decision.entry_price
            )

            with self.db.transaction():
                if fill.status in ("filled", "submitted"):
                    position_id = self.db.insert_position({
                        "condition_id": market.condition_id,
                        "side": decision.side,
                        "shares": fill.shares,
                        "avg_price": fill.price,
                        "cost_basis": fill.notional,
                        "forecast_id": record["forecast_id"],
                        "kelly_fraction": decision.kelly_scaled,
                        "edge_at_entry": decision.edge,
                    })
                    placed += 1
                    deployed += fill.notional
                else:
                    position_id = None

                self.db.insert_trade({
                    "position_id": position_id,
                    "condition_id": market.condition_id,
                    "token_id": market.token_for(decision.side),
                    "side": decision.side,
                    "action": "buy",
                    "shares": fill.shares or decision.shares,
                    "price": fill.price or decision.entry_price,
                    "notional": fill.notional,
                    "fees": fill.fees,
                    "mode": self.executor.mode,
                    "status": fill.status,
                    "external_order_id": fill.external_order_id,
                    "error": fill.error or None,
                })

            if fill.status not in ("filled", "submitted"):
                LOG.warning("order not placed for %s: %s", market.condition_id, fill.error)

        return placed, deployed

    async def run_cycle(self):
        cycle_id = self.db.start_cycle()
        stats = {
            "markets_fetched": 0,
            "flags_raised": 0,
            "screened": 0,
            "promoted": 0,
            "forecasts": 0,
            "orders_placed": 0,
            "capital_deployed": 0.0,
        }
        try:
            markets = await self.fetch_universe()
            stats["markets_fetched"] = len(markets)
            self.persist_universe(markets)

            run_id, flags_by_market = run_scan(self.db, self.config, markets)
            stats["flags_raised"] = sum(len(items) for items in flags_by_market.values())

            candidates = rank_candidates(
                markets, flags_by_market, self.config.screen_candidates_per_cycle
            )
            screenings = self.screen_stage(candidates, run_id)
            stats["screened"] = len(screenings)

            promoted = []
            for market, flags in candidates:
                screening = screenings.get(market.condition_id)
                if not screening or not screening["promote"]:
                    continue
                promoted.append((market, flags, screening))
            promoted.sort(
                key=lambda item: item[2]["tractability"] * item[2]["confidence"], reverse=True
            )
            promoted = promoted[: self.config.forecast_candidates_per_cycle]
            stats["promoted"] = len(promoted)

            calibration.score_pending(self.db)
            calibrator = calibration.Calibrator(self.db, self.config)

            forecasts = self.forecast_stage(promoted, calibrator)
            stats["forecasts"] = len(forecasts)

            placed, deployed = self.execute_stage(forecasts)
            stats["orders_placed"] = placed
            stats["capital_deployed"] = round(deployed, 2)

            self.db.finish_cycle(cycle_id, stats)
            LOG.info("cycle %s complete: %s", cycle_id, stats)
            return stats
        except Exception as exc:
            LOG.exception("cycle %s failed", cycle_id)
            self.db.finish_cycle(cycle_id, stats, status="failed", error=str(exc)[:1000])
            raise

    async def poll_pairs(self, iterations=None):
        seen = 0
        while iterations is None or seen < iterations:
            seen += 1
            try:
                markets = await self.fetch_universe()
                self.persist_universe(markets)
                run_id, flags_by_market = run_scan(self.db, self.config, markets)
                hits = [
                    flag
                    for flags in flags_by_market.values()
                    for flag in flags
                    if flag.edge_type == EDGE_PAIR_ACCUMULATION
                ]
                if hits:
                    LOG.warning("pair-accumulation windows open: %s", len(hits))
                    for flag in hits:
                        LOG.warning("  %s -> %s", flag.condition_id, flag.detail)
            except Exception:
                LOG.exception("poll iteration failed")
            if iterations is None or seen < iterations:
                await asyncio.sleep(self.config.poller_interval_seconds)

    async def sync_resolutions(self):
        rows = self.db.query(
            """
            SELECT DISTINCT p.condition_id FROM positions p
            LEFT JOIN resolutions r ON r.condition_id = p.condition_id
            WHERE r.condition_id IS NULL
            UNION
            SELECT DISTINCT f.condition_id FROM forecasts f
            LEFT JOIN resolutions r2 ON r2.condition_id = f.condition_id
            WHERE r2.condition_id IS NULL
            """
        )
        condition_ids = [row["condition_id"] for row in rows]
        if not condition_ids:
            return 0

        async with GammaClient(self.config) as client:
            payloads = await client.fetch_closed_markets(condition_ids)

        recorded = 0
        with self.db.transaction():
            for condition_id, raw in payloads.items():
                closed = str(raw.get("closed", "")).lower() in ("true", "1")
                if not closed:
                    continue
                resolved_value = _resolved_yes_value(raw)
                if resolved_value is None:
                    continue
                self.db.record_resolution(
                    condition_id,
                    "YES" if resolved_value >= 0.5 else "NO",
                    resolved_value,
                    raw.get("endDate") or iso_now(),
                )
                recorded += 1
        LOG.info("recorded %s market resolutions", recorded)
        return recorded

    def settle_positions(self):
        rows = self.db.query(
            """
            SELECT p.*, r.resolved_value FROM positions p
            JOIN resolutions r ON r.condition_id = p.condition_id
            WHERE p.status = 'open'
            """
        )
        settled = 0
        with self.db.transaction():
            for row in rows:
                resolved = float(row["resolved_value"])
                payout_per_share = resolved if row["side"] == "YES" else 1.0 - resolved
                payout = payout_per_share * float(row["shares"])
                pnl = payout - float(row["cost_basis"])
                self.db.execute(
                    "UPDATE positions SET status = 'settled', closed_at = ?, realized_pnl = ? "
                    "WHERE id = ?",
                    (iso_now(), round(pnl, 4), row["id"]),
                )
                settled += 1
        LOG.info("settled %s positions", settled)
        return settled


def _resolved_yes_value(raw):
    from .gamma import parse_json_field

    outcomes = [str(item).strip().lower() for item in parse_json_field(raw.get("outcomes"), [])]
    prices = [to_float(item) for item in parse_json_field(raw.get("outcomePrices"), [])]
    if len(outcomes) != 2 or len(prices) != 2 or None in prices:
        return None
    yes_index = outcomes.index("yes") if "yes" in outcomes else 0
    value = prices[yes_index]
    if value is None:
        return None
    if value > 0.99:
        return 1.0
    if value < 0.01:
        return 0.0
    return None


async def run_forever(engine):
    while True:
        started = utc_now()
        try:
            await engine.run_cycle()
            await engine.sync_resolutions()
            engine.settle_positions()
        except Exception:
            LOG.exception("cycle failed; continuing to next interval")
        elapsed = (utc_now() - started).total_seconds()
        sleep_for = max(30.0, engine.config.cycle_interval_seconds - elapsed)
        LOG.info("sleeping %.0fs until next cycle", sleep_for)
        await asyncio.sleep(sleep_for)
