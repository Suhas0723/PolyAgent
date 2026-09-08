import argparse
import asyncio
import json
import logging
import sys

from . import __version__, calibration
from .config import Config, ConfigError
from .db import Database
from .edge_scanner import run_scan
from .clob_prices import BookPriceClient
from .gamma import GammaClient, filter_markets
from .pipeline import Engine, run_forever

LOG = logging.getLogger("polyagent")


def setup_logging(level):
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def cmd_init_db(config, args):
    with Database(config.db_path) as db:
        db.initialize()
    print(f"initialized schema at {config.db_path}")
    return 0


def cmd_config(config, args):
    print(json.dumps(config.redacted(), indent=2, default=str))
    return 0


def cmd_scan(config, args):
    async def _run():
        async with GammaClient(config) as client:
            markets = await client.fetch_markets()
        universe = filter_markets(markets, config)
        ranked = sorted(universe, key=lambda m: m.liquidity, reverse=True)
        async with BookPriceClient(config) as book:
            await book.annotate(ranked[: config.book_annotation_limit])
        return universe

    markets = asyncio.run(_run())
    with Database(config.db_path) as db:
        db.initialize()
        with db.transaction():
            for market in markets:
                db.upsert_market(market)
                db.insert_price_snapshot(market)
        run_id, flags_by_market = run_scan(db, config, markets)

    total = sum(len(items) for items in flags_by_market.values())
    print(f"scan run {run_id}: {len(markets)} markets, {total} flags")

    rows = []
    for condition_id, flags in flags_by_market.items():
        for flag in flags:
            rows.append((flag.score, flag.edge_type, condition_id, flag.detail))
    rows.sort(reverse=True)
    for score, edge_type, condition_id, detail in rows[: args.top]:
        print(f"  {score:.3f}  {edge_type:<22} {condition_id[:20]}  {detail}")
    return 0


def cmd_cycle(config, args):
    engine = Engine(config)
    try:
        stats = asyncio.run(engine.run_cycle())
        print(json.dumps(stats, indent=2))
    finally:
        engine.close()
    return 0


def cmd_run(config, args):
    engine = Engine(config)
    try:
        asyncio.run(run_forever(engine))
    except KeyboardInterrupt:
        LOG.info("shutdown requested")
    finally:
        engine.close()
    return 0


def cmd_poll(config, args):
    engine = Engine(config)
    try:
        asyncio.run(engine.poll_pairs(iterations=args.iterations))
    except KeyboardInterrupt:
        LOG.info("shutdown requested")
    finally:
        engine.close()
    return 0


def cmd_settle(config, args):
    engine = Engine(config)
    try:
        recorded = asyncio.run(engine.sync_resolutions())
        settled = engine.settle_positions()
        scored = calibration.score_pending(engine.db)
    finally:
        engine.close()
    print(json.dumps(
        {"resolutions_recorded": recorded, "positions_settled": settled,
         "forecasts_scored": scored},
        indent=2,
    ))
    return 0


def cmd_calibration(config, args):
    with Database(config.db_path) as db:
        db.initialize()
        calibration.score_pending(db)
        report = calibration.category_report(db)
        bins = calibration.reliability_bins(db)

    if not report:
        print("no scored forecasts yet")
        return 0

    print(f"{'slice':<24}{'n':>6}{'model':>10}{'market':>10}{'skill':>10}{'bias':>10}")
    for row in report:
        print(
            f"{row['category'][:23]:<24}{row['samples']:>6}"
            f"{row['model_brier']:>10.4f}{row['market_brier']:>10.4f}"
            f"{row['skill_vs_market']:>10.4f}{row['bias']:>+10.4f}"
        )
    if bins:
        print("\nreliability")
        for bucket in bins:
            print(
                f"  {bucket['range']:<12}n={bucket['samples']:<6}"
                f"predicted={bucket['mean_forecast']:.3f}  observed={bucket['observed_rate']:.3f}"
            )
    return 0


def cmd_status(config, args):
    with Database(config.db_path) as db:
        db.initialize()
        summary = {
            "db_path": config.db_path,
            "execution_mode": config.execution_mode,
            "markets_tracked": db.scalar("SELECT COUNT(*) FROM markets", (), 0),
            "price_snapshots": db.scalar("SELECT COUNT(*) FROM price_snapshots", (), 0),
            "edge_flags": db.scalar("SELECT COUNT(*) FROM edge_flags", (), 0),
            "screenings": db.scalar("SELECT COUNT(*) FROM screenings", (), 0),
            "forecasts": db.scalar("SELECT COUNT(*) FROM forecasts", (), 0),
            "open_positions": db.open_position_count(),
            "open_cost_basis": round(float(db.scalar(
                "SELECT COALESCE(SUM(cost_basis),0) FROM positions WHERE status='open'", (), 0.0
            )), 2),
            "realized_pnl": round(float(db.scalar(
                "SELECT COALESCE(SUM(realized_pnl),0) FROM positions WHERE status='settled'",
                (), 0.0
            )), 2),
            "trades": db.scalar("SELECT COUNT(*) FROM trades", (), 0),
            "llm_errors": db.scalar("SELECT COUNT(*) FROM llm_errors", (), 0),
            "last_cycle": db.query_one(
                "SELECT * FROM cycle_runs ORDER BY id DESC LIMIT 1"
            ),
        }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_positions(config, args):
    with Database(config.db_path) as db:
        db.initialize()
        rows = db.query(
            """
            SELECT p.id, p.condition_id, p.side, p.status, p.shares, p.avg_price,
                   p.cost_basis, p.edge_at_entry, p.realized_pnl, m.question
            FROM positions p LEFT JOIN markets m ON m.condition_id = p.condition_id
            ORDER BY p.id DESC LIMIT ?
            """,
            (args.limit,),
        )
    if not rows:
        print("no positions")
        return 0
    for row in rows:
        pnl = row["realized_pnl"]
        pnl_text = f"{pnl:+.2f}" if pnl is not None else "open"
        print(
            f"#{row['id']:<5}{row['side']:<5}{row['status']:<9}"
            f"{row['shares']:>9.2f} @ {row['avg_price']:.4f}  "
            f"cost={row['cost_basis']:.2f}  pnl={pnl_text}  "
            f"{(row['question'] or row['condition_id'])[:60]}"
        )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="polyagent", description="Polymarket trading agent")
    parser.add_argument("--version", action="version", version=f"polyagent {__version__}")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    sub.add_parser("config").set_defaults(func=cmd_config)

    scan = sub.add_parser("scan")
    scan.add_argument("--top", type=int, default=25)
    scan.set_defaults(func=cmd_scan)

    sub.add_parser("cycle").set_defaults(func=cmd_cycle)
    sub.add_parser("run").set_defaults(func=cmd_run)

    poll = sub.add_parser("poll")
    poll.add_argument("--iterations", type=int, default=None)
    poll.set_defaults(func=cmd_poll)

    sub.add_parser("settle").set_defaults(func=cmd_settle)
    sub.add_parser("calibration").set_defaults(func=cmd_calibration)
    sub.add_parser("status").set_defaults(func=cmd_status)

    positions = sub.add_parser("positions")
    positions.add_argument("--limit", type=int, default=30)
    positions.set_defaults(func=cmd_positions)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.env_file)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    setup_logging(args.log_level.upper() if args.log_level else config.log_level)
    try:
        return args.func(config, args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        LOG.exception("command failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
