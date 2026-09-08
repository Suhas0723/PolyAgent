import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["POLYAGENT_LLM_STUB"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""

from polyagent.calibration import Calibrator, brier, score_pending
from polyagent.config import Config, ConfigError
from polyagent.db import Database, iso_now, utc_now
from polyagent.edge_scanner import (
    detect_pair_accumulation,
    detect_resolution_ambiguity,
    detect_stale_price,
    run_scan,
)
from polyagent.executor import PaperExecutor
from polyagent.gamma import Market, market_from_payload, parse_json_field
from polyagent.kelly import kelly_fraction_binary, size_position
from polyagent.llm import TwoTierPipeline, extract_json, LLMParseError, coerce_probability
from polyagent.pipeline import Engine


def make_config(**overrides):
    config = Config(anthropic_api_key="", llm_stub=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_market(**overrides):
    defaults = dict(
        condition_id="0xabc123",
        question="Will the agency publish the report before the deadline?",
        description=(
            "This market resolves YES if the official report is published according to "
            "the agency website on or before 2026-12-31 at 11:59 PM ET. Resolution will "
            "use official results published at https://example.gov/reports."
        ),
        category="Politics",
        end_date=(utc_now() + timedelta(days=20)).isoformat(),
        liquidity=5000.0,
        volume_24hr=1200.0,
        comment_count=4,
        outcomes=["Yes", "No"],
        outcome_prices=[0.40, 0.60],
        token_ids=["111", "222"],
    )
    defaults.update(overrides)
    return Market(**defaults)


class TestGammaParsing(unittest.TestCase):
    def test_stringified_json_fields(self):
        raw = {
            "conditionId": "0xdeadbeef",
            "question": "Test?",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.62", "0.38"]',
            "clobTokenIds": '["777", "888"]',
            "liquidity": "1500.5",
            "volume24hr": "300",
            "endDate": "2026-11-01T00:00:00Z",
        }
        market = market_from_payload(raw, {"id": 9, "title": "Event"})
        self.assertTrue(market.is_binary)
        self.assertAlmostEqual(market.yes_price, 0.62)
        self.assertAlmostEqual(market.no_price, 0.38)
        self.assertAlmostEqual(market.pair_cost, 1.0)
        self.assertEqual(market.yes_token_id, "777")
        self.assertEqual(market.no_token_id, "888")

    def test_malformed_fields_do_not_raise(self):
        self.assertEqual(parse_json_field("not json"), [])
        self.assertEqual(parse_json_field(None), [])
        self.assertEqual(parse_json_field(""), [])
        self.assertEqual(parse_json_field(["a"]), ["a"])

    def test_no_first_ordering(self):
        raw = {
            "conditionId": "0x1",
            "question": "Test?",
            "outcomes": '["No", "Yes"]',
            "outcomePrices": '["0.30", "0.70"]',
            "clobTokenIds": '["A", "B"]',
        }
        market = market_from_payload(raw)
        self.assertAlmostEqual(market.yes_price, 0.70)
        self.assertEqual(market.yes_token_id, "B")


class TestJsonExtraction(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"probability": 0.4}')["probability"], 0.4)

    def test_fenced_output(self):
        text = '```json\n{"probability": 0.55}\n```'
        self.assertEqual(extract_json(text)["probability"], 0.55)

    def test_preamble_and_postamble(self):
        text = 'Sure, here is my analysis:\n{"probability": 0.31}\nHope that helps!'
        self.assertEqual(extract_json(text)["probability"], 0.31)

    def test_braces_inside_strings(self):
        text = 'noise {"reasoning": "uses {curly} braces and \\"quotes\\"", "probability": 0.2} tail'
        parsed = extract_json(text)
        self.assertEqual(parsed["probability"], 0.2)
        self.assertIn("curly", parsed["reasoning"])

    def test_trailing_comma_repair(self):
        text = '{"probability": 0.9, "confidence": 0.8,}'
        self.assertEqual(extract_json(text)["confidence"], 0.8)

    def test_array_expectation(self):
        text = 'Here you go:\n[{"id": "a", "promote": true}]'
        parsed = extract_json(text, expect="array")
        self.assertEqual(parsed[0]["id"], "a")

    def test_empty_raises(self):
        with self.assertRaises(LLMParseError):
            extract_json("   ")

    def test_percent_probability_normalized(self):
        self.assertAlmostEqual(coerce_probability(65), 0.65)
        self.assertAlmostEqual(coerce_probability("0.65"), 0.65)
        self.assertAlmostEqual(coerce_probability(1.0), 1.0)
        self.assertAlmostEqual(coerce_probability("bad", 0.5), 0.5)


class TestKelly(unittest.TestCase):
    def test_full_kelly_formula(self):
        self.assertAlmostEqual(kelly_fraction_binary(0.6, 0.5), 0.2, places=6)
        self.assertAlmostEqual(kelly_fraction_binary(0.5, 0.5), 0.0, places=6)
        self.assertEqual(kelly_fraction_binary(0.4, 0.5), 0.0)

    def test_negative_edge_rejected(self):
        config = make_config()
        decision = size_position(config, 0.42, 0.9, 0.40, 0.60)
        self.assertFalse(decision.approved)

    def test_approved_position_is_capped(self):
        config = make_config(bankroll=10000.0, max_position_usd=100.0)
        decision = size_position(config, 0.75, 0.9, 0.40, 0.60)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.side, "YES")
        self.assertLessEqual(decision.stake_usd, 100.0)
        self.assertAlmostEqual(decision.shares * decision.entry_price, decision.stake_usd, places=2)

    def test_no_side_selected_when_model_is_low(self):
        config = make_config(bankroll=5000.0)
        decision = size_position(config, 0.20, 0.9, 0.45, 0.55)
        self.assertTrue(decision.approved)
        self.assertEqual(decision.side, "NO")

    def test_low_confidence_rejected(self):
        config = make_config(min_confidence=0.7)
        decision = size_position(config, 0.80, 0.5, 0.40, 0.60)
        self.assertFalse(decision.approved)

    def test_daily_cap_enforced(self):
        config = make_config(bankroll=10000.0, max_daily_deployment_usd=10.0,
                             min_position_usd=25.0)
        decision = size_position(config, 0.80, 0.9, 0.40, 0.60, deployed_today=9.0)
        self.assertFalse(decision.approved)


class TestEdgeDetectors(unittest.TestCase):
    def test_pair_accumulation_live(self):
        config = make_config(fee_rate=0.0)
        market = make_market(outcome_prices=[0.45, 0.55])
        market.yes_ask, market.no_ask = 0.45, 0.50
        flag = detect_pair_accumulation(market, config, [])
        self.assertIsNotNone(flag)
        self.assertEqual(flag.payload["kind"], "simultaneous")
        self.assertEqual(flag.payload["price_source"], "clob_book_ask")

    def test_pair_accumulation_staged_across_snapshots(self):
        config = make_config(fee_rate=0.0)
        market = make_market(outcome_prices=[0.48, 0.52])
        market.yes_ask, market.no_ask = 0.48, 0.54
        snapshots = [{"yes_price": 0.42, "no_price": 0.55, "captured_at": iso_now()}]
        flag = detect_pair_accumulation(market, config, snapshots)
        self.assertIsNotNone(flag)
        self.assertEqual(flag.payload["kind"], "staged")

    def test_no_pair_flag_at_fair_pricing(self):
        config = make_config()
        market = make_market(outcome_prices=[0.50, 0.50])
        market.yes_ask, market.no_ask = 0.50, 0.51
        self.assertIsNone(detect_pair_accumulation(market, config, []))

    def test_gamma_mids_alone_never_flag_pair_accumulation(self):
        config = make_config(fee_rate=0.0)
        market = make_market(outcome_prices=[0.30, 0.70])
        self.assertIsNone(detect_pair_accumulation(market, config, []))

    def test_blank_resolution_source_is_not_ambiguity(self):
        market = make_market(resolution_source="")
        self.assertIsNone(detect_resolution_ambiguity(market))

    def test_missing_description_flags(self):
        market = make_market(description="Resolves yes or no.")
        flag = detect_resolution_ambiguity(market)
        self.assertIsNotNone(flag)
        self.assertEqual(flag.payload["reason"], "missing_description")

    def test_vague_prose_flags(self):
        market = make_market(
            description=(
                "This market resolves YES if the situation is generally considered to have "
                "substantially improved, at the discretion of the resolver, or similar "
                "widely reported outcomes, etc."
            )
        )
        flag = detect_resolution_ambiguity(market)
        self.assertIsNotNone(flag)
        self.assertGreater(flag.score, 0.45)

    def test_stale_price_requires_window(self):
        config = make_config(stale_price_hours=48.0)
        market = make_market()
        now = utc_now()
        snapshots = [
            {"yes_price": 0.40, "captured_at": (now - timedelta(hours=h)).isoformat()}
            for h in (1, 20, 40)
        ]
        flag = detect_stale_price(market, config, snapshots)
        self.assertIsNotNone(flag)
        self.assertEqual(flag.payload["snapshot_count"], 3)

    def test_moving_price_not_stale(self):
        config = make_config()
        market = make_market()
        now = utc_now()
        snapshots = [
            {"yes_price": price, "captured_at": (now - timedelta(hours=h)).isoformat()}
            for price, h in ((0.40, 1), (0.55, 20), (0.30, 40))
        ]
        self.assertIsNone(detect_stale_price(market, config, snapshots))


class TestInformationalDetector(unittest.TestCase):
    def test_zero_flow_market_is_rejected(self):
        from polyagent.edge_scanner import detect_informational
        config = make_config()
        market = make_market(volume_24hr=0.0, volume_1wk=0.0)
        self.assertIsNone(detect_informational(market, config))

    def test_tradeable_quiet_market_flags(self):
        from polyagent.edge_scanner import detect_informational
        config = make_config()
        market = make_market(volume_24hr=800.0, volume_1wk=6000.0, liquidity=6000.0,
                             comment_count=2)
        flag = detect_informational(market, config)
        self.assertIsNotNone(flag)
        self.assertGreaterEqual(flag.score, 0.62)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name).initialize()

    def tearDown(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_upsert_and_snapshot(self):
        market = make_market()
        with self.db.transaction():
            self.db.upsert_market(market)
            self.db.insert_price_snapshot(market)
            self.db.upsert_market(market)
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM markets"), 1)
        self.assertEqual(len(self.db.recent_snapshots(market.condition_id)), 1)

    def test_scan_run_persists_flags(self):
        config = make_config(fee_rate=0.0)
        market = make_market(outcome_prices=[0.44, 0.56])
        market.yes_ask, market.no_ask = 0.44, 0.50
        with self.db.transaction():
            self.db.upsert_market(market)
        run_id, flags = run_scan(self.db, config, [market])
        self.assertGreater(self.db.scalar("SELECT COUNT(*) FROM edge_flags"), 0)
        self.assertIn(market.condition_id, flags)

    def test_risk_accounting(self):
        market = make_market()
        with self.db.transaction():
            self.db.upsert_market(market)
            self.db.insert_position({
                "condition_id": market.condition_id, "side": "YES", "shares": 100.0,
                "avg_price": 0.4, "cost_basis": 40.0,
            })
        self.assertEqual(self.db.open_position_count(), 1)
        self.assertAlmostEqual(self.db.market_exposure(market.condition_id), 40.0)
        self.assertAlmostEqual(self.db.deployed_today(), 40.0)

    def test_calibration_scoring(self):
        market = make_market()
        with self.db.transaction():
            self.db.upsert_market(market)
            forecast_id = self.db.insert_forecast({
                "condition_id": market.condition_id, "model": "m", "raw_probability": 0.7,
                "calibrated_probability": 0.7, "confidence": 0.8, "market_price": 0.4,
                "edge": 0.3, "side": "YES",
            })
            self.db.record_resolution(market.condition_id, "YES", 1.0, iso_now())
        scored = score_pending(self.db)
        self.assertEqual(scored, 1)
        row = self.db.query_one("SELECT * FROM calibration_scores WHERE forecast_id = ?",
                                (forecast_id,))
        self.assertAlmostEqual(row["model_brier"], brier(0.7, 1.0))
        self.assertAlmostEqual(row["market_brier"], brier(0.4, 1.0))
        self.assertEqual(score_pending(self.db), 0)

    def test_calibrator_shrinks_toward_market(self):
        config = make_config(calibration_shrinkage=0.5)
        calibrator = Calibrator(self.db, config)
        value, note = calibrator.apply(0.9, 0.5, "Politics")
        self.assertAlmostEqual(value, 0.7)
        self.assertEqual(note, "shrinkage_only")


class TestExecutor(unittest.TestCase):
    def test_paper_fill_respects_limit(self):
        config = make_config(slippage_buffer=0.01)
        executor = PaperExecutor(config)
        market = make_market()
        good = executor.buy(market, "YES", 100.0, 0.45)
        self.assertEqual(good.status, "filled")
        self.assertAlmostEqual(good.price, 0.41)
        bad = executor.buy(market, "YES", 100.0, 0.30)
        self.assertEqual(bad.status, "rejected")


class TestPipelineEndToEnd(unittest.TestCase):
    def test_full_cycle_with_stubbed_dependencies(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        config = make_config(
            db_path=tmp.name, bankroll=5000.0, min_edge=0.02, min_confidence=0.1,
            fee_rate=0.0, forecast_candidates_per_cycle=5, screen_candidates_per_cycle=20,
        )
        db = Database(tmp.name).initialize()
        engine = Engine(config, db=db, llm=TwoTierPipeline(config),
                        executor=PaperExecutor(config))

        markets = [
            make_market(
                condition_id=f"0xmarket{i}",
                outcome_prices=[0.30 + i * 0.05, 0.70 - i * 0.05],
                volume_24hr=800.0, volume_1wk=6000.0, liquidity=6000.0,
                question=f"Will official data release {i} be published before the deadline?",
            )
            for i in range(6)
        ]

        async def fake_fetch():
            return markets

        engine.fetch_universe = fake_fetch
        stats = asyncio.run(engine.run_cycle())

        self.assertEqual(stats["markets_fetched"], 6)
        self.assertGreater(stats["screened"], 0)
        self.assertGreaterEqual(stats["forecasts"], 0)
        self.assertEqual(
            db.scalar("SELECT status FROM cycle_runs ORDER BY id DESC LIMIT 1"), "complete"
        )

        engine.close()
        os.unlink(tmp.name)


class TestConfigValidation(unittest.TestCase):
    def test_live_mode_requires_key(self):
        config = make_config(execution_mode="live", clob_private_key="")
        with self.assertRaises(ConfigError):
            config.validate()

    def test_bad_kelly_fraction(self):
        with self.assertRaises(ConfigError):
            make_config(kelly_fraction=1.5).validate()

    def test_stub_mode_allows_missing_api_key(self):
        make_config(llm_stub=True, anthropic_api_key="").validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestExecutionPriceConsistency(unittest.TestCase):
    def test_ask_used_for_cost_mid_used_for_edge(self):
        config = make_config(bankroll=5000.0, slippage_buffer=0.01, fee_rate=0.0)
        market = make_market(outcome_prices=[0.40, 0.60])
        market.yes_ask, market.no_ask = 0.44, 0.62
        decision = size_position(
            config, probability=0.75, confidence=0.9,
            yes_price=market.yes_price, no_price=market.no_price,
            yes_cost=market.price_for("YES"), no_cost=market.price_for("NO"),
        )
        self.assertTrue(decision.approved)
        self.assertAlmostEqual(decision.edge, 0.35, places=6)
        self.assertAlmostEqual(decision.entry_price, 0.45, places=6)

    def test_paper_limit_matches_sizing_limit(self):
        config = make_config(bankroll=5000.0, slippage_buffer=0.01, fee_rate=0.0)
        market = make_market(outcome_prices=[0.40, 0.60])
        market.yes_ask, market.no_ask = 0.44, 0.62
        decision = size_position(
            config, probability=0.75, confidence=0.9,
            yes_price=market.yes_price, no_price=market.no_price,
            yes_cost=market.price_for("YES"), no_cost=market.price_for("NO"),
        )
        fill = PaperExecutor(config).buy(
            market, decision.side, decision.shares, decision.entry_price
        )
        self.assertEqual(fill.status, "filled")
        self.assertAlmostEqual(fill.price, decision.entry_price, places=6)
