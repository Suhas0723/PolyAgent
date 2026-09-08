# PolyAgent

A Polymarket prediction-market trading bot. It pulls live market data from Polymarket's
Gamma API, filters for thin, low-attention markets where a generalist analyst has a
plausible edge, then runs candidates through a two-tier Claude pipeline — Haiku screens
the universe cheaply and discards most of it, Sonnet produces calibrated probabilities on
the survivors. An edge scanner flags four structural patterns along the way:
pair-accumulation arbitrage, informational candidates, ambiguous resolution criteria, and
stale pricing. Forecasts are corrected against tracked Brier scores, sized with fractional
Kelly, and executed against the Polymarket CLOB. Everything persists to SQLite. It ships
in paper mode and runs as a systemd service on a VPS.

## Technologies

| Tool | Role |
|---|---|
| **Python 3.10+** | Implementation language |
| **Polymarket Gamma API** | Market discovery — questions, prices, liquidity, resolution text |
| **Polymarket CLOB API** | Real executable bid/ask; order placement in live mode |
| **Claude Haiku 4.5** | Tier-1 screener — cheap triage over the candidate set |
| **Claude Sonnet 5** | Tier-2 forecaster — deep probability estimation |
| **`anthropic`** (>= 1.0.0) | Official Python SDK for the Claude API |
| **`httpx`** | Async HTTP client |
| **SQLite** | Persistence for markets, prices, forecasts, positions, calibration |
| **`py-clob-client`** *(optional)* | Order signing and submission for live execution |
| **systemd** | Process supervision and restart-on-failure |
| **`unittest`** | 42-test suite, no network required |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # set ANTHROPIC_API_KEY
polyagent init-db
polyagent cycle
```

| Command | Purpose |
|---|---|
| `polyagent scan` | Edge scanner only, print ranked flags |
| `polyagent cycle` | One full cycle |
| `polyagent run` | Continuous loop (what systemd runs) |
| `polyagent poll` | Fast pair-accumulation poller |
| `polyagent settle` | Sync resolutions, settle positions, score forecasts |
| `polyagent calibration` | Brier scores by category vs. the market |
| `polyagent status` | Counts, open exposure, realized P&L |

Deploy with `sudo bash deploy/install.sh`. Test with `python -m unittest discover -s tests`.

## Notes

Defaults to paper mode. Live trading requires an explicit config change and separate
credentials — worth leaving off until `polyagent calibration` shows positive
`skill_vs_market` on a real sample.

A few constraints shaped the design: Gamma mid-prices always sum to exactly 1.00, so
arbitrage detection needs real order book asks; `resolutionSource` is usually blank and is
not a signal; edges below ~10 points are consumed by fees and slippage; and the
`anthropic` SDK removed `temperature` in v1.0, so sampling is unpinned and calibration
does the work determinism used to.

Prediction market trading risks total loss of capital. Nothing here is financial advice.
