# PolyAgent

A Polymarket prediction-market trading bot. It pulls live market data from Polymarket's
Gamma API, screens the universe through a two-tier Claude Haiku → Sonnet analysis
pipeline, sizes positions with fractional Kelly, and executes trades against the
Polymarket CLOB. State lives in SQLite. It runs as a systemd service on a Vultr VPS.

It ships in paper mode by default. Live trading requires an explicit config change and
separate credentials.

---

## Tools and dependencies

| Tool | Role |
|---|---|
| **Python 3.10+** | Implementation language for the whole agent |
| **Polymarket Gamma API** | Public, unauthenticated market discovery: questions, prices, liquidity, volume, resolution text |
| **Polymarket CLOB API** | Public `/price` and `/book` endpoints for real executable bid/ask; authenticated order placement in live mode |
| **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) | Tier-1 screener — cheap triage over the whole candidate set |
| **Claude Sonnet 5** (`claude-sonnet-5`) | Tier-2 forecaster — deep probability estimation on survivors |
| **`anthropic`** | Official Python SDK for the Claude API |
| **`httpx`** | Async HTTP client for Gamma and CLOB requests |
| **SQLite** (`sqlite3`, stdlib) | Persistence: markets, price history, edges, forecasts, positions, trades, calibration |
| **`py-clob-client`** *(optional)* | Order signing and submission, only needed for live execution |
| **systemd** | Process supervision, restart-on-failure, log routing |
| **logrotate** | Log rotation on the VPS |
| **`unittest`** (stdlib) | 40-test suite covering parsing, sizing, detectors, and a full stubbed cycle |

No web framework, no ORM, no message queue. The dependency surface is two packages.

---

## How it works

One cycle, run every 15 minutes by default:

```
Gamma API  ──▶  universe filter  ──▶  CLOB book annotation
                                              │
                                              ▼
                                       edge scanner (4 detectors)
                                              │
                                              ▼
                                   Haiku screen  (batched, cheap)
                                              │  promoted only
                                              ▼
                                   Sonnet forecast (one call per market)
                                              │
                                              ▼
                                   calibration correction
                                              │
                                              ▼
                                   fractional Kelly sizing + risk gates
                                              │
                                              ▼
                                   paper or live execution  ──▶  SQLite
```

### 1. Fetch and filter

`gamma.py` pages through `/events` sorted by `volume24hr` ascending, which biases toward
low-volume niche markets. That direction is deliberate: high-volume markets are where
institutional flow lives, and no amount of model quality overcomes an information
disadvantage there.

Gamma returns `outcomes`, `outcomePrices`, and `clobTokenIds` as **stringified JSON
arrays**, not native arrays. They need explicit `json.loads()`. The parser also handles
markets where the outcome order is `["No", "Yes"]` rather than the usual `["Yes", "No"]`,
so the YES side is resolved by label, never by index.

The universe filter drops anything non-binary, illiquid, too heavily traded, priced at
the extremes, or outside the resolution-horizon window.

### 2. Annotate with real book prices

Gamma's `outcomePrices` are **mid-prices that always sum to exactly 1.00**. Verified
against 493 live markets: every single one had `yes + no == 1.0000`. That means any
arbitrage detector built on Gamma prices alone is mathematically dead — it can never
fire.

So the top markets by liquidity get annotated with real asks from the public CLOB
`/price?token_id=…&side=sell` endpoint, which needs no authentication. Those are the
prices you can actually pay, and they are what the pair detector and the position sizer
both use.

### 3. Edge scanner

`edge_scanner.py` runs four independent detectors per market and writes hits to
`edge_scan_runs` / `edge_flags`.

- **`pair_accumulation`** — the arbitrage case. If the YES ask plus the NO ask costs
  less than `1.00 − fees`, buying both locks in profit at settlement regardless of
  outcome. Also checks *staged* windows: the cheapest YES seen at one timestamp plus the
  cheapest NO seen at another, since the two legs rarely go cheap simultaneously. Only
  fires on executable asks, never on mids.
- **`informational`** — quiet, tractable markets: low 24h volume, few comments, a
  resolution horizon around three weeks, and language suggesting the outcome hinges on a
  scheduled public disclosure. Gated on actual recent flow, because a market with zero
  volume is untradeable no matter how mispriced it is.
- **`resolution_ambiguity`** — scores the resolution *prose* for vague language against
  specific criteria. Critically, it **ignores the `resolutionSource` field**. Polymarket
  routinely leaves that field blank even on well-specified markets, embedding the source
  in the description instead; treating blank as ambiguous produced false positives on
  roughly 40% of the universe.
- **`stale_price`** — a price pinned within a tight band across several snapshots over a
  long window, while the resolution date keeps approaching. Requires the agent to have
  been running long enough to have price history.

On a live run this flags roughly 13% of the filtered universe.

### 4. Two-tier LLM pipeline

The two tiers exist for cost control. Deep analysis on every flagged market would be
expensive and mostly wasted; most flags do not survive contact with the actual question.

**Tier 1 — Haiku screener.** Batches markets (default 8 per call) into a single prompt
and asks for one JSON array. It returns a promote/reject decision, a rough probability, a
confidence, and a tractability score. Its job is throwing things away.

**Tier 2 — Sonnet forecaster.** One call per promoted market, with the edge-scanner flags
and the tier-1 verdict included as context. It returns a calibrated probability,
confidence, reasoning, key drivers, and an explicit **resolution risk** note — how the
resolution language could produce an outcome different from the real-world event. That
field exists because being right about the world and wrong about the wording is a
recurring way to lose.

**Output parsing.** Both tiers demand bare JSON, and both get it wrong sometimes in
production. `extract_json` handles: markdown code fences, conversational preamble and
postamble, trailing commas, arrays returned where objects were requested, key aliases
(`yes_probability` for `probability`), probabilities expressed as percentages, and — the
subtle one — a **brace-matching scanner that tracks string state and escapes**, so a `{`
inside a reasoning string doesn't corrupt the extraction. Everything that still fails is
logged to `llm_errors` with the raw output rather than crashing the cycle.

Set `POLYAGENT_LLM_STUB=1` to run the whole pipeline with a deterministic fake backend
and no API key. Useful for testing plumbing without spending tokens.

### 5. Calibration

`calibration.py` tracks Brier scores for every resolved forecast, sliced by market
category, and compares them against the market price's own Brier score over the same
questions. That comparison is the only number that matters: beating your own past
forecasts is meaningless if the market beat both.

The `Calibrator` then corrects new forecasts:

- shrink toward the market price by a configurable factor (the model is not the only
  information source);
- subtract measured directional bias for that category;
- if a category shows **no measured skill against the market**, pull forecasts halfway to
  the market price. The system defers to the market where it has not earned the right not
  to.

Categories below a minimum sample count fall back to global statistics, and with no
history at all only shrinkage applies.

### 6. Kelly sizing

`kelly.py` computes the full Kelly fraction for a binary contract at price `p`:

```
b = (1 − p) / p          net odds received on a winning share
f* = (P·b − q) / b       optimal bankroll fraction
```

Full Kelly is far too aggressive when the probability input is an LLM estimate rather
than a known distribution. It gets scaled down three ways: a fixed fractional multiplier
(default ¼), the model's own confidence, and a hard cap on position fraction.

The sizer separates **fair price** from **execution cost**. Edge is measured against the
mid — that is the market's opinion of fair value. Cost, and therefore the limit price and
share count, use the real ask plus a slippage buffer. Conflating the two silently
mis-sizes every position; an early live run rejected orders because the limit was
computed off the mid while the fill was simulated off the ask.

Every trade must clear, in order: minimum raw edge (default 10 points, since anything
smaller gets eaten by fees and slippage), minimum confidence, positive expected value
*after* fees, open-position cap, per-market exposure cap, daily deployment cap, and
minimum position size. Rejections are logged with the specific reason.

### 7. Execution and settlement

`PaperExecutor` simulates fills at ask plus slippage and honors the limit price.
`ClobExecutor` signs and posts real orders through `py-clob-client`; it refuses to
initialize without a private key. Both write to `trades`, and fills open rows in
`positions`.

`settle` fetches closed markets, records resolutions, computes realized P&L per position,
and feeds the outcomes back into the calibration tables — closing the loop from forecast
to score.

---

## Data model

| Table | Contents |
|---|---|
| `markets` | Market metadata, refreshed each cycle |
| `price_snapshots` | Time series of YES/NO/pair cost — powers stale-price and staged-pair detection |
| `edge_scan_runs`, `edge_flags` | Scanner audit trail |
| `screenings` | Tier-1 verdicts with token usage |
| `forecasts` | Tier-2 output, raw and calibrated probability, edge, chosen side |
| `positions`, `trades` | Holdings and the order log |
| `resolutions` | Settled outcomes |
| `calibration_scores` | Per-forecast Brier scores vs. the market's |
| `cycle_runs` | Per-cycle stats and failures |
| `llm_errors` | Malformed model output, kept with the raw text |

WAL mode, busy timeout, and `BEGIN IMMEDIATE` transactions so the poller and main service
can share one database file.

---

## Setup

```bash
git clone <repo> && cd polyagent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env       # set ANTHROPIC_API_KEY
polyagent init-db
```

## Commands

| Command | Purpose |
|---|---|
| `polyagent init-db` | Create the schema |
| `polyagent config` | Print resolved config, secrets redacted |
| `polyagent scan --top 25` | Run the edge scanner only, print ranked flags |
| `polyagent cycle` | One full cycle |
| `polyagent run` | Continuous loop (this is what systemd runs) |
| `polyagent poll` | Fast pair-accumulation poller |
| `polyagent settle` | Sync resolutions, settle positions, score forecasts |
| `polyagent calibration` | Brier scores by category plus a reliability table |
| `polyagent status` | Counts, open exposure, realized P&L |
| `polyagent positions` | Recent positions |

Start in paper mode and leave it there until `polyagent calibration` shows positive
`skill_vs_market` on a meaningful sample. The system is built to be able to trade; that
is not the same as having earned the right to.

## Deployment (Vultr VPS)

```bash
sudo bash deploy/install.sh
sudo nano /opt/polyagent/.env       # set ANTHROPIC_API_KEY
sudo systemctl start polyagent
journalctl -u polyagent -f
```

The installer creates a locked-down `polyagent` system user, builds a venv under
`/opt/polyagent`, initializes the database, and installs both units plus logrotate.

`polyagent.service` runs the main loop with `Restart=always`, `RestartSec=15`, and
`KillSignal=SIGINT` for clean shutdown mid-cycle. It is sandboxed with `ProtectSystem=strict`,
`PrivateTmp`, `NoNewPrivileges`, and a 1 GB memory cap, with write access limited to the
data and log directories. `polyagent-poller.service` is optional and runs the faster
pair-accumulation poll on a separate schedule.

---

## Testing

```bash
python -m unittest discover -s tests -v
```

40 tests, no network required. Coverage includes stringified-JSON parsing and
reversed-outcome ordering, all seven LLM output-corruption cases, Kelly math against
hand-computed values, every risk gate, each edge detector in both firing and
non-firing states, mid-vs-ask price consistency, settlement and Brier scoring, and a
full cycle end to end against the stub backend.

---

## Design constraints

Findings that shaped the build, several of them the hard way:

- **Gamma mid-prices always sum to 1.00.** Arbitrage detection requires real order book
  asks. Verified across 493 live markets.
- **`resolutionSource` is not a signal.** Blank is the Polymarket norm, not a red flag.
  Resolution criteria live in the prose description.
- **Edges below ~10 points are noise.** Fees and slippage consume them.
- **Adverse selection is architectural.** On heavily traded markets, better prompting
  does not close an information gap. The universe filter avoids them rather than trying
  to compete.
- **Speed arbitrage is out of reach.** Sub-100ms execution needs co-located infrastructure
  and direct RPC access. The bottleneck is network latency, not the model. Every strategy
  here is deliberately latency-tolerant.
- **Calibration beats prompt engineering.** Measuring Brier score against the market and
  correcting for it is a higher-return lever than another round of prompt tuning.

## Disclaimer

Prediction market trading risks total loss of capital. This software is provided as-is,
defaults to paper mode, and nothing here is financial advice. Check that automated
trading on Polymarket is permitted in your jurisdiction before switching to live mode.
