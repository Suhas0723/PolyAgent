import hashlib
import json
import logging
import re
import time

LOG = logging.getLogger("polyagent.llm")

FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")

SCREEN_SYSTEM = """You are the first-stage screener in a prediction market trading system.

You receive a batch of Polymarket binary markets. For each one, decide whether it is
worth spending expensive deep analysis on. Promote a market only when ALL of these hold:

1. The outcome is determinable from public information a careful analyst could gather.
2. The resolution criteria are specific enough that you could argue the outcome.
3. You have a defensible reason to think the market price may be materially wrong.

Do NOT promote markets that are pure coin flips, that hinge on private information,
that depend on unpredictable short-horizon noise, or where the crowd is very likely
better informed than a generalist analyst.

Output rules, which are absolute:
- Reply with a single JSON array and nothing else.
- No prose before or after. No markdown code fences. No trailing commentary.
- One object per input market, in the same order, with these exact keys:
  "id" (string, echo the given id), "promote" (boolean),
  "quick_probability" (number 0-1, your rough probability the YES side resolves true),
  "confidence" (number 0-1), "tractability" (number 0-1, how analyzable this market is),
  "rationale" (string, at most 200 characters)."""

FORECAST_SYSTEM = """You are the deep analysis stage of a prediction market trading system.

You are given one Polymarket binary market, its current price, and structural flags
raised by an automated edge scanner. Produce a calibrated probability that the YES
outcome resolves true.

Discipline you must apply:
- Start from a base rate before adjusting for specifics.
- Treat the market price as informative but not authoritative; state where you diverge and why.
- Read the resolution criteria literally. Many losses come from being right about the
  world and wrong about the resolution language.
- Be honest about confidence. Low confidence is a valid and useful answer.
- Do not anchor to round numbers.

Output rules, which are absolute:
- Reply with a single JSON object and nothing else.
- No prose before or after. No markdown code fences.
- Exact keys: "probability" (number 0-1), "confidence" (number 0-1),
  "reasoning" (string, at most 1200 characters),
  "key_drivers" (array of at most 5 short strings),
  "resolution_risk" (string, at most 300 characters, describing how the resolution
  language could produce an outcome different from the real-world event)."""


class LLMError(RuntimeError):
    pass


class LLMParseError(LLMError):
    pass


def strip_fences(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = FENCE_RE.sub("", cleaned)
    return cleaned.strip()


def _scan_balanced(text, open_char, close_char):
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def extract_json(text, expect="object"):
    if not text or not text.strip():
        raise LLMParseError("empty model output")

    cleaned = strip_fences(text)

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    candidates = []
    if expect == "array":
        candidates.append(_scan_balanced(cleaned, "[", "]"))
        candidates.append(_scan_balanced(cleaned, "{", "}"))
    else:
        candidates.append(_scan_balanced(cleaned, "{", "}"))
        candidates.append(_scan_balanced(cleaned, "[", "]"))

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                continue

    raise LLMParseError(f"no parseable JSON in model output: {cleaned[:300]!r}")


def coerce_probability(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        if default is None:
            raise LLMParseError(f"non-numeric probability: {value!r}")
        return default
    if number != number or number in (float("inf"), float("-inf")):
        if default is None:
            raise LLMParseError(f"non-finite probability: {value!r}")
        return default
    if 1.0 < number <= 100.0:
        number = number / 100.0
    return min(max(number, 0.0), 1.0)


def coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "promote")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def coerce_str(value, limit=2000):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    return str(value)[:limit]


class AnthropicBackend:
    def __init__(self, config):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise LLMError("anthropic package is required; pip install anthropic") from exc
        self.config = config
        self.client = Anthropic(
            api_key=config.anthropic_api_key,
            timeout=config.llm_timeout_seconds,
            max_retries=0,
        )

    def complete(self, model, system, prompt, max_tokens):
        delay = 2.0
        last_error = None
        for attempt in range(self.config.llm_max_retries):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(
                    block.text for block in response.content
                    if getattr(block, "type", "") == "text"
                )
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "output_tokens": getattr(response.usage, "output_tokens", None),
                }
                return text, usage
            except Exception as exc:
                last_error = exc
                LOG.warning("llm call failed (%s/%s) on %s: %s",
                            attempt + 1, self.config.llm_max_retries, model, exc)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise LLMError(f"model {model} failed after retries: {last_error}")


class StubBackend:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _pseudo(seed_text):
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def complete(self, model, system, prompt, max_tokens):
        usage = {"input_tokens": len(prompt) // 4, "output_tokens": 128}
        if "single JSON array" in system:
            ids = re.findall(r'"id":\s*"([^"]+)"', prompt)
            payload = []
            for market_id in ids:
                base = self._pseudo(market_id)
                payload.append({
                    "id": market_id,
                    "promote": base > 0.45,
                    "quick_probability": round(0.15 + base * 0.7, 4),
                    "confidence": round(0.4 + base * 0.4, 4),
                    "tractability": round(0.3 + base * 0.6, 4),
                    "rationale": "stub screener output",
                })
            return json.dumps(payload), usage
        seed = self._pseudo(prompt)
        payload = {
            "probability": round(0.1 + seed * 0.8, 4),
            "confidence": round(0.5 + seed * 0.4, 4),
            "reasoning": "stub forecaster output for offline pipeline testing",
            "key_drivers": ["stub driver a", "stub driver b"],
            "resolution_risk": "stub resolution risk note",
        }
        return json.dumps(payload), usage


def build_backend(config):
    if config.llm_stub:
        LOG.warning("using stub LLM backend; no real analysis is being performed")
        return StubBackend(config)
    return AnthropicBackend(config)


def _market_brief(market, index=None):
    from .db import parse_iso, utc_now

    end = parse_iso(market.end_date)
    days = (end - utc_now()).total_seconds() / 86400.0 if end else None
    lines = [
        f'"id": "{market.condition_id}"',
        f"question: {market.question}",
        f"category: {market.category or 'uncategorized'}",
        f"yes_price: {market.yes_price}",
        f"no_price: {market.no_price}",
        f"liquidity_usd: {market.liquidity:.0f}",
        f"volume_24hr_usd: {market.volume_24hr:.0f}",
        f"days_to_resolution: {days:.1f}" if days is not None else "days_to_resolution: unknown",
        f"resolution_text: {coerce_str(market.description, 1800)}",
    ]
    header = f"--- MARKET {index} ---" if index is not None else "--- MARKET ---"
    return header + "\n" + "\n".join(lines)


def build_screen_prompt(markets):
    blocks = [_market_brief(market, index + 1) for index, market in enumerate(markets)]
    return (
        f"Screen the following {len(markets)} markets.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn the JSON array now."
    )


def build_forecast_prompt(market, flags, screening):
    flag_lines = []
    for flag in flags:
        flag_lines.append(f"- {flag.edge_type} (score {flag.score:.2f}): {flag.detail}")
    flag_text = "\n".join(flag_lines) if flag_lines else "- none"

    screen_text = "none"
    if screening:
        screen_text = (
            f"quick_probability={screening.get('quick_probability')}, "
            f"confidence={screening.get('confidence')}, "
            f"rationale={coerce_str(screening.get('rationale'), 300)}"
        )

    return (
        _market_brief(market)
        + "\n\n--- EDGE SCANNER FLAGS ---\n"
        + flag_text
        + "\n\n--- FIRST-STAGE SCREEN ---\n"
        + screen_text
        + "\n\nReturn the JSON object now."
    )


class TwoTierPipeline:
    def __init__(self, config, backend=None):
        self.config = config
        self.backend = backend or build_backend(config)

    def screen(self, markets, on_error=None):
        results = {}
        batch_size = max(1, self.config.llm_batch_size)
        for start in range(0, len(markets), batch_size):
            batch = markets[start:start + batch_size]
            prompt = build_screen_prompt(batch)
            try:
                text, usage = self.backend.complete(
                    self.config.screen_model,
                    SCREEN_SYSTEM,
                    prompt,
                    self.config.screen_max_tokens,
                )
            except LLMError as exc:
                if on_error:
                    on_error("screen", self.config.screen_model, None, "api_error", exc, None)
                continue

            try:
                parsed = extract_json(text, expect="array")
            except LLMParseError as exc:
                if on_error:
                    on_error("screen", self.config.screen_model, None, "parse_error", exc, text)
                continue

            if isinstance(parsed, dict):
                parsed = parsed.get("results") or parsed.get("markets") or [parsed]
            if not isinstance(parsed, list):
                if on_error:
                    on_error("screen", self.config.screen_model, None, "shape_error",
                             "expected array", text)
                continue

            by_id = {market.condition_id: market for market in batch}
            for position, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                market_id = coerce_str(item.get("id"), 200).strip()
                if market_id not in by_id:
                    if position < len(batch):
                        market_id = batch[position].condition_id
                    else:
                        continue
                try:
                    results[market_id] = {
                        "promote": coerce_bool(item.get("promote")),
                        "quick_probability": coerce_probability(
                            item.get("quick_probability"), 0.5
                        ),
                        "confidence": coerce_probability(item.get("confidence"), 0.3),
                        "tractability": coerce_probability(item.get("tractability"), 0.3),
                        "rationale": coerce_str(item.get("rationale"), 400),
                        "usage": usage,
                        "model": self.config.screen_model,
                    }
                except LLMParseError as exc:
                    if on_error:
                        on_error("screen", self.config.screen_model, market_id,
                                 "field_error", exc, text)
        return results

    def forecast(self, market, flags, screening, on_error=None):
        prompt = build_forecast_prompt(market, flags, screening)
        try:
            text, usage = self.backend.complete(
                self.config.forecast_model,
                FORECAST_SYSTEM,
                prompt,
                self.config.forecast_max_tokens,
            )
        except LLMError as exc:
            if on_error:
                on_error("forecast", self.config.forecast_model, market.condition_id,
                         "api_error", exc, None)
            return None

        try:
            parsed = extract_json(text, expect="object")
        except LLMParseError as exc:
            if on_error:
                on_error("forecast", self.config.forecast_model, market.condition_id,
                         "parse_error", exc, text)
            return None

        if isinstance(parsed, list):
            parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else None
        if not isinstance(parsed, dict):
            if on_error:
                on_error("forecast", self.config.forecast_model, market.condition_id,
                         "shape_error", "expected object", text)
            return None

        if "probability" not in parsed:
            for alias in ("yes_probability", "prob", "p"):
                if alias in parsed:
                    parsed["probability"] = parsed[alias]
                    break

        try:
            probability = coerce_probability(parsed.get("probability"))
        except LLMParseError as exc:
            if on_error:
                on_error("forecast", self.config.forecast_model, market.condition_id,
                         "field_error", exc, text)
            return None

        drivers = parsed.get("key_drivers") or []
        if isinstance(drivers, str):
            drivers = [drivers]
        drivers = [coerce_str(item, 200) for item in list(drivers)[:5]]

        return {
            "probability": probability,
            "confidence": coerce_probability(parsed.get("confidence"), 0.4),
            "reasoning": coerce_str(parsed.get("reasoning"), 4000),
            "key_drivers": drivers,
            "resolution_risk": coerce_str(parsed.get("resolution_risk"), 800),
            "usage": usage,
            "model": self.config.forecast_model,
        }
