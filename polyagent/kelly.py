import logging
import math
from dataclasses import dataclass

LOG = logging.getLogger("polyagent.kelly")


@dataclass
class SizingDecision:
    approved: bool
    side: str
    entry_price: float
    probability: float
    edge: float
    kelly_full: float
    kelly_scaled: float
    stake_usd: float
    shares: float
    expected_value_per_share: float
    reason: str


def choose_side(probability, market_yes_price):
    return "YES" if probability > market_yes_price else "NO"


def side_price(side, yes_price, no_price):
    if side == "YES":
        return yes_price
    if no_price is not None:
        return no_price
    return 1.0 - yes_price


def kelly_fraction_binary(probability, price):
    if price <= 0.0 or price >= 1.0:
        return 0.0
    b = (1.0 - price) / price
    q = 1.0 - probability
    fraction = (probability * b - q) / b
    if fraction != fraction or math.isinf(fraction):
        return 0.0
    return max(0.0, min(1.0, fraction))


def confidence_weight(confidence, floor=0.0):
    return max(floor, min(1.0, confidence))


def size_position(config, probability, confidence, yes_price, no_price,
                  yes_cost=None, no_cost=None, current_market_exposure=0.0,
                  open_positions=0, deployed_today=0.0):
    if yes_price is None or not 0.0 < yes_price < 1.0:
        return SizingDecision(False, "NONE", 0.0, probability, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              "invalid market price")


    side = choose_side(probability, yes_price)
    fair_price = side_price(side, yes_price, no_price)
    entry_price = side_price(
        side,
        yes_cost if yes_cost is not None else yes_price,
        no_cost if no_cost is not None else no_price,
    )
    if entry_price is None or not 0.0 < entry_price < 1.0:
        return SizingDecision(False, side, 0.0, probability, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              "invalid side price")


    win_probability = probability if side == "YES" else 1.0 - probability
    effective_price = min(entry_price + config.slippage_buffer, 0.999)
    raw_edge = win_probability - fair_price
    net_edge = win_probability - effective_price - config.fee_rate * (1.0 - effective_price)

    decision_base = dict(
        side=side,
        entry_price=effective_price,
        probability=win_probability,
        edge=raw_edge,
    )

    def reject(reason, kelly_full=0.0, kelly_scaled=0.0):
        return SizingDecision(
            approved=False, kelly_full=kelly_full, kelly_scaled=kelly_scaled,
            stake_usd=0.0, shares=0.0, expected_value_per_share=round(net_edge, 6),
            reason=reason, **decision_base,
        )

    if raw_edge < config.min_edge:
        return reject(f"edge {raw_edge:.4f} below minimum {config.min_edge:.4f}")

    if confidence < config.min_confidence:
        return reject(
            f"confidence {confidence:.3f} below minimum {config.min_confidence:.3f}"
        )

    if net_edge <= 0:
        return reject("edge does not survive fees and slippage")

    if open_positions >= config.max_open_positions:
        return reject(f"open position cap reached ({open_positions})")

    kelly_full = kelly_fraction_binary(win_probability, effective_price)
    kelly_scaled = kelly_full * config.kelly_fraction * confidence_weight(confidence)
    kelly_scaled = min(kelly_scaled, config.max_position_fraction)

    stake = kelly_scaled * config.bankroll
    stake = min(stake, config.max_position_usd)

    remaining_market = config.max_market_exposure_usd - current_market_exposure
    stake = min(stake, max(0.0, remaining_market))

    remaining_daily = config.max_daily_deployment_usd - deployed_today
    stake = min(stake, max(0.0, remaining_daily))

    if stake < config.min_position_usd:
        return reject(
            f"stake {stake:.2f} below minimum {config.min_position_usd:.2f} after risk caps",
            kelly_full=kelly_full, kelly_scaled=kelly_scaled,
        )

    shares = stake / effective_price

    return SizingDecision(
        approved=True,
        kelly_full=round(kelly_full, 6),
        kelly_scaled=round(kelly_scaled, 6),
        stake_usd=round(stake, 2),
        shares=round(shares, 4),
        expected_value_per_share=round(net_edge, 6),
        reason="approved",
        **decision_base,
    )
