from __future__ import annotations


def american_odds_to_implied_prob(odds: int | float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Total implied probability must be positive.")
    return prob_a / total, prob_b / total


def cents_to_probability(cents: int | float | None) -> float | None:
    if cents is None:
        return None
    return max(0.0, min(float(cents) / 100.0, 1.0))


def midpoint_probability(
    yes_bid: int | float | None,
    yes_ask: int | float | None,
    last_price: int | float | None,
) -> float | None:
    bid_prob = cents_to_probability(yes_bid)
    ask_prob = cents_to_probability(yes_ask)
    if bid_prob is not None and ask_prob is not None:
        return (bid_prob + ask_prob) / 2.0
    return cents_to_probability(last_price)


def gap_to_bps(market_prob: float, model_prob: float) -> int:
    return int(round((market_prob - model_prob) * 10_000))
