from mlpm.normalize.probability import (
    american_odds_to_implied_prob,
    cents_to_probability,
    devig_two_way,
    gap_to_bps,
)


def test_american_odds_to_implied_prob_positive() -> None:
    assert round(american_odds_to_implied_prob(150), 4) == 0.4


def test_american_odds_to_implied_prob_negative() -> None:
    assert round(american_odds_to_implied_prob(-150), 4) == 0.6


def test_devig_two_way() -> None:
    fair_a, fair_b = devig_two_way(0.55, 0.50)
    assert round(fair_a + fair_b, 5) == 1.0


def test_cents_to_probability_bounds() -> None:
    assert cents_to_probability(120) == 1.0
    assert cents_to_probability(-5) == 0.0


def test_gap_to_bps() -> None:
    assert gap_to_bps(0.53, 0.50) == 300
