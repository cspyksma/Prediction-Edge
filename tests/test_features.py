import pandas as pd

from mlpm.features.build_features import build_features


def test_build_features_uses_market_and_model_probabilities() -> None:
    frame = pd.DataFrame(
        [
            {
                "gap_bps": 250,
                "market_prob": 0.55,
                "model_prob": 0.525,
            }
        ]
    )
    result = build_features(frame)
    assert result.iloc[0]["abs_gap_bps"] == 250
    assert round(result.iloc[0]["market_prob_centered"], 3) == 0.05
    assert round(result.iloc[0]["model_prob_centered"], 3) == 0.025
