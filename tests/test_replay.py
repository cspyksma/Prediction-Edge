from __future__ import annotations

import pytest

from mlpm.historical.replay import _snapshot_policy_minutes


def test_snapshot_policy_minutes_supports_expected_policies() -> None:
    assert _snapshot_policy_minutes("last_pregame") == 0
    assert _snapshot_policy_minutes("t_minus_60m") == 60
    assert _snapshot_policy_minutes("t_minus_30m") == 30
    assert _snapshot_policy_minutes("t_minus_10m") == 10


def test_snapshot_policy_minutes_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported snapshot_policy"):
        _snapshot_policy_minutes("t_minus_5m")
