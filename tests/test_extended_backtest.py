from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import extended_backtest as backtest


def test_bootstrap_resamples_whole_weeks_with_unequal_game_counts():
    first = np.array([0.9, 0.9, 0.9, 0.1])
    second = np.full(4, 0.5)
    actual = np.ones(4)
    result = backtest.bootstrap_difference(first, second, actual, np.array([1, 1, 1, 2]))
    assert result["brier"]["difference"] == pytest.approx(-0.04)
    assert result["brier"]["ci95"] == pytest.approx([-0.24, 0.56])
    assert result["status"] == "no confirmed improvement"
    repeat = backtest.bootstrap_difference(first, second, actual, np.array([1, 1, 1, 2]))
    assert repeat == result


def test_identical_sources_do_not_produce_a_gain():
    result = backtest.bootstrap_difference(np.array([0.4, 0.8]), np.array([0.4, 0.8]), np.array([0, 1]), np.array([1, 2]))
    assert result["brier"] == {"difference": 0.0, "ci95": [0.0, 0.0]}
    assert result["logLoss"]["difference"] == pytest.approx(0, abs=1e-15)
    assert result["status"] == "no confirmed improvement"


def test_one_week_does_not_support_a_confidence_interval():
    result = backtest.bootstrap_difference(np.array([0.8, 0.8]), np.array([0.5, 0.5]), np.array([1, 1]), np.array([1, 1]))
    assert result["brier"]["ci95"] is None
    assert result["status"] == "insufficient week blocks"


def test_multiple_comparison_intervals_are_wider():
    first = np.linspace(0.1, 0.9, 20)
    result = backtest.bootstrap_difference(first, np.full(20, 0.5), np.ones(20), np.arange(20), family_size=6)
    for metric in ("brier", "logLoss"):
        ordinary = result[metric]["ci95"]
        adjusted = result[metric]["ciAdjusted"]
        assert adjusted[0] <= ordinary[0]
        assert adjusted[1] >= ordinary[1]
        assert result[metric]["adjustedConfidence"] == pytest.approx(1 - 0.05 / 6)
