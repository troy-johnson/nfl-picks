from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import recency_experiments as recency


def history_fixture(weeks=13):
    features, stats = [], {}
    for week in range(1, weeks + 1):
        earlier = list(range(max(1, week - 8), week))
        control = float(np.mean(earlier)) if earlier else np.nan
        features.append({"game_id": f"2021_{week:02d}_B_A", "season": 2021, "week": week, "home_team": "A", "away_team": "B", **{f"diff_{metric}": control for metric in recency.predict.METRICS}})
        stats[(2021, week, "A")] = {metric: float(week) for metric in recency.predict.METRICS}
        stats[(2021, week, "B")] = {metric: 0.0 for metric in recency.predict.METRICS}
    return pd.DataFrame(features), stats


def test_windows_use_only_prior_games_and_reproduce_the_eight_game_control():
    features, stats = history_fixture()
    frame, difference = recency.build_recency(features, stats)
    last = frame.iloc[-1]
    assert difference == 0
    assert last.core4_diff_off_epa == 10.5
    assert last.core8_diff_off_epa == 8.5
    assert last.core12_diff_off_epa == 6.5
    stats[(2021, 13, "A")] = {metric: 9999.0 for metric in recency.predict.METRICS}
    changed, _ = recency.build_recency(features, stats)
    pd.testing.assert_frame_equal(frame, changed)


def test_exponential_weights_skip_missing_values_per_metric():
    history = [{metric: value for metric in recency.predict.METRICS} for value in (0.0, np.nan, 8.0)]
    result = recency.average(history, 24, 1)
    assert result["off_epa"] == pytest.approx(6.4)
    assert np.isnan(recency.average([], 8, None)["off_epa"])


def test_variants_replace_the_original_strength_columns():
    for candidate in recency.candidates()[2:]:
        assert not any(column.startswith("diff_") and column.removeprefix("diff_") in recency.predict.METRICS for column in candidate.features)
        rolling = [column for column in candidate.features if column.startswith(tuple(f"{name}_diff_" for name in recency.VARIANTS))]
        assert len(rolling) == len(recency.predict.METRICS)
        assert {"rest_diff", "neutral_site"} <= set(candidate.features)
