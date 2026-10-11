from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import matchup_experiments as matchup


def fixture():
    features, stats, coaches = [], [], []
    for week in (1, 2, 3):
        game_id = f"2021_{week:02d}_B_A"
        features.append({"game_id": game_id, "season": 2021, "week": week, "home_team": "A", "away_team": "B", "home_score": 20, "away_score": 10})
        coaches.append({"game_id": game_id, "home_coach": "Coach A", "away_coach": "Coach B"})
        for team, value in [("A", 2), ("B", 1)]:
            stats.append({"game_id": game_id, "season": 2021, "week": week, "team": team, "season_type": "REG", "attempts": 30, "sacks_suffered": value, "carries": 20, "passing_epa": value * week, "rushing_epa": value, "passing_interceptions": value - 1, "fumbles_lost_total": 0})
    return pd.DataFrame(features), pd.DataFrame(coaches), pd.DataFrame(stats)


def test_target_week_stats_and_coaches_cannot_change_target_features():
    features, schedule, stats = fixture()
    original = matchup.build_features(features, schedule, stats)
    stats.loc[stats.week >= 2, ["passing_epa", "rushing_epa", "sacks_suffered"]] = [500, -500, 100]
    schedule.loc[1:, "home_coach"] = "New coach"
    features.loc[features.week >= 2, ["home_score", "away_score"]] = [0, 100]
    changed = matchup.build_features(features, schedule, stats)
    columns = [column for column in original if column.startswith(("team", "weighted", "offseason", "match_", "coach_"))]
    pd.testing.assert_frame_equal(original.loc[original.week == 2, columns], changed.loc[changed.week == 2, columns])
    assert original.loc[original.week == 3, "team8_pass_epa"].iloc[0] != changed.loc[changed.week == 3, "team8_pass_epa"].iloc[0]
    assert changed.loc[changed.week == 3, "coach_recent_change_diff"].iloc[0] == 1


def test_rates_weight_opportunities_and_reduce_previous_season_weight():
    history = []
    for season, numerator, denominator in [(2021, 1, 1), (2022, 27, 9)]:
        history.append({"season": season, **{f"{metric}_{part}": value for metric in matchup.METRICS for part, value in [("num", numerator), ("den", denominator)]}})
    ordinary = matchup.rate_features(history, 2022, 8, None, 1)
    weighted = matchup.rate_features(history, 2022, 8, 1, 1)
    offseason = matchup.rate_features(history, 2022, 8, 1, 0.25)
    assert ordinary["pass_epa"] == pytest.approx(2.8)
    assert weighted["pass_epa"] == pytest.approx(27.5 / 9.5)
    assert offseason["pass_epa"] == pytest.approx(27.125 / 9.125)
    assert np.isnan(matchup.rate_features([], 2022, 8, None, 1)["pass_epa"])


def test_stats_for_another_game_fail_instead_of_entering_history():
    features, schedule, stats = fixture()
    stats.loc[0, "game_id"] = "another-game"
    with pytest.raises(ValueError, match="do not match game"):
        matchup.build_features(features, schedule, stats)
