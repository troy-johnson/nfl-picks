from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import context_experiments as context


def inputs():
    stadiums = pd.DataFrame([
        {"stadium_id": "WEST", "lat": 37.4, "lon": -122.0, "altitude": 5, "tz": "America/Los_Angeles", "roof_type": "Outdoors"},
        {"stadium_id": "EAST", "lat": 40.8, "lon": -74.0, "altitude": 5, "tz": "America/New_York", "roof_type": "Outdoors"},
    ])
    rows = []
    for game_id, day, home, away, stadium in [
        ("prior-west", "2022-09-01", "W", "E", "WEST"),
        ("prior-east", "2022-09-08", "E", "W", "EAST"),
        ("target", "2023-09-10", "E", "W", "EAST"),
        ("future", "2023-09-17", "W", "E", "WEST"),
    ]:
        rows.append({"game_id": game_id, "gameday": day, "season": int(day[:4]), "game_type": "REG", "gametime": "13:00", "home_team": home, "away_team": away, "stadium_id": stadium, "home_score": 20, "away_score": 10, "location": "Home", "home_rest": 7, "away_rest": 7, "surface": "grass", "div_game": 0, "roof": "outdoors", "wind": 5, "temp": 70, "home_qb_id": "recorded-home", "away_qb_id": "recorded-away"})
    features = pd.DataFrame([{"game_id": "target", "home_recent_qb_id": "old", "away_recent_qb_id": "young"}])
    players = pd.DataFrame([{"gsis_id": "old", "birth_date": "1985-09-10"}, {"gsis_id": "young", "birth_date": "2001-09-10"}])
    return features, pd.DataFrame(rows), stadiums, players


def test_target_outcomes_starters_and_future_games_cannot_change_features():
    features, schedule, stadiums, players = inputs()
    first, _ = context.build_context(features, schedule, stadiums, players)
    schedule.loc[schedule.game_id.isin(["target", "future"]), ["home_score", "away_score"]] = [0, 100]
    schedule.loc[schedule.game_id == "target", ["roof", "home_qb_id", "away_qb_id", "wind", "temp"]] = ["closed", "young", "old", 100, -20]
    schedule.loc[schedule.game_id == "future", "stadium_id"] = "EAST"
    second, _ = context.build_context(features, schedule, stadiums, players)
    columns = list(context.VENUE_FEATURES + context.TRAVEL_FEATURES + context.AGE_FEATURES)
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert first.home_qb_age.iloc[0] == pytest.approx(38, abs=0.01)
    assert first.away_qb_age.iloc[0] == pytest.approx(22, abs=0.01)
    assert first.road_streak_diff.iloc[0] == 0
    schedule.loc[schedule.game_id == "prior-west", "stadium_id"] = "EAST"
    third, _ = context.build_context(features, schedule, stadiums, players)
    assert third.away_travel_km.iloc[0] != first.away_travel_km.iloc[0]


def test_body_clock_handles_daylight_saving_and_invalid_source_alias():
    phoenix = {"lat": 33, "lon": -112, "tz": "America/Phoenix"}
    east = {"lat": 40, "lon": -74, "tz": "America/New_York"}
    summer = context.travel_context(phoenix, east, datetime(2023, 9, 10, 17, tzinfo=timezone.utc))
    winter = context.travel_context(phoenix, east, datetime(2023, 12, 10, 18, tzinfo=timezone.utc))
    assert summer[1:] == (3, 2)
    assert winter[1:] == (2, 1)
    munich = {"lat": 48, "lon": 11, "tz": "Europe/Munich"}
    assert np.isfinite(context.travel_context(east, munich, datetime(2023, 11, 5, 14, 30, tzinfo=timezone.utc))[0])


def test_candidate_features_exclude_recorded_weather_and_target_starters():
    forbidden = {"observed_wind", "observed_temp", "home_score", "away_score", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name"}
    for candidate in context.candidates():
        assert not forbidden.intersection(candidate.features)


def test_split_report_joins_prediction_and_feature_metadata():
    features, schedule, stadiums, players = inputs()
    features["season"], features["week"] = 2023, 1
    features["home_rest"], features["away_rest"] = 7, 7
    features["neutral_site"] = 0
    features, _ = context.build_context(features, schedule, stadiums, players)
    result = {"predictions": [{"gameId": "target", "season": 2023, "week": 1, "homeWin": 1, "marketHomeProbability": 0.6, "candidates": {"baseline": {"probability": 0.6, "blendedProbability": 0.6}}}]}
    rows = {row["split"]: row for row in context.splits(result, features)}
    assert rows["early_season"]["games"] == 1
    assert rows["low_confidence"]["blend"]["brier"] == pytest.approx(0.16)
