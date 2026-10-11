from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import recency_forward as forward


@pytest.fixture
def fixture():
    spec = {"candidate": "baseline_core_exp4", "frozenAt": "2026-10-10T00:00:00Z", "sourceHashes": {"predict.py": "frozen"}, "evaluation": {"minimumCompletedGames": 250}}
    games = [{"gameId": "g1", "kickoffAt": "2026-10-11T17:00:00Z", "marketHomeProbability": 0.5,
              "baseline": {"probability": 0.5, "blendedProbability": 0.5}, "candidate": {"probability": 0.8, "blendedProbability": 0.6}}]
    archive = {"specHash": "spec", "predictionSourceHashes": spec["sourceHashes"], "savedAt": "2026-10-11T16:00:00Z", "capturedAt": "2026-10-11T15:59:00Z", "games": games}
    schedule = pd.DataFrame([{"game_id": "g1", "season": 2026, "week": 5, "game_type": "REG", "home_score": 20, "away_score": 10, "gameday": "2026-10-11", "gametime": "13:00"}])
    return spec, archive, schedule


def test_archive_is_immutable_and_rejects_late_save(tmp_path, fixture):
    spec, archive, _ = fixture
    path = tmp_path / "capture.json"
    assert forward.save_capture(path, archive, spec)
    original = path.read_bytes()
    changed = deepcopy(archive)
    changed["games"][0]["candidate"]["probability"] = 0.1
    assert not forward.save_capture(path, changed, spec)
    assert path.read_bytes() == original
    archive["savedAt"] = archive["games"][0]["kickoffAt"]
    with pytest.raises(ValueError, match="before kickoff"):
        forward.save_capture(tmp_path / "late.json", archive, spec)


def test_evaluation_deduplicates_games_and_requires_minimum_sample(fixture):
    spec, archive, schedule = fixture
    result = forward.evaluate([archive, deepcopy(archive)], schedule, spec, "spec")
    assert result["games"] == 1
    assert result["metrics"]["candidate"]["brier"] == pytest.approx(0.04)
    assert result["metrics"]["baseline"]["brier"] == 0.25
    assert result["status"] == "insufficient games: 1 of 250"
    assert result["comparisons"]["candidate_vs_baseline"]["status"] == "insufficient week blocks"


@pytest.mark.parametrize("invalid", ["late", "before_freeze", "odds_after_save", "wrong_spec", "wrong_source", "tie", "unfinished", "invalid_probability", "earlier_kickoff"])
def test_evaluation_excludes_invalid_games(fixture, invalid):
    spec, archive, schedule = fixture
    if invalid == "late":
        archive["savedAt"] = "2026-10-11T17:00:00Z"
    elif invalid == "before_freeze":
        archive["savedAt"] = archive["capturedAt"] = "2026-10-09T16:00:00Z"
    elif invalid == "odds_after_save":
        archive["capturedAt"] = "2026-10-11T16:01:00Z"
    elif invalid == "wrong_spec":
        archive["specHash"] = "different"
    elif invalid == "wrong_source":
        archive["predictionSourceHashes"] = {}
    elif invalid == "tie":
        schedule.loc[0, "away_score"] = 20
    elif invalid == "unfinished":
        schedule.loc[0, "home_score"] = np.nan
    elif invalid == "invalid_probability":
        archive["games"][0]["candidate"]["probability"] = np.nan
    else:
        schedule.loc[0, "gametime"] = "11:00"
    assert forward.evaluate([archive], schedule, spec, "spec")["games"] == 0


def test_forecast_excludes_all_target_week_results_and_stats(monkeypatch):
    schedule = pd.DataFrame([{"game_id": f"g{week}", "season": 2026, "week": week, "game_type": "REG", "home_team": "A", "away_team": "B", "gameday": f"2026-09-{week:02d}", "gametime": "13:00", "home_score": 20, "away_score": 10, "home_rest": 7, "away_rest": 7, "location": "Home"} for week in range(1, 4)])
    stats = {(2026, week, team): {metric: float(week) if team == "A" else 0.0 for metric in forward.predict.METRICS} for week in range(1, 4) for team in ("A", "B")}
    fitted = []
    class Pipeline:
        def predict_proba(self, frame):
            return np.tile([0.4, 0.6], (len(frame), 1))
    def train(frame, season):
        fitted.append(frame.copy())
        return SimpleNamespace(pipeline=Pipeline(), market_weight=0.95, validation_season=2025, training_seasons=[2026])
    monkeypatch.setattr(forward.predict, "train_model", train)
    first = forward.forecast(schedule, stats, 2026, 3, {"g3"})
    assert all(frame.loc[frame.week == 3, "home_win"].isna().all() for frame in fitted)
    assert first["g3"]["baseline"]["features"]["diff_off_epa"] == 1.5
    schedule.loc[schedule.week == 3, "home_score"] = 0
    stats[(2026, 3, "A")] = {metric: 9999.0 for metric in forward.predict.METRICS}
    assert forward.forecast(schedule, stats, 2026, 3, {"g3"}) == first


def test_frozen_sources_still_match():
    assert forward.specification()["candidate"] == "baseline_core_exp4"


def test_missed_morning_capture_keeps_only_remaining_games():
    schedule = pd.DataFrame([
        {"game_id": "morning", "season": 2026, "game_type": "REG", "gameday": "2026-10-11", "gametime": "09:30"},
        {"game_id": "afternoon", "season": 2026, "game_type": "REG", "gameday": "2026-10-11", "gametime": "13:00"},
        {"game_id": "evening", "season": 2026, "game_type": "REG", "gameday": "2026-10-11", "gametime": "20:15"},
    ])
    now = forward.timestamp("2026-10-11T15:00:00Z")
    day, kickoff, games = forward.remaining_game_day(schedule, 2026, now)
    assert day == "2026-10-11"
    assert kickoff == forward.timestamp("2026-10-11T17:00:00Z")
    assert set(games.game_id) == {"afternoon", "evening"}
    assert forward.predict.odds_capture_due(now, kickoff)
