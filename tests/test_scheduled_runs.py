from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import scheduled_runs as runs

predict = runs.predict

# Week 7 of 2026 has a 09:30 ET London game. 2026-10-25 is in EDT (UTC-4).
WEEK = pd.DataFrame([
    {"game_id": "thu", "season": 2026, "game_type": "REG", "week": 7, "gameday": "2026-10-22", "gametime": "20:15"},
    {"game_id": "london", "season": 2026, "game_type": "REG", "week": 7, "gameday": "2026-10-25", "gametime": "09:30"},
    {"game_id": "early", "season": 2026, "game_type": "REG", "week": 7, "gameday": "2026-10-25", "gametime": "13:00"},
    {"game_id": "late", "season": 2026, "game_type": "REG", "week": 7, "gameday": "2026-10-25", "gametime": "16:25"},
    {"game_id": "mon", "season": 2026, "game_type": "REG", "week": 7, "gameday": "2026-10-26", "gametime": "20:15"},
])
BEFORE_LONDON = datetime(2026, 10, 25, 11, 47, tzinfo=timezone.utc)  # 07:47 ET
AFTER_LONDON = datetime(2026, 10, 25, 14, 52, tzinfo=timezone.utc)  # 10:52 ET
BETWEEN_WINDOWS = datetime(2026, 10, 25, 16, 30, tzinfo=timezone.utc)  # 12:30 ET


def test_remaining_game_day_measures_from_next_unstarted_kickoff():
    day, kickoff, games = runs.remaining_game_day(WEEK, 2026, AFTER_LONDON)
    assert day == "2026-10-25"
    assert kickoff == datetime(2026, 10, 25, 17, tzinfo=timezone.utc)
    assert list(games.game_id) == ["early", "late"]
    _, old_kickoff, _ = predict.upcoming_game_day(WEEK, 2026, AFTER_LONDON)
    assert old_kickoff == datetime(2026, 10, 25, 13, 30, tzinfo=timezone.utc)


def test_remaining_game_day_includes_london_before_it_starts():
    _, kickoff, games = runs.remaining_game_day(WEEK, 2026, BEFORE_LONDON)
    assert kickoff == datetime(2026, 10, 25, 13, 30, tzinfo=timezone.utc)
    assert list(games.game_id) == ["london", "early", "late"]


@pytest.mark.parametrize("now, expected", [(BEFORE_LONDON, "final"), (AFTER_LONDON, "final")])
def test_preflight_runs_final_before_each_sunday_window(monkeypatch, now, expected):
    monkeypatch.setattr(predict, "current_nfl_season", lambda now=None: 2026)
    assert runs.preflight(False, now, WEEK) == expected
    _, week_kickoff = predict.upcoming_week_and_kickoff(WEEK, 2026, now)
    assert predict.refresh_stage(now, week_kickoff) is None


def test_preflight_exits_between_windows(monkeypatch):
    monkeypatch.setattr(predict, "current_nfl_season", lambda now=None: 2026)
    with pytest.raises(SystemExit):
        runs.preflight(False, BETWEEN_WINDOWS, WEEK)
    assert runs.preflight(True, BETWEEN_WINDOWS, WEEK) == "forced"


def test_capture_odds_saves_afternoon_games_after_london_kickoff(monkeypatch, tmp_path):
    calls = {}

    def normalize(events, games, season, game_day, first_kickoff, captured_at, published):
        calls.update(games=list(games.game_id), first_kickoff=first_kickoff)
        return {"games": [{"gameId": game_id} for game_id in games.game_id]}

    monkeypatch.setattr(predict, "current_nfl_season", lambda now=None: 2026)
    monkeypatch.setattr(predict, "fetch_current_odds", lambda api_key: [])
    monkeypatch.setattr(predict, "published_picks", lambda season: {})
    monkeypatch.setattr(predict, "normalize_odds_snapshot", normalize)
    monkeypatch.setenv("THE_ODDS_API_KEY", "test")
    assert runs.capture_odds(AFTER_LONDON, tmp_path, WEEK) is not None
    assert calls == {"games": ["early", "late"], "first_kickoff": datetime(2026, 10, 25, 17, tzinfo=timezone.utc)}
    assert (tmp_path / "2026" / "2026-10-25.json").exists()
    assert runs.capture_odds(BETWEEN_WINDOWS, tmp_path / "other", WEEK) is None


def game(game_id, gameday, gametime, pick="A"):
    return {"gameId": game_id, "gameday": gameday, "gametime": gametime, "pick": pick}


def test_keep_started_games_preserves_published_picks_in_kickoff_order():
    previous = {"season": 2026, "week": 8, "games": [game("thu", "2026-10-22", "20:15", "old"), game("london", "2026-10-25", "09:30", "old"), game("early", "2026-10-25", "13:00", "old")]}
    payload = {"season": 2026, "week": 8, "firstGame": "2026-10-25", "games": [game("early", "2026-10-25", "13:00", "new")]}
    merged = runs.keep_started_games(previous, payload)
    assert [(g["gameId"], g["pick"]) for g in merged["games"]] == [("thu", "old"), ("london", "old"), ("early", "new")]
    assert merged["firstGame"] == "2026-10-22"


def test_keep_started_games_ignores_another_week():
    payload = {"season": 2026, "week": 9, "firstGame": "2026-10-29", "games": [game("next", "2026-10-29", "20:15")]}
    assert runs.keep_started_games({"season": 2026, "week": 8, "games": [game("thu", "2026-10-22", "20:15")]}, payload) == payload
    assert runs.keep_started_games(None, payload) == payload
