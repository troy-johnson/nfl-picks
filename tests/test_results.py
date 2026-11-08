from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import results

NOW = datetime(2026, 11, 8, 18, 10, tzinfo=timezone.utc)


def payload():
    return {"season": 2026, "week": 9, "generatedAt": "2026-11-05T12:00:00Z", "games": [{
        "gameId": "2026_09_LA_WAS", "awayTeam": "LA", "homeTeam": "WAS",
        "pick": "LA", "winProbability": .65, "generatedAt": "2026-11-05T12:00:00Z",
        "entryCheck": {"submittedPick": "WAS", "status": "different"},
    }]}


def scoreboard(name="STATUS_FINAL", completed=True, away="17", home="24"):
    return {"events": [{
        "season": {"year": 2026, "type": 2}, "week": {"number": 9},
        "status": {"type": {"name": name, "state": "in" if name == "STATUS_IN_PROGRESS" else "pre" if name in {"STATUS_SCHEDULED", "STATUS_POSTPONED"} else "post", "completed": completed, "shortDetail": "Final"}},
        "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"abbreviation": "WSH"}, "score": home},
            {"homeAway": "away", "team": {"abbreviation": "LAR"}, "score": away},
        ]}],
    }]}


@pytest.mark.parametrize("away,home,outcome,winner", [("17", "24", "win", "WAS"), ("24", "17", "loss", "LA"), ("17", "17", "tie", None), ("0", "7", "win", "WAS")])
def test_results_grade_actual_pick_and_preserve_forecast(away, home, outcome, winner):
    data = payload()
    original = deepcopy(data)
    results.attach(data, scoreboard(away=away, home=home), NOW)
    game = data["games"][0]
    assert game["result"]["pickResult"] == outcome
    assert game["result"]["winner"] == winner
    assert game["result"]["submittedPick"] == "WAS"
    assert game["result"]["awayScore"] == int(away)
    assert {key: value for key, value in game.items() if key != "result"} == original["games"][0]
    assert data["generatedAt"] == original["generatedAt"]


@pytest.mark.parametrize("name,completed,status", [("STATUS_IN_PROGRESS", False, "in_progress"), ("STATUS_SCHEDULED", False, "scheduled"), ("STATUS_POSTPONED", False, "unknown"), ("STATUS_CANCELED", True, "unknown"), ("STATUS_FINAL", False, "unknown")])
def test_nonfinal_games_do_not_score_picks(name, completed, status):
    data = results.attach(payload(), scoreboard(name, completed), NOW)
    result = data["games"][0]["result"]
    assert result["status"] == status
    assert result["pickResult"] == "pending"
    assert result["winner"] is None
    if status in {"scheduled", "unknown"}:
        assert result["homeScore"] is None


@pytest.mark.parametrize("check,outcome", [({}, "unavailable"), ({"submittedPick": None, "status": "unavailable"}, "unavailable"), ({"submittedPick": None, "status": "missing"}, "missing")])
def test_final_without_verified_pick_does_not_grade_recommendation(check, outcome):
    data = payload()
    data["games"][0]["entryCheck"] = check
    results.attach(data, scoreboard(), NOW)
    assert data["games"][0]["result"]["pickResult"] == outcome


@pytest.mark.parametrize("change", [{"season": {"year": 2025, "type": 2}}, {"week": {"number": 8}}, {"season": {"year": 2026, "type": 3}}])
def test_other_weeks_and_seasons_cannot_supply_results(change):
    source = scoreboard()
    source["events"][0].update(change)
    with pytest.raises(ValueError, match="missing"):
        results.attach(payload(), source, NOW)


def test_incomplete_feed_and_regressed_final_leave_results_unchanged():
    data = results.attach(payload(), scoreboard(), NOW)
    original = deepcopy(data)
    with pytest.raises(ValueError, match="regressed"):
        results.attach(data, scoreboard("STATUS_IN_PROGRESS", False), NOW)
    assert data == original
    with pytest.raises(ValueError, match="score is missing"):
        results.attach(data, scoreboard(home=None), NOW)
    assert data == original


def test_refresh_failure_leaves_published_file_unchanged(monkeypatch, tmp_path):
    path = tmp_path / "current.json"
    path.write_text(json.dumps(payload()))
    original = path.read_bytes()
    monkeypatch.setattr(results.crowd, "fetch_json", lambda url: {"events": []})
    monkeypatch.setattr(results.espn_pool, "load_context", lambda season, week: None)
    monkeypatch.setattr(results.espn_pool, "attach", lambda data, context, now: data)
    with pytest.raises(ValueError, match="missing"):
        results.refresh(path, NOW)
    assert path.read_bytes() == original


def test_result_refresh_works_after_last_kickoff_without_generating_forecasts(monkeypatch, tmp_path):
    path = tmp_path / "current.json"
    path.write_text(json.dumps(payload()))
    monkeypatch.setattr(results.crowd, "fetch_json", lambda url: scoreboard())
    monkeypatch.setattr(results.espn_pool, "load_context", lambda season, week: None)
    monkeypatch.setattr(results.espn_pool, "attach", lambda data, context, now: data)
    updated = results.refresh(path, NOW)
    assert json.loads(path.read_text()) == updated
    assert updated["games"][0]["result"]["pickResult"] == "win"
    assert updated["generatedAt"] == payload()["generatedAt"]
