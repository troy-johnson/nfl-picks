from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import crowd
import espn_pool
import live_odds
import live_picks
import predict
import scheduled_runs
import snapshots

NOW = datetime(2026, 10, 22, 22, 45, tzinfo=timezone.utc)
GAME = {"game_id": "2026_07_NE_CHI", "season": 2026, "week": 7, "game_type": "REG", "gameday": "2026-10-22", "gametime": "20:15", "away_team": "NE", "home_team": "CHI", "home_score": np.nan, "away_score": np.nan}


def event():
    return {
        "id": "event", "home_team": "Chicago Bears", "away_team": "New England Patriots", "commence_time": "2026-10-23T00:15:00Z",
        "bookmakers": [{"key": key, "title": key, "last_update": "2026-10-22T22:40:00Z", "markets": [{"key": "h2h", "outcomes": [{"name": "Chicago Bears", "price": -150}, {"name": "New England Patriots", "price": 130}]}]} for key in ("a", "b")],
    }


def test_consensus_removes_each_books_margin_and_rejects_stale_books():
    source = event()
    stale = deepcopy(source["bookmakers"][0])
    stale.update(key="stale", last_update="2026-10-22T21:00:00Z")
    source["bookmakers"].append(stale)
    quote = live_odds.quotes([source], pd.DataFrame([GAME]), NOW)[GAME["game_id"]]
    assert quote["marketHomeProbability"] == pytest.approx(predict.devig_home_probability(-150, 130))
    assert [book["key"] for book in quote["bookmakers"]] == ["a", "b"]
    game = {"homeTeam": "CHI", "awayTeam": "NE", "statisticalHomeProbability": .4, "marketHomeProbability": .3}
    live_picks.apply_quote(game, quote, .95, NOW)
    assert game["pick"] == "CHI"
    assert game["homeWinProbability"] == round(.95 * quote["marketHomeProbability"] + .05 * .4, 4)
    assert game["statisticalHomeProbability"] == .4
    assert game["marketAgeMinutes"] == 5
    assert game["marketBookCount"] == 2


@pytest.mark.parametrize("problem", ["duplicate", "future", "old_market", "bad_price", "different_kickoff", "already_started"])
def test_unusable_odds_cannot_satisfy_final_refresh(problem):
    source = event()
    if problem == "duplicate":
        source["bookmakers"][1]["key"] = "a"
    elif problem == "future":
        source["bookmakers"][1]["last_update"] = "2026-10-22T23:00:00Z"
    elif problem == "old_market":
        source["bookmakers"][1]["markets"][0]["last_update"] = "2026-10-22T21:00:00Z"
    elif problem == "bad_price":
        source["bookmakers"][1]["markets"][0]["outcomes"][0]["price"] = float("nan")
    elif problem == "different_kickoff":
        source["commence_time"] = "2026-10-23T04:15:00Z"
    else:
        source["commence_time"] = live_odds.iso(NOW)
    games = pd.DataFrame([GAME])
    quotes = live_odds.quotes([source], games, NOW)
    assert quotes == {}
    with pytest.raises(RuntimeError, match="Fresh odds"):
        live_odds.require_due_quotes(games, quotes, NOW)


def test_missing_later_week_odds_are_explicit_fallbacks():
    later = {**GAME, "gameday": "2026-10-25"}
    live_odds.require_due_quotes(pd.DataFrame([later]), {}, NOW)
    game = {"homeTeam": "CHI", "awayTeam": "NE", "statisticalHomeProbability": .4, "marketHomeProbability": .6}
    live_picks.apply_quote(game, None, .95, NOW)
    assert game["marketSource"] == "nflverse fallback"
    assert game["marketAgeMinutes"] is None
    assert game["marketBookCount"] == 0


def test_current_season_pbp_failure_cannot_publish_prior_season_only():
    prior = {**GAME, "game_id": "prior", "week": 6, "home_score": 24, "away_score": 10}
    with pytest.raises(RuntimeError, match="prior"):
        live_picks.check_team_coverage(pd.DataFrame([prior, GAME]), {(2025, 18, "CHI"): {}}, 2026, 7)
    live_picks.check_team_coverage(pd.DataFrame([GAME]), {}, 2026, 7)
    live_picks.check_team_coverage(pd.DataFrame([prior, GAME]), {(2026, 6, "CHI"): {}, (2026, 6, "NE"): {}}, 2026, 7)


def mock_generation(monkeypatch, tmp_path):
    features = pd.DataFrame([{**GAME, **{name: 0 for name in predict.FEATURES}, "market_home_prob": .3}])
    model = SimpleNamespace(pipeline=SimpleNamespace(predict_proba=lambda values: np.array([[.6, .4]])), market_weight=.95, training_seasons=[2025, 2026], validation_season=2025, validation_accuracy=.6, validation_brier=.21, market_brier=.21)
    monkeypatch.setenv("THE_ODDS_API_KEY", "test")
    monkeypatch.delenv("ESPN_ENTRY_FILE", raising=False)
    monkeypatch.setattr(predict, "load_inputs", lambda seasons: (pd.DataFrame([GAME]), pd.DataFrame()))
    monkeypatch.setattr(predict, "aggregate_team_weeks", lambda pbp: {})
    monkeypatch.setattr(predict, "build_game_features", lambda schedule, weeks: features)
    monkeypatch.setattr(predict, "train_model", lambda features, season: model)
    monkeypatch.setattr(predict, "fetch_current_odds", lambda key: [event()])
    monkeypatch.setattr(predict, "news_for_teams", lambda teams, now: {})
    monkeypatch.setattr(crowd, "week_shares", lambda *args: {})
    monkeypatch.setattr(espn_pool, "load_context", lambda *args: None)
    monkeypatch.setattr(predict, "OUTPUT", tmp_path / "current.json")
    monkeypatch.setattr(predict, "SNAPSHOT_DIR", tmp_path / "initial")
    monkeypatch.setattr(predict, "ODDS_SNAPSHOT_DIR", tmp_path / "odds")
    monkeypatch.setattr(scheduled_runs, "preflight", lambda force, now: "final")


def test_refresh_publishes_consensus_and_archives_the_identical_inputs(monkeypatch, tmp_path):
    mock_generation(monkeypatch, tmp_path)
    payload = scheduled_runs.refresh(False, True, NOW)
    game = payload["games"][0]
    assert game["pick"] == "CHI"
    assert game["poolPick"] == game["pick"]
    assert game["statisticalHomeProbability"] == .4
    assert json.loads(predict.OUTPUT.read_text()) == payload
    captures = snapshots.load_snapshots(predict.ODDS_SNAPSHOT_DIR)
    assert len(captures) == 1
    archived = captures[0]["games"][0]
    assert archived["published"]["marketCapturedAt"] == captures[0]["capturedAt"]
    assert archived["published"]["homeWinProbability"] == game["homeWinProbability"]
    assert snapshots.snapshot_frame(captures).iloc[0][snapshots.PUBLISHED] == game["homeWinProbability"]
    assert json.loads((predict.SNAPSHOT_DIR / "2026-07-initial.json").read_text()) == payload


def test_failed_refresh_keeps_published_and_initial_files(monkeypatch, tmp_path):
    mock_generation(monkeypatch, tmp_path)
    scheduled_runs.refresh(False, True, NOW)
    previous = predict.OUTPUT.read_bytes()
    initial = (predict.SNAPSHOT_DIR / "2026-07-initial.json").read_bytes()
    monkeypatch.setattr(predict, "fetch_current_odds", lambda key: [])
    with pytest.raises(RuntimeError, match="Fresh odds"):
        scheduled_runs.refresh(False, True, NOW + timedelta(minutes=10))
    assert predict.OUTPUT.read_bytes() == previous
    assert (predict.SNAPSHOT_DIR / "2026-07-initial.json").read_bytes() == initial
    assert len(snapshots.load_snapshots(predict.ODDS_SNAPSHOT_DIR)) == 1


def test_new_capture_does_not_overwrite_earlier_archive_or_started_pick(monkeypatch, tmp_path):
    mock_generation(monkeypatch, tmp_path)
    first = scheduled_runs.refresh(False, True, NOW)
    old = {**first["games"][0], "gameId": "started", "gameday": "2026-10-21", "pick": "NE"}
    first["games"].append(old)
    predict.OUTPUT.write_text(json.dumps(first))
    first_path = next(predict.ODDS_SNAPSHOT_DIR.rglob("*.json"))
    original = first_path.read_bytes()
    second = scheduled_runs.refresh(False, True, NOW + timedelta(minutes=10))
    assert next(game for game in second["games"] if game["gameId"] == "started") == old
    assert first_path.read_bytes() == original
    assert len(snapshots.load_snapshots(predict.ODDS_SNAPSHOT_DIR)) == 2


def test_archive_refuses_save_at_kickoff(tmp_path):
    quotes = live_odds.quotes([event()], pd.DataFrame([GAME]), NOW)
    with pytest.raises(RuntimeError, match="after kickoff"):
        live_odds.archive(quotes, 2026, None, datetime(2026, 10, 23, 0, 15, tzinfo=timezone.utc), tmp_path)
    assert list(tmp_path.rglob("*.json")) == []
