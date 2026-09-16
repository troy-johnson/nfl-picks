from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib.util
import sys

import pandas as pd

MODEL_DIR = Path(__file__).parents[1] / "model"


def load_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


predict = load_module("predict")
crowd = load_module("crowd")

NOW = datetime(2026, 9, 17, 22, 0, tzinfo=timezone.utc)


def counter(count: int, percentage: float, format_id: int) -> dict:
    return {"count": count, "percentage": percentage, "scoringFormatId": format_id}


def outcome(outcome_id: str, abbrev: str, side: str, count: int, share: float, moneyline: str) -> dict:
    return {
        "id": outcome_id,
        "abbrev": abbrev,
        "subType": side,
        "mappings": [{"type": "BETTING_LINE", "value": moneyline}],
        "choiceCounters": [counter(1, 0.5, 3), counter(count, share, 1), counter(2, 0.4, 2)],
    }


def proposition(name: str, away: tuple, home: tuple, lock: datetime, correct: list[str] | None = None, week: int = 2) -> dict:
    return {
        "name": name,
        "scoringPeriodId": week,
        "spread": 2.5,
        "lockDate": int(lock.timestamp() * 1000),
        "correctOutcomes": correct or [],
        "mappings": [{"type": "EVENT_ID", "value": "401"}],
        "possibleOutcomes": [outcome("a1", away[0], "AWAY", away[1], away[2], "-135"), outcome("h1", home[0], "HOME", home[1], home[2], "+115")],
    }


def payload(now: datetime) -> dict:
    return {
        "propositions": [
            proposition("CAR @ ATL", ("CAR", 800, 0.8), ("ATL", 200, 0.2), now + timedelta(hours=1)),
            proposition("WSH @ LAR", ("WSH", 300, 0.3), ("LAR", 700, 0.7), now + timedelta(days=3), correct=["h1"]),
            proposition("BYE", ("KC", 1, 1.0), ("KC", 0, 0.0), now, week=3),
        ]
    }


def test_challenge_keys_cover_old_and_new_names():
    challenges = [
        {"key": "nfl-pigskin-pickem-2025"},
        {"key": "nfl-pickem-2026"},
        {"key": "nfl-pigskin-playoff-pickem-2021"},
        {"key": "nba-pickem-2026"},
    ]
    assert crowd.challenge_keys_by_season(challenges) == {2025: "nfl-pigskin-pickem-2025", 2026: "nfl-pickem-2026"}


def test_normalize_uses_straight_format_and_nflverse_codes():
    games = crowd.normalize_pick_shares(payload(NOW), 2026, 2, NOW)
    assert [(g["awayTeam"], g["homeTeam"]) for g in games] == [("CAR", "ATL"), ("WAS", "LA")]
    first = games[0]
    assert first["homeShare"] == 0.2 and first["awayShare"] == 0.8 and first["entries"] == 1000
    assert first["espnAwayMoneyline"] == "-135" and first["espnSpread"] == 2.5
    assert first["homeWin"] is None
    assert games[1]["homeWin"] is True
    assert first["lockAt"].endswith("Z")


def test_merge_week_keeps_last_pre_lock_shares_for_locked_games():
    first = crowd.normalize_pick_shares(payload(NOW), 2026, 2, NOW)
    later = NOW + timedelta(hours=2)
    fresh = crowd.normalize_pick_shares(payload(NOW), 2026, 2, later)
    for game in fresh:
        game["homeShare"], game["awayShare"] = 0.5, 0.5
    merged = {(g["awayTeam"], g["homeTeam"]): g for g in crowd.merge_week(first, fresh, later)}
    assert merged[("CAR", "ATL")]["homeShare"] == 0.2, "locked game keeps its earlier shares"
    assert merged[("WAS", "LA")]["homeShare"] == 0.5, "open game takes the fresh shares"


def test_merge_week_backfills_results_for_locked_games():
    first = crowd.normalize_pick_shares(payload(NOW), 2026, 2, NOW)
    fresh = crowd.normalize_pick_shares(payload(NOW), 2026, 2, NOW)
    fresh[0]["homeWin"] = False
    merged = {(g["awayTeam"], g["homeTeam"]): g for g in crowd.merge_week(first, fresh, NOW + timedelta(hours=2))}
    assert merged[("CAR", "ATL")]["homeWin"] is False


def test_capture_due_window():
    kickoff = NOW + timedelta(minutes=90)
    assert crowd.capture_due(NOW, kickoff)
    assert not crowd.capture_due(NOW - timedelta(minutes=60), kickoff)
    assert not crowd.capture_due(kickoff, kickoff)


def test_join_market_and_evaluate():
    crowd_frame = pd.DataFrame([
        {"season": 2025, "week": 1, "awayTeam": "A", "homeTeam": "B", "homeShare": 0.9, "awayShare": 0.1, "entries": 10},
        {"season": 2025, "week": 1, "awayTeam": "C", "homeTeam": "D", "homeShare": 0.3, "awayShare": 0.7, "entries": 10},
        {"season": 2025, "week": 1, "awayTeam": "E", "homeTeam": "F", "homeShare": 0.6, "awayShare": 0.4, "entries": 10},
    ])
    schedule = pd.DataFrame([
        {"game_id": "1", "season": 2025, "week": 1, "game_type": "REG", "away_team": "A", "home_team": "B", "home_score": 20, "away_score": 10, "home_moneyline": -150, "away_moneyline": 130, "spread_line": 3, "gameday": "2025-09-07"},
        {"game_id": "2", "season": 2025, "week": 1, "game_type": "REG", "away_team": "C", "home_team": "D", "home_score": 24, "away_score": 21, "home_moneyline": -120, "away_moneyline": 100, "spread_line": 1, "gameday": "2025-09-07"},
        {"game_id": "3", "season": 2025, "week": 1, "game_type": "REG", "away_team": "E", "home_team": "F", "home_score": None, "away_score": None, "home_moneyline": -110, "away_moneyline": -110, "spread_line": 0, "gameday": "2025-09-07"},
        {"game_id": "4", "season": 2025, "week": 1, "game_type": "POST", "away_team": "A", "home_team": "B", "home_score": 1, "away_score": 0, "home_moneyline": -110, "away_moneyline": -110, "spread_line": 0, "gameday": "2026-01-10"},
    ])
    joined = crowd.join_market(crowd_frame, schedule)
    assert len(joined) == 3
    result = crowd.evaluate(joined)
    assert result["games"] == 2, "unplayed game is excluded"
    assert result["marketFavoriteAccuracy"] == 1.0
    assert result["crowdFavoriteAccuracy"] == 0.5
    assert result["disagreements"] == 1
    assert sum(row["games"] for row in result["byMarketBucket"]) == 2
    assert result["bySeason"][2025]["games"] == 2


def test_weekly_report_ranks_by_contrarian_value():
    current = {"games": [
        {"gameId": "1", "awayTeam": "A", "homeTeam": "B", "pick": "B", "homeWinProbability": 0.62, "marketHomeProbability": 0.60},
        {"gameId": "2", "awayTeam": "C", "homeTeam": "D", "pick": "D", "homeWinProbability": 0.55, "marketHomeProbability": 0.52},
        {"gameId": "3", "awayTeam": "X", "homeTeam": "Y", "pick": "Y", "homeWinProbability": 0.7, "marketHomeProbability": 0.7},
    ]}
    shares = [
        {"awayTeam": "A", "homeTeam": "B", "homeShare": 0.85, "awayShare": 0.15, "entries": 100},
        {"awayTeam": "C", "homeTeam": "D", "homeShare": 0.58, "awayShare": 0.42, "entries": 100},
    ]
    rows = crowd.weekly_report(current, shares)
    assert [row["gameId"] for row in rows] == ["1", "2"], "games without crowd data are dropped; largest value first"
    top = rows[0]
    assert top["contrarianTeam"] == "A"
    assert abs(top["contrarianTeamMarketProbability"] - 0.40) < 1e-9
    assert abs(top["contrarianTeamCrowdShare"] - 0.15) < 1e-9
    assert abs(top["contrarianValue"] - 0.25) < 1e-9
    assert abs(top["expectedPointCost"] - 0.20) < 1e-9
    assert abs(rows[1]["expectedPointCost"] - 0.04) < 1e-9
    assert "A@B" in crowd.format_report(rows)
