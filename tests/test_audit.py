from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import audit

# Sunday 2026-10-25 is in EDT (UTC-4): 09:30 ET is 13:30 UTC and 13:00 ET is 17:00 UTC.
SCHEDULE = pd.DataFrame([
    {"game_id": "london", "season": 2026, "game_type": "REG", "week": 7, "gameday": "2026-10-25", "gametime": "09:30", "away_team": "JAX", "home_team": "LA"},
    {"game_id": "early", "season": 2026, "game_type": "REG", "week": 7, "gameday": "2026-10-25", "gametime": "13:00", "away_team": "NE", "home_team": "CHI"},
    {"game_id": "next", "season": 2026, "game_type": "REG", "week": 8, "gameday": "2026-11-01", "gametime": "13:00", "away_team": "BUF", "home_team": "MIA"},
])
AFTER = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
REFRESH = [datetime(2026, 10, 25, 11, 50, tzinfo=timezone.utc)]


def write(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def complete_day(root):
    write(root, "docs/recency-candidate.json", {"frozenAt": "2026-10-10T23:28:24-06:00"})
    games = [{"gameId": "london"}, {"gameId": "early"}]
    write(root, "data/odds-snapshots/2026/2026-10-25.json", {"capturedAt": "2026-10-25T11:47:00Z", "games": games})
    write(root, "data/model-predictions/recency/2026/2026-10-25.json", {"savedAt": "2026-10-25T11:48:00+00:00", "games": games})
    write(root, "data/crowd-picks/2026/week-07.json", {"games": [
        {"awayTeam": "JAX", "homeTeam": "LA", "capturedAt": "2026-10-25T11:49:00Z"},
        {"awayTeam": "NE", "homeTeam": "CHI", "capturedAt": "2026-10-25T16:30:00Z"},
    ]})


def test_complete_day_passes_and_unfinished_days_are_skipped(tmp_path):
    complete_day(tmp_path)
    assert audit.audit(SCHEDULE, 2026, AFTER, REFRESH, root=tmp_path) == []


def test_missing_files_fail(tmp_path):
    write(tmp_path, "docs/recency-candidate.json", {"frozenAt": "2026-10-10T23:28:24-06:00"})
    assert audit.audit(SCHEDULE, 2026, AFTER, [], root=tmp_path) == [
        "2026-10-25: no odds snapshot",
        "2026-10-25: no crowd file for week 7",
        "2026-10-25: no recency candidate file",
        "2026-10-25: no pick refresh between 10:30 and 13:30 UTC",
    ]


def test_capture_after_the_london_kickoff_fails_for_the_london_game(tmp_path):
    complete_day(tmp_path)
    late = {"capturedAt": "2026-10-25T14:50:00Z", "savedAt": "2026-10-25T14:50:00+00:00", "games": [{"gameId": "london"}, {"gameId": "early"}]}
    write(tmp_path, "data/odds-snapshots/2026/2026-10-25.json", late)
    write(tmp_path, "data/model-predictions/recency/2026/2026-10-25.json", {**late, "games": [{"gameId": "early"}]})
    write(tmp_path, "data/crowd-picks/2026/week-07.json", {"games": [{"awayTeam": "NE", "homeTeam": "CHI", "capturedAt": "2026-10-25T16:30:00Z"}]})
    refresh_after_london = [datetime(2026, 10, 25, 14, 52, tzinfo=timezone.utc)]
    assert audit.audit(SCHEDULE, 2026, AFTER, refresh_after_london, root=tmp_path) == [
        "2026-10-25: odds snapshot lacks london",
        "2026-10-25: no pre-kickoff crowd share for london",
        "2026-10-25: recency candidate file lacks london",
        "2026-10-25: no pick refresh between 10:30 and 13:30 UTC",
    ]


def test_since_skips_earlier_days_and_frozen_period_skips_recency(tmp_path):
    complete_day(tmp_path)
    write(tmp_path, "docs/recency-candidate.json", {"frozenAt": "2026-10-30T00:00:00Z"})
    (tmp_path / "data/model-predictions/recency/2026/2026-10-25.json").unlink()
    assert audit.audit(SCHEDULE, 2026, AFTER, REFRESH, root=tmp_path) == []
    assert audit.audit(SCHEDULE, 2026, AFTER, [], since=date(2026, 10, 26), root=tmp_path) == []


def test_playoff_days_check_only_odds(tmp_path):
    write(tmp_path, "docs/recency-candidate.json", {"frozenAt": "2026-10-10T23:28:24-06:00"})
    playoff = pd.DataFrame([{"game_id": "sb", "season": 2026, "game_type": "POST", "week": 22, "gameday": "2027-02-07", "gametime": "18:30", "away_team": "BUF", "home_team": "SF"}])
    assert audit.audit(playoff, 2026, datetime(2027, 2, 8, tzinfo=timezone.utc), [], root=tmp_path) == ["2027-02-07: no odds snapshot"]
