from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "model"))
import crowd
import espn_pool

NOW = datetime(2026, 10, 22, 22, 45, tzinfo=timezone.utc)
GAME = {"gameId": "2026_07_NE_CHI", "awayTeam": "NE", "homeTeam": "CHI", "gameday": "2026-10-22", "gametime": "20:15", "pick": "CHI", "poolPick": "NE"}


def context():
    return {
        "config": {"season": 2026, "entryId": "own", "groupId": "group", "challengeKey": "nfl-pickem-2026"},
        "remainingGames": 179,
        "group": {
            "size": 9, "groupSettings": {"name": "Pool"},
            "challengeSettings": {"scoringFormatId": 1, "scoringSettings": {"propPointsByPeriod": {str(week): 10 for week in range(1, 19)}, "excludeLowScoringPeriodScore": True, "ignoreTiebreak": False}},
            "entries": [
                {"id": "own", "name": "Our entry", "picks": [], "score": {"rank": 9, "overallScore": 450, "scoreByPeriod": {"3": {"score": 80, "invalid": True}}}},
                {"id": "leader", "name": "Leader", "score": {"rank": 1, "overallScore": 490}},
            ],
        },
        "challenge": {"propositions": [{"id": "prop", "scoringPeriodId": 7, "lockDate": int((NOW + timedelta(minutes=90)).timestamp() * 1000), "possibleOutcomes": [{"id": "a", "subType": "AWAY", "abbrev": "NE"}, {"id": "h", "subType": "HOME", "abbrev": "CHI"}]}]},
    }


def test_group_uses_actual_rules_and_standings():
    result = espn_pool.group_summary(context(), NOW)
    assert result["entries"] == 9
    assert result["dropLowestWeek"] is True
    assert result["tiebreakers"] is True
    assert result["correctPicksBehind"] == 4
    assert result["droppedWeeks"] == [3]
    assert result["remainingGames"] == 179


def test_public_hidden_picks_are_unknown_not_missing():
    result = espn_pool.compare_picks(context(), [GAME], 7, NOW)
    assert result["counts"] == {"matched": 0, "different": 0, "missing": 0, "unavailable": 1}
    assert result["games"][0]["submittedPick"] is None


def test_public_visible_pick_is_compared_without_changing_it():
    source = context()
    source["group"]["entries"][0]["picks"] = [{"propositionId": "prop", "outcomesPicked": [{"outcomeId": "a"}], "lastUpdateInfo": {"date": int(NOW.timestamp() * 1000)}}]
    result = espn_pool.compare_picks(source, [GAME], 7, NOW)
    assert result["counts"]["different"] == 1
    assert result["games"][0]["submittedPick"] == "NE"
    assert result["games"][0]["recommendedPick"] == "CHI"
    assert source["group"]["entries"][0]["picks"][0]["outcomesPicked"][0]["outcomeId"] == "a"


def test_locked_absent_pick_is_missing_but_missing_proposition_is_unknown():
    source = context()
    result = espn_pool.compare_picks(source, [GAME], 7, NOW + timedelta(hours=2))
    assert result["counts"]["missing"] == 1
    source["challenge"]["propositions"] = []
    assert espn_pool.compare_picks(source, [GAME], 7, NOW)["counts"]["unavailable"] == 1


def test_complete_recent_entry_export_can_confirm_missing_open_picks(tmp_path):
    source = context()
    exported = {"entryId": "own", "season": 2026, "week": 7, "capturedAt": NOW.isoformat(), "complete": True, "picks": {}}
    path = tmp_path / "entry.json"
    path.write_text(json.dumps(exported))
    validated = espn_pool.entry_snapshot(path, source["config"], 7, NOW)
    assert espn_pool.compare_picks(source, [GAME], 7, NOW, validated)["counts"]["missing"] == 1
    exported["picks"][GAME["gameId"]] = "CHI"
    assert espn_pool.compare_picks(source, [GAME], 7, NOW, exported)["counts"]["matched"] == 1


@pytest.mark.parametrize("change", [{"entryId": "other"}, {"week": 6}, {"complete": False}, {"capturedAt": "2026-10-22T21:00:00Z"}, {"capturedAt": "2026-10-23T01:00:00Z"}])
def test_incorrect_or_stale_entry_exports_cannot_claim_visibility(tmp_path, change):
    exported = {"entryId": "own", "season": 2026, "week": 7, "capturedAt": NOW.isoformat(), "complete": True, "picks": {}, **change}
    path = tmp_path / "entry.json"
    path.write_text(json.dumps(exported))
    with pytest.raises(ValueError):
        espn_pool.entry_snapshot(path, context()["config"], 7, NOW)


def test_accuracy_mode_does_not_fade_a_favorite_because_the_entry_is_behind(monkeypatch):
    monkeypatch.delenv("ESPN_ENTRY_FILE", raising=False)
    payload = {"season": 2026, "week": 7, "games": [dict(GAME)]}
    result = espn_pool.attach(payload, context(), NOW)
    assert result["games"][0]["poolPick"] == "CHI"
    assert result["pool"]["correctPicksBehind"] == 4
    assert result["games"][0]["entryCheck"]["status"] == "unavailable"


def test_dropped_week_is_calculated_per_entry():
    weeks = np.array([1, 1, 2, 2, 3, 3])
    correct = np.array([[1, 1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 1]])
    assert crowd.pool_scores(correct, weeks, False).tolist() == [4, 5]
    assert crowd.pool_scores(correct, weeks, True).tolist() == [4, 4]
    assert crowd.pool_scores(correct[None, :, :], weeks, True).tolist() == [[4, 4]]
