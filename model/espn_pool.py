"""Read group rules and compare visible ESPN picks with current recommendations."""
from __future__ import annotations

import json
import os
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import crowd
import live_odds
import predict

CONFIG = predict.ROOT / "data" / "pool-config.json"


def load_context(season: int, week: int) -> dict[str, Any] | None:
    config = json.loads(CONFIG.read_text())
    if config["season"] != season:
        return None
    base = f"{crowd.CHALLENGES_URL}/{config['challengeKey']}"
    group = crowd.fetch_json(f"{base}/groups/{config['groupId']}")
    challenge = crowd.fetch_challenge_week(config["challengeKey"], week)
    return {"config": config, "group": group, "challenge": challenge}


def group_summary(context: dict[str, Any], now: datetime) -> dict[str, Any]:
    group, config = context["group"], context["config"]
    entries = group.get("entries", [])
    own = next((entry for entry in entries if entry["id"] == config["entryId"]), None)
    if own is None:
        raise ValueError("The configured entry is absent from the ESPN group")
    settings = group["challengeSettings"]
    scoring = settings["scoringSettings"]
    points = scoring.get("propPointsByPeriod", {})
    values = set(points.values())
    if settings.get("scoringFormatId") != 1 or len(values) != 1 or next(iter(values)) <= 0:
        raise ValueError("The group must use equal-point straight-up scoring")
    if scoring.get("jokerCount", 0) or scoring.get("perfectPropositionBonus", 0) or scoring.get("perfectScoringPeriodBonus", 0):
        raise ValueError("The group has unsupported bonus scoring")
    score = own["score"]
    leader = max(entry["score"]["overallScore"] for entry in entries)
    per_pick = next(iter(values))
    return {
        "status": "available", "checkedAt": live_odds.iso(now), "name": group["groupSettings"]["name"],
        "url": f"https://fantasy.espn.com/games/{config['challengeKey']}/group?id={config['groupId']}",
        "entryName": own["name"], "entries": group["size"], "rank": score["rank"],
        "points": score["overallScore"], "leaderPoints": leader,
        "correctPicksBehind": (leader - score["overallScore"]) / per_pick,
        "pointsPerPick": per_pick, "dropLowestWeek": bool(scoring.get("excludeLowScoringPeriodScore")),
        "tiebreakers": not scoring.get("ignoreTiebreak", False),
        "droppedWeeks": [int(week) for week, result in score.get("scoreByPeriod", {}).items() if result.get("invalid")],
        "remainingGames": context.get("remainingGames"),
        "strategy": "accuracy", "strategyReason": "Choose the higher win probability to maximize expected correct picks.",
    }


def entry_snapshot(path: Path, config: dict[str, Any], week: int, now: datetime) -> dict[str, Any]:
    """An explicit, complete export can confirm missing picks before kickoff."""
    snapshot = json.loads(path.read_text())
    if (snapshot.get("entryId"), snapshot.get("season"), snapshot.get("week")) != (config["entryId"], config["season"], week):
        raise ValueError("The entry export does not match this entry and week")
    captured = live_odds.timestamp(snapshot["capturedAt"])
    if not now - timedelta(minutes=30) <= captured <= now:
        raise ValueError("The entry export must be at most 30 minutes old")
    if snapshot.get("complete") is not True or not isinstance(snapshot.get("picks"), dict):
        raise ValueError("The entry export must declare a complete set of submitted picks")
    return snapshot


def compare_picks(context: dict[str, Any], games: list[dict[str, Any]], week: int, now: datetime, exported: dict[str, Any] | None = None) -> dict[str, Any]:
    """Absence in the public feed means unknown until the game locks."""
    config, group, challenge = context["config"], context["group"], context["challenge"]
    entry = next(item for item in group["entries"] if item["id"] == config["entryId"])
    submitted = {pick["propositionId"]: pick for pick in entry.get("picks", [])}
    propositions = {}
    for proposition in challenge.get("propositions", []):
        if proposition.get("scoringPeriodId") != week:
            continue
        sides = {outcome.get("subType", "").lower(): crowd.nflverse_team(outcome["abbrev"]) for outcome in proposition.get("possibleOutcomes", [])}
        if "home" in sides and "away" in sides:
            propositions[(sides["away"], sides["home"])] = proposition
    checks = []
    valid_ids = {f"{config['season']}_{week:02d}_{away}_{home}" for away, home in propositions}
    if exported is not None and set(exported["picks"]) - valid_ids:
        raise ValueError("The entry export contains games outside this ESPN week")
    for game in games:
        proposition = propositions.get((game["awayTeam"], game["homeTeam"]))
        team, updated, status = None, None, "unavailable"
        if proposition is not None:
            lock = datetime.fromtimestamp(proposition["lockDate"] / 1000, tz=now.tzinfo)
            pick = submitted.get(proposition["id"], {})
            choices = pick.get("outcomesPicked") or []
            by_id = {outcome["id"]: crowd.nflverse_team(outcome["abbrev"]) for outcome in proposition["possibleOutcomes"]}
            if len(choices) == 1:
                team = by_id.get(choices[0].get("outcomeId"))
                millis = (pick.get("lastUpdateInfo") or {}).get("date")
                updated = None if millis is None else live_odds.iso(datetime.fromtimestamp(millis / 1000, tz=now.tzinfo))
            if exported is not None and lock > now:
                team = exported["picks"].get(game["gameId"])
                updated = None
                if team is not None and team not in (game["awayTeam"], game["homeTeam"]):
                    raise ValueError(f"Invalid submitted team for {game['gameId']}")
                status = "missing" if team is None else "matched" if team == game["pick"] else "different"
            elif team is not None:
                status = "matched" if team == game["pick"] else "different"
            elif lock <= now:
                status = "missing"
        checks.append({
            "gameId": game["gameId"], "status": status, "submittedPick": team,
            "recommendedPick": game["pick"], "submittedAt": updated,
        })
    return {
        "checkedAt": exported["capturedAt"] if exported is not None else live_odds.iso(now), "source": "entry export" if exported is not None else "ESPN public feed",
        "visibilityNote": "ESPN can hide open picks. Unavailable does not mean missing.",
        "counts": {status: sum(check["status"] == status for check in checks) for status in ("matched", "different", "missing", "unavailable")},
        "games": checks,
    }


def attach(payload: dict[str, Any], context: dict[str, Any] | None, now: datetime) -> dict[str, Any]:
    """Publish group context and use standard picks for the accuracy objective."""
    for game in payload["games"]:
        if predict.kickoff_at(pd.Series(game)) > now:
            game["poolPick"] = game["pick"]
            game["poolPickReason"] = "accuracy: higher win probability"
    if context is None:
        payload["pool"] = {"status": "unavailable", "checkedAt": live_odds.iso(now), "strategy": "accuracy"}
        return payload
    summary = group_summary(context, now)
    export = None
    if os.environ.get("ESPN_ENTRY_FILE"):
        export = entry_snapshot(Path(os.environ["ESPN_ENTRY_FILE"]), context["config"], payload["week"], now)
    checks = compare_picks(context, payload["games"], payload["week"], now, export)
    by_game = {check["gameId"]: check for check in checks["games"]}
    for game in payload["games"]:
        game["entryCheck"] = by_game[game["gameId"]]
    payload["pool"] = {**summary, "pickCheck": {key: value for key, value in checks.items() if key != "games"}}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare submitted ESPN picks with published recommendations")
    parser.add_argument("--entry-file", type=Path, help="A complete, recent export of your own submitted picks")
    parser.add_argument("--picks", type=Path, default=predict.OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.picks.read_text())
    now = datetime.now(timezone.utc)
    context = load_context(payload["season"], payload["week"])
    if context is None:
        raise SystemExit("No ESPN group is configured for this season")
    exported = None if args.entry_file is None else entry_snapshot(args.entry_file, context["config"], payload["week"], now)
    report = {"group": group_summary(context, now), "recommendationsGeneratedAt": payload["generatedAt"],
              "pickCheck": compare_picks(context, payload["games"], payload["week"], now, exported)}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
