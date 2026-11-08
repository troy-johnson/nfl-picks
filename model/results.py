"""Refresh game scores and submitted-pick results without changing forecasts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import crowd
import espn_pool
import live_odds
import predict
from scheduled_runs import atomic_json

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


def attach(payload: dict[str, Any], scoreboard: dict[str, Any], now: datetime) -> dict[str, Any]:
    events = {}
    for event in scoreboard.get("events", []):
        if event.get("season", {}).get("year") != payload["season"] or event.get("week", {}).get("number") != payload["week"]:
            continue
        if event.get("season", {}).get("type") != 2:
            continue
        for competition in event.get("competitions", []):
            sides = {team["homeAway"]: team for team in competition.get("competitors", [])}
            if "home" in sides and "away" in sides:
                key = tuple(crowd.nflverse_team(sides[side]["team"]["abbreviation"]) for side in ("away", "home"))
                events[key] = (event, sides)
    updates = []
    for game in payload["games"]:
        match = events.get((game["awayTeam"], game["homeTeam"]))
        if match is None:
            raise ValueError(f"ESPN scoreboard is missing {game['gameId']}")
        event, sides = match
        state = event.get("status", {}).get("type", {})
        final = state.get("completed") is True and state.get("name", "").startswith("STATUS_FINAL")
        status = "final" if final else "in_progress" if state.get("state") == "in" else "scheduled" if state.get("name") == "STATUS_SCHEDULED" else "unknown"
        scores = {}
        for side in ("away", "home"):
            value = sides[side].get("score")
            scores[side] = int(value) if str(value).isdigit() and status in {"final", "in_progress"} else None
        if final and any(value is None for value in scores.values()):
            raise ValueError(f"ESPN final score is missing for {game['gameId']}")
        if game.get("result", {}).get("status") == "final" and not final:
            raise ValueError(f"ESPN final status regressed for {game['gameId']}")
        winner = None
        if final and scores["away"] != scores["home"]:
            winner = game["awayTeam"] if scores["away"] > scores["home"] else game["homeTeam"]
        check = game.get("entryCheck", {})
        submitted = check.get("submittedPick")
        if not final:
            outcome = "pending"
        elif submitted not in (game["awayTeam"], game["homeTeam"]):
            outcome = "missing" if check.get("status") == "missing" else "unavailable"
        else:
            outcome = "tie" if winner is None else "win" if submitted == winner else "loss"
        updates.append({
            "status": status, "detail": state.get("shortDetail") or state.get("description") or "Status unavailable",
            "awayScore": scores["away"], "homeScore": scores["home"], "winner": winner,
            "submittedPick": submitted, "pickResult": outcome, "source": "ESPN", "checkedAt": live_odds.iso(now),
        })
    for game, result in zip(payload["games"], updates):
        game["result"] = result
    payload["resultsUpdatedAt"] = live_odds.iso(now)
    return payload


def refresh(path: Path = predict.OUTPUT, now: datetime | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    now = now or datetime.now(timezone.utc)
    scoreboard = crowd.fetch_json(f"{SCOREBOARD_URL}?dates={payload['season']}&seasontype=2&week={payload['week']}")
    context = espn_pool.load_context(payload["season"], payload["week"])
    if context is not None:
        context["remainingGames"] = payload.get("pool", {}).get("remainingGames")
    payload = espn_pool.attach(payload, context, now)
    attach(payload, scoreboard, now)
    atomic_json(path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks", type=Path, default=predict.OUTPUT)
    args = parser.parse_args()
    payload = refresh(args.picks)
    for game in payload["games"]:
        result = game["result"]
        print(f"{game['awayTeam']} at {game['homeTeam']}: {result['detail']} · your pick {result['submittedPick'] or 'unavailable'} · {result['pickResult']}")


if __name__ == "__main__":
    main()
