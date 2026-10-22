"""Timing and entry point for the scheduled refresh and odds-capture workflows.

`predict.py` is a frozen source of the recency forward test
(`docs/recency-candidate.json`). A change to it stops the candidate capture.
Its timing helpers measure windows from the first kickoff of a game day or
week, even after that kickoff has passed. This module measures each window
from the next kickoff that has not started, and leaves `predict.py` unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict  # noqa: E402


def remaining_game_day(schedule: pd.DataFrame, season: int, now: datetime, game_types: tuple[str, ...] = ("REG", "POST")) -> tuple[str, datetime, pd.DataFrame]:
    """Next game day, its earliest kickoff after `now`, and its games that have not started."""
    games = schedule[(schedule.season == season) & schedule.game_type.isin(game_types)].copy()
    games["_kickoff"] = games.apply(predict.kickoff_at, axis=1)
    upcoming = games[games._kickoff > now]
    if upcoming.empty:
        raise RuntimeError(f"No upcoming games found for {season}")
    game_day = str(upcoming.sort_values("_kickoff").iloc[0].gameday)
    day_games = upcoming[upcoming.gameday.astype(str) == game_day].copy()
    return game_day, min(day_games._kickoff), day_games


def preflight(force: bool, now: datetime, schedule: pd.DataFrame | None = None) -> str:
    season = predict.current_nfl_season(now)
    schedule = predict.load_schedule([season]) if schedule is None else schedule
    week, _ = predict.upcoming_week_and_kickoff(schedule, season, now)
    if force:
        return "forced"
    game_day, kickoff, _ = remaining_game_day(schedule, season, now, ("REG",))
    stage = predict.refresh_stage(now, kickoff)
    if stage is None:
        print(f"Week {week} refresh is not due yet for {game_day}; leaving published picks unchanged")
        raise SystemExit(0)
    return stage


def keep_started_games(previous: dict[str, Any] | None, payload: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Keep the published picks for games that started before this refresh.

    The generator returns only games that have not started. Without this
    step, a Sunday refresh would remove the Thursday game from the week.
    """
    if not previous or (previous.get("season"), previous.get("week")) != (payload["season"], payload["week"]):
        return payload
    fresh = {game["gameId"] for game in payload["games"]}
    now = now or datetime.now(timezone.utc)
    started = [game for game in previous.get("games", []) if game["gameId"] not in fresh and predict.kickoff_at(pd.Series(game)) <= now]
    games = sorted(started + payload["games"], key=lambda game: (game["gameday"], game.get("gametime") or "", game["gameId"]))
    return {**payload, "firstGame": min((game["gameday"] for game in games), default=None), "games": games}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def refresh(force: bool, no_weather: bool, now: datetime | None = None) -> dict[str, Any]:
    import espn_pool
    import live_odds
    import live_picks

    stage = preflight(force, now or datetime.now(timezone.utc))
    previous = json.loads(predict.OUTPUT.read_text()) if predict.OUTPUT.exists() else None
    current_conditions = stage in {"final", "forced"}
    payload, quotes, context = live_picks.generate(current_conditions and not no_weather, current_conditions, now)
    if previous and (previous.get("season"), previous.get("week")) == (payload["season"], payload["week"]):
        old_games = {game["gameId"]: game for game in previous.get("games", [])}
        for game in payload["games"]:
            old = old_games.get(game["gameId"], {})
            if old.get("pick") and old["pick"] != game["pick"]:
                game["previousPick"] = old["pick"]
                game["recommendationChangedAt"] = payload["generatedAt"]
            elif old.get("previousPick"):
                game["previousPick"] = old["previousPick"]
                game["recommendationChangedAt"] = old["recommendationChangedAt"]
            if game.get("previousPick"):
                game["flags"].append(f"Recommendation changed: {game['previousPick']} → {game['pick']}")
    saved_at = now or datetime.now(timezone.utc)
    payload = keep_started_games(previous, payload, saved_at)
    payload = espn_pool.attach(payload, context, saved_at)
    saved_at = now or datetime.now(timezone.utc)
    live_odds.archive(quotes, payload["season"], payload, saved_at, predict.ODDS_SNAPSHOT_DIR)
    snapshot_path, baseline = predict.baseline_for(payload["season"], payload["week"])
    if baseline is None and not snapshot_path.exists():
        atomic_json(snapshot_path, payload)
    atomic_json(predict.OUTPUT, payload)
    return payload


def capture_odds(now: datetime | None = None, output_dir: Path = predict.ODDS_SNAPSHOT_DIR, schedule: pd.DataFrame | None = None) -> dict[str, Any] | None:
    import live_odds

    started = now or datetime.now(timezone.utc)
    season = predict.current_nfl_season(started)
    schedule = predict.load_schedule([season]) if schedule is None else schedule
    game_day, first_kickoff, games = remaining_game_day(schedule, season, started)
    if not predict.odds_capture_due(started, first_kickoff):
        print(f"Odds capture for {game_day} is not due yet")
        return None
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set THE_ODDS_API_KEY before capturing odds")
    events = predict.fetch_current_odds(api_key)
    captured = now or datetime.now(timezone.utc)
    if any(predict.kickoff_at(row) <= captured for _, row in games.iterrows()):
        raise RuntimeError("Odds capture crossed a kickoff")
    quotes = live_odds.quotes(events, games, captured)
    live_odds.require_due_quotes(games, quotes, captured)
    if not quotes:
        raise RuntimeError("The odds feed has no fresh multi-book prices")
    paths = live_odds.archive(quotes, season, None, now or datetime.now(timezone.utc), output_dir)
    snapshot = json.loads(paths[0].read_text())
    print(f"Captured odds for {len(quotes)} games on {game_day}")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled pick refresh and odds capture")
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--capture-odds", action="store_true", help="Save the multi-book moneyline snapshot when a game-day capture is due")
    args = parser.parse_args()
    if args.capture_odds:
        capture_odds()
        return
    payload = refresh(args.force, args.no_weather)
    print(f"Generated {len(payload['games'])} picks for {payload['season']} week {payload['week']}")
    print(f"Market weight: {payload['model']['marketWeight']:.0%}")


if __name__ == "__main__":
    main()
