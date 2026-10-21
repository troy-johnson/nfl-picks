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


def keep_started_games(previous: dict[str, Any] | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the published picks for games that started before this refresh.

    `predict.generate` publishes only games that have not started. Without this
    step, a Sunday refresh would remove the Thursday game from the week.
    """
    if not previous or (previous.get("season"), previous.get("week")) != (payload["season"], payload["week"]):
        return payload
    fresh = {game["gameId"] for game in payload["games"]}
    started = [game for game in previous.get("games", []) if game["gameId"] not in fresh]
    games = sorted(started + payload["games"], key=lambda game: (game["gameday"], game.get("gametime") or "", game["gameId"]))
    return {**payload, "firstGame": min((game["gameday"] for game in games), default=None), "games": games}


def refresh(force: bool, no_weather: bool, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    stage = preflight(force, now)
    previous = json.loads(predict.OUTPUT.read_text()) if predict.OUTPUT.exists() else None
    current_conditions = stage in {"final", "forced"}
    payload = keep_started_games(previous, predict.generate(current_conditions and not no_weather, current_conditions, now))
    predict.OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def capture_odds(now: datetime | None = None, output_dir: Path = predict.ODDS_SNAPSHOT_DIR, schedule: pd.DataFrame | None = None) -> dict[str, Any] | None:
    now = now or datetime.now(timezone.utc)
    season = predict.current_nfl_season(now)
    schedule = predict.load_schedule([season]) if schedule is None else schedule
    game_day, first_kickoff, games = remaining_game_day(schedule, season, now)
    if not predict.odds_capture_due(now, first_kickoff):
        print(f"Odds capture for {game_day} is not due yet")
        return None
    path = predict.odds_snapshot_path(season, game_day, output_dir)
    if path.exists():
        print(f"Odds snapshot already exists for {game_day}")
        return json.loads(path.read_text())
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set THE_ODDS_API_KEY before capturing odds")
    snapshot = predict.normalize_odds_snapshot(predict.fetch_current_odds(api_key), games, season, game_day, first_kickoff, now, predict.published_picks(season))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"Captured odds for {len(snapshot['games'])} games on {game_day}")
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
