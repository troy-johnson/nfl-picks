"""Fail when a completed game day lacks the data the scheduled workflows should save.

The capture and refresh jobs exit 0 when they skip work, so a missed window
leaves no failed run. This audit checks the saved files instead:

- odds: every game is in the day's odds snapshot, captured before kickoff.
- crowd: every regular-season game has an ESPN pick share captured before kickoff.
- recency: every regular-season game after the freeze has a candidate forecast saved before kickoff.
- refresh: `public/data/current.json` changed in the final window before the day's first regular-season kickoff.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict  # noqa: E402

ROOT = predict.ROOT
SPEC = ROOT / "docs" / "recency-candidate.json"


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def check_odds(day: str, games: pd.DataFrame, root: Path) -> list[str]:
    directory = root / "data" / "odds-snapshots" / str(games.season.iloc[0])
    paths = sorted((directory / day).glob("*.json"))
    legacy = directory / f"{day}.json"
    if legacy.exists():
        paths.append(legacy)
    if not paths:
        return [f"{day}: no odds snapshot"]
    saved = {}
    paired = set()
    version_two = False
    for path in paths:
        snapshot = read_json(path)
        captured = timestamp(snapshot["capturedAt"])
        finished = timestamp(snapshot.get("savedAt", snapshot["capturedAt"]))
        version_two = version_two or snapshot.get("formatVersion") == 2
        for game in snapshot["games"]:
            saved.setdefault(game["gameId"], []).append((captured, finished))
            published = game.get("published") or {}
            generated = timestamp(published["generatedAt"]) if published.get("generatedAt") else None
            if (published.get("marketCapturedAt") == snapshot["capturedAt"] and generated is not None
                    and captured <= generated <= finished and published.get("homeWinProbability") is not None
                    and published.get("statisticalHomeProbability") is not None):
                paired.add((game["gameId"], captured, finished))
    missing = []
    no_forecast = []
    for row in games.itertuples():
        opens = row.kickoff - timedelta(minutes=predict.CAPTURE_WINDOW_MAX_MINUTES)
        valid = [(captured, finished) for captured, finished in saved.get(row.game_id, []) if captured <= finished < row.kickoff and (not version_two or captured >= opens)]
        if not valid:
            missing.append(row.game_id)
        if version_two and row.game_type == "REG" and not any((row.game_id, captured, finished) in paired for captured, finished in valid):
            no_forecast.append(row.game_id)
    problems = [f"{day}: odds snapshot lacks {', '.join(missing)}"] if missing else []
    if no_forecast:
        problems.append(f"{day}: no fresh paired forecast for {', '.join(no_forecast)}")
    return problems


def check_crowd(day: str, games: pd.DataFrame, root: Path) -> list[str]:
    games = games[games.game_type == "REG"]
    if games.empty:
        return []
    season, week = int(games.season.iloc[0]), int(games.week.iloc[0])
    week_file = read_json(root / "data" / "crowd-picks" / str(season) / f"week-{week:02d}.json")
    if week_file is None:
        return [f"{day}: no crowd file for week {week}"]
    captured = {(game["awayTeam"], game["homeTeam"]): timestamp(game["capturedAt"]) for game in week_file["games"] if game.get("capturedAt")}
    missing = [row.game_id for row in games.itertuples() if not captured.get((row.away_team, row.home_team), row.kickoff) < row.kickoff]
    return [f"{day}: no pre-kickoff crowd share for {', '.join(missing)}"] if missing else []


def check_recency(day: str, games: pd.DataFrame, root: Path, frozen_at: datetime) -> list[str]:
    games = games[(games.game_type == "REG") & (games.kickoff > frozen_at)]
    if games.empty:
        return []
    archive = read_json(root / "data" / "model-predictions" / "recency" / str(games.season.iloc[0]) / f"{day}.json")
    if archive is None:
        return [f"{day}: no recency candidate file"]
    saved = timestamp(archive["savedAt"])
    forecast = {game["gameId"] for game in archive["games"]}
    missing = [row.game_id for row in games.itertuples() if row.game_id not in forecast or saved >= row.kickoff]
    return [f"{day}: recency candidate file lacks {', '.join(missing)}"] if missing else []


def check_refresh(day: str, games: pd.DataFrame, refreshes: list[datetime]) -> list[str]:
    games = games[games.game_type == "REG"]
    if games.empty:
        return []
    kickoff = min(games.kickoff)
    opens = kickoff - timedelta(minutes=predict.CAPTURE_WINDOW_MAX_MINUTES)
    if any(opens <= refreshed < kickoff for refreshed in refreshes):
        return []
    return [f"{day}: no pick refresh between {opens:%H:%M} and {kickoff:%H:%M} UTC"]


def audit(schedule: pd.DataFrame, season: int, now: datetime, refreshes: list[datetime], since: date | None = None, root: Path = ROOT) -> list[str]:
    """Problems for each game day of `season` whose last kickoff is before `now`."""
    frozen_at = timestamp(json.loads((root / "docs" / "recency-candidate.json").read_text())["frozenAt"])
    games = schedule[(schedule.season == season) & schedule.game_type.isin(["REG", "POST"])].copy()
    games["kickoff"] = games.apply(predict.kickoff_at, axis=1)
    problems = []
    for day, day_games in games.groupby(games.gameday.astype(str)):
        if max(day_games.kickoff) >= now or (since is not None and date.fromisoformat(day) < since):
            continue
        problems += check_odds(day, day_games, root)
        problems += check_crowd(day, day_games, root)
        problems += check_recency(day, day_games, root, frozen_at)
        problems += check_refresh(day, day_games, refreshes)
    return problems


def refresh_times(root: Path = ROOT) -> list[datetime]:
    """Commit times of every change to the published picks. Needs the full git history."""
    log = subprocess.run(["git", "log", "--format=%cI", "--", "public/data/current.json"], cwd=root, check=True, capture_output=True, text=True)
    return [timestamp(line) for line in log.stdout.split()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail when a completed game day lacks expected data")
    parser.add_argument("--days", type=int, help="Check only game days in the last N days (default: the whole season)")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    season = predict.current_nfl_season(now)
    since = None if args.days is None else (now - timedelta(days=args.days)).date()
    problems = audit(predict.load_schedule([season]), season, now, refresh_times(), since)
    for problem in problems:
        print(problem)
    if problems:
        raise SystemExit(f"{len(problems)} data problems found")
    print(f"No data problems found for {season}" + ("" if since is None else f" since {since}"))


if __name__ == "__main__":
    main()
