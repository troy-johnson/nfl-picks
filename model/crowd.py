"""Public pick shares from ESPN's NFL Pick'em game.

ESPN publishes, for every game, the share of public entries that picked each
team. This module archives those shares, keeps a live weekly file updated until
each game locks, and evaluates the crowd against the recorded market.

The crowd share is not a win probability. It measures where the pick'em field
stands, which is what matters when the goal is to finish ahead of a league
rather than to beat the market. Nothing here changes live picks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict  # noqa: E402

ROOT = predict.ROOT
CROWD_DIR = ROOT / "data" / "crowd-picks"
CHALLENGES_URL = "https://gambit-api.fantasy.espn.com/apis/v1/challenges"
CHALLENGE_KEY_PATTERN = re.compile(r"^nfl-(?:pigskin-)?pickem-(\d{4})$")
STRAIGHT_FORMAT_ID = 1
ESPN_TEAM_ALIASES = {"WSH": "WAS", "LAR": "LA"}
CAPTURE_WINDOW_MINUTES = 120
REGULAR_SEASON_WEEKS = 18


def nflverse_team(abbrev: str) -> str:
    return ESPN_TEAM_ALIASES.get(abbrev, abbrev)


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-picks/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode())


def challenge_keys_by_season(challenges: list[dict[str, Any]]) -> dict[int, str]:
    """Map each season to its ESPN challenge key. ESPN renamed the game in 2026."""
    keys: dict[int, str] = {}
    for challenge in challenges:
        match = CHALLENGE_KEY_PATTERN.match(str(challenge.get("key", "")))
        if match:
            keys[int(match.group(1))] = str(challenge["key"])
    return keys


def challenge_key_for_season(season: int) -> str:
    keys = challenge_keys_by_season(fetch_json(CHALLENGES_URL))
    if season not in keys:
        raise RuntimeError(f"No ESPN pick'em challenge found for {season}")
    return keys[season]


def fetch_challenge_week(key: str, week: int) -> dict[str, Any]:
    return fetch_json(f"{CHALLENGES_URL}/{key}?scoringPeriodId={week}&view=picks")


def _mapping(items: list[dict[str, Any]] | None, kind: str) -> str | None:
    for item in items or []:
        if item.get("type") == kind:
            return item.get("value")
    return None


def _iso(milliseconds: Any) -> str | None:
    if milliseconds is None:
        return None
    return datetime.fromtimestamp(int(milliseconds) / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_pick_shares(payload: dict[str, Any], season: int, week: int, captured_at: datetime) -> list[dict[str, Any]]:
    """Reduce one ESPN week payload to one record per game with straight-up shares."""
    games = []
    for proposition in payload.get("propositions", []):
        if proposition.get("scoringPeriodId") not in (None, week):
            continue
        sides: dict[str, dict[str, Any]] = {}
        for outcome in proposition.get("possibleOutcomes", []):
            side = str(outcome.get("subType", "")).lower()
            if side not in ("home", "away"):
                continue
            counter = next((item for item in outcome.get("choiceCounters", []) if item.get("scoringFormatId") == STRAIGHT_FORMAT_ID), None)
            if counter is None:
                continue
            sides[side] = {
                "team": nflverse_team(str(outcome.get("abbrev"))),
                "count": int(counter.get("count", 0)),
                "share": float(counter.get("percentage", 0.0)),
                "won": outcome.get("id") in (proposition.get("correctOutcomes") or []),
                "espnMoneyline": _mapping(outcome.get("mappings"), "BETTING_LINE"),
            }
        if set(sides) != {"home", "away"}:
            continue
        correct = proposition.get("correctOutcomes") or []
        games.append({
            "season": season,
            "week": week,
            "awayTeam": sides["away"]["team"],
            "homeTeam": sides["home"]["team"],
            "espnEventId": _mapping(proposition.get("mappings"), "EVENT_ID"),
            "lockAt": _iso(proposition.get("lockDate")),
            "entries": sides["home"]["count"] + sides["away"]["count"],
            "homeShare": sides["home"]["share"],
            "awayShare": sides["away"]["share"],
            "espnSpread": proposition.get("spread"),
            "espnHomeMoneyline": sides["home"]["espnMoneyline"],
            "espnAwayMoneyline": sides["away"]["espnMoneyline"],
            "homeWin": (sides["home"]["won"] if correct else None),
            "capturedAt": captured_at.isoformat().replace("+00:00", "Z"),
        })
    return games


def week_path(season: int, week: int, output_dir: Path = CROWD_DIR) -> Path:
    return output_dir / str(season) / f"week-{week:02d}.json"


def merge_week(existing: list[dict[str, Any]], fresh: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Update shares for games that have not locked. Keep the last pre-lock capture for locked games."""
    by_game = {(game["awayTeam"], game["homeTeam"]): dict(game) for game in existing}
    for game in fresh:
        key = (game["awayTeam"], game["homeTeam"])
        previous = by_game.get(key)
        lock_at = game.get("lockAt")
        locked = lock_at is not None and datetime.fromisoformat(lock_at.replace("Z", "+00:00")) <= now
        if previous is not None and locked:
            if previous.get("homeWin") is None and game.get("homeWin") is not None:
                previous["homeWin"] = game["homeWin"]
            continue
        by_game[key] = game
    return sorted(by_game.values(), key=lambda game: (game.get("lockAt") or "", game["awayTeam"]))


def write_week(season: int, week: int, games: list[dict[str, Any]], output_dir: Path = CROWD_DIR) -> Path:
    path = week_path(season, week, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"provider": "ESPN Pick'em", "season": season, "week": week, "scoringFormat": "straight", "games": games}, indent=2) + "\n")
    return path


def read_week(season: int, week: int, output_dir: Path = CROWD_DIR) -> list[dict[str, Any]]:
    path = week_path(season, week, output_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("games", [])


def archive_seasons(from_season: int, to_season: int, output_dir: Path = CROWD_DIR) -> None:
    """Download completed seasons. ESPN keeps the final shares for past games."""
    keys = challenge_keys_by_season(fetch_json(CHALLENGES_URL))
    now = datetime.now(timezone.utc)
    for season in range(from_season, to_season + 1):
        if season not in keys:
            print(f"No ESPN challenge for {season}; skipping")
            continue
        for week in range(1, REGULAR_SEASON_WEEKS + 1):
            games = normalize_pick_shares(fetch_challenge_week(keys[season], week), season, week, now)
            if not games:
                continue
            write_week(season, week, games, output_dir)
            print(f"Archived {season} week {week}: {len(games)} games")


def capture_due(now: datetime, first_kickoff: datetime) -> bool:
    minutes_until_kickoff = (first_kickoff - now).total_seconds() / 60
    return 0 < minutes_until_kickoff <= CAPTURE_WINDOW_MINUTES


def capture_current_week(now: datetime | None = None, output_dir: Path = CROWD_DIR, force: bool = False) -> Path | None:
    """Refresh the current week's shares when a game day is about to start."""
    now = now or datetime.now(timezone.utc)
    season = predict.current_nfl_season(now)
    game_day, first_kickoff, day_games = predict.upcoming_game_day(predict.load_schedule([season]), season, now)
    if not force and not capture_due(now, first_kickoff):
        print(f"Crowd capture for {game_day} is not due yet")
        return None
    week = int(day_games.week.iloc[0])
    fresh = normalize_pick_shares(fetch_challenge_week(challenge_key_for_season(season), week), season, week, now)
    if not fresh:
        raise RuntimeError(f"ESPN returned no pick shares for {season} week {week}")
    merged = merge_week(read_week(season, week, output_dir), fresh, now)
    path = write_week(season, week, merged, output_dir)
    print(f"Captured crowd shares for {len(fresh)} games in {season} week {week}")
    return path


def load_archive(from_season: int, to_season: int, output_dir: Path = CROWD_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in range(from_season, to_season + 1):
        for week in range(1, REGULAR_SEASON_WEEKS + 1):
            rows.extend(read_week(season, week, output_dir))
    return pd.DataFrame(rows)


def join_market(crowd: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach the recorded market probability and the outcome from the nflverse schedule."""
    games = schedule[schedule.game_type == "REG"].copy()
    games["market_home_prob"] = games.apply(predict.market_home_probability, axis=1)
    games["home_win"] = (games.home_score > games.away_score).astype(float)
    games.loc[games.home_score.isna() | games.away_score.isna(), "home_win"] = float("nan")
    games.loc[games.home_score == games.away_score, "home_win"] = 0.5
    merged = crowd.merge(
        games[["game_id", "season", "week", "away_team", "home_team", "market_home_prob", "home_win", "gameday"]],
        left_on=["season", "week", "awayTeam", "homeTeam"],
        right_on=["season", "week", "away_team", "home_team"],
        how="inner",
    )
    return merged.drop(columns=["away_team", "home_team"])


def evaluate(joined: pd.DataFrame) -> dict[str, Any]:
    """Compare the crowd favourite to the market favourite on the same games."""
    scored = joined.dropna(subset=["market_home_prob", "home_win"]).copy()
    scored = scored[scored.home_win != 0.5]
    actual = scored.home_win.to_numpy()
    crowd_home = scored.homeShare.to_numpy()
    market_home = scored.market_home_prob.to_numpy()
    crowd_pick_correct = ((crowd_home >= 0.5) == (actual == 1)).mean()
    market_pick_correct = ((market_home >= 0.5) == (actual == 1)).mean()
    disagree = (crowd_home >= 0.5) != (market_home >= 0.5)
    result: dict[str, Any] = {
        "games": int(len(scored)),
        "crowdFavoriteAccuracy": float(crowd_pick_correct),
        "marketFavoriteAccuracy": float(market_pick_correct),
        "crowdAsProbability": predict.probability_metrics(actual, crowd_home),
        "market": predict.probability_metrics(actual, market_home),
        "disagreements": int(disagree.sum()),
        "marketFavoriteAccuracyWhenDisagree": float(((market_home[disagree] >= 0.5) == (actual[disagree] == 1)).mean()) if disagree.any() else None,
        "byMarketBucket": [],
        "bySeason": {},
    }
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
    favorite_prob = pd.Series([max(p, 1 - p) for p in market_home])
    favorite_share = pd.Series([h if p >= 0.5 else 1 - h for p, h in zip(market_home, crowd_home)])
    favorite_won = pd.Series([(a == 1) == (p >= 0.5) for p, a in zip(market_home, actual)])
    bucket = pd.cut(favorite_prob, bins=bins, include_lowest=True, right=False)
    for label, index in bucket.groupby(bucket, observed=True).groups.items():
        result["byMarketBucket"].append({
            "marketFavoriteRange": str(label),
            "games": int(len(index)),
            "meanMarketFavoriteProbability": float(favorite_prob[index].mean()),
            "meanCrowdFavoriteShare": float(favorite_share[index].mean()),
            "favoriteWinRate": float(favorite_won[index].mean()),
        })
    for season, group in scored.groupby("season"):
        season_actual = group.home_win.to_numpy()
        result["bySeason"][int(season)] = {
            "games": int(len(group)),
            "crowdFavoriteAccuracy": float(((group.homeShare.to_numpy() >= 0.5) == (season_actual == 1)).mean()),
            "marketFavoriteAccuracy": float(((group.market_home_prob.to_numpy() >= 0.5) == (season_actual == 1)).mean()),
        }
    return result


def contrarian_value(market_prob: float, crowd_share: float) -> float:
    """Market probability of a team minus the share of the field that picked it.

    Positive values mark teams the field under-picks relative to their chance
    of winning. Those are the cheapest places to differ from a league.
    """
    return market_prob - crowd_share


def weekly_report(current: dict[str, Any], crowd_games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join live picks with the crowd shares for the same week."""
    crowd_by_game = {(game["awayTeam"], game["homeTeam"]): game for game in crowd_games}
    rows = []
    for game in current.get("games", []):
        crowd = crowd_by_game.get((game["awayTeam"], game["homeTeam"]))
        if crowd is None:
            continue
        market_home = game.get("marketHomeProbability")
        if market_home is None:
            market_home = game["homeWinProbability"]
        home_value = contrarian_value(market_home, crowd["homeShare"])
        contrarian_team = game["homeTeam"] if home_value > 0 else game["awayTeam"]
        contrarian_prob = market_home if contrarian_team == game["homeTeam"] else 1 - market_home
        rows.append({
            "gameId": game["gameId"],
            "awayTeam": game["awayTeam"],
            "homeTeam": game["homeTeam"],
            "pick": game["pick"],
            "modelHomeProbability": game["homeWinProbability"],
            "marketHomeProbability": market_home,
            "crowdHomeShare": crowd["homeShare"],
            "entries": crowd["entries"],
            "contrarianTeam": contrarian_team,
            "contrarianTeamMarketProbability": contrarian_prob,
            "contrarianTeamCrowdShare": crowd["homeShare"] if contrarian_team == game["homeTeam"] else crowd["awayShare"],
            "contrarianValue": abs(home_value),
            "expectedPointCost": max(0.0, 1 - 2 * contrarian_prob),
        })
    return sorted(rows, key=lambda row: -row["contrarianValue"])


def format_report(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'game':<10} {'pick':<5} {'model':>6} {'market':>7} {'crowd':>6} {'contrarian':<11} {'mkt':>5} {'crowd':>6} {'value':>6} {'cost':>6}"]
    for row in rows:
        lines.append(
            f"{row['awayTeam'] + '@' + row['homeTeam']:<10} {row['pick']:<5} {row['modelHomeProbability']:>6.3f} {row['marketHomeProbability']:>7.3f} {row['crowdHomeShare']:>6.3f} "
            f"{row['contrarianTeam']:<11} {row['contrarianTeamMarketProbability']:>5.2f} {row['contrarianTeamCrowdShare']:>6.2f} {row['contrarianValue']:>6.3f} {row['expectedPointCost']:>6.3f}"
        )
    lines.append("model/market/crowd columns are home-team values")
    lines.append("value = contrarian team's market probability minus its crowd share; cost = expected points given up by picking it instead of the market favorite")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ESPN Pick'em crowd shares")
    parser.add_argument("--archive", action="store_true", help="Download completed seasons into data/crowd-picks")
    parser.add_argument("--capture", action="store_true", help="Refresh the current week's shares when a game day is due")
    parser.add_argument("--force", action="store_true", help="Capture even when no game day is due")
    parser.add_argument("--evaluate", action="store_true", help="Compare crowd and market favorites on archived seasons")
    parser.add_argument("--report", action="store_true", help="Show contrarian value for the current week's live picks")
    parser.add_argument("--from-season", type=int, default=2021)
    parser.add_argument("--to-season", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "artifacts" / "crowd-evaluation.json")
    args = parser.parse_args()

    if args.archive:
        archive_seasons(args.from_season, args.to_season)
    if args.capture:
        capture_current_week(force=args.force)
    if args.evaluate:
        crowd = load_archive(args.from_season, args.to_season)
        if crowd.empty:
            raise SystemExit("No archived crowd picks. Run --archive first.")
        joined = join_market(crowd, predict.load_schedule(list(range(args.from_season, args.to_season + 1))))
        result = evaluate(joined)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key not in ("byMarketBucket", "bySeason")}, indent=2))
        print(f"{'market fav range':<16} {'games':>5} {'mkt fav':>8} {'crowd fav':>10} {'fav won':>8}")
        for row in result["byMarketBucket"]:
            print(f"{row['marketFavoriteRange']:<16} {row['games']:>5} {row['meanMarketFavoriteProbability']:>8.3f} {row['meanCrowdFavoriteShare']:>10.3f} {row['favoriteWinRate']:>8.3f}")
        for season, row in result["bySeason"].items():
            print(f"{season}: {row['games']} games, crowd favorite {row['crowdFavoriteAccuracy']:.3f}, market favorite {row['marketFavoriteAccuracy']:.3f}")
    if args.report:
        current = json.loads(predict.OUTPUT.read_text())
        season, week = int(current["season"]), int(current["week"])
        rows = weekly_report(current, read_week(season, week))
        if not rows:
            raise SystemExit(f"No crowd shares archived for {season} week {week}. Run --capture --force first.")
        print(format_report(rows))


if __name__ == "__main__":
    main()
