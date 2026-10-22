"""Fresh consensus odds and immutable captures for live predictions."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

import predict

MAX_BOOK_AGE_MINUTES = 30
MIN_BOOKMAKERS = 2


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("An odds timestamp must include a time zone")
    return result


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def quotes(events: list[dict[str, Any]], games: pd.DataFrame, captured_at: datetime) -> dict[str, dict[str, Any]]:
    """Reject old prices, duplicate books, and events with a different kickoff."""
    result = {}
    for row in games.itertuples(index=False):
        game = pd.Series(row._asdict())
        kickoff = predict.kickoff_at(game)
        if kickoff <= captured_at:
            continue
        home, away = predict.TEAM_NAMES[str(row.home_team)], predict.TEAM_NAMES[str(row.away_team)]
        matches = []
        for event in events:
            if (event.get("home_team"), event.get("away_team")) != (home, away):
                continue
            try:
                start = timestamp(event["commence_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if start > captured_at and abs((start - kickoff).total_seconds()) <= 600:
                matches.append(event)
        if len(matches) != 1:
            continue
        event = matches[0]
        books = {}
        for book in event.get("bookmakers", []):
            market = next((item for item in book.get("markets", []) if item.get("key") == "h2h"), None)
            if market is None or not book.get("key"):
                continue
            try:
                updated = timestamp(market.get("last_update") or book["last_update"])
                age = (captured_at - updated).total_seconds() / 60
                prices = {item["name"]: float(item["price"]) for item in market["outcomes"]}
                if set(prices) != {home, away} or any(not math.isfinite(value) or abs(value) < 100 for value in prices.values()):
                    continue
                probability = predict.devig_home_probability(prices[home], prices[away])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not 0 <= age <= MAX_BOOK_AGE_MINUTES or probability is None:
                continue
            previous = books.get(book["key"])
            if previous and timestamp(previous["lastUpdate"]) >= updated:
                continue
            books[book["key"]] = {
                "key": book["key"], "title": book.get("title", book["key"]), "lastUpdate": iso(updated),
                "homeMoneyline": prices[home], "awayMoneyline": prices[away], "homeProbability": probability,
            }
        if len(books) < MIN_BOOKMAKERS:
            continue
        ordered = sorted(books.values(), key=lambda book: book["key"])
        result[str(row.game_id)] = {
            "gameId": str(row.game_id), "homeTeam": str(row.home_team), "awayTeam": str(row.away_team),
            "kickoffAt": iso(kickoff), "oddsEventId": event.get("id"),
            "marketHomeProbability": sum(book["homeProbability"] for book in ordered) / len(ordered),
            "bookmakers": ordered, "capturedAt": iso(captured_at),
            "oldestBookUpdate": min(ordered, key=lambda book: timestamp(book["lastUpdate"]))["lastUpdate"],
        }
    return result


def require_due_quotes(games: pd.DataFrame, available: dict[str, dict[str, Any]], now: datetime) -> None:
    missing = []
    for _, row in games.iterrows():
        minutes = (predict.kickoff_at(row) - now).total_seconds() / 60
        if 0 < minutes <= predict.CAPTURE_WINDOW_MAX_MINUTES and str(row.game_id) not in available:
            missing.append(str(row.game_id))
    if missing:
        raise RuntimeError(f"Fresh odds from at least {MIN_BOOKMAKERS} books are missing: {', '.join(missing)}")


def archive(available: dict[str, dict[str, Any]], season: int, payload: dict[str, Any] | None, saved_at: datetime, output_dir: Path = predict.ODDS_SNAPSHOT_DIR) -> list[Path]:
    """Save each capture. A later kickoff can have a new capture on the same day."""
    published = {game["gameId"]: game for game in (payload or {}).get("games", [])}
    days: dict[str, list[dict[str, Any]]] = {}
    for quote in available.values():
        kickoff = timestamp(quote["kickoffAt"])
        if saved_at >= kickoff:
            raise RuntimeError(f"Capture finished after kickoff: {quote['gameId']}")
        day = kickoff.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        game = published.get(quote["gameId"])
        frozen = None if game is None else {
            "generatedAt": game["generatedAt"], "marketCapturedAt": game.get("marketCapturedAt"),
            "pick": game["pick"], "poolPick": game.get("poolPick"),
            "homeWinProbability": game["homeWinProbability"],
            "statisticalHomeProbability": game["statisticalHomeProbability"],
            "marketHomeProbability": game["marketHomeProbability"], "marketWeight": payload["model"]["marketWeight"],
        }
        days.setdefault(day, []).append({**quote, "published": frozen})
    paths = []
    for day, games in days.items():
        captured_at = min(timestamp(game["capturedAt"]) for game in games)
        first = min(timestamp(game["kickoffAt"]) for game in games)
        record = {
            "formatVersion": 2, "provider": "The Odds API", "season": season, "gameDay": day,
            "capturedAt": iso(captured_at), "savedAt": iso(saved_at), "firstKickoffAt": iso(first),
            "captureWindowOpensAt": iso(first - timedelta(minutes=predict.CAPTURE_WINDOW_MAX_MINUTES)),
            "captureWindowClosesAt": iso(first - timedelta(minutes=predict.CAPTURE_WINDOW_MIN_MINUTES)),
            "market": "h2h", "region": "us", "games": games,
        }
        path = output_dir / str(season) / day / f"{saved_at.astimezone(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as output:
                output.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        except FileExistsError:
            if json.loads(path.read_text()) != record:
                raise RuntimeError(f"A different capture already exists: {path}")
        paths.append(path)
    return paths
