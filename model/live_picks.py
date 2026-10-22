"""Live publication using the frozen statistical model and fresh market odds."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

import crowd
import espn_pool
import live_odds
import predict


def check_team_coverage(schedule: pd.DataFrame, team_weeks: dict, season: int, week: int) -> None:
    completed = schedule[(schedule.season == season) & (schedule.game_type == "REG") & (schedule.week < week) & schedule.home_score.notna() & schedule.away_score.notna()]
    missing = [str(row.game_id) for row in completed.itertuples() if any((season, int(row.week), str(team)) not in team_weeks for team in (row.home_team, row.away_team))]
    if missing:
        raise RuntimeError(f"Current-season play-by-play is incomplete: {', '.join(missing)}")


def apply_quote(game: dict[str, Any], quote: dict[str, Any] | None, weight: float, now: datetime) -> None:
    market = game["marketHomeProbability"] if quote is None else quote["marketHomeProbability"]
    probability = game["statisticalHomeProbability"] if market is None else weight * market + (1 - weight) * game["statisticalHomeProbability"]
    game.update({
        "pick": game["homeTeam"] if probability >= 0.5 else game["awayTeam"],
        "homeWinProbability": round(probability, 4), "winProbability": round(max(probability, 1 - probability), 4),
        "marketHomeProbability": None if market is None else round(market, 4), "confidence": predict.confidence(probability),
        "marketSource": "The Odds API" if quote else "nflverse fallback" if market is not None else "no market",
        "marketCapturedAt": None if quote is None else quote["capturedAt"],
        "marketUpdatedAt": None if quote is None else quote["oldestBookUpdate"],
        "marketBookCount": 0 if quote is None else len(quote["bookmakers"]),
        "marketAgeMinutes": None if quote is None else round((now - live_odds.timestamp(quote["oldestBookUpdate"])).total_seconds() / 60, 1),
        "generatedAt": live_odds.iso(now),
    })


def generate(refresh_weather: bool, refresh_news: bool, now: datetime | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any] | None]:
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set THE_ODDS_API_KEY before generating live picks")
    started = now or datetime.now(timezone.utc)
    season = predict.current_nfl_season(started)
    schedule, pbp = predict.load_inputs(list(range(predict.START_SEASON, season + 1)))
    week, _ = predict.upcoming_week_and_kickoff(schedule, season, started)
    team_weeks = predict.aggregate_team_weeks(pbp)
    check_team_coverage(schedule, team_weeks, season, week)
    features = predict.build_game_features(schedule, team_weeks)
    model = predict.train_model(features, season)
    current = features[(features.season == season) & (features.week == week)].copy()
    current = current[current.apply(predict.kickoff_at, axis=1) > started].sort_values(["gameday", "gametime", "game_id"])
    if current.empty:
        raise RuntimeError("No games remain in the selected week")
    probabilities = model.pipeline.predict_proba(current[predict.FEATURES])[:, 1]
    stadiums = predict.load_stadiums() if refresh_weather else {}
    news = predict.news_for_teams(set(current.home_team) | set(current.away_team), started) if refresh_news else {}
    _, baseline = predict.baseline_for(season, week)
    shares = crowd.week_shares(season, week, started)
    try:
        context = espn_pool.load_context(season, week)
    except Exception as error:
        print(f"ESPN group unavailable: {type(error).__name__}")
        context = None
    games = []
    for (_, row), stat in zip(current.iterrows(), probabilities):
        market = row.get("market_home_prob")
        games.append({
            "gameId": str(row.game_id), "awayTeam": str(row.away_team), "homeTeam": str(row.home_team),
            "gameday": str(row.gameday), "gametime": None if pd.isna(row.get("gametime")) else str(row.gametime),
            "stadium": None if pd.isna(row.get("stadium")) else str(row.stadium),
            "roof": None if pd.isna(row.get("roof")) else str(row.roof),
            "statisticalHomeProbability": float(stat), "marketHomeProbability": None if pd.isna(market) else float(market),
            "spreadLine": None if pd.isna(row.get("spread_line")) else float(row.spread_line),
            "homeQb": None if pd.isna(row.get("home_qb_name")) else str(row.home_qb_name),
            "awayQb": None if pd.isna(row.get("away_qb_name")) else str(row.away_qb_name),
            "weather": predict.weather_for_game(row, stadiums) if refresh_weather else None,
            "news": (news.get(str(row.away_team), []) + news.get(str(row.home_team), []))[:2], "flags": [],
        })
    events = predict.fetch_current_odds(api_key)
    captured = now or datetime.now(timezone.utc)
    if any(predict.kickoff_at(row) <= captured for _, row in current.iterrows()):
        raise RuntimeError("Generation crossed a kickoff; keep the previous published picks")
    available = live_odds.quotes(events, current, captured)
    live_odds.require_due_quotes(current, available, captured)
    if not available:
        raise RuntimeError("The odds feed has no fresh multi-book prices for this week")
    if context is not None:
        regular = schedule[(schedule.season == season) & (schedule.game_type == "REG")]
        context["remainingGames"] = int((regular.apply(predict.kickoff_at, axis=1) > captured).sum())
    for game in games:
        apply_quote(game, available.get(game["gameId"]), model.market_weight, captured)
        game["statisticalHomeProbability"] = round(game["statisticalHomeProbability"], 4)
        game.update(predict.pool_fields(game, shares))
        game["flags"] = predict.game_flags(game, baseline)
        if game["marketCapturedAt"] is None:
            game["flags"].append("Fresh consensus odds unavailable; fallback odds have unknown age")
    payload = {
        "season": season, "week": week, "generatedAt": live_odds.iso(captured),
        "firstGame": min(game["gameday"] for game in games),
        "model": {"trainingSeasons": model.training_seasons, "validationSeason": model.validation_season,
                  "marketWeight": round(model.market_weight, 2), "validationAccuracy": round(model.validation_accuracy, 4),
                  "validationBrier": round(model.validation_brier, 4),
                  "marketBrier": None if model.market_brier is None else round(model.market_brier, 4)},
        "games": games,
    }
    return payload, available, context
