from __future__ import annotations

import argparse
import csv
import json
import math
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "data" / "current.json"
SNAPSHOT_DIR = ROOT / "public" / "data" / "snapshots"
BACKTEST_OUTPUT = ROOT / "model" / "artifacts" / "backtest.json"
ODDS_SNAPSHOT_DIR = ROOT / "data" / "odds-snapshots"
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"
STADIUMS_URL = "https://raw.githubusercontent.com/greerreNFL/Stadiums/main/data/stadiums.csv"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
GOOGLE_NEWS = "https://news.google.com/rss/search"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
ROLLING_GAMES = 8
START_SEASON = 2021
ELO_INITIAL_RATING = 1500.0
ELO_HOME_ADVANTAGE = 65.0
ELO_K_FACTOR = 20.0
ELO_SEASON_REGRESSION = 0.67
METRICS = ["off_epa", "off_success", "pass_epa", "rush_epa", "def_epa_allowed", "def_success_allowed", "def_pass_epa_allowed", "def_rush_epa_allowed"]
FEATURES = [f"diff_{m}" for m in METRICS] + ["rest_diff", "neutral_site"]
ELO_FEATURES = FEATURES + ["elo_diff"]
QB_FEATURES = FEATURES + ["diff_qb_epa_per_dropback", "diff_qb_cpoe", "diff_qb_log_dropbacks"]
ELO_QB_FEATURES = ELO_FEATURES + QB_FEATURES[len(FEATURES):]
OPPONENT_ADJUSTED_FEATURES = FEATURES + [f"diff_adjusted_{metric}" for metric in METRICS]
OPPONENT_METRICS = {
    "off_epa": "def_epa_allowed",
    "off_success": "def_success_allowed",
    "pass_epa": "def_pass_epa_allowed",
    "rush_epa": "def_rush_epa_allowed",
    "def_epa_allowed": "off_epa",
    "def_success_allowed": "off_success",
    "def_pass_epa_allowed": "pass_epa",
    "def_rush_epa_allowed": "rush_epa",
}
TEAM_NAMES = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers", "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}


def current_nfl_season(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def american_implied(odds: float | int | None) -> float | None:
    if odds is None or pd.isna(odds) or float(odds) == 0:
        return None
    value = float(odds)
    return 100 / (value + 100) if value > 0 else -value / (-value + 100)


def devig_home_probability(home_odds: Any, away_odds: Any) -> float | None:
    home, away = american_implied(home_odds), american_implied(away_odds)
    if home is None or away is None:
        return None
    return home / (home + away)


def spread_home_probability(spread_line: Any) -> float | None:
    if spread_line is None or pd.isna(spread_line):
        return None
    return 1 / (1 + math.exp(-float(spread_line) / 6.5))


def market_home_probability(row: pd.Series) -> float | None:
    ml = devig_home_probability(row.get("home_moneyline"), row.get("away_moneyline"))
    return ml if ml is not None else spread_home_probability(row.get("spread_line"))


def odds_snapshot_path(season: int, game_day: str, output_dir: Path = ODDS_SNAPSHOT_DIR) -> Path:
    return output_dir / str(season) / f"{game_day}.json"


def upcoming_game_day(schedule: pd.DataFrame, season: int, now: datetime) -> tuple[str, datetime, pd.DataFrame]:
    games = schedule[(schedule.season == season) & schedule.game_type.isin(["REG", "POST"])].copy()
    games["_kickoff"] = games.apply(kickoff_at, axis=1)
    upcoming = games[games._kickoff > now]
    if upcoming.empty:
        raise RuntimeError(f"No upcoming games found for {season}")
    first_game = upcoming.sort_values("_kickoff").iloc[0]
    game_day = str(first_game.gameday)
    day_games = games[games.gameday.astype(str) == game_day].copy()
    return game_day, min(day_games._kickoff), day_games


def odds_capture_due(now: datetime, first_kickoff: datetime) -> bool:
    minutes_until_kickoff = (first_kickoff - now).total_seconds() / 60
    return 60 <= minutes_until_kickoff <= 120


def fetch_current_odds(api_key: str) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"})
    request = urllib.request.Request(f"{ODDS_API_URL}?{params}", headers={"User-Agent": "nfl-picks/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode())
    if not isinstance(data, list):
        raise RuntimeError("The Odds API returned an unexpected response")
    return data


def published_picks(season: int, output: Path = OUTPUT) -> dict[str, dict[str, Any]]:
    """Live picks from the published week file, keyed by game id, so a snapshot freezes what users saw."""
    if not output.exists():
        return {}
    payload = json.loads(output.read_text())
    if int(payload.get("season", -1)) != season:
        return {}
    generated_at = payload.get("generatedAt")
    return {
        str(game["gameId"]): {
            "generatedAt": generated_at,
            "pick": game.get("pick"),
            "homeWinProbability": game.get("homeWinProbability"),
            "statisticalHomeProbability": game.get("statisticalHomeProbability"),
            "marketHomeProbability": game.get("marketHomeProbability"),
            "marketWeight": payload.get("model", {}).get("marketWeight"),
        }
        for game in payload.get("games", [])
    }


def normalize_odds_snapshot(events: list[dict[str, Any]], games: pd.DataFrame, season: int, game_day: str, first_kickoff: datetime, captured_at: datetime, published: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    events_by_teams = {(str(event.get("home_team")), str(event.get("away_team"))): event for event in events}
    published = published or {}
    snapshot_games = []
    for _, row in games.sort_values(["gameday", "gametime", "game_id"]).iterrows():
        home_name, away_name = TEAM_NAMES.get(str(row.home_team)), TEAM_NAMES.get(str(row.away_team))
        event = events_by_teams.get((home_name, away_name))
        if event is None:
            continue
        bookmakers = []
        for bookmaker in event.get("bookmakers", []):
            market = next((item for item in bookmaker.get("markets", []) if item.get("key") == "h2h"), None)
            if market is None:
                continue
            prices = {str(outcome.get("name")): outcome.get("price") for outcome in market.get("outcomes", [])}
            home_odds, away_odds = prices.get(home_name), prices.get(away_name)
            probability = devig_home_probability(home_odds, away_odds)
            if probability is None:
                continue
            bookmakers.append({
                "key": bookmaker.get("key"),
                "title": bookmaker.get("title"),
                "lastUpdate": bookmaker.get("last_update"),
                "homeMoneyline": home_odds,
                "awayMoneyline": away_odds,
                "homeProbability": probability,
            })
        if not bookmakers:
            continue
        snapshot_games.append({
            "gameId": str(row.game_id),
            "kickoffAt": kickoff_at(row).isoformat().replace("+00:00", "Z"),
            "homeTeam": str(row.home_team),
            "awayTeam": str(row.away_team),
            "oddsEventId": event.get("id"),
            "marketHomeProbability": sum(book["homeProbability"] for book in bookmakers) / len(bookmakers),
            "published": published.get(str(row.game_id)),
            "bookmakers": bookmakers,
        })
    if not snapshot_games:
        raise RuntimeError("The Odds API returned no usable odds for the scheduled games")
    target = first_kickoff - timedelta(minutes=90)
    return {
        "provider": "The Odds API",
        "season": season,
        "gameDay": game_day,
        "captureTargetAt": target.isoformat().replace("+00:00", "Z"),
        "capturedAt": captured_at.isoformat().replace("+00:00", "Z"),
        "firstKickoffAt": first_kickoff.isoformat().replace("+00:00", "Z"),
        "market": "h2h",
        "region": "us",
        "games": snapshot_games,
    }


def capture_odds(now: datetime | None = None, output_dir: Path = ODDS_SNAPSHOT_DIR) -> dict[str, Any] | None:
    now = now or datetime.now(timezone.utc)
    season = current_nfl_season(now)
    game_day, first_kickoff, games = upcoming_game_day(load_schedule([season]), season, now)
    if not odds_capture_due(now, first_kickoff):
        print(f"Odds capture for {game_day} is not due yet")
        return None
    path = odds_snapshot_path(season, game_day, output_dir)
    if path.exists():
        print(f"Odds snapshot already exists for {game_day}")
        return json.loads(path.read_text())
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set THE_ODDS_API_KEY before capturing odds")
    snapshot = normalize_odds_snapshot(fetch_current_odds(api_key), games, season, game_day, first_kickoff, now, published_picks(season))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"Captured odds for {len(snapshot['games'])} games on {game_day}")
    return snapshot


def load_schedule(seasons: list[int]) -> pd.DataFrame:
    cols = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "away_score", "home_team", "home_score", "location", "away_rest", "home_rest", "away_moneyline", "home_moneyline", "spread_line", "roof", "home_qb_name", "away_qb_name", "stadium_id", "stadium"]
    df = pd.read_csv(SCHEDULE_URL, usecols=cols, low_memory=False)
    return df[df.season.isin(seasons)].copy()


def load_inputs(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = load_schedule(seasons)
    cols = ["game_id", "season", "season_type", "week", "posteam", "defteam", "play_type", "epa", "success", "pass", "rush", "passer_player_id", "qb_dropback", "qb_epa", "cpoe"]
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            frames.append(pd.read_csv(PBP_URL.format(season=season), usecols=cols, low_memory=False))
        except Exception as exc:
            if season < max(seasons):
                raise RuntimeError(f"Could not load {season} play-by-play") from exc
    if not frames:
        raise RuntimeError("No play-by-play data available")
    return schedule, pd.concat(frames, ignore_index=True)


def aggregate_team_weeks(pbp: pd.DataFrame) -> dict[tuple[int, int, str], dict[str, float]]:
    plays = pbp[(pbp.season_type == "REG") & pbp.posteam.notna() & pbp.defteam.notna() & pbp.epa.notna() & pbp.play_type.isin(["pass", "run"])].copy()
    plays["is_pass"] = (plays["pass"] == 1) | (plays.play_type == "pass")
    plays["is_rush"] = (plays["rush"] == 1) | (plays.play_type == "run")
    offense = plays.groupby(["season", "week", "posteam"], as_index=False).agg(off_epa=("epa", "mean"), off_success=("success", "mean")).rename(columns={"posteam": "team"})
    pass_off = plays[plays.is_pass].groupby(["season", "week", "posteam"], as_index=False).epa.mean().rename(columns={"posteam": "team", "epa": "pass_epa"})
    rush_off = plays[plays.is_rush].groupby(["season", "week", "posteam"], as_index=False).epa.mean().rename(columns={"posteam": "team", "epa": "rush_epa"})
    defense = plays.groupby(["season", "week", "defteam"], as_index=False).agg(def_epa_allowed=("epa", "mean"), def_success_allowed=("success", "mean")).rename(columns={"defteam": "team"})
    pass_def = plays[plays.is_pass].groupby(["season", "week", "defteam"], as_index=False).epa.mean().rename(columns={"defteam": "team", "epa": "def_pass_epa_allowed"})
    rush_def = plays[plays.is_rush].groupby(["season", "week", "defteam"], as_index=False).epa.mean().rename(columns={"defteam": "team", "epa": "def_rush_epa_allowed"})
    team_week = offense
    for part in (pass_off, rush_off, defense, pass_def, rush_def):
        team_week = team_week.merge(part, on=["season", "week", "team"], how="outer")
    result = {}
    for row in team_week.to_dict("records"):
        result[(int(row["season"]), int(row["week"]), str(row["team"]))] = {m: float(row[m]) if pd.notna(row.get(m)) else np.nan for m in METRICS}
    return result


def aggregate_qb_games(pbp: pd.DataFrame) -> dict[str, list[dict[str, float | str]]]:
    dropbacks = pbp[pbp.passer_player_id.notna() & pbp.posteam.notna() & (pbp.qb_dropback == 1)].copy()
    grouped = dropbacks.groupby(["game_id", "posteam", "passer_player_id"], as_index=False).agg(
        dropbacks=("qb_dropback", "sum"),
        qb_epa=("qb_epa", lambda values: values.sum(min_count=1)),
        cpoe=("cpoe", "mean"),
    )
    games: defaultdict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in grouped.itertuples(index=False):
        games[str(row.game_id)].append({
            "id": str(row.passer_player_id),
            "team": str(row.posteam),
            "dropbacks": float(row.dropbacks),
            "qb_epa": float(row.qb_epa) if pd.notna(row.qb_epa) else np.nan,
            "cpoe": float(row.cpoe) if pd.notna(row.cpoe) else np.nan,
        })
    return dict(games)


def qb_history_features(history: deque[dict[str, float | str]]) -> dict[str, float]:
    if not history:
        return {"qb_epa_per_dropback": np.nan, "qb_cpoe": np.nan, "qb_log_dropbacks": np.nan}
    dropbacks = sum(float(game["dropbacks"]) for game in history)
    epa = sum(float(game["qb_epa"]) for game in history if not pd.isna(game["qb_epa"]))
    cpoe_games = [game for game in history if not pd.isna(game["cpoe"])]
    cpoe_dropbacks = sum(float(game["dropbacks"]) for game in cpoe_games)
    return {
        "qb_epa_per_dropback": epa / dropbacks if dropbacks else np.nan,
        "qb_cpoe": sum(float(game["cpoe"]) * float(game["dropbacks"]) for game in cpoe_games) / cpoe_dropbacks if cpoe_dropbacks else np.nan,
        "qb_log_dropbacks": math.log1p(dropbacks),
    }


def build_qb_features(features: pd.DataFrame, qb_games: dict[str, list[dict[str, float | str]]]) -> pd.DataFrame:
    """Add prior-game primary-QB features without using target-game starter data."""
    games = features.copy()
    histories: defaultdict[str, deque[dict[str, float | str]]] = defaultdict(lambda: deque(maxlen=ROLLING_GAMES))
    latest_qbs: dict[str, str] = {}
    games["_kickoff"] = games.apply(kickoff_at, axis=1)
    for kickoff, simultaneous_games in games.groupby("_kickoff", sort=True):
        for index, game in simultaneous_games.iterrows():
            home_qb = latest_qbs.get(str(game.home_team))
            away_qb = latest_qbs.get(str(game.away_team))
            home = qb_history_features(histories[home_qb]) if home_qb else qb_history_features(deque())
            away = qb_history_features(histories[away_qb]) if away_qb else qb_history_features(deque())
            games.loc[index, "home_recent_qb_id"] = home_qb
            games.loc[index, "away_recent_qb_id"] = away_qb
            for metric in ("qb_epa_per_dropback", "qb_cpoe", "qb_log_dropbacks"):
                games.loc[index, f"diff_{metric}"] = home[metric] - away[metric]
        for _, game in simultaneous_games.iterrows():
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            stats = qb_games.get(str(game.game_id), [])
            for stat in stats:
                histories[str(stat["id"])].append(stat)
            for team in {str(stat["team"]) for stat in stats}:
                primary = max((stat for stat in stats if str(stat["team"]) == team), key=lambda stat: (float(stat["dropbacks"]), str(stat["id"])))
                latest_qbs[team] = str(primary["id"])
    return games.drop(columns="_kickoff")


def history_average(history: deque[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {m: np.nan for m in METRICS}
    return {m: float(np.mean([x[m] for x in history if not pd.isna(x.get(m))])) if any(not pd.isna(x.get(m)) for x in history) else np.nan for m in METRICS}


def opponent_adjusted_stats(team_stats: dict[str, float], opponent_stats: dict[str, float]) -> dict[str, float]:
    return {
        metric: team_stats[metric] - opponent_stats[OPPONENT_METRICS[metric]]
        if not pd.isna(team_stats.get(metric)) and not pd.isna(opponent_stats.get(OPPONENT_METRICS[metric]))
        else np.nan
        for metric in METRICS
    }


def pregame_elo_differences(games: pd.DataFrame) -> dict[str, float]:
    """Calculate pregame Elo rating gaps and update ratings after each completed week."""
    ratings: defaultdict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    differences: dict[str, float] = {}
    games = games.sort_values(["season", "week", "gameday", "gametime", "game_id"])
    previous_season: int | None = None
    for season, week in games[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).itertuples(index=False):
        if previous_season is not None and season != previous_season:
            for team, rating in ratings.items():
                ratings[team] = ELO_INITIAL_RATING + ELO_SEASON_REGRESSION * (rating - ELO_INITIAL_RATING)
        week_games = games[(games.season == season) & (games.week == week)]
        week_predictions: list[tuple[pd.Series, float]] = []
        for _, game in week_games.iterrows():
            home, away = str(game.home_team), str(game.away_team)
            difference = ratings[home] - ratings[away]
            if str(game.get("location", "")).lower() != "neutral":
                difference += ELO_HOME_ADVANTAGE
            differences[str(game.game_id)] = difference
            week_predictions.append((game, 1 / (1 + 10 ** (-difference / 400))))
        for game, probability in week_predictions:
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            outcome = 1.0 if float(game.home_score) > float(game.away_score) else 0.0 if float(game.home_score) < float(game.away_score) else 0.5
            change = ELO_K_FACTOR * (outcome - probability)
            ratings[str(game.home_team)] += change
            ratings[str(game.away_team)] -= change
        previous_season = int(season)
    return differences


def build_game_features(schedule: pd.DataFrame, team_weeks: dict[tuple[int, int, str], dict[str, float]]) -> pd.DataFrame:
    games = schedule[schedule.game_type == "REG"].copy()
    games["_date"] = pd.to_datetime(games.gameday, errors="coerce")
    games = games.sort_values(["season", "week", "_date", "game_id"])
    elo_differences = pregame_elo_differences(games)
    histories: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=ROLLING_GAMES))
    adjusted_histories: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=ROLLING_GAMES))
    rows = []
    for _, game in games.iterrows():
        home, away = str(game.home_team), str(game.away_team)
        ha, aa = history_average(histories[home]), history_average(histories[away])
        ha_adjusted, aa_adjusted = history_average(adjusted_histories[home]), history_average(adjusted_histories[away])
        record = {k: game.get(k) for k in games.columns if k != "_date"}
        for metric in METRICS:
            record[f"diff_{metric}"] = ha[metric] - aa[metric]
            record[f"diff_adjusted_{metric}"] = ha_adjusted[metric] - aa_adjusted[metric]
        record["rest_diff"] = float(game.home_rest) - float(game.away_rest) if pd.notna(game.home_rest) and pd.notna(game.away_rest) else 0.0
        record["neutral_site"] = 1.0 if str(game.get("location", "")).lower() == "neutral" else 0.0
        record["elo_diff"] = elo_differences[str(game.game_id)]
        record["market_home_prob"] = market_home_probability(game)
        if pd.notna(game.home_score) and pd.notna(game.away_score) and float(game.home_score) != float(game.away_score):
            record["home_win"] = int(float(game.home_score) > float(game.away_score))
        else:
            record["home_win"] = np.nan
        rows.append(record)
        season, week = int(game.season), int(game.week)
        for team, stats, opponent_stats in ((home, team_weeks.get((season, week, home)), aa), (away, team_weeks.get((season, week, away)), ha)):
            if stats:
                histories[team].append(stats)
                adjusted_histories[team].append(opponent_adjusted_stats(stats, opponent_stats))
    return pd.DataFrame(rows)


@dataclass
class ModelResult:
    pipeline: Pipeline
    validation_season: int
    market_weight: float
    validation_accuracy: float
    validation_brier: float
    market_brier: float | None
    training_seasons: list[int]


def new_pipeline() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, C=0.6))])


def select_market_weight(actual: np.ndarray, statistical: np.ndarray, market: np.ndarray) -> tuple[float, float]:
    scored = [(float(weight), float(brier_score_loss(actual, weight * market + (1 - weight) * statistical))) for weight in np.linspace(0, 1, 21)]
    return min(scored, key=lambda item: item[1])


def probability_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    probabilities = np.clip(probabilities.astype(float), 0.0001, 0.9999)
    return {
        "games": int(len(actual)),
        "accuracy": float(accuracy_score(actual, probabilities >= 0.5)),
        "brier": float(brier_score_loss(actual, probabilities)),
        "logLoss": float(log_loss(actual, probabilities, labels=[0, 1])),
    }


def elo_home_probability(home_rating: float, away_rating: float, neutral_site: bool) -> float:
    home_advantage = 0.0 if neutral_site else ELO_HOME_ADVANTAGE
    return 1 / (1 + 10 ** (-(home_rating + home_advantage - away_rating) / 400))


def elo_probabilities(features: pd.DataFrame, from_season: int, to_season: int) -> dict[str, float]:
    differences = pregame_elo_differences(features)
    scored = features[(features.season >= from_season) & (features.season <= to_season) & features.home_win.notna()]
    return {str(game.game_id): 1 / (1 + 10 ** (-differences[str(game.game_id)] / 400)) for _, game in scored.iterrows()}


def train_model(features: pd.DataFrame, current_season: int) -> ModelResult:
    completed = features[features.home_win.notna()].copy()
    seasons = sorted(int(s) for s in completed.season.unique())
    validation_season = current_season - 1 if current_season - 1 in seasons else seasons[-1]
    train, validation = completed[completed.season < validation_season], completed[completed.season == validation_season]
    if len(train) < 200:
        raise RuntimeError("Not enough historical games to train")
    pipe = new_pipeline()
    pipe.fit(train[FEATURES], train.home_win.astype(int))
    stat = pipe.predict_proba(validation[FEATURES])[:, 1]
    y = validation.home_win.astype(int).to_numpy()
    accuracy = float(accuracy_score(y, stat >= 0.5))
    has_market = validation.market_home_prob.notna().to_numpy()
    weight, market_brier = 0.7, None
    blend_brier = float(brier_score_loss(y, stat))
    if has_market.sum() >= 100:
        market = validation.loc[has_market, "market_home_prob"].astype(float).to_numpy()
        stat_m, y_m = stat[has_market], y[has_market]
        market_brier = float(brier_score_loss(y_m, market))
        weight, blend_brier = select_market_weight(y_m, stat_m, market)
    pipe.fit(completed[FEATURES], completed.home_win.astype(int))
    return ModelResult(pipe, validation_season, weight, accuracy, blend_brier, market_brier, seasons)


def backtest(features: pd.DataFrame, from_season: int, to_season: int) -> dict[str, Any]:
    """Replay completed weeks with only results available before each week's kickoff."""
    completed = features[features.home_win.notna()].copy()
    targets = completed[(completed.season >= from_season) & (completed.season <= to_season)]
    predictions: list[dict[str, Any]] = []
    skipped_weeks = 0
    for season, week in targets[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).itertuples(index=False):
        games = completed[(completed.season == season) & (completed.week == week)].copy()
        history = completed[(completed.season < season) | ((completed.season == season) & (completed.week < week))]
        prior_seasons = sorted(int(value) for value in history.loc[history.season < season, "season"].unique())
        if len(history) < 200 or len(prior_seasons) < 2:
            skipped_weeks += 1
            continue
        validation_season = prior_seasons[-1]
        validation_train = history[history.season < validation_season]
        validation = history[history.season == validation_season]
        if len(validation_train) < 200 or validation.empty:
            skipped_weeks += 1
            continue
        validation_pipe = new_pipeline()
        validation_pipe.fit(validation_train[FEATURES], validation_train.home_win.astype(int))
        validation_stat = validation_pipe.predict_proba(validation[FEATURES])[:, 1]
        validation_elo_feature_pipe = new_pipeline()
        validation_elo_feature_pipe.fit(validation_train[ELO_FEATURES], validation_train.home_win.astype(int))
        validation_elo_feature_stat = validation_elo_feature_pipe.predict_proba(validation[ELO_FEATURES])[:, 1]
        validation_qb_feature_pipe = new_pipeline()
        validation_qb_feature_pipe.fit(validation_train[QB_FEATURES], validation_train.home_win.astype(int))
        validation_qb_feature_stat = validation_qb_feature_pipe.predict_proba(validation[QB_FEATURES])[:, 1]
        validation_elo_qb_feature_pipe = new_pipeline()
        validation_elo_qb_feature_pipe.fit(validation_train[ELO_QB_FEATURES], validation_train.home_win.astype(int))
        validation_elo_qb_feature_stat = validation_elo_qb_feature_pipe.predict_proba(validation[ELO_QB_FEATURES])[:, 1]
        validation_opponent_adjusted_pipe = new_pipeline()
        validation_opponent_adjusted_pipe.fit(validation_train[OPPONENT_ADJUSTED_FEATURES], validation_train.home_win.astype(int))
        validation_opponent_adjusted_stat = validation_opponent_adjusted_pipe.predict_proba(validation[OPPONENT_ADJUSTED_FEATURES])[:, 1]
        validation_market = validation.market_home_prob.notna().to_numpy()
        market_weight = 0.7
        elo_feature_market_weight = 0.7
        qb_feature_market_weight = 0.7
        elo_qb_feature_market_weight = 0.7
        opponent_adjusted_market_weight = 0.7
        if validation_market.sum() >= 100:
            market_weight, _ = select_market_weight(
                validation.home_win.astype(int).to_numpy()[validation_market],
                validation_stat[validation_market],
                validation.loc[validation_market, "market_home_prob"].astype(float).to_numpy(),
            )
            elo_feature_market_weight, _ = select_market_weight(
                validation.home_win.astype(int).to_numpy()[validation_market],
                validation_elo_feature_stat[validation_market],
                validation.loc[validation_market, "market_home_prob"].astype(float).to_numpy(),
            )
            qb_feature_market_weight, _ = select_market_weight(
                validation.home_win.astype(int).to_numpy()[validation_market],
                validation_qb_feature_stat[validation_market],
                validation.loc[validation_market, "market_home_prob"].astype(float).to_numpy(),
            )
            elo_qb_feature_market_weight, _ = select_market_weight(
                validation.home_win.astype(int).to_numpy()[validation_market],
                validation_elo_qb_feature_stat[validation_market],
                validation.loc[validation_market, "market_home_prob"].astype(float).to_numpy(),
            )
            opponent_adjusted_market_weight, _ = select_market_weight(
                validation.home_win.astype(int).to_numpy()[validation_market],
                validation_opponent_adjusted_stat[validation_market],
                validation.loc[validation_market, "market_home_prob"].astype(float).to_numpy(),
            )
        pipe = new_pipeline()
        pipe.fit(history[FEATURES], history.home_win.astype(int))
        statistical = pipe.predict_proba(games[FEATURES])[:, 1]
        elo_feature_pipe = new_pipeline()
        elo_feature_pipe.fit(history[ELO_FEATURES], history.home_win.astype(int))
        elo_feature_statistical = elo_feature_pipe.predict_proba(games[ELO_FEATURES])[:, 1]
        qb_feature_pipe = new_pipeline()
        qb_feature_pipe.fit(history[QB_FEATURES], history.home_win.astype(int))
        qb_feature_statistical = qb_feature_pipe.predict_proba(games[QB_FEATURES])[:, 1]
        elo_qb_feature_pipe = new_pipeline()
        elo_qb_feature_pipe.fit(history[ELO_QB_FEATURES], history.home_win.astype(int))
        elo_qb_feature_statistical = elo_qb_feature_pipe.predict_proba(games[ELO_QB_FEATURES])[:, 1]
        opponent_adjusted_pipe = new_pipeline()
        opponent_adjusted_pipe.fit(history[OPPONENT_ADJUSTED_FEATURES], history.home_win.astype(int))
        opponent_adjusted_statistical = opponent_adjusted_pipe.predict_proba(games[OPPONENT_ADJUSTED_FEATURES])[:, 1]
        for (_, game), stat, elo_feature_stat, qb_feature_stat, elo_qb_feature_stat, opponent_adjusted_stat in zip(games.iterrows(), statistical, elo_feature_statistical, qb_feature_statistical, elo_qb_feature_statistical, opponent_adjusted_statistical):
            market = None if pd.isna(game.market_home_prob) else float(game.market_home_prob)
            blended = float(stat) if market is None else market_weight * market + (1 - market_weight) * float(stat)
            elo_feature_blended = float(elo_feature_stat) if market is None else elo_feature_market_weight * market + (1 - elo_feature_market_weight) * float(elo_feature_stat)
            qb_feature_blended = float(qb_feature_stat) if market is None else qb_feature_market_weight * market + (1 - qb_feature_market_weight) * float(qb_feature_stat)
            elo_qb_feature_blended = float(elo_qb_feature_stat) if market is None else elo_qb_feature_market_weight * market + (1 - elo_qb_feature_market_weight) * float(elo_qb_feature_stat)
            opponent_adjusted_blended = float(opponent_adjusted_stat) if market is None else opponent_adjusted_market_weight * market + (1 - opponent_adjusted_market_weight) * float(opponent_adjusted_stat)
            predictions.append({
                "gameId": str(game.game_id),
                "season": int(game.season),
                "week": int(game.week),
                "awayTeam": str(game.away_team),
                "homeTeam": str(game.home_team),
                "homeWin": int(game.home_win),
                "statisticalHomeProbability": float(stat),
                "statisticalWithEloHomeProbability": float(elo_feature_stat),
                "statisticalWithQbHomeProbability": float(qb_feature_stat),
                "statisticalWithEloAndQbHomeProbability": float(elo_qb_feature_stat),
                "statisticalWithOpponentAdjustmentHomeProbability": float(opponent_adjusted_stat),
                "marketHomeProbability": market,
                "blendedHomeProbability": blended,
                "blendedWithEloHomeProbability": elo_feature_blended,
                "blendedWithQbHomeProbability": qb_feature_blended,
                "blendedWithEloAndQbHomeProbability": elo_qb_feature_blended,
                "blendedWithOpponentAdjustmentHomeProbability": opponent_adjusted_blended,
                "marketWeight": market_weight,
                "marketWeightWithElo": elo_feature_market_weight,
                "marketWeightWithQb": qb_feature_market_weight,
                "marketWeightWithEloAndQb": elo_qb_feature_market_weight,
                "marketWeightWithOpponentAdjustment": opponent_adjusted_market_weight,
                "trainingGames": int(len(history)),
                "validationSeason": validation_season,
            })
    if not predictions:
        raise RuntimeError("No weeks could be backtested; add more historical seasons")
    results = pd.DataFrame(predictions)
    results["eloHomeProbability"] = results.gameId.map(elo_probabilities(features, from_season, to_season))
    if results.eloHomeProbability.isna().any():
        raise RuntimeError("Elo backtest did not produce a probability for every scored game")
    actual = results.homeWin.to_numpy()
    market_results = results[results.marketHomeProbability.notna()]
    market_actual = market_results.homeWin.to_numpy()
    return {
        "fromSeason": from_season,
        "toSeason": to_season,
        "method": "expanding weekly retraining with prior-season blend selection",
        "marketOddsNote": "Historical nflverse moneyline timing may differ from the odds available at a live prediction run.",
        "skippedWeeks": skipped_weeks,
        "allGames": {
            "statistical": probability_metrics(actual, results.statisticalHomeProbability.to_numpy()),
            "statisticalWithElo": probability_metrics(actual, results.statisticalWithEloHomeProbability.to_numpy()),
            "statisticalWithQb": probability_metrics(actual, results.statisticalWithQbHomeProbability.to_numpy()),
            "statisticalWithEloAndQb": probability_metrics(actual, results.statisticalWithEloAndQbHomeProbability.to_numpy()),
            "statisticalWithOpponentAdjustment": probability_metrics(actual, results.statisticalWithOpponentAdjustmentHomeProbability.to_numpy()),
            "elo": probability_metrics(actual, results.eloHomeProbability.to_numpy()),
            "blended": probability_metrics(actual, results.blendedHomeProbability.to_numpy()),
            "blendedWithElo": probability_metrics(actual, results.blendedWithEloHomeProbability.to_numpy()),
            "blendedWithQb": probability_metrics(actual, results.blendedWithQbHomeProbability.to_numpy()),
            "blendedWithEloAndQb": probability_metrics(actual, results.blendedWithEloAndQbHomeProbability.to_numpy()),
            "blendedWithOpponentAdjustment": probability_metrics(actual, results.blendedWithOpponentAdjustmentHomeProbability.to_numpy()),
        },
        "marketGames": {
            "statistical": probability_metrics(market_actual, market_results.statisticalHomeProbability.to_numpy()),
            "statisticalWithElo": probability_metrics(market_actual, market_results.statisticalWithEloHomeProbability.to_numpy()),
            "statisticalWithQb": probability_metrics(market_actual, market_results.statisticalWithQbHomeProbability.to_numpy()),
            "statisticalWithEloAndQb": probability_metrics(market_actual, market_results.statisticalWithEloAndQbHomeProbability.to_numpy()),
            "statisticalWithOpponentAdjustment": probability_metrics(market_actual, market_results.statisticalWithOpponentAdjustmentHomeProbability.to_numpy()),
            "elo": probability_metrics(market_actual, market_results.eloHomeProbability.to_numpy()),
            "market": probability_metrics(market_actual, market_results.marketHomeProbability.to_numpy()),
            "blended": probability_metrics(market_actual, market_results.blendedHomeProbability.to_numpy()),
            "blendedWithElo": probability_metrics(market_actual, market_results.blendedWithEloHomeProbability.to_numpy()),
            "blendedWithQb": probability_metrics(market_actual, market_results.blendedWithQbHomeProbability.to_numpy()),
            "blendedWithEloAndQb": probability_metrics(market_actual, market_results.blendedWithEloAndQbHomeProbability.to_numpy()),
            "blendedWithOpponentAdjustment": probability_metrics(market_actual, market_results.blendedWithOpponentAdjustmentHomeProbability.to_numpy()),
        },
        "predictions": predictions,
    }


def confidence(prob: float) -> str:
    strength = max(prob, 1 - prob)
    return "high" if strength >= 0.72 else "medium" if strength >= 0.62 else "low"


def load_stadiums() -> dict[str, dict[str, str]]:
    try:
        with urllib.request.urlopen(STADIUMS_URL, timeout=12) as response:
            rows = csv.DictReader(response.read().decode().splitlines())
            return {row["stadium_id"]: row for row in rows if row.get("stadium_id")}
    except Exception:
        return {}


def weather_for_game(game: pd.Series, stadiums: dict[str, dict[str, str]]) -> dict[str, Any] | None:
    if str(game.get("roof") or "").lower() in {"dome", "closed"}:
        return None
    stadium = stadiums.get(str(game.get("stadium_id") or ""))
    if not stadium:
        return None
    try:
        lat, lon = float(stadium["lat"]), float(stadium["lon"])
        tz_name = stadium.get("tz") or "America/New_York"
        eastern = datetime.fromisoformat(f"{game.gameday}T{game.get('gametime') or '13:00'}").replace(tzinfo=ZoneInfo("America/New_York"))
        local = eastern.astimezone(ZoneInfo(tz_name))
        params = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m", "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": tz_name, "forecast_days": 16})
        with urllib.request.urlopen(f"{OPEN_METEO}?{params}", timeout=12) as response:
            data = json.loads(response.read().decode())
        times = data.get("hourly", {}).get("time", [])
        if not times:
            return None
        target = local.replace(minute=0, second=0, microsecond=0)
        idx = min(range(len(times)), key=lambda i: abs(datetime.fromisoformat(times[i]) - target.replace(tzinfo=None)))
        h = data["hourly"]
        temp, precip, wind, gust = h["temperature_2m"][idx], h["precipitation_probability"][idx], h["wind_speed_10m"][idx], h["wind_gusts_10m"][idx]
        parts = [f"{round(temp)}°F", f"wind {round(wind)} mph"]
        if precip is not None and precip >= 30:
            parts.append(f"{round(precip)}% precip")
        return {"summary": " · ".join(parts), "temperatureF": round(float(temp), 1), "precipitationPct": round(float(precip)) if precip is not None else None, "windMph": round(float(wind), 1), "gustMph": round(float(gust), 1)}
    except Exception:
        return None


def recent_team_news(team: str, now: datetime) -> list[dict[str, str]]:
    query = f'{TEAM_NAMES.get(team, team)} NFL injury starter'
    try:
        url = f"{GOOGLE_NEWS}?{urllib.parse.urlencode({'q': query, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})}"
        request = urllib.request.Request(url, headers={"User-Agent": "nfl-picks/1.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            root = ET.fromstring(response.read())
        recent = []
        for item in root.findall("./channel/item"):
            published = item.findtext("pubDate")
            if not published or parsedate_to_datetime(published).astimezone(timezone.utc) < now - timedelta(days=7):
                continue
            title, url = item.findtext("title"), item.findtext("link")
            if not title or not url:
                continue
            source = item.find("source")
            recent.append({"title": title, "source": source.text if source is not None and source.text else "Google News", "url": url})
            if len(recent) == 2:
                break
        return recent
    except Exception:
        return []


def news_for_teams(teams: set[str], now: datetime) -> dict[str, list[dict[str, str]]]:
    return {team: recent_team_news(team, now) for team in teams}


def baseline_for(season: int, week: int) -> tuple[Path, dict[str, Any] | None]:
    path = SNAPSHOT_DIR / f"{season}-{week:02d}-initial.json"
    try:
        return path, json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return path, None


def game_flags(game: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    flags = []
    weather = game.get("weather")
    if weather:
        if (weather.get("windMph") or 0) >= 20 or (weather.get("gustMph") or 0) >= 30:
            flags.append("Weather: high wind may materially affect the game")
        if (weather.get("precipitationPct") or 0) >= 60:
            flags.append("Weather: meaningful precipitation risk")
        if (weather.get("temperatureF") or 100) <= 25:
            flags.append("Weather: very cold conditions")
    if not baseline:
        return flags
    old = next((g for g in baseline.get("games", []) if g.get("gameId") == game["gameId"]), None)
    if not old:
        return flags
    om, nm = old.get("marketHomeProbability"), game.get("marketHomeProbability")
    if om is not None and nm is not None:
        move = abs(float(nm) - float(om))
        if move >= 0.05:
            flags.append(f"Market moved {round(move * 100)} pts since initial snapshot")
        if (float(om) >= 0.5) != (float(nm) >= 0.5):
            flags.append("Market favorite flipped since initial snapshot")
    if old.get("homeQb") and game.get("homeQb") and old["homeQb"] != game["homeQb"]:
        flags.append(f"Home QB changed: {old['homeQb']} → {game['homeQb']}")
    if old.get("awayQb") and game.get("awayQb") and old["awayQb"] != game["awayQb"]:
        flags.append(f"Away QB changed: {old['awayQb']} → {game['awayQb']}")
    return flags


def kickoff_at(row: pd.Series) -> datetime:
    eastern = datetime.fromisoformat(f"{row.gameday}T{row.get('gametime') or '13:00'}").replace(tzinfo=ZoneInfo("America/New_York"))
    return eastern.astimezone(timezone.utc)


def upcoming_week_and_kickoff(schedule: pd.DataFrame, season: int, now: datetime) -> tuple[int, datetime]:
    regular = schedule[(schedule.season == season) & (schedule.game_type == "REG")].copy()
    regular["_kickoff"] = regular.apply(kickoff_at, axis=1)
    upcoming = regular[regular._kickoff > now]
    if upcoming.empty:
        raise RuntimeError(f"No upcoming regular-season games found for {season}")
    week = int(upcoming.sort_values("_kickoff").iloc[0].week)
    games = regular[regular.week == week]
    return week, min(games._kickoff)


def refresh_stage(now: datetime, first_kickoff: datetime) -> str | None:
    minutes_until_kickoff = (first_kickoff - now).total_seconds() / 60
    if 60 <= minutes_until_kickoff <= 120:
        return "final"
    eastern = now.astimezone(ZoneInfo("America/New_York"))
    if eastern.weekday() == 2 and 10 <= eastern.hour < 18:
        return "initial"
    return None


def preflight(force: bool, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    season = current_nfl_season()
    schedule = load_schedule([season])
    week, first = upcoming_week_and_kickoff(schedule, season, now)
    if force:
        return "forced"
    stage = refresh_stage(now, first)
    if stage is None:
        if now >= first:
            print(f"Week {week} has already kicked off; leaving published picks unchanged")
        else:
            print(f"Week {week} refresh is not due yet; leaving published picks unchanged")
        raise SystemExit(0)
    return stage


def generate(refresh_weather: bool, refresh_news: bool, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    season = current_nfl_season()
    seasons = list(range(START_SEASON, season + 1))
    schedule, pbp = load_inputs(seasons)
    features = build_game_features(schedule, aggregate_team_weeks(pbp))
    model = train_model(features, season)
    week, _ = upcoming_week_and_kickoff(schedule, season, now)
    current = features[(features.season == season) & (features.week == week)].copy()
    current["_kickoff"] = current.apply(kickoff_at, axis=1)
    current = current[current._kickoff > now].sort_values(["gameday", "gametime", "game_id"])
    stat_probs = model.pipeline.predict_proba(current[FEATURES])[:, 1]
    stadiums = load_stadiums() if refresh_weather else {}
    team_news = news_for_teams(set(current.home_team) | set(current.away_team), now) if refresh_news else {}
    snapshot_path, baseline = baseline_for(season, week)
    games = []
    for (_, row), stat_home in zip(current.iterrows(), stat_probs):
        market_home = None if pd.isna(row.get("market_home_prob")) else float(row.market_home_prob)
        final_home = float(stat_home) if market_home is None else model.market_weight * market_home + (1 - model.market_weight) * float(stat_home)
        pick = str(row.home_team) if final_home >= 0.5 else str(row.away_team)
        game = {"gameId": str(row.game_id), "awayTeam": str(row.away_team), "homeTeam": str(row.home_team), "gameday": str(row.gameday), "gametime": None if pd.isna(row.get("gametime")) else str(row.gametime), "stadium": None if pd.isna(row.get("stadium")) else str(row.stadium), "roof": None if pd.isna(row.get("roof")) else str(row.roof), "pick": pick, "winProbability": round(max(final_home, 1 - final_home), 4), "homeWinProbability": round(final_home, 4), "statisticalHomeProbability": round(float(stat_home), 4), "marketHomeProbability": None if market_home is None else round(market_home, 4), "spreadLine": None if pd.isna(row.get("spread_line")) else float(row.spread_line), "confidence": confidence(final_home), "homeQb": None if pd.isna(row.get("home_qb_name")) else str(row.home_qb_name), "awayQb": None if pd.isna(row.get("away_qb_name")) else str(row.away_qb_name), "weather": weather_for_game(row, stadiums) if refresh_weather else None, "news": (team_news.get(str(row.away_team), []) + team_news.get(str(row.home_team), []))[:2], "flags": []}
        game["flags"] = game_flags(game, baseline)
        games.append(game)
    payload = {"season": season, "week": week, "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "firstGame": min((g["gameday"] for g in games), default=None), "model": {"trainingSeasons": model.training_seasons, "validationSeason": model.validation_season, "marketWeight": round(model.market_weight, 2), "validationAccuracy": round(model.validation_accuracy, 4), "validationBrier": round(model.validation_brier, 4), "marketBrier": None if model.market_brier is None else round(model.market_brier, 4)}, "games": games}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if baseline is None:
        snapshot_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--backtest", action="store_true", help="Replay completed games using expanding weekly training windows")
    parser.add_argument("--capture-odds", action="store_true", help="Save the current multi-book moneyline snapshot when a game-day capture is due")
    parser.add_argument("--from-season", type=int, help="First season to score during a backtest")
    parser.add_argument("--to-season", type=int, help="Last season to score during a backtest")
    parser.add_argument("--backtest-output", type=Path, default=BACKTEST_OUTPUT)
    args = parser.parse_args()
    if args.capture_odds:
        capture_odds()
        return
    if args.backtest:
        latest_completed = current_nfl_season() - 1
        to_season = args.to_season or latest_completed
        from_season = args.from_season or max(START_SEASON + 2, to_season - 2)
        if from_season > to_season:
            parser.error("--from-season must not be after --to-season")
        schedule, pbp = load_inputs(list(range(START_SEASON, to_season + 1)))
        features = build_game_features(schedule, aggregate_team_weeks(pbp))
        features = build_qb_features(features, aggregate_qb_games(pbp))
        results = backtest(features, from_season, to_season)
        args.backtest_output.parent.mkdir(parents=True, exist_ok=True)
        args.backtest_output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Backtested {results['allGames']['statistical']['games']} games from {from_season}-{to_season}")
        print(f"Elo Brier: {results['marketGames']['elo']['brier']:.4f}; statistical Brier: {results['marketGames']['statistical']['brier']:.4f}")
        print(f"Statistical with Elo Brier: {results['marketGames']['statisticalWithElo']['brier']:.4f}")
        print(f"Statistical with QB Brier: {results['marketGames']['statisticalWithQb']['brier']:.4f}")
        print(f"Statistical with Elo and QB Brier: {results['marketGames']['statisticalWithEloAndQb']['brier']:.4f}")
        print(f"Statistical with opponent adjustment Brier: {results['marketGames']['statisticalWithOpponentAdjustment']['brier']:.4f}")
        print(f"Blended Brier: {results['marketGames']['blended']['brier']:.4f}; market Brier: {results['marketGames']['market']['brier']:.4f}")
        print(f"Wrote {args.backtest_output}")
        return
    stage = preflight(args.force)
    refresh_current_conditions = stage in {"final", "forced"}
    payload = generate(refresh_current_conditions and not args.no_weather, refresh_current_conditions)
    print(f"Generated {len(payload['games'])} picks for {payload['season']} week {payload['week']}")
    print(f"Market weight: {payload['model']['marketWeight']:.0%}")


if __name__ == "__main__":
    main()
