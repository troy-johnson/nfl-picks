from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "data" / "current.json"
SNAPSHOT_DIR = ROOT / "public" / "data" / "snapshots"
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"
STADIUMS_URL = "https://raw.githubusercontent.com/greerreNFL/Stadiums/main/data/stadiums.csv"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
ROLLING_GAMES = 8
START_SEASON = 2021
METRICS = ["off_epa", "off_success", "pass_epa", "rush_epa", "def_epa_allowed", "def_success_allowed", "def_pass_epa_allowed", "def_rush_epa_allowed"]
FEATURES = [f"diff_{m}" for m in METRICS] + ["rest_diff", "neutral_site"]


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


def load_schedule(seasons: list[int]) -> pd.DataFrame:
    cols = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "away_score", "home_team", "home_score", "location", "away_rest", "home_rest", "away_moneyline", "home_moneyline", "spread_line", "roof", "home_qb_name", "away_qb_name", "stadium_id", "stadium"]
    df = pd.read_csv(SCHEDULE_URL, usecols=cols, low_memory=False)
    return df[df.season.isin(seasons)].copy()


def load_inputs(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = load_schedule(seasons)
    cols = ["season", "season_type", "week", "posteam", "defteam", "play_type", "epa", "success", "pass", "rush"]
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


def history_average(history: deque[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {m: np.nan for m in METRICS}
    return {m: float(np.mean([x[m] for x in history if not pd.isna(x.get(m))])) if any(not pd.isna(x.get(m)) for x in history) else np.nan for m in METRICS}


def build_game_features(schedule: pd.DataFrame, team_weeks: dict[tuple[int, int, str], dict[str, float]]) -> pd.DataFrame:
    games = schedule[schedule.game_type == "REG"].copy()
    games["_date"] = pd.to_datetime(games.gameday, errors="coerce")
    games = games.sort_values(["season", "week", "_date", "game_id"])
    histories: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=ROLLING_GAMES))
    rows = []
    for _, game in games.iterrows():
        home, away = str(game.home_team), str(game.away_team)
        ha, aa = history_average(histories[home]), history_average(histories[away])
        record = {k: game.get(k) for k in games.columns if k != "_date"}
        for metric in METRICS:
            record[f"diff_{metric}"] = ha[metric] - aa[metric]
        record["rest_diff"] = float(game.home_rest) - float(game.away_rest) if pd.notna(game.home_rest) and pd.notna(game.away_rest) else 0.0
        record["neutral_site"] = 1.0 if str(game.get("location", "")).lower() == "neutral" else 0.0
        record["market_home_prob"] = market_home_probability(game)
        if pd.notna(game.home_score) and pd.notna(game.away_score) and float(game.home_score) != float(game.away_score):
            record["home_win"] = int(float(game.home_score) > float(game.away_score))
        else:
            record["home_win"] = np.nan
        rows.append(record)
        season, week = int(game.season), int(game.week)
        for team in (home, away):
            stats = team_weeks.get((season, week, team))
            if stats:
                histories[team].append(stats)
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


def train_model(features: pd.DataFrame, current_season: int) -> ModelResult:
    completed = features[features.home_win.notna()].copy()
    seasons = sorted(int(s) for s in completed.season.unique())
    validation_season = current_season - 1 if current_season - 1 in seasons else seasons[-1]
    train, validation = completed[completed.season < validation_season], completed[completed.season == validation_season]
    if len(train) < 200:
        raise RuntimeError("Not enough historical games to train")
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, C=0.6))])
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
        scored = [(float(w), float(brier_score_loss(y_m, w * market + (1 - w) * stat_m))) for w in np.linspace(0, 1, 21)]
        weight, blend_brier = min(scored, key=lambda x: x[1])
    pipe.fit(completed[FEATURES], completed.home_win.astype(int))
    return ModelResult(pipe, validation_season, weight, accuracy, blend_brier, market_brier, seasons)


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


def first_week_and_kickoff(schedule: pd.DataFrame, season: int) -> tuple[int, datetime]:
    regular = schedule[(schedule.season == season) & (schedule.game_type == "REG")].copy()
    unplayed = regular[regular.home_score.isna()]
    if unplayed.empty:
        raise RuntimeError(f"No unplayed regular-season games found for {season}")
    week = int(unplayed.week.min())
    games = regular[regular.week == week]
    kickoffs = []
    for _, row in games.iterrows():
        eastern = datetime.fromisoformat(f"{row.gameday}T{row.get('gametime') or '13:00'}").replace(tzinfo=ZoneInfo("America/New_York"))
        kickoffs.append(eastern.astimezone(timezone.utc))
    return week, min(kickoffs)


def preflight(force: bool) -> None:
    season = current_nfl_season()
    schedule = load_schedule([season])
    week, first = first_week_and_kickoff(schedule, season)
    if datetime.now(timezone.utc) >= first and not force:
        print(f"Week {week} has already kicked off; leaving published picks unchanged")
        raise SystemExit(0)


def generate(refresh_weather: bool) -> dict[str, Any]:
    season = current_nfl_season()
    seasons = list(range(START_SEASON, season + 1))
    schedule, pbp = load_inputs(seasons)
    features = build_game_features(schedule, aggregate_team_weeks(pbp))
    model = train_model(features, season)
    current = features[(features.season == season) & features.home_win.isna()].copy()
    week = int(current.week.min())
    current = current[current.week == week].sort_values(["gameday", "gametime", "game_id"])
    stat_probs = model.pipeline.predict_proba(current[FEATURES])[:, 1]
    stadiums = load_stadiums() if refresh_weather else {}
    snapshot_path, baseline = baseline_for(season, week)
    games = []
    for (_, row), stat_home in zip(current.iterrows(), stat_probs):
        market_home = None if pd.isna(row.get("market_home_prob")) else float(row.market_home_prob)
        final_home = float(stat_home) if market_home is None else model.market_weight * market_home + (1 - model.market_weight) * float(stat_home)
        pick = str(row.home_team) if final_home >= 0.5 else str(row.away_team)
        game = {"gameId": str(row.game_id), "awayTeam": str(row.away_team), "homeTeam": str(row.home_team), "gameday": str(row.gameday), "gametime": None if pd.isna(row.get("gametime")) else str(row.gametime), "stadium": None if pd.isna(row.get("stadium")) else str(row.stadium), "roof": None if pd.isna(row.get("roof")) else str(row.roof), "pick": pick, "winProbability": round(max(final_home, 1 - final_home), 4), "homeWinProbability": round(final_home, 4), "statisticalHomeProbability": round(float(stat_home), 4), "marketHomeProbability": None if market_home is None else round(market_home, 4), "spreadLine": None if pd.isna(row.get("spread_line")) else float(row.spread_line), "confidence": confidence(final_home), "homeQb": None if pd.isna(row.get("home_qb_name")) else str(row.home_qb_name), "awayQb": None if pd.isna(row.get("away_qb_name")) else str(row.away_qb_name), "weather": weather_for_game(row, stadiums) if refresh_weather else None, "flags": []}
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
    args = parser.parse_args()
    preflight(args.force)
    payload = generate(not args.no_weather)
    print(f"Generated {len(payload['games'])} picks for {payload['season']} week {payload['week']}")
    print(f"Market weight: {payload['model']['marketWeight']:.0%}")


if __name__ == "__main__":
    main()
