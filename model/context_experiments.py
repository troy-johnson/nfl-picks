"""Test venue, travel, and QB age features with chronological weekly training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import experiments
import predict

ARTIFACTS = predict.ROOT / "model" / "artifacts"
VENUE_FEATURES = ("venue_enclosure", "venue_artificial", "venue_altitude_km", "division_game")
TRAVEL_FEATURES = ("travel_distance_diff", "eastward_shift_diff", "early_body_clock_diff", "short_rest_diff", "bye_rest_diff", "road_streak_diff")
AGE_FEATURES = ("qb_age_diff", "qb_age_squared_diff", "qb_over_35_diff", "qb_under_24_diff")
TIMEZONE_ALIASES = {"Europe/Munich": "Europe/Berlin", "Europe/Frankfurt": "Europe/Berlin"}
INPUT_URLS = {
    "schedule": predict.SCHEDULE_URL,
    "stadiums": predict.STADIUMS_URL,
    "players": "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv",
}


def distance_km(origin: dict, destination: dict) -> float:
    """Great-circle distance between two stadiums."""
    lat1, lat2 = math.radians(float(origin["lat"])), math.radians(float(destination["lat"]))
    dlat = lat2 - lat1
    dlon = math.radians(float(destination["lon"]) - float(origin["lon"]))
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1.0, value)))


def travel_context(origin: dict | None, venue: dict | None, kickoff: datetime) -> tuple[float, float, float]:
    """Use the team's home stadium as its travel origin, not an assumed itinerary."""
    if origin is None or venue is None:
        return (np.nan, np.nan, np.nan)
    origin_zone = TIMEZONE_ALIASES.get(str(origin["tz"]), str(origin["tz"]))
    venue_zone = TIMEZONE_ALIASES.get(str(venue["tz"]), str(venue["tz"]))
    origin_time = kickoff.astimezone(ZoneInfo(origin_zone))
    venue_time = kickoff.astimezone(ZoneInfo(venue_zone))
    shift = (venue_time.utcoffset() - origin_time.utcoffset()).total_seconds() / 3600
    hour = origin_time.hour + origin_time.minute / 60
    return distance_km(origin, venue) / 1000, max(0.0, shift), max(0.0, 12 - hour)


def build_context(features: pd.DataFrame, schedule: pd.DataFrame, stadiums: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build context before each game day. Update team stadiums only after that day."""
    venues = stadiums.set_index("stadium_id").to_dict("index")
    births = pd.to_datetime(players.set_index("gsis_id")["birth_date"], errors="coerce").to_dict()
    bases: dict[str, dict] = {}
    road_streaks: dict[str, int] = defaultdict(int)
    enclosed: set[str] = set()
    rows = []
    games = schedule[schedule.game_type.isin(["REG", "POST"])].sort_values(["gameday", "game_id"])
    targets = features.set_index("game_id")
    previous_season = None
    for day, day_games in games.groupby("gameday", sort=True):
        season = int(day_games.iloc[0].season)
        if season != previous_season:
            road_streaks.clear()
        previous_season = season
        for _, game in day_games.iterrows():
            if game.game_id not in targets.index:
                continue
            venue = venues.get(str(game.stadium_id))
            home, away = str(game.home_team), str(game.away_team)
            neutral = str(game.location).lower() == "neutral"
            kickoff = predict.kickoff_at(game)
            home_travel = travel_context(bases.get(home), venue, kickoff)
            away_travel = travel_context(bases.get(away), venue, kickoff)
            if not neutral and venue is not None:
                home_travel = (0.0, 0.0, 0.0)
            ages = {}
            for side in ("home", "away"):
                player_id = targets.loc[game.game_id, f"{side}_recent_qb_id"]
                birth = births.get(player_id, pd.NaT)
                ages[side] = (pd.Timestamp(day) - birth).days / 365.2425 if pd.notna(birth) else np.nan
            home_age, away_age = ages["home"], ages["away"]
            surface = str(game.surface).lower()
            row = {
                "game_id": game.game_id,
                "venue_enclosure": np.nan if venue is None else float(str(venue["roof_type"]).lower() == "dome" or str(game.stadium_id) in enclosed),
                "venue_artificial": np.nan if pd.isna(game.surface) else float(surface != "grass"),
                "venue_altitude_km": np.nan if venue is None else float(venue["altitude"]) / 1000,
                "division_game": float(game.div_game),
                "travel_distance_diff": home_travel[0] - away_travel[0],
                "eastward_shift_diff": home_travel[1] - away_travel[1],
                "early_body_clock_diff": home_travel[2] - away_travel[2],
                "short_rest_diff": float(game.home_rest <= 5) - float(game.away_rest <= 5),
                "bye_rest_diff": float(game.home_rest >= 12) - float(game.away_rest >= 12),
                "road_streak_diff": float(min(road_streaks[home], 3)) - float(min(road_streaks[away], 3)),
                "qb_age_diff": home_age - away_age,
                "qb_age_squared_diff": (home_age - 28) ** 2 - (away_age - 28) ** 2,
                "qb_over_35_diff": max(0.0, home_age - 35) - max(0.0, away_age - 35) if np.isfinite(home_age + away_age) else np.nan,
                "qb_under_24_diff": max(0.0, 24 - home_age) - max(0.0, 24 - away_age) if np.isfinite(home_age + away_age) else np.nan,
                "away_travel_km": away_travel[0] * 1000,
                "away_early_body_clock": away_travel[2],
                "home_qb_age": home_age,
                "away_qb_age": away_age,
                "observed_wind": game.wind,
                "observed_temp": game.temp,
            }
            rows.append(row)
        for _, game in day_games.iterrows():
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            home, away = str(game.home_team), str(game.away_team)
            venue = venues.get(str(game.stadium_id))
            neutral = str(game.location).lower() == "neutral"
            road_streaks[away] += 1
            road_streaks[home] = road_streaks[home] + 1 if neutral else 0
            if not neutral and venue is not None:
                bases[home] = venue
            if str(game.roof).lower() in {"dome", "closed"}:
                enclosed.add(str(game.stadium_id))
    context = pd.DataFrame(rows)
    merged = features.merge(context, on="game_id", validate="one_to_one")
    if len(merged) != len(features):
        raise ValueError("The context join lost feature rows")
    coverage = {column: int(merged[column].isna().sum()) for column in VENUE_FEATURES + TRAVEL_FEATURES + AGE_FEATURES}
    return merged, coverage


def candidates() -> list[experiments.Candidate]:
    base = tuple(predict.FEATURES)
    strong = experiments.ELO_TUNED_INJURY_FEATURES
    qb = tuple(predict.QB_FEATURES)
    return [
        experiments.CANDIDATES["baseline"],
        experiments.CANDIDATES["elo_tuned_injury"],
        experiments.CANDIDATES["qb"],
        experiments.Candidate("venue", "Baseline plus venue and division context", base + VENUE_FEATURES, experiments.logistic_pipeline),
        experiments.Candidate("travel", "Baseline plus travel and rest context", base + TRAVEL_FEATURES, experiments.logistic_pipeline),
        experiments.Candidate("context", "Baseline plus venue and travel context", base + VENUE_FEATURES + TRAVEL_FEATURES, experiments.logistic_pipeline),
        experiments.Candidate("strong_context", "Tuned Elo and injury features plus venue and travel context", strong + VENUE_FEATURES + TRAVEL_FEATURES, experiments.logistic_pipeline),
        experiments.Candidate("qb_age", "Prior-game QB features plus age terms", qb + AGE_FEATURES, experiments.logistic_pipeline),
        experiments.CANDIDATES["market_stack"],
        experiments.Candidate("market_context", "Market stacking plus venue and travel context", tuple(experiments.INJURY_FEATURES) + ("market_logit",) + VENUE_FEATURES + TRAVEL_FEATURES, experiments.logistic_pipeline, uses_market=True),
    ]


def comparisons(result: dict) -> dict:
    pairs = {"venue": "baseline", "travel": "baseline", "context": "baseline", "strong_context": "elo_tuned_injury", "qb_age": "qb", "market_context": "market_stack"}
    frame = pd.DataFrame(result["predictions"])
    frame = frame[frame.marketHomeProbability.notna()]
    answer = {}
    for candidate, control in pairs.items():
        answer[candidate] = {}
        for period, group in [("pooled", frame), *[(str(s), g) for s, g in frame.groupby("season")]]:
            entry = {"games": len(group), "control": control}
            for field in ("probability", "blendedProbability"):
                first = group.candidates.map(lambda c: c[candidate][field]).to_numpy()
                second = group.candidates.map(lambda c: c[control][field]).to_numpy()
                entry[field] = experiments.paired_difference(first, second, group.homeWin.to_numpy(), (group.season * 100 + group.week).to_numpy())
            answer[candidate][period] = entry
    return answer


def splits(result: dict, features: pd.DataFrame) -> list[dict]:
    frame = pd.DataFrame(result["predictions"]).merge(features, left_on="gameId", right_on="game_id", validate="one_to_one", suffixes=("", "_feature"))
    frame = frame[frame.marketHomeProbability.notna()].copy()
    frame["baseline_probability"] = frame.candidates.map(lambda c: c["baseline"]["probability"])
    frame["baseline_blend"] = frame.candidates.map(lambda c: c["baseline"]["blendedProbability"])
    frame["confidence"] = np.maximum(frame.baseline_blend, 1 - frame.baseline_blend)
    groups = {
        "market_home_favorite": frame.marketHomeProbability >= 0.5,
        "market_away_favorite": frame.marketHomeProbability < 0.5,
        "low_confidence": frame.confidence < 0.62,
        "medium_confidence": (frame.confidence >= 0.62) & (frame.confidence < 0.72),
        "high_confidence": frame.confidence >= 0.72,
        "early_season": frame.week <= 4,
        "late_season": frame.week >= 14,
        "enclosed_venue": frame.venue_enclosure == 1,
        "open_venue": frame.venue_enclosure == 0,
        "artificial_surface": frame.venue_artificial == 1,
        "grass_surface": frame.venue_artificial == 0,
        "high_altitude": frame.venue_altitude_km >= 1,
        "long_away_trip": frame.away_travel_km >= 2500,
        "early_away_body_clock": frame.away_early_body_clock > 0,
        "short_rest": (frame.home_rest <= 5) | (frame.away_rest <= 5),
        "neutral_site": frame.neutral_site == 1,
        "qb_age_35_plus": (frame.home_qb_age >= 35) | (frame.away_qb_age >= 35),
        "observed_wind_15_plus": frame.observed_wind >= 15,
        "observed_temp_40_or_less": frame.observed_temp <= 40,
    }
    rows = []
    for label, mask in groups.items():
        group = frame[mask]
        if group.empty:
            continue
        actual = group.homeWin.to_numpy()
        rows.append({"split": label, "games": len(group), **{name: predict.probability_metrics(actual, group[column].to_numpy()) for name, column in [("raw", "baseline_probability"), ("blend", "baseline_blend"), ("market", "marketHomeProbability")]}})
    return rows


def write_report(result: dict, path: Path) -> None:
    summary = result["summary"]
    lines = ["# Venue, travel, and QB age tests", "", f"Scored {summary['marketGames']} regular-season games from 2023–2025 with expanding weekly training.", "", "Each week uses earlier games for training and the prior season for blend selection.", "These are retrospective tests. The 2025 results are a separate temporal check, not a new untouched holdout.", "", "## Probability scores", "", "Lower Brier and log loss are better.", "", "| Candidate | Accuracy | Raw Brier | Raw log loss | Blended Brier |", "| --- | ---: | ---: | ---: | ---: |"]
    market = summary["market"]
    lines.append(f"| Recorded closing market | {market['accuracy']:.1%} | {market['brier']:.5f} | {market['logLoss']:.5f} | {market['brier']:.5f} |")
    for name, value in summary["candidates"].items():
        raw, blend = value["marketGames"]["raw"], value["marketGames"]["blended"]
        lines.append(f"| {name} | {raw['accuracy']:.1%} | {raw['brier']:.5f} | {raw['logLoss']:.5f} | {blend['brier']:.5f} |")
    improvements = []
    confirmed = []
    for name, periods in result["featureComparisons"].items():
        value = periods["pooled"]["probability"]
        if value["brier"] < 0 and value["logLoss"] < 0:
            improvements.append(name)
        if value["brier"] < -2 * value["brierWeekBlockStandardError"] and value["logLoss"] < -2 * value["logLossWeekBlockStandardError"]:
            confirmed.append(name)
    lines += ["", "## Findings", "", f"Additions that improve both raw scores: {', '.join(improvements) or 'none'}.", f"Additions that improve both raw scores beyond two week-block standard errors: {', '.join(confirmed) or 'none'}.", "These exploratory comparisons need confirmation with later games before a model change.", "", "## Added-feature comparisons", "", "Negative differences favor the added features. SE is the standard error across week blocks.", "", "| Candidate vs control | Period | Brier difference | SE | Log-loss difference | SE |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, periods in result["featureComparisons"].items():
        for period, entry in periods.items():
            value = entry["probability"]
            lines.append(f"| {name} vs {entry['control']} | {period} | {value['brier']:+.5f} | {value['brierWeekBlockStandardError']:.5f} | {value['logLoss']:+.5f} | {value['logLossWeekBlockStandardError']:.5f} |")
    lines += ["", "## Blended comparisons", "", "| Candidate vs control | Brier difference | SE | Log-loss difference | SE |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, periods in result["featureComparisons"].items():
        entry = periods["pooled"]
        value = entry["blendedProbability"]
        lines.append(f"| {name} vs {entry['control']} | {value['brier']:+.5f} | {value['brierWeekBlockStandardError']:.5f} | {value['logLoss']:+.5f} | {value['logLossWeekBlockStandardError']:.5f} |")
    lines += ["", "## Error splits", "", "These overlapping groups are descriptive. Small groups do not establish a predictive effect."]
    if summary["candidates"]["baseline"]["marketWeight"]["min"] == 1:
        lines += ["The baseline blend selects 100% market in every replay week, so its split scores equal the market scores."]
    lines += ["", "| Split | Games | Raw Brier | Blended Brier | Market Brier | Blend accuracy |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["splits"]:
        lines.append(f"| {row['split']} | {row['games']} | {row['raw']['brier']:.5f} | {row['blend']['brier']:.5f} | {row['market']['brier']:.5f} | {row['blend']['accuracy']:.1%} |")
    lines += ["", "## Limits", "", "- Weather splits use recorded game weather. No weather feature enters a candidate model.", "- A weather model needs historical forecasts saved before kickoff.", "- Travel distance uses the team's latest prior home stadium. Actual travel itineraries are unknown.", "- Time-zone features use the kickoff date, including daylight saving time.", "- Roof enclosure uses stadium metadata and earlier roof observations, not the target game's roof status.", "- Surface uses the historical schedule, rather than today's stadium surface.", "- Age terms use the prior game's primary QB and birthdate. They do not use the target game's recorded starter.", "- QB age terms test an age adjustment. They do not estimate a complete player aging curve.", "- Recorded closing odds cannot establish an advantage against odds available when a live pick was published.", "- Multiple exploratory comparisons increase the chance of a false finding.", "", "## Data coverage", "", "Missing feature values use training-set medians.", "", "```json", json.dumps(result["missingFeatures"], indent=2), "```", "", "## Sources", "", "- [nflverse schedules](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)", "- [Stadium coordinates and time zones](https://github.com/greerreNFL/Stadiums)", "- [nflverse player birthdates](https://nflreadr.nflverse.com/reference/load_players.html)", "", "## Reproduce", "", "Build the existing feature cache if it is missing.", "Then run the context experiments.", "", "```bash", "uv run --project model python model/experiments.py --candidates baseline --from-season 2023 --to-season 2025", "uv run --project model python model/context_experiments.py --download-inputs", "```", "", "The JSON report records input hashes. Source files can change after this run.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ARTIFACTS / "features-v6-2021-2025.pkl")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "context-experiments.json")
    parser.add_argument("--report", type=Path, default=predict.ROOT / "docs" / "context-experiments.md")
    parser.add_argument("--download-inputs", action="store_true")
    args = parser.parse_args()
    if args.download_inputs:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for name, url in INPUT_URLS.items():
            request = urllib.request.Request(url, headers={"User-Agent": "nfl-picks/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                (ARTIFACTS / f"context-{name}.csv").write_bytes(response.read())
    features = pd.read_pickle(args.cache)["features"]
    schedule = pd.read_csv(ARTIFACTS / "context-schedule.csv", low_memory=False)
    stadiums = pd.read_csv(ARTIFACTS / "context-stadiums.csv")
    players = pd.read_csv(ARTIFACTS / "context-players.csv", low_memory=False)
    features, coverage = build_context(features, schedule, stadiums, players)
    print("Missing features:", json.dumps(coverage), flush=True)
    result = experiments.run_experiments(features, 2023, 2025, candidates())
    result["inputs"] = {name: {"url": url, "sha256": hashlib.sha256((ARTIFACTS / f"context-{name}.csv").read_bytes()).hexdigest()} for name, url in INPUT_URLS.items()}
    result["featureCacheSha256"] = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    result["featureComparisons"] = comparisons(result)
    result["splits"] = splits(result, features)
    result["missingFeatures"] = coverage
    result["generatedAt"] = datetime.now().astimezone().isoformat()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    write_report(result, args.report)
    print(experiments.format_table(result["summary"]))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
