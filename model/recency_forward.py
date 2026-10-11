"""Capture and score future predictions from the frozen recency candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import extended_backtest as extended
import predict
import recency_experiments as recency

SPEC = predict.ROOT / "docs" / "recency-candidate.json"
ARCHIVE = predict.ROOT / "data" / "model-predictions" / "recency"
REPORT = predict.ROOT / "data" / "model-evaluation" / "recency.json"
PBP_COLUMNS = ["game_id", *extended.PBP_COLUMNS]


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Archive timestamps must include a time zone")
    return parsed


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specification() -> dict:
    spec = json.loads(SPEC.read_text())
    for name, expected in spec["sourceHashes"].items():
        if digest(predict.ROOT / "model" / name) != expected:
            raise RuntimeError(f"Frozen prediction source changed: {name}")
    return spec


def load_team_weeks(schedule: pd.DataFrame, season: int, week: int) -> tuple[dict, dict]:
    """Read one season at a time and exclude target-week plays."""
    weeks, hashes = {}, {}
    for year in sorted(schedule.season.unique()):
        path = predict.ROOT / "model" / "artifacts" / f"forward-pbp-{year}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or year == season:
            import urllib.request

            request = urllib.request.Request(predict.PBP_URL.format(season=year) + ".gz", headers={"User-Agent": "nfl-picks/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                path.write_bytes(response.read())
        hashes[path.name] = digest(path)
        pbp = pd.read_csv(path, usecols=PBP_COLUMNS, low_memory=False)
        available = schedule[schedule.home_score.notna() & schedule.away_score.notna()]
        available = available[(available.season < season) | ((available.season == season) & (available.week < week))]
        pbp = pbp[pbp.game_id.isin(available.game_id)]
        weeks.update(predict.aggregate_team_weeks(pbp))
    return weeks, hashes


def forecast(schedule: pd.DataFrame, team_weeks: dict, season: int, week: int, game_ids: set[str]) -> dict:
    """Exclude the entire target week from training and team histories."""
    history = schedule[(schedule.season < season) | ((schedule.season == season) & (schedule.week <= week))].copy()
    target = (history.season == season) & (history.week == week)
    history.loc[target, ["home_score", "away_score"]] = np.nan
    eligible = {(int(row.season), int(row.week), str(team)) for row in history[~target].itertuples() if pd.notna(row.home_score) and pd.notna(row.away_score) for team in (row.home_team, row.away_team)}
    filtered = {key: value for key, value in team_weeks.items() if key in eligible}
    base = predict.build_game_features(history, filtered)
    features, _ = recency.build_recency(base, filtered)
    candidate = features.copy()
    for metric in predict.METRICS:
        candidate[f"diff_{metric}"] = candidate[f"core_exp4_diff_{metric}"]
    models = {"baseline": (features, predict.train_model(features, season)), "candidate": (candidate, predict.train_model(candidate, season))}
    result = {game_id: {} for game_id in game_ids}
    for name, (frame, model) in models.items():
        current = frame[frame.game_id.isin(game_ids)]
        if set(current.game_id) != game_ids:
            raise ValueError("A capture game has no model features")
        probabilities = model.pipeline.predict_proba(current[predict.FEATURES])[:, 1]
        for (_, row), probability in zip(current.iterrows(), probabilities):
            result[str(row.game_id)][name] = {
                "probability": float(probability), "marketWeight": model.market_weight,
                "validationSeason": model.validation_season, "trainingSeasons": model.training_seasons,
                "features": {name: None if pd.isna(row[name]) else float(row[name]) for name in predict.FEATURES},
            }
    return result


def save_capture(path: Path, payload: dict, spec: dict) -> bool:
    """Preserve the first archive and refuse predictions saved after kickoff."""
    now = timestamp(payload["savedAt"])
    if now < timestamp(spec["frozenAt"]) or any(not timestamp(spec["frozenAt"]) < timestamp(game["kickoffAt"]) or not now < timestamp(game["kickoffAt"]) for game in payload["games"]):
        raise ValueError("Predictions must be saved before kickoff and after the candidate freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as output:
            output.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except FileExistsError:
        return False
    return True


def remaining_game_day(schedule: pd.DataFrame, season: int, now: datetime) -> tuple[str, datetime, pd.DataFrame]:
    """Allow a missed morning capture to save games that have not started."""
    day, _, games = predict.upcoming_game_day(schedule, season, now)
    games = games[games.apply(predict.kickoff_at, axis=1) > now].copy()
    return day, min(games.apply(predict.kickoff_at, axis=1)), games


def capture() -> None:
    spec = specification()
    now = datetime.now(timezone.utc)
    season = predict.current_nfl_season(now)
    schedule = predict.load_schedule(list(range(spec["trainingStartSeason"], season + 1)))
    regular = schedule[schedule.game_type == "REG"]
    if regular.empty or not any(predict.kickoff_at(row) > now for _, row in regular[regular.season == season].iterrows()):
        print("No upcoming regular-season games")
        return
    day, kickoff, games = remaining_game_day(regular, season, now)
    path = ARCHIVE / str(season) / f"{day}.json"
    if path.exists() or not predict.odds_capture_due(now, kickoff):
        print(f"Candidate capture is already saved or not due for {day}")
        return
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Set THE_ODDS_API_KEY before capturing candidate forecasts")
    all_forecasts, source_hashes = {}, {}
    for week, group in games.groupby("week"):
        weeks, hashes = load_team_weeks(schedule, season, int(week))
        source_hashes.update(hashes)
        all_forecasts.update(forecast(schedule, weeks, season, int(week), set(group.game_id)))
    events = predict.fetch_current_odds(api_key)
    captured_at = datetime.now(timezone.utc)
    payload = predict.normalize_odds_snapshot(events, games, season, day, kickoff, captured_at)
    payload.update({"candidate": spec["candidate"], "frozenAt": spec["frozenAt"], "specHash": digest(SPEC), "sourceHashes": source_hashes,
                    "predictionSourceHashes": {**spec["sourceHashes"], Path(__file__).name: digest(Path(__file__))},
                    "scheduleHash": hashlib.sha256(schedule.to_csv(index=False).encode()).hexdigest()})
    for game in payload["games"]:
        game["week"] = int(games.loc[games.game_id == game["gameId"], "week"].iloc[0])
        game.update(all_forecasts[game["gameId"]])
        for name in ("baseline", "candidate"):
            entry = game[name]
            entry["blendedProbability"] = entry["marketWeight"] * game["marketHomeProbability"] + (1 - entry["marketWeight"]) * entry["probability"]
    payload["savedAt"] = datetime.now(timezone.utc).isoformat()
    saved = save_capture(path, payload, spec)
    print(f"Saved {len(payload['games']) if saved else 0} candidate forecasts for {day}")


def evaluate(archives: list[dict], schedule: pd.DataFrame, spec: dict, spec_hash: str) -> dict:
    """Score only valid, completed future games from this frozen specification."""
    results = schedule.set_index("game_id")
    rows, seen = [], set()
    for archive in sorted(archives, key=lambda item: item["savedAt"]):
        if archive.get("specHash") != spec_hash or any(archive.get("predictionSourceHashes", {}).get(name) != expected for name, expected in spec["sourceHashes"].items()):
            continue
        saved, odds_at = timestamp(archive["savedAt"]), timestamp(archive["capturedAt"])
        for game in archive["games"]:
            game_id = game["gameId"]
            kickoff = timestamp(game["kickoffAt"])
            if game_id in seen or game_id not in results.index or not timestamp(spec["frozenAt"]) < kickoff or not timestamp(spec["frozenAt"]) <= odds_at <= saved < kickoff:
                continue
            row = results.loc[game_id]
            if row.game_type != "REG" or pd.isna(row.home_score) or pd.isna(row.away_score) or row.home_score == row.away_score:
                continue
            if saved >= predict.kickoff_at(row):
                continue
            probabilities = {"baseline": game["baseline"]["probability"], "candidate": game["candidate"]["probability"],
                             "baselineBlend": game["baseline"]["blendedProbability"], "candidateBlend": game["candidate"]["blendedProbability"], "market": game["marketHomeProbability"]}
            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
                continue
            rows.append({"gameId": game_id, "actual": int(row.home_score > row.away_score), "block": int(row.season) * 100 + int(row.week), **probabilities})
            seen.add(game_id)
    minimum = spec["evaluation"]["minimumCompletedGames"]
    report = {"candidate": spec["candidate"], "frozenAt": spec["frozenAt"], "specHash": spec_hash, "games": len(rows), "minimumCompletedGames": minimum,
              "status": f"insufficient games: {len(rows)} of {minimum}", "metrics": {}, "comparisons": {}, "scoredGameIds": [row["gameId"] for row in rows]}
    if not rows:
        return report
    actual, blocks = np.array([row["actual"] for row in rows]), np.array([row["block"] for row in rows])
    arrays = {name: np.array([row[name] for row in rows]) for name in ("baseline", "candidate", "baselineBlend", "candidateBlend", "market")}
    report["metrics"] = {name: predict.probability_metrics(actual, values) for name, values in arrays.items()}
    for first, second in (("candidate", "baseline"), ("candidate", "market"), ("candidateBlend", "baselineBlend"), ("candidateBlend", "market")):
        report["comparisons"][f"{first}_vs_{second}"] = extended.bootstrap_difference(arrays[first], arrays[second], actual, blocks)
    if len(rows) >= minimum:
        report["status"] = report["comparisons"]["candidate_vs_baseline"]["status"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.capture:
        capture()
        return
    spec = specification()
    archives = [json.loads(path.read_text()) for path in ARCHIVE.glob("*/*.json")]
    schedule = predict.load_schedule(sorted({int(item["season"]) for item in archives})) if archives else pd.DataFrame(columns=["game_id"])
    report = evaluate(archives, schedule, spec, digest(SPEC))
    if REPORT.exists():
        previous = json.loads(REPORT.read_text())
        previous.pop("generatedAt", None)
        if previous == report:
            print(report["status"])
            return
    report["generatedAt"] = datetime.now(timezone.utc).isoformat()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"])


if __name__ == "__main__":
    main()
