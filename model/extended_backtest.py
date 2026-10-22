"""Validate the fixed four-game challenger outside its development seasons."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import experiments
import matchup_experiments as matchup
import predict
import snapshots

ARTIFACTS = predict.ROOT / "model" / "artifacts"
PBP_COLUMNS = ["season", "season_type", "week", "posteam", "defteam", "play_type", "epa", "success", "pass", "rush"]
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20261010


def download_inputs() -> None:
    """Download the source files needed by both validation periods."""
    sources = {ARTIFACTS / "extended-schedule.csv": predict.SCHEDULE_URL}
    for season in [*range(2016, 2023), 2025, 2026]:
        sources[ARTIFACTS / f"extended-pbp-{season}.csv.gz"] = predict.PBP_URL.format(season=season) + ".gz"
    for season in range(2016, 2027):
        sources[ARTIFACTS / f"matchup-stats-{season}.csv"] = matchup.STATS_URL.format(season=season)
    for season in [*range(2016, 2023), 2026]:
        sources[ARTIFACTS / f"extended-injuries-{season}.csv"] = experiments.INJURY_URL.format(season=season)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for path, url in sources.items():
        request = urllib.request.Request(url, headers={"User-Agent": "nfl-picks/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            path.write_bytes(response.read())
        print(f"Downloaded {path.name}", flush=True)


def clear_feature_caches() -> None:
    """Remove only this runner's generated feature caches."""
    for pattern in ("extended-team-weeks-*.json", "extended-base-*.pkl"):
        for path in ARTIFACTS.glob(pattern):
            path.unlink()


def aggregate_inputs(seasons: list[int]) -> dict:
    """Read compressed play-by-play one season at a time and retain weekly aggregates."""
    team_weeks = {}
    for season in seasons:
        path = ARTIFACTS / f"extended-team-weeks-{season}.json"
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            pbp = pd.read_csv(ARTIFACTS / f"extended-pbp-{season}.csv.gz", usecols=PBP_COLUMNS, low_memory=False)
            for column in ("posteam", "defteam"):
                pbp[column] = pbp[column].replace(experiments.ELO_TEAM_ALIASES)
            aggregate = predict.aggregate_team_weeks(pbp)
            rows = [{"season": s, "week": w, "team": t, **values} for (s, w, t), values in aggregate.items()]
            path.write_text(json.dumps(rows) + "\n")
        for row in rows:
            team_weeks[(row["season"], row["week"], row["team"])] = {metric: row[metric] for metric in predict.METRICS}
        print(f"Aggregated {season}: {len(rows)} team weeks", flush=True)
    return team_weeks


def build_base(schedule: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Rebuild the live strength features, tuned Elo columns, and injury features."""
    cache = ARTIFACTS / f"extended-base-{min(seasons)}-{max(seasons)}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    selected = experiments.aliased_elo_schedule(schedule[schedule.season.isin(seasons)], experiments.ELO_TEAM_ALIASES)
    features = predict.build_game_features(selected, aggregate_inputs(seasons))
    warm = schedule[(schedule.season >= experiments.ELO_WARMUP_FIRST_SEASON) & (schedule.season <= max(seasons))]
    for column, params in experiments.ELO_GRID_PARAMS.items():
        elo = experiments.pregame_elo_v2(warm, k=params["k"], home_advantage=params["homeAdvantage"], season_regression=params["seasonRegression"], mov_cap=experiments.ELO_MOV_CAP, signed_mov=True, aliases=experiments.ELO_TEAM_ALIASES)
        features[column] = features.game_id.astype(str).map(elo)
    injuries = []
    for season in seasons:
        path = ARTIFACTS / f"extended-injuries-{season}.csv"
        if not path.exists():
            if season == 2025:
                continue
            raise FileNotFoundError(path)
        frame = experiments.normalize_injury_frame(pd.read_csv(path, low_memory=False), season)
        frame["team"] = frame.team.replace(experiments.ELO_TEAM_ALIASES)
        injuries.append(frame)
    features = experiments.build_injury_features(features, pd.concat(injuries, ignore_index=True))
    pd.to_pickle(features, cache)
    return features


def bootstrap_difference(first: np.ndarray, second: np.ndarray, actual: np.ndarray, blocks: np.ndarray, repeats: int = BOOTSTRAP_REPEATS, seed: int = BOOTSTRAP_SEED, family_size: int = 1) -> dict:
    """Resample complete season-week blocks and preserve paired game comparisons."""
    first, second, actual, blocks = (np.asarray(value) for value in (first, second, actual, blocks))
    if not len(actual) or any(len(value) != len(actual) for value in (first, second, blocks)):
        raise ValueError("Paired samples must have equal, nonzero lengths")
    if family_size < 1:
        raise ValueError("The comparison family must contain at least one test")
    if not np.isfinite(np.column_stack((first, second, actual))).all():
        raise ValueError("Paired samples must contain finite probabilities and outcomes")
    first, second = np.clip(first, experiments.PROBABILITY_FLOOR, 1 - experiments.PROBABILITY_FLOOR), np.clip(second, experiments.PROBABILITY_FLOOR, 1 - experiments.PROBABILITY_FLOOR)
    differences = pd.DataFrame({
        "block": blocks,
        "brier": (first - actual) ** 2 - (second - actual) ** 2,
        "logLoss": -(actual * np.log(first) + (1 - actual) * np.log1p(-first)) + actual * np.log(second) + (1 - actual) * np.log1p(-second),
    })
    grouped = differences.groupby("block").agg(brier=("brier", "sum"), logLoss=("logLoss", "sum"), games=("brier", "size"))
    result = {"games": len(actual), "weekBlocks": len(grouped), "repeats": repeats}
    if len(grouped) < 2:
        return {**result, "status": "insufficient week blocks", "brier": {"difference": float(differences.brier.mean()), "ci95": None}, "logLoss": {"difference": float(differences.logLoss.mean()), "ci95": None}}
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(grouped), size=(repeats, len(grouped)))
    counts = grouped.games.to_numpy()[indexes].sum(axis=1)
    for metric in ("brier", "logLoss"):
        bootstrap = grouped[metric].to_numpy()[indexes].sum(axis=1) / counts
        result[metric] = {"difference": float(differences[metric].mean()), "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist()}
        if family_size > 1:
            tail = 0.025 / family_size
            result[metric]["ciAdjusted"] = np.quantile(bootstrap, [tail, 1 - tail]).tolist()
            result[metric]["adjustedConfidence"] = 1 - 0.05 / family_size
    result["status"] = "both intervals below zero" if all(result[metric]["ci95"][1] < 0 for metric in ("brier", "logLoss")) else "no confirmed improvement"
    if family_size > 1:
        result["adjustedStatus"] = "both adjusted intervals below zero" if all(result[metric]["ciAdjusted"][1] < 0 for metric in ("brier", "logLoss")) else "no confirmed improvement after adjustment"
    return result


def score_period(predictions: list[dict]) -> dict:
    rows = [row for row in predictions if row["marketHomeProbability"] is not None]
    actual = np.array([row["homeWin"] for row in rows])
    market = np.array([row["marketHomeProbability"] for row in rows])
    blocks = np.array([row["season"] * 100 + row["week"] for row in rows])
    probabilities = {"market": market}
    for name in ("baseline", "elo_tuned_injury", "team8", "team4"):
        for field, suffix in (("probability", "raw"), ("blendedProbability", "blend")):
            probabilities[f"{name}_{suffix}"] = np.array([row["candidates"][name][field] for row in rows])
    comparisons = [("team4_blend", "market"), ("team4_raw", "team8_raw"), ("team4_raw", "elo_tuned_injury_raw"), ("team4_blend", "elo_tuned_injury_blend")]
    result = {
        "games": len(rows),
        "metrics": {name: predict.probability_metrics(actual, values) for name, values in probabilities.items()},
        "comparisons": {f"{first}_vs_{second}": bootstrap_difference(probabilities[first], probabilities[second], actual, blocks) for first, second in comparisons},
    }
    for name, values in probabilities.items():
        metrics = result["metrics"][name]
        correct = int(((values >= 0.5) == actual).sum())
        metrics["correct"] = correct
        metrics["incorrect"] = len(rows) - correct
    return result


def score_same_time(predictions: list[dict]) -> dict:
    """Compare reconstructed challenger probabilities to the stored pregame odds."""
    by_id = {row["gameId"]: row for row in predictions}
    frame = snapshots.snapshot_frame(snapshots.load_snapshots(predict.ODDS_SNAPSHOT_DIR / "2026"))
    rows = [(by_id[game["game_id"]], game[snapshots.SAME_TIME]) for game in frame.to_dict("records") if game["game_id"] in by_id]
    if not rows:
        return {"games": 0}
    actual = np.array([row["homeWin"] for row, _ in rows])
    market = np.array([probability for _, probability in rows])
    raw = np.array([row["candidates"]["team4"]["probability"] for row, _ in rows])
    blend = np.array([row["candidates"]["team4"]["blendedProbability"] for row, _ in rows])
    blocks = np.array([row["season"] * 100 + row["week"] for row, _ in rows])
    return {"games": len(rows), "rawVersusSameTime": bootstrap_difference(raw, market, actual, blocks), "blendVersusSameTime": bootstrap_difference(blend, market, actual, blocks), "note": "The reconstructed blend uses closing odds. Only the raw statistical challenger is independent of those target-game closing odds."}


def write_report(result: dict, path: Path) -> None:
    lines = ["# Extended four-game backtest", "", "The four-game specification stays fixed. No new feature or parameter is selected from these results.", "Training expands by week. Blend selection uses only the latest prior season.", "", "## Results", "", "Lower Brier and log loss are better.", "", "| Period | Games | Four-game blend accuracy | Blend Brier | Market Brier | Difference | 95% interval |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    displayed_periods = {**result["periods"], "All periods, descriptive": result["combinedAll"], "2026 without undated injuries": result["noCurrentInjuries"]}
    for name, period in displayed_periods.items():
        blend, market = period["metrics"]["team4_blend"], period["metrics"]["market"]
        comparison = period["comparisons"]["team4_blend_vs_market"]["brier"]
        interval = comparison["ci95"]
        formatted = "insufficient blocks" if interval is None else f"[{interval[0]:+.5f}, {interval[1]:+.5f}]"
        lines.append(f"| {name} | {period['games']} | {blend['accuracy']:.1%} | {blend['brier']:.5f} | {market['brier']:.5f} | {comparison['difference']:+.5f} | {formatted} |")
    lines += ["", "A negative difference favors the four-game blend. An interval that crosses zero does not establish an improvement.", "", "## Paired comparisons", "", "Intervals use 10,000 paired bootstrap samples of complete season-week blocks.", "They preserve unequal game counts across weeks.", "", "| Period | Comparison | Brier difference | 95% interval | Log-loss difference | 95% interval |", "| --- | --- | ---: | --- | ---: | --- |"]
    for name, period in result["periods"].items():
        for comparison_name, comparison in period["comparisons"].items():
            brier, log = comparison["brier"], comparison["logLoss"]
            brier_ci, log_ci = brier["ci95"], log["ci95"]
            brier_text = "insufficient blocks" if brier_ci is None else f"[{brier_ci[0]:+.5f}, {brier_ci[1]:+.5f}]"
            log_text = "insufficient blocks" if log_ci is None else f"[{log_ci[0]:+.5f}, {log_ci[1]:+.5f}]"
            lines.append(f"| {name} | {comparison_name} | {brier['difference']:+.5f} | {brier_text} | {log['difference']:+.5f} | {log_text} |")
    lines += ["", "## Design and limits", "", "- The development period is 2023–2025. Its predictions come from the original experiment report without refitting.", "- The earlier-period replay scores 2018–2022 and starts training in 2016.", "- The 2026 replay preserves the original 2021–2025 training features and adds completed 2026 games.", "- The 2026 sample is small and covers only games completed when the source files were downloaded.", "- The 2026 injury file lacks report timestamps. Its features cannot be verified as available before kickoff.", "- The sensitivity replay sets both injury differences to zero for 2026. Historical training and blend selection stay unchanged.", "- The earlier replay uses more training history than the original development replay. Its market and model controls share that history.", "- Franchise aliases preserve team histories across relocations in the earlier replay.", "- The challenger adds four-game team rates to the existing Elo, injury, and eight-game play-by-play features.", "- The blend uses recorded closing odds. This test cannot establish a tradable advantage against earlier odds.", "- Bootstrap intervals describe sampling uncertainty. They do not correct for earlier exploration across multiple candidates.", "- The combined result includes the development sample. It is descriptive, not an independent validation result.", "- These are reconstructed historical predictions. The challenger was not published before these games.", "", "## Same-time archive", "", f"Matched {result['sameTime2026']['games']} completed 2026 games to stored pregame odds.", "The JSON report compares raw statistical probabilities with those odds.", "The reconstructed blend uses closing odds, so its same-time comparison is not a prospective market test.", "", "## Reproduce", "", "First run the original matchup experiment to create its feature cache and development report.", "The runner uses compressed play-by-play, injury reports, and weekly team stats cached in `model/artifacts/`.", "The JSON report records hashes of the source files, original feature cache, and original experiment report.", "", "```bash", "uv run --project model python model/extended_backtest.py --download-inputs", "```", "", "Use `--refresh-features` to rebuild caches after a feature-builder change.", "", "## Sources", "", "- [nflverse play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp)", "- [nflverse team stats](https://nflreadr.nflverse.com/reference/load_team_stats.html)", "- [nflverse injuries](https://github.com/nflverse/nflverse-data/releases/tag/injuries)", "- [nflverse schedule](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-inputs", action="store_true", help="Refresh source files and rebuild derived caches")
    parser.add_argument("--refresh-features", action="store_true", help="Rebuild derived caches from existing source files")
    args = parser.parse_args()
    if args.download_inputs:
        download_inputs()
    if args.download_inputs or args.refresh_features:
        clear_feature_caches()
    schedule = pd.read_csv(ARTIFACTS / "extended-schedule.csv", low_memory=False)
    stats = pd.concat([pd.read_csv(ARTIFACTS / f"matchup-stats-{year}.csv") for year in range(2016, 2027)], ignore_index=True)
    stats["team"] = stats.team.replace(experiments.ELO_TEAM_ALIASES)
    stats["opponent_team"] = stats.opponent_team.replace(experiments.ELO_TEAM_ALIASES)
    selected_candidates = [candidate for candidate in matchup.candidates() if candidate.name in {"baseline", "elo_tuned_injury", "team8", "team4"}]
    earlier = matchup.build_features(build_base(schedule, list(range(2016, 2023))), schedule, stats)
    print("Scoring 2018–2022", flush=True)
    past = experiments.run_experiments(earlier, 2018, 2022, selected_candidates)
    original_features = pd.read_pickle(ARTIFACTS / "features-v6-2021-2025.pkl")["features"]
    current = build_base(schedule, [2025, 2026])
    current = pd.concat([original_features, current[current.season == 2026]], ignore_index=True)
    current = matchup.build_features(current, schedule, stats)
    print("Scoring completed 2026 games", flush=True)
    present = experiments.run_experiments(current, 2026, 2026, selected_candidates)
    without_injuries = current.copy()
    without_injuries.loc[without_injuries.season == 2026, ["diff_injury_load", "diff_qb_injury"]] = 0.0
    print("Scoring 2026 without undated injury features", flush=True)
    sensitivity = experiments.run_experiments(without_injuries, 2026, 2026, selected_candidates)
    original = json.loads((ARTIFACTS / "matchup-experiments.json").read_text())
    result = {
        "generatedAt": datetime.now().astimezone().isoformat(),
        "periods": {"2018–2022 validation": score_period(past["predictions"]), "2023–2025 development": score_period(original["predictions"]), "2026 validation": score_period(present["predictions"])},
        "bySeason": {str(season): score_period([row for row in rows if row["season"] == season]) for rows in (past["predictions"], original["predictions"], present["predictions"]) for season in sorted({row["season"] for row in rows})},
        "sameTime2026": score_same_time(present["predictions"]),
        "earlierReplay": past,
        "currentReplay": present,
        "combinedAll": score_period(past["predictions"] + original["predictions"] + present["predictions"]),
        "combinedValidation": score_period(past["predictions"] + present["predictions"]),
        "noCurrentInjuries": score_period(sensitivity["predictions"]),
        "injurySensitivityReplay": sensitivity,
    }
    inputs = sorted(set(ARTIFACTS.glob("extended-pbp-*.csv.gz")) | set(ARTIFACTS.glob("extended-injuries-*.csv")) | set(ARTIFACTS.glob("matchup-stats-*.csv")) | {ARTIFACTS / "extended-schedule.csv", ARTIFACTS / "features-v6-2021-2025.pkl", ARTIFACTS / "matchup-experiments.json"})
    result["sourceHashes"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    (ARTIFACTS / "extended-backtest.json").write_text(json.dumps(result, indent=2) + "\n")
    report = predict.ROOT / "docs" / "extended-backtest.md"
    write_report(result, report)
    for name, period in result["periods"].items():
        print(name, json.dumps(period["comparisons"]["team4_blend_vs_market"]), flush=True)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
