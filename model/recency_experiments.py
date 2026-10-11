"""Replace the core rolling averages while holding other model inputs fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import experiments
import extended_backtest as extended
import predict

ARTIFACTS = predict.ROOT / "model" / "artifacts"
VARIANTS = {"core4": (4, None), "core8": (8, None), "core12": (12, None), "core_exp4": (24, 4.0)}


def average(history: list[dict], window: int, half_life: float | None) -> dict[str, float]:
    """Average game-level metrics with equal or exponentially decreasing weights."""
    recent = history[-window:]
    weights = np.array([1.0 if half_life is None else 0.5 ** ((len(recent) - i - 1) / half_life) for i in range(len(recent))])
    values = {}
    for metric in predict.METRICS:
        samples = np.array([row.get(metric, np.nan) for row in recent], dtype=float)
        valid = np.isfinite(samples)
        values[metric] = float(np.average(samples[valid], weights=weights[valid])) if valid.any() else np.nan
    return values


def build_recency(features: pd.DataFrame, team_weeks: dict) -> tuple[pd.DataFrame, float]:
    """Freeze team histories for each target week and replace all eight rolling metrics."""
    histories = defaultdict(lambda: deque(maxlen=24))
    rows = []
    for (season, week), games in features.sort_values(["season", "week", "game_id"]).groupby(["season", "week"], sort=True):
        for _, game in games.iterrows():
            row = {"game_id": game.game_id}
            for name, parameters in VARIANTS.items():
                home = average(list(histories[str(game.home_team)]), *parameters)
                away = average(list(histories[str(game.away_team)]), *parameters)
                for metric in predict.METRICS:
                    row[f"{name}_diff_{metric}"] = home[metric] - away[metric]
            rows.append(row)
        for _, game in games.iterrows():
            for side in ("home", "away"):
                team = str(game[f"{side}_team"])
                stats = team_weeks.get((int(season), int(week), team))
                if stats:
                    histories[team].append(stats)
    merged = features.merge(pd.DataFrame(rows), on="game_id", validate="one_to_one")
    differences = []
    for metric in predict.METRICS:
        original = merged[f"diff_{metric}"].to_numpy(dtype=float)
        rebuilt = merged[f"core8_diff_{metric}"].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(original), np.isnan(rebuilt)):
            raise ValueError(f"Eight-game control has different missing values: {metric}")
        valid = np.isfinite(original) & np.isfinite(rebuilt)
        differences.extend(np.abs(original[valid] - rebuilt[valid]).tolist())
        if not np.allclose(original, rebuilt, atol=1e-10, rtol=0, equal_nan=True):
            raise ValueError(f"Eight-game control differs from frozen inputs: {metric}")
    return merged, max(differences, default=0.0)


def candidates() -> list[experiments.Candidate]:
    result = [experiments.CANDIDATES["baseline"], experiments.CANDIDATES["elo_tuned_injury"]]
    for name in ("core4", "core12", "core_exp4"):
        metrics = tuple(f"{name}_diff_{metric}" for metric in predict.METRICS)
        for prefix, extra in (("baseline", ()), ("strong", (experiments.ELO_TUNED_COLUMN, "diff_injury_load", "diff_qb_injury"))):
            result.append(experiments.Candidate(f"{prefix}_{name}", f"Replace the core rolling averages with {name}", metrics + ("rest_diff", "neutral_site") + extra, experiments.logistic_pipeline))
    return result


def score(rows: list[dict]) -> dict:
    rows = [row for row in rows if row["marketHomeProbability"] is not None]
    actual = np.array([row["homeWin"] for row in rows])
    market = np.array([row["marketHomeProbability"] for row in rows])
    blocks = np.array([row["season"] * 100 + row["week"] for row in rows])
    result = {"games": len(rows), "market": predict.probability_metrics(actual, market), "candidates": {}}
    for candidate in candidates():
        name = candidate.name
        control = "baseline" if name.startswith("baseline") else "elo_tuned_injury"
        raw = np.array([row["candidates"][name]["probability"] for row in rows])
        blend = np.array([row["candidates"][name]["blendedProbability"] for row in rows])
        control_raw = np.array([row["candidates"][control]["probability"] for row in rows])
        control_blend = np.array([row["candidates"][control]["blendedProbability"] for row in rows])
        result["candidates"][name] = {
            "control": control,
            "raw": predict.probability_metrics(actual, raw),
            "blend": predict.probability_metrics(actual, blend),
            "rawVsControl": extended.bootstrap_difference(raw, control_raw, actual, blocks, family_size=6),
            "blendVsControl": extended.bootstrap_difference(blend, control_blend, actual, blocks),
            "blendVsMarket": extended.bootstrap_difference(blend, market, actual, blocks),
        }
    return result


def write_report(result: dict, path: Path) -> None:
    combined = result["periods"]["All periods"]
    lines = ["# Core recency tests", "", f"Scored {combined['games']} games from 2018–2026 with recorded market odds.", "Each candidate replaces the eight rolling strength features. No additional team-rate features enter the model.", "Rest, neutral-site context, estimator, regularization, Elo, and injury definitions stay fixed within each control pair.", "Each week trains on earlier weeks. The prior season selects the market blend.", "", "## Combined scores", "", "Lower Brier and log loss are better.", "", "| Candidate | Raw accuracy | Raw Brier | Raw log loss | Blended Brier | Difference vs market | 95% Brier interval |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    market = combined["market"]
    lines.append(f"| Recorded closing market | {market['accuracy']:.1%} | {market['brier']:.5f} | {market['logLoss']:.5f} | {market['brier']:.5f} | 0 | — |")
    for name, entry in combined["candidates"].items():
        raw, blend = entry["raw"], entry["blend"]
        delta = entry["blendVsMarket"]["brier"]
        lower, upper = delta["ci95"]
        lines.append(f"| {name} | {raw['accuracy']:.1%} | {raw['brier']:.5f} | {raw['logLoss']:.5f} | {blend['brier']:.5f} | {delta['difference']:+.5f} | [{lower:+.5f}, {upper:+.5f}] |")
    lines += ["", "## Adjustment for six variants", "", "These Bonferroni-adjusted bootstrap intervals use 99.17% coverage for each raw-score comparison.", "They allow for the six new variants in this experiment. They do not allow for all earlier model exploration.", "", "| Candidate | Raw Brier difference | Adjusted interval | Raw log-loss difference | Adjusted interval |", "| --- | ---: | --- | ---: | --- |"]
    for name, entry in combined["candidates"].items():
        if name in ("baseline", "elo_tuned_injury"):
            continue
        value = entry["rawVsControl"]
        brier, log = value["brier"], value["logLoss"]
        lo, hi = brier["ciAdjusted"]
        loglo, loghi = log["ciAdjusted"]
        lines.append(f"| {name} | {brier['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {log['difference']:+.5f} | [{loglo:+.5f}, {loghi:+.5f}] |")
    lines += ["", "## Comparisons with the eight-game control", "", "Negative differences favor the candidate. Intervals use 10,000 paired samples of complete season-week blocks.", "", "| Period | Candidate | Raw Brier difference | 95% interval | Raw log-loss difference | 95% interval |", "| --- | --- | ---: | --- | ---: | --- |"]
    for period, scores in result["periods"].items():
        for name, entry in scores["candidates"].items():
            if name in ("baseline", "elo_tuned_injury"):
                continue
            difference = entry["rawVsControl"]
            brier, log = difference["brier"], difference["logLoss"]
            lo, hi = brier["ci95"]
            loglo, loghi = log["ci95"]
            lines.append(f"| {period} | {name} | {brier['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {log['difference']:+.5f} | [{loglo:+.5f}, {loghi:+.5f}] |")
    lines += ["", "## Control checks", "", f"Maximum difference between rebuilt and original eight-game inputs: {result['controlMaxDifference']:.3g}.", "All variants use the same game-level EPA and success-rate definitions.", "The exponentially weighted variant uses a four-game half-life and the latest 24 completed regular-season games.", "Equal-weight variants use four, eight, or twelve games.", "", "## Limits", "", "- Earlier-period training begins in 2016. Recent-period training begins in 2021, matching the prior experiment setup.", "- This experiment changes the averaging window, not the underlying game metrics or feature count.", "- Elo selection uses the same prior-game grid for every variant. Market weights can differ because validation scores differ.", "- The 2026 injury reports lack timestamps. The JSON report includes a sensitivity replay with current-season injury differences set to zero.", "- The 2026 sample has only five week blocks. Its confidence intervals can be unstable.", "- All periods have now been inspected. A selected candidate needs a frozen specification and evaluation on future games.", "- Multiple exploratory comparisons increase the chance of a false finding. These intervals do not correct for that exploration.", "- Market comparisons use closing odds. They do not prove an advantage at the time of a published pick.", "- This offline replay does not change live picks.", "", "## Reproduce", "", "Reuse the original feature cache and the extended backtest inputs.", "The JSON output records source hashes, candidate definitions, and per-game predictions.", "", "```bash", "uv run --project model python model/recency_experiments.py --download-inputs", "```", "", "- [nflverse play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp)", "- [Earlier backtest](extended-backtest.md)", ""]
    path.write_text("\n".join(lines))


def freeze_candidate(result: dict) -> None:
    """Save the promising live-baseline challenger specification for future evaluation."""
    name = "baseline_core_exp4"
    comparison = result["periods"]["All periods"]["candidates"][name]["rawVsControl"]
    if not all(comparison[metric]["ciAdjusted"][1] < 0 for metric in ("brier", "logLoss")):
        return
    path = predict.ROOT / "docs" / "recency-candidate.json"
    if path.exists():
        return
    specification = {
        "candidate": name,
        "frozenAt": datetime.now().astimezone().isoformat(),
        "status": "specification frozen; future prediction capture is not configured",
        "metrics": predict.METRICS,
        "historyGames": 24,
        "halfLifeGames": 4,
        "weightFormula": "0.5 ** (gamesAgo / 4)",
        "average": "weighted game-level means; omit missing values per metric",
        "contextFeatures": ["rest_diff", "neutral_site"],
        "trainingStartSeason": 2021,
        "training": "expanding weekly training; target-week games excluded",
        "estimator": {"name": "logistic regression", "C": 0.6, "maxIter": 2000, "imputation": "training-set median", "scaling": "training-set standard scaler"},
        "marketBlend": "select on the latest completed prior season using existing Brier selection",
        "evaluation": {"primaryControl": "original eight-game statistical baseline", "primaryMetric": "raw Brier score", "confirmationMetric": "raw log loss", "minimumCompletedGames": 250, "sample": "only games with kickoff after frozenAt and predictions saved before kickoff"},
        "sourceHashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), predict.ROOT / "model" / "predict.py", predict.ROOT / "model" / "experiments.py")},
    }
    path.write_text(json.dumps(specification, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-inputs", action="store_true")
    args = parser.parse_args()
    if args.download_inputs:
        extended.download_inputs()
        for season in (2023, 2024):
            request = urllib.request.Request(predict.PBP_URL.format(season=season) + ".gz", headers={"User-Agent": "nfl-picks/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                (ARTIFACTS / f"extended-pbp-{season}.csv.gz").write_bytes(response.read())
        extended.clear_feature_caches()
    schedule = pd.read_csv(ARTIFACTS / "extended-schedule.csv", low_memory=False)
    team_weeks = extended.aggregate_inputs(list(range(2016, 2027)))
    earlier = extended.build_base(schedule, list(range(2016, 2023)))
    frozen = pd.read_pickle(ARTIFACTS / "features-v6-2021-2025.pkl")["features"]
    current = extended.build_base(schedule, [2025, 2026])
    recent = pd.concat([frozen, current[current.season == 2026]], ignore_index=True)
    earlier, old_delta = build_recency(earlier, team_weeks)
    recent, recent_delta = build_recency(recent, team_weeks)
    print(f"Eight-game control reproduced: max difference {max(old_delta, recent_delta):.3g}", flush=True)
    print("Scoring 2018–2022", flush=True)
    past = experiments.run_experiments(earlier, 2018, 2022, candidates())
    print("Scoring 2023–2026", flush=True)
    present = experiments.run_experiments(recent, 2023, 2026, candidates())
    no_injuries = recent.copy()
    no_injuries.loc[no_injuries.season == 2026, ["diff_injury_load", "diff_qb_injury"]] = 0.0
    sensitivity = experiments.run_experiments(no_injuries, 2026, 2026, candidates())
    development = [row for row in present["predictions"] if row["season"] < 2026]
    year2026 = [row for row in present["predictions"] if row["season"] == 2026]
    all_rows = past["predictions"] + present["predictions"]
    result = {
        "generatedAt": datetime.now().astimezone().isoformat(),
        "controlMaxDifference": max(old_delta, recent_delta),
        "specifications": VARIANTS,
        "periods": {"2018–2022": score(past["predictions"]), "2023–2025": score(development), "2026": score(year2026), "All periods": score(all_rows)},
        "bySeason": {str(season): score([row for row in all_rows if row["season"] == season]) for season in sorted({row["season"] for row in all_rows})},
        "withoutCurrentInjuries": score(sensitivity["predictions"]),
        "earlierReplay": past,
        "recentReplay": present,
        "injurySensitivityReplay": sensitivity,
    }
    sources = sorted(set(ARTIFACTS.glob("extended-pbp-*.csv.gz")) | set(ARTIFACTS.glob("extended-injuries-*.csv")) | {ARTIFACTS / "extended-schedule.csv", ARTIFACTS / "features-v6-2021-2025.pkl"})
    result["sourceHashes"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    output = ARTIFACTS / "recency-experiments.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    report = predict.ROOT / "docs" / "recency-experiments.md"
    write_report(result, report)
    freeze_candidate(result)
    for name, entry in result["periods"]["All periods"]["candidates"].items():
        print(name, "raw Brier", entry["raw"]["brier"], "blend Brier", entry["blend"]["brier"], "vs control", entry["rawVsControl"]["brier"], flush=True)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
