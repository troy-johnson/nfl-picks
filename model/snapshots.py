"""Score the live picks against the same-time odds archive.

`predict.py --capture-odds` stores one multi-book moneyline snapshot per game
day, taken 60-120 minutes before the day's first kickoff, together with the
pick that was published at that moment. This module joins those snapshots to
final scores and scores every probability source on identical games:

- sameTimeMarket: the de-vigged multi-book average in the snapshot.
- closingMarket: the nflverse recorded closing moneyline (the backtest benchmark).
- publishedPick: the blended home probability users saw.
- publishedStatistical: the statistical model's raw home probability.
- candidate:<name>: optional challenger probabilities from experiments.json.

The same-time market is the only fair benchmark for a market-beating claim,
because it is the price that was available when the pick was made. Nothing
here changes live picks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments  # noqa: E402
import predict  # noqa: E402

ROOT = predict.ROOT
SNAPSHOT_OUTPUT = ROOT / "model" / "artifacts" / "snapshot-evaluation.json"
MIN_GAMES_FOR_CLAIM = 250
DETECTABLE_BRIER_DIFFERENCES = (0.005, 0.010)
SAME_TIME = "sameTimeMarket"
CLOSING = "closingMarket"
PUBLISHED = "publishedPick"
STATISTICAL = "publishedStatistical"
CANDIDATE_PREFIX = "candidate:"
COMPARISONS = ((PUBLISHED, SAME_TIME), (STATISTICAL, SAME_TIME), (PUBLISHED, CLOSING), (SAME_TIME, CLOSING))


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def load_snapshots(snapshot_dir: Path = predict.ODDS_SNAPSHOT_DIR) -> list[dict[str, Any]]:
    """Every stored game-day snapshot, oldest first."""
    if not snapshot_dir.exists():
        return []
    return [json.loads(path.read_text()) for path in sorted(snapshot_dir.glob("*/*.json"))]


def snapshot_frame(snapshots: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per snapshot game with the same-time market and the frozen published pick."""
    rows = []
    for snapshot in snapshots:
        captured_at, first_kickoff = parse_timestamp(snapshot.get("capturedAt")), parse_timestamp(snapshot.get("firstKickoffAt"))
        minutes_before = None if captured_at is None or first_kickoff is None else (first_kickoff - captured_at).total_seconds() / 60
        for game in snapshot.get("games", []):
            published = game.get("published") or {}
            rows.append({
                "game_id": str(game["gameId"]),
                "season": int(snapshot["season"]),
                "gameDay": str(snapshot["gameDay"]),
                "capturedAt": snapshot.get("capturedAt"),
                "minutesBeforeFirstKickoff": minutes_before,
                "bookCount": len(game.get("bookmakers", [])),
                SAME_TIME: float(game["marketHomeProbability"]),
                PUBLISHED: published.get("homeWinProbability"),
                STATISTICAL: published.get("statisticalHomeProbability"),
                "publishedMarketHomeProbability": published.get("marketHomeProbability"),
                "publishedAt": published.get("generatedAt"),
            })
    columns = ["game_id", "season", "gameDay", "capturedAt", "minutesBeforeFirstKickoff", "bookCount", SAME_TIME, PUBLISHED, STATISTICAL, "publishedMarketHomeProbability", "publishedAt"]
    frame = pd.DataFrame(rows, columns=columns)
    for column in (PUBLISHED, STATISTICAL, "publishedMarketHomeProbability"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def join_results(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach week, final result, and the recorded closing line. Keeps completed, non-tied games only."""
    schedule = schedule.copy()
    schedule[CLOSING] = schedule.apply(predict.market_home_probability, axis=1)
    columns = ["game_id", "week", "game_type", "home_team", "away_team", "home_score", "away_score", CLOSING]
    joined = frame.merge(schedule[columns], on="game_id", how="left", validate="one_to_one")
    completed = joined.home_score.notna() & joined.away_score.notna()
    decided = completed & (joined.home_score != joined.away_score)
    joined = joined[decided].copy()
    joined["home_win"] = (joined.home_score > joined.away_score).astype(int)
    joined["week"] = joined.week.astype(int)
    joined["weekBlock"] = joined.season * 100 + joined.week
    return joined.sort_values(["season", "week", "gameDay", "game_id"]).reset_index(drop=True)


def attach_candidates(frame: pd.DataFrame, experiment_results: dict[str, Any] | None) -> pd.DataFrame:
    """Add candidate:<name> columns from an experiments.json payload, matched by game id."""
    if not experiment_results:
        return frame
    rows: dict[str, dict[str, float]] = {}
    for prediction in experiment_results.get("predictions", []):
        rows[str(prediction["gameId"])] = {f"{CANDIDATE_PREFIX}{name}": float(value["probability"]) for name, value in prediction.get("candidates", {}).items()}
    if not rows:
        return frame
    candidates = pd.DataFrame.from_dict(rows, orient="index")
    candidates.index.name = "game_id"
    return frame.merge(candidates.reset_index(), on="game_id", how="left")


def probability_sources(frame: pd.DataFrame) -> list[str]:
    fixed = [name for name in (SAME_TIME, CLOSING, PUBLISHED, STATISTICAL) if name in frame.columns]
    return fixed + sorted(column for column in frame.columns if column.startswith(CANDIDATE_PREFIX))


def detectable_games(standard_error: float | None, games: int, difference: float) -> int | None:
    """Games needed for a two-standard-error test of `difference`, extrapolated from the observed week-block SE."""
    if standard_error is None or games < 2 or not math.isfinite(standard_error) or standard_error <= 0:
        return None
    return int(math.ceil(games * (2 * standard_error / difference) ** 2))


def claim_status(comparison: dict[str, float], games: int, min_games: int = MIN_GAMES_FOR_CLAIM) -> str:
    """Whether the first source can be said to beat the second. Requires enough games and two SE on both scores."""
    if games < min_games:
        return f"insufficient games: {games} of {min_games} needed before any claim"
    brier_se, log_se = comparison["brierWeekBlockStandardError"], comparison["logLossWeekBlockStandardError"]
    if brier_se is None or log_se is None or not (math.isfinite(brier_se) and math.isfinite(log_se)):
        return "insufficient week blocks"
    if comparison["brier"] < -2 * brier_se and comparison["logLoss"] < -2 * log_se:
        return "beats second source on Brier and log loss at two standard errors"
    if comparison["brier"] > 2 * brier_se and comparison["logLoss"] > 2 * log_se:
        return "worse than second source on Brier and log loss at two standard errors"
    return "no significant difference"


def evaluate(joined: pd.DataFrame, min_games: int = MIN_GAMES_FOR_CLAIM) -> dict[str, Any]:
    """Metrics for every probability source on identical games, plus paired differences."""
    if joined.empty:
        return {"games": 0, "seasons": [], "sources": {}, "comparisons": [], "bySeason": {}, "sameTimeVersusClosing": None, "capture": None}
    sources = probability_sources(joined)
    scored = joined.dropna(subset=sources).copy()
    actual = scored.home_win.to_numpy()
    weeks = scored.weekBlock.to_numpy()
    result: dict[str, Any] = {
        "games": int(len(scored)),
        "snapshotGames": int(len(joined)),
        "seasons": sorted(int(value) for value in scored.season.unique()),
        "weekBlocks": int(scored.weekBlock.nunique()),
        "minimumGamesForClaim": min_games,
        "sources": {},
        "comparisons": [],
        "bySeason": {},
    }
    for source in sources:
        probabilities = scored[source].to_numpy(dtype=float)
        result["sources"][source] = {**predict.probability_metrics(actual, probabilities), "calibration": experiments.calibration_report(actual, probabilities)}
    pairs = [pair for pair in COMPARISONS if pair[0] in sources and pair[1] in sources]
    pairs += [(source, SAME_TIME) for source in sources if source.startswith(CANDIDATE_PREFIX)]
    for first, second in pairs:
        comparison = experiments.paired_difference(scored[first].to_numpy(dtype=float), scored[second].to_numpy(dtype=float), actual, weeks)
        result["comparisons"].append({
            "first": first,
            "second": second,
            **{key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in comparison.items()},
            "status": claim_status(comparison, len(scored), min_games),
            "gamesToDetect": {f"{difference:.3f}": detectable_games(comparison["brierWeekBlockStandardError"], len(scored), difference) for difference in DETECTABLE_BRIER_DIFFERENCES},
        })
    for season, group in scored.groupby("season"):
        season_actual = group.home_win.to_numpy()
        result["bySeason"][str(int(season))] = {
            "games": int(len(group)),
            **{source: predict.probability_metrics(season_actual, group[source].to_numpy(dtype=float)) for source in sources},
        }
    if SAME_TIME in sources and CLOSING in sources:
        same_time, closing = scored[SAME_TIME].to_numpy(dtype=float), scored[CLOSING].to_numpy(dtype=float)
        result["sameTimeVersusClosing"] = {
            "meanAbsoluteMove": float(np.abs(closing - same_time).mean()),
            "favoriteAgreement": float(((same_time >= 0.5) == (closing >= 0.5)).mean()),
            "correlation": float(np.corrcoef(same_time, closing)[0, 1]) if len(scored) > 1 else None,
        }
    minutes = scored.minutesBeforeFirstKickoff.dropna()
    result["capture"] = {
        "meanBookCount": float(scored.bookCount.mean()),
        "minutesBeforeFirstKickoff": None if minutes.empty else {"min": float(minutes.min()), "mean": float(minutes.mean()), "max": float(minutes.max())},
        "gamesWithPublishedPick": int(joined[PUBLISHED].notna().sum()),
    }
    return result


def format_report(result: dict[str, Any]) -> str:
    if not result["games"]:
        return "No completed games with same-time odds yet."
    lines = [f"{result['games']} completed games with same-time odds across {result['weekBlocks']} weeks ({', '.join(map(str, result['seasons']))})", ""]
    lines.append(f"{'source':<32}{'acc':>8}{'brier':>10}{'logloss':>10}{'slope':>8}")
    for name, metrics in result["sources"].items():
        lines.append(f"{name:<32}{metrics['accuracy']:>8.3f}{metrics['brier']:>10.5f}{metrics['logLoss']:>10.5f}{metrics['calibration']['slope']:>8.2f}")
    lines.append("")
    lines.append("paired differences (first minus second; negative favours first)")
    for comparison in result["comparisons"]:
        brier, brier_se = comparison["brier"], comparison["brierWeekBlockStandardError"]
        log, log_se = comparison["logLoss"], comparison["logLossWeekBlockStandardError"]
        se = lambda value: "n/a" if value is None else f"{value:.5f}"  # noqa: E731
        lines.append(f"  {comparison['first']} vs {comparison['second']}: Brier {brier:+.5f} (SE {se(brier_se)}), log loss {log:+.5f} (SE {se(log_se)}); {comparison['status']}")
        needed = comparison["gamesToDetect"]
        lines.append(f"    games for two SE: {', '.join(f'{key}: {value}' for key, value in needed.items())}")
    move = result.get("sameTimeVersusClosing")
    if move:
        lines.append("")
        lines.append(f"same-time vs closing: mean absolute move {move['meanAbsoluteMove']:.4f}, favourite agreement {move['favoriteAgreement']:.3f}")
    return "\n".join(lines)


def run(snapshot_dir: Path, experiment_path: Path | None, min_games: int) -> tuple[dict[str, Any], pd.DataFrame]:
    snapshots = load_snapshots(snapshot_dir)
    frame = snapshot_frame(snapshots)
    if frame.empty:
        return evaluate(frame, min_games), frame
    schedule = predict.load_schedule(sorted(int(value) for value in frame.season.unique()))
    joined = join_results(frame, schedule)
    if experiment_path and experiment_path.exists():
        joined = attach_candidates(joined, json.loads(experiment_path.read_text()))
    return evaluate(joined, min_games), joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Score published picks against the same-time odds archive")
    parser.add_argument("--snapshots-dir", type=Path, default=predict.ODDS_SNAPSHOT_DIR)
    parser.add_argument("--experiments", type=Path, default=None, help="experiments.json whose candidate probabilities are scored on the snapshot games")
    parser.add_argument("--min-games", type=int, default=MIN_GAMES_FOR_CLAIM)
    parser.add_argument("--output", type=Path, default=SNAPSHOT_OUTPUT)
    args = parser.parse_args()
    result, joined = run(args.snapshots_dir, args.experiments, args.min_games)
    result["generatedAt"] = datetime.now().astimezone().isoformat()
    if not joined.empty:
        result["gameDetails"] = json.loads(joined.drop(columns=["home_score", "away_score"]).to_json(orient="records"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(format_report(result))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
