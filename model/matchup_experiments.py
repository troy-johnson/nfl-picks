"""Replay matchup, recency, and observed coaching-continuity challengers."""

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
import predict

ARTIFACTS = predict.ROOT / "model" / "artifacts"
STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
METRICS = ("pass_epa", "rush_epa", "pass_allowed", "rush_allowed", "sacks_allowed", "sacks_forced", "giveaways", "takeaways")
WINDOWS = {"team8": (8, None, 1.0), "team4": (4, None, 1.0), "team16": (16, None, 1.0), "weighted3": (16, 3.0, 1.0), "offseason25": (16, 3.0, 0.25)}
MATCHUP_FEATURES = ("match_pass", "match_rush", "match_sacks")
COACH_FEATURES = ("coach_tenure_diff", "coach_recent_change_diff", "coach_new_season_diff")


def rate_features(history: list[dict], season: int, window: int, half_life: float | None, previous_season_weight: float) -> dict[str, float]:
    """Weight completed games before computing rates from summed counts."""
    recent = history[-window:]
    if not recent:
        return dict.fromkeys(METRICS, np.nan)
    weights = np.array([1.0 if half_life is None else 0.5 ** ((len(recent) - 1 - i) / half_life) for i in range(len(recent))])
    weights *= np.array([1.0 if game["season"] == season else previous_season_weight for game in recent])
    result = {}
    for metric in METRICS:
        numerator = np.array([game[f"{metric}_num"] for game in recent], dtype=float)
        denominator = np.array([game[f"{metric}_den"] for game in recent], dtype=float)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
        total = np.sum(weights[valid] * denominator[valid])
        result[metric] = float(np.sum(weights[valid] * numerator[valid]) / total) if total else np.nan
    return result


def team_game(team: pd.Series, opponent: pd.Series, season: int) -> dict:
    """Join each team's offensive counts to its opponent's offensive counts."""
    own_pass, opp_pass = float(team.attempts + team.sacks_suffered), float(opponent.attempts + opponent.sacks_suffered)
    own_plays, opp_plays = own_pass + float(team.carries), opp_pass + float(opponent.carries)
    values = {
        "pass_epa": (team.passing_epa, own_pass),
        "rush_epa": (team.rushing_epa, team.carries),
        "pass_allowed": (opponent.passing_epa, opp_pass),
        "rush_allowed": (opponent.rushing_epa, opponent.carries),
        "sacks_allowed": (team.sacks_suffered, own_pass),
        "sacks_forced": (opponent.sacks_suffered, opp_pass),
        "giveaways": (team.passing_interceptions + team.fumbles_lost_total, own_plays),
        "takeaways": (opponent.passing_interceptions + opponent.fumbles_lost_total, opp_plays),
    }
    return {"season": season, **{f"{metric}_{part}": float(value) for metric, pair in values.items() for part, value in zip(("num", "den"), pair)}}


def build_features(features: pd.DataFrame, schedule: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Freeze histories for the entire target week before adding that week's stats."""
    stats = stats[stats.season_type == "REG"].set_index(["season", "week", "team"])
    coaching = schedule.set_index("game_id")
    if not stats.index.is_unique or not coaching.index.is_unique:
        raise ValueError("Stats and schedules must have unique game keys")
    histories: dict[str, deque] = defaultdict(lambda: deque(maxlen=16))
    coaches: dict[str, dict] = {}
    rows = []
    for (season, week), games in features.sort_values(["season", "week", "game_id"]).groupby(["season", "week"], sort=True):
        season, week = int(season), int(week)
        for _, game in games.iterrows():
            home, away = str(game.home_team), str(game.away_team)
            row = {"game_id": game.game_id}
            for name, parameters in WINDOWS.items():
                home_rates = rate_features(list(histories[home]), season, *parameters)
                away_rates = rate_features(list(histories[away]), season, *parameters)
                for metric in METRICS:
                    row[f"{name}_{metric}"] = home_rates[metric] - away_rates[metric]
                if name == "team8":
                    row["match_pass"] = home_rates["pass_epa"] * away_rates["pass_allowed"] - away_rates["pass_epa"] * home_rates["pass_allowed"]
                    row["match_rush"] = home_rates["rush_epa"] * away_rates["rush_allowed"] - away_rates["rush_epa"] * home_rates["rush_allowed"]
                    row["match_sacks"] = home_rates["sacks_allowed"] * away_rates["sacks_forced"] - away_rates["sacks_allowed"] * home_rates["sacks_forced"]
            home_coach, away_coach = coaches.get(home), coaches.get(away)
            for name, getter in [
                ("coach_tenure_diff", lambda c: np.log1p(c["tenure"])),
                ("coach_recent_change_diff", lambda c: float(c["changed"] and c["tenure"] <= 4)),
                ("coach_new_season_diff", lambda c: float(c["start_season"] == season and c["changed"])),
            ]:
                row[name] = getter(home_coach) - getter(away_coach) if home_coach and away_coach else np.nan
            rows.append(row)
        for _, game in games.iterrows():
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            for side, opponent_side in (("home", "away"), ("away", "home")):
                team, opponent = str(game[f"{side}_team"]), str(game[f"{opponent_side}_team"])
                key, opponent_key = (season, week, team), (season, week, opponent)
                if key not in stats.index or opponent_key not in stats.index:
                    raise ValueError(f"Missing completed-game stats: {game.game_id}")
                if stats.loc[key, "game_id"] != game.game_id or stats.loc[opponent_key, "game_id"] != game.game_id:
                    raise ValueError(f"Team stats do not match game: {game.game_id}")
                histories[team].append(team_game(stats.loc[key], stats.loc[opponent_key], season))
                coach = coaching.loc[game.game_id, f"{side}_coach"]
                if pd.notna(coach):
                    prior = coaches.get(team)
                    if prior and prior["name"] == coach:
                        prior["tenure"] += 1
                    else:
                        coaches[team] = {"name": coach, "tenure": 1, "changed": prior is not None, "start_season": season}
    return features.merge(pd.DataFrame(rows), on="game_id", validate="one_to_one")


def candidates() -> list[experiments.Candidate]:
    base = experiments.ELO_TUNED_INJURY_FEATURES
    rates = lambda name: tuple(f"{name}_{metric}" for metric in METRICS)
    result = [experiments.CANDIDATES["baseline"], experiments.CANDIDATES["elo_tuned_injury"]]
    for name in WINDOWS:
        result.append(experiments.Candidate(name, f"Tuned Elo and injury features plus {name} team rates", base + rates(name), experiments.logistic_pipeline))
    result += [
        experiments.Candidate("matchup", "Eight-game team rates plus opponent interactions", base + rates("team8") + MATCHUP_FEATURES, experiments.logistic_pipeline),
        experiments.Candidate("coaching", "Tuned Elo and injury features plus observed coaching continuity", base + COACH_FEATURES, experiments.logistic_pipeline),
        experiments.Candidate("weighted_coaching", "Weighted team rates plus observed coaching continuity", base + rates("weighted3") + COACH_FEATURES, experiments.logistic_pipeline),
    ]
    return result


def compare(result: dict) -> dict:
    controls = {"team8": "elo_tuned_injury", "team4": "team8", "team16": "team8", "weighted3": "team8", "offseason25": "weighted3", "matchup": "team8", "coaching": "elo_tuned_injury", "weighted_coaching": "weighted3"}
    frame = pd.DataFrame(result["predictions"])
    frame = frame[frame.marketHomeProbability.notna()]
    comparisons = {}
    for name, control in controls.items():
        comparisons[name] = {}
        for period, group in [("pooled", frame), *[(str(s), g) for s, g in frame.groupby("season")]]:
            entry = {"control": control, "games": len(group)}
            for field in ("probability", "blendedProbability"):
                first = group.candidates.map(lambda c: c[name][field]).to_numpy()
                second = group.candidates.map(lambda c: c[control][field]).to_numpy()
                entry[field] = experiments.paired_difference(first, second, group.homeWin.to_numpy(), (group.season * 100 + group.week).to_numpy())
            comparisons[name][period] = entry
    return comparisons


def write_report(result: dict, path: Path) -> None:
    summary = result["summary"]
    lines = ["# Matchup, recency, and coaching tests", "", f"Scored {summary['marketGames']} games from 2023–2025 with expanding weekly training.", "Each week trains on earlier weeks. The prior season selects the market blend.", "These tests are retrospective. The 2025 results are a temporal check, not a new untouched holdout.", "", "## Probability scores", "", "Lower Brier and log loss are better.", "", "| Candidate | Accuracy | Raw Brier | Raw log loss | Blended Brier |", "| --- | ---: | ---: | ---: | ---: |"]
    market = summary["market"]
    lines.append(f"| Recorded closing market | {market['accuracy']:.1%} | {market['brier']:.5f} | {market['logLoss']:.5f} | {market['brier']:.5f} |")
    for name, value in summary["candidates"].items():
        raw, blend = value["marketGames"]["raw"], value["marketGames"]["blended"]
        lines.append(f"| {name} | {raw['accuracy']:.1%} | {raw['brier']:.5f} | {raw['logLoss']:.5f} | {blend['brier']:.5f} |")
    confirmed = []
    for name, periods in result["featureComparisons"].items():
        value = periods["pooled"]["probability"]
        if value["brier"] < -2 * value["brierWeekBlockStandardError"] and value["logLoss"] < -2 * value["logLossWeekBlockStandardError"]:
            confirmed.append(name)
    lines += ["", "## Findings", "", f"Raw improvements beyond two week-block standard errors: {', '.join(confirmed) or 'none'}.", "", "## Blends against the market", "", "Negative differences favor the blend. A small gain relative to its SE does not establish an improvement.", "", "| Candidate | Brier difference | SE | Log-loss difference | SE |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, entry in summary["candidates"].items():
        value = entry["marketGames"]["blendedVersusMarket"]
        lines.append(f"| {name} | {value['brier']:+.5f} | {value['brierWeekBlockStandardError']:.5f} | {value['logLoss']:+.5f} | {value['logLossWeekBlockStandardError']:.5f} |")
    lines += ["", "## Added-feature comparisons", "", "Negative differences favor the candidate. SE is the standard error across week blocks.", "", "| Candidate vs control | Period | Brier difference | SE | Log-loss difference | SE |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, periods in result["featureComparisons"].items():
        for period, entry in periods.items():
            value = entry["probability"]
            lines.append(f"| {name} vs {entry['control']} | {period} | {value['brier']:+.5f} | {value['brierWeekBlockStandardError']:.5f} | {value['logLoss']:+.5f} | {value['logLossWeekBlockStandardError']:.5f} |")
    lines += ["", "## Blended comparisons", "", "| Candidate vs control | Brier difference | SE | Log-loss difference | SE |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, periods in result["featureComparisons"].items():
        entry = periods["pooled"]
        value = entry["blendedProbability"]
        lines.append(f"| {name} vs {entry['control']} | {value['brier']:+.5f} | {value['brierWeekBlockStandardError']:.5f} | {value['logLoss']:+.5f} | {value['logLossWeekBlockStandardError']:.5f} |")
    lines += ["", "## Feature definitions", "", "- Team rates use pass EPA, rush EPA, sacks, and turnovers on offense and defense.", "- Each rate divides weighted total counts by weighted total opportunities.", "- The windows contain four, eight, or sixteen completed regular-season games.", "- `weighted3` uses sixteen games with a three-game half-life.", "- `offseason25` adds a 0.25 multiplier for games before the current season.", "- Matchup features multiply each offense's rates by the opposing defense's rates and take the home-minus-away difference.", "- Coaching features use log tenure, a recent observed change, and a change first observed this season.", "", "## Limits", "", "- Coaching features use the coach from the last completed regular-season game. They cannot detect a change before the new coach's first game.", "- Coach tenure counts games in the available history, which begins in 2021. It excludes postseason games.", "- These are coaching-continuity features, not estimates of each coach's skill.", "- Team rates come from weekly box-score aggregates. Their definitions differ from the live play-by-play averages.", "- Window comparisons use matching team-rate features, so they compare recency rather than different stat definitions.", "- No feature uses target-week statistics or the target game's recorded coach.", "- All specifications were fixed before this replay. Multiple exploratory tests can still produce false findings.", "- Closing odds cannot establish an advantage against odds available when a live pick was published.", "", "## Sources and reproduction", "", "- [nflverse team stats](https://nflreadr.nflverse.com/reference/load_team_stats.html)", "- [Team-stat definitions](https://nflreadr.nflverse.com/articles/dictionary_team_stats.html)", "- [nflverse schedule](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)", "", "Reuse the existing `features-v6-2021-2025.pkl` cache.", "The JSON report records source hashes and per-game predictions.", "", "```bash", "uv run --project model python model/matchup_experiments.py --download-inputs", "```", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-inputs", action="store_true")
    args = parser.parse_args()
    paths = [ARTIFACTS / f"matchup-stats-{season}.csv" for season in range(2021, 2026)]
    sources = {str(path): STATS_URL.format(season=season) for path, season in zip(paths, range(2021, 2026))}
    sources[str(ARTIFACTS / "context-schedule.csv")] = predict.SCHEDULE_URL
    if args.download_inputs:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for path, url in sources.items():
            request = urllib.request.Request(url, headers={"User-Agent": "nfl-picks/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                Path(path).write_bytes(response.read())
    cache = ARTIFACTS / "features-v6-2021-2025.pkl"
    features = pd.read_pickle(cache)["features"]
    schedule = pd.read_csv(ARTIFACTS / "context-schedule.csv", low_memory=False)
    stats = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    features = build_features(features, schedule, stats)
    result = experiments.run_experiments(features, 2023, 2025, candidates())
    result["featureComparisons"] = compare(result)
    result["inputs"] = {path: {"url": url, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path, url in sources.items()}
    result["featureCacheSha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    result["missingFeatures"] = {column: int(features[column].isna().sum()) for candidate in candidates() for column in candidate.features if column not in experiments.ELO_TUNED_INJURY_FEATURES}
    result["generatedAt"] = datetime.now().astimezone().isoformat()
    output = ARTIFACTS / "matchup-experiments.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    report = predict.ROOT / "docs" / "matchup-experiments.md"
    write_report(result, report)
    print(experiments.format_table(result["summary"]))
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
