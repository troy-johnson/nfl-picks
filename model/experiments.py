"""Offline experiment queue for pre-kickoff NFL win-probability challengers.

This module never changes live picks. It reuses the feature builders in
``predict.py`` and replays completed weeks with expanding chronological
retraining. Every choice that could leak future information (hyperparameters,
calibration, and the market blend weight) is fitted on the most recent prior
season only, then the final model is refitted on all earlier completed games.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import predict

EXPERIMENT_OUTPUT = predict.ROOT / "model" / "artifacts" / "experiments.json"
FEATURE_CACHE_DIR = predict.ROOT / "model" / "artifacts"
PROBABILITY_FLOOR = 0.0001
DEFAULT_MARKET_WEIGHT = 0.7
MIN_TRAINING_GAMES = 200
MIN_MARKET_GAMES = 100
CALIBRATION_BINS = 10
INJURY_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
INJURY_STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}
INJURY_FEATURES = predict.ELO_FEATURES + ["diff_injury_load", "diff_qb_injury"]
ELO_WARMUP_FIRST_SEASON = 2010
ELO_MOV_CAP = 24.0
ELO_WARM_FEATURES = tuple(predict.FEATURES) + ("elo_diff_warm",)
ELO_MOV_FEATURES = tuple(predict.FEATURES) + ("elo_diff_mov",)
ELO_MOV_INJURY_FEATURES = ELO_MOV_FEATURES + ("diff_injury_load", "diff_qb_injury")
ELO_TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
ELO_EXPECTED_TEAM_COUNT = 32
ELO_GRID_K_VALUES = (16.0, 20.0, 24.0)
ELO_GRID_HOME_ADVANTAGES = (48.0, 65.0)
ELO_GRID_REGRESSIONS = (0.5, 0.67, 0.8)
ELO_GRID_PARAMS: dict[str, dict[str, float]] = {
    f"elo_grid_k{k:g}_h{h:g}_r{r:g}": {"k": k, "homeAdvantage": h, "seasonRegression": r}
    for k in ELO_GRID_K_VALUES
    for h in ELO_GRID_HOME_ADVANTAGES
    for r in ELO_GRID_REGRESSIONS
}
ELO_MOV2_COLUMN = "elo_diff_mov2"
ELO_TUNED_COLUMN = "elo_diff_tuned"
ELO_MOV2_FEATURES = tuple(predict.FEATURES) + (ELO_MOV2_COLUMN,)
ELO_MOV2_INJURY_FEATURES = ELO_MOV2_FEATURES + ("diff_injury_load", "diff_qb_injury")
ELO_TUNED_FEATURES = tuple(predict.FEATURES) + (ELO_TUNED_COLUMN,)
ELO_TUNED_INJURY_FEATURES = ELO_TUNED_FEATURES + ("diff_injury_load", "diff_qb_injury")
MARKET_RATING_COLUMN = "market_rating_diff"
ELO_SEED_COLUMN = "elo_diff_market_seed"
MARKET_RATING_RIDGE = 0.1
MARKET_PREVIOUS_SEASON_WEIGHT = 0.5
ELO_SEED_SPREAD_TO_ELO = 25.0
ELO_SEED_K = 16.0
ELO_SEED_HOME_ADVANTAGE = 48.0
ELO_SEED_REGRESSION = 0.67
MARKET_RATING_INJURY_FEATURES = tuple(predict.FEATURES) + (MARKET_RATING_COLUMN, "diff_injury_load", "diff_qb_injury")
ELO_SEED_INJURY_FEATURES = tuple(predict.FEATURES) + (ELO_SEED_COLUMN, "diff_injury_load", "diff_qb_injury")
ELO_SEED_MARKET_RATING_INJURY_FEATURES = tuple(predict.FEATURES) + (ELO_SEED_COLUMN, MARKET_RATING_COLUMN, "diff_injury_load", "diff_qb_injury")


@dataclass(frozen=True)
class Candidate:
    """One challenger definition. ``grid`` lists the hyperparameter sets to search."""

    name: str
    description: str
    features: tuple[str, ...]
    estimator: Callable[..., BaseEstimator]
    grid: tuple[dict[str, Any], ...] = ({},)
    calibrate: bool = False
    uses_market: bool = False

    def build(self, params: dict[str, Any]) -> BaseEstimator:
        return self.estimator(**params)


@dataclass(frozen=True)
class PlattCalibrator:
    """Two-parameter logistic recalibration in logit space."""

    intercept: float = 0.0
    slope: float = 1.0

    @classmethod
    def fit(cls, probabilities: np.ndarray, actual: np.ndarray) -> "PlattCalibrator":
        logits = logit(probabilities).reshape(-1, 1)
        if len(np.unique(actual)) < 2:
            return cls()
        model = LogisticRegression(C=np.inf, max_iter=5000)
        model.fit(logits, actual)
        return cls(float(model.intercept_[0]), float(model.coef_[0][0]))

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.intercept + self.slope * logit(probabilities))))


IDENTITY_CALIBRATOR = PlattCalibrator()


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return np.log(clipped / (1 - clipped))


def logistic_pipeline(C: float = 0.6) -> Pipeline:
    """Match ``predict.new_pipeline`` so the baseline candidate reproduces live probabilities."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, C=C)),
    ])


def boosted_trees(max_depth: int = 2, max_iter: int = 100, learning_rate: float = 0.05) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=max_depth,
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def random_forest(max_depth: int = 4, min_samples_leaf: int = 20) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0, n_jobs=-1)),
    ])


def load_elo_schedule(first_season: int, last_season: int) -> pd.DataFrame:
    """Load schedule rows for the Elo walk; warm-up seasons need no play-by-play."""
    return predict.load_schedule(list(range(first_season, last_season + 1)))


def aliased_elo_schedule(games: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Copy the schedule frame and map old franchise codes to their current codes.

    Aliasing applies inside the Elo walk only. Feature rows keep the team codes the
    schedule publishes, so no feature or injury join changes.
    """
    result = games.copy()
    if aliases:
        result["home_team"] = result.home_team.astype(str).map(lambda team: aliases.get(team, team))
        result["away_team"] = result.away_team.astype(str).map(lambda team: aliases.get(team, team))
    return result


def assert_elo_team_codes(schedule: pd.DataFrame, aliases: dict[str, str]) -> list[str]:
    """Return the distinct aliased team codes; fail unless the walk sees exactly 32."""
    aliased = aliased_elo_schedule(schedule, aliases)
    codes = sorted(set(aliased.home_team.astype(str)) | set(aliased.away_team.astype(str)))
    assert len(codes) == ELO_EXPECTED_TEAM_COUNT, (
        f"Aliased Elo schedule has {len(codes)} team codes, expected {ELO_EXPECTED_TEAM_COUNT}: {codes}"
    )
    return codes


def fit_spread_ratings(
    games: pd.DataFrame,
    *,
    ridge: float = MARKET_RATING_RIDGE,
    weights: np.ndarray | None = None,
) -> tuple[dict[str, float], float]:
    """Least-squares team ratings and one shared home-field allowance from closing spreads.

    The model is ``spread_line = rating_home - rating_away + hfa`` over the completed
    games given. ``spread_line`` is positive when the home team is favoured; the sign
    was verified against the recorded moneylines (99.5% agreement on the 2010-2025
    schedule). Neutral-site games carry no home-field term. A small ridge penalty pulls
    the ratings toward zero so the rating sum is identifiable; the home-field term is
    not penalised. ``weights`` reweight rows, which the daily fit uses to discount the
    previous season. Only completed games with a recorded spread are fitted; teams that
    appear in no fitted game return no rating.
    """
    usable = games[games.spread_line.notna() & games.home_score.notna() & games.away_score.notna()]
    teams = sorted(set(usable.home_team.astype(str)) | set(usable.away_team.astype(str)))
    if usable.empty or not teams:
        return {}, 0.0
    positions = {team: position for position, team in enumerate(teams)}
    count = len(usable)
    design = np.zeros((count, len(teams) + 1))
    rows = np.arange(count)
    design[rows, usable.home_team.astype(str).map(positions).to_numpy(dtype=int)] = 1.0
    design[rows, usable.away_team.astype(str).map(positions).to_numpy(dtype=int)] = -1.0
    neutral = usable.get("location", pd.Series("Home", index=usable.index)).astype(str).str.lower().eq("neutral")
    design[:, -1] = (~neutral.to_numpy()).astype(float)
    row_weights = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    normal = design.T @ (design * row_weights[:, None])
    penalty = np.eye(len(teams) + 1) * ridge
    penalty[-1, -1] = 0.0
    target = design.T @ (row_weights * usable.spread_line.to_numpy(dtype=float))
    solution = np.linalg.solve(normal + penalty, target)
    ratings = {team: float(solution[positions[team]]) for team in teams}
    return ratings, float(solution[-1])


def market_rating_differences(
    schedule: pd.DataFrame,
    target_games: pd.DataFrame,
    *,
    previous_season_weight: float = MARKET_PREVIOUS_SEASON_WEIGHT,
    ridge: float = MARKET_RATING_RIDGE,
    aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    """Spread-based power-rating difference for each target game, fitted once per game day.

    Leakage rule: for a target game on day D, only completed games with ``gameday``
    strictly before D contribute spreads. The trailing window is the current season's
    earlier games at weight 1.0 plus the whole previous season, regular and postseason,
    at weight ``previous_season_weight`` (0.5). All games on one day share one fit, so
    no same-day or later spread can leak. ``aliases`` keep relocated franchises on one
    rating; input frames are never modified. Returns game_id ->
    ``rating_home - rating_away`` with no home-field term, because the logistic model
    carries its own intercept and the neutral-site flag.
    """
    history = aliased_elo_schedule(schedule, aliases or {})
    history["_date"] = pd.to_datetime(history.gameday, errors="coerce")
    completed = history[history._date.notna() & history.spread_line.notna() & history.home_score.notna() & history.away_score.notna()]
    targets = target_games[["game_id", "season", "gameday", "home_team", "away_team"]].copy()
    targets["_date"] = pd.to_datetime(targets.gameday, errors="coerce")
    mapping = aliases or {}
    targets["home_code"] = targets.home_team.astype(str).map(lambda team: mapping.get(team, team))
    targets["away_code"] = targets.away_team.astype(str).map(lambda team: mapping.get(team, team))
    differences: dict[str, float] = {}
    for (season, date), group in targets.groupby(["season", "_date"], sort=True):
        window = completed[(completed._date < date) & completed.season.isin([season - 1, season])]
        row_weights = np.where(window.season.to_numpy() == season, 1.0, previous_season_weight)
        ratings, _ = fit_spread_ratings(window, ridge=ridge, weights=row_weights)
        for row in group.to_dict("records"):
            differences[str(row["game_id"])] = ratings.get(row["home_code"], 0.0) - ratings.get(row["away_code"], 0.0)
    return differences


def market_season_priors(
    schedule: pd.DataFrame,
    *,
    spread_to_elo: float = ELO_SEED_SPREAD_TO_ELO,
    ridge: float = MARKET_RATING_RIDGE,
    aliases: dict[str, str] | None = None,
) -> dict[tuple[int, str], float]:
    """Elo season-start targets per ``(season, team)`` from the previous season's closing spreads.

    For each season boundary after the first schedule season, one least-squares fit on
    the previous season's completed regular and postseason games gives each team a
    spread rating, and the Elo target is ``1500 + spread_to_elo * rating``. The
    25-point scale is the FiveThirtyEight Elo-per-spread-point conversion. An NFL
    season always completes before the next season's first kickoff, so no same-season
    or later spread can enter a boundary fit. Keys use the post-alias team codes; the
    first schedule season keeps the plain 1500 start.
    """
    history = aliased_elo_schedule(schedule, aliases or {})
    completed = history[history.spread_line.notna() & history.home_score.notna() & history.away_score.notna()]
    seasons = sorted(int(value) for value in completed.season.unique())
    priors: dict[tuple[int, str], float] = {}
    for season in seasons[1:]:
        ratings, _ = fit_spread_ratings(completed[completed.season == season - 1], ridge=ridge)
        for team, rating in ratings.items():
            priors[(season, team)] = predict.ELO_INITIAL_RATING + spread_to_elo * rating
    return priors


def pregame_elo_v2(
    games: pd.DataFrame,
    *,
    k: float = 20.0,
    home_advantage: float = 65.0,
    season_regression: float = 0.67,
    mov_cap: float | None = None,
    signed_mov: bool = False,
    aliases: dict[str, str] | None = None,
    season_prior: dict[tuple[int, str], float] | None = None,
) -> dict[str, float]:
    """Elo walk identical to ``predict.pregame_elo_differences`` plus an optional FiveThirtyEight margin-of-victory multiplier.

    With ``mov_cap=None`` the output is exactly ``predict.pregame_elo_differences`` on the
    same frame. With ``mov_cap`` set, each rating change is multiplied by
    ``ln(min(abs(margin), mov_cap) + 1) * 2.2 / (2.2 + 0.001 * elo_diff_winner)``,
    where ``elo_diff_winner`` is the pregame difference from the winner's point of view.
    With ``signed_mov=False`` (the default) the absolute value is used, so favourites and
    underdogs are damped equally. With ``signed_mov=True`` the signed value is used, as
    FiveThirtyEight does: favourites that win get a smaller update and underdogs that win
    get a larger one. A tie moves no rating because that factor is zero at margin zero.
    ``aliases`` maps old franchise codes (for example ``OAK``) to current codes on a copy
    of the frame; the input frame is never modified. ``season_prior`` maps
    ``(season, team)`` with post-alias team codes to an Elo target: at each season
    boundary a rating regresses toward that target instead of toward 1500, as
    ``target + season_regression * (rating - target)``. Teams without an entry for the
    new season regress toward 1500, and the first walk season keeps the 1500 start.
    The default ``season_prior=None`` reproduces the plain walk exactly, so the
    ``elo_diff_warm`` / ``elo_diff_mov`` / ``elo_diff_mov2`` columns are unchanged.
    Postseason games update ratings, but their game ids never match a regular-season
    feature row, so they never appear as prediction rows. The whole week is predicted
    first and updated after, so only games strictly before the target week move ratings.
    """
    if aliases:
        games = aliased_elo_schedule(games, aliases)
    ratings: defaultdict[str, float] = defaultdict(lambda: predict.ELO_INITIAL_RATING)
    differences: dict[str, float] = {}
    games = games.sort_values(["season", "week", "gameday", "gametime", "game_id"])
    previous_season: int | None = None
    for season, week in games[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).itertuples(index=False):
        if previous_season is not None and season != previous_season:
            for team, rating in ratings.items():
                target = predict.ELO_INITIAL_RATING
                if season_prior is not None:
                    target = season_prior.get((int(season), team), predict.ELO_INITIAL_RATING)
                ratings[team] = target + season_regression * (rating - target)
        week_games = games[(games.season == season) & (games.week == week)]
        week_predictions: list[tuple[pd.Series, float]] = []
        for _, game in week_games.iterrows():
            home, away = str(game.home_team), str(game.away_team)
            difference = ratings[home] - ratings[away]
            if str(game.get("location", "")).lower() != "neutral":
                difference += home_advantage
            differences[str(game.game_id)] = difference
            week_predictions.append((game, 1 / (1 + 10 ** (-difference / 400))))
        for game, probability in week_predictions:
            if pd.isna(game.home_score) or pd.isna(game.away_score):
                continue
            home_score, away_score = float(game.home_score), float(game.away_score)
            outcome = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
            factor = 1.0
            if mov_cap is not None and outcome == 0.5:
                factor = 0.0
            elif mov_cap is not None:
                margin = min(abs(home_score - away_score), mov_cap)
                winner_difference = differences[str(game.game_id)] if outcome == 1.0 else -differences[str(game.game_id)]
                damping_difference = winner_difference if signed_mov else abs(winner_difference)
                factor = math.log(margin + 1) * 2.2 / (damping_difference * 0.001 + 2.2)
            change = k * (outcome - probability) * factor
            ratings[str(game.home_team)] += change
            ratings[str(game.away_team)] -= change
        previous_season = int(season)
    return differences


INJURY_COLUMNS = ["season", "game_type", "team", "week", "position", "report_status", "date_modified"]


def normalize_injury_frame(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    """Standardize one season's injury frame, accepting files published without date_modified."""
    if "date_modified" in frame.columns:
        return frame[INJURY_COLUMNS].assign(has_timestamp=True)
    print(f"Injury report for {season} has no date_modified column; its rows are accepted without a timestamp check")
    return frame[[column for column in INJURY_COLUMNS if column != "date_modified"]].assign(date_modified=np.nan, has_timestamp=False)


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    """Download one nflverse injury report per season, skipping and noting any season file that is missing."""
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            frame = pd.read_csv(INJURY_URL.format(season=season), low_memory=False)
        except Exception as error:
            print(f"Injury report for {season} could not be downloaded ({error}); continuing without it")
            continue
        frames.append(normalize_injury_frame(frame, season))
    if not frames:
        return pd.DataFrame(columns=INJURY_COLUMNS + ["has_timestamp"])
    return pd.concat(frames, ignore_index=True)


def build_injury_features(features: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Add pregame injury-load and QB-injury differences for each game.

    Leakage rule: an injury row with ``has_timestamp`` True counts only when the UTC
    date of its ``date_modified`` is strictly earlier than the game's ``gameday``
    date. Rows modified on game day or later, and rows without a parseable
    ``date_modified``, are excluded, so no game-day or post-kickoff information can
    leak. Rows with ``has_timestamp`` False (season files published without a
    ``date_modified`` column) are counted for their season and week without a date
    check. For 2021-2024, 99.7% of weighted report rows carry a ``date_modified``
    at least one day before ``gameday`` and about 90% two days before, so the file
    is a final practice-week snapshot, not a game-day inactives list, and the
    timestamp check discards almost nothing. Rows with a NaN ``report_status``
    contribute zero, and a team with no rows for that week scores zero.
    """
    games = features.copy()
    games["_game_date"] = pd.to_datetime(games.gameday, errors="coerce").dt.tz_localize("UTC")
    if injuries.empty or games.empty:
        for column in ("home_injury_load", "away_injury_load", "home_qb_injury", "away_qb_injury"):
            games[column] = 0.0
        games["diff_injury_load"] = games.home_injury_load - games.away_injury_load
        games["diff_qb_injury"] = games.home_qb_injury - games.away_qb_injury
        return games.drop(columns="_game_date")
    report = injuries[injuries.season.isin(games.season.unique())].copy()
    report = report[report.game_type == "REG"]
    schedule_teams = set(games.home_team.astype(str)) | set(games.away_team.astype(str))
    unknown = set(report.team.astype(str)) - schedule_teams
    assert not unknown, f"Injury team codes missing from the schedule: {sorted(unknown)}"
    report["team"] = report.team.astype(str)
    report["position"] = report.position.astype(str).str.strip()
    report["weight"] = report.report_status.astype(str).map(INJURY_STATUS_WEIGHT).fillna(0.0)
    if "has_timestamp" not in report.columns:
        report["has_timestamp"] = True
    report["_report_date"] = pd.to_datetime(report.date_modified, utc=True, errors="coerce").dt.normalize()
    is_qb = report.position == "QB"
    report["_load"] = report.weight.where(~is_qb, 0.0)
    report["_qb"] = report.weight.where(is_qb, 0.0)
    per_day = report.groupby(["season", "week", "team", "_report_date", "has_timestamp"], as_index=False, dropna=False)[["_load", "_qb"]].sum()
    game_ids = games.game_id.astype(str)
    for side in ("home", "away"):
        side_games = pd.DataFrame({
            "game_id": game_ids,
            "season": games.season,
            "week": games.week,
            "team": games[f"{side}_team"].astype(str),
            "_game_date": games["_game_date"],
        })
        merged = side_games.merge(per_day, on=["season", "week", "team"], how="left")
        merged = merged[merged.has_timestamp.eq(False) | (merged._report_date < merged._game_date)]
        sums = merged.groupby("game_id", as_index=False)[["_load", "_qb"]].sum().set_index("game_id")
        games[f"{side}_injury_load"] = game_ids.map(sums["_load"]).fillna(0.0).to_numpy()
        games[f"{side}_qb_injury"] = game_ids.map(sums["_qb"]).fillna(0.0).to_numpy()
    games["diff_injury_load"] = games.home_injury_load - games.away_injury_load
    games["diff_qb_injury"] = games.home_qb_injury - games.away_qb_injury
    return games.drop(columns="_game_date")


def add_market_logit(frame: pd.DataFrame) -> pd.DataFrame:
    """Append market_logit = log(p / (1 - p)) from market_home_prob, clipped with PROBABILITY_FLOOR."""
    result = frame.copy()
    result["market_logit"] = logit(result.market_home_prob.astype(float).to_numpy())
    return result


def market_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Append market_logit and drop rows without a recorded market probability."""
    return add_market_logit(frame[frame.market_home_prob.notna()])


def market_logit_coefficient(model: Pipeline, features: list[str]) -> float:
    """Read the fitted market_logit coefficient on the raw logit scale, undoing the StandardScaler."""
    index = features.index("market_logit")
    return float(model.named_steps["model"].coef_[0][index] / model.named_steps["scale"].scale_[index])


CANDIDATES: dict[str, Candidate] = {
    candidate.name: candidate
    for candidate in (
        Candidate("baseline", "Live logistic regression on rolling EPA, rest, and venue features", tuple(predict.FEATURES), logistic_pipeline),
        Candidate("elo", "Baseline features plus pregame Elo difference", tuple(predict.ELO_FEATURES), logistic_pipeline),
        Candidate("qb", "Baseline features plus prior-game primary passer features", tuple(predict.QB_FEATURES), logistic_pipeline),
        Candidate("elo_qb", "Baseline features plus Elo and primary passer features", tuple(predict.ELO_QB_FEATURES), logistic_pipeline),
        Candidate("elo_injury", "Elo features plus pregame injury-report load differences", tuple(INJURY_FEATURES), logistic_pipeline),
        Candidate("elo_warm", "Baseline features plus Elo warmed up from 2010 (no margin term)", ELO_WARM_FEATURES, logistic_pipeline),
        Candidate("elo_mov", "Baseline features plus warm-up Elo with the FiveThirtyEight margin-of-victory multiplier", ELO_MOV_FEATURES, logistic_pipeline),
        Candidate("elo_mov_injury", "Warm-up margin-of-victory Elo plus pregame injury-report load differences", ELO_MOV_INJURY_FEATURES, logistic_pipeline),
        Candidate("elo_mov2", "Baseline features plus aliased warm-up Elo with the signed FiveThirtyEight margin multiplier", ELO_MOV2_FEATURES, logistic_pipeline),
        Candidate("elo_mov2_injury", "Signed-margin warm-up Elo plus pregame injury-report load differences", ELO_MOV2_INJURY_FEATURES, logistic_pipeline),
        Candidate("elo_tuned_injury", "Per-week-tuned signed-margin warm-up Elo plus pregame injury-report load differences", ELO_TUNED_INJURY_FEATURES, logistic_pipeline),
        Candidate("market_rating_injury", "Baseline features plus the trailing spread-based power rating and injury-report load differences", MARKET_RATING_INJURY_FEATURES, logistic_pipeline),
        Candidate("elo_seed_injury", "Baseline features plus market-seeded warm-up Elo and injury-report load differences", ELO_SEED_INJURY_FEATURES, logistic_pipeline),
        Candidate("elo_seed_market_rating_injury", "Baseline features plus market-seeded Elo, the spread-based power rating, and injury-report load differences", ELO_SEED_MARKET_RATING_INJURY_FEATURES, logistic_pipeline),
        Candidate("market_stack", "Elo and injury features stacked on the recorded market logit", tuple(INJURY_FEATURES + ["market_logit"]), functools.partial(logistic_pipeline, C=1.0), uses_market=True),
        Candidate(
            "regularized_logistic",
            "Elo plus QB features with the L2 strength chosen on the prior season",
            tuple(predict.ELO_QB_FEATURES),
            logistic_pipeline,
            grid=tuple({"C": C} for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)),
        ),
        Candidate(
            "boosted_trees",
            "Shallow histogram gradient boosting on Elo plus QB features, Platt-calibrated on the prior season",
            tuple(predict.ELO_QB_FEATURES),
            boosted_trees,
            grid=tuple({"max_depth": depth, "max_iter": iterations} for depth in (2, 3) for iterations in (100, 300)),
            calibrate=True,
        ),
        Candidate(
            "random_forest",
            "Shallow random forest on Elo plus QB features, Platt-calibrated on the prior season. "
            "Opt-in only: the 2023-2025 replay showed 0.93 correlation with boosted trees and worse scores.",
            tuple(predict.ELO_QB_FEATURES),
            random_forest,
            grid=tuple({"max_depth": depth} for depth in (3, 5)),
            calibrate=True,
        ),
    )
}
# random_forest is excluded from the default queue. It adds little diversity to boosted_trees
# (0.93 probability correlation in the 2023-2025 replay) and scores worse on every metric.
# Run it with `--candidates ...,random_forest` when a new feature set may change that result.
DEFAULT_QUEUE = ("baseline", "elo", "qb", "elo_qb", "elo_injury", "elo_warm", "elo_mov", "elo_mov_injury", "elo_mov2", "elo_mov2_injury", "elo_tuned_injury", "market_rating_injury", "elo_seed_injury", "elo_seed_market_rating_injury", "market_stack", "regularized_logistic", "boosted_trees")


def select_hyperparameters(candidate: Candidate, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray]:
    """Pick the grid point with the lowest validation log loss. Returns the point and its validation probabilities."""
    features = list(candidate.features)
    y_train, y_validation = train.home_win.astype(int).to_numpy(), validation.home_win.astype(int).to_numpy()
    scored: list[tuple[float, dict[str, Any], np.ndarray]] = []
    for params in candidate.grid:
        model = candidate.build(params)
        model.fit(train[features], y_train)
        probabilities = model.predict_proba(validation[features])[:, 1]
        scoring = PlattCalibrator.fit(probabilities, y_validation).apply(probabilities) if candidate.calibrate else probabilities
        scored.append((float(log_loss(y_validation, np.clip(scoring, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR), labels=[0, 1])), params, probabilities))
    _, params, probabilities = min(scored, key=lambda item: item[0])
    return params, probabilities


def pure_elo_log_loss(differences: np.ndarray, actual: np.ndarray) -> float:
    """Log loss of the pure Elo probability ``1 / (1 + 10 ** (-difference / 400))``."""
    probabilities = np.clip(1 / (1 + 10 ** (-np.asarray(differences, dtype=float) / 400)), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return float(log_loss(actual, probabilities, labels=[0, 1]))


def select_elo_parameters(history: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Pick the grid column whose pure Elo probability has the lowest log loss on the history rows.

    Selection uses every completed row strictly before the target week, never the target
    week itself and never the validation season alone. Ties keep the first column in the
    registry order, so the choice is deterministic.
    """
    actual = history.home_win.astype(int).to_numpy()
    best_column: str | None = None
    best_params: dict[str, float] = {}
    best_score = math.inf
    for column, params in ELO_GRID_PARAMS.items():
        if column not in history.columns:
            continue
        differences = history[column].to_numpy(dtype=float)
        mask = np.isfinite(differences)
        if not mask.any():
            continue
        score = pure_elo_log_loss(differences[mask], actual[mask])
        if score < best_score:
            best_column, best_params, best_score = column, params, score
    assert best_column is not None, "No elo_grid_* columns found in the feature frame"
    return best_column, best_params


@dataclass(frozen=True)
class WeekFit:
    hyperparameters: dict[str, Any]
    calibrator: PlattCalibrator
    market_weight: float
    probabilities: np.ndarray
    market_logit_coefficient: float | None = None


def fit_candidate_week(
    candidate: Candidate,
    history: pd.DataFrame,
    validation_train: pd.DataFrame,
    validation: pd.DataFrame,
    games: pd.DataFrame,
    min_market_games: int,
) -> WeekFit:
    """Fit one candidate for one target week using only games before that week."""
    if candidate.uses_market:
        history, validation_train, validation = (market_training_frame(frame) for frame in (history, validation_train, validation))
        games = add_market_logit(games)
    params, validation_probabilities = select_hyperparameters(candidate, validation_train, validation)
    y_validation = validation.home_win.astype(int).to_numpy()
    calibrator = PlattCalibrator.fit(validation_probabilities, y_validation) if candidate.calibrate else IDENTITY_CALIBRATOR
    calibrated_validation = calibrator.apply(validation_probabilities)
    has_market = validation.market_home_prob.notna().to_numpy()
    market_weight = DEFAULT_MARKET_WEIGHT
    if has_market.sum() >= min_market_games:
        market_weight, _ = predict.select_market_weight(y_validation[has_market], calibrated_validation[has_market], validation.loc[has_market, "market_home_prob"].astype(float).to_numpy())
    features = list(candidate.features)
    model = candidate.build(params)
    scored = games.market_home_prob.notna().to_numpy() if candidate.uses_market else np.ones(len(games), dtype=bool)
    probabilities = np.full(len(games), np.nan)
    if scored.any():
        model.fit(history[features], history.home_win.astype(int).to_numpy())
        probabilities[scored] = calibrator.apply(model.predict_proba(games[scored][features])[:, 1])
    coefficient = market_logit_coefficient(model, features) if candidate.uses_market and scored.any() else None
    return WeekFit(params, calibrator, market_weight, probabilities, coefficient)


def blend(probability: float, market: float | None, market_weight: float) -> float:
    return probability if market is None else market_weight * market + (1 - market_weight) * probability


def run_experiments(
    features: pd.DataFrame,
    from_season: int,
    to_season: int,
    candidates: list[Candidate],
    min_training_games: int = MIN_TRAINING_GAMES,
    min_market_games: int = MIN_MARKET_GAMES,
) -> dict[str, Any]:
    """Replay each completed week with expanding retraining for every candidate on identical games."""
    completed = features[features.home_win.notna()].copy()
    tuned = any(ELO_TUNED_COLUMN in candidate.features for candidate in candidates)
    targets = completed[(completed.season >= from_season) & (completed.season <= to_season)]
    predictions: list[dict[str, Any]] = []
    weeks: list[dict[str, Any]] = []
    skipped_weeks = 0
    for season, week in targets[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).itertuples(index=False):
        games = completed[(completed.season == season) & (completed.week == week)].copy()
        history = completed[(completed.season < season) | ((completed.season == season) & (completed.week < week))]
        prior_seasons = sorted(int(value) for value in history.loc[history.season < season, "season"].unique())
        if len(history) < min_training_games or len(prior_seasons) < 2:
            skipped_weeks += 1
            continue
        validation_season = prior_seasons[-1]
        elo_parameters: dict[str, float] | None = None
        if tuned:
            history = history.copy()
            column, elo_parameters = select_elo_parameters(history)
            history[ELO_TUNED_COLUMN] = history[column].to_numpy()
            games[ELO_TUNED_COLUMN] = games[column].to_numpy()
        validation_train = history[history.season < validation_season]
        validation = history[history.season == validation_season]
        if len(validation_train) < min_training_games or validation.empty:
            skipped_weeks += 1
            continue
        fits = {candidate.name: fit_candidate_week(candidate, history, validation_train, validation, games, min_market_games) for candidate in candidates}
        weeks.append({
            "season": int(season),
            "week": int(week),
            "validationSeason": validation_season,
            "trainingGames": int(len(history)),
            **({"eloParameters": elo_parameters} if elo_parameters is not None else {}),
            "candidates": {
                name: {
                    "hyperparameters": fit.hyperparameters,
                    "calibration": {"intercept": fit.calibrator.intercept, "slope": fit.calibrator.slope},
                    "marketWeight": fit.market_weight,
                    **({"marketLogitCoefficient": fit.market_logit_coefficient} if fit.market_logit_coefficient is not None else {}),
                }
                for name, fit in fits.items()
            },
        })
        for position, (_, game) in enumerate(games.iterrows()):
            market = None if pd.isna(game.market_home_prob) else float(game.market_home_prob)
            predictions.append({
                "gameId": str(game.game_id),
                "season": int(game.season),
                "week": int(game.week),
                "awayTeam": str(game.away_team),
                "homeTeam": str(game.home_team),
                "homeWin": int(game.home_win),
                "marketHomeProbability": market,
                "trainingGames": int(len(history)),
                "validationSeason": validation_season,
                "candidates": {
                    name: {
                        "probability": float(fit.probabilities[position]),
                        "blendedProbability": blend(float(fit.probabilities[position]), market, fit.market_weight),
                        "marketWeight": fit.market_weight,
                    }
                    for name, fit in fits.items()
                },
            })
    if not predictions:
        raise RuntimeError("No weeks could be scored; add more historical seasons")
    return {
        "fromSeason": from_season,
        "toSeason": to_season,
        "method": "expanding weekly retraining; hyperparameters, calibration, and market blend fitted on the latest prior season",
        "marketOddsNote": "Recorded nflverse moneylines are closing lines, not odds captured at prediction time. Use data/odds-snapshots for the market-beating test.",
        "skippedWeeks": skipped_weeks,
        "candidates": {candidate.name: {"description": candidate.description, "features": list(candidate.features), "calibrated": candidate.calibrate, "usesMarket": candidate.uses_market, "gridSize": len(candidate.grid)} for candidate in candidates},
        "summary": summarize(predictions, weeks, [candidate.name for candidate in candidates]),
        "weeks": weeks,
        "predictions": predictions,
    }


def calibration_report(actual: np.ndarray, probabilities: np.ndarray, bins: int = CALIBRATION_BINS) -> dict[str, Any]:
    """Equal-width reliability bins, expected calibration error, and logit-space intercept/slope."""
    edges = np.linspace(0, 1, bins + 1)
    indexes = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        mask = indexes == index
        if not mask.any():
            continue
        mean_predicted, observed = float(probabilities[mask].mean()), float(actual[mask].mean())
        ece += mask.sum() / len(actual) * abs(mean_predicted - observed)
        rows.append({"lower": float(edges[index]), "upper": float(edges[index + 1]), "games": int(mask.sum()), "meanPredicted": mean_predicted, "observedRate": observed})
    fitted = PlattCalibrator.fit(probabilities, actual)
    return {"bins": rows, "expectedCalibrationError": float(ece), "intercept": fitted.intercept, "slope": fitted.slope}


def paired_difference(first: np.ndarray, second: np.ndarray, actual: np.ndarray, weeks: np.ndarray) -> dict[str, float]:
    """Mean per-game Brier and log-loss difference (first minus second) with week-block standard errors."""
    first, second = np.clip(first, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR), np.clip(second, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    brier = (first - actual) ** 2 - (second - actual) ** 2
    log = -(actual * np.log(first) + (1 - actual) * np.log(1 - first)) + (actual * np.log(second) + (1 - actual) * np.log(1 - second))
    frame = pd.DataFrame({"week": weeks, "brier": brier, "logLoss": log}).groupby("week").mean()
    blocks = len(frame)
    return {
        "brier": float(brier.mean()),
        "brierWeekBlockStandardError": float(frame.brier.std(ddof=1) / math.sqrt(blocks)) if blocks > 1 else float("nan"),
        "logLoss": float(log.mean()),
        "logLossWeekBlockStandardError": float(frame.logLoss.std(ddof=1) / math.sqrt(blocks)) if blocks > 1 else float("nan"),
        "weekBlocks": blocks,
    }


def verdict(candidate: dict[str, float | int], baseline: dict[str, float | int]) -> str:
    better_scores = candidate["brier"] < baseline["brier"] and candidate["logLoss"] < baseline["logLoss"]
    if better_scores:
        return "improves baseline Brier and log loss"
    if candidate["accuracy"] > baseline["accuracy"]:
        return "rejected: accuracy gain without Brier and log-loss gain"
    return "rejected: does not improve baseline"


def season_breakdown(scored: pd.DataFrame, name: str, comparisons: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Per-season raw, market, and paired-difference metrics for one candidate's scored games."""
    breakdown: dict[str, dict[str, Any]] = {}
    for season, group in scored.groupby("season"):
        actual = group.homeWin.to_numpy()
        raw = group[f"{name}__probability"].to_numpy()
        market = group.marketHomeProbability.astype(float).to_numpy()
        weeks = (group.season.astype(str) + "-" + group.week.astype(str).str.zfill(2)).to_numpy()
        entry: dict[str, Any] = {
            "games": int(len(group)),
            "raw": predict.probability_metrics(actual, raw),
            "market": predict.probability_metrics(actual, market),
            "rawVersusMarket": paired_difference(raw, market, actual, weeks),
        }
        for key, column in comparisons.items():
            entry[key] = paired_difference(raw, group[column].to_numpy(), actual, weeks)
        breakdown[str(int(season))] = entry
    return breakdown


def summarize(predictions: list[dict[str, Any]], weeks: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    results = pd.DataFrame(predictions)
    for name in names:
        results[f"{name}__probability"] = results.candidates.map(lambda item, name=name: item[name]["probability"])
        results[f"{name}__blended"] = results.candidates.map(lambda item, name=name: item[name]["blendedProbability"])
    market_games = results[results.marketHomeProbability.notna()].copy()
    actual, market_actual = results.homeWin.to_numpy(), market_games.homeWin.to_numpy()
    market = market_games.marketHomeProbability.astype(float).to_numpy()
    market_metrics = predict.probability_metrics(market_actual, market)
    baseline_name = "baseline" if "baseline" in names else names[0]
    baseline_metrics = predict.probability_metrics(market_actual, market_games[f"{baseline_name}__probability"].to_numpy())
    summary: dict[str, Any] = {"games": int(len(results)), "marketGames": int(len(market_games)), "market": {**market_metrics, "calibration": calibration_report(market_actual, market)}, "baselineCandidate": baseline_name, "candidates": {}}
    summary["marketBySeason"] = {
        str(int(season)): predict.probability_metrics(group.homeWin.to_numpy(), group.marketHomeProbability.astype(float).to_numpy())
        for season, group in market_games.groupby("season")
    }
    for name in names:
        scored = market_games[market_games[f"{name}__probability"].notna()]
        raw, blended = scored[f"{name}__probability"].to_numpy(), scored[f"{name}__blended"].to_numpy()
        scored_actual = scored.homeWin.to_numpy()
        scored_market = scored.marketHomeProbability.astype(float).to_numpy()
        scored_weeks = (scored.season.astype(str) + "-" + scored.week.astype(str).str.zfill(2)).to_numpy()
        raw_metrics = predict.probability_metrics(scored_actual, raw)
        weights = [week["candidates"][name]["marketWeight"] for week in weeks]
        valid = results[results[f"{name}__probability"].notna()]
        comparisons: dict[str, str] = {}
        if name != baseline_name:
            comparisons["rawVersusBaseline"] = f"{baseline_name}__probability"
        if "elo" in names and name != "elo":
            comparisons["rawVersusElo"] = "elo__probability"
        if "elo_injury" in names and name != "elo_injury":
            comparisons["rawVersusEloInjury"] = "elo_injury__probability"
        if "elo_mov_injury" in names and name != "elo_mov_injury":
            comparisons["rawVersusEloMovInjury"] = "elo_mov_injury__probability"
        if "elo_tuned_injury" in names and name != "elo_tuned_injury":
            comparisons["rawVersusEloTunedInjury"] = "elo_tuned_injury__probability"
        summary["candidates"][name] = {
            "allGames": predict.probability_metrics(valid.homeWin.to_numpy(), valid[f"{name}__probability"].to_numpy()),
            "bySeason": season_breakdown(scored, name, comparisons),
            "marketGames": {
                "raw": raw_metrics,
                "blended": predict.probability_metrics(scored_actual, blended),
                "rawVersusMarket": paired_difference(raw, scored_market, scored_actual, scored_weeks),
                "blendedVersusMarket": paired_difference(blended, scored_market, scored_actual, scored_weeks),
                **{key: paired_difference(raw, scored[column].to_numpy(), scored_actual, scored_weeks) for key, column in comparisons.items()},
            },
            "calibration": calibration_report(scored_actual, raw),
            "marketWeight": {"mean": float(np.mean(weights)), "min": float(min(weights)), "max": float(max(weights)), "weeksAtFullMarket": int(sum(weight >= 1.0 for weight in weights))},
            "hyperparameterCounts": [{"hyperparameters": dict(params), "weeks": count} for params, count in Counter(tuple(sorted(week["candidates"][name]["hyperparameters"].items())) for week in weeks).most_common()],
            "verdict": verdict(raw_metrics, baseline_metrics) if name != baseline_name else "baseline",
            "rawBeatsRecordedMarket": bool(raw_metrics["brier"] < market_metrics["brier"] and raw_metrics["logLoss"] < market_metrics["logLoss"]),
        }
    correlation = market_games[[f"{name}__probability" for name in names]].assign(market=market).corr()
    correlation.index = correlation.columns = [column.replace("__probability", "") for column in correlation.columns]
    summary["probabilityCorrelation"] = {row: {column: float(value) for column, value in values.items()} for row, values in correlation.to_dict(orient="index").items()}
    summary["rankingByLogLoss"] = sorted(names, key=lambda name: summary["candidates"][name]["marketGames"]["raw"]["logLoss"])
    tuned_weeks = [week["eloParameters"] for week in weeks if "eloParameters" in week]
    if tuned_weeks:
        counts: Counter[tuple[tuple[str, float], ...]] = Counter(tuple(sorted(parameters.items())) for parameters in tuned_weeks)
        column_by_params = {tuple(sorted(params.items())): column for column, params in ELO_GRID_PARAMS.items()}
        summary["eloParameterCounts"] = [
            {"hyperparameters": dict(params), "column": column_by_params[params], "weeks": count}
            for params, count in counts.most_common()
        ]
    return summary


def print_elo_missing(features: pd.DataFrame) -> None:
    """Print how many feature rows lack each derived Elo or market column; the count must be zero."""
    for column in ("elo_diff_warm", "elo_diff_mov", ELO_MOV2_COLUMN, *ELO_GRID_PARAMS, MARKET_RATING_COLUMN, ELO_SEED_COLUMN):
        if column in features.columns:
            print(f"Derived column {column}: {int(features[column].isna().sum())} feature rows missing")


def load_features(to_season: int, cache: Path | None, refresh: bool) -> pd.DataFrame:
    if cache is not None and cache.exists() and not refresh:
        features = pd.read_pickle(cache)["features"]
        print_elo_missing(features)
        return features
    seasons = list(range(predict.START_SEASON, to_season + 1))
    schedule, pbp = predict.load_inputs(seasons)
    features = predict.build_game_features(schedule, predict.aggregate_team_weeks(pbp))
    features = predict.build_qb_features(features, predict.aggregate_qb_games(pbp))
    elo_schedule = load_elo_schedule(ELO_WARMUP_FIRST_SEASON, to_season)
    codes = assert_elo_team_codes(elo_schedule, ELO_TEAM_ALIASES)
    print(f"Elo walk team codes after aliasing: {len(codes)}")
    for column, mov_cap in (("elo_diff_warm", None), ("elo_diff_mov", ELO_MOV_CAP)):
        features[column] = features.game_id.astype(str).map(pregame_elo_v2(elo_schedule, mov_cap=mov_cap))
    aliased_walks = [(ELO_MOV2_COLUMN, {"k": 20.0, "homeAdvantage": 65.0, "seasonRegression": 0.67}), *ELO_GRID_PARAMS.items()]
    for column, params in aliased_walks:
        differences = pregame_elo_v2(
            elo_schedule,
            k=params["k"],
            home_advantage=params["homeAdvantage"],
            season_regression=params["seasonRegression"],
            mov_cap=ELO_MOV_CAP,
            signed_mov=True,
            aliases=ELO_TEAM_ALIASES,
        )
        features[column] = features.game_id.astype(str).map(differences)
    scored_lines = elo_schedule[elo_schedule.spread_line.notna() & elo_schedule.home_moneyline.notna() & elo_schedule.away_moneyline.notna()]
    agreement = ((scored_lines.spread_line > 0) == (scored_lines.home_moneyline < scored_lines.away_moneyline)).mean()
    print(f"Spread sign check versus moneyline favourite: {agreement:.1%} of {len(scored_lines)} games with both lines")
    features[MARKET_RATING_COLUMN] = features.game_id.astype(str).map(
        market_rating_differences(elo_schedule, features, aliases=ELO_TEAM_ALIASES)
    )
    season_priors = market_season_priors(elo_schedule, aliases=ELO_TEAM_ALIASES)
    features[ELO_SEED_COLUMN] = features.game_id.astype(str).map(
        pregame_elo_v2(
            elo_schedule,
            k=ELO_SEED_K,
            home_advantage=ELO_SEED_HOME_ADVANTAGE,
            season_regression=ELO_SEED_REGRESSION,
            mov_cap=ELO_MOV_CAP,
            signed_mov=True,
            aliases=ELO_TEAM_ALIASES,
            season_prior=season_priors,
        )
    )
    print_elo_missing(features)
    features = build_injury_features(features, load_injuries(seasons))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"features": features, "elo_schedule": elo_schedule}, cache)
    return features


def format_table(summary: dict[str, Any]) -> str:
    header = f"{'candidate':<22}{'acc':>7}{'brier':>9}{'logloss':>9}{'dBrier':>9}{'dLogLoss':>10}{'blend':>7}{'slope':>7}  verdict"
    lines = [header]
    market = summary["market"]
    lines.append(f"{'recorded market':<22}{market['accuracy']:>7.3f}{market['brier']:>9.5f}{market['logLoss']:>9.5f}{0:>9.5f}{0:>10.5f}{'':>7}{market['calibration']['slope']:>7.2f}  reference")
    for name in summary["rankingByLogLoss"]:
        item = summary["candidates"][name]
        raw, delta = item["marketGames"]["raw"], item["marketGames"]["rawVersusMarket"]
        lines.append(f"{name:<22}{raw['accuracy']:>7.3f}{raw['brier']:>9.5f}{raw['logLoss']:>9.5f}{delta['brier']:>+9.5f}{delta['logLoss']:>+10.5f}{item['marketWeight']['mean']:>7.2f}{item['calibration']['slope']:>7.2f}  {item['verdict']}")
        seasonal = "  ".join(
            f"{season}: {entry['raw']['brier']:.5f} ({entry.get('rawVersusBaseline', {'brier': 0.0})['brier']:+.5f})"
            for season, entry in sorted(item["bySeason"].items())
        )
        lines.append(f"{'':<22}per-season brier  {seasonal}")
    return "\n".join(lines)


def print_injury_coverage(features: pd.DataFrame) -> None:
    """Print each season's game count and the share of games with a nonzero home injury load."""
    for season, group in features.groupby("season"):
        share = (group.home_injury_load > 0).mean()
        print(f"Injury coverage {int(season)}: {len(group)} games, {share:.1%} with home_injury_load > 0")


def parse_candidates(value: str | None) -> list[Candidate]:
    names = DEFAULT_QUEUE if not value else tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in names if name not in CANDIDATES]
    if unknown:
        raise SystemExit(f"Unknown candidates: {', '.join(unknown)}. Known: {', '.join(CANDIDATES)}")
    return [CANDIDATES[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue offline win-probability challengers; live picks are unchanged")
    parser.add_argument("--from-season", type=int, default=2023, help="First season to score")
    parser.add_argument("--to-season", type=int, default=2025, help="Last season to score")
    parser.add_argument("--candidates", help="Comma-separated candidate names (default: full queue)")
    parser.add_argument("--list", action="store_true", help="List registered candidates and exit")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_OUTPUT)
    parser.add_argument("--features-cache", type=Path, help="Pickle path for built features (default: model/artifacts/features-v6-<start>-<to>.pkl)")
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write the feature cache")
    parser.add_argument("--refresh-features", action="store_true", help="Rebuild the feature cache from nflverse")
    args = parser.parse_args()
    if args.list:
        for candidate in CANDIDATES.values():
            print(f"{candidate.name:<22}grid={len(candidate.grid):<3}calibrated={str(candidate.calibrate):<6}uses_market={str(candidate.uses_market):<6}{candidate.description}")
        return
    if args.from_season > args.to_season:
        parser.error("--from-season must not be after --to-season")
    candidates = parse_candidates(args.candidates)
    cache = None if args.no_cache else args.features_cache or FEATURE_CACHE_DIR / f"features-v6-{predict.START_SEASON}-{args.to_season}.pkl"
    features = load_features(args.to_season, cache, args.refresh_features)
    print_injury_coverage(features)
    results = run_experiments(features, args.from_season, args.to_season, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Scored {results['summary']['games']} games ({results['summary']['marketGames']} with recorded odds) from {args.from_season}-{args.to_season}; skipped {results['skippedWeeks']} weeks")
    print(format_table(results["summary"]))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
