from pathlib import Path
import importlib.util
import math
import sys

import numpy as np
import pandas as pd
import pytest

MODEL_DIR = Path(__file__).parents[1] / "model"


def load_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


predict = load_module("predict")
experiments = load_module("experiments")

OUTCOME_COLUMNS = {"home_win", "home_score", "away_score", "home_qb_name", "away_qb_name", "market_home_prob", "home_moneyline", "away_moneyline", "spread_line"}
MARKET_COLUMNS = {"market_home_prob", "market_logit"}
FAST_QUEUE = ["baseline", "regularized_logistic", "boosted_trees"]


def synthetic_features(seed: int = 0, seasons=(2021, 2022, 2023), weeks: int = 4, games_per_week: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = sorted({feature for candidate in experiments.CANDIDATES.values() for feature in candidate.features})
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for game in range(games_per_week):
                values = {column: float(rng.normal()) for column in columns}
                values["neutral_site"] = 0.0
                strength = values["diff_off_epa"] - values["diff_def_epa_allowed"] + 0.5 * values["elo_diff"]
                probability = 1 / (1 + math.exp(-strength))
                rows.append({
                    "game_id": f"{season}_{week:02d}_A{game}_H{game}",
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-09-{week + 8:02d}",
                    "gametime": "13:00",
                    "home_team": f"H{game}",
                    "away_team": f"A{game}",
                    "home_win": int(rng.random() < probability),
                    "market_home_prob": float(np.clip(probability + rng.normal(0, 0.05), 0.05, 0.95)),
                    **values,
                })
    return pd.DataFrame(rows)


def run_fast(features: pd.DataFrame, names=FAST_QUEUE) -> dict:
    return experiments.run_experiments(features, 2023, 2023, [experiments.CANDIDATES[name] for name in names], min_training_games=8, min_market_games=4)


def test_registry_covers_requested_candidates():
    assert {"baseline", "elo", "qb", "elo_qb", "elo_injury", "elo_warm", "elo_mov", "elo_mov_injury", "elo_mov2", "elo_mov2_injury", "elo_tuned_injury", "market_stack", "regularized_logistic", "boosted_trees", "random_forest"} <= set(experiments.CANDIDATES)
    assert experiments.CANDIDATES["baseline"].features == tuple(predict.FEATURES)
    assert experiments.CANDIDATES["elo"].features == tuple(predict.ELO_FEATURES)
    assert experiments.CANDIDATES["qb"].features == tuple(predict.QB_FEATURES)
    assert experiments.CANDIDATES["elo_qb"].features == tuple(predict.ELO_QB_FEATURES)
    assert experiments.CANDIDATES["elo_injury"].features == tuple(experiments.INJURY_FEATURES)
    assert not experiments.CANDIDATES["elo_injury"].uses_market
    assert experiments.CANDIDATES["elo_warm"].features == tuple(predict.FEATURES) + ("elo_diff_warm",)
    assert experiments.CANDIDATES["elo_mov"].features == tuple(predict.FEATURES) + ("elo_diff_mov",)
    assert experiments.CANDIDATES["elo_mov_injury"].features == tuple(predict.FEATURES) + ("elo_diff_mov", "diff_injury_load", "diff_qb_injury")
    assert experiments.CANDIDATES["elo_mov2"].features == tuple(predict.FEATURES) + ("elo_diff_mov2",)
    assert experiments.CANDIDATES["elo_mov2_injury"].features == tuple(predict.FEATURES) + ("elo_diff_mov2", "diff_injury_load", "diff_qb_injury")
    assert experiments.CANDIDATES["elo_tuned_injury"].features == tuple(predict.FEATURES) + ("elo_diff_tuned", "diff_injury_load", "diff_qb_injury")
    assert experiments.ELO_TEAM_ALIASES == {"OAK": "LV", "SD": "LAC", "STL": "LA"}
    assert len(experiments.ELO_GRID_PARAMS) == 18
    assert experiments.CANDIDATES["market_stack"].features == tuple(experiments.INJURY_FEATURES + ["market_logit"])
    assert experiments.CANDIDATES["market_stack"].uses_market
    assert len(experiments.CANDIDATES["regularized_logistic"].grid) > 1
    assert experiments.CANDIDATES["boosted_trees"].calibrate
    for name in experiments.DEFAULT_QUEUE:
        assert name in experiments.CANDIDATES
    assert {"elo_injury", "elo_warm", "elo_mov", "elo_mov_injury", "elo_mov2", "elo_mov2_injury", "elo_tuned_injury", "market_stack"} <= set(experiments.DEFAULT_QUEUE)
    queue = list(experiments.DEFAULT_QUEUE)
    assert queue[queue.index("elo_injury") + 1:queue.index("elo_injury") + 4] == ["elo_warm", "elo_mov", "elo_mov_injury"]
    assert queue[queue.index("elo_mov_injury") + 1:queue.index("elo_mov_injury") + 4] == ["elo_mov2", "elo_mov2_injury", "elo_tuned_injury"]
    assert "random_forest" not in experiments.DEFAULT_QUEUE
    assert experiments.parse_candidates("boosted_trees,random_forest")[-1].name == "random_forest"


def test_no_candidate_uses_outcome_or_target_game_starter_columns():
    for candidate in experiments.CANDIDATES.values():
        assert not OUTCOME_COLUMNS & set(candidate.features), candidate.name
        if not candidate.uses_market:
            assert not MARKET_COLUMNS & set(candidate.features), candidate.name
        else:
            assert "market_logit" in candidate.features, candidate.name


def test_baseline_candidate_reproduces_live_pipeline():
    features = synthetic_features()
    live, candidate = predict.new_pipeline(), experiments.CANDIDATES["baseline"].build({})
    live.fit(features[predict.FEATURES], features.home_win)
    candidate.fit(features[list(experiments.CANDIDATES["baseline"].features)], features.home_win)
    assert np.allclose(live.predict_proba(features[predict.FEATURES])[:, 1], candidate.predict_proba(features[predict.FEATURES])[:, 1])


def test_run_experiments_scores_identical_games_for_every_candidate():
    results = run_fast(synthetic_features())
    predictions = results["predictions"]
    assert len(predictions) == 16
    assert len(results["weeks"]) == 4
    assert results["skippedWeeks"] == 0
    for prediction in predictions:
        assert set(prediction["candidates"]) == set(FAST_QUEUE)
        for item in prediction["candidates"].values():
            assert 0 <= item["probability"] <= 1
            assert 0 <= item["blendedProbability"] <= 1
            assert 0 <= item["marketWeight"] <= 1
    for week in results["weeks"]:
        assert week["validationSeason"] == 2022
        assert set(week["candidates"]) == set(FAST_QUEUE)
        assert week["candidates"]["regularized_logistic"]["hyperparameters"] in experiments.CANDIDATES["regularized_logistic"].grid
        assert week["candidates"]["boosted_trees"]["calibration"]["slope"] != 1.0
        assert week["candidates"]["baseline"]["calibration"] == {"intercept": 0.0, "slope": 1.0}
    summary = results["summary"]
    assert set(summary["candidates"]) == set(FAST_QUEUE)
    assert summary["games"] == 16
    for item in summary["candidates"].values():
        assert item["marketGames"]["raw"]["games"] == summary["marketGames"]
        assert sum(row["games"] for row in item["calibration"]["bins"]) == summary["marketGames"]
        assert {"brier", "logLoss", "accuracy"} <= set(item["marketGames"]["blended"])
        assert "brier" in item["marketGames"]["rawVersusMarket"]
    assert summary["candidates"]["baseline"]["verdict"] == "baseline"
    assert set(summary["rankingByLogLoss"]) == set(FAST_QUEUE)
    assert set(summary["probabilityCorrelation"]) == set(FAST_QUEUE) | {"market"}


def test_training_games_count_only_earlier_weeks():
    results = run_fast(synthetic_features())
    for prediction in results["predictions"]:
        expected = 2 * 16 + (prediction["week"] - 1) * 4
        assert prediction["trainingGames"] == expected


def test_target_week_and_later_outcomes_do_not_change_predictions():
    features = synthetic_features()
    altered = features.copy()
    later = (altered.season == 2023) & (altered.week >= 3)
    altered.loc[later, "home_win"] = 1 - altered.loc[later, "home_win"]
    original, changed = run_fast(features), run_fast(altered)
    original_rows = {row["gameId"]: row for row in original["predictions"] if row["week"] <= 3}
    changed_rows = {row["gameId"]: row for row in changed["predictions"] if row["week"] <= 3}
    assert original_rows.keys() == changed_rows.keys()
    for game_id, row in original_rows.items():
        for name in FAST_QUEUE:
            assert row["candidates"][name] == changed_rows[game_id]["candidates"][name], (game_id, name)
    week_four = [row for row in changed["predictions"] if row["week"] == 4]
    assert week_four and week_four[0]["candidates"]["baseline"] != next(row for row in original["predictions"] if row["gameId"] == week_four[0]["gameId"])["candidates"]["baseline"]


def test_select_hyperparameters_returns_lowest_validation_log_loss():
    features = synthetic_features()
    candidate = experiments.CANDIDATES["regularized_logistic"]
    train, validation = features[features.season == 2021], features[features.season == 2022]
    params, probabilities = experiments.select_hyperparameters(candidate, train, validation)
    assert params in candidate.grid
    assert len(probabilities) == len(validation)
    scores = {}
    for grid_params in candidate.grid:
        model = candidate.build(grid_params)
        model.fit(train[list(candidate.features)], train.home_win)
        scores[grid_params["C"]] = experiments.log_loss(validation.home_win, model.predict_proba(validation[list(candidate.features)])[:, 1], labels=[0, 1])
    assert params["C"] == min(scores, key=scores.get)


def test_platt_calibrator_is_near_identity_for_calibrated_probabilities():
    rng = np.random.default_rng(1)
    probabilities = rng.uniform(0.05, 0.95, 20000)
    actual = (rng.random(20000) < probabilities).astype(int)
    calibrator = experiments.PlattCalibrator.fit(probabilities, actual)
    assert abs(calibrator.intercept) < 0.1
    assert abs(calibrator.slope - 1) < 0.1
    assert np.allclose(experiments.IDENTITY_CALIBRATOR.apply(probabilities), probabilities)


def test_platt_calibrator_corrects_overconfident_probabilities():
    rng = np.random.default_rng(2)
    true = rng.uniform(0.3, 0.7, 20000)
    actual = (rng.random(20000) < true).astype(int)
    overconfident = 1 / (1 + np.exp(-3 * experiments.logit(true)))
    calibrator = experiments.PlattCalibrator.fit(overconfident, actual)
    assert calibrator.slope < 0.5
    assert abs(calibrator.apply(overconfident).mean() - true.mean()) < 0.02


def test_calibration_report_covers_every_game():
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0, 1, 500)
    actual = (rng.random(500) < probabilities).astype(int)
    report = experiments.calibration_report(actual, probabilities)
    assert sum(row["games"] for row in report["bins"]) == 500
    assert 0 <= report["expectedCalibrationError"] <= 1
    assert all(row["lower"] <= row["meanPredicted"] <= row["upper"] for row in report["bins"])


def test_paired_difference_is_zero_for_identical_probabilities():
    probabilities = np.array([0.6, 0.4, 0.7, 0.2])
    actual = np.array([1, 0, 0, 1])
    weeks = np.array(["w1", "w1", "w2", "w2"])
    result = experiments.paired_difference(probabilities, probabilities, actual, weeks)
    assert result["brier"] == 0
    assert result["logLoss"] == 0
    assert result["weekBlocks"] == 2


def test_verdict_rejects_accuracy_only_gains():
    baseline = {"accuracy": 0.62, "brier": 0.2277, "logLoss": 0.6465}
    assert experiments.verdict({"accuracy": 0.65, "brier": 0.2300, "logLoss": 0.6500}, baseline).startswith("rejected: accuracy gain")
    assert experiments.verdict({"accuracy": 0.61, "brier": 0.2250, "logLoss": 0.6400}, baseline).startswith("improves")
    assert experiments.verdict({"accuracy": 0.60, "brier": 0.2300, "logLoss": 0.6500}, baseline) == "rejected: does not improve baseline"


def test_parse_candidates_rejects_unknown_names():
    assert [candidate.name for candidate in experiments.parse_candidates(None)] == list(experiments.DEFAULT_QUEUE)
    assert [candidate.name for candidate in experiments.parse_candidates("elo, qb")] == ["elo", "qb"]
    try:
        experiments.parse_candidates("nope")
    except SystemExit as exc:
        assert "Unknown candidates" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def synthetic_schedule(seed: int = 5, seasons=(2021, 2022, 2023), weeks: int = 6, games_per_week: int = 4) -> pd.DataFrame:
    """Schedule-like frame with decided scores for the Elo walk tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for game in range(games_per_week):
                home_score, away_score = int(rng.integers(0, 45)), int(rng.integers(0, 45))
                if home_score == away_score:
                    away_score += 3
                rows.append({
                    "game_id": f"{season}_{week:02d}_A{game}_H{game}",
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-10-{week:02d}",
                    "gametime": "13:00",
                    "home_team": f"H{game}",
                    "away_team": f"A{game}",
                    "home_score": home_score,
                    "away_score": away_score,
                })
    return pd.DataFrame(rows)


def test_pregame_elo_v2_without_mov_reproduces_predict():
    frame = synthetic_schedule()
    assert experiments.pregame_elo_v2(frame) == predict.pregame_elo_differences(frame)


def test_mov_factor_caps_margin_and_grows_with_margin():
    def margin_schedule(margin: int) -> pd.DataFrame:
        return pd.DataFrame([
            {"game_id": "2021_01_A0_H0", "season": 2021, "week": 1, "gameday": "2021-10-01", "gametime": "13:00", "home_team": "H0", "away_team": "A0", "home_score": margin, "away_score": 0},
            {"game_id": "2021_02_A1_H0", "season": 2021, "week": 2, "gameday": "2021-10-08", "gametime": "13:00", "home_team": "H0", "away_team": "A1", "home_score": 20, "away_score": 17},
        ])

    capped = experiments.pregame_elo_v2(margin_schedule(30), mov_cap=24)
    at_cap = experiments.pregame_elo_v2(margin_schedule(24), mov_cap=24)
    small = experiments.pregame_elo_v2(margin_schedule(3), mov_cap=24)
    assert capped == at_cap
    assert capped["2021_02_A1_H0"] > small["2021_02_A1_H0"]


def test_mov_scores_do_not_leak_into_earlier_weeks():
    frame = synthetic_schedule()
    altered = frame.copy()
    target = (altered.season == 2021) & (altered.week == 2) & (altered.game_id == "2021_02_A0_H0")
    row = altered[target].iloc[0]
    base_margin = abs(int(row.home_score) - int(row.away_score))
    new_margin = 3 if min(base_margin, 24) == 24 else 30
    if row.home_score > row.away_score:
        altered.loc[target, "home_score"] = row.away_score + new_margin
    else:
        altered.loc[target, "away_score"] = row.home_score + new_margin
    base, changed = experiments.pregame_elo_v2(frame, mov_cap=24), experiments.pregame_elo_v2(altered, mov_cap=24)
    earlier = {game_id for game_id in base if int(game_id.split("_")[0]) == 2021 and int(game_id.split("_")[1]) <= 2}
    for game_id in earlier:
        assert changed[game_id] == base[game_id], game_id
    affected = [game_id for game_id in base if int(game_id.split("_")[0]) == 2021 and int(game_id.split("_")[1]) == 3 and ("A0" in game_id or "H0" in game_id)]
    assert affected and any(changed[game_id] != base[game_id] for game_id in affected)


def test_warmup_seasons_break_the_cold_start():
    warm = experiments.pregame_elo_v2(synthetic_schedule(seasons=(2019, 2020, 2021)))
    cold = experiments.pregame_elo_v2(synthetic_schedule(seasons=(2021,)))
    warm_first_week = [value for game_id, value in warm.items() if game_id.startswith("2021_01")]
    cold_first_week = [value for game_id, value in cold.items() if game_id.startswith("2021_01")]
    assert set(cold_first_week) == {predict.ELO_HOME_ADVANTAGE}
    assert len(set(warm_first_week)) > 1


def relocation_schedule() -> pd.DataFrame:
    """Two-season schedule where one franchise switches codes from OAK to LV after season 1."""
    return pd.DataFrame([
        {"game_id": "1_01_OPP_OAK", "season": 1, "week": 1, "gameday": "1-10-01", "gametime": "13:00", "home_team": "OAK", "away_team": "OPP", "home_score": 30, "away_score": 20},
        {"game_id": "2_01_OPP_LV", "season": 2, "week": 1, "gameday": "2-10-01", "gametime": "13:00", "home_team": "LV", "away_team": "NEW", "home_score": 21, "away_score": 10},
    ])


def test_elo_aliases_carry_ratings_across_relocation():
    frame = relocation_schedule()
    unaliased = experiments.pregame_elo_v2(frame)
    assert unaliased["2_01_OPP_LV"] == predict.ELO_HOME_ADVANTAGE
    aliased = experiments.pregame_elo_v2(frame, aliases={"OAK": "LV"})
    probability = 1 / (1 + 10 ** (-predict.ELO_HOME_ADVANTAGE / 400))
    oak_final = predict.ELO_INITIAL_RATING + predict.ELO_K_FACTOR * (1 - probability)
    oak_regressed = predict.ELO_INITIAL_RATING + predict.ELO_SEASON_REGRESSION * (oak_final - predict.ELO_INITIAL_RATING)
    assert aliased["2_01_OPP_LV"] == pytest.approx(oak_regressed - predict.ELO_INITIAL_RATING + predict.ELO_HOME_ADVANTAGE)


def test_elo_aliases_do_not_mutate_the_input_frame():
    frame = relocation_schedule()
    before = frame.copy(deep=True)
    experiments.pregame_elo_v2(frame, aliases={"OAK": "LV"})
    experiments.pregame_elo_v2(frame)
    pd.testing.assert_frame_equal(frame, before)


def upset_margin_schedules(margin: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same week-1 margin, once won by the home favourite and once by the away underdog."""
    favourite = pd.DataFrame([
        {"game_id": "2021_01_A0_H0", "season": 2021, "week": 1, "gameday": "2021-10-01", "gametime": "13:00", "home_team": "H0", "away_team": "A0", "home_score": 20 + margin, "away_score": 20},
        {"game_id": "2021_02_A1_H0", "season": 2021, "week": 2, "gameday": "2021-10-08", "gametime": "13:00", "home_team": "H0", "away_team": "A1", "home_score": 20, "away_score": 17},
    ])
    underdog = pd.DataFrame([
        {"game_id": "2021_01_A0_H0", "season": 2021, "week": 1, "gameday": "2021-10-01", "gametime": "13:00", "home_team": "H0", "away_team": "A0", "home_score": 20, "away_score": 20 + margin},
        {"game_id": "2021_02_A1_A0", "season": 2021, "week": 2, "gameday": "2021-10-08", "gametime": "13:00", "home_team": "A0", "away_team": "A1", "home_score": 20, "away_score": 17},
    ])
    return favourite, underdog


def test_signed_mov_factor_moves_upset_wins_more_than_expected_wins():
    favourite, underdog = upset_margin_schedules(10)
    unsigned_favourite = experiments.pregame_elo_v2(favourite, mov_cap=24)
    unsigned_underdog = experiments.pregame_elo_v2(underdog, mov_cap=24)
    signed_favourite = experiments.pregame_elo_v2(favourite, mov_cap=24, signed_mov=True)
    signed_underdog = experiments.pregame_elo_v2(underdog, mov_cap=24, signed_mov=True)
    # An expected win keeps the same factor signed or not, because the winner's gap is positive.
    assert signed_favourite["2021_02_A1_H0"] == unsigned_favourite["2021_02_A1_H0"]
    # An upset gets a larger factor when the gap is signed, so the underdog carries more rating forward.
    assert signed_underdog["2021_02_A1_A0"] > unsigned_underdog["2021_02_A1_A0"]
    # For the same margin, the underdog winner moves more than the favourite winner.
    assert signed_underdog["2021_02_A1_A0"] > signed_favourite["2021_02_A1_H0"]
    probability = 1 / (1 + 10 ** (-predict.ELO_HOME_ADVANTAGE / 400))
    signed_underdog_gain = predict.ELO_K_FACTOR * probability * math.log(10 + 1) * 2.2 / (2.2 - predict.ELO_HOME_ADVANTAGE * 0.001)
    assert signed_underdog["2021_02_A1_A0"] == pytest.approx(signed_underdog_gain + predict.ELO_HOME_ADVANTAGE)


def tuned_features(seed: int = 3) -> pd.DataFrame:
    """Synthetic frame that also carries the 18 grid columns as noisy Elo differences."""
    features = synthetic_features(seed)
    rng = np.random.default_rng(seed + 50)
    for column in experiments.ELO_GRID_PARAMS:
        features[column] = rng.normal(0, 60, size=len(features))
    return features


def test_select_elo_parameters_uses_all_history_rows_not_validation_alone():
    history = tuned_features()[lambda frame: frame.season < 2023].copy().reset_index(drop=True)
    validation = history.season == 2022
    perfect_column, steady_column = "elo_grid_k16_h48_r0.5", "elo_grid_k20_h65_r0.8"
    history[perfect_column] = np.where(history.home_win == 1, 800.0, -800.0)
    history.loc[~validation, perfect_column] = np.where(history.loc[~validation, "home_win"] == 1, -800.0, 800.0)
    history[steady_column] = np.where(history.home_win == 1, 200.0, -200.0)
    column, params = experiments.select_elo_parameters(history)
    assert column == steady_column
    assert params == experiments.ELO_GRID_PARAMS[steady_column]


def test_tuned_elo_selection_does_not_leak_target_week_outcomes():
    features = tuned_features()
    altered = features.copy()
    later = (altered.season == 2023) & (altered.week >= 3)
    altered.loc[later, "home_win"] = 1 - altered.loc[later, "home_win"]
    names = ["elo_mov2_injury", "elo_tuned_injury"]
    original, changed = run_fast(features, names), run_fast(altered, names)
    original_weeks = {week["week"]: week for week in original["weeks"]}
    changed_weeks = {week["week"]: week for week in changed["weeks"]}
    assert set(original_weeks) == {1, 2, 3, 4}
    for week in (1, 2, 3):
        assert original_weeks[week]["eloParameters"] == changed_weeks[week]["eloParameters"], week
        assert original_weeks[week]["candidates"] == changed_weeks[week]["candidates"], week
        assert original_weeks[week]["eloParameters"] in experiments.ELO_GRID_PARAMS.values()
    original_rows = {row["gameId"]: row for row in original["predictions"] if row["week"] <= 3}
    changed_rows = {row["gameId"]: row for row in changed["predictions"] if row["week"] <= 3}
    for game_id, row in original_rows.items():
        for name in names:
            assert row["candidates"][name] == changed_rows[game_id]["candidates"][name], (game_id, name)
    week_four = [row for row in changed["predictions"] if row["week"] == 4]
    assert week_four[0]["candidates"]["elo_tuned_injury"] != next(row for row in original["predictions"] if row["gameId"] == week_four[0]["gameId"])["candidates"]["elo_tuned_injury"]
    counts = original["summary"]["eloParameterCounts"]
    assert sum(entry["weeks"] for entry in counts) == 4
    assert all(entry["column"] in experiments.ELO_GRID_PARAMS for entry in counts)


def injury_test_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two 2023 week-5 games plus injury rows hitting every rule of build_injury_features."""
    games = pd.DataFrame({
        "game_id": ["2023_05_BUF_KC", "2023_05_SF_ARI"],
        "season": [2023, 2023],
        "week": [5, 5],
        "gameday": ["2023-10-08", "2023-10-08"],
        "home_team": ["KC", "ARI"],
        "away_team": ["BUF", "SF"],
    })
    injuries = pd.DataFrame({
        "season": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023],
        "game_type": ["REG", "REG", "REG", "REG", "REG", "REG", "REG", "REG", "WC"],
        "team": ["KC", "KC", "KC", "KC", "KC", "BUF", "BUF", "BUF", "KC"],
        "week": [5, 5, 5, 5, 5, 5, 5, 5, 5],
        "position": ["WR", "QB", "LB", "TE", "RB", "QB", "WR", "RB", "WR"],
        "report_status": ["Out", "Doubtful", "Questionable", np.nan, "Out", "Questionable", "Out", "Out", "Out"],
        "date_modified": [
            "2023-10-07T20:00:00Z",  # KC WR Out, day before gameday -> home load 1.0
            "2023-10-07T20:00:00Z",  # KC QB Doubtful, day before -> home qb 0.75
            "2023-10-09T20:00:00Z",  # KC LB Questionable, after gameday -> excluded
            "2023-10-07T18:00:00Z",  # KC TE NaN status -> contributes 0
            np.nan,                  # KC RB Out but no date_modified -> excluded
            "2023-10-07T02:00:00Z",  # BUF QB Questionable, day before -> away qb 0.25
            "2023-10-06T23:00:00Z",  # BUF WR Out, two days before -> away load 1.0
            "2023-10-08T01:00:00Z",  # BUF RB Out, on gameday UTC date -> excluded
            "2023-10-07T20:00:00Z",  # playoff row, filtered by game_type -> excluded
        ],
    })
    return games, injuries


def test_build_injury_features_applies_weights_and_pregame_date_rule():
    games, injuries = injury_test_frames()
    result = experiments.build_injury_features(games, injuries)
    assert list(result.home_injury_load) == [1.0, 0.0]
    assert list(result.away_injury_load) == [1.0, 0.0]
    assert list(result.home_qb_injury) == [0.75, 0.0]
    assert list(result.away_qb_injury) == [0.25, 0.0]
    assert list(result.diff_injury_load) == [0.0, 0.0]
    assert list(result.diff_qb_injury) == [0.5, 0.0]


def test_build_injury_features_defaults_to_zero_without_injuries():
    games, _ = injury_test_frames()
    result = experiments.build_injury_features(games, pd.DataFrame())
    for column in ("home_injury_load", "away_injury_load", "home_qb_injury", "away_qb_injury", "diff_injury_load", "diff_qb_injury"):
        assert list(result[column]) == [0.0, 0.0], column


def test_build_injury_features_rejects_unknown_team_codes():
    games, injuries = injury_test_frames()
    injuries.loc[len(injuries)] = [2023, "REG", "WSH", 5, "WR", "Out", "2023-10-07T20:00:00Z"]
    with pytest.raises(AssertionError, match="missing from the schedule"):
        experiments.build_injury_features(games, injuries)


def test_normalize_injury_frame_accepts_missing_date_modified():
    frame = pd.DataFrame({
        "season": [2025, 2025],
        "game_type": ["REG", "REG"],
        "team": ["KC", "BUF"],
        "week": [3, 3],
        "position": ["WR", "QB"],
        "report_status": ["Out", "Doubtful"],
    })
    normalized = experiments.normalize_injury_frame(frame, 2025)
    assert list(normalized.columns) == experiments.INJURY_COLUMNS + ["has_timestamp"]
    assert normalized.has_timestamp.eq(False).all()
    assert normalized.date_modified.isna().all()
    assert list(normalized.report_status) == ["Out", "Doubtful"]
    with_timestamp = experiments.normalize_injury_frame(frame.assign(date_modified="2025-09-19T20:00:00Z"), 2025)
    assert with_timestamp.has_timestamp.eq(True).all()
    assert with_timestamp.date_modified.notna().all()


def test_build_injury_features_counts_rows_without_timestamp():
    games = pd.DataFrame({
        "game_id": ["2025_05_BUF_KC"],
        "season": [2025],
        "week": [5],
        "gameday": ["2025-10-08"],
        "home_team": ["KC"],
        "away_team": ["BUF"],
    })
    injuries = pd.DataFrame({
        "season": [2025, 2025, 2025, 2025],
        "game_type": ["REG", "REG", "REG", "REG"],
        "team": ["KC", "KC", "BUF", "BUF"],
        "week": [5, 5, 5, 5],
        "position": ["WR", "QB", "WR", "RB"],
        "report_status": ["Out", "Questionable", "Out", "Out"],
        "date_modified": [np.nan, "2025-10-09T20:00:00Z", np.nan, "2025-10-08T20:00:00Z"],
        "has_timestamp": [False, True, False, True],
    })
    result = experiments.build_injury_features(games, injuries)
    assert list(result.home_injury_load) == [1.0]
    assert list(result.home_qb_injury) == [0.0]
    assert list(result.away_injury_load) == [1.0]
    assert list(result.away_qb_injury) == [0.0]


def test_summary_by_season_matches_pooled_games():
    names = FAST_QUEUE + ["elo"]
    results = run_fast(synthetic_features(), names)
    summary = results["summary"]
    seasons = sorted({row["season"] for row in results["predictions"]})
    assert summary["marketGames"] == len(results["predictions"])
    for name in names:
        by_season = summary["candidates"][name]["bySeason"]
        assert sorted(int(key) for key in by_season) == seasons
        assert sum(entry["games"] for entry in by_season.values()) == summary["games"]
        expected_keys = {"games", "raw", "market", "rawVersusMarket"}
        expected_keys |= set() if name == "baseline" else {"rawVersusBaseline"}
        for entry in by_season.values():
            assert expected_keys <= set(entry)
        assert ("rawVersusElo" in by_season["2023"]) == (name != "elo")
    market_by_season = summary["marketBySeason"]
    assert sorted(int(key) for key in market_by_season) == seasons
    assert sum(entry["games"] for entry in market_by_season.values()) == summary["games"]


def test_paired_difference_restricted_to_one_season_counts_its_weeks():
    seasons = np.array(["2023"] * 6 + ["2024"] * 4)
    weeks = np.array(["2023-01", "2023-01", "2023-02", "2023-02", "2023-03", "2023-03", "2024-01", "2024-01", "2024-02", "2024-02"])
    probabilities = np.full(10, 0.6)
    actual = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    one_season = seasons == "2023"
    result = experiments.paired_difference(probabilities[one_season], probabilities[one_season] + 0.1, actual[one_season], weeks[one_season])
    assert result["weekBlocks"] == 3


def market_driven_features(seed: int = 11, seasons=(2020, 2021, 2022, 2023), weeks: int = 18, games_per_week: int = 14) -> pd.DataFrame:
    """Synthetic frame where home_win is drawn from market_home_prob and every model feature is noise."""
    rng = np.random.default_rng(seed)
    columns = list(experiments.CANDIDATES["market_stack"].features)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for game in range(games_per_week):
                probability = float(rng.uniform(0.1, 0.9))
                rows.append({
                    "game_id": f"{season}_{week:02d}_A{game}_H{game}",
                    "season": season,
                    "week": week,
                    "gameday": f"{season}-10-{week + 6:02d}",
                    "home_team": f"H{game}",
                    "away_team": f"A{game}",
                    "home_win": int(rng.random() < probability),
                    "market_home_prob": probability,
                    **{column: float(rng.normal()) for column in columns if column != "market_logit"},
                })
    return pd.DataFrame(rows)


def test_market_stack_recovers_market_probabilities():
    features = market_driven_features()
    results = experiments.run_experiments(
        features, 2023, 2023, [experiments.CANDIDATES["market_stack"]], min_training_games=8, min_market_games=4
    )
    coefficients = [week["candidates"]["market_stack"]["marketLogitCoefficient"] for week in results["weeks"]]
    assert coefficients and all(0.7 <= value <= 1.3 for value in coefficients), coefficients
    summary = results["summary"]["candidates"]["market_stack"]["marketGames"]["raw"]
    market = results["summary"]["market"]
    assert abs(summary["logLoss"] - market["logLoss"]) < 0.01


def test_market_stack_outputs_nan_when_target_game_lacks_market():
    features = market_driven_features()
    features.loc[(features.season == 2023) & (features.week == 1) & (features.game_id.str.endswith("_H0")), "market_home_prob"] = np.nan
    total = int((features.season == 2023).sum())
    results = experiments.run_experiments(
        features, 2023, 2023, [experiments.CANDIDATES["baseline"], experiments.CANDIDATES["market_stack"]], min_training_games=8, min_market_games=4
    )
    missing = [row for row in results["predictions"] if row["marketHomeProbability"] is None]
    assert len(missing) == 1
    assert math.isnan(missing[0]["candidates"]["market_stack"]["probability"])
    assert not math.isnan(missing[0]["candidates"]["baseline"]["probability"])
    candidates = results["summary"]["candidates"]
    assert results["summary"]["marketGames"] == total - 1
    assert candidates["market_stack"]["allGames"]["games"] == total - 1
    assert candidates["baseline"]["allGames"]["games"] == total
    assert candidates["market_stack"]["marketGames"]["raw"]["games"] == total - 1


def test_new_candidates_match_output_shape():
    names = FAST_QUEUE + ["elo_injury", "market_stack"]
    results = run_fast(synthetic_features(), names)
    expected_keys = {"probability", "blendedProbability", "marketWeight"}
    for prediction in results["predictions"]:
        assert set(prediction["candidates"]) == set(names)
        keys = {frozenset(item) for item in prediction["candidates"].values()}
        assert keys == {frozenset(expected_keys)}
        for item in prediction["candidates"].values():
            assert 0 <= item["probability"] <= 1
            assert 0 <= item["blendedProbability"] <= 1
    for week in results["weeks"]:
        assert set(week["candidates"]) == set(names)
        assert "marketLogitCoefficient" in week["candidates"]["market_stack"]
        assert "marketLogitCoefficient" not in week["candidates"]["baseline"]
    summary = results["summary"]
    assert set(summary["candidates"]) == set(names)
    assert set(summary["rankingByLogLoss"]) == set(names)
    assert set(summary["probabilityCorrelation"]) == set(names) | {"market"}
