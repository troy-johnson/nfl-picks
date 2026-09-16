from pathlib import Path
import importlib.util
import json
import math
import sys

import numpy as np
import pandas as pd

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
snapshots = load_module("snapshots")


def snapshot(season: int, game_day: str, games: list[tuple[str, str, str, float, float | None]]) -> dict:
    """games: (game_id, away, home, sameTimeHomeProbability, publishedHomeProbability or None)."""
    return {
        "provider": "The Odds API",
        "season": season,
        "gameDay": game_day,
        "capturedAt": f"{game_day}T15:30:00Z",
        "firstKickoffAt": f"{game_day}T17:00:00Z",
        "games": [
            {
                "gameId": game_id,
                "homeTeam": home,
                "awayTeam": away,
                "marketHomeProbability": market,
                "published": None if published is None else {"generatedAt": f"{game_day}T12:00:00Z", "pick": home if published >= 0.5 else away, "homeWinProbability": published, "statisticalHomeProbability": round(published - 0.05, 4), "marketHomeProbability": market, "marketWeight": 0.9},
                "bookmakers": [{"key": "a", "homeProbability": market}, {"key": "b", "homeProbability": market}],
            }
            for game_id, away, home, market, published in games
        ],
    }


def write_snapshots(root: Path, items: list[dict]) -> Path:
    for item in items:
        path = root / str(item["season"]) / f"{item['gameDay']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(item))
    return root


def synthetic_archive(weeks: int = 6, games_per_week: int = 4, seed: int = 0) -> tuple[list[dict], pd.DataFrame]:
    """Weekly snapshots plus a schedule whose outcomes follow the closing line; the same-time line is noisier."""
    rng = np.random.default_rng(seed)
    items, schedule_rows = [], []
    for week in range(1, weeks + 1):
        game_day = f"2026-09-{6 + 7 * week:02d}" if 6 + 7 * week <= 30 else f"2026-10-{7 * week - 24:02d}"
        games = []
        for index in range(games_per_week):
            home, away = f"H{week}{index}", f"A{week}{index}"
            game_id = f"2026_{week:02d}_{away}_{home}"
            closing = float(rng.uniform(0.3, 0.8))
            same_time = float(np.clip(closing + rng.normal(0, 0.03), 0.05, 0.95))
            published = float(np.clip(same_time + rng.normal(0, 0.02), 0.05, 0.95))
            home_win = int(rng.uniform() < closing)
            games.append((game_id, away, home, same_time, published))
            home_ml = -100 * closing / (1 - closing) if closing >= 0.5 else 100 * (1 - closing) / closing
            away_ml = 100 * closing / (1 - closing) if closing >= 0.5 else -100 * (1 - closing) / closing
            schedule_rows.append({"game_id": game_id, "season": 2026, "week": week, "game_type": "REG", "gameday": game_day, "home_team": home, "away_team": away, "home_score": 20 + home_win * 7, "away_score": 20 + (1 - home_win) * 7, "home_moneyline": home_ml, "away_moneyline": away_ml, "spread_line": None})
        items.append(snapshot(2026, game_day, games))
    return items, pd.DataFrame(schedule_rows)


def test_snapshot_frame_reads_same_time_market_and_published_pick():
    items = [snapshot(2026, "2026-09-17", [("2026_02_DET_BUF", "DET", "BUF", 0.66, 0.6569), ("2026_02_SEA_LA", "SEA", "LA", 0.55, None)])]
    frame = snapshots.snapshot_frame(items)
    assert list(frame.game_id) == ["2026_02_DET_BUF", "2026_02_SEA_LA"]
    assert frame.loc[0, snapshots.SAME_TIME] == 0.66
    assert frame.loc[0, snapshots.PUBLISHED] == 0.6569
    assert frame.loc[0, snapshots.STATISTICAL] == 0.6069
    assert pd.isna(frame.loc[1, snapshots.PUBLISHED]), "a game without a frozen pick stays unscored for the pick sources"
    assert frame.loc[0, "minutesBeforeFirstKickoff"] == 90
    assert frame.loc[0, "bookCount"] == 2


def test_join_results_keeps_completed_non_tied_games_and_adds_closing_line():
    items, schedule = synthetic_archive(weeks=2)
    schedule.loc[0, ["home_score", "away_score"]] = [None, None]
    schedule.loc[1, ["home_score", "away_score"]] = [24, 24]
    joined = snapshots.join_results(snapshots.snapshot_frame(items), schedule)
    assert len(joined) == 6, "one unplayed and one tied game are dropped"
    assert set(joined.columns) >= {snapshots.CLOSING, "home_win", "week", "weekBlock"}
    assert joined.weekBlock.tolist() == [202601, 202601, 202602, 202602, 202602, 202602]
    closing = schedule.set_index("game_id").loc[joined.game_id.iloc[0]]
    expected = predict.devig_home_probability(closing.home_moneyline, closing.away_moneyline)
    assert math.isclose(joined[snapshots.CLOSING].iloc[0], expected)


def test_evaluate_scores_every_source_on_identical_games():
    items, schedule = synthetic_archive(weeks=6)
    joined = snapshots.join_results(snapshots.snapshot_frame(items), schedule)
    result = snapshots.evaluate(joined, min_games=10)
    assert result["games"] == 24
    assert set(result["sources"]) == {snapshots.SAME_TIME, snapshots.CLOSING, snapshots.PUBLISHED, snapshots.STATISTICAL}
    for metrics in result["sources"].values():
        assert metrics["games"] == 24
        assert set(metrics) >= {"accuracy", "brier", "logLoss", "calibration"}
    pairs = {(c["first"], c["second"]) for c in result["comparisons"]}
    assert pairs == set(snapshots.COMPARISONS)
    same_time_vs_closing = next(c for c in result["comparisons"] if c["first"] == snapshots.SAME_TIME)
    assert same_time_vs_closing["weekBlocks"] == 6
    assert same_time_vs_closing["gamesToDetect"]["0.005"] >= same_time_vs_closing["gamesToDetect"]["0.010"]
    assert result["bySeason"]["2026"]["games"] == 24
    assert 0 <= result["sameTimeVersusClosing"]["favoriteAgreement"] <= 1
    assert result["capture"]["gamesWithPublishedPick"] == 24


def test_evaluate_drops_games_missing_any_source_so_comparisons_are_paired():
    items, schedule = synthetic_archive(weeks=3)
    items[0]["games"][0]["published"] = None
    joined = snapshots.join_results(snapshots.snapshot_frame(items), schedule)
    result = snapshots.evaluate(joined, min_games=5)
    assert result["snapshotGames"] == 12
    assert result["games"] == 11
    assert all(metrics["games"] == 11 for metrics in result["sources"].values())


def test_attach_candidates_adds_prefixed_columns_and_compares_to_same_time_market():
    items, schedule = synthetic_archive(weeks=3)
    joined = snapshots.join_results(snapshots.snapshot_frame(items), schedule)
    payload = {"predictions": [{"gameId": game_id, "candidates": {"elo_tuned_injury": {"probability": 0.5}}} for game_id in joined.game_id]}
    with_candidates = snapshots.attach_candidates(joined, payload)
    assert "candidate:elo_tuned_injury" in with_candidates.columns
    result = snapshots.evaluate(with_candidates, min_games=5)
    assert math.isclose(result["sources"]["candidate:elo_tuned_injury"]["brier"], 0.25)
    assert ("candidate:elo_tuned_injury", snapshots.SAME_TIME) in {(c["first"], c["second"]) for c in result["comparisons"]}


def test_claim_status_requires_games_and_two_standard_errors_on_both_scores():
    strong = {"brier": -0.01, "brierWeekBlockStandardError": 0.002, "logLoss": -0.02, "logLossWeekBlockStandardError": 0.005}
    assert snapshots.claim_status(strong, 100, min_games=250).startswith("insufficient games")
    assert snapshots.claim_status(strong, 300, min_games=250).startswith("beats second source")
    mixed = {**strong, "logLoss": -0.001}
    assert snapshots.claim_status(mixed, 300, min_games=250) == "no significant difference"
    worse = {"brier": 0.01, "brierWeekBlockStandardError": 0.002, "logLoss": 0.02, "logLossWeekBlockStandardError": 0.005}
    assert snapshots.claim_status(worse, 300, min_games=250).startswith("worse than second source")
    assert snapshots.claim_status({**strong, "brierWeekBlockStandardError": float("nan")}, 300, min_games=250) == "insufficient week blocks"


def test_detectable_games_scales_with_standard_error():
    assert snapshots.detectable_games(0.005, 100, 0.010) == 100
    assert snapshots.detectable_games(0.005, 100, 0.005) == 400
    assert snapshots.detectable_games(float("nan"), 100, 0.005) is None


def test_empty_archive_reports_no_games(tmp_path):
    result, frame = snapshots.run(tmp_path / "missing", None, 250)
    assert result["games"] == 0
    assert frame.empty
    assert snapshots.format_report(result) == "No completed games with same-time odds yet."


def test_load_snapshots_reads_season_folders_in_order(tmp_path):
    items, _ = synthetic_archive(weeks=2)
    root = write_snapshots(tmp_path, list(reversed(items)))
    loaded = snapshots.load_snapshots(root)
    assert [item["gameDay"] for item in loaded] == sorted(item["gameDay"] for item in items)


def test_single_week_archive_serialises_without_nan():
    items, schedule = synthetic_archive(weeks=1)
    joined = snapshots.join_results(snapshots.snapshot_frame(items), schedule)
    result = snapshots.evaluate(joined, min_games=250)
    comparison = result["comparisons"][0]
    assert comparison["brierWeekBlockStandardError"] is None
    assert comparison["gamesToDetect"]["0.005"] is None
    assert "NaN" not in json.dumps(result)
    assert "SE n/a" in snapshots.format_report(result)
