from pathlib import Path
import importlib.util
import sys
from datetime import datetime, timezone
import pandas as pd
SPEC=importlib.util.spec_from_file_location("predict",Path(__file__).parents[1]/"model"/"predict.py")
predict=importlib.util.module_from_spec(SPEC);sys.modules["predict"]=predict
assert SPEC.loader is not None
SPEC.loader.exec_module(predict)
def test_american_implied_probability():
    assert round(predict.american_implied(-150),3)==0.6
    assert round(predict.american_implied(150),3)==0.4
def test_devig_probability_removes_vig():
    assert round(predict.devig_home_probability(-110,-110),3)==0.5
def test_favorite_confidence_uses_stronger_side():
    assert predict.confidence(0.74)=="high"
    assert predict.confidence(0.30)=="medium"
    assert predict.confidence(0.58)=="low"

def test_upcoming_week_uses_kickoff_not_missing_scores():
    schedule=pd.DataFrame([
        {"season":2026,"game_type":"REG","week":1,"gameday":"2026-09-10","gametime":"20:15","home_score":None},
        {"season":2026,"game_type":"REG","week":1,"gameday":"2026-09-14","gametime":"20:15","home_score":None},
        {"season":2026,"game_type":"REG","week":2,"gameday":"2026-09-17","gametime":"20:15","home_score":None},
    ])
    now=datetime(2026,9,15,1,tzinfo=timezone.utc)
    week,kickoff=predict.upcoming_week_and_kickoff(schedule,2026,now)
    assert week==2
    assert kickoff==datetime(2026,9,18,0,15,tzinfo=timezone.utc)

def test_refresh_stage_allows_midweek_and_pre_kickoff_runs():
    kickoff=datetime(2026,9,18,0,15,tzinfo=timezone.utc)
    assert predict.refresh_stage(datetime(2026,9,16,16,tzinfo=timezone.utc),kickoff)=="initial"
    assert predict.refresh_stage(datetime(2026,9,17,22,45,tzinfo=timezone.utc),kickoff)=="final"
    assert predict.refresh_stage(datetime(2026,9,15,16,tzinfo=timezone.utc),kickoff) is None


def test_odds_capture_due_only_in_final_window():
    kickoff=datetime(2026,9,18,0,15,tzinfo=timezone.utc)
    assert predict.odds_capture_due(datetime(2026,9,17,22,45,tzinfo=timezone.utc),kickoff)
    assert predict.odds_capture_due(datetime(2026,9,17,21,30,tzinfo=timezone.utc),kickoff)
    assert not predict.odds_capture_due(datetime(2026,9,17,21,0,tzinfo=timezone.utc),kickoff)
    assert not predict.odds_capture_due(datetime(2026,9,17,23,45,tzinfo=timezone.utc),kickoff)


def test_upcoming_game_day_includes_playoff_games():
    schedule=pd.DataFrame([{
        "game_id":"2026_21_A_B", "season":2026, "game_type":"POST",
        "gameday":"2027-02-07", "gametime":"18:30",
    }])
    game_day,kickoff,games=predict.upcoming_game_day(
        schedule, 2026, datetime(2027,2,7,20,tzinfo=timezone.utc),
    )
    assert game_day=="2027-02-07"
    assert kickoff==datetime(2027,2,7,23,30,tzinfo=timezone.utc)
    assert len(games)==1


def test_normalize_odds_snapshot_devigs_then_averages_books():
    games=pd.DataFrame([{
        "game_id":"2026_01_DAL_PHI", "gameday":"2026-09-10", "gametime":"20:15",
        "home_team":"PHI", "away_team":"DAL",
    }])
    events=[{
        "id":"event-1", "home_team":"Philadelphia Eagles", "away_team":"Dallas Cowboys",
        "bookmakers":[
            {"key":"book-a", "title":"Book A", "last_update":"2026-09-10T22:00:00Z", "markets":[{"key":"h2h", "outcomes":[{"name":"Philadelphia Eagles", "price":-110}, {"name":"Dallas Cowboys", "price":-110}]}]},
            {"key":"book-b", "title":"Book B", "last_update":"2026-09-10T22:00:00Z", "markets":[{"key":"h2h", "outcomes":[{"name":"Philadelphia Eagles", "price":-120}, {"name":"Dallas Cowboys", "price":100}]}]},
        ],
    }]
    snapshot=predict.normalize_odds_snapshot(
        events, games, 2026, "2026-09-10",
        datetime(2026,9,11,0,15,tzinfo=timezone.utc),
        datetime(2026,9,10,22,45,tzinfo=timezone.utc),
    )
    assert len(snapshot["games"])==1
    assert len(snapshot["games"][0]["bookmakers"])==2
    assert round(snapshot["games"][0]["marketHomeProbability"],3)==0.511
    assert snapshot["games"][0]["published"] is None


def test_normalize_odds_snapshot_matches_rams_schedule_code_and_freezes_published_pick(tmp_path):
    games=pd.DataFrame([{"game_id":"2026_01_SEA_LA", "gameday":"2026-09-13", "gametime":"16:05", "home_team":"LA", "away_team":"SEA"}])
    events=[{"id":"event-2", "home_team":"Los Angeles Rams", "away_team":"Seattle Seahawks", "bookmakers":[
        {"key":"book-a", "title":"Book A", "last_update":"2026-09-13T18:00:00Z", "markets":[{"key":"h2h", "outcomes":[{"name":"Los Angeles Rams", "price":-150}, {"name":"Seattle Seahawks", "price":130}]}]},
    ]}]
    output=tmp_path/"current.json"
    output.write_text('{"season":2026,"week":1,"generatedAt":"2026-09-10T22:00:00Z","model":{"marketWeight":0.95},"games":[{"gameId":"2026_01_SEA_LA","pick":"LA","homeWinProbability":0.61,"statisticalHomeProbability":0.58,"marketHomeProbability":0.6116}]}')
    published=predict.published_picks(2026, output)
    snapshot=predict.normalize_odds_snapshot(events, games, 2026, "2026-09-13", datetime(2026,9,13,20,5,tzinfo=timezone.utc), datetime(2026,9,13,18,35,tzinfo=timezone.utc), published)
    assert len(snapshot["games"])==1, "schedule uses LA for the Rams; the odds provider uses the full name"
    assert snapshot["games"][0]["published"]=={"generatedAt":"2026-09-10T22:00:00Z", "pick":"LA", "homeWinProbability":0.61, "statisticalHomeProbability":0.58, "marketHomeProbability":0.6116, "marketWeight":0.95}
    assert predict.published_picks(2025, output)=={}, "a stale week file must not be attached"


def test_qb_features_use_prior_games_and_latest_completed_primary_passer():
    features=pd.DataFrame([
        {"game_id":"2021_01_A_B", "gameday":"2021-09-12", "gametime":"13:00", "home_team":"A", "away_team":"B", "home_score":20, "away_score":10},
        {"game_id":"2021_02_A_B", "gameday":"2021-09-19", "gametime":"13:00", "home_team":"A", "away_team":"B", "home_score":None, "away_score":None},
    ])
    qb_games={"2021_01_A_B":[
        {"id":"A1", "team":"A", "dropbacks":30.0, "qb_epa":6.0, "cpoe":0.2},
        {"id":"B1", "team":"B", "dropbacks":25.0, "qb_epa":-2.0, "cpoe":-0.1},
    ]}
    result=predict.build_qb_features(features, qb_games)
    assert pd.isna(result.loc[0,"diff_qb_epa_per_dropback"])
    assert result.loc[1,"home_recent_qb_id"]=="A1"
    assert round(result.loc[1,"diff_qb_epa_per_dropback"],2)==0.28
    assert round(result.loc[1,"diff_qb_cpoe"],2)==0.30


def test_opponent_adjusted_stats_use_pregame_opponent_strength():
    team={metric:0.1 for metric in predict.METRICS}
    opponent={metric:0.05 for metric in predict.METRICS}
    adjusted=predict.opponent_adjusted_stats(team,opponent)
    assert adjusted["off_epa"]==0.05
    assert adjusted["def_epa_allowed"]==0.05


def test_probability_metrics_scores_probability_quality():
    metrics=predict.probability_metrics(
        pd.Series([1,0]).to_numpy(),
        pd.Series([0.9,0.1]).to_numpy(),
    )
    assert metrics["games"]==2
    assert metrics["accuracy"]==1
    assert round(metrics["brier"],2)==0.01
    assert round(metrics["logLoss"],3)==0.105


def test_select_market_weight_returns_best_grid_value():
    weight,brier=predict.select_market_weight(
        pd.Series([1,0]).to_numpy(),
        pd.Series([0.6,0.4]).to_numpy(),
        pd.Series([0.9,0.1]).to_numpy(),
    )
    assert weight==1
    assert round(brier,2)==0.01


def test_elo_home_probability_includes_home_advantage():
    assert predict.elo_home_probability(1500,1500,True)==0.5
    assert predict.elo_home_probability(1500,1500,False)>0.5


def test_elo_updates_after_completed_weeks_only():
    features=pd.DataFrame([
        {"game_id":"2021_01_A_B","season":2021,"week":1,"gameday":"2021-09-12","gametime":"13:00","home_team":"B","away_team":"A","home_score":20,"away_score":10,"home_win":1,"location":"Home"},
        {"game_id":"2021_02_A_B","season":2021,"week":2,"gameday":"2021-09-19","gametime":"13:00","home_team":"B","away_team":"A","home_score":10,"away_score":20,"home_win":0,"location":"Home"},
    ])
    probabilities=predict.elo_probabilities(features,2021,2021)
    assert probabilities["2021_01_A_B"]>0.5
    assert probabilities["2021_02_A_B"]>probabilities["2021_01_A_B"]


def test_pregame_elo_difference_excludes_current_week_result():
    games=pd.DataFrame([
        {"game_id":"2021_01_A_B","season":2021,"week":1,"gameday":"2021-09-12","gametime":"13:00","home_team":"B","away_team":"A","home_score":20,"away_score":10,"location":"Home"},
        {"game_id":"2021_01_C_D","season":2021,"week":1,"gameday":"2021-09-12","gametime":"16:00","home_team":"D","away_team":"C","home_score":10,"away_score":20,"location":"Home"},
        {"game_id":"2021_02_A_B","season":2021,"week":2,"gameday":"2021-09-19","gametime":"13:00","home_team":"B","away_team":"A","home_score":10,"away_score":20,"location":"Home"},
    ])
    differences=predict.pregame_elo_differences(games)
    assert differences["2021_01_A_B"]==differences["2021_01_C_D"]
    assert differences["2021_02_A_B"]>differences["2021_01_A_B"]
