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
