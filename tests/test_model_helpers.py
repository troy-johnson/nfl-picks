from pathlib import Path
import importlib.util
import sys
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
