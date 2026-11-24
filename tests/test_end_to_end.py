import math

import pytest

from readmission.data import features_and_target, load_dataset
from readmission.evaluation import choose_threshold
from readmission.experiment import holdout_split
from readmission.predict import ReadmissionPredictor

RECORD = {"age": 67, "gender": "F", "bmi": 31.2, "waist_cm": 104, "fasting_glucose": 182, "hba1c": 8.4,
          "bp_systolic": 138, "cholesterol_total": 201, "hdl": 44, "triglycerides": 190, "years_since_diagnosis": 14,
          "medication": "Insulin", "physical_activity": "Low", "smoking_status": "Former", "visits_last_year": 2,
          "length_of_stay_days": 7}


def test_run_writes_reports_and_model(trained):
    ctx, base, _ = trained
    assert (base / "models" / "readmission_model.joblib").exists()
    assert (base / "reports" / "RESULTS.md").exists() and (base / "reports" / "tables" / "model_comparison.csv").exists()
    assert ctx["selected_model"] != "dummy_prior" and 0 < ctx["threshold"] < 1
    assert 0 <= ctx["final_metrics"]["at_threshold"]["roc_auc"] <= 1


def test_threshold_comes_from_development_oof_only(trained):
    ctx, _, path = trained
    dev, hold = holdout_split(load_dataset(path))
    _, y_dev = features_and_target(dev)
    assert ctx["n_development"] == len(dev) and ctx["n_final_set"] == len(hold)
    assert ctx["threshold"] == choose_threshold(y_dev, ctx["oof"][ctx["selected_model"]])


def test_prediction_contract(trained):
    predictor = ReadmissionPredictor.load(trained[1] / "models")
    out = predictor.predict_one(RECORD)
    assert {"probability", "predicted_class", "risk_category", "model_version"} <= set(out)
    assert 0 <= out["probability"] <= 1 and out["model_version"] == "1.0.0"
    assert out["predicted_class"] == int(out["probability"] >= predictor.threshold)
    assert out["risk_category"] in {"Low", "Elevated", "High"}


def test_invalid_and_unseen_inputs_are_handled(trained):
    predictor = ReadmissionPredictor.load(trained[1] / "models")
    bad = {**RECORD, "age": "abc", "medication": "NewDrug", "hba1c": "7.5%", "gender": None}
    out = predictor.predict_one(bad)
    assert math.isfinite(out["probability"])
    text = " ".join(out["warnings"])
    assert "age" in text and "NewDrug" in text and "gender missing" in text
    with pytest.raises(TypeError):
        predictor.predict_one(["not", "a", "dict"])


def test_column_order_and_forbidden_fields_do_not_change_prediction(trained):
    predictor = ReadmissionPredictor.load(trained[1] / "models")
    base = predictor.predict_one(RECORD)["probability"]
    shuffled = dict(reversed(list(RECORD.items())))
    leaky = {**RECORD, "readmission_ward": "ICU", "readmitted_30d": 1, "patient_id": "X1"}
    assert predictor.predict_one(shuffled)["probability"] == base
    assert predictor.predict_one(leaky)["probability"] == base
