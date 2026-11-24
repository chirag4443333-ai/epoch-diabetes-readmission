import pytest
from conftest import make_synthetic
from sklearn.linear_model import LogisticRegression

from readmission.config import FEATURES
from readmission.data import features_and_target, harmonise
from readmission.leakage import LeakageError, assert_no_forbidden_features, audit_leakage
from readmission.preprocessing import build_pipeline


@pytest.mark.parametrize("col", ["readmission_ward", "patient_id", "admission_date", "readmitted_30d", "bmi_was_missing"])
def test_forbidden_columns_are_rejected(col):
    with pytest.raises(LeakageError):
        assert_no_forbidden_features(FEATURES + [col])


def test_training_fails_if_leaky_column_reaches_the_pipeline():
    df = make_synthetic()
    with pytest.raises(LeakageError):
        build_pipeline(LogisticRegression()).fit(df[FEATURES + ["readmission_ward"]], df["readmitted_30d"])


def test_feature_matrix_is_the_whitelist_in_training_order():
    X, _ = features_and_target(harmonise(make_synthetic()))
    assert list(X.columns) == FEATURES and "readmission_ward" not in X


def test_audit_confirms_ward_leak_and_catches_new_perfect_feature():
    df = harmonise(make_synthetic())
    assert audit_leakage(df)["readmission_ward_auc_if_used"] == 1.0
    df["hba1c"] = df["readmitted_30d"] * 5.0
    with pytest.raises(LeakageError):
        audit_leakage(df)
