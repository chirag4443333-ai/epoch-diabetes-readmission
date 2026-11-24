import numpy as np
from conftest import make_synthetic
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from readmission.config import NUMERIC_FEATURES
from readmission.data import features_and_target, harmonise, restore_missing
from readmission.experiment import cv_folds, holdout_split
from readmission.preprocessing import build_pipeline


def _data(n=500, seed=0):
    raw = make_synthetic(n, seed)
    return restore_missing(harmonise(raw), raw)


def test_holdout_is_stratified_disjoint_and_reproducible():
    df = _data()
    dev, hold = holdout_split(df)
    _, hold2 = holdout_split(df)
    assert len(hold) == 100 and set(dev.patient_id).isdisjoint(hold.patient_id)
    assert abs(dev.readmitted_30d.mean() - hold.readmitted_30d.mean()) < 0.03
    assert list(hold.patient_id) == list(hold2.patient_id)


def test_cv_folds_are_disjoint_and_cover_every_row_once():
    X, y = features_and_target(_data())
    seen = []
    for train_idx, val_idx in cv_folds().split(X, y):
        assert set(train_idx).isdisjoint(val_idx)
        seen.extend(val_idx)
    assert sorted(seen) == list(range(len(X)))


def test_imputer_is_learned_from_training_rows_only():
    X, y = features_and_target(_data())
    train = X.iloc[:300]
    pipe = build_pipeline(LogisticRegression(max_iter=2000), "median", scale=False).fit(train, y.iloc[:300])
    imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, train[NUMERIC_FEATURES].median().values)
    assert not np.allclose(imputer.statistics_, X[NUMERIC_FEATURES].median().values)


def test_iterative_imputation_and_unseen_categories_do_not_crash():
    X, y = features_and_target(_data())
    pipe = build_pipeline(LogisticRegression(max_iter=2000), "iterative", add_indicator=True).fit(X, y)
    new = X.head(3).copy()
    new["medication"], new["hba1c"] = "Experimental drug", np.nan
    proba = pipe.predict_proba(new)[:, 1]
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()


def test_training_is_reproducible():
    X, y = features_and_target(_data())
    a = build_pipeline(RandomForestClassifier(50, random_state=42), scale=False).fit(X, y).predict_proba(X)
    b = build_pipeline(RandomForestClassifier(50, random_state=42), scale=False).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
