"""Automatic leakage guards. Training fails if a forbidden column reaches the model."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

from .config import FEATURES, FORBIDDEN_FEATURES, LEAKAGE_COL, MISSING_FLAG_SUFFIX, TARGET


class LeakageError(RuntimeError):
    """Raised when forbidden or unexpected columns would enter the feature matrix."""


def assert_no_forbidden_features(columns):
    columns = list(columns)
    forbidden = [c for c in columns if c in FORBIDDEN_FEATURES or c.endswith(MISSING_FLAG_SUFFIX)]
    if forbidden:
        raise LeakageError(f"forbidden columns in feature matrix: {forbidden}")
    if columns != FEATURES:
        raise LeakageError(f"feature matrix must be exactly {FEATURES} (in order); got {columns}")


class FeatureGuard(BaseEstimator, TransformerMixin):
    """First pipeline step: re-checks the columns on every fit / transform call."""

    def fit(self, X, y=None):
        assert_no_forbidden_features(X.columns)
        return self

    def transform(self, X):
        assert_no_forbidden_features(X.columns)
        return X


def audit_leakage(df, max_single_feature_auc=0.95):
    """Prove the known leak exists and that no allowed feature is suspiciously perfect."""
    y = df[TARGET].astype(int)
    report = {}
    if LEAKAGE_COL in df:
        ward_known = df[LEAKAGE_COL].fillna("Not readmitted").str.strip().str.lower().ne("not readmitted")
        report["readmission_ward_auc_if_used"] = round(float(roc_auc_score(y, ward_known)), 4)
        report["readmission_ward_filled_only_for_readmitted"] = bool((ward_known == y.astype(bool)).all())
    single = {}
    for col in FEATURES:
        x = df[col]
        if pd.api.types.is_numeric_dtype(x):
            mask = x.notna()
            auc = roc_auc_score(y[mask], x[mask])
        else:
            auc = roc_auc_score(y, x.map(y.groupby(x).mean()))  # category -> its own rate (in-sample)
        single[col] = round(float(max(auc, 1 - auc)), 4)
    report["single_feature_auc"] = single
    suspicious = [c for c, a in single.items() if a >= max_single_feature_auc]
    if suspicious:
        raise LeakageError(f"allowed features are near-perfect predictors (possible leak): {suspicious}")
    return report
