"""Model explanation and cautious subgroup checks."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from .config import FEATURES, RANDOM_STATE
from .evaluation import metrics_at_threshold
from .experiment import cv_folds

AGE_BANDS, AGE_LABELS = [0, 45, 55, 65, 200], ["<45", "45-54", "55-64", "65+"]


def logistic_coefficients(pipeline):
    """Coefficients on standardised numeric inputs / one-hot categories (log-odds per SD or vs. absent)."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def native_importance(pipeline):
    """Impurity / split-based importance for tree models (None for other models)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return (pd.DataFrame({"feature": names, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def cv_permutation_importance(pipeline, X, y, n_repeats=5):
    """Drop in validation ROC-AUC when one raw input column is shuffled.
    Fitted on training folds, measured on validation folds (development data only)."""
    per_fold = []
    for train_idx, val_idx in cv_folds().split(X, y):
        fitted = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(fitted, X.iloc[val_idx], y.iloc[val_idx], scoring="roc_auc",
                                        n_repeats=n_repeats, random_state=RANDOM_STATE)
        per_fold.append(result.importances_mean)
    arr = np.array(per_fold)
    return (pd.DataFrame({"feature": FEATURES, "auc_drop_mean": arr.mean(0), "auc_drop_sd": arr.std(0, ddof=1)})
            .sort_values("auc_drop_mean", ascending=False).reset_index(drop=True))


def subgroup_table(frame, y, proba, threshold, min_n=30):
    """Metrics by gender and age band. Small or single-class groups get no metrics."""
    age_band = pd.cut(frame["age"], AGE_BANDS, right=False, labels=AGE_LABELS).astype(str)
    groups = {"gender": frame["gender"].astype(str), "age_band": age_band.replace("nan", "age missing")}
    y, proba = np.asarray(y).astype(int), np.asarray(proba, dtype=float)
    rows = []
    for group, values in groups.items():
        for level in sorted(values.unique()):
            mask = (values == level).values
            row = {"group": group, "level": level, "n": int(mask.sum()), "readmitted": int(y[mask].sum())}
            if mask.sum() >= min_n and 0 < y[mask].sum() < mask.sum():
                m = metrics_at_threshold(y[mask], proba[mask], threshold)
                row.update({"observed_rate": y[mask].mean(), "mean_predicted": proba[mask].mean(),
                            **{k: m[k] for k in ["roc_auc", "recall", "precision", "specificity", "flagged_rate"]}})
            rows.append(row)
    return pd.DataFrame(rows)
