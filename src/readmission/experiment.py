"""Validation protocol and model comparison (shared by train.py and the notebook)."""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from .config import ID_COL, N_SPLITS, RANDOM_STATE, TARGET, TEST_SIZE
from .evaluation import choose_threshold
from .models import candidate_models, reference_models
from .preprocessing import build_pipeline

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}
PREPROCESSING_VARIANTS = [("median", False), ("median", True), ("iterative", False), ("iterative", True)]


def holdout_split(df):
    """Stratified 80/20 split (seed 42). The 20% is untouched until the final evaluation."""
    dev, holdout = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)
    if ID_COL in df and not set(dev[ID_COL]).isdisjoint(holdout[ID_COL]):
        raise RuntimeError("a patient appears in both development and holdout data")
    return dev.reset_index(drop=True), holdout.reset_index(drop=True)


def cv_folds():
    """Same stratified 5 folds for every experiment, so all comparisons are like-for-like."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def compare_preprocessing(X, y):
    """Median vs iterative imputation, with / without missing indicators, for two fixed models."""
    rows = []
    for model_name, (estimator, scale) in reference_models().items():
        for imputation, indicator in PREPROCESSING_VARIANTS:
            pipe = build_pipeline(clone(estimator), imputation, indicator, scale)
            res = cross_validate(pipe, X, y, cv=cv_folds(), scoring=SCORING)
            rows.append({"model": model_name, "imputation": imputation, "missing_indicators": indicator,
                         "cv_roc_auc_mean": res["test_roc_auc"].mean(),
                         "cv_roc_auc_sd": res["test_roc_auc"].std(ddof=1),
                         "cv_pr_auc_mean": res["test_pr_auc"].mean(),
                         "cv_brier_mean": -res["test_brier"].mean()})
    return pd.DataFrame(rows)


def choose_preprocessing(table):
    """Rule fixed in advance: highest ROC-AUC averaged over the two reference models."""
    avg = table.groupby(["imputation", "missing_indicators"])["cv_roc_auc_mean"].mean()
    imputation, indicator = avg.idxmax()
    return imputation, bool(indicator)


def compare_models(X, y, imputation, add_indicator, quick=False):
    """Grid-search every candidate on the same folds. Returns the results table,
    per-fold ROC-AUC of each tuned model and its out-of-fold (OOF) probabilities."""
    rows, fold_scores, oof = [], {}, {}
    for name, (estimator, grid, scale) in candidate_models(quick).items():
        start = time.time()
        pipe = build_pipeline(estimator, imputation, add_indicator, scale)
        res = GridSearchCV(pipe, grid, cv=cv_folds(), scoring=SCORING, refit=False).fit(X, y).cv_results_
        best = int(np.argmax(res["mean_test_roc_auc"]))
        params = res["params"][best]
        folds = np.array([res[f"split{k}_test_roc_auc"][best] for k in range(N_SPLITS)])
        pr_folds = np.array([res[f"split{k}_test_pr_auc"][best] for k in range(N_SPLITS)])
        oof[name] = cross_val_predict(clone(pipe).set_params(**params), X, y, cv=cv_folds(),
                                      method="predict_proba")[:, 1]
        fold_scores[name] = folds.round(4).tolist()
        rows.append({"model": name, "best_params": {k.replace("model__", ""): v for k, v in params.items()},
                     "configs_tried": len(res["params"]),
                     "cv_roc_auc_mean": folds.mean(), "cv_roc_auc_sd": folds.std(ddof=1),
                     "cv_pr_auc_mean": pr_folds.mean(), "cv_pr_auc_sd": pr_folds.std(ddof=1),
                     "cv_brier_mean": -res["mean_test_brier"][best], "seconds": round(time.time() - start, 1)})
    return pd.DataFrame(rows), fold_scores, oof


def select_model(results):
    """Highest mean CV ROC-AUC, but prefer the simplest model within one standard error of it
    (candidates are ordered simple -> complex, so the first eligible one wins)."""
    cands = results[results["model"] != "dummy_prior"].reset_index(drop=True)
    best = cands.loc[cands["cv_roc_auc_mean"].idxmax()]
    one_se = best["cv_roc_auc_sd"] / np.sqrt(N_SPLITS)
    chosen = cands[cands["cv_roc_auc_mean"] >= best["cv_roc_auc_mean"] - one_se].iloc[0]
    reason = (f"best mean CV ROC-AUC = {best['cv_roc_auc_mean']:.4f} ({best['model']}); one standard error = "
              f"{one_se:.4f}; simplest model within one SE = {chosen['model']} "
              f"({chosen['cv_roc_auc_mean']:.4f} +/- {chosen['cv_roc_auc_sd']:.4f} SD over folds)")
    return chosen["model"], reason


def build_final_pipeline(name, best_params, imputation, add_indicator, quick=False):
    estimator, _, scale = candidate_models(quick)[name]
    pipe = build_pipeline(clone(estimator), imputation, add_indicator, scale)
    return pipe.set_params(**{f"model__{k}": v for k, v in best_params.items()})


def threshold_stability(y, oof_proba, beta=2.0):
    """F-beta-optimal threshold re-computed inside each validation fold (how stable is it?)."""
    y = pd.Series(np.asarray(y))
    return [choose_threshold(y.iloc[val], np.asarray(oof_proba)[val], beta)
            for _, val in cv_folds().split(np.zeros(len(y)), y)]
