"""Candidate models with small, deliberate search spaces (ordered simple -> complex)."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_STATE

# class_weight is left at None on purpose: re-weighting distorts predicted probabilities,
# and the class imbalance is handled by choosing the decision threshold instead.


def candidate_models(quick=False):
    """name -> (estimator, parameter grid, needs scaling). `quick` = tiny grids for tests."""
    trees = 40 if quick else 300
    models = {
        "dummy_prior": (DummyClassifier(strategy="prior"), {}, False),
        "logistic_regression": (LogisticRegression(max_iter=5000),
                                {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, True),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=200, l2_regularization=1.0,
                                                                  early_stopping=False, random_state=RANDOM_STATE),
                                   {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15],
                                    "model__min_samples_leaf": [20, 80]}, False),
        "random_forest": (RandomForestClassifier(n_estimators=trees, random_state=RANDOM_STATE, n_jobs=-1),
                          {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [5, 20]}, False),
        "extra_trees": (ExtraTreesClassifier(n_estimators=trees, random_state=RANDOM_STATE, n_jobs=-1),
                        {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [5, 20]}, False),
    }
    if quick:
        first = lambda grid: {k: v[:1] for k, v in grid.items()}  # noqa: E731
        models = {name: (est, first(grid), scale) for name, (est, grid, scale) in models.items()}
    return models


def reference_models():
    """Two fixed models used to compare preprocessing options before tuning."""
    return {
        "logistic_regression": (LogisticRegression(C=0.1, max_iter=5000), True),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                  max_leaf_nodes=15, l2_regularization=1.0,
                                                                  early_stopping=False, random_state=RANDOM_STATE), False),
    }
