"""Model pipelines. Imputation, scaling and encoding are fitted on training rows only."""
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, CATEGORY_LEVELS, NUMERIC_FEATURES, RANDOM_STATE, UNKNOWN
from .leakage import FeatureGuard


def build_preprocessor(imputation="iterative", add_indicator=False, scale=True):
    """Numeric: impute (median or iterative regression) [+ missing flags] [+ scaling].
    Categorical: one-hot with fixed levels; unseen levels are encoded as all zeros."""
    if imputation == "median":
        imputer = SimpleImputer(strategy="median", add_indicator=add_indicator)
    elif imputation == "iterative":
        imputer = IterativeImputer(max_iter=25, random_state=RANDOM_STATE, add_indicator=add_indicator)
    else:
        raise ValueError(f"unknown imputation '{imputation}'")
    numeric = Pipeline([("impute", imputer)] + ([("scale", StandardScaler())] if scale else []))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=UNKNOWN)),
        ("onehot", OneHotEncoder(categories=[CATEGORY_LEVELS[c] for c in CATEGORICAL_FEATURES],
                                 handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
                             remainder="drop", verbose_feature_names_out=False)


def build_pipeline(estimator, imputation="iterative", add_indicator=False, scale=True):
    """guard (leakage check) -> preprocessing -> estimator, as one fit-able object."""
    return Pipeline([
        ("guard", FeatureGuard()),
        ("preprocess", build_preprocessor(imputation, add_indicator, scale)),
        ("model", estimator),
    ])
