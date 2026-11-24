"""Load, harmonise and validate the diabetes readmission data.

One code path handles both supported inputs:
* the Phase 1 cleaned file (30 columns incl. ``*_was_missing`` flags), and
* the raw competition format (20 messy text columns, e.g. a hidden test file).

The row-level rules of CLEANING_LOG.md are applied (they are idempotent on the
cleaned file). Values flagged as originally missing are set back to NaN in
memory so that imputation is learned inside each training fold only.
No statistics are learned here, so this step cannot leak information.
"""
import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (CATEGORICAL_FEATURES, CATEGORY_LEVELS, FEATURES, GLUCOSE_MMOL_CUTOFF, ID_COL,
                     LEAKAGE_COL, MISSING_FLAG_SUFFIX, MMOL_TO_MGDL, NUMERIC_FEATURES,
                     PLAUSIBLE_RANGES, TARGET, UNKNOWN)


class DataValidationError(ValueError):
    """Raised when a dataset does not match the expected schema."""


_GENDER_ALIASES = {"male": "M", "female": "F"}
_TARGET_VALUES = {"yes": 1, "y": 1, "1": 1, "1.0": 1, "true": 1,
                  "no": 0, "n": 0, "0": 0, "0.0": 0, "false": 0}


def canonical_category(value, column):
    """Map a messy spelling (case, spaces, one-letter typos) to a known level, else NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = " ".join(str(value).strip().lower().split())
    if text in ("", "nan", "none"):
        return np.nan
    lookup = {level.lower(): level for level in CATEGORY_LEVELS[column]}
    if column == "gender":
        lookup.update(_GENDER_ALIASES)
    if text in lookup:
        return lookup[text]
    match = difflib.get_close_matches(text, list(lookup), n=1, cutoff=0.75)  # same rule as Phase 1
    return lookup[match[0]] if match else np.nan


def parse_target(value):
    """Map the 14 observed spellings of the label (Yes/no/Y/0/True...) to 1/0."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return _TARGET_VALUES.get(str(value).strip().lower(), np.nan)


def to_number(series):
    """Parse numbers stored as text (e.g. '7.0%'); anything unparseable becomes NaN."""
    text = series.astype(str).str.strip().str.replace("%", "", regex=False)
    return pd.to_numeric(text.replace({"": None, "nan": None, "None": None}), errors="coerce")


def harmonise(raw):
    """Return a standard frame (id, features, target, leakage column for audit only)."""
    df = pd.DataFrame(index=raw.index)
    if ID_COL in raw:
        df[ID_COL] = raw[ID_COL].astype(str).str.strip().str.upper()
    for col in NUMERIC_FEATURES:
        df[col] = to_number(raw[col]) if col in raw else np.nan
    glucose = df["fasting_glucose"]
    df["fasting_glucose"] = glucose.where(~(glucose < GLUCOSE_MMOL_CUTOFF), (glucose * MMOL_TO_MGDL).round())
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan
    df.loc[df["years_since_diagnosis"] > df["age"], "years_since_diagnosis"] = np.nan
    for col in CATEGORICAL_FEATURES:
        values = raw[col] if col in raw else pd.Series(np.nan, index=raw.index)
        df[col] = values.map(lambda v, c=col: canonical_category(v, c)).fillna(UNKNOWN)
    if TARGET in raw:
        df[TARGET] = raw[TARGET].map(parse_target)
    if LEAKAGE_COL in raw:  # kept only for the leakage audit - never a feature
        df[LEAKAGE_COL] = raw[LEAKAGE_COL]
    return df


def restore_missing(df, raw):
    """Set values flagged by Phase 1 as originally missing back to NaN (copy, file untouched)."""
    out = df.copy()
    for col in NUMERIC_FEATURES:
        flag = col + MISSING_FLAG_SUFFIX
        if flag in raw:
            out.loc[pd.to_numeric(raw[flag], errors="coerce").fillna(0).astype(int).values == 1, col] = np.nan
    return out


def detect_format(raw):
    return "cleaned" if any(c.endswith(MISSING_FLAG_SUFFIX) for c in raw.columns) else "raw"


def load_dataset(source, is_training=True):
    """Read a CSV path (or an already-loaded DataFrame) in either format -> standard frame.

    Training files are de-duplicated (Phase 1 rule: identical records after
    harmonisation, first copy kept). Test files are never filtered, so every
    row receives a prediction.
    """
    raw = pd.read_csv(source, dtype=str) if isinstance(source, (str, Path)) else source.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    df = restore_missing(harmonise(raw), raw)
    df.attrs["source_format"] = detect_format(raw)
    df.attrs["rows_read"] = len(raw)
    if is_training:
        key = [c for c in [ID_COL, *FEATURES, TARGET] if c in df]
        df = df.loc[~df.duplicated(subset=key, keep="first")].reset_index(drop=True)
        df.attrs["duplicates_removed"] = df.attrs["rows_read"] - len(df)
    return df


def validate_dataset(df, require_target=True):
    """Fail loudly on schema problems; return a small aggregate summary."""
    missing_cols = [c for c in FEATURES if c not in df]
    if missing_cols:
        raise DataValidationError(f"missing feature columns: {missing_cols}")
    for col in CATEGORICAL_FEATURES:
        bad = set(df[col].dropna()) - set(CATEGORY_LEVELS[col]) - {UNKNOWN}
        if bad:
            raise DataValidationError(f"unexpected levels in {col}: {sorted(bad)}")
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"{col} is not numeric")
    summary = {"rows": int(len(df)),
               "missing_values_by_column": {c: int(n) for c, n in df[FEATURES].isna().sum().items() if n}}
    if require_target:
        if TARGET not in df:
            raise DataValidationError(f"target column '{TARGET}' is missing")
        if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
            raise DataValidationError("target must be binary 0/1 with no missing values")
        summary.update(positives=int(df[TARGET].sum()), negatives=int((df[TARGET] == 0).sum()),
                       prevalence=round(float(df[TARGET].mean()), 4))
    if ID_COL in df and require_target and df[ID_COL].duplicated().any():
        raise DataValidationError("duplicate patient identifiers in training data")
    return summary


def features_and_target(df):
    """Feature matrix in the exact training order (whitelist) and the target."""
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int) if TARGET in df else None
    return X, y
