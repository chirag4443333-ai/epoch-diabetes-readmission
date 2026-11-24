"""Shared fixtures. Tests use synthetic records, so they run without the (private) event data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from readmission.config import CATEGORY_LEVELS, DEFAULT_TRAIN_PATH  # noqa: E402

REAL_CLEANED = Path(DEFAULT_TRAIN_PATH)
REAL_RAW = REAL_CLEANED.parent / "train.csv"
requires_real_data = pytest.mark.skipif(not REAL_CLEANED.exists(), reason="event data not present (never committed)")
FLAGGED = ["age", "bmi", "waist_cm", "hba1c", "bp_systolic", "cholesterol_total", "hdl", "triglycerides",
           "years_since_diagnosis"]


def make_synthetic(n=400, seed=0):
    """Random records in the cleaned-file schema (30 columns). Not real patients."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": [f"SYN{i:05d}" for i in range(n)], "age": rng.integers(25, 90, n),
        "gender": rng.choice(["M", "F"], n), "admission_date": "2025-01-01",
        "bmi": rng.normal(29, 5, n).round(1).clip(17, 47), "waist_cm": rng.normal(97, 15, n).round().clip(60, 156),
        "fasting_glucose": rng.normal(140, 35, n).round().clip(60, 280),
        "hba1c": rng.normal(6.5, 1.1, n).round(1).clip(4, 10.6), "bp_systolic": rng.normal(128, 14, n).round(),
        "cholesterol_total": rng.normal(195, 37, n).round(), "hdl": rng.normal(51, 10, n).round(),
        "triglycerides": rng.normal(162, 70, n).round().clip(40, 700), "years_since_diagnosis": rng.integers(0, 20, n),
        "medication": rng.choice(CATEGORY_LEVELS["medication"], n),
        "physical_activity": rng.choice(CATEGORY_LEVELS["physical_activity"], n),
        "smoking_status": rng.choice(CATEGORY_LEVELS["smoking_status"], n),
        "visits_last_year": rng.integers(0, 12, n), "length_of_stay_days": rng.integers(1, 14, n),
    })
    logit = (-1.3 + 0.9 * (df.hba1c - 6.5) + 0.3 * (df.length_of_stay_days - 4)
             + 0.8 * df.medication.isin(["Insulin", "Combination"]))
    df["readmitted_30d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["readmission_ward"] = np.where(df.readmitted_30d == 1, "ICU", "Not readmitted")
    for col in FLAGGED:
        df[col + "_was_missing"] = (rng.random(n) < 0.08).astype(int)
    df["physical_activity_was_missing"] = (df.physical_activity == "Unknown").astype(int)
    return df


@pytest.fixture
def synthetic_csv(tmp_path):
    path = tmp_path / "synthetic_cleaned.csv"
    make_synthetic().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """One quick end-to-end training run on synthetic data, shared by several tests."""
    from readmission.train import run
    base = tmp_path_factory.mktemp("run")
    path = base / "synthetic.csv"
    make_synthetic(n=500, seed=1).to_csv(path, index=False)
    ctx = run(path, report_dir=base / "reports", model_dir=base / "models", quick=True, figures=False,
              log=lambda *a, **k: None)
    return ctx, base, path
