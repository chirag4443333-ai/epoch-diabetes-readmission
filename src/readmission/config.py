"""Single place for column roles, leakage rules, seeds and default paths."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20  # untouched stratified holdout (used only when no official test file is given)
N_SPLITS = 5  # stratified K-fold inside the development data
MODEL_VERSION = "1.0.0"
HIGH_RISK_PROBABILITY = 0.5  # prototype "High" tier: predicted risk is more likely than not

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "diabetes_train_cleaned.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

TARGET = "readmitted_30d"
ID_COL = "patient_id"
LEAKAGE_COL = "readmission_ward"

NUMERIC_FEATURES = [
    "age", "bmi", "waist_cm", "fasting_glucose", "hba1c", "bp_systolic",
    "cholesterol_total", "hdl", "triglycerides", "years_since_diagnosis",
    "visits_last_year", "length_of_stay_days",
]
CATEGORICAL_FEATURES = ["gender", "medication", "physical_activity", "smoking_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # the exact model input order

# Fixed category levels -> identical one-hot columns in every fold and at inference time.
CATEGORY_LEVELS = {
    "gender": ["F", "M"],
    "medication": ["Combination", "Insulin", "Metformin", "No medication"],
    "physical_activity": ["High", "Low", "Medium", "Unknown"],  # Unknown = explicit category
    "smoking_status": ["Current", "Former", "Never"],
}
UNKNOWN = "Unknown"

# Columns that must never reach the feature matrix (training fails if they do).
FORBIDDEN_FEATURES = {
    LEAKAGE_COL: "ward of the readmission - only known after the outcome (perfect target leakage)",
    TARGET: "the prediction target itself",
    ID_COL: "patient identifier - no clinical meaning",
    "admission_date": "period-specific date with no established predictive value",
}
MISSING_FLAG_SUFFIX = "_was_missing"  # flags from Phase 1; used only to rebuild missingness

# Row-level rules from CLEANING_LOG.md (applied to raw-format files, e.g. a hidden test set).
PLAUSIBLE_RANGES = {"age": (1, 120), "bmi": (10, 80), "waist_cm": (40, 250), "bp_systolic": (50, 300)}
GLUCOSE_MMOL_CUTOFF = 30  # fasting glucose < 30 is mmol/L -> multiply by 18 to get mg/dL
MMOL_TO_MGDL = 18
