"""Reusable inference interface: patient record(s) in, risk estimate(s) out.

    predictor = ReadmissionPredictor.load("models")
    predictor.predict_one({"age": 67, "gender": "F", "hba1c": 8.1, "medication": "Insulin"})

Command line:
    python -m readmission.predict --json '{"age": 67, "gender": "F", "hba1c": 8.1}'
    python -m readmission.predict --input new_patients.csv --output predictions.csv
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (CATEGORICAL_FEATURES, DEFAULT_MODEL_DIR, FEATURES, FORBIDDEN_FEATURES, GLUCOSE_MMOL_CUTOFF,
                     ID_COL, MODEL_VERSION, NUMERIC_FEATURES, UNKNOWN)
from .data import harmonise, to_number

MODEL_FILE = "readmission_model.joblib"
USE_NOTE = ("Prototype decision support for prioritising post-discharge follow-up only. Not a diagnosis, "
            "not clinically validated and not a treatment recommendation; every flag needs human review.")


def _blank(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip().lower() in ("", "nan", "none")


class ReadmissionPredictor:
    """The fitted pipeline + operating threshold + model version, saved and loaded together."""

    def __init__(self, pipeline, threshold, high_risk_threshold, model_version=MODEL_VERSION,
                 model_name="", feature_ranges=None):
        if not 0.0 < threshold < high_risk_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 < threshold < high_risk_threshold <= 1")
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.high_risk_threshold = float(high_risk_threshold)
        self.model_version = model_version
        self.model_name = model_name
        self.feature_ranges = feature_ranges or {}

    def save(self, model_dir=DEFAULT_MODEL_DIR):
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(self.__dict__), path / MODEL_FILE)
        return path / MODEL_FILE

    @classmethod
    def load(cls, model_dir=DEFAULT_MODEL_DIR):
        return cls(**joblib.load(Path(model_dir) / MODEL_FILE))

    def risk_category(self, probability):
        if probability >= self.high_risk_threshold:
            return "High"
        return "Elevated" if probability >= self.threshold else "Low"

    def _input_notes(self, records, X):
        """Human-readable notes on anything that was imputed, converted or not recognised."""
        notes = []
        for i in range(len(records)):
            row, out, n = records.iloc[i], X.iloc[i], []
            for col in NUMERIC_FEATURES:
                given = row.get(col)
                if _blank(given):
                    n.append(f"{col} missing - imputed")
                elif np.isnan(out[col]):
                    n.append(f"{col}={given!r} invalid or implausible - treated as missing and imputed")
                else:
                    if col == "fasting_glucose" and to_number(pd.Series([given])).iloc[0] < GLUCOSE_MMOL_CUTOFF:
                        n.append("fasting_glucose < 30 read as mmol/L and converted to mg/dL")
                    low, high = self.feature_ranges.get(col, (-np.inf, np.inf))
                    if not low <= out[col] <= high:
                        n.append(f"{col}={out[col]:g} outside the training range [{low:g}, {high:g}]")
            for col in CATEGORICAL_FEATURES:
                given = row.get(col)
                if _blank(given):
                    n.append(f"{col} missing - treated as '{UNKNOWN}'")
                elif out[col] == UNKNOWN and str(given).strip().lower() != UNKNOWN.lower():
                    n.append(f"{col}={given!r} not recognised - treated as '{UNKNOWN}'")
            notes.append(n)
        return notes

    def predict_frame(self, records):
        """records: DataFrame with raw or clean values. Extra columns (identifiers, the leakage
        column, the target) are never used; the model sees FEATURES only, in training order."""
        records = records.reset_index(drop=True)
        X = harmonise(records)[FEATURES]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories")
            proba = self.pipeline.predict_proba(X)[:, 1]
        ignored = sorted(c for c in records.columns if c in FORBIDDEN_FEATURES and c != ID_COL)
        notes = self._input_notes(records, X)
        if ignored:
            notes = [n + [f"ignored non-model columns: {ignored}"] for n in notes]
        out = pd.DataFrame({
            "probability": proba.round(4),
            "predicted_class": (proba >= self.threshold).astype(int),
            "risk_category": [self.risk_category(p) for p in proba],
            "model_version": self.model_version,
            "input_notes": ["; ".join(n) for n in notes],
        })
        if ID_COL in records:  # echoed back for linking only - never a model input
            out.insert(0, ID_COL, records[ID_COL].values)
        return out

    def predict_one(self, record):
        if not isinstance(record, dict):
            raise TypeError("record must be a dict of feature name -> value")
        row = self.predict_frame(pd.DataFrame([record])).iloc[0]
        return {"probability": float(row["probability"]), "predicted_class": int(row["predicted_class"]),
                "risk_category": row["risk_category"], "model_version": self.model_version,
                "threshold": self.threshold, "input_notes": [s for s in row["input_notes"].split("; ") if s],
                "note": USE_NOTE}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score patients with the trained readmission model.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="one patient record as a JSON object")
    group.add_argument("--input", help="CSV file with one patient per row")
    parser.add_argument("--output", help="where to write predictions for --input (default: print)")
    args = parser.parse_args(argv)
    predictor = ReadmissionPredictor.load(args.model_dir)
    if args.json:
        print(json.dumps(predictor.predict_one(json.loads(args.json)), indent=2))
        return
    result = predictor.predict_frame(pd.read_csv(args.input, dtype=str))
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"wrote {len(result)} predictions to {args.output}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
