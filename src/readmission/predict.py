"""Reusable inference interface: patient record(s) in, risk estimate out.

    python -m readmission.predict --json '{"age": 67, "medication": "Insulin", "hba1c": 8.1}'
    python -m readmission.predict --input new_patients.csv --output predictions.csv
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

from .config import CATEGORICAL_FEATURES, DEFAULT_MODEL_DIR, FEATURES, GLUCOSE_MMOL_CUTOFF, NUMERIC_FEATURES, UNKNOWN
from .data import harmonise, to_number

MODEL_FILE = "readmission_model.joblib"
NOTE = ("Prototype decision support for prioritising post-discharge follow-up. Not a diagnosis, not clinically "
        "validated and not a treatment recommendation - every flag needs human review.")


def _blank(value):
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


class ReadmissionPredictor:
    """Wraps the fitted pipeline with its operating threshold and version."""

    def __init__(self, pipeline, threshold, high_risk_threshold, model_version, model_name="", feature_ranges=None):
        if not 0.0 < threshold < high_risk_threshold <= 1.0:
            raise ValueError("need 0 < threshold < high_risk_threshold <= 1")
        self.pipeline, self.threshold = pipeline, float(threshold)
        self.high_risk_threshold, self.model_version = float(high_risk_threshold), model_version
        self.model_name, self.feature_ranges = model_name, feature_ranges or {}

    def save(self, model_dir=DEFAULT_MODEL_DIR):
        path = Path(model_dir) / MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(self.__dict__), path)
        return path

    @classmethod
    def load(cls, model_dir=DEFAULT_MODEL_DIR):
        return cls(**joblib.load(Path(model_dir) / MODEL_FILE))

    def risk_category(self, probability):
        if probability >= self.high_risk_threshold:
            return "High"
        return "Elevated" if probability >= self.threshold else "Low"

    def _prepare(self, records):
        """Same row rules as training, exact training column order, readable warnings per row."""
        records = records.reset_index(drop=True)
        std = harmonise(records)
        notes = [[] for _ in range(len(records))]
        ignored = [c for c in records.columns if c not in FEATURES]
        for i in range(len(records)):
            if ignored:
                notes[i].append(f"ignored (not model inputs): {', '.join(ignored)}")
            for col in NUMERIC_FEATURES:
                given = records.at[i, col] if col in records else None
                value = std.at[i, col]
                if _blank(given):
                    notes[i].append(f"{col} missing -> imputed")
                elif pd.isna(value):
                    notes[i].append(f"{col}={given!r} invalid or implausible -> treated as missing and imputed")
                else:
                    if col == "fasting_glucose" and to_number(pd.Series([given]))[0] < GLUCOSE_MMOL_CUTOFF:
                        notes[i].append("fasting_glucose < 30 read as mmol/L and converted to mg/dL")
                    low, high = self.feature_ranges.get(col, (None, None))
                    if low is not None and not low <= value <= high:
                        notes[i].append(f"{col}={value} outside training range [{low}, {high}]")
            for col in CATEGORICAL_FEATURES:
                given = records.at[i, col] if col in records else None
                if _blank(given):
                    notes[i].append(f"{col} missing -> '{UNKNOWN}'")
                elif std.at[i, col] == UNKNOWN and str(given).strip().lower() != UNKNOWN.lower():
                    notes[i].append(f"{col}={given!r} not recognised -> '{UNKNOWN}'")
        return std[FEATURES], notes

    def predict_frame(self, records):
        if isinstance(records, dict):
            records = pd.DataFrame([records])
        if not isinstance(records, pd.DataFrame):
            raise TypeError("records must be a dict or a pandas DataFrame")
        X, notes = self._prepare(records)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # unseen categories -> all-zero one-hot
            proba = self.pipeline.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "probability": proba.round(4),
            "predicted_class": (proba >= self.threshold).astype(int),
            "risk_category": [self.risk_category(p) for p in proba],
            "model_version": self.model_version,
            "warnings": ["; ".join(n) for n in notes],
        })

    def predict_one(self, record):
        if not isinstance(record, dict):
            raise TypeError("record must be a dict of field -> value")
        row = self.predict_frame(record).iloc[0]
        return {"probability": float(row["probability"]), "predicted_class": int(row["predicted_class"]),
                "risk_category": row["risk_category"], "model_version": self.model_version,
                "threshold": self.threshold, "warnings": [w for w in row["warnings"].split("; ") if w],
                "note": NOTE}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score patient records with the trained model.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="one patient record as a JSON object")
    group.add_argument("--input", help="CSV of patient records (raw or cleaned format)")
    parser.add_argument("--output", help="where to write predictions for --input (keep it out of git)")
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
        print(result["risk_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
