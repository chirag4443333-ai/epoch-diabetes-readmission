import numpy as np
import pandas as pd
import pytest
from conftest import FLAGGED, REAL_CLEANED, REAL_RAW, make_synthetic, requires_real_data

from readmission.config import FEATURES, TARGET
from readmission.data import DataValidationError, harmonise, load_dataset, parse_target, validate_dataset


def test_flagged_values_restored_as_missing_and_file_untouched(synthetic_csv):
    before = synthetic_csv.read_bytes()
    raw, df = pd.read_csv(synthetic_csv), load_dataset(synthetic_csv)
    for col in FLAGGED:
        assert df[col].isna().sum() == raw[col + "_was_missing"].sum()
        assert df.loc[raw[col + "_was_missing"] == 0, col].notna().all()
    assert synthetic_csv.read_bytes() == before


def test_unknown_activity_is_kept_as_explicit_category(synthetic_csv):
    df = load_dataset(synthetic_csv)
    assert "Unknown" in set(df["physical_activity"]) and df["physical_activity"].notna().all()


def test_raw_format_rules_from_cleaning_log():
    raw = pd.DataFrame([{"patient_id": " db1 ", "age": "250", "gender": "female", "bmi": "0", "waist_cm": "999",
                         "fasting_glucose": "5.5", "hba1c": "7.0%", "bp_systolic": "400", "cholesterol_total": "200",
                         "hdl": "50", "triglycerides": "150", "years_since_diagnosis": "5", "medication": "Metformn",
                         "physical_activity": "", "smoking_status": "Nver", "visits_last_year": "2",
                         "length_of_stay_days": "3", "readmitted_30d": "Yes"}])
    out = harmonise(raw).iloc[0]
    assert all(np.isnan(out[c]) for c in ["age", "bmi", "waist_cm", "bp_systolic"])
    assert out.fasting_glucose == 99 and out.hba1c == 7.0 and out.patient_id == "DB1"
    assert (out.gender, out.medication, out.physical_activity, out.smoking_status) == ("F", "Metformin", "Unknown", "Never")
    assert out.readmitted_30d == 1


@pytest.mark.parametrize("value,expected", [("Yes", 1), ("y", 1), ("True", 1), ("1", 1), ("no", 0), ("N", 0), ("False", 0)])
def test_target_spellings(value, expected):
    assert parse_target(value) == expected


def test_invalid_target_and_schema_fail(tmp_path):
    df = load_dataset(_write(make_synthetic(), tmp_path))
    df.loc[0, TARGET] = 2
    with pytest.raises(DataValidationError):
        validate_dataset(df)
    with pytest.raises(DataValidationError):
        validate_dataset(df.drop(columns=["hba1c"]))


def test_training_files_deduplicated_but_test_files_never_filtered(tmp_path):
    raw = make_synthetic(n=50)
    path = _write(pd.concat([raw, raw.head(5)]), tmp_path)
    assert len(load_dataset(path, is_training=True)) == 50
    assert len(load_dataset(path, is_training=False)) == 55


@requires_real_data
def test_real_dataset_characteristics():
    raw = pd.read_csv(REAL_CLEANED)
    assert raw.shape == (4000, 30) and raw.isna().sum().sum() == 0 and not raw.duplicated().any()
    assert raw[TARGET].value_counts().to_dict() == {0: 3094, 1: 906}
    assert (raw["readmission_ward"].ne("Not readmitted") == raw[TARGET].astype(bool)).all()  # the leak
    summary = validate_dataset(load_dataset(REAL_CLEANED))
    assert summary["missing_values_by_column"]["hba1c"] == 391 and summary["positives"] == 906


@pytest.mark.skipif(not REAL_RAW.exists(), reason="raw event file not present")
def test_raw_file_reproduces_cleaned_file_exactly():
    clean, raw = load_dataset(REAL_CLEANED), load_dataset(REAL_RAW)
    assert raw.attrs["rows_read"] == 4120 and raw.attrs["duplicates_removed"] == 120
    pd.testing.assert_frame_equal(clean[FEATURES + [TARGET]], raw[FEATURES + [TARGET]], check_dtype=False)


def _write(df, tmp_path):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path
