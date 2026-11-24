# EPOCH Datathon 2026 - 30-day readmission risk for diabetes patients

**User:** a discharge coordinator or diabetes care team. **Decision:** which patients get extra
post-discharge follow-up (an early call or clinic slot). **When:** at discharge.
**Target:** `readmitted_30d` (1 = readmitted within 30 days). This is a decision-support *prototype*.
It is not a diagnostic tool, it is not clinically validated and it gives no treatment advice.
A human reviews every flag.

## Results at a glance (executed run - full tables in `reports/RESULTS.md`)
| | value |
|---|---|
| Selected model | Logistic regression (L2, C = 0.03), iterative imputation + missing indicators |
| 5-fold CV ROC-AUC (development, n = 3,200) | 0.813 +/- 0.015 SD |
| **Holdout ROC-AUC (untouched 20 %, n = 800)** | **0.779** (95 % bootstrap CI 0.738 - 0.818) |
| Holdout PR-AUC / Brier | 0.533 (CI 0.460 - 0.612) / 0.142 (prevalence 0.226) |
| Operating threshold (max F2 on development OOF) | 0.15 (per-fold optimum 0.12 - 0.17) |
| Holdout at 0.15 | recall 0.768, precision 0.359, specificity 0.599, 48.4 % flagged |
| Holdout counts at 0.15 | 181 readmissions: 139 detected, 42 missed; 248 flagged but not readmitted |
| For comparison at 0.5 | recall 0.354, precision 0.593, 13.5 % flagged |

Model comparison (same folds, CV ROC-AUC): logistic regression **0.813**, HistGradientBoosting 0.804,
random forest 0.799, extra trees 0.794, dummy 0.500. Predicting "no readmission" for everyone gives
77.4 % accuracy, so accuracy is not used to pick models.

An additional median-imputation benchmark of the candidate model families is documented in
[`reports/model-benchmark-2026-11-24.md`](reports/model-benchmark-2026-11-24.md); it did not establish a reliable improvement over the primary model.

## Repository layout
| path | purpose |
|---|---|
| `src/readmission/config.py` | column roles, forbidden features, seeds, paths |
| `src/readmission/data.py` | load either file format, apply cleaning-log row rules, restore missing values, validate |
| `src/readmission/leakage.py` | FeatureGuard pipeline step + leakage audit (training fails on forbidden columns) |
| `src/readmission/preprocessing.py` | imputation / scaling / one-hot pipelines, fitted inside training folds only |
| `src/readmission/models.py` | candidate models and small search grids |
| `src/readmission/experiment.py` | holdout split, CV folds, preprocessing + model comparison, selection rule |
| `src/readmission/evaluation.py` | metrics, threshold table, bootstrap CIs, figures |
| `src/readmission/interpret.py` | permutation importance, coefficients, tree importance, subgroups |
| `src/readmission/train.py` | one command: train, compare, select, threshold, evaluate, save |
| `src/readmission/predict.py` | ReadmissionPredictor (record in: probability, class, risk band, version out) + CLI |
| `scripts/build_notebook.py` | builds the self-contained competition notebook from the package |
| `notebooks/TeamName_Diabetes_Notebook.ipynb` | competition notebook (outputs stripped) |
| `reports/RESULTS.md` | executed results - aggregate only |
| `MODEL_CARD.md` | intended use, performance, limitations, oversight |
| `tests/` | pytest suite (synthetic data; real-data checks run only when the data is present) |

## Data (not in this repository)
The event data must not be published. Put `diabetes_train_cleaned.csv` (Phase 1 output) in `data/`.
Optionally add the raw `train.csv` there too, which enables the "raw reproduces cleaned" test. `.gitignore`
blocks `data/`, all CSVs except the aggregate tables, models, predictions and executed notebooks. The Phase 1
cleaning log (`CLEANING_LOG.md`) is kept with the data in the team Drive.

Checked before modelling: 4,000 unique patients, 30 columns, zero missing cells, 906 positives / 3,094
negatives (22.65 %). Running the same row rules on the raw file reproduces the cleaned file exactly:
4,120 rows, 120 duplicates removed, identical values. That supports the cleaning log's counts.

## Features
- **Used (16):** age, gender, bmi, waist_cm, fasting_glucose, hba1c, bp_systolic, cholesterol_total, hdl,
  triglycerides, years_since_diagnosis, medication, physical_activity (`Unknown` kept as its own category),
  smoking_status, visits_last_year, length_of_stay_days. Length of stay is allowed because the prediction is
  made at discharge (assumption).
- **Excluded:**
  - `readmission_ward`: describes the readmission itself, is filled only for readmitted patients, and would
    give ROC-AUC 1.0 (target leakage).
  - `patient_id`: an identifier.
  - `admission_date`: period-specific; readmission rate is flat across quarters.
  - `*_was_missing` flags: used only to restore missing values.

## How leakage is prevented
1. Feature whitelist in a fixed order. `FeatureGuard` is the first pipeline step and raises `LeakageError` if a
   forbidden column or a missing-value flag reaches `fit` or `transform`.
2. A leakage audit proves the ward leak, then fails if any allowed feature predicts the target almost
   perfectly (single-feature AUC >= 0.95).
3. Phase 1 imputed values using all 4,000 rows. Here those values are set back to NaN in memory (the CSV is
   unchanged) and imputation is re-learned inside each training fold.
4. The 20 % holdout (stratified, seed 42, disjoint patient IDs) is used once, after preprocessing, model,
   hyperparameters and threshold are all frozen.

## Method
- **Validation:** stratified 80/20 split, then stratified 5-fold CV (seed 42) on the 80 %. Every comparison
  uses the same folds. If an official labelled test file is passed (`--test-path` / `TEST_PATH`), all
  training rows become development data and that file is the final test.
- **Preprocessing comparison:** median vs iterative regression imputation, with and without missing
  indicators, for logistic regression and HistGB. The rule was fixed in advance (best mean AUC). All variants
  fall within 0.002 AUC, which is noise. Iterative + indicators was selected. Scaling is applied for logistic
  regression only. One-hot encoding uses fixed levels, and unseen levels are encoded as all zeros.
- **Models:** dummy, logistic regression, HistGradientBoosting, random forest and extra trees, each tuned on
  a small grid. XGBoost and LightGBM were not used: HistGB already covers the same algorithm family and the
  tree models did not beat the linear model. Class weights were not used, which keeps probabilities
  calibrated. Imbalance is handled by the threshold.
- **Selection:** the simplest model within one standard error of the best mean CV ROC-AUC. Logistic
  regression was the best model outright.
- **Threshold:** maximise F2 on development out-of-fold predictions, because missing a readmission is assumed
  to cost more than an extra follow-up call. Risk bands are Low < 0.15 <= Elevated < 0.50 <= High. The F2
  threshold flags about half of all patients. If follow-up capacity is limited, use the threshold table in
  `reports/RESULTS.md`; for example, 0.25 gives recall 0.70 with 33 % flagged. These are prototype settings,
  not clinically optimal ones.

## What drives predictions
These are associations, not causes:
- **Permutation importance (CV drop in AUC):** hba1c 0.045, visits_last_year 0.036, length_of_stay 0.031,
  fasting_glucose 0.013, medication 0.010, age 0.005. Everything else is about 0.
- **Coefficients:** higher HbA1c, a longer stay, combination therapy and older age raise risk. More clinic
  visits in the past year lower risk.
- **Random forest cross-check:** ranks the same top features.
- **Checks for suspicious reliance:**
  - HbA1c and glucose are strongly correlated (r = 0.89), so their weight is shared between them.
  - Missing-indicator coefficients rest on very few rows (for example, 20 rows for bp_systolic) and add no
    measurable AUC. Treat them as artefacts.
  - The low-value pile-ups (HbA1c 4.0, glucose 60) follow the same trend as other low values.

## Reproduce (from the repository root)
1. Create a fresh virtual environment (Python 3.10+).
2. Use pip to install the packages in `requirements.txt`, then install this project in editable mode (`-e .`).
3. Run:
```bash
python -m readmission.train                                 # reports/ + models/ (~3 min on 1 CPU)
python -m readmission.train --test-path path/to/test.csv    # official / hidden labelled test file
python -m pytest -q                                         # test suite
python scripts/build_notebook.py --execute                  # rebuild + run the competition notebook
```

## Predict
```bash
python -m readmission.predict --input new_patients.csv --output local_predictions.csv   # keep out of git
```
In Python: `ReadmissionPredictor.load("models").predict_one(record)` returns probability,
predicted_class, risk_category, model_version, threshold, warnings and a note. The predictor applies the
training-time row rules and uses the exact training column order. Missing, invalid or implausible values are
imputed with a warning. Unknown categories become `Unknown`, and non-feature fields such as `patient_id` or
`readmission_ward` are ignored.

## Competition notebook
`notebooks/TeamName_Diabetes_Notebook.ipynb`:
- `TRAIN_PATH` and `TEST_PATH` are defined and loaded in the first cell, and the final score is printed at
  the end.
- It handles raw-format and cleaned-format files, and contains an AI usage note.
- Before you submit:
  1. Rename `TeamName` to your registered team name.
  2. Edit the AI usage note so it describes what your team did.
  3. Upload it with the data to Colab and run all cells.

## Limitations and next steps
- One small dataset, with no external or temporal validation.
- The holdout AUC (0.779) is below the CV estimate (0.813). The holdout figure is the honest one.
- Subgroups are small. The holdout 45-54 band shows lower recall (0.50, n = 199); monitor it, don't
  over-interpret it.
- Important information is unknown or absent: units for several labs, causes of readmission, social
  factors, and what happened after discharge.
- Next steps:
  - Confirm the glucose-unit and floor-value questions with a mentor.
  - Validate on later admissions.
  - Calibrate on local data.
  - Agree the threshold with clinicians according to follow-up capacity.
