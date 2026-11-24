# Model card - 30-day readmission risk prototype (v1.0.0)

**Model:** scikit-learn pipeline with these steps: feature guard, then iterative imputation with missing
indicators, standard scaling and one-hot encoding, then L2 logistic regression (C = 0.03).
It was trained on the 3,200-patient development split. Model version `1.0.0`.

## Intended use
Help a discharge coordinator or diabetes care team decide which patients to review for extra
post-discharge follow-up. It produces a probability, a flag (threshold 0.15) and a band:
Low < 0.15 <= Elevated < 0.50 <= High. Every output is reviewed by a clinician. Hospital follow-up
policies remain in charge.

## Out of scope
- Diagnosis.
- Treatment, medication or discharge decisions.
- Denying care or resources.
- Use on populations or sites different from this dataset.
- Any use as a validated clinical tool.

## Data
EPOCH Datathon 2026 diabetes dataset (event data: public or synthetic, no real patient records according to
the rulebook). Raw file: 4,120 rows. After Phase 1 de-duplication: 4,000 patients and 906 readmissions
(22.65 %). Phase 1 imputed values are restored to missing and re-imputed inside training folds. Excluded:
`readmission_ward` (leakage), `patient_id` and `admission_date`.

## Performance (executed run, see reports/RESULTS.md)
- 5-fold CV ROC-AUC on development data: 0.813 +/- 0.015.
- Untouched holdout (n = 800): ROC-AUC 0.779 (95 % CI 0.738 - 0.818), PR-AUC 0.533, Brier 0.142.
- At threshold 0.15: recall 0.768, precision 0.359, specificity 0.599, balanced accuracy 0.684,
  48.4 % flagged.
- Holdout confusion matrix at 0.15: TP 139, FN 42, FP 248, TN 371.
- At 0.5: recall 0.354, precision 0.593, 13.5 % flagged.

## Subgroups (descriptive, uncertain)
- **Holdout ROC-AUC:** F 0.75 / M 0.80; age <45 0.82, 45-54 0.67, 55-64 0.83, 65+ 0.78.
- **Flag rate:** rises with age (36 % under 45, 63 % at 65+), which follows the higher observed
  readmission rates in older patients.
- The 45-54 holdout group (n = 199) shows lower recall (0.50) than in development (0.82). Monitor it.
- Groups are too small to support fairness claims.

## Limitations and risks
- **Validation:** single dataset; no temporal or external validation.
- **Data quality:** units are partly assumed (glucose mmol/L converted to mg/dL) and some values look
  truncated at floors.
- **Interpretation:** associations only. Medication effects are confounded with disease severity.
- **Workload:** the threshold flags about half of patients, which means many false alarms. It must be
  matched to real follow-up capacity.
- **Behaviour:** inputs outside the training range produce warnings. Missing inputs are imputed, which
  lowers reliability.

## Human oversight
- Flags only prioritise review.
- Clinicians can override any output.
- Monitor recall, precision, calibration and subgroup performance on new admissions before any real use.
- Retrain or retire the model if performance drifts.
