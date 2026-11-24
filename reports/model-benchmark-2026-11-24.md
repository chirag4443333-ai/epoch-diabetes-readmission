# Model benchmark on the supplied diabetes dataset

Run date: 2026-11-24

Dataset: 4,000 labeled rows; 22.7% positive. No row-level data was added to this repository.

## Protocol

Used the notebook's cleaning rules and the same deterministic, stratified 80/20 split (random seed 42): 3,200 development rows and 800 holdout rows. Candidate model families and hyperparameter grids came from `src/readmission/models.py`. Tuning used five-fold stratified cross-validation on the development split. Each threshold was selected from development out-of-fold predictions by maximizing F2; the holdout was used only for final evaluation.

This run uses median imputation and the notebook's feature preparation to keep the comparison directly aligned with the notebook benchmark. It does not compare every preprocessing option in the repository.

## Results

| Model | CV ROC-AUC | CV PR-AUC | Holdout ROC-AUC | Holdout PR-AUC | F2 threshold | Accuracy | Recall | Precision | Flagged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.813 | 0.588 | 0.786 | 0.534 | 0.15 | 0.636 | 0.790 | 0.361 | 49.5% |
| Hist Gradient Boosting | 0.807 | 0.577 | 0.786 | 0.509 | 0.14 | 0.639 | 0.807 | 0.365 | 50.0% |
| Random Forest | 0.801 | 0.559 | 0.782 | 0.512 | 0.17 | 0.649 | 0.785 | 0.370 | 48.0% |
| Extra Trees | 0.797 | 0.557 | 0.770 | 0.506 | 0.17 | 0.663 | 0.751 | 0.377 | 45.1% |

The full aggregate table is in [`reports/results/diabetes-model-benchmark-2026-11-24.csv`](results/diabetes-model-benchmark-2026-11-24.csv).

## Findings and limits

Logistic Regression had the highest mean cross-validation ROC-AUC and the highest holdout PR-AUC. Hist Gradient Boosting had nearly identical holdout ROC-AUC, but it did not improve PR-AUC. No tested candidate demonstrates a reliable predictive improvement over the notebook's holdout ROC-AUC of 0.785; the 0.001 difference for Logistic Regression is too small to treat as a real gain from one split.

The thresholds prioritize recall through F2, so accuracy is intentionally lower than the 77.4% accuracy of predicting no readmission for everyone. That all-negative rule would miss every readmission, so plain accuracy is a poor success measure for this follow-up-slot problem. Compare recall, precision, flagged share, PR-AUC, and top-k capture at the actual call capacity.

## Recommendation

Keep Logistic Regression as the primary model for now. The current experiment does not support claiming higher accuracy. Next, evaluate top-k recall/precision at the available follow-up capacity and validate on a separate future cohort before making a clinical claim.
