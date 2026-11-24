"""Build the self-contained competition notebook from the package source.

    python scripts/build_notebook.py            # writes notebooks/TeamName_Diabetes_Notebook.ipynb
    python scripts/build_notebook.py --execute  # also runs it end-to-end (needs the data file)

Judges re-run the notebook on a hidden dataset, so it cannot import a local package:
every module in src/readmission is inlined (one cell per module). The package stays
the single source of truth - edit the modules, then rebuild.
"""
import argparse
import ast
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "readmission"
OUT = ROOT / "notebooks" / "TeamName_Diabetes_Notebook.ipynb"
EXECUTED = ROOT / "notebook_outputs" / "TeamName_Diabetes_Notebook.executed.ipynb"
MODULES = ["config", "data", "leakage", "preprocessing", "models", "evaluation", "experiment", "interpret",
           "predict", "train"]

FIRST_CELL = '''# Rulebook: data is loaded through TRAIN_PATH and TEST_PATH in the first cell.
# TEST_PATH = None -> the final test set is a stratified 20% holdout of TRAIN_PATH (seed 42).
# For the hidden dataset set TEST_PATH to that file (raw competition format or cleaned format both work).
import pandas as pd

TRAIN_PATH = "data/diabetes_train_cleaned.csv"
TEST_PATH = None

train_input = pd.read_csv(TRAIN_PATH, dtype=str)
test_input = pd.read_csv(TEST_PATH, dtype=str) if TEST_PATH else None
print("train file:", train_input.shape, "| test file:", None if test_input is None else test_input.shape)'''

INTRO = '''# EPOCH Datathon 2026 - Diabetes theme: 30-day readmission risk at discharge

**Target user:** a hospital discharge coordinator / diabetes care team deciding which patients get extra
post-discharge follow-up (e.g. an early phone call or clinic slot). **Prediction point:** discharge.
**Target:** `readmitted_30d` (1 = readmitted within 30 days). Clinical decision-support *prototype* only -
not a diagnosis, not clinically validated, no treatment recommendations; every flag is reviewed by a human.

**Pipeline in one line:** validated data -> leakage audit -> stratified 80/20 split (seed 42) -> 5-fold CV on the
80% for preprocessing choice, model choice, tuning and threshold -> one final evaluation on the untouched 20%.

Cleaning (Phase 1) is documented in `CLEANING_LOG.md`. The same row-level rules are re-applied here so that a
raw hidden test file is handled exactly like the training data.

## AI usage note (rulebook requirement - team: edit so it matches what you did)
- **Team:** chose the theme, target user and decision; ran Phase 1 cleaning and wrote/approved the cleaning log;
  set the requirements for this phase (no `readmission_ward`, `patient_id`, `admission_date`; imputation inside
  folds; 80/20 + 5-fold CV; recall-oriented threshold; no holdout tuning); reviewed the results.
- **AI (Claude, via Composio):** wrote the Phase 2 code (package, tests, this generated notebook), ran the
  experiments and drafted the documentation. All numbers shown are produced by running this notebook.'''


def inline_source(name):
    """Module source without relative imports / __main__ blocks (all names share one namespace)."""
    src = (PKG / f"{name}.py").read_text()
    drop = set()
    for node in ast.parse(src).body:
        relative = isinstance(node, ast.ImportFrom) and node.level > 0
        main_guard = isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
        if relative or main_guard:
            drop.update(range(node.lineno, node.end_lineno + 1))
    code = "\n".join(line for i, line in enumerate(src.splitlines(), 1) if i not in drop).strip() + "\n"
    return re.sub(r"\bev\.", "", code) if name == "train" else code


def story():
    md, code = new_markdown_cell, new_code_cell
    fig = 'display(Image(filename="notebook_outputs/figures/{}"))'
    return [
        md("## 2. Run the complete experiment\nOne call runs every step below on the development data, then evaluates "
           "the selected model once on the untouched final set. Patient-level data is never printed."),
        code('import warnings\nfrom IPython.display import Image, display\nwarnings.filterwarnings("ignore")\n\n'
             'ctx = run(TRAIN_PATH, TEST_PATH, report_dir="notebook_outputs", model_dir="notebook_outputs/model")\n'
             'T = ctx["frames"]'),
        md("## 3. Data validation and leakage audit\n`readmission_ward` is filled only for readmitted patients "
           "(post-outcome information). It would give ROC-AUC 1.0, so it is excluded; the pipeline's first step "
           "raises an error if it (or `patient_id`, `admission_date`, the target, or any `*_was_missing` flag) "
           "ever reaches the model. Values that Phase 1 imputed on all 4,000 rows are set back to missing and "
           "re-imputed inside each training fold."),
        code('display(pd.Series({k: v for k, v in ctx["data"].items()}, name="value").to_frame())\n'
             'print({k: v for k, v in ctx["leakage"].items() if k != "single_feature_auc"})\n'
             'print("single-feature ROC-AUC (none near 1.0):", ctx["leakage"]["single_feature_auc"])'),
        md("## 4. Preprocessing comparison (5-fold stratified CV, development data only)\nMedian vs iterative "
           "regression imputation, with / without missing indicators, for two fixed reference models. Rule fixed "
           "in advance: best mean ROC-AUC across both models. Differences are small - read them as 'within noise'."),
        code('display(T["preprocessing_comparison"].round(4))\nprint("chosen:", ctx["preprocessing"])'),
        md("## 5. Model comparison (identical folds, small grids)\nSelection: best mean CV ROC-AUC, but the simplest "
           "model within one standard error wins (interpretability)."),
        code('display(T["model_comparison"].round(4))\nprint(ctx["selection_reason"])\n' + fig.format("cv_model_comparison.png")),
        md("## 6. Operating threshold (development out-of-fold predictions only)\nMissing a readmission is assumed "
           "to cost more than an extra follow-up call, so the threshold maximises F2 (recall weighted above "
           "precision). It is a prototype setting, not a clinically optimal one; the table shows the trade-off."),
        code('display(T["threshold_table_dev_oof"].round(3))\nprint("threshold:", ctx["threshold"], "| per-fold:", '
             'ctx["threshold_per_fold"], "| High-risk band from:", ctx["high_risk_threshold"])\n'
             + fig.format("threshold_tradeoff.png")),
        md("## 7. Final evaluation (computed once, after every decision was frozen)"),
        code('fm = ctx["final_metrics"]\nif fm:\n    m = fm["at_threshold"]\n'
             '    display(pd.DataFrame({"at chosen threshold": fm["at_threshold"], "at 0.5": fm["at_0.5"]}).round(3))\n'
             '    print(f"{m[\'tp\'] + m[\'fn\']} readmissions: {m[\'tp\']} detected, {m[\'fn\']} missed; "\n'
             '          f"{m[\'fp\']} patients flagged who were not readmitted; {m[\'flagged_rate\']:.1%} of patients flagged.")\n'
             '    print("95% bootstrap CI ROC-AUC:", [round(v, 3) for v in fm["roc_auc_95ci"]], "| PR-AUC:", '
             '[round(v, 3) for v in fm["pr_auc_95ci"]])\n'
             '    display(Image(filename="notebook_outputs/figures/calibration_confusion.png"))\n'
             + fig.format("roc_pr_curves.png")),
        md("## 8. What drives the predictions\nPermutation importance (drop in validation ROC-AUC, CV on development "
           "data), logistic-regression coefficients (standardised inputs) and a tree-model cross-check. These are "
           "associations in this dataset, not causes."),
        code('display(T["permutation_importance"].round(4))\nif T["logistic_coefficients"] is not None:\n'
             '    display(T["logistic_coefficients"].round(3))\ndisplay(T["tree_importance"].head(10).round(4))\n'
             + fig.format("permutation_importance.png")),
        md("## 9. Subgroups (descriptive only)\nGroups are small, so differences are uncertain; no fairness claim is made."),
        code('if T["subgroups"] is not None:\n    display(T["subgroups"].round(3))'),
        md("## 10. Using the model for a new patient (made-up example record)"),
        code('predictor = ReadmissionPredictor.load("notebook_outputs/model")\nexample = {"age": 67, "gender": "F", '
             '"bmi": 31.2, "waist_cm": 104, "fasting_glucose": 182, "hba1c": 8.4, "bp_systolic": 138,\n'
             '           "cholesterol_total": 201, "hdl": 44, "triglycerides": 190, "years_since_diagnosis": 14,\n'
             '           "medication": "Insulin", "physical_activity": "Low", "smoking_status": "Former",\n'
             '           "visits_last_year": 2, "length_of_stay_days": 7}\npredictor.predict_one(example)'),
        md("## 11. Limitations\nSingle synthetic/event dataset; no external or temporal validation; CV score is "
           "slightly optimistic (tuned on the same folds) - the final-set score is the honest estimate; threshold "
           "and risk bands are prototype choices; subgroup results are uncertain; correlation is not causation."),
        code('fm = ctx["final_metrics"]\nif fm:\n    m = fm["at_threshold"]\n'
             '    print(f"FINAL EVALUATION SCORE - ROC-AUC on the {fm[\'name\']} set (n={fm[\'n\']}): {m[\'roc_auc\']:.4f}")\n'
             '    print(f"PR-AUC {m[\'pr_auc\']:.4f} | threshold {ctx[\'threshold\']}: recall {m[\'recall\']:.3f}, "\n'
             '          f"precision {m[\'precision\']:.3f}, specificity {m[\'specificity\']:.3f}")\nelse:\n'
             '    predictions = ReadmissionPredictor.load("notebook_outputs/model").predict_frame(test_input)\n'
             '    predictions.to_csv("notebook_outputs/test_predictions.csv", index=False)\n'
             '    print("TEST_PATH has no labels -> predictions saved to notebook_outputs/test_predictions.csv")'),
    ]


def build():
    cells = [new_code_cell(FIRST_CELL), new_markdown_cell(INTRO),
             new_markdown_cell("## 1. Project code\nInlined from `src/readmission/` so the notebook runs anywhere. "
                               "Each cell is one module; read the docstrings for what it does.")]
    for name in MODULES:
        cells += [new_markdown_cell(f"### `{name}.py`"), new_code_cell(inline_source(name))]
    nb = new_notebook(cells=cells + story())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    OUT.parent.mkdir(exist_ok=True)
    nbformat.write(nb, OUT)
    return nb


def execute(nb):
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    EXECUTED.parent.mkdir(exist_ok=True)
    nbformat.write(nb, EXECUTED)  # outputs stay local (git-ignored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebook = build()
    print("wrote", OUT.relative_to(ROOT))
    if args.execute:
        execute(notebook)
        print("executed OK ->", EXECUTED.relative_to(ROOT))
