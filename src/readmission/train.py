"""End-to-end training and evaluation; writes aggregate reports and the model bundle.

    python -m readmission.train                               # stratified 80/20 holdout of the training file
    python -m readmission.train --test-path path/to/test.csv  # official / hidden test file instead
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from . import evaluation as ev
from .config import DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR, DEFAULT_TRAIN_PATH, MODEL_VERSION, NUMERIC_FEATURES
from .data import features_and_target, load_dataset, validate_dataset
from .experiment import (build_final_pipeline, choose_preprocessing, compare_models, compare_preprocessing,
                         holdout_split, select_model, threshold_stability)
from .interpret import cv_permutation_importance, logistic_coefficients, native_importance, subgroup_table
from .leakage import assert_no_forbidden_features, audit_leakage
from .predict import ReadmissionPredictor


def run(train_path=DEFAULT_TRAIN_PATH, test_path=None, report_dir=DEFAULT_REPORT_DIR, model_dir=DEFAULT_MODEL_DIR,
        quick=False, figures=True, log=print):
    report_dir = Path(report_dir)
    tables, figs = report_dir / "tables", report_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # 1. data checks + leakage audit
    data = load_dataset(train_path, is_training=True)
    summary = {**validate_dataset(data), **data.attrs}
    leakage = audit_leakage(data)
    log("[data]", summary)
    log("[leakage]", {k: v for k, v in leakage.items() if k != "single_feature_auc"})

    # 2. validation design
    if test_path:
        dev, held, set_name = data, load_dataset(test_path, is_training=False), "test"
        validate_dataset(held, require_target=False)
    else:
        (dev, held), set_name = holdout_split(data), "holdout"
    X_dev, y_dev = features_and_target(dev)
    X_held, y_held = features_and_target(held)
    assert_no_forbidden_features(X_dev.columns)
    assert_no_forbidden_features(X_held.columns)

    # 3-4. preprocessing and model comparison (development data only)
    prep = compare_preprocessing(X_dev, y_dev)
    imputation, indicator = choose_preprocessing(prep)
    log("[preprocessing] chosen:", imputation, "imputation, missing indicators =", indicator)
    results, fold_scores, oof = compare_models(X_dev, y_dev, imputation, indicator, quick)
    chosen, reason = select_model(results)
    params = results.set_index("model").loc[chosen, "best_params"]
    log(results[["model", "cv_roc_auc_mean", "cv_roc_auc_sd", "cv_pr_auc_mean", "cv_brier_mean"]].round(4).to_string(index=False))
    log("[selection]", reason)

    # 5. operating threshold from development out-of-fold predictions only
    threshold = ev.choose_threshold(y_dev, oof[chosen])
    high = round(min(0.95, max(0.5, threshold + 0.1)), 2)
    thr_table = ev.threshold_table(y_dev, oof[chosen])
    stability = threshold_stability(y_dev, oof[chosen])
    log("[threshold] F2-optimal on OOF =", threshold, "per fold =", stability)

    # 6. final fit on development data; one evaluation on the untouched holdout / test data
    pipeline = build_final_pipeline(chosen, params, imputation, indicator, quick).fit(X_dev, y_dev)
    ranges = {c: [float(X_dev[c].min()), float(X_dev[c].max())] for c in NUMERIC_FEATURES}
    ReadmissionPredictor(pipeline, threshold, high, MODEL_VERSION, chosen, ranges).save(model_dir)
    proba = pipeline.predict_proba(X_held)[:, 1]
    final, subgroups = None, None
    if y_held is not None:
        labelled = y_held.notna().values
        yh, ph = y_held[labelled].astype(int).values, proba[labelled]
        final = {"name": set_name, "n": int(labelled.sum()), "prevalence": float(yh.mean()),
                 "at_threshold": ev.metrics_at_threshold(yh, ph, threshold),
                 "at_0.5": ev.metrics_at_threshold(yh, ph, 0.5),
                 "roc_auc_95ci": ev.bootstrap_ci(yh, ph, roc_auc_score),
                 "pr_auc_95ci": ev.bootstrap_ci(yh, ph, average_precision_score)}
        subgroups = pd.concat([subgroup_table(dev, y_dev, oof[chosen], threshold).assign(data="development OOF"),
                               subgroup_table(held[labelled], yh, ph, threshold).assign(data=set_name)])
        log("[" + set_name + "]", {k: round(v, 4) for k, v in final["at_threshold"].items()})

    # 7. interpretation on development data (never used for any decision above)
    perm = cv_permutation_importance(build_final_pipeline(chosen, params, imputation, indicator, quick),
                                     X_dev, y_dev, n_repeats=2 if quick else 5)
    coefs = logistic_coefficients(pipeline) if chosen == "logistic_regression" else None
    forest = results[results["model"].isin(["random_forest", "extra_trees"])].sort_values("cv_roc_auc_mean").iloc[-1]
    tree_imp = native_importance(build_final_pipeline(forest["model"], forest["best_params"], imputation,
                                                      indicator, quick).fit(X_dev, y_dev))

    ctx = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"), "model_version": MODEL_VERSION,
           "sklearn_version": sklearn.__version__, "data": summary, "leakage": leakage, "final_set": set_name,
           "n_development": int(len(dev)), "n_final_set": int(len(held)),
           "preprocessing": {"imputation": imputation, "missing_indicators": indicator},
           "selected_model": chosen, "selected_params": params, "selection_reason": reason,
           "threshold": threshold, "high_risk_threshold": high, "threshold_per_fold": stability,
           "oof_at_threshold": ev.metrics_at_threshold(y_dev, oof[chosen], threshold),
           "oof_at_0.5": ev.metrics_at_threshold(y_dev, oof[chosen], 0.5), "fold_roc_auc": fold_scores,
           "final_metrics": final, "tree_model_for_importance": forest["model"]}
    frames = {"preprocessing_comparison": prep,
              "model_comparison": results.assign(best_params=results["best_params"].astype(str)),
              "threshold_table_dev_oof": thr_table, "permutation_importance": perm, "tree_importance": tree_imp,
              "logistic_coefficients": coefs, "subgroups": subgroups}
    for name, frame in frames.items():
        if frame is not None:
            frame.round(4).to_csv(tables / (name + ".csv"), index=False)
    (report_dir / "results.json").write_text(json.dumps(ctx, indent=2, default=str))
    write_results_markdown(ctx, frames, report_dir / "RESULTS.md")
    if figures:
        ev.plot_cv_comparison(fold_scores, figs / "cv_model_comparison.png")
        curves = {"development OOF": (y_dev, oof[chosen])}
        if final:
            curves[set_name] = (yh, ph)
            ev.plot_calibration_and_confusion(yh, ph, threshold, figs / "calibration_confusion.png")
        ev.plot_roc_pr(curves, figs / "roc_pr_curves.png")
        ev.plot_threshold_tradeoff(thr_table, threshold, figs / "threshold_tradeoff.png")
        ev.plot_importance(perm, "auc_drop_mean", "auc_drop_sd", "Permutation importance (drop in CV ROC-AUC)",
                           figs / "permutation_importance.png")
    ctx.update(frames=frames, pipeline=pipeline, oof=oof)
    return ctx


def md_table(df, digits=3):
    def cell(v):
        return str(round(v, digits)) if isinstance(v, float) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "".join("\n| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False))


def write_results_markdown(c, frames, path):
    f, d = c["final_metrics"], c["data"]
    keep = ["model", "best_params", "cv_roc_auc_mean", "cv_roc_auc_sd", "cv_pr_auc_mean", "cv_pr_auc_sd", "cv_brier_mean"]
    lines = ["# Results (generated by python -m readmission.train, " + c["created_utc"] + ")", "",
             "All numbers come from the executed run that wrote this file; nothing is typed in by hand.", "",
             "## Data and leakage checks",
             f"- rows {d['rows']} (read {d['rows_read']}, duplicates removed {d['duplicates_removed']}), "
             f"positives {d['positives']}, prevalence {d['prevalence']}",
             f"- originally-missing values restored as NaN (imputed inside folds): {d['missing_values_by_column']}",
             f"- readmission_ward ROC-AUC if it were used: {c['leakage'].get('readmission_ward_auc_if_used')} "
             f"(filled only for readmitted patients: {c['leakage'].get('readmission_ward_filled_only_for_readmitted')}) -> excluded",
             f"- development rows {c['n_development']}, {c['final_set']} rows {c['n_final_set']}", "",
             "## Preprocessing comparison (5-fold stratified CV, development data)",
             md_table(frames["preprocessing_comparison"], 4), "",
             f"Chosen: **{c['preprocessing']['imputation']} imputation, missing indicators = "
             f"{c['preprocessing']['missing_indicators']}**", "",
             "## Model comparison (same 5 folds for every model)", md_table(frames["model_comparison"][keep], 4), "",
             "Selection: " + c["selection_reason"], "",
             "## Threshold analysis (development out-of-fold predictions)", md_table(frames["threshold_table_dev_oof"], 3), "",
             f"Operating threshold (max F2 on OOF): **{c['threshold']}**; F2-optimal threshold per fold: "
             f"{c['threshold_per_fold']}. High-risk band starts at {c['high_risk_threshold']}.", ""]
    if f:
        t, h = f["at_threshold"], f["at_0.5"]
        lines += [f"## Final {f['name']} metrics (n = {f['n']}, prevalence {round(f['prevalence'], 3)}) - computed once", "",
                  f"| metric | at threshold {c['threshold']} | at 0.5 |", "|---|---|---|"]
        for k in ["roc_auc", "pr_auc", "brier", "recall", "precision", "specificity", "f1", "balanced_accuracy",
                  "accuracy", "flagged_rate", "tp", "fn", "fp", "tn"]:
            lines.append(f"| {k} | {round(t[k], 3)} | {round(h[k], 3)} |")
        lo, hi = f["roc_auc_95ci"]
        plo, phi = f["pr_auc_95ci"]
        lines += ["", f"95 percent bootstrap CI: ROC-AUC {round(lo, 3)} to {round(hi, 3)}, "
                      f"PR-AUC {round(plo, 3)} to {round(phi, 3)}", ""]
    lines += ["## Permutation importance (drop in validation ROC-AUC, 5-fold CV)",
              md_table(frames["permutation_importance"], 4), ""]
    if frames["logistic_coefficients"] is not None:
        lines += ["## Logistic regression coefficients (standardised inputs)", md_table(frames["logistic_coefficients"], 3), ""]
    lines += ["## Tree-based importance (" + c["tree_model_for_importance"] + ", cross-check)",
              md_table(frames["tree_importance"].head(12), 4), ""]
    if frames["subgroups"] is not None:
        lines += ["## Subgroups (descriptive only - small groups, wide uncertainty)", md_table(frames["subgroups"], 3), ""]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the 30-day readmission model.")
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=None, help="labelled test file; default = 20 percent stratified holdout")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--quick", action="store_true", help="tiny grids (smoke tests only)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    ctx = run(args.train_path, args.test_path, args.report_dir, args.model_dir, args.quick, not args.no_figures)
    if ctx["final_metrics"]:
        print("FINAL", ctx["final_set"].upper(), "ROC-AUC:", round(ctx["final_metrics"]["at_threshold"]["roc_auc"], 4))


if __name__ == "__main__":
    main()
