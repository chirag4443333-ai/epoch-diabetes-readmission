"""Metrics beyond accuracy, threshold analysis, bootstrap intervals and figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from .config import RANDOM_STATE


def _ratio(num, den):
    return float(num / den) if den else 0.0


def metrics_at_threshold(y_true, proba, threshold):
    """Ranking metrics (ROC-AUC, PR-AUC, Brier) plus the confusion matrix at one threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    precision, recall, specificity = _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "recall": recall, "precision": precision, "specificity": specificity,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "f2": _ratio(5 * precision * recall, 4 * precision + recall),
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": _ratio(tp + tn, len(y)),
        "flagged_rate": _ratio(tp + fp, len(y)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(len(y)),
    }


def threshold_table(y_true, proba, thresholds=None):
    """Effect of the threshold on recall, precision, specificity and share of patients flagged."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.751, 0.05), 2)
    cols = ["threshold", "flagged_rate", "recall", "precision", "specificity", "f1", "f2",
            "balanced_accuracy", "tp", "fp", "fn", "tn"]
    return pd.DataFrame([metrics_at_threshold(y_true, proba, t) for t in thresholds])[cols]


def choose_threshold(y_true, proba, beta=2.0):
    """Threshold maximising F-beta (beta=2: recall weighted above precision) on out-of-fold
    development predictions. Never call this with holdout / test labels."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    best_t, best_score = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        precision, recall = _ratio(tp, pred.sum()), _ratio(tp, (y == 1).sum())
        score = _ratio((1 + beta**2) * precision * recall, beta**2 * precision + recall)
        if score > best_score + 1e-12:
            best_t, best_score = float(t), score
    return best_t


def bootstrap_ci(y_true, proba, metric, n_boot=1000, seed=RANDOM_STATE):
    """Percentile 95% interval of metric(y, p) by resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            stats.append(metric(y[idx], p[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


# ----------------------------------------------------------------------------- figures
def plot_cv_comparison(fold_scores, path=None):
    """Box plot of per-fold ROC-AUC for every model ({name: [fold scores]})."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(fold_scores)
    ax.boxplot([fold_scores[n] for n in names], vert=False)
    ax.set_yticks(range(1, len(names) + 1), names)
    ax.set_xlabel("ROC-AUC per validation fold (5-fold CV, development data)")
    ax.set_title("Model comparison")
    return _finish(fig, path)


def plot_roc_pr(curves, path=None):
    """ROC and precision-recall curves for {label: (y_true, proba)}."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y, p):.3f})")
        ax2.plot(rec, prec, label=f"{label} (AP {average_precision_score(y, p):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    prevalence = float(np.mean(next(iter(curves.values()))[0]))
    ax2.axhline(prevalence, color="k", ls="--", lw=0.8, label=f"prevalence {prevalence:.3f}")
    ax1.set(xlabel="False positive rate", ylabel="Recall (sensitivity)", title="ROC curve")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    return _finish(fig, path)


def plot_threshold_tradeoff(table, chosen, path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in ["recall", "precision", "specificity", "flagged_rate"]:
        ax.plot(table["threshold"], table[col], marker="o", ms=3, label=col)
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"chosen {chosen:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Value", title="Threshold trade-off (development OOF)")
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_calibration_and_confusion(y_true, proba, threshold, path=None):
    m = metrics_at_threshold(y_true, proba, threshold)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    ax1.plot(mean_pred, frac_pos, marker="o", label="model")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax1.set(xlabel="Mean predicted probability", ylabel="Observed readmission rate",
            title=f"Calibration (Brier {m['brier']:.3f})")
    ax1.legend(fontsize=8)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax2.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax2.text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax2.set_xticks([0, 1], ["not flagged", "flagged"])
    ax2.set_yticks([0, 1], ["not readmitted", "readmitted"])
    ax2.set_title(f"Confusion matrix at threshold {threshold:.2f}")
    return _finish(fig, path)


def plot_importance(table, value_col, error_col=None, title="", path=None, top=16):
    data = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 1.2))
    ax.barh(data["feature"], data[value_col], xerr=data[error_col] if error_col else None, color="tab:blue")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title)
    return _finish(fig, path)


def _finish(fig, path):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
