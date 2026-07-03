"""Metrics, plots and threshold selection for the fraud classifier.

The dataset is ~577:1 imbalanced (492 fraud out of 284,807 transactions), so
accuracy is meaningless (predicting "not fraud" for everyone scores 99.8%).
We instead lean on precision/recall, PR-AUC (average precision) and a
business-cost-aware decision threshold.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def find_best_threshold_by_f1(y_true, y_proba) -> float:
    """Threshold that maximizes F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1_scores[:-1])  # last point has no matching threshold
    return float(thresholds[best_idx])


def find_best_threshold_by_cost(y_true, y_proba, cost_fp=1.0, cost_fn=100.0) -> float:
    """Threshold that minimizes expected business cost.

    A missed fraud (false negative) is assumed to cost far more than a
    flagged-but-legitimate transaction (false positive, e.g. a manual
    review). Tune cost_fp/cost_fn to your actual business numbers.

    Vectorized via a single sort + cumulative sum instead of recomputing a
    confusion matrix per candidate threshold (which is O(n_thresholds * n)
    and gets very slow once y_proba has tens of thousands of unique values).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    order = np.argsort(-y_proba)  # descending: threshold sweeps from high to low
    sorted_proba = y_proba[order]
    sorted_labels = y_true[order]

    total_positives = sorted_labels.sum()
    total_negatives = len(sorted_labels) - total_positives

    # after predicting the top-k highest-probability points as positive:
    cum_tp = np.cumsum(sorted_labels)
    cum_fp = np.cumsum(1 - sorted_labels)
    fn = total_positives - cum_tp
    fp = cum_fp
    cost = fp * cost_fp + fn * cost_fn

    best_idx = np.argmin(cost)
    return float(sorted_proba[best_idx])


def plot_confusion_matrix(y_true, y_pred, title: str, ax=None):
    ax = ax or plt.gca()
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    return ax


def plot_pr_curve(y_true, y_proba, label: str, ax=None):
    ax = ax or plt.gca()
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    return ax


def plot_roc_curve(y_true, y_proba, label: str, ax=None):
    ax = ax or plt.gca()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend()
    return ax


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)


def metrics_to_dataframe(results: dict) -> pd.DataFrame:
    """results: {model_name: metrics_dict}"""
    return pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
