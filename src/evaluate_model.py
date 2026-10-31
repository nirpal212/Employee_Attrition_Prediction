"""
Compute classification metrics from true labels and model outputs.

Metrics used in this project:
    Accuracy  — overall share of correct predictions (can look high on imbalanced data).
    Precision — of employees flagged as leaving, how many actually left.
    Recall    — of employees who left, how many the model found (important in attrition).
    F1-score  — harmonic mean of precision and recall.
    ROC-AUC   — ranking quality using predicted probabilities, when available.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Return a JSON-serializable metrics dictionary."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 2:
            # Probability of the positive class (column 1 when classes_ are [0, 1]).
            positive_scores = y_proba[:, 1]
        else:
            positive_scores = y_proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def confusion_matrix_values(y_true, y_pred) -> dict:
    """Return TN, FP, FN, TP for plotting and the report."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
