"""Model evaluation utilities for PlantOps AI."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true,
    probabilities,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate binary probabilities at a supplied decision threshold."""
    y_true_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)

    predictions = (probability_array >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true_array, probability_array)),
        "pr_auc": float(
            average_precision_score(y_true_array, probability_array)
        ),
        "precision": float(
            precision_score(y_true_array, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_array, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def threshold_table(
    y_true,
    probabilities,
    thresholds,
) -> list[dict[str, float | int]]:
    """Evaluate a sequence of decision thresholds."""
    return [
        evaluate_binary_classifier(
            y_true,
            probabilities,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]
