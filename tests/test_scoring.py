"""Tests for PlantOps AI model evaluation and risk scoring."""

import numpy as np

from plantops_ai.evaluate import evaluate_binary_classifier, threshold_table
from plantops_ai.scoring import risk_band, risk_bands


def test_binary_evaluation_returns_expected_confusion_matrix() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.8, 0.2])

    metrics = evaluate_binary_classifier(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1


def test_binary_evaluation_metrics_are_bounded() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.6, 0.9])

    metrics = evaluate_binary_classifier(y_true, probabilities)

    for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        assert 0.0 <= metrics[metric] <= 1.0


def test_threshold_table_returns_one_result_per_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    thresholds = [0.3, 0.5, 0.7]

    results = threshold_table(
        y_true,
        probabilities,
        thresholds,
    )

    assert len(results) == 3
    assert [result["threshold"] for result in results] == thresholds


def test_risk_band_boundaries() -> None:
    assert risk_band(0.00) == "LOW"
    assert risk_band(0.29) == "LOW"
    assert risk_band(0.30) == "MEDIUM"
    assert risk_band(0.59) == "MEDIUM"
    assert risk_band(0.60) == "HIGH"
    assert risk_band(1.00) == "HIGH"


def test_risk_bands_returns_one_band_per_score() -> None:
    scores = np.array([0.1, 0.4, 0.8])

    bands = risk_bands(scores)

    assert bands.tolist() == ["LOW", "MEDIUM", "HIGH"]
