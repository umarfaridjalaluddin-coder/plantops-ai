"""Tests for PlantOps AI baseline and model-training methods."""

import numpy as np

from plantops_ai.baseline import (
    operational_rule_alert,
    prevalence_baseline_predictions,
    prevalence_baseline_probability,
)
from plantops_ai.features import build_model_frame, split_by_equipment
from plantops_ai.generate_data import generate_synthetic_data
from plantops_ai.train import build_logistic_regression_pipeline


def test_prevalence_baseline_uses_training_target() -> None:
    df = generate_synthetic_data()

    probability = prevalence_baseline_probability(df["failure_next_7d"])

    assert probability == df["failure_next_7d"].mean()


def test_prevalence_predictions_are_constant() -> None:
    df = generate_synthetic_data()

    predictions = prevalence_baseline_predictions(
        df["failure_next_7d"],
        n_rows=10,
    )

    assert len(predictions) == 10
    assert np.unique(predictions).size == 1


def test_operational_rule_returns_binary_values() -> None:
    df = generate_synthetic_data()

    alerts = operational_rule_alert(df)

    assert set(alerts.unique()).issubset({0, 1})
    assert len(alerts) == len(df)


def test_logistic_pipeline_trains_and_predicts_probabilities() -> None:
    df = generate_synthetic_data()
    train_df, test_df = split_by_equipment(df)

    X_train, y_train = build_model_frame(train_df)
    X_test, _ = build_model_frame(test_df)

    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    assert len(probabilities) == len(test_df)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_hist_gradient_boosting_trains_and_predicts_probabilities() -> None:
    from sklearn.utils.class_weight import compute_sample_weight

    from plantops_ai.features import split_train_validation_test_by_equipment
    from plantops_ai.train import build_hist_gradient_boosting_pipeline

    df = generate_synthetic_data()
    train_df, validation_df, _ = split_train_validation_test_by_equipment(df)

    X_train, y_train = build_model_frame(train_df)
    X_validation, _ = build_model_frame(validation_df)

    pipeline = build_hist_gradient_boosting_pipeline()

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    pipeline.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weight,
    )

    probabilities = pipeline.predict_proba(X_validation)[:, 1]

    assert len(probabilities) == len(validation_df)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
