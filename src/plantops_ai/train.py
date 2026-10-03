"""Model training utilities for PlantOps AI."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from plantops_ai.config import RANDOM_SEED
from plantops_ai.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_logistic_regression_pipeline() -> Pipeline:
    """Build the interpretable Logistic Regression baseline pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def build_hist_gradient_boosting_pipeline() -> Pipeline:
    """Build the nonlinear HistGradientBoosting candidate pipeline."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
