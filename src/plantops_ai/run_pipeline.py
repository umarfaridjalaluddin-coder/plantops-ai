"""Train and persist the frozen PlantOps AI prototype model."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from plantops_ai.config import (
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    MODELS_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from plantops_ai.evaluate import evaluate_binary_classifier
from plantops_ai.features import (
    MODEL_FEATURES,
    build_model_frame,
    split_train_validation_test_by_equipment,
)
from plantops_ai.train import build_logistic_regression_pipeline

MODEL_VERSION = "plantops-lr-v1"


def train_frozen_model() -> tuple[object, dict[str, object]]:
    """Train the frozen model and return it with reproducibility metadata."""
    df = pd.read_csv(RAW_DATA_DIR / "equipment_operations.csv")

    train_df, validation_df, test_df = (
        split_train_validation_test_by_equipment(df)
    )

    X_train, y_train = build_model_frame(train_df)

    model = build_logistic_regression_pipeline()
    model.fit(X_train, y_train)

    metadata: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "model_type": "LogisticRegression",
        "target": TARGET_COLUMN,
        "model_features": list(MODEL_FEATURES),
        "operating_threshold": HIGH_RISK_THRESHOLD,
        "medium_risk_threshold": MEDIUM_RISK_THRESHOLD,
        "score_interpretation": "uncalibrated_risk_score",
        "decision_support_only": True,
        "synthetic_data": True,
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "training_equipment": int(train_df["equipment_id"].nunique()),
        "validation_equipment": int(validation_df["equipment_id"].nunique()),
        "test_equipment": int(test_df["equipment_id"].nunique()),
        "training_failures": int(train_df[TARGET_COLUMN].sum()),
        "validation_failures": int(validation_df[TARGET_COLUMN].sum()),
        "test_failures": int(test_df[TARGET_COLUMN].sum()),
        "sklearn_version": sklearn.__version__,
    }

    return model, metadata


def persist_model(
    model,
    metadata: dict[str, object],
) -> None:
    """Persist the model artifact and metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "plantops_lr_v1.joblib"
    metadata_path = MODELS_DIR / "plantops_lr_v1_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        **metadata,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")


def evaluate_frozen_model(model) -> dict[str, float | int]:
    """Reproduce the documented frozen holdout evaluation."""
    df = pd.read_csv(RAW_DATA_DIR / "equipment_operations.csv")

    _, _, test_df = split_train_validation_test_by_equipment(df)

    X_test, y_test = build_model_frame(test_df)
    scores = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_binary_classifier(
        y_test,
        scores,
        threshold=HIGH_RISK_THRESHOLD,
    )

    metrics["roc_auc"] = float(
        roc_auc_score(y_test, scores)
    )
    metrics["pr_auc"] = float(
        average_precision_score(y_test, scores)
    )

    return metrics


def save_metrics(metrics: dict[str, float | int]) -> None:
    """Persist frozen evaluation metrics."""
    metrics_dir = REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    output_path = metrics_dir / "final_test_metrics.json"

    output_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved metrics: {output_path}")


def main() -> None:
    """Run the frozen training and persistence workflow."""
    model, metadata = train_frozen_model()

    persist_model(model, metadata)

    metrics = evaluate_frozen_model(model)
    save_metrics(metrics)

    print()
    print("--- FROZEN MODEL ---")
    print("Version:", metadata["model_version"])
    print("Score interpretation:", metadata["score_interpretation"])
    print("Decision support only:", metadata["decision_support_only"])

    print()
    print("--- REPRODUCED FINAL TEST METRICS ---")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
