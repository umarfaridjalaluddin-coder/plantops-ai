"""Generate Power BI-ready decision-support data for PlantOps AI."""

from __future__ import annotations

import joblib
import pandas as pd

from plantops_ai.baseline import operational_rule_alert
from plantops_ai.config import (
    HIGH_RISK_THRESHOLD,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from plantops_ai.features import build_model_frame
from plantops_ai.run_pipeline import MODEL_VERSION
from plantops_ai.scoring import risk_bands

MODEL_PATH = MODELS_DIR / "plantops_lr_v1.joblib"
OUTPUT_PATH = PROCESSED_DATA_DIR / "plantops_powerbi.csv"


def build_powerbi_dataset() -> pd.DataFrame:
    """Build the synthetic operational decision-support dataset."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model artifact is missing. "
            "Run `python -m plantops_ai.run_pipeline` first."
        )

    input_path = RAW_DATA_DIR / "equipment_operations.csv"

    if not input_path.exists():
        raise RuntimeError(
            "Synthetic operational dataset is missing."
        )

    df = pd.read_csv(
        input_path,
        parse_dates=["timestamp"],
    )

    model = joblib.load(MODEL_PATH)

    X, _ = build_model_frame(df)

    scores = model.predict_proba(X)[:, 1]

    output = df.copy()

    output["risk_score"] = scores
    output["risk_band"] = risk_bands(scores)
    output["ml_alert"] = (
        output["risk_score"] >= HIGH_RISK_THRESHOLD
    )
    output["rule_alert"] = (
        operational_rule_alert(output).astype(bool)
    )

    output["combined_alert"] = (
        output["ml_alert"] | output["rule_alert"]
    )

    output["model_version"] = MODEL_VERSION
    output["score_interpretation"] = "uncalibrated_risk_score"
    output["decision_support_only"] = True
    output["synthetic_data"] = True

    return output


def save_powerbi_dataset(df: pd.DataFrame) -> None:
    """Persist the Power BI-ready CSV."""
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")


def main() -> None:
    """Build and save the Power BI-ready dataset."""
    df = build_powerbi_dataset()
    save_powerbi_dataset(df)


if __name__ == "__main__":
    main()
