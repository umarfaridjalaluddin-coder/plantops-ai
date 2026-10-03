"""Monitoring utilities for PlantOps AI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plantops_ai.config import TARGET_COLUMN
from plantops_ai.features import NUMERIC_FEATURES


def population_stability_index(
    reference,
    current,
    n_bins: int = 10,
) -> float:
    """Calculate PSI using reference quantile boundaries."""
    reference_array = np.asarray(reference, dtype=float)
    current_array = np.asarray(current, dtype=float)

    if reference_array.size == 0 or current_array.size == 0:
        raise ValueError("Reference and current samples must not be empty.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)

    boundaries = np.unique(
        np.quantile(
            reference_array,
            quantiles,
        )
    )

    if len(boundaries) < 2:
        return 0.0

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference_array,
        bins=boundaries,
    )
    current_counts, _ = np.histogram(
        current_array,
        bins=boundaries,
    )

    reference_pct = reference_counts / reference_counts.sum()
    current_pct = current_counts / current_counts.sum()

    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None,
    )
    current_pct = np.clip(
        current_pct,
        epsilon,
        None,
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return PSI values for numeric model features."""
    rows = []

    for feature in NUMERIC_FEATURES:
        psi = population_stability_index(
            reference_df[feature],
            current_df[feature],
        )

        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "drift_band": drift_band(psi),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )


def drift_band(psi: float) -> str:
    """Convert PSI into a conventional monitoring band."""
    if psi >= 0.25:
        return "HIGH"

    if psi >= 0.10:
        return "WATCH"

    return "LOW"


def data_quality_summary(df: pd.DataFrame) -> dict[str, object]:
    """Return operational data-quality monitoring statistics."""
    return {
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_equipment_timestamp": int(
            df.duplicated(
                ["equipment_id", "timestamp"]
            ).sum()
        ),
        "equipment_count": int(
            df["equipment_id"].nunique()
        ),
        "target_available": TARGET_COLUMN in df.columns,
    }


def scoring_monitoring_summary(
    scored_df: pd.DataFrame,
) -> dict[str, object]:
    """Return monitoring statistics for scored operational data."""
    required_columns = {
        "risk_score",
        "risk_band",
        "ml_alert",
        "rule_alert",
        "combined_alert",
    }

    missing = sorted(
        required_columns - set(scored_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing scoring columns: {missing}"
        )

    risk_counts = (
        scored_df["risk_band"]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH"],
            fill_value=0,
        )
    )

    return {
        "rows": len(scored_df),
        "mean_risk_score": float(
            scored_df["risk_score"].mean()
        ),
        "low_risk": int(risk_counts["LOW"]),
        "medium_risk": int(risk_counts["MEDIUM"]),
        "high_risk": int(risk_counts["HIGH"]),
        "ml_alert_rate": float(
            scored_df["ml_alert"].mean()
        ),
        "rule_alert_rate": float(
            scored_df["rule_alert"].mean()
        ),
        "combined_alert_rate": float(
            scored_df["combined_alert"].mean()
        ),
    }
