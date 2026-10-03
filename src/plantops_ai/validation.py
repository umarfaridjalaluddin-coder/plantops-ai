"""Data validation for PlantOps AI operational datasets."""

from __future__ import annotations

import pandas as pd

from plantops_ai.config import EQUIPMENT_TYPES, TARGET_COLUMN

REQUIRED_COLUMNS = (
    "timestamp",
    "equipment_id",
    "equipment_type",
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "motor_current_a",
    "runtime_hours",
    "days_since_maintenance",
    "load_pct",
    TARGET_COLUMN,
)


def validate_operational_data(df: pd.DataFrame) -> None:
    """Validate the structural and domain contract of an operational dataset."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset must not be empty.")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns contain missing values.")

    duplicate_count = df.duplicated(["equipment_id", "timestamp"]).sum()
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate equipment/timestamp observations."
        )

    observed_types = set(df["equipment_type"].unique())
    invalid_types = sorted(observed_types - set(EQUIPMENT_TYPES))
    if invalid_types:
        raise ValueError(f"Invalid equipment types: {invalid_types}")

    target_values = set(df[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")

    numeric_ranges = {
        "temperature_c": (0.0, 150.0),
        "vibration_mm_s": (0.0, 20.0),
        "pressure_bar": (0.0, 30.0),
        "motor_current_a": (0.0, 150.0),
        "runtime_hours": (0.0, None),
        "days_since_maintenance": (0.0, None),
        "load_pct": (0.0, 100.0),
    }

    for column, (minimum, maximum) in numeric_ranges.items():
        if (df[column] < minimum).any():
            raise ValueError(f"{column} contains values below {minimum}.")

        if maximum is not None and (df[column] > maximum).any():
            raise ValueError(f"{column} contains values above {maximum}.")


def validation_summary(df: pd.DataFrame) -> dict[str, object]:
    """Return a concise validation summary."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "equipment_count": int(df["equipment_id"].nunique()),
        "equipment_types": sorted(df["equipment_type"].unique().tolist()),
        "duplicate_equipment_timestamp": int(
            df.duplicated(["equipment_id", "timestamp"]).sum()
        ),
        "missing_values": int(df.isna().sum().sum()),
        "failure_rows": int(df[TARGET_COLUMN].sum()),
        "failure_prevalence": float(df[TARGET_COLUMN].mean()),
    }
