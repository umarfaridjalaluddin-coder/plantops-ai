"""Feature preparation and leakage-safe dataset splitting for PlantOps AI."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from plantops_ai.config import RANDOM_SEED, TARGET_COLUMN

NUMERIC_FEATURES = (
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "motor_current_a",
    "runtime_hours",
    "days_since_maintenance",
    "load_pct",
)

CATEGORICAL_FEATURES = ("equipment_type",)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and binary target without identifier leakage."""
    X = df[list(MODEL_FEATURES)].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    return X, y


def split_by_equipment(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data so an equipment ID cannot appear in both train and test."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_SEED,
    )

    train_index, test_index = next(
        splitter.split(
            df,
            y=df[TARGET_COLUMN],
            groups=df["equipment_id"],
        )
    )

    train_df = df.iloc[train_index].copy()
    test_df = df.iloc[test_index].copy()

    return train_df, test_df


def split_train_validation_test_by_equipment(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible 60/20/20 partitions with no equipment overlap."""
    train_df, holdout_df = split_by_equipment(
        df,
        test_size=0.40,
    )

    holdout_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )

    validation_index, test_index = next(
        holdout_splitter.split(
            holdout_df,
            y=holdout_df[TARGET_COLUMN],
            groups=holdout_df["equipment_id"],
        )
    )

    validation_df = holdout_df.iloc[validation_index].copy()
    test_df = holdout_df.iloc[test_index].copy()

    return train_df, validation_df, test_df
