"""Baseline methods for PlantOps AI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plantops_ai.config import (
    LOAD_ALERT_PCT,
    TEMPERATURE_ALERT_C,
    VIBRATION_ALERT_MM_S,
)


def prevalence_baseline_probability(train_target: pd.Series) -> float:
    """Return training-set positive prevalence as a constant probability."""
    return float(train_target.mean())


def prevalence_baseline_predictions(
    train_target: pd.Series,
    n_rows: int,
) -> np.ndarray:
    """Return constant probabilities based only on training prevalence."""
    probability = prevalence_baseline_probability(train_target)
    return np.full(n_rows, probability, dtype=float)


def operational_rule_alert(df: pd.DataFrame) -> pd.Series:
    """Return deterministic prototype operational alerts."""
    return (
        (df["temperature_c"] >= TEMPERATURE_ALERT_C)
        | (df["vibration_mm_s"] >= VIBRATION_ALERT_MM_S)
        | (df["load_pct"] >= LOAD_ALERT_PCT)
    ).astype(int)
