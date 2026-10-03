"""Operational risk scoring utilities for PlantOps AI."""

from __future__ import annotations

import numpy as np

from plantops_ai.config import (
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)


def risk_band(probability: float) -> str:
    """Convert a model risk score into a prototype decision-support band."""
    if probability >= HIGH_RISK_THRESHOLD:
        return "HIGH"

    if probability >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"

    return "LOW"


def risk_bands(probabilities) -> np.ndarray:
    """Convert multiple model risk scores into risk bands."""
    return np.asarray(
        [risk_band(float(probability)) for probability in probabilities]
    )
