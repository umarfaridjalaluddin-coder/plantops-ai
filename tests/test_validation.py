"""Tests for PlantOps AI data validation."""

import pytest

from plantops_ai.generate_data import generate_synthetic_data
from plantops_ai.validation import validate_operational_data


def test_valid_generated_data_passes_validation() -> None:
    df = generate_synthetic_data()

    validate_operational_data(df)


def test_missing_required_column_fails() -> None:
    df = generate_synthetic_data().drop(columns=["temperature_c"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_operational_data(df)


def test_duplicate_observation_fails() -> None:
    df = generate_synthetic_data()
    duplicate = df.iloc[[0]]
    df = __import__("pandas").concat([df, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate equipment/timestamp"):
        validate_operational_data(df)


def test_invalid_load_fails() -> None:
    df = generate_synthetic_data()
    df.loc[0, "load_pct"] = 101.0

    with pytest.raises(ValueError, match="load_pct"):
        validate_operational_data(df)


def test_invalid_target_fails() -> None:
    df = generate_synthetic_data()
    df.loc[0, "failure_next_7d"] = 2

    with pytest.raises(ValueError, match="must contain only 0 and 1"):
        validate_operational_data(df)
