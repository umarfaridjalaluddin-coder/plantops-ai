"""Tests for the PlantOps AI synthetic data generator."""

from plantops_ai.config import EQUIPMENT_TYPES, N_DAYS, N_EQUIPMENT, TARGET_COLUMN
from plantops_ai.generate_data import generate_synthetic_data


def test_generated_dataset_shape() -> None:
    df = generate_synthetic_data()

    assert len(df) == N_EQUIPMENT * N_DAYS
    assert df["equipment_id"].nunique() == N_EQUIPMENT


def test_generated_equipment_types() -> None:
    df = generate_synthetic_data()

    assert set(df["equipment_type"].unique()) == set(EQUIPMENT_TYPES)


def test_generated_data_has_no_missing_values() -> None:
    df = generate_synthetic_data()

    assert not df.isna().any().any()


def test_equipment_timestamp_is_unique() -> None:
    df = generate_synthetic_data()

    assert not df.duplicated(["equipment_id", "timestamp"]).any()


def test_target_is_binary_and_contains_both_classes() -> None:
    df = generate_synthetic_data()

    assert set(df[TARGET_COLUMN].unique()) == {0, 1}


def test_generation_is_reproducible() -> None:
    first = generate_synthetic_data()
    second = generate_synthetic_data()

    assert first.equals(second)
