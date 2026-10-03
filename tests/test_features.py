"""Tests for feature preparation and leakage-safe splitting."""

from plantops_ai.config import TARGET_COLUMN
from plantops_ai.features import MODEL_FEATURES, build_model_frame, split_by_equipment
from plantops_ai.generate_data import generate_synthetic_data


def test_model_frame_excludes_identifiers_and_target() -> None:
    df = generate_synthetic_data()
    X, y = build_model_frame(df)

    assert list(X.columns) == list(MODEL_FEATURES)
    assert "equipment_id" not in X.columns
    assert "timestamp" not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == len(df)


def test_split_has_no_equipment_overlap() -> None:
    df = generate_synthetic_data()
    train_df, test_df = split_by_equipment(df)

    train_equipment = set(train_df["equipment_id"])
    test_equipment = set(test_df["equipment_id"])

    assert train_equipment.isdisjoint(test_equipment)


def test_split_preserves_all_rows() -> None:
    df = generate_synthetic_data()
    train_df, test_df = split_by_equipment(df)

    assert len(train_df) + len(test_df) == len(df)


def test_split_is_reproducible() -> None:
    df = generate_synthetic_data()

    train_a, test_a = split_by_equipment(df)
    train_b, test_b = split_by_equipment(df)

    assert train_a.index.equals(train_b.index)
    assert test_a.index.equals(test_b.index)


def test_three_way_split_has_no_equipment_overlap() -> None:
    from plantops_ai.features import split_train_validation_test_by_equipment

    df = generate_synthetic_data()
    train_df, validation_df, test_df = split_train_validation_test_by_equipment(df)

    train_ids = set(train_df["equipment_id"])
    validation_ids = set(validation_df["equipment_id"])
    test_ids = set(test_df["equipment_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_three_way_split_preserves_all_rows() -> None:
    from plantops_ai.features import split_train_validation_test_by_equipment

    df = generate_synthetic_data()
    train_df, validation_df, test_df = split_train_validation_test_by_equipment(df)

    assert len(train_df) + len(validation_df) + len(test_df) == len(df)
    assert len(train_df) == 1800
    assert len(validation_df) == 600
    assert len(test_df) == 600


def test_three_way_split_is_reproducible() -> None:
    from plantops_ai.features import split_train_validation_test_by_equipment

    df = generate_synthetic_data()

    first = split_train_validation_test_by_equipment(df)
    second = split_train_validation_test_by_equipment(df)

    for first_df, second_df in zip(first, second, strict=True):
        assert first_df.index.equals(second_df.index)
