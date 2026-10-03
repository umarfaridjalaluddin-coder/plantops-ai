"""Tests for the PlantOps AI Power BI export."""

from plantops_ai.powerbi_export import build_powerbi_dataset


def test_powerbi_dataset_shape() -> None:
    df = build_powerbi_dataset()

    assert len(df) == 3000
    assert df["equipment_id"].nunique() == 100


def test_powerbi_dataset_contains_decision_support_columns() -> None:
    df = build_powerbi_dataset()

    expected_columns = {
        "risk_score",
        "risk_band",
        "ml_alert",
        "rule_alert",
        "combined_alert",
        "model_version",
        "score_interpretation",
        "decision_support_only",
        "synthetic_data",
    }

    assert expected_columns.issubset(df.columns)


def test_powerbi_risk_scores_are_bounded() -> None:
    df = build_powerbi_dataset()

    assert df["risk_score"].between(0.0, 1.0).all()


def test_powerbi_risk_bands_are_valid() -> None:
    df = build_powerbi_dataset()

    assert set(df["risk_band"].unique()).issubset(
        {"LOW", "MEDIUM", "HIGH"}
    )


def test_powerbi_metadata_contract() -> None:
    df = build_powerbi_dataset()

    assert (df["model_version"] == "plantops-lr-v1").all()
    assert (
        df["score_interpretation"]
        == "uncalibrated_risk_score"
    ).all()
    assert df["decision_support_only"].all()
    assert df["synthetic_data"].all()
