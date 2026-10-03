"""Tests for PlantOps AI monitoring utilities."""

import pandas as pd

from plantops_ai.generate_data import generate_synthetic_data
from plantops_ai.monitoring import (
    data_quality_summary,
    drift_band,
    feature_drift_report,
    population_stability_index,
    scoring_monitoring_summary,
)
from plantops_ai.powerbi_export import build_powerbi_dataset


def test_identical_population_has_zero_psi() -> None:
    values = pd.Series(
        range(1, 101),
        dtype=float,
    )

    psi = population_stability_index(
        values,
        values,
    )

    assert abs(psi) < 1e-12


def test_shifted_population_has_positive_psi() -> None:
    reference = pd.Series(
        range(1, 101),
        dtype=float,
    )
    current = reference + 50.0

    psi = population_stability_index(
        reference,
        current,
    )

    assert psi > 0.0


def test_drift_band_boundaries() -> None:
    assert drift_band(0.05) == "LOW"
    assert drift_band(0.10) == "WATCH"
    assert drift_band(0.249) == "WATCH"
    assert drift_band(0.25) == "HIGH"


def test_feature_drift_report_contains_numeric_features() -> None:
    df = generate_synthetic_data()

    reference = df.iloc[:1500]
    current = df.iloc[1500:]

    report = feature_drift_report(
        reference,
        current,
    )

    assert len(report) == 7
    assert set(report.columns) == {
        "feature",
        "psi",
        "drift_band",
    }


def test_data_quality_summary_for_generated_data() -> None:
    df = generate_synthetic_data()

    summary = data_quality_summary(df)

    assert summary["rows"] == 3000
    assert summary["missing_values"] == 0
    assert summary["duplicate_equipment_timestamp"] == 0
    assert summary["equipment_count"] == 100
    assert summary["target_available"] is True


def test_scoring_monitoring_summary() -> None:
    df = build_powerbi_dataset()

    summary = scoring_monitoring_summary(df)

    assert summary["rows"] == 3000
    assert 0.0 <= summary["mean_risk_score"] <= 1.0
    assert (
        summary["low_risk"]
        + summary["medium_risk"]
        + summary["high_risk"]
        == 3000
    )
    assert 0.0 <= summary["ml_alert_rate"] <= 1.0
    assert 0.0 <= summary["rule_alert_rate"] <= 1.0
    assert 0.0 <= summary["combined_alert_rate"] <= 1.0
