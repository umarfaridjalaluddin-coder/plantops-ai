"""Tests for the PlantOps AI FastAPI service."""

from fastapi.testclient import TestClient

from plantops_ai.api import app
from plantops_ai.config import EQUIPMENT_TYPES

client = TestClient(app)


VALID_PAYLOAD = {
    "equipment_type": "PUMP",
    "temperature_c": 82.0,
    "vibration_mm_s": 4.8,
    "pressure_bar": 6.5,
    "motor_current_a": 42.0,
    "runtime_hours": 5200.0,
    "days_since_maintenance": 70,
    "load_pct": 84.0,
}


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["metadata_available"] is True
    assert body["model_version"] == "plantops-lr-v1"


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "plantops-lr-v1"
    assert body["operating_threshold"] == 0.6
    assert body["medium_risk_threshold"] == 0.3
    assert body["score_interpretation"] == "uncalibrated_risk_score"
    assert body["decision_support_only"] is True
    assert body["synthetic_data"] is True


def test_equipment_endpoint() -> None:
    response = client.get("/equipment")

    assert response.status_code == 200

    body = response.json()

    assert body["synthetic_data"] is True
    assert body["count"] == 100
    assert len(body["equipment"]) == 100


def test_predict_endpoint() -> None:
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "plantops-lr-v1"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["ml_alert"], bool)
    assert isinstance(body["rule_alert"], bool)
    assert body["decision_support_only"] is True
    assert body["score_interpretation"] == "uncalibrated_risk_score"


def test_predict_rejects_invalid_numeric_input() -> None:
    payload = {
        **VALID_PAYLOAD,
        "load_pct": 120.0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_predict_rejects_unknown_equipment_type() -> None:
    payload = {
        **VALID_PAYLOAD,
        "equipment_type": "TRACTOR",
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    body = response.json()

    assert body["detail"]["message"] == "Unsupported equipment_type."
    assert set(body["detail"]["allowed_values"]) == set(EQUIPMENT_TYPES)


def test_high_operational_values_trigger_rule_alert() -> None:
    payload = {
        **VALID_PAYLOAD,
        "temperature_c": 95.0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["rule_alert"] is True
