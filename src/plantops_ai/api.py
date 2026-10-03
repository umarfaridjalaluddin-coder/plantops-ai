"""FastAPI service for PlantOps AI decision support."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from plantops_ai.baseline import operational_rule_alert
from plantops_ai.config import (
    EQUIPMENT_TYPES,
    HIGH_RISK_THRESHOLD,
    MODELS_DIR,
    RAW_DATA_DIR,
)
from plantops_ai.run_pipeline import MODEL_VERSION
from plantops_ai.scoring import risk_band

MODEL_PATH = MODELS_DIR / "plantops_lr_v1.joblib"
METADATA_PATH = MODELS_DIR / "plantops_lr_v1_metadata.json"

app = FastAPI(
    title="PlantOps AI",
    version="0.1.0",
    description=(
        "Synthetic prototype API for AI-assisted operational monitoring "
        "and predictive-maintenance decision support."
    ),
)


class PredictionRequest(BaseModel):
    """Validated operational inputs for one equipment observation."""

    equipment_type: str
    temperature_c: Annotated[float, Field(ge=0.0, le=150.0)]
    vibration_mm_s: Annotated[float, Field(ge=0.0, le=20.0)]
    pressure_bar: Annotated[float, Field(ge=0.0, le=30.0)]
    motor_current_a: Annotated[float, Field(ge=0.0, le=150.0)]
    runtime_hours: Annotated[float, Field(ge=0.0)]
    days_since_maintenance: Annotated[int, Field(ge=0)]
    load_pct: Annotated[float, Field(ge=0.0, le=100.0)]


class PredictionResponse(BaseModel):
    """Decision-support output for one equipment observation."""

    model_version: str
    risk_score: float
    risk_band: str
    ml_alert: bool
    rule_alert: bool
    decision_support_only: bool
    score_interpretation: str


@lru_cache(maxsize=1)
def load_model():
    """Load the persisted frozen model."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model artifact is missing. "
            "Run `python -m plantops_ai.run_pipeline` first."
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, object]:
    """Load persisted model metadata."""
    if not METADATA_PATH.exists():
        raise RuntimeError(
            "Model metadata is missing. "
            "Run `python -m plantops_ai.run_pipeline` first."
        )

    return json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )


@app.get("/health")
def health() -> dict[str, object]:
    """Return service and model-artifact health."""
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "metadata_available": METADATA_PATH.exists(),
        "model_version": MODEL_VERSION,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Return safe model metadata for API consumers."""
    metadata = load_metadata()

    return {
        "model_version": metadata["model_version"],
        "model_type": metadata["model_type"],
        "target": metadata["target"],
        "operating_threshold": metadata["operating_threshold"],
        "medium_risk_threshold": metadata["medium_risk_threshold"],
        "score_interpretation": metadata["score_interpretation"],
        "decision_support_only": metadata["decision_support_only"],
        "synthetic_data": metadata["synthetic_data"],
        "model_features": metadata["model_features"],
    }


@app.get("/equipment")
def equipment() -> dict[str, object]:
    """Return equipment IDs available in the synthetic prototype dataset."""
    data_path = RAW_DATA_DIR / "equipment_operations.csv"

    if not data_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Synthetic operational dataset is unavailable.",
        )

    df = pd.read_csv(
        data_path,
        usecols=["equipment_id", "equipment_type"],
    )

    equipment_records = (
        df.drop_duplicates()
        .sort_values("equipment_id")
        .to_dict(orient="records")
    )

    return {
        "synthetic_data": True,
        "count": len(equipment_records),
        "equipment": equipment_records,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return ML and deterministic-rule decision support."""
    if request.equipment_type not in EQUIPMENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unsupported equipment_type.",
                "allowed_values": list(EQUIPMENT_TYPES),
            },
        )

    model = load_model()
    metadata = load_metadata()

    row = pd.DataFrame(
        [
            {
                "temperature_c": request.temperature_c,
                "vibration_mm_s": request.vibration_mm_s,
                "pressure_bar": request.pressure_bar,
                "motor_current_a": request.motor_current_a,
                "runtime_hours": request.runtime_hours,
                "days_since_maintenance": request.days_since_maintenance,
                "load_pct": request.load_pct,
                "equipment_type": request.equipment_type,
            }
        ]
    )

    score = float(model.predict_proba(row)[:, 1][0])

    rule_input = row[
        [
            "temperature_c",
            "vibration_mm_s",
            "load_pct",
        ]
    ]

    rule_alert = bool(
        operational_rule_alert(rule_input).iloc[0]
    )

    return PredictionResponse(
        model_version=str(metadata["model_version"]),
        risk_score=score,
        risk_band=risk_band(score),
        ml_alert=score >= HIGH_RISK_THRESHOLD,
        rule_alert=rule_alert,
        decision_support_only=True,
        score_interpretation="uncalibrated_risk_score",
    )
