"""Central configuration for PlantOps AI."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Business / ML problem
# ---------------------------------------------------------------------------

TARGET_COLUMN = "failure_next_7d"
PREDICTION_HORIZON_DAYS = 7

EQUIPMENT_TYPES = (
    "PUMP",
    "MOTOR",
    "CONVEYOR",
    "PRESS",
    "BOILER_FEED",
)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

N_EQUIPMENT = 100
N_DAYS = 30


# ---------------------------------------------------------------------------
# Operational rule thresholds
#
# These are prototype/demo thresholds for synthetic data.
# They are NOT claimed to be real industrial operating limits.
# ---------------------------------------------------------------------------

TEMPERATURE_ALERT_C = 90.0
VIBRATION_ALERT_MM_S = 6.0
LOAD_ALERT_PCT = 95.0


# ---------------------------------------------------------------------------
# Model risk bands
#
# Prototype decision-support bands. They must not be interpreted as
# validated maintenance or safety thresholds.
# ---------------------------------------------------------------------------

MEDIUM_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60
