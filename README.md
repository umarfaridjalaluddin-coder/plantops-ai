# PlantOps AI

## AI-Assisted Operational Monitoring & Predictive Maintenance

PlantOps AI is an independent, non-confidential engineering prototype demonstrating how data engineering, deterministic operational rules, machine learning, monitoring, APIs, and analytics can support equipment-monitoring and predictive-maintenance workflows.

> **Disclaimer:** This project uses entirely synthetic data. It is not affiliated with, deployed by, or based on operational data from Tradewinds Plantation Berhad.

## Business Problem

Equipment teams often need to prioritise which assets deserve investigation when operational conditions begin to change.

PlantOps AI demonstrates the following decision-support question:

> Given the operational information available for an equipment observation, what is the estimated risk that the equipment will experience a failure within the next 7 days?

The system does **not** automatically stop equipment or prescribe maintenance.

The intended workflow is:

```text
Operational Data
      |
      v
Validation
      |
      +-------------------+
      |                   |
      v                   v
Operational Rules     ML Risk Model
      |                   |
      +---------+---------+
                |
                v
        Decision Support
                |
                v
      Engineer Investigation
                |
                v
        Decision / Action
```

## What the Prototype Includes

- reproducible synthetic operational data;
- data-quality validation;
- equipment-grouped train/validation/test splitting;
- deterministic operational alerts;
- Logistic Regression risk model;
- imbalanced-class evaluation;
- threshold and risk-band selection;
- model explainability;
- calibration diagnostics;
- model versioning and metadata;
- reproducible model artifacts;
- FastAPI prediction service;
- Power BI-ready scoring export;
- data and feature-drift monitoring;
- AI lifecycle and governance documentation;
- automated Ruff and pytest checks through GitHub Actions.

## Synthetic Dataset

The reproducible dataset contains:

- 3,000 observations;
- 100 equipment units;
- 30 daily observations per unit;
- 5 equipment types;
- 44 synthetic positive failure labels;
- approximately 1.47% overall positive prevalence.

Equipment types:

- PUMP
- MOTOR
- CONVEYOR
- PRESS
- BOILER_FEED

Example operational features include temperature, vibration, pressure, motor current, runtime, maintenance age, and equipment load.

## ML Design

### Target

`failure_next_7d`

### Prediction horizon

7 days.

### Leakage control

Equipment units are separated across development partitions:

| Partition | Rows | Equipment |
|---|---:|---:|
| Train | 1,800 | 60 |
| Validation | 600 | 20 |
| Test | 600 | 20 |

There is no equipment overlap between these partitions.

### Model

Frozen model version:

`plantops-lr-v1`

The selected model is a class-weighted Logistic Regression pipeline.

A HistGradientBoosting candidate was also evaluated during model selection.

### Risk bands

- LOW: `< 0.30`
- MEDIUM: `0.30 to < 0.60`
- HIGH: `>= 0.60`

These are prototype decision thresholds, not industrial safety limits.

## Final Locked Holdout Results

The final test set contains 600 observations from 20 unseen equipment units and 10 positive synthetic labels.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.7473 |
| PR-AUC | 0.0523 |
| Precision | 0.0364 |
| Recall | 0.4000 |
| F1 | 0.0667 |
| True Positive | 4 |
| False Positive | 106 |
| False Negative | 6 |
| True Negative | 484 |

These results demonstrate ranking signal, but also a substantial false-positive burden.

The prototype should therefore be interpreted as an engineering and decision-support demonstration, **not a production-ready predictive-maintenance model**.

## Important Score Interpretation

Because the Logistic Regression model uses class weighting, its output is not calibrated to the real failure prevalence.

The output is therefore called:

**risk score**

rather than:

**failure probability**

Production calibration would require substantially more reliable labelled failure data and an appropriately separated calibration process.

## Rules + ML

PlantOps AI combines two complementary approaches.

Deterministic prototype rules identify known threshold conditions such as:

- temperature >= 90 C;
- vibration >= 6 mm/s;
- load >= 95%.

The ML model identifies multivariate patterns across several measurements.

The prototype thresholds are illustrative assumptions and are not claimed to represent TPB or universal equipment limits.

Neither mechanism automatically initiates a safety-critical action.

## API

FastAPI provides:

```text
GET  /health
GET  /model-info
GET  /equipment
POST /predict
```

Example response:

```json
{
  "model_version": "plantops-lr-v1",
  "risk_score": 0.19105406080358228,
  "risk_band": "LOW",
  "ml_alert": false,
  "rule_alert": false,
  "decision_support_only": true,
  "score_interpretation": "uncalibrated_risk_score"
}
```

## Power BI Layer

The project generates:

`data/processed/plantops_powerbi.csv`

The current reproducible export contains 3,000 observations and 20 columns.

Suggested dashboard views include:

- equipment count;
- high-risk observations;
- combined alerts;
- average risk score;
- risk-band distribution;
- equipment-type comparison;
- equipment-level operating trends.

See [`powerbi/README.md`](powerbi/README.md).

## Monitoring

The monitoring layer demonstrates:

- missing-value monitoring;
- duplicate detection;
- equipment coverage;
- feature drift using PSI;
- risk-score distribution;
- risk-band distribution;
- ML alert rate;
- rule alert rate;
- combined alert rate.

Drift is treated as an **investigation signal**, not an automatic retraining trigger.

## Reproduce the Project

### 1. Create a Python 3.12 environment

Example:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 3. Generate synthetic data

```bash
python -m plantops_ai.generate_data
```

### 4. Build the frozen model artifacts

```bash
python -m plantops_ai.run_pipeline
```

### 5. Generate the Power BI export

```bash
python -m plantops_ai.powerbi_export
```

### 6. Run quality checks

```bash
ruff check src tests
pytest -q
```

The current verified test suite contains **49 tests**.

## Run the API

After generating the data and model artifacts:

```bash
uvicorn plantops_ai.api:app --host 127.0.0.1 --port 8000
```

Swagger/OpenAPI documentation is then available through the FastAPI `/docs` endpoint.

## Reproducibility

Generated runtime artifacts are intentionally excluded from Git, including:

- synthetic CSV data;
- trained model artifacts;
- model metadata;
- metrics files;
- Power BI export data.

The pipeline regenerates them from source.

GitHub Actions follows the same sequence:

```text
Checkout
   ↓
Python 3.12
   ↓
Install dependencies
   ↓
Generate synthetic data
   ↓
Build model artifacts
   ↓
Ruff
   ↓
pytest
   ↓
Generate Power BI export
```

## Documentation

Detailed documentation is available in:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/model_card.md`](docs/model_card.md)
- [`docs/model_selection.md`](docs/model_selection.md)
- [`docs/ai_lifecycle.md`](docs/ai_lifecycle.md)
- [`docs/interview_story.md`](docs/interview_story.md)
- [`powerbi/README.md`](powerbi/README.md)

## Production Evolution

A real implementation would first validate the organisation's actual equipment, sensor availability, failure definitions, maintenance records, operating context, and engineering requirements.

Potential production components could include:

- historian / SCADA / IoT integration;
- batch or streaming ingestion;
- governed operational storage;
- orchestration;
- authenticated services;
- model registry;
- CMMS/work-order integration;
- monitoring and alerting;
- engineer feedback;
- controlled CI/CD and rollback.

Technology choices should follow the actual operational environment rather than being imposed by this prototype.

## Project Status

Core prototype complete:

- data generation and validation;
- ML development and evaluation;
- deterministic rules;
- explainability and calibration diagnostics;
- reproducible model artifacts;
- prediction API;
- Power BI-ready export;
- monitoring;
- governance documentation;
- automated CI configuration.

The remaining work is presentation/demo preparation and optional dashboard implementation.
