# PlantOps AI — Architecture

## Objective

PlantOps AI demonstrates an end-to-end pattern for AI-assisted operational monitoring and predictive-maintenance decision support.

The prototype deliberately separates:

- data engineering;
- deterministic monitoring rules;
- machine-learning risk scoring;
- API/dashboard consumption;
- monitoring and governance;
- human engineering decisions.

## Logical Architecture

```text
Synthetic Operational Data
          |
          v
+-------------------------+
| Data Validation         |
| schema / ranges / nulls |
| duplicates / categories |
+-------------------------+
          |
          v
+-------------------------+
| Feature Preparation     |
| numeric + equipment type|
+-------------------------+
          |
          +----------------------+
          |                      |
          v                      v
+------------------+    +------------------+
| Deterministic    |    | ML Risk Model    |
| Operational Rules|    | Logistic Reg.    |
+------------------+    +------------------+
          |                      |
          +-----------+----------+
                      |
                      v
             +------------------+
             | Decision Support |
             | score / band /   |
             | combined alerts  |
             +------------------+
                |            |
                v            v
          +----------+  +----------+
          | FastAPI  |  | Power BI |
          +----------+  +----------+
                \            /
                 \          /
                  v        v
              +--------------+
              | Engineer     |
              | Investigation|
              +--------------+
                     |
                     v
              Decision / Action
                     |
                     v
              Recorded Outcome
                     |
                     v
          +----------------------+
          | Monitoring &         |
          | Feedback Lifecycle   |
          +----------------------+
```

## Data Layer

The current prototype generates synthetic daily equipment observations.

Inputs include:

- temperature;
- vibration;
- pressure;
- motor current;
- runtime;
- maintenance age;
- equipment load;
- equipment type.

The synthetic dataset is reproducible using a fixed random seed.

## Validation Layer

Before modelling, PlantOps AI validates:

- required columns;
- missing values;
- equipment/timestamp uniqueness;
- equipment categories;
- target values;
- numeric ranges.

In a production implementation this layer would also detect schema evolution, sensor-quality problems, late data, and source-system failures.

## Feature Layer

Identifiers are separated from model features.

Equipment-based train/validation/test splitting prevents observations from the same equipment unit appearing across model-development partitions.

This reduces leakage and gives a more realistic test of generalisation to unseen equipment.

## Rules Layer

Deterministic rules represent known operational conditions.

Prototype examples:

- high temperature;
- high vibration;
- high load.

Rules remain explicit and auditable.

They should be configured using approved engineering limits in a real implementation.

## Machine-Learning Layer

The frozen prototype model is:

`plantops-lr-v1`

Model:

Logistic Regression with preprocessing and class weighting.

Output:

Uncalibrated risk score.

The ML model complements rather than replaces deterministic rules.

## Decision-Support Layer

Each scored observation can expose:

- risk score;
- LOW/MEDIUM/HIGH risk band;
- ML alert;
- rule alert;
- combined alert;
- model version;
- score interpretation.

The output prioritises investigation.

It does not automatically trigger a safety shutdown.

## API Layer

FastAPI exposes:

- `GET /health`
- `GET /model-info`
- `GET /equipment`
- `POST /predict`

OpenAPI/Swagger documentation is generated automatically.

## Analytics Layer

A Power BI-ready export combines operational measurements with decision-support outputs.

Suggested dashboard views include:

- equipment count;
- high-risk observations;
- combined alerts;
- average risk score;
- risk distribution;
- equipment-type comparison;
- equipment-level investigation trends.

All dashboard data is synthetic.

## Monitoring Layer

The prototype monitoring layer includes:

- data-quality summaries;
- feature PSI;
- risk distribution;
- ML alert rate;
- rule alert rate;
- combined alert rate.

In production, labelled outcomes would additionally support continuous precision, recall, PR-AUC, false-positive, and false-negative monitoring.

## Human-in-the-Loop Control

The intended workflow is:

```text
Alert
  |
  v
Engineer reviews operating context
  |
  +--> sensor/data issue -> correct data/instrumentation
  |
  +--> genuine abnormal condition -> inspect equipment
  |
  +--> acceptable operating condition -> document feedback
  |
  v
Engineering decision
  |
  v
Action and outcome recorded
```

This feedback can later improve rules, data quality, thresholds, and model development.

## Production Evolution

A production implementation could replace the synthetic/file-based components with:

- historian or SCADA/IoT sources;
- streaming or scheduled ingestion;
- governed cloud/on-premise storage;
- orchestration;
- model registry;
- authenticated API services;
- CMMS/work-order integration;
- monitoring and alerting infrastructure;
- controlled CI/CD.

Technology choices should depend on the organisation's actual environment rather than being imposed by this prototype.

## Failure and Recovery Considerations

A production design should define behaviour when:

- source data stops arriving;
- sensors become unreliable;
- validation fails;
- the model artifact is unavailable;
- the API is unavailable;
- monitoring detects abnormal distributions.

For safety-critical conditions, approved deterministic controls and engineering procedures must not depend solely on an ML service.

## Security and Governance

Production controls should include:

- authentication;
- authorisation;
- encryption;
- secrets management;
- network controls;
- audit logging;
- artifact integrity;
- controlled deployments;
- model/version traceability;
- rollback.

## Disclaimer

This architecture is an independent prototype using synthetic data. It does not represent the current architecture, systems, equipment, or operational practices of Tradewinds Plantation Berhad.
