# PlantOps AI — Model Card

## Model

**Version:** plantops-lr-v1

**Model family:** Logistic Regression

**Purpose:** Estimate an equipment observation's relative risk of a synthetic failure event within the next 7 days.

The output is an uncalibrated risk score for decision support. It is not a literal probability of equipment failure.

## Intended Use

PlantOps AI demonstrates how operational measurements, deterministic rules, machine learning, monitoring, APIs, and dashboards can support predictive-maintenance workflows.

Intended workflow:

Operational observation → risk assessment → engineer investigation → engineering decision.

The model must not independently:

- stop equipment;
- prescribe maintenance;
- make safety-critical decisions;
- replace engineering judgement.

## Data

The prototype uses reproducible synthetic data.

Current dataset:

- 3,000 observations
- 100 equipment units
- 30 daily observations per equipment unit
- 5 equipment types
- 44 synthetic positive failure labels
- overall failure prevalence approximately 1.47%

Equipment types:

- PUMP
- MOTOR
- CONVEYOR
- PRESS
- BOILER_FEED

No Tradewinds Plantation Berhad operational data was used.

## Target

`failure_next_7d`

Binary synthetic indicator representing whether the observation is associated with a failure event within the prototype's 7-day prediction horizon.

## Features

Numeric:

- temperature_c
- vibration_mm_s
- pressure_bar
- motor_current_a
- runtime_hours
- days_since_maintenance
- load_pct

Categorical:

- equipment_type

Equipment ID and timestamp are not model features.

## Leakage Control

Equipment is separated across train, validation, and test partitions.

Current split:

- Train: 1,800 rows / 60 equipment
- Validation: 600 rows / 20 equipment
- Test: 600 rows / 20 equipment

There is no equipment overlap between these partitions.

The final test partition was not used for model or threshold selection.

## Model Selection

Two ML candidates were compared on validation data:

- Logistic Regression
- HistGradientBoostingClassifier

Logistic Regression was selected because it produced stronger validation ranking performance and offered straightforward interpretability for this prototype.

The model uses class weighting because positive failure examples are rare.

## Operating Threshold

HIGH-risk / ML alert threshold:

`risk_score >= 0.60`

The threshold was selected using validation data with a prototype decision rule:

1. require validation recall of at least 0.50;
2. among eligible thresholds, minimise false positives;
3. prefer higher precision;
4. then prefer the higher threshold.

The final test set was not used to alter this threshold.

Risk bands:

- LOW: score < 0.30
- MEDIUM: 0.30 <= score < 0.60
- HIGH: score >= 0.60

These thresholds are prototype decision conventions, not industrial safety limits.

## Final Locked Test Results

Final test population:

- 600 observations
- 20 unseen equipment units
- 10 positive synthetic failure labels
- prevalence: 1.67%

Metrics:

- ROC-AUC: 0.7473
- PR-AUC: 0.0523
- Precision: 0.0364
- Recall: 0.4000
- F1: 0.0667
- True positives: 4
- False positives: 106
- False negatives: 6
- True negatives: 484

## Interpretation

The model demonstrates ranking signal on the synthetic holdout data, but the operating threshold produces many false positives.

This is not production-ready predictive-maintenance performance.

The model is best presented as a prototype showing the architecture, engineering workflow, evaluation discipline, and decision-support concept.

## Calibration

The class-balanced Logistic Regression output is not calibrated.

On the final test set:

- observed positive prevalence: approximately 0.0167
- mean model score: approximately 0.4154

Therefore the score must be described as a **risk score**, not as a literal probability of failure.

Calibration would require more labelled failure events and an appropriately separated calibration dataset or group-safe calibration process.

## Explainability

The strongest fitted Logistic Regression coefficients include:

- days_since_maintenance
- load_pct
- vibration_mm_s
- pressure_bar
- runtime_hours

Coefficients describe associations learned by this fitted synthetic model. They are not causal effects.

## Deterministic Rules

The prototype also contains rule-based alerts for known operational thresholds:

- temperature >= 90 C
- vibration >= 6 mm/s
- load >= 95%

These are prototype assumptions and are not claimed to be TPB or universal equipment limits.

Rules and ML serve different purposes:

- deterministic rules capture known threshold conditions;
- ML captures multivariate risk patterns.

## Monitoring

Proposed production monitoring includes:

- data quality;
- schema and category changes;
- feature drift;
- risk-score distribution;
- risk-band distribution;
- ML alert rate;
- rule alert rate;
- combined alert rate;
- labelled predictive performance when outcomes become available.

PSI is included as a prototype feature-drift diagnostic.

Drift should trigger investigation, not automatic retraining.

## Limitations

- Synthetic data only.
- Very small number of positive examples.
- Uncalibrated scores.
- High false-positive burden at the frozen operating threshold.
- No real sensor integration.
- No maintenance-management-system integration.
- No live feedback loop.
- No automatic retraining.
- No safety certification.
- No production security controls.

## Governance

Any production implementation should require:

- approved model versions;
- documented data and feature contracts;
- controlled threshold changes;
- audit logging;
- access controls;
- monitoring;
- engineer feedback;
- candidate-versus-current model validation;
- rollback capability.

## Disclaimer

PlantOps AI is an independent, non-confidential prototype using synthetic data. It is not affiliated with or deployed by Tradewinds Plantation Berhad.
