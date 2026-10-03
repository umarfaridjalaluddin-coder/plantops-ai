# PlantOps AI — AI Lifecycle and Monitoring

## Purpose

PlantOps AI is a synthetic predictive-maintenance decision-support prototype.

The model does not automatically stop equipment, prescribe maintenance, or replace engineering judgement.

## Lifecycle

The proposed lifecycle is:

1. Collect operational data.
2. Validate schema, completeness, ranges, and duplicates.
3. Prepare model features.
4. Apply deterministic operational rules.
5. Generate an ML risk score.
6. Assign LOW, MEDIUM, or HIGH risk.
7. Present alerts through an API or dashboard.
8. Engineer investigates the equipment and operating context.
9. Record the action and eventual outcome.
10. Monitor data quality, drift, alert behaviour, and later model performance.
11. Review whether retraining is justified.
12. Validate a candidate model before controlled promotion.

## Monitoring Layers

### Data quality

Monitor:

- missing values;
- duplicate observations;
- schema changes;
- invalid ranges;
- equipment coverage;
- unexpected categories.

Bad input data should be investigated before relying on model output.

### Data and feature drift

PlantOps AI includes Population Stability Index (PSI) as a simple prototype drift diagnostic for numeric features.

Prototype interpretation:

- PSI < 0.10: LOW
- 0.10 <= PSI < 0.25: WATCH
- PSI >= 0.25: HIGH

These are monitoring conventions for this prototype, not universal industrial limits.

Drift is an investigation signal. It does not automatically prove that model performance has degraded.

### Score and alert monitoring

Monitor:

- mean risk score;
- LOW/MEDIUM/HIGH distribution;
- ML alert rate;
- deterministic rule alert rate;
- combined alert rate.

A sudden distribution change may indicate a genuine operational change, sensor/data problems, or model-input drift.

### Performance monitoring

True predictive performance can only be measured when reliable outcome labels become available.

Relevant metrics include:

- precision;
- recall;
- PR-AUC;
- ROC-AUC;
- false-positive burden;
- false-negative events.

Because the prototype uses synthetic data, current evaluation results must not be presented as expected production performance.

## Retraining

Retraining should not be triggered automatically by one drift statistic.

A production process should consider:

- sustained data drift;
- degradation in labelled performance;
- new equipment or operating regimes;
- sensor or process changes;
- sufficient new labelled examples;
- engineering feedback.

A retrained model should be validated against the current approved model before promotion.

## Human in the Loop

PlantOps AI is designed around:

Risk signal → engineer investigation → engineering decision → action → recorded outcome.

Known engineering limits remain deterministic rules.

ML complements these rules by identifying multivariate patterns that may deserve investigation.

## Governance

Each deployed model should record:

- model version;
- training-data period;
- feature contract;
- model family;
- thresholds;
- evaluation results;
- known limitations;
- responsible owner;
- approval date.

Rollback to a previous approved model should remain possible.

## Security

A production implementation should include:

- authentication and authorisation;
- least-privilege access;
- encrypted transport;
- protected model artifacts;
- controlled configuration changes;
- audit logging;
- secrets outside source code;
- appropriate network segmentation.

## Current Prototype Limitations

- Synthetic data only.
- Small number of positive failure examples.
- Risk scores are not calibrated probabilities.
- No real sensor integration.
- No real maintenance work-order integration.
- No automated retraining.
- No automatic safety or shutdown decisions.
