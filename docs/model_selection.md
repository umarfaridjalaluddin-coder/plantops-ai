# PlantOps AI Model Selection

## Experimental design

PlantOps AI uses a reproducible equipment-grouped split:

- Training: 60 equipment / 1,800 observations
- Validation: 20 equipment / 600 observations
- Test: 20 equipment / 600 observations

Equipment IDs do not overlap across partitions.

The final test partition was kept locked during model-family and threshold selection.

## Model-family decision

Two candidate models were compared on validation data.

Logistic Regression:

- ROC-AUC: 0.7323
- PR-AUC: 0.0601

HistGradientBoosting:

- ROC-AUC: 0.5710
- PR-AUC: 0.0287

Validation positive prevalence was 0.0150.

Logistic Regression was selected as the primary model because it provided stronger validation ranking performance while remaining comparatively interpretable.

## Prototype operating threshold

The prototype selection rule was defined as:

> Maintain validation failure recall of at least 50%, then select the tested threshold with the fewest false-positive alerts.

Among thresholds tested from 0.30 to 0.90 in increments of 0.05, the selected operating threshold was 0.60.

Validation results at 0.60:

- Recall: 0.5556
- Precision: 0.0407
- True positives: 5
- False positives: 118
- False negatives: 4
- True negatives: 473
- Alerts: 123 / 600 observations

This threshold is a prototype decision-support assumption and is not a validated industrial maintenance or safety threshold.

## Risk bands

The prototype uses:

- LOW: score < 0.30
- MEDIUM: 0.30 <= score < 0.60
- HIGH: score >= 0.60

The 0.60 HIGH boundary corresponds to the validation-selected operating threshold.

The 0.30 MEDIUM boundary is a prototype watch-tier convention and was not statistically optimized.

Model outputs are treated as risk scores rather than calibrated probabilities.

## Limitations

The dataset is synthetic.

There were only 44 positive observations in the complete dataset, including 25 in training, 9 in validation, and 10 in the locked test partition.

Therefore, performance estimates have substantial statistical uncertainty and should not be interpreted as production performance.
