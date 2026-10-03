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

## Final locked holdout evaluation

After the model family, operating threshold, and risk-band policy were frozen using training and validation data, the final test partition was evaluated once.

Final test partition:

- 20 previously unseen equipment units
- 600 observations
- 10 positive observations
- Positive prevalence: 0.0167

Frozen Logistic Regression results:

- ROC-AUC: 0.7473
- PR-AUC: 0.0523
- Operating threshold: 0.60
- Precision: 0.0364
- Recall: 0.4000
- F1: 0.0667
- True positives: 4
- False positives: 106
- False negatives: 6
- True negatives: 484

Test risk-band counts:

- LOW: 182
- MEDIUM: 308
- HIGH: 110

The final test result was not used to change the selected model, operating threshold, or risk-band policy.

The model demonstrates useful ranking signal on this synthetic holdout dataset, but the operating point still produces many false-positive alerts. This prototype therefore represents decision support and prioritisation rather than production-ready automated maintenance decisions.

## Probability interpretation

The class-balanced Logistic Regression was trained to improve learning from a highly imbalanced target.

Its raw scores should not currently be interpreted as literal probabilities of failure.

On the final test partition:

- Observed failure prevalence: 0.0167
- Mean model score: 0.4154

This difference motivates explicit calibration analysis before probability-based interpretation.
