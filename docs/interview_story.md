# PlantOps AI — Interview Story

## 30-Second Summary

PlantOps AI is a synthetic end-to-end prototype showing how I would approach AI-assisted operational monitoring and predictive maintenance.

I deliberately started with the engineering workflow rather than just training a model: data validation, leakage-safe model development, deterministic operational rules, ML risk scoring, an API, Power BI-ready analytics, monitoring, governance, and a human-in-the-loop decision process.

The prototype uses no company operational data and is intended to demonstrate the architecture and engineering approach rather than claim production performance.

## Problem

The business question is:

> Given current equipment operating information, which observations should engineers prioritise for investigation because they show elevated risk of a failure within the next seven days?

The objective is prioritisation and earlier investigation.

It is not automatic maintenance or automatic equipment shutdown.

## Why Combine Rules and ML?

Known engineering limits are best represented explicitly.

For example, if an approved engineering limit says vibration must not exceed a certain value, that condition should remain transparent and deterministic.

ML addresses a different problem: several individually acceptable measurements may combine into an unusual operating pattern.

Therefore:

- rules capture known conditions;
- ML captures multivariate patterns;
- engineers make the final operational decision.

## Data Approach

For this demonstration I generated synthetic operational data for 100 equipment units across five equipment types.

The data contains operational variables such as temperature, vibration, pressure, current, runtime, maintenance age, and load.

I also built validation for missing values, duplicates, categories, ranges, and schema expectations.

With real operational data, I would first validate sensor reliability, maintenance records, failure definitions, sampling frequency, and equipment context before selecting an ML approach.

## Leakage Control

A key design decision was splitting by equipment rather than randomly splitting individual rows.

That prevents observations from the same equipment unit appearing in both development and test partitions.

The final split is:

- 60 equipment for training;
- 20 for validation;
- 20 for the locked final test.

This makes the final evaluation closer to the question of whether the model generalises to unseen equipment.

## Model Development

I compared a class-weighted Logistic Regression model with a HistGradientBoosting candidate.

Logistic Regression performed better on validation ranking metrics in this experiment and was also easier to explain.

I therefore froze Logistic Regression as `plantops-lr-v1`.

I did not tune the model against the final test set.

## Why PR-AUC Matters

The synthetic failure class is rare.

Overall prevalence is only about 1.47%.

That means accuracy alone can be misleading because a model predicting no failures would still appear highly accurate.

I therefore considered:

- PR-AUC;
- ROC-AUC;
- precision;
- recall;
- confusion matrix;
- operational alert burden.

## Threshold Decision

The HIGH-risk threshold was selected using validation data, not test data.

The prototype requirement was to maintain at least 50% validation recall and then reduce false positives among eligible thresholds.

That resulted in a threshold of 0.60.

The final locked test produced lower recall, at 40%, which is exactly why I keep the final test separate and do not retune after seeing it.

## Final Test Result

On 600 observations from 20 unseen synthetic equipment units:

- ROC-AUC: 0.7473
- PR-AUC: 0.0523
- precision: 0.0364
- recall: 0.4000
- TP: 4
- FP: 106
- FN: 6
- TN: 484

The model shows ranking signal but generates many false positives.

I would not describe this as production-ready performance.

The important demonstration is the disciplined development process and the architecture around the model.

## Why Risk Score Instead of Probability?

The Logistic Regression model uses class weighting to handle the rare positive class.

As a result, the raw output is not calibrated to the actual failure prevalence.

I therefore expose it as an **uncalibrated risk score**, not a literal probability of failure.

With sufficient real labelled data, calibration could be evaluated separately.

## Explainability

The fitted model places relatively strong weight on variables including:

- days since maintenance;
- load;
- vibration;
- pressure;
- runtime.

These are model associations in synthetic data, not causal conclusions.

The purpose of explainability here is to help an engineer understand what is contributing to the risk signal.

## API

I wrapped the decision-support layer with FastAPI.

The API exposes health, model metadata, equipment information, and prediction endpoints.

A prediction returns:

- model version;
- risk score;
- risk band;
- ML alert;
- rule alert;
- decision-support flag;
- score interpretation.

This makes the model usable by another application rather than leaving it as a notebook experiment.

## Power BI

I also generate a Power BI-ready scoring dataset.

The dashboard concept gives an operations or engineering team a way to view:

- equipment risk;
- alert counts;
- operating measurements;
- equipment trends;
- equipment requiring investigation.

This separates the ML backend from the operational consumption layer.

## Monitoring

Deployment is not the end of the AI lifecycle.

The prototype monitors:

- data quality;
- feature drift;
- risk distribution;
- ML alert rate;
- rule alert rate;
- combined alert rate.

In one temporal comparison, `days_since_maintenance` produces strong PSI drift.

That is explainable because maintenance age naturally increases through the synthetic timeline.

This demonstrates why a drift statistic should trigger investigation rather than automatic retraining.

## Human in the Loop

The intended production workflow is:

```text
Risk / rule alert
       ↓
Engineer investigates
       ↓
Engineering decision
       ↓
Action
       ↓
Outcome recorded
       ↓
Monitoring and future improvement
```

The ML system assists prioritisation.

It does not replace engineering responsibility.

## Cost-Saving Framework

I would not claim a monetary saving without actual operational data.

Instead I would quantify a production business case using measurable components such as:

```text
Avoided downtime value
+ avoided secondary damage
+ maintenance-efficiency improvement
+ reduced manual monitoring effort
- implementation cost
- operating cost
- false-alert investigation cost
```

A pilot should establish these values using real equipment and maintenance history.

## If Real Data Is Not Ready

I would not force an AI model.

I would first establish:

1. equipment and sensor inventory;
2. failure and maintenance definitions;
3. reliable timestamped data capture;
4. data-quality monitoring;
5. deterministic operational KPIs and alerts;
6. labelled maintenance outcomes.

Only then would I determine whether ML adds enough value beyond rules and conventional analytics.

## Potential Extension Beyond Predictive Maintenance

The same engineering approach can support other use cases, for example:

- process-efficiency anomaly detection;
- energy or resource-consumption monitoring;
- quality/yield monitoring;
- technical-support issue classification and knowledge retrieval.

I would prioritise based on data readiness, measurable business value, operational risk, and implementation effort.

## Closing Message

PlantOps AI is not presented as a finished industrial AI product.

It demonstrates how I approach the complete problem:

**data → validation → engineering rules → ML → API/analytics → monitoring → human decision → feedback.**

The key principle is that AI should support reliable operational decisions and measurable process improvement, not exist as an isolated model.
