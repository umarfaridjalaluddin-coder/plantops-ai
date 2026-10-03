# PlantOps AI — Power BI Dashboard

## Purpose

This dashboard presents synthetic PlantOps AI operational-monitoring and predictive-maintenance decision-support data.

It is an independent prototype and is not based on Tradewinds Plantation Berhad operational data.

Model outputs are uncalibrated risk scores and must not be interpreted as literal failure probabilities.

## Data Source

Generate the dashboard dataset with:

```bash
python -m plantops_ai.powerbi_export
```

Import:

`data/processed/plantops_powerbi.csv`

Current generated dataset:

- 3,000 observations
- 100 synthetic equipment units
- 5 equipment types
- 20 columns
- Model version: plantops-lr-v1

## Dashboard Page 1 — Operations Overview

### KPI cards

1. Equipment Count
2. High-Risk Observations
3. Combined Alerts
4. Average Risk Score

### Visuals

- Donut chart: observations by risk band
- Clustered bar chart: high-risk observations by equipment type
- Line chart: average risk score by timestamp
- Table: highest-risk equipment observations

### Filters

- Equipment type
- Equipment ID
- Risk band
- Timestamp

## Dashboard Page 2 — Equipment Investigation

Use a selected equipment ID to display:

- Risk score over time
- Temperature over time
- Vibration over time
- Load over time
- Days since maintenance
- ML alert
- Rule alert
- Combined alert

This page supports engineer investigation rather than automatic maintenance decisions.

## Suggested DAX Measures

```DAX
Equipment Count =
DISTINCTCOUNT(plantops_powerbi[equipment_id])

Observation Count =
COUNTROWS(plantops_powerbi)

Average Risk Score =
AVERAGE(plantops_powerbi[risk_score])

High-Risk Observations =
CALCULATE(
    COUNTROWS(plantops_powerbi),
    plantops_powerbi[risk_band] = "HIGH"
)

Medium-Risk Observations =
CALCULATE(
    COUNTROWS(plantops_powerbi),
    plantops_powerbi[risk_band] = "MEDIUM"
)

Low-Risk Observations =
CALCULATE(
    COUNTROWS(plantops_powerbi),
    plantops_powerbi[risk_band] = "LOW"
)

ML Alerts =
CALCULATE(
    COUNTROWS(plantops_powerbi),
    plantops_powerbi[ml_alert] = TRUE()
)

Rule Alerts =
CALCULATE(
    COUNTROWS(plantops_powerbi),
    plantops_powerbi[rule_alert] = TRUE()
)

Combined Alerts =
CALCULATE(
    COUNTROWS(plantops_powerbi),
    plantops_powerbi[combined_alert] = TRUE()
)

High-Risk Rate =
DIVIDE(
    [High-Risk Observations],
    [Observation Count]
)

Combined Alert Rate =
DIVIDE(
    [Combined Alerts],
    [Observation Count]
)
```

## Prototype Reference Values

The current reproducible synthetic export produces:

- LOW: 782 observations
- MEDIUM: 1,623 observations
- HIGH: 595 observations
- ML alerts: 595
- Rule alerts: 178
- Combined alerts: 656

These values are verification references for the current synthetic dataset, not operational targets or production KPIs.

## Dashboard Disclaimer

Display this text visibly on the dashboard:

> Synthetic prototype data — decision support only. Risk scores are uncalibrated and are not literal probabilities of equipment failure.

## Interview Story

The dashboard demonstrates the final consumption layer of the PlantOps AI architecture:

Operational data → validation → feature preparation → ML + deterministic rules → risk scoring → API/dashboard → engineer investigation and action.

The design deliberately keeps a human in the loop. A HIGH risk score or rule alert prioritises an observation for investigation; it does not automatically shut down equipment or prescribe maintenance.
