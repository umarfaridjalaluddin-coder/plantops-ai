# Open PlantOps AI PBIP

Open this file in Power BI Desktop:

`/home/umarfarid/projects/plantops-ai/powerbi/PlantOps AI - Operational Risk Dashboard.PBIP/PlantOps AI - Operational Risk Dashboard.pbip`

If Power BI Desktop cannot reach the source CSV, open Transform Data > Manage Parameters and update `CsvPath`.

Default Windows CSV path:

`\\wsl.localhost\Ubuntu\home\umarfarid\projects\plantops-ai\data\processed\plantops_powerbi.csv`

Equivalent Linux/WSL source path:

`/home/umarfarid/projects/plantops-ai/data/processed/plantops_powerbi.csv`

After opening, select Refresh. The verified full-dataset dashboard values should be:

- Equipment Count: 100
- High Risk Observations: 595
- Combined Alerts: 656
- Average Risk Score: approximately 0.4336
- ML Alerts: 595
- Rule Alerts: 178
- Total Observations: 3000

These are synthetic full-dataset visualization counts, not holdout model-performance metrics. The Python/ML backend, model `plantops-lr-v1`, threshold `0.60`, and risk-band logic are not modified by this PBIP.
