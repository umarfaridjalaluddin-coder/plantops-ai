"""Generate reproducible synthetic operational data for PlantOps AI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from plantops_ai.config import (
    EQUIPMENT_TYPES,
    N_DAYS,
    N_EQUIPMENT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)


def generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic equipment observations and a 7-day failure target."""
    rng = np.random.default_rng(RANDOM_SEED)

    equipment_ids = [f"EQ-{i:03d}" for i in range(1, N_EQUIPMENT + 1)]
    equipment_type_map = {
        equipment_id: EQUIPMENT_TYPES[(i - 1) % len(EQUIPMENT_TYPES)]
        for i, equipment_id in enumerate(equipment_ids, start=1)
    }

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=N_DAYS,
        freq="D",
    )

    rows: list[dict[str, object]] = []

    for equipment_id in equipment_ids:
        equipment_type = equipment_type_map[equipment_id]

        base_runtime = rng.uniform(500, 8000)
        base_maintenance_age = int(rng.integers(0, 90))

        for day_index, timestamp in enumerate(timestamps):
            runtime_hours = base_runtime + (day_index * rng.uniform(8, 24))

            days_since_maintenance = base_maintenance_age + day_index
            if days_since_maintenance > 120 and rng.random() < 0.12:
                days_since_maintenance = int(rng.integers(0, 8))
                base_maintenance_age = days_since_maintenance - day_index

            load_pct = float(np.clip(rng.normal(72, 14), 25, 100))

            temperature_c = float(
                np.clip(
                    rng.normal(68, 7)
                    + max(load_pct - 75, 0) * 0.18
                    + days_since_maintenance * 0.025,
                    40,
                    110,
                )
            )

            vibration_mm_s = float(
                np.clip(
                    rng.normal(2.8, 0.9)
                    + max(load_pct - 80, 0) * 0.035
                    + days_since_maintenance * 0.008,
                    0.3,
                    10,
                )
            )

            pressure_bar = float(
                np.clip(
                    rng.normal(6.2, 1.1) + (load_pct - 70) * 0.015,
                    1.5,
                    12,
                )
            )

            motor_current_a = float(
                np.clip(
                    rng.normal(32, 5) + load_pct * 0.12,
                    10,
                    70,
                )
            )

            # Synthetic latent failure mechanism.
            #
            # This intentionally combines several operational signals rather
            # than making failure depend on one threshold. It gives the later
            # ML models a multivariate pattern to learn.
            risk_score = (
                -5.4
                + 0.055 * max(temperature_c - 70, 0)
                + 0.38 * max(vibration_mm_s - 3.0, 0)
                + 0.018 * max(load_pct - 75, 0)
                + 0.009 * max(days_since_maintenance - 45, 0)
                + 0.00006 * max(runtime_hours - 3000, 0)
            )

            failure_probability = 1.0 / (1.0 + np.exp(-risk_score))
            failure_next_7d = int(rng.random() < failure_probability)

            rows.append(
                {
                    "timestamp": timestamp,
                    "equipment_id": equipment_id,
                    "equipment_type": equipment_type,
                    "temperature_c": round(temperature_c, 2),
                    "vibration_mm_s": round(vibration_mm_s, 2),
                    "pressure_bar": round(pressure_bar, 2),
                    "motor_current_a": round(motor_current_a, 2),
                    "runtime_hours": round(runtime_hours, 2),
                    "days_since_maintenance": days_since_maintenance,
                    "load_pct": round(load_pct, 2),
                    TARGET_COLUMN: failure_next_7d,
                }
            )

    return pd.DataFrame(rows)


def save_synthetic_data(df: pd.DataFrame) -> None:
    """Save the synthetic dataset as a CSV file."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DATA_DIR / "equipment_operations.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Failure rows: {int(df[TARGET_COLUMN].sum()):,}")
    print(f"Failure prevalence: {df[TARGET_COLUMN].mean():.2%}")


def main() -> None:
    """Generate and save the PlantOps AI synthetic dataset."""
    df = generate_synthetic_data()
    save_synthetic_data(df)


if __name__ == "__main__":
    main()
