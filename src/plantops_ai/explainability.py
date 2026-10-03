"""Explainability and calibration diagnostics for PlantOps AI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_diagnostics(
    y_true,
    probabilities,
    n_bins: int = 5,
) -> tuple[float, pd.DataFrame]:
    """Return Brier score and a calibration table."""
    y_true_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)

    brier_score = float(
        brier_score_loss(
            y_true_array,
            probability_array,
        )
    )

    observed_fraction, mean_predicted = calibration_curve(
        y_true_array,
        probability_array,
        n_bins=n_bins,
        strategy="quantile",
    )

    table = pd.DataFrame(
        {
            "mean_predicted_score": mean_predicted,
            "observed_failure_rate": observed_fraction,
        }
    )

    return brier_score, table


def logistic_feature_coefficients(model) -> pd.DataFrame:
    """Return transformed Logistic Regression feature coefficients."""
    preprocessing = model.named_steps["preprocessing"]
    classifier = model.named_steps["model"]

    feature_names = preprocessing.get_feature_names_out()
    coefficients = classifier.coef_[0]

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    result["absolute_coefficient"] = result["coefficient"].abs()

    return result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)
