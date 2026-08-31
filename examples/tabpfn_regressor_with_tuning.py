#  Copyright (c) Prior Labs GmbH 2026.
"""Example of calibrating the predictive distribution of a TabPFNRegressor.

`TabPFNRegressor` sharpens every ensemble member's bucket logits by the
`softmax_temperature` its checkpoint declares (0.9 for the current models).
Passing `tuning_config={"calibrate_temperature": True}`
allows tuning the sharpness of the ensemble distribution logits:
log(sum(TabPFNRegressor_i.predict(X))/ensemble_softmax_temperature_).
This does not touch the softmax_temperature of individual ensemble members.

`eval_metric` chooses by what metric to calibrate. See
[tabpfn.inference_tuning.RegressorEvalMetrics][] for the full set of options.
"""

import numpy as np
import torch
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor
from tabpfn.inference_tuning import (
    RegressorEvalMetrics,
    compute_regression_metric_to_minimize,
)

X, y = make_friedman1(n_samples=3_000, n_features=8, noise=8.0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
print(f"Train/test split: {len(X_train)}/{len(X_test)} rows")


QUANTILE_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def evaluate(regressor: TabPFNRegressor) -> tuple[float, float, dict[float, float]]:
    """Return test-set NLL, CRPS and the empirical coverage of the quantiles."""
    output = regressor.predict(X_test, output_type="full", quantiles=QUANTILE_LEVELS)

    # `criterion` is the bar distribution in the units of the original target, so it
    # scores the raw `y_test` directly. Both metrics return one loss per row.
    criterion = output["criterion"]
    y_test_tensor = torch.as_tensor(
        y_test, dtype=criterion.borders.dtype, device=output["logits"].device
    )
    nll = float(criterion(output["logits"], y_test_tensor).mean())
    crps = float(
        compute_regression_metric_to_minimize(
            metric_name=RegressorEvalMetrics.CRPS,
            raw_space_bardist=criterion,
            logits=output["logits"],
            y_true=y_test_tensor,
        ).mean()
    )

    # A well-calibrated distribution puts a `q` fraction of the targets below its
    # `q`-quantile. Under-dispersed predictions undershoot in both tails.
    coverage = {
        q: float(np.mean(y_test <= prediction))
        for q, prediction in zip(QUANTILE_LEVELS, output["quantiles"], strict=True)
    }
    return nll, crps, coverage


def report(name: str, regressor: TabPFNRegressor) -> tuple[float, float, float]:
    nll, crps, coverage = evaluate(regressor)
    coverage_error = float(np.mean([abs(c - q) for q, c in coverage.items()]))
    print(f"\n{name}")
    print(
        f"  ensemble_softmax_temperature_: "
        f"{regressor.ensemble_softmax_temperature_:.4f}"
    )
    print(f"  test NLL:  {nll:.4f}")
    print(f"  test CRPS: {crps:.4f}")
    formatted = "  ".join(f"q{q:g}: {c:.3f}" for q, c in coverage.items())
    print(f"  empirical coverage ({formatted})")
    print(f"  mean |coverage - nominal|: {coverage_error:.4f}")
    return nll, crps, coverage_error


# Baseline: the fixed per-estimator temperature of 0.9, no calibration.
reg_baseline = TabPFNRegressor(random_state=42)
reg_baseline.fit(X_train, y_train)
nll_baseline, crps_baseline, coverage_error_baseline = report(
    "Without calibration", reg_baseline
)

# With calibration: `fit()` additionally fits throwaway regressors on cross-validation
# splits of the training data and tunes the temperature on their pooled holdout rows.
calibrated = {}
results = {}
for metric in ("nll", "crps"):
    regressor = TabPFNRegressor(
        random_state=42,
        eval_metric=metric,
        tuning_config={"calibrate_temperature": True},
    )
    regressor.fit(X_train, y_train)
    calibrated[metric] = regressor
    results[metric] = report(f"Calibrated on {metric.upper()}", regressor)

print("\nImprovement over the uncalibrated baseline (positive is better):")
print(f"  {'calibrated on':<16}{'NLL gain':>10}{'CRPS gain':>12}{'coverage gain':>16}")
for metric, (nll, crps, coverage_error) in results.items():
    print(
        f"  {metric.upper():<16}{nll_baseline - nll:>+10.4f}"
        f"{crps_baseline - crps:>+12.4f}"
        f"{coverage_error_baseline - coverage_error:>+16.4f}"
    )

# The point predictions are barely affected -- calibration reshapes the distribution
# rather than moving its centre -- so the gain shows up in NLL, CRPS and the quantiles.
mean_shift = np.abs(
    calibrated["nll"].predict(X_test) - reg_baseline.predict(X_test)
).mean()
print(f"\nMean absolute change in point predictions: {mean_shift:.4f}")
