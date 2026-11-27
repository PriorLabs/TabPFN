#  Copyright (c) Prior Labs GmbH 2026.
"""Batched forecasting over many M4 series with TabPFNRegressor.predict_batched.

Follows the TabPFN time series framing: a forecast is just tabular regression
where the features describe *when* a point occurs, not its past values. Each
series becomes its own small table of calendar features, so the series are
independent and identically shaped, which is exactly what `predict_batched`
wants. Instead of one forward pass per series per estimator, it stacks the
series on the model's batch dimension and runs one fused forward per estimator.
This times that against the sequential fit + predict loop.
"""

import time

import numpy as np
import pandas as pd

from tabpfn import TabPFNRegressor

M4_HOURLY = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Hourly-train.csv"
CONTEXT, HORIZON, N_SERIES = 400, 48, 20


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """M4's own metric. Scale free, so series in the hundreds and in the
    thousands contribute equally to the average.
    """
    return 200 * np.mean(
        np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast))
    )


def timestamp_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Describe each point by when it happens, plus a running index for trend.

    The sin/cos pairs give the daily and weekly cycles a smooth, wrap-around
    encoding, so hour 23 sits next to hour 0 rather than at the far end.
    """
    daily = 2 * np.pi * index.hour / 24
    weekly = 2 * np.pi * (index.dayofweek * 24 + index.hour) / 168
    return pd.DataFrame(
        {
            "step": np.arange(len(index)),
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "daily_sin": np.sin(daily),
            "daily_cos": np.cos(daily),
            "weekly_sin": np.sin(weekly),
            "weekly_cos": np.cos(weekly),
        }
    )


def to_forecasting_task(values: np.ndarray) -> tuple:
    """Split one series into a train table and the HORIZON steps to forecast."""
    values = values[-(CONTEXT + HORIZON) :]
    index = pd.date_range("2010-01-01", periods=len(values), freq="h")
    features = timestamp_features(index)
    return (
        features[:CONTEXT],
        values[:CONTEXT],
        features[CONTEXT:],
        values[CONTEXT:],
    )


raw = pd.read_csv(M4_HOURLY).iloc[:, 1:]
series = [row[~np.isnan(row)] for row in raw.to_numpy(dtype=np.float64)]
series = [s for s in series if len(s) >= CONTEXT + HORIZON][:N_SERIES]
tasks = [to_forecasting_task(s) for s in series]
X_trains, y_trains, X_tests, y_tests = map(list, zip(*tasks, strict=True))
print(f"{len(tasks)} series, {CONTEXT} hours of context, {HORIZON} hours to forecast")

reg = TabPFNRegressor()
reg.fit(X_trains[0], y_trains[0])  # warm up: load the checkpoint outside the timings

start = time.perf_counter()
sequential = [reg.fit(X, y).predict(X_test) for X, y, X_test, _ in tasks]
t_sequential = time.perf_counter() - start

start = time.perf_counter()
batched = reg.predict_batched(X_trains, y_trains, X_tests)
t_batched = time.perf_counter() - start

score = np.mean([smape(t, p) for t, p in zip(y_tests, batched, strict=True)])
# Sanity check: repeating the last observed day should be beaten.
naive = np.mean(
    [
        smape(t, np.tile(y[-24:], HORIZON // 24))
        for y, t in zip(y_trains, y_tests, strict=True)
    ]
)
# Relative, because M4 hourly series live on very different scales. What is left
# is float accumulation order: batched and sequential matmuls do not associate
# identically, more so under autocast.
drift = max(
    np.abs(a - b).max() / np.abs(a).mean()
    for a, b in zip(sequential, batched, strict=True)
)
print(f"sMAPE {score:.2f}% (seasonal naive {naive:.2f}%)")
print(f"max relative deviation from sequential {drift:.1e}")
print(f"sequential {t_sequential:.1f}s, batched {t_batched:.1f}s")
print(f"speedup {t_sequential / t_batched:.1f}x")
