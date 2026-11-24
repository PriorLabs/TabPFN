#  Copyright (c) Prior Labs GmbH 2026.

"""A float32 table predicts as a float64 table when float32 rounds values together."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

import tabpfn.base
import tabpfn.classifier
from tabpfn import TabPFNClassifier
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.configs import PreprocessorConfig

N_ROWS = 600
N_TEST = 200


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    float64_table: bool,
    by_ranges: bool = False,
    categorical: list[int] | None = None,
    tuning_config: dict | None = None,
    **config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """The cleaned fit table and the test rows' probabilities."""
    captured: dict[str, Any] = {}

    def fit_spy(*args: Any, **kwargs: Any) -> Any:
        captured["fit"] = tabpfn.base.validate_and_clean_fit_input(*args, **kwargs)
        return captured["fit"]

    n_rows = len(X) - N_TEST
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tabpfn.classifier, "validate_and_clean_fit_input", fit_spy)
        if float64_table:
            patch.setattr(
                tabpfn.base, "cleaned_table_dtype", lambda _: np.dtype(np.float64)
            )
        if by_ranges:
            patch.setattr(tabpfn.base, "_RANGE_BYTES", 2 * n_rows * 32)
        model = TabPFNClassifier(
            n_estimators=2,
            device="cpu",
            random_state=0,
            categorical_features_indices=categorical,
            tuning_config=tuning_config,
            inference_config={"ENABLE_GPU_PREPROCESSING": True, **config},
        )
        model.fit(X[:n_rows], y[:n_rows])
        return captured["fit"][0], model.predict_proba(X[n_rows:])


def _with_tiny_steps(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Features whose last column has steps too small for float32, and targets."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    X[:, 4] = 1 + 1e-12 * np.arange(n)
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


@pytest.mark.parametrize("by_ranges", [False, True])
def test__column_constant_only_in_float32__predicts_as_a_float64_table(
    by_ranges: bool,
) -> None:
    X, y = _with_tiny_steps(N_ROWS + N_TEST)
    assert np.unique(X[:N_ROWS, 4].astype(np.float32)).size == 1

    table, proba = _fit(X, y, float64_table=False, by_ranges=by_ranges)
    _, expected = _fit(X, y, float64_table=True)

    assert table.dtype == np.float32
    np.testing.assert_array_equal(proba, expected)


@pytest.mark.parametrize("by_ranges", [False, True])
def test__tuning__column_constant_only_in_float32__predicts_as_a_float64_table(
    by_ranges: bool,
) -> None:
    """The holdout models that tuning fits read the table too. The column carries
    the target in steps too small for float32, so dropping it moves the temperature.
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, N_ROWS + N_TEST)
    X = rng.standard_normal((N_ROWS + N_TEST, 6))
    X[:, 0] = 1.0 + (y + rng.integers(0, 3, len(y))) * 1e-12
    tuning = {"calibrate_temperature": True}

    _, proba = _fit(
        X, y, float64_table=False, by_ranges=by_ranges, tuning_config=tuning
    )
    _, expected = _fit(X, y, float64_table=True, tuning_config=tuning)

    np.testing.assert_array_equal(proba, expected)


@pytest.mark.parametrize("by_ranges", [False, True])
def test__column_constant_in_float32_on_one_members_rows__predicts_as_float64(
    by_ranges: bool,
) -> None:
    X, y = _with_tiny_steps(N_ROWS + N_TEST)
    # Both columns are constant only on rows that leave out the first three: the
    # last column only in float32, the column before it also in float64.
    X[:, 3] = 1.0
    X[:3, 3:] = 5.0

    table, proba = _fit(
        X, y, float64_table=False, by_ranges=by_ranges, SUBSAMPLE_SAMPLES=300
    )
    _, expected = _fit(X, y, float64_table=True, SUBSAMPLE_SAMPLES=300)

    assert table.dtype == np.float32
    np.testing.assert_array_equal(proba, expected)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="LightGBM requires libomp which is not available on macOS CI",
)
def test__importance_order__predicts_as_a_float64_table() -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    n = 1500 + N_TEST
    X = rng.standard_normal((n, 10))
    # The target is the low bit of the id. float32 rounds the ids into fewer groups,
    # and each group mixes both targets.
    ids = rng.integers(0, 8, n)
    X[:, 0] = 2.0**25 + ids
    y = ids % 2
    X[:, 1] = y + 1.5 * rng.standard_normal(n)
    config: dict[str, Any] = {
        "FEATURE_SUBSAMPLING_METHOD": "gini_feature_importance",
        "FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT": 1,
        "PREPROCESS_TRANSFORMS": [
            PreprocessorConfig(
                "none",
                categorical_name="ordinal_shuffled",
                max_features_per_estimator=4,
            )
        ],
    }

    table, proba = _fit(X, y, float64_table=False, **config)
    _, expected = _fit(X, y, float64_table=True, **config)

    assert table.dtype == np.float32
    np.testing.assert_array_equal(proba, expected)


@pytest.mark.parametrize("by_ranges", [False, True])
def test__codes_float32_cannot_hold__keep_a_float64_table(
    by_ranges: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y = _with_tiny_steps(N_ROWS + N_TEST)
    X[:, 2] = np.arange(N_ROWS + N_TEST) % 8
    # Codes above 2 stand for codes float32 cannot hold.
    monkeypatch.setattr(tabpfn.base, "_FLOAT32_EXACT_INTEGERS", 2)

    table, proba = _fit(X, y, float64_table=False, by_ranges=by_ranges, categorical=[2])
    _, expected = _fit(X, y, float64_table=True, categorical=[2])

    assert table.dtype == np.float64
    np.testing.assert_array_equal(proba, expected)


def test__initialize_dataset_preprocessing__keeps_a_float64_table() -> None:
    """Its callers, such as the fine-tuning datasets, fit members on the table
    without the decisions, so it keeps the cleaned values.
    """
    X, y = _with_tiny_steps(N_ROWS)
    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        inference_config={
            "FINGERPRINT_FEATURE": False,
            "POLYNOMIAL_FEATURES": "no",
            "PREPROCESS_TRANSFORMS": [
                PreprocessorConfig("none", categorical_name="numeric")
            ],
        },
    )
    model._initialize_model_variables()

    _, table, _ = model._initialize_dataset_preprocessing(X=X, y=y, random_state=0)

    assert table.dtype == np.float64


def test__fit__scalar_input__raises_a_validation_error() -> None:
    with pytest.raises(TabPFNValidationError, match="2D"):
        TabPFNClassifier(n_estimators=1, device="cpu").fit(np.array(1.0), [0])
