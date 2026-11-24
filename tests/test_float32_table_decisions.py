#  Copyright (c) Prior Labs GmbH 2026.

"""A float32 table predicts as a float64 one where float32 rounds values together."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import tabpfn.base
import tabpfn.classifier
from tabpfn import TabPFNClassifier

N_ROWS = 600
N_TEST = 200


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    float64_table: bool,
    by_ranges: bool = False,
    categorical: list[int] | None = None,
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
            inference_config={"ENABLE_GPU_PREPROCESSING": True, **config},
        )
        model.fit(X[:n_rows], y[:n_rows])
        return captured["fit"][0], model.predict_proba(X[n_rows:])


def _with_tiny_steps(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Features with a last column of steps float32 rounds away, and targets."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    X[:, 4] = 1 + 1e-12 * np.arange(n)
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


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
