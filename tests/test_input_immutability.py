#  Copyright (c) Prior Labs GmbH 2026.

"""Fitting and predicting must leave the caller's data exactly as it was.

`clean_data` hands a numeric array straight back rather than casting it into one
it owns, so the array the wrapper carries through `fit` *is* the caller's. What
keeps that safe is `PreprocessingPipeline._process_steps` taking a private copy
before any step mutates anything -- one guarantee, in one place, rather than a
defensive copy at every boundary. Several things inside cleaning and preprocessing
write in place quite deliberately (`_extract_inf_masks` NaNs out infinities,
`process_text_na_dataframe` is documented to mutate the frame it is given), so the
question is not whether anything mutates but whether any of it reaches back to the
caller.

Every input is checked, not just the training features: `y` is label-encoded and
z-normalised on the way through, and the test features take a different route
again -- through `clean_data_transform` and a fitted encoder rather than a fresh
one. Each input kind below reaches cleaning by a different path: a plain numeric
array takes the single-cast shortcut, an object array and a DataFrame go through
pandas, and NaN and +/-inf are rewritten in place internally before the steps run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing import pipeline_interface

N_ROWS = 80
N_TEST = 20
N_FEATURES = 5
# +/-inf reaches the ordinal encoder only with this on; without it validation
# rejects the input before any of the code under test runs.
PASSTHROUGH_INF = {"PASSTHROUGH_INF": True}


def _numeric(dtype: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_ROWS + N_TEST, N_FEATURES)).astype(dtype)


def _numeric_with_non_finite() -> np.ndarray:
    X = _numeric("float64")
    X[3, 1] = np.nan
    X[7, 2] = np.inf
    X[11, 0] = -np.inf
    X[N_ROWS + 2, 4] = np.nan
    X[N_ROWS + 5, 3] = np.inf
    return X


def _object_mixed() -> np.ndarray:
    rng = np.random.default_rng(1)
    n = N_ROWS + N_TEST
    X = np.empty((n, 4), dtype=object)
    X[:, 0] = rng.standard_normal(n)
    # Three levels, so it detects as categorical rather than as free text.
    X[:, 1] = [f"lvl{i % 3}" for i in range(n)]
    X[:, 2] = rng.integers(0, 5, n)
    X[:, 3] = [None if i % 17 == 0 else f"k{i % 2}" for i in range(n)]
    return X


def _dataframe_mixed() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = N_ROWS + N_TEST
    numeric = rng.standard_normal(n)
    numeric[5] = np.nan
    return pd.DataFrame(
        {
            "num": numeric,
            "cat": pd.Categorical([f"c{i % 3}" for i in range(n)]),
            "text": [f"s{i % 2}" for i in range(n)],
            "count": rng.integers(0, 4, n),
        }
    )


INPUTS = {
    "numeric_float32": lambda: _numeric("float32"),
    "numeric_float64": lambda: _numeric("float64"),
    "numeric_with_nan_and_inf": _numeric_with_non_finite,
    "object_mixed": _object_mixed,
    "dataframe_mixed": _dataframe_mixed,
}


def _split(X: Any) -> tuple[Any, Any]:
    """Train/test split that keeps the container type it was given."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:N_ROWS], X.iloc[N_ROWS:]
    return X[:N_ROWS], X[N_ROWS:]


def _snapshot(data: Any) -> Any:
    """A copy to compare against afterwards, deep enough to catch a write."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.copy(deep=True)
    return data.copy()


def _assert_unchanged(after: Any, before: Any, what: str) -> None:
    """Bit-for-bit, with NaN comparing equal to NaN in the same place."""
    if isinstance(after, pd.DataFrame):
        pd.testing.assert_frame_equal(after, before, check_exact=True)
    elif isinstance(after, pd.Series):
        pd.testing.assert_series_equal(after, before, check_exact=True)
    else:
        np.testing.assert_array_equal(after, before, err_msg=f"{what} was mutated")


@pytest.mark.parametrize(
    "estimator_cls", [TabPFNClassifier, TabPFNRegressor], ids=lambda c: c.__name__
)
@pytest.mark.parametrize("input_kind", list(INPUTS), ids=list(INPUTS))
def test__fit_predict__leave_every_input_untouched(
    estimator_cls: type, input_kind: str
) -> None:
    """A whole fit + predict changes nothing the caller passed in."""
    X_train, X_test = _split(INPUTS[input_kind]())
    rng = np.random.default_rng(3)
    y_train = (
        rng.integers(0, 3, N_ROWS)
        if estimator_cls is TabPFNClassifier
        else rng.standard_normal(N_ROWS)
    )

    before = {
        "X_train": _snapshot(X_train),
        "X_test": _snapshot(X_test),
        "y_train": _snapshot(y_train),
    }

    model = estimator_cls(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config=PASSTHROUGH_INF,
    ).fit(X_train, y_train)
    model.predict(X_test)
    if estimator_cls is TabPFNClassifier:
        model.predict_proba(X_test)

    after = {"X_train": X_train, "X_test": X_test, "y_train": y_train}
    for name, original in before.items():
        _assert_unchanged(after[name], original, name)


@pytest.mark.parametrize(
    "estimator_cls", [TabPFNClassifier, TabPFNRegressor], ids=lambda c: c.__name__
)
def test__fit_predict__leave_a_pandas_target_untouched(estimator_cls: type) -> None:
    """A `Series` target is label-encoded or z-normalised, never in place.

    Separate from the sweep above because only a pandas target carries an index
    and a name that a mutation could disturb as well as its values.
    """
    X = _numeric("float64")
    X_train, X_test = _split(X)
    rng = np.random.default_rng(4)
    values = (
        rng.integers(0, 3, N_ROWS)
        if estimator_cls is TabPFNClassifier
        else rng.standard_normal(N_ROWS)
    )
    y_train = pd.Series(values, name="target", index=[f"r{i}" for i in range(N_ROWS)])
    before = _snapshot(y_train)

    estimator_cls(n_estimators=2, device="cpu", random_state=0).fit(
        X_train, y_train
    ).predict(X_test)

    _assert_unchanged(y_train, before, "y_train")


def test__fit__copies_before_anything_writes_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline must copy before the first in-place writer sees the data.

    Comparing the caller's array before and after `fit` is not enough to pin this.
    `_extract_inf_masks` NaNs infinities out in place and `_restore_inf_masks`
    writes them back at the end, so a missing copy corrupts the caller's array for
    the duration of the call and then tidies up after itself -- measured, on a
    build with the copy removed: `user_infs 2 -> 0` inside `_process_steps`, and
    `infs 2` again by the time `fit` returns. Every end-to-end check in this module
    passes against that build.

    So watch the boundary instead of the outcome: whatever `_process_steps` hands
    to the first in-place writer must not be the caller's object. That also covers
    the case the round trip hides -- a step raising between the two, which would
    leave the caller holding NaN where their infinities were.
    """
    X_train = _numeric_with_non_finite()[:N_ROWS].copy()
    y_train = np.random.default_rng(5).integers(0, 3, N_ROWS)

    seen: dict[str, bool] = {}
    real_extract = pipeline_interface._extract_inf_masks

    def spy(X: np.ndarray, feature_schema: Any) -> Any:
        seen["ran"] = True
        # `is`, not `shares_memory`: a view shares memory with its base, so the
        # weaker question cannot tell a copy from a slice of the original.
        seen["got_callers_array"] = X is X_train
        return real_extract(X, feature_schema)

    monkeypatch.setattr(pipeline_interface, "_extract_inf_masks", spy)

    TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        random_state=0,
        # Pinned rather than defaulted: dispatching preprocessing to worker
        # processes would put the writes in a child, where they could not reach
        # the caller's array whether the copy was there or not.
        n_preprocessing_jobs=1,
        inference_config=PASSTHROUGH_INF,
    ).fit(X_train, y_train)

    assert seen.get("ran"), (
        "the in-place writer this guards never ran, so the test proved nothing; "
        "point it at whatever replaced `_extract_inf_masks`"
    )
    assert seen["got_callers_array"] is False
