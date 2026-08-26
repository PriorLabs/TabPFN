#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `expand_date_features`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.clean import clean_data
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.date_encoding import expand_date_features

N = 20


def _numeric_and_date_schema() -> FeatureSchema:
    return FeatureSchema(
        features=[
            Feature(name="input_num", modality=FeatureModality.NUMERICAL),
            Feature(name="input_signed_on", modality=FeatureModality.DATE),
        ]
    )


def _numeric_and_date_frame(dates: list[str]) -> np.ndarray:
    return np.column_stack(
        [np.arange(len(dates), dtype=object), np.array(dates, dtype=object)]
    )


def _dates(n: int = N) -> list[str]:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return dates.strftime("%Y-%m-%d").tolist()


def test__no_date_columns__is_a_noop() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    schema = FeatureSchema(
        features=[
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.NUMERICAL),
        ]
    )
    X_out, schema_out, fitted = expand_date_features(X, schema)
    assert X_out is X
    assert schema_out is schema
    assert fitted == {}


def test__fit__removes_raw_column_and_appends_numeric_features() -> None:
    X = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()

    X_out, schema_out, fitted = expand_date_features(X, schema)

    assert schema_out is not None
    assert schema_out.indices_for(FeatureModality.DATE) == []
    assert all(f.modality is FeatureModality.NUMERICAL for f in schema_out.features)
    assert schema_out.num_columns == X_out.shape[1]
    assert X_out.shape[0] == N
    # More than just the original numeric column survives: the date expanded.
    assert X_out.shape[1] > 2

    assert list(fitted) == [1]
    fitted_encoder = fitted[1]
    assert len(fitted_encoder.output_names) == X_out.shape[1] - 1
    assert all(
        name.startswith("input_signed_on_") for name in fitted_encoder.output_names
    )
    # Every expanded feature is real-valued for a fully populated date column.
    assert np.isfinite(X_out[:, 1:].astype(float)).all()


def test__fit__output_names_avoid_collision_with_existing_columns() -> None:
    """A pre-existing column can happen to look like a generated output name."""
    X = np.column_stack([np.array(_dates(), dtype=object), np.zeros(N, dtype=object)])
    schema = FeatureSchema(
        features=[
            Feature(name="input_signed_on", modality=FeatureModality.DATE),
            Feature(name="input_signed_on_0", modality=FeatureModality.NUMERICAL),
        ]
    )

    _, schema_out, fitted = expand_date_features(X, schema)

    names = schema_out.feature_names
    assert len(names) == len(set(names))
    assert "input_signed_on_0" in names
    assert fitted[0].output_names[0] != "input_signed_on_0"


def test__expand_before_clean__vs__clean_before_expand() -> None:
    """Ordering guard: `clean_data` has no notion of `DATE` at all.

    Expand-then-clean (what fit/predict actually do) keeps the real date
    value: the year feature reads 2020. Clean-before-expand -- swapping the
    two calls -- doesn't error; `clean_data` just doesn't recognize the
    still-`DATE`-tagged column as anything special and silently ordinal-codes
    it like any other string column, replacing the dates with 0, 1, 2, ...
    """
    X = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()

    # Correct order: expand, then clean.
    X_expanded, schema_expanded, _ = expand_date_features(X, schema)
    X_right_order, _, schema_right_order = clean_data(X_expanded, schema_expanded)
    year_index = schema_right_order.feature_names.index("input_signed_on_0")
    np.testing.assert_array_equal(X_right_order[:, year_index], 2020.0)

    # Swapped order: clean first, on the still-DATE-tagged column.
    X_wrong_order, _, _ = clean_data(X, schema)
    date_column_index = 1  # unchanged: clean_data never expands/removes columns
    assert not np.allclose(X_wrong_order[:, date_column_index], 2020.0)
    np.testing.assert_array_equal(
        X_wrong_order[:, date_column_index], np.arange(N, dtype=float)
    )


def test__predict__reapplies_fitted_encoder_positionally() -> None:
    X_fit = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()
    X_fit_out, _, fitted = expand_date_features(X_fit, schema)

    X_test = _numeric_and_date_frame(_dates())
    X_test_out, schema_out, fitted_out = expand_date_features(
        X_test, feature_schema=None, fitted=fitted
    )

    assert schema_out is None
    assert fitted_out is fitted
    assert X_test_out.shape[1] == X_fit_out.shape[1]
    # Same dates in, same encoded values out.
    np.testing.assert_array_equal(
        X_test_out[:, 1:].astype(float), X_fit_out[:, 1:].astype(float)
    )


def test__predict__value_that_no_longer_parses_as_a_date__becomes_nan() -> None:
    """A predict-time column can drift dtype like any other fitted column; a
    value that no longer parses coerces to NaN rather than crashing.
    """
    X_fit = _numeric_and_date_frame(_dates())
    schema = _numeric_and_date_schema()
    _, _, fitted = expand_date_features(X_fit, schema)

    dates = _dates()
    dates[3] = "not a date at all"
    X_test = _numeric_and_date_frame(dates)

    X_test_out, _, _ = expand_date_features(X_test, feature_schema=None, fitted=fitted)

    assert np.isnan(X_test_out[3, 1:].astype(float)).all()
    # Every other row is unaffected.
    other_rows = [i for i in range(N) if i != 3]
    assert np.isfinite(X_test_out[other_rows][:, 1:].astype(float)).all()


def test__mixed_date_and_datetime_string_formats__all_parse() -> None:
    """Regression: naive `pd.to_datetime` infers a format from an early value
    and silently coerces a later, differently-shaped but valid value to NaT.
    Verified directly: mixing "2020-01-01" and "2020-06-15 13:45:30" drops the
    second to NaT under the default (non-"mixed") format inference.
    """
    dates = ["2020-01-01", "2020-06-15 13:45:30", "2021-12-31 23:59:59"]
    X = _numeric_and_date_frame(dates)
    schema = _numeric_and_date_schema()

    X_out, _, _ = expand_date_features(X, schema)

    assert np.isfinite(X_out[:, 1:].astype(float)).all()


def _classification_or_regression_target(
    estimator_cls: type, rng: np.random.Generator, n: int
) -> np.ndarray:
    if estimator_cls is TabPFNClassifier:
        return rng.integers(0, 2, size=n)
    return rng.normal(size=n)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_predict__use_dates__expands_date_and_predicts(
    estimator_cls: type,
) -> None:
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D").strftime(
                "%Y-%m-%d"
            ),
        }
    )
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"USE_DATES": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X, y)

    assert model.inferred_feature_schema_.indices_for(FeatureModality.DATE) == []
    assert 1 in model.date_encoders_  # "signed_on" is the second input column

    if estimator_cls is TabPFNClassifier:
        out = model.predict_proba(X)
    else:
        out = model.predict(X)
    assert np.isfinite(out).all()


def test__predict_proba_batched__use_dates__reapplies_encoder_on_worker() -> None:
    """The ensemble-worker predict path also reapplies the fitted date encoder,
    not just the direct-`self` path.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D").strftime(
                "%Y-%m-%d"
            ),
        }
    ).to_numpy(dtype=object)
    y = rng.integers(0, 2, size=n)

    clf = TabPFNClassifier(
        n_estimators=1, device="cpu", inference_config={"USE_DATES": True}
    )
    proba = clf.predict_proba_batched([X], [y], [X[:5]])
    assert proba.shape == (1, 5, 2)
    assert np.isfinite(proba).all()
