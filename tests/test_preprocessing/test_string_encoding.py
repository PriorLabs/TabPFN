#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `expand_text_features`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.string_encoding import (
    DEFAULT_TEXT_N_COMPONENTS,
    expand_text_features,
)

N = 40


def _numeric_and_text_schema() -> FeatureSchema:
    return FeatureSchema(
        features=[
            Feature(name="input_num", modality=FeatureModality.NUMERICAL),
            Feature(name="input_review", modality=FeatureModality.TEXT),
        ]
    )


def _numeric_and_text_frame(texts: list) -> np.ndarray:
    return np.column_stack(
        [np.arange(len(texts), dtype=object), np.array(texts, dtype=object)]
    )


def _texts(n: int = N) -> list[str]:
    return [f"this is review number {i} about a product" for i in range(n)]


def test__no_text_columns__is_a_noop() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    schema = FeatureSchema(
        features=[
            Feature(name="a", modality=FeatureModality.NUMERICAL),
            Feature(name="b", modality=FeatureModality.NUMERICAL),
        ]
    )
    X_out, schema_out, fitted = expand_text_features(X, schema, use_text=True)
    assert X_out is X
    assert schema_out is schema
    assert fitted == {}


def test__use_text_false__is_a_noop_even_with_a_text_column() -> None:
    """A `TEXT` tag survives on the schema regardless of the flag (unlike
    `DATE`), so `use_text` has to gate fitting here directly.
    """
    X = _numeric_and_text_frame(_texts())
    schema = _numeric_and_text_schema()

    X_out, schema_out, fitted = expand_text_features(X, schema, use_text=False)

    assert X_out is X
    assert schema_out is schema
    assert fitted == {}


def test__fit__removes_raw_column_and_appends_numeric_features() -> None:
    X = _numeric_and_text_frame(_texts())
    schema = _numeric_and_text_schema()

    X_out, schema_out, fitted = expand_text_features(X, schema, use_text=True)

    assert schema_out is not None
    assert schema_out.indices_for(FeatureModality.TEXT) == []
    assert all(f.modality is FeatureModality.NUMERICAL for f in schema_out.features)
    assert schema_out.num_columns == X_out.shape[1]
    assert X_out.shape[0] == N

    assert list(fitted) == [1]
    fitted_encoder = fitted[1]
    assert len(fitted_encoder.output_names) == DEFAULT_TEXT_N_COMPONENTS
    assert all(name.startswith("input_review_") for name in fitted_encoder.output_names)
    assert np.isfinite(X_out[:, 1:].astype(float)).all()


def test__fit__mixed_dtype_column__stringifies_and_fits() -> None:
    values: list = list(_texts())
    values[3] = 123
    values[7] = None
    values[11] = 4.5
    X = _numeric_and_text_frame(values)
    schema = _numeric_and_text_schema()

    X_out, schema_out, fitted = expand_text_features(X, schema, use_text=True)

    assert schema_out.indices_for(FeatureModality.TEXT) == []
    assert X_out.shape[0] == N
    assert list(fitted) == [1]


def test__fit__output_names_avoid_collision_with_existing_columns() -> None:
    """A pre-existing column can happen to look like a generated output name."""
    X = np.column_stack([np.array(_texts(), dtype=object), np.zeros(N, dtype=object)])
    schema = FeatureSchema(
        features=[
            Feature(name="input_review", modality=FeatureModality.TEXT),
            Feature(name="input_review_0", modality=FeatureModality.NUMERICAL),
        ]
    )

    _, schema_out, fitted = expand_text_features(X, schema, use_text=True)

    names = schema_out.feature_names
    assert len(names) == len(set(names))
    assert "input_review_0" in names
    assert fitted[0].output_names[0] != "input_review_0"


def test__predict__reapplies_fitted_encoder_positionally() -> None:
    X_fit = _numeric_and_text_frame(_texts())
    schema = _numeric_and_text_schema()
    X_fit_out, _, fitted = expand_text_features(X_fit, schema, use_text=True)

    X_test = _numeric_and_text_frame(_texts())
    X_test_out, schema_out, fitted_out = expand_text_features(
        X_test, feature_schema=None, fitted=fitted
    )

    assert schema_out is None
    assert fitted_out is fitted
    assert X_test_out.shape[1] == X_fit_out.shape[1]
    np.testing.assert_array_equal(
        X_test_out[:, 1:].astype(float), X_fit_out[:, 1:].astype(float)
    )


def test__predict__unseen_and_missing_values__become_zero_vectors() -> None:
    """Unlike dates (which coerce a bad value to NaN), a `StringEncoder` never
    raises on drifted predict-time input -- every unrecognized row (missing,
    non-string) comes out as a zero vector, verified directly against skrub.
    """
    X_fit = _numeric_and_text_frame(_texts())
    schema = _numeric_and_text_schema()
    _, _, fitted = expand_text_features(X_fit, schema, use_text=True)

    test_values: list = _texts(5)
    test_values[1] = None
    test_values[2] = 123
    X_test = _numeric_and_text_frame(test_values)

    X_test_out, _, _ = expand_text_features(X_test, feature_schema=None, fitted=fitted)

    assert np.all(X_test_out[1, 1:].astype(float) == 0.0)
    assert np.isfinite(X_test_out[:, 1:].astype(float)).all()


def test__fit__all_token_empty_values__raises_empty_vocabulary() -> None:
    """Accepted gap: detection can't structurally rule this out (a `TEXT`
    column just needs enough unique values, not extractable tokens), and
    skrub raises outright rather than silently producing garbage.
    """
    values = [" " * (i + 1) for i in range(33)]
    X = _numeric_and_text_frame(values)
    schema = _numeric_and_text_schema()

    with pytest.raises(ValueError, match="empty vocabulary"):
        expand_text_features(X, schema, use_text=True)


def _classification_or_regression_target(
    estimator_cls: type, rng: np.random.Generator, n: int
) -> np.ndarray:
    if estimator_cls is TabPFNClassifier:
        return rng.integers(0, 2, size=n)
    return rng.normal(size=n)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_predict__use_text__expands_text_and_predicts(
    estimator_cls: type,
) -> None:
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"num": rng.normal(size=n), "review": _texts(n)})
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"USE_TEXT": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X, y)

    assert model.inferred_feature_schema_.indices_for(FeatureModality.TEXT) == []
    assert 1 in model.text_encoders_  # "review" is the second input column

    if estimator_cls is TabPFNClassifier:
        out = model.predict_proba(X)
    else:
        out = model.predict(X)
    assert np.isfinite(out).all()


def test__predict_proba_batched__use_text__reapplies_encoder_on_worker() -> None:
    """The ensemble-worker predict path also reapplies the fitted text
    encoder, not just the direct-`self` path.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"num": rng.normal(size=n), "review": _texts(n)}).to_numpy(
        dtype=object
    )
    y = rng.integers(0, 2, size=n)

    clf = TabPFNClassifier(
        n_estimators=1, device="cpu", inference_config={"USE_TEXT": True}
    )
    proba = clf.predict_proba_batched([X], [y], [X[:5]])
    assert proba.shape == (1, 5, 2)
    assert np.isfinite(proba).all()
