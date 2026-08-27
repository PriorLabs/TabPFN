#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `encode_multimodal_data` composing date and text expansion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.multimodal_encoding import encode_multimodal_data

N = 40


def _numeric_date_and_text_schema() -> FeatureSchema:
    return FeatureSchema(
        features=[
            Feature(name="input_num", modality=FeatureModality.NUMERICAL),
            Feature(name="input_signed_on", modality=FeatureModality.DATE),
            Feature(name="input_review", modality=FeatureModality.TEXT),
        ]
    )


def _numeric_date_and_text_frame(dates: list[str], texts: list[str]) -> np.ndarray:
    return np.column_stack(
        [
            np.arange(len(dates), dtype=object),
            np.array(dates, dtype=object),
            np.array(texts, dtype=object),
        ]
    )


def _dates(n: int = N) -> list[str]:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return dates.strftime("%Y-%m-%d").tolist()


def _texts(n: int = N) -> list[str]:
    return [f"this is review number {i} about a product" for i in range(n)]


def test__fit__both_flags_on__expands_dates_then_text() -> None:
    X = _numeric_date_and_text_frame(_dates(), _texts())
    schema = _numeric_date_and_text_schema()

    X_out, schema_out, date_encoders, text_encoders = encode_multimodal_data(
        X, schema, use_text=True
    )

    assert schema_out.indices_for(FeatureModality.DATE) == []
    assert schema_out.indices_for(FeatureModality.TEXT) == []
    assert all(f.modality is FeatureModality.NUMERICAL for f in schema_out.features)
    assert schema_out.num_columns == X_out.shape[1]

    # The date column (original index 1) expands first and is dropped, so the
    # text column (originally after it, at index 2) shifts left to fill the
    # gap before `expand_text_features` ever looks at indices.
    assert list(date_encoders) == [1]
    assert list(text_encoders) == [1]

    names = schema_out.feature_names
    assert len(names) == len(set(names))
    assert np.isfinite(X_out[:, 1:].astype(float)).all()


def test__fit__use_text_false__only_dates_expand() -> None:
    X = _numeric_date_and_text_frame(_dates(), _texts())
    schema = _numeric_date_and_text_schema()

    X_out, schema_out, date_encoders, text_encoders = encode_multimodal_data(
        X, schema, use_text=False
    )

    assert schema_out.indices_for(FeatureModality.DATE) == []
    # TEXT is untouched: still tagged TEXT, exactly as before this function ran.
    # The date column (index 1) was dropped and its outputs appended at the
    # end, so text (originally after it, at index 2) shifted left to index 1.
    assert schema_out.indices_for(FeatureModality.TEXT) == [1]
    assert date_encoders
    assert text_encoders == {}
    assert X_out.shape[1] == X.shape[1] - 1 + len(date_encoders[1].output_names)


def test__predict__reapplies_both_encoders_positionally() -> None:
    X_fit = _numeric_date_and_text_frame(_dates(), _texts())
    schema = _numeric_date_and_text_schema()
    X_fit_out, _, date_encoders, text_encoders = encode_multimodal_data(
        X_fit, schema, use_text=True
    )

    X_test = _numeric_date_and_text_frame(_dates(), _texts())
    X_test_out, schema_out, date_out, text_out = encode_multimodal_data(
        X_test,
        feature_schema=None,
        date_fitted=date_encoders,
        text_fitted=text_encoders,
    )

    assert schema_out is None
    assert date_out is date_encoders
    assert text_out is text_encoders
    assert X_test_out.shape == X_fit_out.shape
    np.testing.assert_array_equal(
        X_test_out[:, 1:].astype(float), X_fit_out[:, 1:].astype(float)
    )


def test__fit__text_output_names_avoid_collision_with_date_output_names() -> None:
    """A text column's generated name can collide with one the date encoder
    already produced for an earlier column; `existing_names` must carry over.

    Both columns share a name here, so both would naturally generate the same
    `f"{name}_{i}"` candidates -- the date encoder's outputs land first, so
    every colliding text-generated name must be deduped against them.
    """
    schema = FeatureSchema(
        features=[
            Feature(name="input_shared", modality=FeatureModality.DATE),
            Feature(name="input_shared", modality=FeatureModality.TEXT),
        ]
    )
    X = np.column_stack(
        [np.array(_dates(), dtype=object), np.array(_texts(), dtype=object)]
    )

    _, schema_out, date_encoders, text_encoders = encode_multimodal_data(
        X, schema, use_text=True
    )

    date_names = date_encoders[0].output_names
    # The date column (index 0) is dropped and its outputs appended at the end,
    # so the text column (originally at index 1) shifts left to fill the gap.
    text_names = text_encoders[0].output_names
    # Some of text's raw candidates (`input_shared_0`, ...) genuinely collide
    # with date's own outputs -- confirmed real, not merely asserted:
    raw_text_candidates = [f"input_shared_{i}" for i in range(len(text_names))]
    assert set(raw_text_candidates) & set(date_names)
    # Deduping resolves every one of them.
    assert not set(date_names) & set(text_names)
    all_names = schema_out.feature_names
    assert len(all_names) == len(set(all_names))
