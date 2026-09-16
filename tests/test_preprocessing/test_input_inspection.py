#  Copyright (c) Prior Labs GmbH 2026.
"""Tests for the inspection fit keeps of its input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from tabpfn.preprocessing.clean import PANDAS_BELOW_3
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.input_inspection import (
    MAX_EXAMPLE_LENGTH,
    MAX_EXAMPLES,
    InputInspection,
    build_input_inspection,
)
from tabpfn.preprocessing.modality_detection import (
    ModalityDecision,
    detect_feature_modalities,
)


def _mixed_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame(
        {
            "amount": rng.normal(size=n),
            "level": pd.Series(np.arange(n) % 3).astype("category"),
            "city": np.where(
                np.arange(n) % 7 == 0, None, [f"c{i % 8}" for i in range(n)]
            ),
            "note": [
                f"quite a long free text value number {i} in this cell"
                for i in range(n)
            ],
            "empty": [np.nan] * n,
        }
    )


def _decisions(X: pd.DataFrame | np.ndarray) -> list[ModalityDecision]:
    values = X.to_numpy(dtype=object) if isinstance(X, pd.DataFrame) else X
    _, decisions = detect_feature_modalities(
        values,
        feature_names=None,
        min_samples_for_inference=100,
        max_unique_for_category=30,
        min_unique_for_numerical=4,
        min_cardinality_for_text=30,
    )
    return decisions


def test__build_input_inspection__mixed_frame__keeps_every_fact():
    X = _mixed_frame()

    inspection = build_input_inspection(X, decisions=_decisions(X))

    assert inspection.labels == ("amount", "level", "city", "note", "empty")
    string = "object" if PANDAS_BELOW_3 else "str"
    assert inspection.dtypes == ("float64", "category", string, string, "float64")
    assert inspection.category_dtype_positions == (1,)
    assert len(inspection.decisions) == X.shape[1]
    amount, level, city, note, empty = inspection.examples
    assert len(amount) == MAX_EXAMPLES
    assert set(level) == {"0", "1", "2"}
    assert city[:2] == ("'c1'", "'c2'")
    assert None not in city
    assert all(len(value) == MAX_EXAMPLE_LENGTH for value in note)
    assert empty == ()


def test__build_input_inspection__array_input__has_positional_facts():
    X = np.column_stack([np.arange(10, dtype=float), np.arange(10) % 2])

    inspection = build_input_inspection(X, decisions=_decisions(X))

    assert inspection.labels is None
    assert inspection.dtypes == ("float64", "float64")
    assert inspection.category_dtype_positions == ()
    assert inspection.examples[1] == ("0.0", "1.0")
    assert inspection.examples[0] == ("0.0", "1.0", "2.0", "3.0", "4.0")


def test__build_input_inspection__list_input__has_no_dtypes():
    X = [[1.0, "a"], [2.0, "b"], [3.0, "a"]]

    inspection = build_input_inspection(
        X, decisions=_decisions(np.asarray(X, dtype=object))
    )

    assert inspection.labels is None
    assert inspection.dtypes is None
    assert inspection.examples == (("1.0", "2.0", "3.0"), ("'a'", "'b'"))


def test__build_input_inspection__integer_labels__are_kept_as_text():
    X = pd.DataFrame(np.arange(20, dtype=float).reshape(10, 2))

    inspection = build_input_inspection(X, decisions=_decisions(X))

    assert inspection.labels == ("0", "1")


def _fit_data(estimator_cls: type) -> tuple[pd.DataFrame, np.ndarray]:
    X = _mixed_frame()
    rng = np.random.default_rng(seed=1)
    n = len(X)
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )
    return X, y


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__sets_input_inspection__matching_the_input_and_the_schema(
    estimator_cls: type,
) -> None:
    X, y = _fit_data(estimator_cls)

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.warns(UserWarning, match="look like free text"):
        model.fit(X, y)

    inspection = model.input_inspection_
    assert isinstance(inspection, InputInspection)
    assert inspection.labels == tuple(X.columns)
    assert inspection.category_dtype_positions == (1,)
    assert len(inspection.decisions) == model.inferred_feature_schema_.num_columns
    assert [d.modality for d in inspection.decisions] == [
        f.modality for f in model.inferred_feature_schema_.features
    ]
    assert inspection.decisions[3].modality is FeatureModality.TEXT
    assert inspection.decisions[4].modality is FeatureModality.CONSTANT


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit__with_transform_dates__decisions_cover_the_expanded_columns(
    estimator_cls: type,
) -> None:
    """The decisions describe the input after expansion, so an expanded datetime
    column contributes one decision per generated feature, not one for itself.
    """
    X, y = _fit_data(estimator_cls)
    X = X.assign(when=pd.date_range("2024-01-01", periods=len(X), freq="D"))

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    with pytest.warns(UserWarning, match="look like free text"):
        model.fit(X, y)

    inspection = model.input_inspection_
    assert inspection.labels == tuple(X.columns)
    assert inspection.dtypes[-1].startswith("datetime64")
    assert inspection.examples[-1][0] == "2024-01-01 00:00:00"
    assert len(inspection.decisions) == model.inferred_feature_schema_.num_columns
    assert len(inspection.decisions) > X.shape[1]


def test__save_and_load_fitted_model__keeps_input_inspection(tmp_path: Path) -> None:
    X, y = _fit_data(TabPFNClassifier)
    model = TabPFNClassifier(n_estimators=1, device="cpu")
    with pytest.warns(UserWarning, match="look like free text"):
        model.fit(X, y)

    save_fitted_tabpfn_model(model, tmp_path / "model.tabpfn_fit")
    loaded = load_fitted_tabpfn_model(tmp_path / "model.tabpfn_fit", device="cpu")

    assert loaded.input_inspection_ == model.input_inspection_
