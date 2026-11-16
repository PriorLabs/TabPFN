#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.group_aggregates import GroupAggregator, pool_rows
from tabpfn.preprocessing.group_ids import GroupIdEncoder

GROUPS = np.array([0, 0, 1, 1, 1])


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"num": [1.0, 3.0, 5.0, np.nan, 2.0], "cat": ["a", "b", "a", "a", None]}
    )


@pytest.mark.parametrize(
    ("y", "groups"),
    [([0, 1, 1, 1, 1], GROUPS), ([0, 1, 2, 3, 4], np.arange(5))],
    ids=["label_varies_in_a_group", "one_row_per_group"],
)
def test__no_one_label_per_group__adds_nothing(y: list, groups: np.ndarray) -> None:
    X = _frame()

    out = GroupAggregator().fit_transform(X, y, groups, exclude=None)

    assert out is X


def test__aggregates__per_group_values() -> None:
    aggregator = GroupAggregator(max_features=100)

    out = aggregator.fit_transform(_frame(), [0, 0, 1, 1, 1], GROUPS, exclude=None)

    expected = {
        "num_mean": [2.0, 2.0, 3.5, 3.5, 3.5],
        "num_min": [1.0, 1.0, 2.0, 2.0, 2.0],
        "num_max": [3.0, 3.0, 5.0, 5.0, 5.0],
        "num_last": [3.0, 3.0, 2.0, 2.0, 2.0],
        "num_std": [np.sqrt(2), np.sqrt(2), np.sqrt(4.5), np.sqrt(4.5), np.sqrt(4.5)],
        "cat_count": [2, 2, 2, 2, 2],
        "cat_last": ["b", "b", "a", "a", "a"],
        "cat_nunique": [2, 2, 1, 1, 1],
    }
    assert sorted(aggregator.names_) == sorted(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(out[name].to_numpy(), values, err_msg=name)


def test__selection__keeps_the_largest_variance_and_breaks_ties_by_name() -> None:
    X = pd.DataFrame({"b": [0.0, 0.0, 9.0, 9.0], "a": [0.0, 0.0, 9.0, 9.0]})

    aggregator = GroupAggregator(max_features=3)
    aggregator.fit_transform(X, [0, 0, 1, 1], np.array([0, 0, 1, 1]), exclude=None)

    # Every mean, min, max and last has the same variance; std has none.
    assert aggregator.names_ == ["a_last", "a_max", "a_mean"]


def test__exclude__is_not_aggregated() -> None:
    aggregator = GroupAggregator()

    aggregator.fit_transform(_frame(), [0, 0, 1, 1, 1], GROUPS, exclude="cat")

    assert all(name.startswith("num_") for name in aggregator.names_)


def test__transform__aggregates_over_the_rows_passed_in() -> None:
    aggregator = GroupAggregator()
    aggregator.fit_transform(_frame(), [0, 0, 1, 1, 1], GROUPS, exclude=None)
    X_test = pd.DataFrame({"num": [10.0, 20.0, 7.0], "cat": ["c", "c", "d"]})

    out = aggregator.transform(X_test, np.array([0, 0, 1]))

    np.testing.assert_array_equal(out["num_mean"], [15.0, 15.0, 7.0])


def test__row_groups__codes_composite_keys_within_the_input() -> None:
    X = pd.DataFrame({"site": ["s", "s", "t"], "patient": [1, 2, 1], "x": [0, 0, 0]})
    encoder = GroupIdEncoder(group_col=["site", "patient"]).fit(X)
    X_new = pd.DataFrame(
        {"site": ["u", "s", "u"], "patient": [9, 1, 9], "x": [0, 0, 0]}
    )

    np.testing.assert_array_equal(encoder.row_groups(X_new), [0, 1, 0])


def test__pool_rows__replaces_each_row_by_its_group_mean() -> None:
    values = torch.tensor([[[1.0], [7.0], [5.0]], [[0.0], [9.0], [4.0]]])

    out = pool_rows(values, np.array([0, 1, 0]))

    expected = torch.tensor([[[3.0], [7.0], [3.0]], [[2.0], [9.0], [2.0]]])
    torch.testing.assert_close(out, expected)


def _grouped_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 4)
    return pd.DataFrame({"num": rng.normal(size=40), "g": groups}), groups % 2


@pytest.mark.parametrize(
    ("estimator_cls", "method"),
    [(TabPFNClassifier, "predict_proba"), (TabPFNRegressor, "predict")],
)
def test__estimator__appends_aggregates_and_pools_the_predictions_of_a_group(
    estimator_cls: type, method: str
) -> None:
    X, y = _grouped_data()
    estimator = estimator_cls(
        n_estimators=1,
        group_col="g",
        inference_config={"USE_GROUP_AGGREGATES": True},
    ).fit(X, y)

    by_group = getattr(estimator, method)(X).reshape(10, 4, -1)

    names = [f.name for f in estimator.inferred_feature_schema_.features]
    assert "input_num_mean" in names
    assert estimator.group_feature_index_ is not None
    np.testing.assert_allclose(by_group, by_group[:, :1].repeat(4, axis=1), rtol=1e-5)
    assert len(np.unique(by_group[:, 0].round(5), axis=0)) > 1


@pytest.mark.parametrize("several_labels", [True, False], ids=["labels", "flag_off"])
@pytest.mark.parametrize(
    ("estimator_cls", "method"),
    [(TabPFNClassifier, "predict_proba"), (TabPFNRegressor, "predict")],
)
def test__estimator__is_unchanged_without_the_flag_or_one_label_per_group(
    estimator_cls: type, method: str, several_labels: bool
) -> None:
    X, y = _grouped_data()
    if several_labels:
        y = np.arange(40) % 2
    estimator = estimator_cls(
        n_estimators=1,
        group_col="g",
        inference_config={"USE_GROUP_AGGREGATES": several_labels},
    ).fit(X, y)

    pred = getattr(estimator, method)(X).reshape(10, 4, -1)

    assert estimator.group_aggregator_.names_ == []
    assert (pred != pred[:, :1]).any()
