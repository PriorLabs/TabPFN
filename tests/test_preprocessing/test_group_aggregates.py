#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

from typing import Any

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
    [([0, 1, 1, 1, 1], GROUPS), ([0, 1, 2, 3, 4], np.arange(5)), ([0, 0], GROUPS)],
    ids=["label_varies_in_a_group", "one_row_per_group", "y_of_another_length"],
)
def test__no_one_label_per_group__adds_nothing(y: list, groups: np.ndarray) -> None:
    X = _frame()

    out = GroupAggregator(append=True).fit_transform(X, y, groups, exclude=None)

    assert out is X


def test__column_vector_y__is_read_as_one_label_per_row() -> None:
    aggregator = GroupAggregator(append=True)

    aggregator.fit_transform(
        _frame(), np.array([[0, 0, 1, 1, 1]]).T, GROUPS, exclude=None
    )

    assert aggregator.one_label_per_group_


def test__aggregates__per_group_values() -> None:
    aggregator = GroupAggregator(append=True, max_features=100)

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

    aggregator = GroupAggregator(append=True, max_features=3)
    aggregator.fit_transform(X, [0, 0, 1, 1], np.array([0, 0, 1, 1]), exclude=None)

    # Every mean, min, max and last has the same variance; std has none.
    assert aggregator.names_ == ["a_last", "a_max", "a_mean"]


def test__exclude__is_not_aggregated() -> None:
    aggregator = GroupAggregator(append=True)

    aggregator.fit_transform(_frame(), [0, 0, 1, 1, 1], GROUPS, exclude="cat")

    assert all(name.startswith("num_") for name in aggregator.names_)


def test__transform__aggregates_over_the_rows_passed_in() -> None:
    aggregator = GroupAggregator(append=True)
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

    out = pool_rows(values, np.array([0, 1, 0]), logits=False)

    expected = torch.tensor([[[3.0], [7.0], [3.0]], [[2.0], [9.0], [2.0]]])
    torch.testing.assert_close(out, expected)


def test__pool_rows__averages_logits_as_probabilities() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.5, 0.5]])

    out = pool_rows(probabilities.log() + 3.0, np.array([0, 0]), logits=True)

    torch.testing.assert_close(out.exp(), torch.tensor([[0.7, 0.3], [0.7, 0.3]]))


def _grouped_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 4)
    return pd.DataFrame({"num": rng.normal(size=40), "g": groups}), groups % 2


def _fit(estimator_cls: type, y: np.ndarray, **inference_config: bool) -> Any:
    X, _ = _grouped_data()
    return estimator_cls(
        n_estimators=1, group_col="g", inference_config=inference_config
    ).fit(X, y)


def _pooled(pred: np.ndarray) -> bool:
    by_group = pred.reshape(10, 4, -1)
    return np.allclose(by_group, by_group[:, :1], rtol=1e-5)


ESTIMATORS = pytest.mark.parametrize(
    ("estimator_cls", "method"),
    [(TabPFNClassifier, "predict_proba"), (TabPFNRegressor, "predict")],
)


@ESTIMATORS
@pytest.mark.parametrize(("append", "pool"), [(True, False), (False, True)])
def test__estimator__each_flag_switches_on_its_own_mode(
    estimator_cls: type, method: str, append: bool, pool: bool
) -> None:
    X, y = _grouped_data()
    estimator = _fit(
        estimator_cls, y, APPEND_GROUP_AGGREGATES=append, POOL_GROUP_PREDICTIONS=pool
    )

    pred = getattr(estimator, method)(X)

    names = [f.name for f in estimator.inferred_feature_schema_.features]
    assert ("input_num_mean" in names) is append
    assert estimator.group_feature_index_ is not None
    assert _pooled(pred) is pool
    assert len(np.unique(pred.round(5), axis=0)) > 1


@ESTIMATORS
def test__estimator__several_labels_in_a_group__switch_both_modes_off(
    estimator_cls: type, method: str
) -> None:
    X, _ = _grouped_data()
    estimator = _fit(
        estimator_cls,
        np.arange(40) % 2,
        APPEND_GROUP_AGGREGATES=True,
        POOL_GROUP_PREDICTIONS=True,
    )

    assert estimator.group_aggregator_.names_ == []
    assert not _pooled(getattr(estimator, method)(X))


def test__classifier__pooled_logits_match_the_pooled_probabilities() -> None:
    X, y = _grouped_data()
    classifier = _fit(TabPFNClassifier, y, POOL_GROUP_PREDICTIONS=True)

    from_logits = torch.as_tensor(classifier.predict_logits(X)).softmax(dim=-1)

    np.testing.assert_allclose(from_logits, classifier.predict_proba(X), atol=1e-5)


def test__pool_rows__geometric__renormalised_geometric_mean() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    expected = (probabilities[0] * probabilities[1]).sqrt()
    expected = (expected / expected.sum()).expand(2, 2)

    from_probs = pool_rows(probabilities, np.array([0, 0]), logits=False, geometric=True)
    from_logits = pool_rows(
        probabilities.log() + 3.0, np.array([0, 0]), logits=True, geometric=True
    )

    torch.testing.assert_close(from_probs, expected)
    torch.testing.assert_close(from_logits.exp(), expected)


def test__classifier__geometric_pooling__logits_match_probabilities() -> None:
    X, y = _grouped_data()
    classifier = _fit(
        TabPFNClassifier, y, POOL_GROUP_PREDICTIONS=True, POOL_GROUP_RULE="geometric"
    )

    proba = classifier.predict_proba(X)
    from_logits = torch.as_tensor(classifier.predict_logits(X)).softmax(dim=-1)

    assert _pooled(proba)
    np.testing.assert_allclose(from_logits, proba, atol=1e-5)
