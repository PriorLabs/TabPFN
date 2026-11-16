#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import warnings
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


def _aggregator(*, append: bool = True, pool: bool = False, **kwargs: int) -> Any:
    return GroupAggregator(append=append, pool=pool, **kwargs)


def test__one_row_per_group__adds_nothing_and_warns() -> None:
    X = _frame()

    with pytest.warns(UserWarning, match="APPEND_GROUP_AGGREGATES"):
        out = _aggregator().fit_transform(
            X, [0, 1, 2, 3, 4], np.arange(5), exclude=None
        )

    assert out is X


def test__label_varying_in_a_group__still_gets_aggregates() -> None:
    aggregator = _aggregator()

    aggregator.fit_transform(_frame(), [0, 1, 1, 1, 1], GROUPS, exclude=None)

    assert "num_mean" in aggregator.names_


@pytest.mark.parametrize(
    ("y", "groups", "pooled"),
    [
        ([0, 0, 1, 1, 1], GROUPS, True),
        (np.array([[0, 0, 1, 1, 1]]).T, GROUPS, True),
        ([0, 1, 1, 1, 1], GROUPS, False),
        ([0, 1, 2, 3, 4], np.arange(5), False),
        ([0, 0], GROUPS, False),
    ],
    ids=["one_label", "column_vector", "label_varies", "one_row", "wrong_length"],
)
def test__pooling__needs_one_label_per_group_and_warns_without(
    y: Any, groups: np.ndarray, pooled: bool, caplog: pytest.LogCaptureFixture
) -> None:
    aggregator = _aggregator(append=False, pool=True)

    with warnings.catch_warnings(record=True) as caught, caplog.at_level("INFO"):
        warnings.simplefilter("always")
        aggregator.fit_transform(_frame(), y, groups, exclude=None)

    assert aggregator.pool_ is pooled
    assert ("one label" in caplog.text) is pooled
    assert any("POOL_GROUP_PREDICTIONS" in str(w.message) for w in caught) is not pooled


def test__aggregates__per_group_values() -> None:
    aggregator = _aggregator(max_features=100)

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

    aggregator = _aggregator(max_features=3)
    aggregator.fit_transform(X, [0, 0, 1, 1], np.array([0, 0, 1, 1]), exclude=None)

    # Every mean, min, max and last has the same variance; std has none.
    assert aggregator.names_ == ["a_last", "a_max", "a_mean"]


def test__exclude__is_not_aggregated() -> None:
    aggregator = _aggregator()

    aggregator.fit_transform(_frame(), [0, 0, 1, 1, 1], GROUPS, exclude="cat")

    assert all(name.startswith("num_") for name in aggregator.names_)


def test__transform__aggregates_over_the_rows_passed_in() -> None:
    aggregator = _aggregator()
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

    out = pool_rows(values, np.array([0, 1, 0]), logits=False, geometric=False)

    expected = torch.tensor([[[3.0], [7.0], [3.0]], [[2.0], [9.0], [2.0]]])
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    ("geometric", "expected"), [(False, [0.7, 0.3]), (True, [0.75, 0.25])]
)
@pytest.mark.parametrize("logits", [True, False])
def test__pool_rows__pools_logits_and_probabilities_alike(
    geometric: bool, expected: list[float], logits: bool
) -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    values = probabilities.log() + 3.0 if logits else probabilities

    out = pool_rows(values, np.array([0, 0]), logits=logits, geometric=geometric)

    out = out.softmax(dim=-1) if logits else out
    torch.testing.assert_close(out, torch.tensor([expected, expected]))


@pytest.mark.parametrize("logits", [True, False])
def test__pool_rows__geometric_mean_of_rows_with_disjoint_support__is_finite(
    logits: bool,
) -> None:
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)
    values = probabilities.log() if logits else probabilities

    out = pool_rows(values, np.array([0, 0]), logits=logits, geometric=True)

    out = out.softmax(dim=-1) if logits else out
    torch.testing.assert_close(out, torch.full((2, 2), 0.5))


def _grouped_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 4)
    return pd.DataFrame({"num": rng.normal(size=40), "g": groups}), groups % 2


def _fit(estimator_cls: type, y: np.ndarray, **inference_config: Any) -> Any:
    X, _ = _grouped_data()
    return estimator_cls(
        n_estimators=1, group_col="g", inference_config=inference_config
    ).fit(X, y)


def _pooled(pred: np.ndarray) -> bool:
    by_group = pred.reshape(10, 4, -1)
    # float32 decoding of equal logits differs in the last digits near zero.
    return np.allclose(by_group, by_group[:, :1], rtol=1e-5, atol=1e-6)


ESTIMATORS = pytest.mark.parametrize(
    ("estimator_cls", "method"),
    [(TabPFNClassifier, "predict_proba"), (TabPFNRegressor, "predict")],
)


@ESTIMATORS
@pytest.mark.parametrize(
    ("append", "pool"), [(True, None), (False, "mean"), (False, "geometric_mean")]
)
def test__estimator__each_flag_switches_on_its_own_mode(
    estimator_cls: type, method: str, append: bool, pool: str | None
) -> None:
    X, y = _grouped_data()
    estimator = _fit(
        estimator_cls, y, APPEND_GROUP_AGGREGATES=append, POOL_GROUP_PREDICTIONS=pool
    )

    pred = getattr(estimator, method)(X)

    names = [f.name for f in estimator.inferred_feature_schema_.features]
    assert ("input_num_mean" in names) is append
    assert estimator.group_feature_index_ is not None
    assert _pooled(pred) is (pool is not None)
    assert len(np.unique(pred.round(5), axis=0)) > 1


@ESTIMATORS
def test__estimator__several_labels_in_a_group__get_aggregates_but_no_pooling(
    estimator_cls: type, method: str
) -> None:
    X, _ = _grouped_data()
    with pytest.warns(UserWarning, match="POOL_GROUP_PREDICTIONS"):
        estimator = _fit(
            estimator_cls,
            np.arange(40) % 2,
            APPEND_GROUP_AGGREGATES=True,
            POOL_GROUP_PREDICTIONS="mean",
        )

    assert "num_mean" in estimator.group_aggregator_.names_
    assert not _pooled(getattr(estimator, method)(X))


@pytest.mark.parametrize("pool", ["mean", "geometric_mean"])
def test__classifier__pooled_logits_match_the_pooled_probabilities(pool: str) -> None:
    X, y = _grouped_data()
    classifier = _fit(TabPFNClassifier, y, POOL_GROUP_PREDICTIONS=pool)

    from_logits = torch.as_tensor(classifier.predict_logits(X)).softmax(dim=-1)

    np.testing.assert_allclose(from_logits, classifier.predict_proba(X), atol=1e-5)
