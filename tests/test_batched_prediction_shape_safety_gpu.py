#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.finetuning.data_util import _group_batches_by_shape

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _features() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    constant = rng.normal(size=(60, 12)).astype(np.float32)
    constant[:, :5] = 0
    dense = rng.normal(size=(60, 12)).astype(np.float32)
    return [constant, dense, constant.copy()]


def _assert_two_stable_groups(spy: mock.Mock) -> None:
    groups = _group_batches_by_shape(spy.call_args.args[0])
    assert [[index for index, _ in group] for group in groups] == [[0, 2], [1]]


def test_classifier_heterogeneous_widths_match_serial() -> None:
    features = _features()
    labels = np.tile(np.arange(3), 20)
    kwargs = {
        "n_estimators": 2,
        "device": "cuda",
        "random_state": 42,
        "inference_precision": torch.float32,
    }
    with mock.patch(
        "tabpfn.finetuning.data_util._group_batches_by_shape",
        wraps=_group_batches_by_shape,
    ) as spy:
        batched = TabPFNClassifier(**kwargs).predict_proba_batched(
            features, [labels] * 3, [X[:6] for X in features]
        )
    _assert_two_stable_groups(spy)
    for index in range(2):
        reference = TabPFNClassifier(**kwargs).fit(features[index], labels)
        np.testing.assert_allclose(
            batched[index], reference.predict_proba(features[index][:6]), atol=2e-3
        )
    np.testing.assert_allclose(batched[0], batched[2], atol=2e-3)


def test_regressor_heterogeneous_widths_match_serial() -> None:
    features = _features()
    targets = [X[:, 5] - X[:, 6] for X in features]
    kwargs = {
        "n_estimators": 2,
        "device": "cuda",
        "random_state": 42,
        "inference_precision": torch.float32,
    }
    with mock.patch(
        "tabpfn.finetuning.data_util._group_batches_by_shape",
        wraps=_group_batches_by_shape,
    ) as spy:
        batched = TabPFNRegressor(**kwargs).predict_batched(
            features, targets, [X[:6] for X in features]
        )
    _assert_two_stable_groups(spy)
    for index in range(2):
        reference = TabPFNRegressor(**kwargs).fit(features[index], targets[index])
        np.testing.assert_allclose(
            batched[index], reference.predict(features[index][:6]), atol=2e-2
        )
    np.testing.assert_allclose(batched[0], batched[2], atol=2e-2)
