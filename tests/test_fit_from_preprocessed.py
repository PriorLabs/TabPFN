#  Copyright (c) Prior Labs GmbH 2026.

"""The contract of a custom fine-tuning loop built on `fit_from_preprocessed`.

`forward` after `fit_from_preprocessed` must depend only on the batch. An estimator
that has run `fit()` or an earlier batch must give the same output as a fresh one.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.finetuning.data_util import (
    ClassifierBatch,
    RegressorBatch,
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)

ModelType = Literal["classifier", "regressor"]
History = Literal["fit", "fit_from_preprocessed"]


def _make_estimator(model_type: ModelType) -> TabPFNClassifier | TabPFNRegressor:
    # Majority downsampling makes `fit()` store a prior-shift correction, one more
    # piece of per-dataset state that must not reach the fine-tuning batch.
    kwargs = {
        "n_estimators": 2,
        "device": "cpu",
        "random_state": 0,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": 90,
            "SAMPLE_SUBSAMPLING_METHOD": "majority_downsample",
        },
    }
    if model_type == "classifier":
        return TabPFNClassifier(**kwargs)
    return TabPFNRegressor(**kwargs)


def _dataset(
    model_type: ModelType, *, seed: int, n_features: int, y_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """A dataset whose most frequent target value is 0, so downsampling applies."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, n_features))
    is_rare = rng.random(200) < 0.15
    X[is_rare] += 1.0
    if model_type == "classifier":
        return X, is_rare.astype(int)
    y = np.where(is_rare, y_scale * (rng.exponential(size=200) + 0.5), 0.0)
    return X, y


def _batch(
    estimator: TabPFNClassifier | TabPFNRegressor,
    X: np.ndarray,
    y: np.ndarray,
    model_type: ModelType,
) -> ClassifierBatch | RegressorBatch:
    chunks = get_preprocessed_dataset_chunks(
        estimator,
        X,
        y,
        train_test_split,
        100,
        model_type=model_type,
        equal_split_size=True,
        data_shuffle_seed=42,
        preprocessing_random_state=42,
    )
    return next(
        iter(DataLoader(chunks, batch_size=1, collate_fn=meta_dataset_collator))
    )


def _forward_on_batch(
    estimator: TabPFNClassifier | TabPFNRegressor,
    batch: ClassifierBatch | RegressorBatch,
) -> list[torch.Tensor]:
    """Run one step of a user-written fine-tuning loop and return its outputs.

    This is all a custom loop does before `forward`. A change that needs more
    steps here breaks every such loop outside this repository.
    """
    estimator.fit_from_preprocessed(
        batch.X_context,
        batch.y_context,
        batch.cat_indices,
        batch.configs,
        performance_options=PerformanceOptions(),
    )
    if isinstance(estimator, TabPFNClassifier):
        return [estimator.forward(batch.X_query).detach()]

    assert isinstance(batch, RegressorBatch)
    estimator.znorm_space_bardist_ = batch.znorm_space_bardist
    estimator.raw_space_bardist_ = batch.raw_space_bardist
    _, outputs, borders = estimator.forward(batch.X_query)
    return [o.detach() for o in outputs] + [torch.as_tensor(b) for b in borders]


@pytest.mark.parametrize("model_type", ["classifier", "regressor"])
@pytest.mark.parametrize("history", ["fit", "fit_from_preprocessed"])
def test__fit_from_preprocessed__earlier_use__output_matches_fresh_estimator(
    model_type: ModelType, history: History
) -> None:
    X, y = _dataset(model_type, seed=0, n_features=4, y_scale=1.0)
    # Different width, seed and target scale, so any leaked state shows.
    X_other, y_other = _dataset(model_type, seed=1, n_features=6, y_scale=1e3)

    fresh = _make_estimator(model_type)
    batch = _batch(fresh, X, y, model_type)
    expected = _forward_on_batch(fresh, batch)

    used = _make_estimator(model_type)
    if history == "fit":
        used.fit(X_other, y_other)
    else:
        _forward_on_batch(used, _batch(used, X_other, y_other, model_type))
    got = _forward_on_batch(used, batch)

    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        torch.testing.assert_close(g, e)
