"""Module for fitting and transforming preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING, Literal

import joblib
import numpy as np
import torch

from tabpfn.constants import (
    PARALLEL_MODE_TO_RETURN_AS,
    SUPPORTS_RETURN_AS,
)
from tabpfn.preprocessing.ensemble import (
    ClassifierEnsembleConfig,
    RegressorEnsembleConfig,
)
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    from tabpfn.preprocessing.configs import EnsembleConfig
    from tabpfn.preprocessing.datamodel import FeatureSchema
    from tabpfn.preprocessing.pipeline_interface import PreprocessingPipeline


def _fit_preprocessing_one(
    config: EnsembleConfig,
    X_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    random_state: int | np.random.Generator | None = None,
    *,
    feature_schema: FeatureSchema,
) -> tuple[
    EnsembleConfig,
    PreprocessingPipeline,
    np.ndarray,
    np.ndarray,
    FeatureSchema,
]:
    """Fit preprocessing pipeline for a single ensemble configuration.

    Args:
        config: Ensemble configuration.
        X_train: Training data.
        y_train: Training target.
        random_state: Random seed.
        feature_schema: feature schema.

    Returns:
        Tuple containing the ensemble configuration, the fitted preprocessing pipeline,
        the transformed training data, the transformed target, and the indices of
        categorical features.
    """
    static_seed, _ = infer_random_state(random_state)
    if config.subsample_ix is not None:
        X_train = X_train[config.subsample_ix]
        y_train = y_train[config.subsample_ix]
    if not isinstance(X_train, torch.Tensor):
        X_train = X_train.copy()
        y_train = y_train.copy()

    preprocessor = create_preprocessing_pipeline(config, random_state=static_seed)
    res = preprocessor.fit_transform(X_train, feature_schema)

    y_train_processed = _transform_labels_one(config, y_train)

    return (
        config,
        preprocessor,
        res.X,
        y_train_processed,
        res.feature_schema,
    )


def _transform_labels_one(
    config: EnsembleConfig, y_train: np.ndarray | torch.Tensor
) -> np.ndarray:
    """Transform the labels for one ensemble config.
        for both regression or classification.

    Args:
        config: Ensemble config.
        y_train: The unprocessed labels.

    Return: The processed labels.
    """
    if isinstance(config, RegressorEnsembleConfig):
        if config.target_transform is not None:
            y_train = config.target_transform.fit_transform(
                y_train.reshape(-1, 1),
            ).ravel()
    elif isinstance(config, ClassifierEnsembleConfig):
        if config.class_permutation is not None:
            y_train = config.class_permutation[y_train]
    else:
        raise ValueError(f"Invalid ensemble config type: {type(config)}")
    return y_train


def fit_preprocessing(
    configs: Sequence[EnsembleConfig],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int | np.random.Generator | None,
    feature_schema: FeatureSchema,
    n_preprocessing_jobs: int,
    parallel_mode: Literal["block", "as-ready", "in-order"],
) -> Iterator[
    tuple[
        EnsembleConfig,
        PreprocessingPipeline,
        np.ndarray,
        np.ndarray,
        FeatureSchema,
    ]
]:
    """Fit preprocessing pipelines in parallel.

    Args:
        configs: List of ensemble configurations.
        X_train: Training data.
        y_train: Training target.
        random_state: Random number generator.
        feature_schema: feature schema.
        n_preprocessing_jobs: Number of worker processes to use.
            If `1`, then the preprocessing is performed in the current process. This
                avoids multiprocessing overheads, but may not be able to full saturate
                the CPU. Note that the preprocessing itself will parallelise over
                multiple cores, so one job is often enough.
            If `>1`, then different estimators are dispatched to different proceses,
                which allows more parallelism but incurs some overhead.
            If `-1`, then creates as many workers as CPU cores. As each worker itself
                uses multiple cores, this is likely too many.
            It is best to select this value by benchmarking.
        parallel_mode:
            Parallel mode to use.

            * `"block"`: Blocks until all workers are done. Returns in order.
            * `"as-ready"`: Returns results as they are ready. Any order.
            * `"in-order"`: Returns results in order, blocking only in the order that
                needs to be returned in.

    Returns:
        Iterator of tuples containing the ensemble configuration, the fitted
        preprocessing pipeline, the transformed training data, the transformed target,
        and the indices of categorical features.
    """
    _, rng = infer_random_state(random_state)

    if SUPPORTS_RETURN_AS:
        return_as = PARALLEL_MODE_TO_RETURN_AS[parallel_mode]
        executor = joblib.Parallel(
            n_jobs=n_preprocessing_jobs,
            return_as=return_as,
            batch_size="auto",
        )
    else:
        executor = joblib.Parallel(n_jobs=n_preprocessing_jobs, batch_size="auto")
    func = partial(_fit_preprocessing_one, feature_schema=feature_schema)
    worker_func = joblib.delayed(func)

    seeds = rng.integers(0, np.iinfo(np.int32).max, len(configs))
    yield from executor(  # type: ignore[misc]
        [
            worker_func(config, X_train, y_train, seed)
            for config, seed in zip(configs, seeds)
        ],
    )
