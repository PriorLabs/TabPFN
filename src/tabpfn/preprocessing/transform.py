"""Module for fitting and transforming preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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
    pipeline: PreprocessingPipeline | None = None,
    feature_indices: np.ndarray | None = None,
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
        pipeline: Preprocessing pipeline. If not provided, a new pipeline is created
            on-the-fly.
        feature_indices: Indices of features to select. If not provided, all features
            are used.

    Returns:
        Tuple containing the ensemble configuration, the fitted preprocessing pipeline,
        the transformed training data, the transformed target, and the indices of
        categorical features.
    """
    if config.subsample_ix is not None:
        X_train = X_train[config.subsample_ix]
        y_train = y_train[config.subsample_ix]
    if feature_indices is not None:
        X_train = X_train[..., feature_indices]
        feature_schema = feature_schema.slice_for_indices(feature_indices.tolist())
    if not isinstance(X_train, torch.Tensor):
        X_train = X_train.copy()
        y_train = y_train.copy()

    static_seed, _ = infer_random_state(random_state)

    if pipeline is None:
        pipeline = create_preprocessing_pipeline(config, random_state=static_seed)
    res = pipeline.fit_transform(X_train, feature_schema)

    y_train_processed = _transform_labels_one(config, y_train)

    return (
        config,
        pipeline,
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
    pipelines: Sequence[PreprocessingPipeline | None] | None = None,
    subsample_feature_indices: list[np.ndarray | None] | None = None,
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
        pipelines: Preprocessing pipelines. If not provided, a new pipeline is created
            on-the-fly for each configuration.
        subsample_feature_indices: Indices of features to subsample. If not provided,
            no features are subsampled.

    Returns:
        Iterator of tuples containing the ensemble configuration, the fitted
        preprocessing pipeline, the transformed training data, the transformed target,
        and the indices of categorical features.
    """
    _, rng = infer_random_state(random_state)

    if pipelines is None:
        pipelines = [None] * len(configs)
    assert len(pipelines) == len(configs)

    if subsample_feature_indices is None:
        subsample_feature_indices = [None] * len(configs)
    assert len(subsample_feature_indices) == len(configs)

    if SUPPORTS_RETURN_AS:
        return_as = PARALLEL_MODE_TO_RETURN_AS[parallel_mode]
        executor = joblib.Parallel(
            n_jobs=n_preprocessing_jobs,
            return_as=return_as,
            batch_size="auto",
        )
    else:
        executor = joblib.Parallel(n_jobs=n_preprocessing_jobs, batch_size="auto")

    seeds = rng.integers(0, np.iinfo(np.int32).max, len(configs))
    yield from executor(  # type: ignore[misc]
        joblib.delayed(_fit_preprocessing_one)(
            config,
            X_train,
            y_train,
            seed,
            feature_schema=feature_schema,
            pipeline=pipeline,
            feature_indices=feat_idx,
        )
        for config, seed, pipeline, feat_idx in zip(
            configs, seeds, pipelines, subsample_feature_indices
        )
    )
