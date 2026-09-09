#  Copyright (c) Prior Labs GmbH 2026.

"""Module for generating ensemble configurations."""

from __future__ import annotations

import copy
import dataclasses
import math
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import chain, product, repeat
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import torch

from tabpfn.constants import (
    AUTO_FEATURE_SUBSAMPLING_IMPORTANCE_MIN_SAMPLES,
    AUTO_FEATURE_SUBSAMPLING_TOP_K,
    AUTO_FEATURE_SUBSAMPLING_TOP_K_MIN_FEATURES,
    CLASS_SHUFFLE_OVERESTIMATE_FACTOR,
    FEATURE_IMPORTANCE_MAX_SAMPLES,
    MAXIMUM_FEATURE_SHIFT,
)
from tabpfn.preprocessing.configs import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    FeatureSubsamplingMethod,
    RegressorEnsembleConfig,
    SampleSubsamplingMethod,
)
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline
from tabpfn.preprocessing.torch import (
    FeatureSchema,
    TorchPreprocessingPipeline,
    create_gpu_preprocessing_pipeline,
)
from tabpfn.preprocessing.transform import fit_preprocessing
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline

    from tabpfn.preprocessing.configs import PreprocessorConfig
    from tabpfn.preprocessing.pipeline_interface import PreprocessingPipeline

T = TypeVar("T")


@dataclasses.dataclass
class TabPFNEnsembleMember:
    """Holds data, config, and preprocessors for a single ensemble member.

    The data is preprocessed on the CPU but this member also holds a torch preprocessor
    pipeline to be run before inference on the GPU.
    """

    config: EnsembleConfig
    cpu_preprocessor: PreprocessingPipeline
    gpu_preprocessor: TorchPreprocessingPipeline | None
    X_train: np.ndarray | torch.Tensor
    y_train: np.ndarray | torch.Tensor
    feature_schema: FeatureSchema
    feature_indices: np.ndarray | None = None

    def transform_X_test(
        self, X: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Transform the test data."""
        if self.feature_indices is not None:
            X = X[..., self.feature_indices]
        return self.cpu_preprocessor.transform(X).X


class TabPFNEnsemblePreprocessor:
    """Orchestrates the creation of ensemble members.

    - Generates preprocessing pipelines.
    - Iterates over cpu preprocessing.
    - Creates TabPFNEnsembleMember objects with all necessary information to process
        a single ensemble member.
    - Can use global data information and pipelines to perform balanced data slicing
       (e.g. sample/feature subsampling) per ensemble member.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        configs: list[ClassifierEnsembleConfig] | list[RegressorEnsembleConfig],
        n_samples: int,
        feature_schema: FeatureSchema,
        random_state: int | np.random.Generator,
        n_preprocessing_jobs: int,
        keep_fitted_cache: bool = False,
        enable_gpu_preprocessing: bool = False,
        feature_subsampling_method: FeatureSubsamplingMethod = FeatureSubsamplingMethod.RANDOM,  # noqa: E501
        constant_feature_count: int = 50,
        subsample_samples: int | float | list[np.ndarray] | None = None,
        sample_subsampling_method: SampleSubsamplingMethod = SampleSubsamplingMethod.AUTO,  # noqa: E501
        importance_top_k_count: int | float | Literal["auto"] = "auto",
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | torch.Tensor | None = None,
        task_type: Literal["classifier", "regressor"] = "classifier",
    ) -> None:
        """Init.

        Args:
            configs: List of ensemble configurations.
            n_samples: Number of training samples.
            feature_schema: Feature schema of the dataset.
            random_state: Random state object for preprocessing. If int, the
                preprocessing will use the same random seed across calls to fit().
            n_preprocessing_jobs: Number of preprocessing jobs to use.
            keep_fitted_cache: Whether to keep the fitted cache for gpu preprocessing.
                For the cpu preprocessors, the cache is always kept implicitly in the
                preprocessor objects.
            enable_gpu_preprocessing: Whether to move quantile/SVD/shuffle to GPU.
            feature_subsampling_method: Method for subsampling features. One of
                "balanced", "random", "constant_and_balanced", or "feature_importance".
            constant_feature_count: Number of leading features to always include
                when using the "constant_and_balanced" method.
            subsample_samples: Method to subsample rows per estimator. If int,
                subsample that many samples. If float, subsample that fraction of
                samples. If a list of index arrays, use those indices directly. If
                ``None``, no row subsampling is done.
            sample_subsampling_method: How rows are drawn per estimator when
                ``subsample_samples`` is an int or float. One of "auto", "balanced",
                "stratified", or "majority_downsample". "auto" resolves to
                "stratified" for classifiers and "balanced" for regressors. The
                target-aware methods require ``y_train``; "stratified" additionally
                requires a classifier task.
            importance_top_k_count: Number of top-important features always included
                per estimator when feature_subsampling_method is an importance-based
                method. If float in (0, 1], resolved as ceil(value * n_total_features).
                If "auto", uses 150 when n_features > 200 and n_samples > 100_000,
                otherwise keeps all features (no importance filtering).
            X_train: Training features used to compute feature importance. Required
                when feature_subsampling_method is "feature_importance".
            y_train: Training targets used to compute feature importance or
                target-aware row subsampling. Required when feature_subsampling_method
                is "feature_importance" or sample_subsampling_method is target-aware.
            task_type: ``"classifier"`` or ``"regressor"``, controls whether
                ExtraTreesClassifier or ExtraTreesRegressor is used and resolves
                task-dependent sample subsampling behavior.
        """
        super().__init__()
        self.configs = configs
        self.feature_schema = feature_schema
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.keep_fitted_cache = keep_fitted_cache

        self.random_state = random_state
        self.enable_gpu_preprocessing = enable_gpu_preprocessing
        _, rng = infer_random_state(random_state)
        # Derive independent seeds for each random step in one batch so that
        # each step's stream is unaffected by what happens in the others.
        seed_pipelines, seed_features, seed_rows = rng.integers(
            0, np.iinfo(np.int64).max, 3
        )
        rng_pipelines = np.random.default_rng(seed=seed_pipelines)
        rng_features = np.random.default_rng(seed=seed_features)
        rng_rows = np.random.default_rng(seed=seed_rows)

        self.pipeline_seeds = rng_pipelines.integers(
            0, np.iinfo(np.int32).max, len(self.configs)
        )
        self.pipelines = [
            create_preprocessing_pipeline(
                config,
                random_state=int(seed),
                enable_gpu_preprocessing=enable_gpu_preprocessing,
            )
            for config, seed in zip(self.configs, self.pipeline_seeds, strict=True)
        ]

        n_total_features = feature_schema.num_columns
        resolved_top_k = _resolve_importance_top_k(
            importance_top_k_count=importance_top_k_count,
            n_total_features=n_total_features,
        )

        max_features_per_estimator = [
            c.preprocess_config.max_features_per_estimator for c in self.configs
        ]

        importance_feature_order: np.ndarray | None = None
        needs_subsampling = any(
            s < n_total_features for s in max_features_per_estimator
        )

        feature_subsampling_method = _resolve_feature_subsampling_method(
            method=feature_subsampling_method,
            needs_subsampling=needs_subsampling,
            n_samples=n_samples,
        )

        is_feature_importance_subsampling = (
            feature_subsampling_method
            == FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE
        )

        if (
            is_feature_importance_subsampling
            and needs_subsampling
            and resolved_top_k < n_total_features
        ):
            if X_train is None or y_train is None:
                raise ValueError(
                    "X_train and y_train must be provided when using a "
                    "feature_importance subsampling method."
                )
            cat_indices = (
                self.feature_schema.indices_for(FeatureModality.CATEGORICAL) or None
            )
            y_for_importance = _targets_to_numpy(y_train)
            importance_feature_order = _compute_feature_importance_order(
                X=X_train,
                y=y_for_importance,
                task_type=task_type,
                categorical_feature_indices=cat_indices,
                rng=rng_features,
            )

        self.subsample_feature_indices = _get_subsample_feature_indices(
            pipelines=self.pipelines,
            n_samples=n_samples,
            feature_schema=self.feature_schema,
            max_features_per_estimator=max_features_per_estimator,
            rng=rng_features,
            feature_subsampling_method=feature_subsampling_method,
            constant_feature_count=constant_feature_count,
            importance_feature_order=importance_feature_order,
            importance_top_k_count=resolved_top_k,
        )

        resolved_sample_subsampling_method = SampleSubsamplingMethod(
            sample_subsampling_method
        )
        if isinstance(subsample_samples, (int, float)):
            resolved_sample_subsampling_method = _resolve_sample_subsampling_method(
                resolved_sample_subsampling_method,
                task_type=task_type,
            )

        self.subsample_row_indices = _get_subsample_indices_for_estimators(
            subsample_samples=subsample_samples,
            num_estimators=len(self.configs),
            n_samples=n_samples,
            rng=rng_rows,
            method=resolved_sample_subsampling_method,
            y=y_train,
            task_type=task_type,
        )

    def any_estimator_uses_gpu_svd(self) -> bool:
        """True if any ensemble estimator will run SVD on the GPU.

        Used to gate the LAPACK lazy-wrapper pre-warm in
        ``parallel_execute``: pre-warm is only needed when the parallel
        functions will hit ``torch.svd_lowrank`` -> ``torch.linalg.qr``.
        Mirrors the SVD-step inclusion logic in ``create_preprocessing_pipeline``
        and ``create_gpu_preprocessing_pipeline``.
        """
        if not self.enable_gpu_preprocessing:
            return False
        return any(
            not c.preprocess_config.differentiable
            and c.preprocess_config.global_transformer_name is not None
            and c.preprocess_config.global_transformer_name != "None"
            for c in self.configs
        )

    def fit_transform_ensemble_members_iterator(
        self,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        parallel_mode: Literal["block", "as-ready", "in-order"],
    ) -> Iterator[TabPFNEnsembleMember]:
        """Get an iterator over the fit and transform data."""
        preprocessed_data_iterator = fit_preprocessing(
            configs=self.configs,
            X_train=X_train,
            y_train=y_train,
            feature_schema=self.feature_schema,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            parallel_mode=parallel_mode,
            pipelines=self.pipelines,
            subsample_feature_indices=self.subsample_feature_indices,
            subsample_row_indices=self.subsample_row_indices,
        )

        if not self.enable_gpu_preprocessing:
            # Legacy path: create GPU pipelines upfront (before CPU
            # preprocessing) since they only contain the outlier removal step
            # and don't need CPU metadata.
            gpu_preprocessors = [
                create_gpu_preprocessing_pipeline(
                    config=config,
                    keep_fitted_cache=self.keep_fitted_cache,
                )
                for config in self.configs
            ]

        for (
            config_index,
            config,
            cpu_preprocessor,
            X_train_preprocessed,
            y_train_preprocessed,
            feature_schema_preprocessed,
        ) in preprocessed_data_iterator:
            if self.enable_gpu_preprocessing:
                # The CPU output schema carries scheduled_gpu_transform
                # annotations set by ReshapeFeatureDistributionsStep,
                # so the GPU factory can read target indices directly.
                gpu_preprocessor = create_gpu_preprocessing_pipeline(
                    config=config,
                    keep_fitted_cache=self.keep_fitted_cache,
                    enable_gpu_preprocessing=True,
                    feature_schema=feature_schema_preprocessed,
                    n_train_samples=X_train_preprocessed.shape[0],
                    random_state=int(self.pipeline_seeds[config_index]),
                )
            else:
                gpu_preprocessor = gpu_preprocessors[config_index]  # type: ignore

            yield TabPFNEnsembleMember(
                config=config,
                cpu_preprocessor=cpu_preprocessor,
                gpu_preprocessor=gpu_preprocessor,
                X_train=X_train_preprocessed,
                y_train=y_train_preprocessed,
                feature_schema=feature_schema_preprocessed,
                feature_indices=self.subsample_feature_indices[config_index],
            )

    def fit_transform_ensemble_members(
        self,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
    ) -> list[TabPFNEnsembleMember]:
        """Fit and transform the ensemble members."""
        return list(
            self.fit_transform_ensemble_members_iterator(
                X_train=X_train,
                y_train=y_train,
                parallel_mode="block",
            )
        )


def _balance(x: Iterable[T], n: int) -> list[T]:
    """Take a list of elements and make a new list where each appears `n` times.

    E.g. balance([1, 2, 3], 2) -> [1, 1, 2, 2, 3, 3]
    """
    return list(chain.from_iterable(repeat(elem, n) for elem in x))


def _subsample_rows_balanced(
    subsample_size: int,
    n_rows: int,
    num_estimators: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | None:
    """Balanced round-robin row subsampling from a shared shuffled pool.

    Rows are globally shuffled once so consecutive pool positions correspond to
    unrelated original rows. Each estimator draws ``subsample_size`` slots from
    a shared pool that refills when exhausted. This ensures every row appears
    approximately the same number of times across all estimators.
    """
    if subsample_size >= n_rows:
        return None

    shuffled_order = rng.permutation(n_rows)
    result: list[np.ndarray | None] = []
    pool: list[int] = []

    for _ in range(num_estimators):
        slots, pool = _draw_balanced_from_pool(pool, subsample_size, n_rows, rng)
        original_indices = shuffled_order[slots]
        result.append(np.sort(original_indices))

    return result


def _compute_stratified_class_counts(
    class_sizes: np.ndarray,
    subsample_size: int,
) -> np.ndarray:
    """Compute per-class target sample counts that sum to exactly ``subsample_size``.

    Every class is guaranteed at least one slot. One slot is reserved per class
    upfront; the remaining slots are distributed proportionally with
    largest-remainder rounding. Raises ``ValueError`` if
    ``subsample_size < n_classes``.

    Args:
        class_sizes: 1-D integer array of per-class row counts.
        subsample_size: Total number of rows to allocate across classes.

    Returns:
        1-D integer array of length ``len(class_sizes)`` where each entry is the
        number of rows to draw from the corresponding class.  Entries sum to
        exactly ``subsample_size``.
    """
    assert class_sizes.sum() > 0
    n_classes = len(class_sizes)

    if subsample_size < n_classes:
        raise ValueError(
            f"subsample_size ({subsample_size}) must be >= number of classes "
            f"({n_classes}) so that every class can receive at least one sample."
        )

    # Reserve 1 slot per class, then distribute the remainder proportionally so
    # that every class is always represented in every estimator subsample.
    remaining = subsample_size - n_classes
    class_fracs = class_sizes / class_sizes.sum()
    raw = class_fracs * remaining
    counts = np.floor(raw).astype(int) + 1
    leftover = subsample_size - counts.sum()
    if leftover > 0:
        fracs = raw - np.floor(raw)
        top = np.argpartition(fracs, -leftover)[-leftover:]
        counts[top] += 1
    return counts


def _subsample_rows_stratified(
    subsample_size: int,
    y: np.ndarray,
    num_estimators: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | None:
    """Stratified row subsampling that preserves class proportions.

    Each estimator draws a subsample of size ``subsample_size`` where every class
    is represented by at least one row (provided ``subsample_size >= n_classes``).
    Slot counts approximate the original class distribution via proportional
    allocation with a guaranteed minimum of one per class. Within each class a
    balanced round-robin pool ensures every row appears approximately the same
    number of times across estimators; pool refills allow oversampling of classes
    with fewer rows than the per-class target.

    Args:
        subsample_size: Number of rows to subsample for each estimator.
        y: Class labels.
        num_estimators: Number of estimators to generate subsample indices for.
        rng: Random number generator.

    Returns:
        List of row-index arrays (one per estimator), or ``None`` entries when no
        subsampling is needed.
    """
    n_rows = len(y)
    if subsample_size >= n_rows:
        return None

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    class_sizes = np.bincount(inverse, minlength=n_classes)

    class_indices: list[np.ndarray] = [
        np.where(inverse == c)[0] for c in range(n_classes)
    ]

    target_counts = _compute_stratified_class_counts(
        class_sizes=class_sizes,
        subsample_size=subsample_size,
    )

    pools: list[list[int]] = [[] for _ in range(n_classes)]
    result: list[np.ndarray] = []

    for _ in range(num_estimators):
        estimator_indices: list[np.ndarray] = []
        for c in range(n_classes):
            count = int(target_counts[c])
            if count == 0:
                continue
            slots, pools[c] = _draw_balanced_from_pool(
                pools[c], count, len(class_indices[c]), rng
            )
            estimator_indices.append(class_indices[c][np.array(slots)])
        if estimator_indices:
            result.append(np.sort(np.concatenate(estimator_indices).astype(np.int64)))

    return result


def _compute_majority_downsample_group_counts(
    group_sizes: np.ndarray,
    subsample_size: int,
) -> np.ndarray:
    """Keep every non-majority row and allocate the rest to the majority group.

    The majority is the single group whose size is strictly larger than every
    other group's size. All other groups are kept whole. The subsampling budget
    must therefore be large enough to contain every non-majority row. Tied
    largest groups are rejected because there is no unique majority group to
    downsample.

    Args:
        group_sizes: 1-D integer array of per-group row counts.
        subsample_size: Total number of rows to allocate across groups. Must be
            ``<= group_sizes.sum()``.

    Returns:
        1-D integer array of length ``len(group_sizes)``. Every non-majority
        count equals its group size, and the counts sum to ``subsample_size``.

    Raises:
        ValueError: If there is no unique majority group or the subsampling
            budget cannot contain all non-majority rows plus one majority row.
    """
    assert 0 < subsample_size <= group_sizes.sum()

    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    if subsample_size == group_sizes.sum():
        return group_sizes.copy()

    largest_size = int(group_sizes.max())
    majority_groups = np.flatnonzero(group_sizes == largest_size)
    if len(majority_groups) != 1:
        raise ValueError(
            "majority_downsample requires one unique majority target value, but "
            f"{len(majority_groups)} target values are tied at {largest_size} rows."
        )

    majority_group = int(majority_groups[0])
    non_majority_size = int(group_sizes.sum() - largest_size)
    if subsample_size <= non_majority_size:
        raise ValueError(
            f"subsample_size ({subsample_size}) must be greater than the number "
            f"of non-majority rows ({non_majority_size}) when using "
            "majority_downsample. Increase SUBSAMPLE_SAMPLES so every "
            "non-majority row and at least one majority row can be kept."
        )

    counts = group_sizes.copy()
    counts[majority_group] = subsample_size - non_majority_size
    return counts


def _subsample_rows_majority_downsample(
    subsample_size: int,
    y: np.ndarray,
    num_estimators: int,
    rng: np.random.Generator,
    *,
    task_type: Literal["classifier", "regressor"],
) -> list[np.ndarray] | None:
    """Row subsampling that downsamples only the majority target value.

    Rows are grouped by exact target value. Every estimator receives all rows
    except those belonging to the single most frequent target value, then fills
    the rest of its ``subsample_size`` budget from that majority group. A
    balanced round-robin pool ensures every majority row appears approximately
    the same number of times across estimators. Non-majority rows are identical
    across estimators.

    This mode is designed for datasets with one dominant target value. For
    classification the groups are the classes. For regression this targets
    zero-inflated or otherwise spiky targets: the repeated value is downsampled
    while all other values are kept. When there is no unique majority value,
    this warns and falls back to stratified sampling for classification or
    balanced sampling for regression. Budgets too small to keep all
    non-majority rows plus at least one majority row are rejected.

    Args:
        subsample_size: Number of rows to subsample for each estimator.
        y: Target values.
        num_estimators: Number of estimators to generate subsample indices for.
        rng: Random number generator.
        task_type: Determines the fallback method when there is no unique
            majority target value.

    Returns:
        List of row-index arrays (one per estimator), or ``None`` when no
        subsampling is needed.
    """
    n_rows = len(y)
    if subsample_size >= n_rows:
        return None

    _, inverse = np.unique(y, return_inverse=True)
    inverse = inverse.reshape(-1)
    group_sizes = np.bincount(inverse)
    largest_size = int(group_sizes.max())
    majority_groups = np.flatnonzero(group_sizes == largest_size)
    if len(majority_groups) != 1:
        fallback_method = _resolve_sample_subsampling_method(
            SampleSubsamplingMethod.AUTO,
            task_type=task_type,
        )
        warnings.warn(
            "majority_downsample requires one unique majority target value, but "
            f"{len(majority_groups)} target values are tied at {largest_size} rows; "
            f"falling back to {fallback_method.value!r} row subsampling.",
            UserWarning,
            stacklevel=2,
        )
        if fallback_method == SampleSubsamplingMethod.STRATIFIED:
            return _subsample_rows_stratified(
                subsample_size=subsample_size,
                y=y,
                num_estimators=num_estimators,
                rng=rng,
            )
        return _subsample_rows_balanced(
            subsample_size=subsample_size,
            n_rows=n_rows,
            num_estimators=num_estimators,
            rng=rng,
        )

    target_counts = _compute_majority_downsample_group_counts(
        group_sizes=group_sizes,
        subsample_size=subsample_size,
    )

    majority_group = int(majority_groups[0])
    if task_type == "classifier" and len(group_sizes) > 1:
        largest_non_majority_size = int(np.delete(group_sizes, majority_group).max())
        sampled_majority_size = int(target_counts[majority_group])
        if sampled_majority_size < largest_non_majority_size:
            warnings.warn(
                "majority_downsample changes the class prior so the original "
                f"majority class has {sampled_majority_size} rows, fewer than "
                f"another class with {largest_non_majority_size} rows. Predicted "
                "probabilities may require calibration.",
                UserWarning,
                stacklevel=2,
            )

    # Non-majority groups are identical for every estimator; only the single
    # majority group needs the round-robin pool.
    whole_groups = np.flatnonzero(target_counts == group_sizes)
    whole_indices = np.flatnonzero(np.isin(inverse, whole_groups))
    downsampled_groups = np.flatnonzero(
        (target_counts > 0) & (target_counts < group_sizes)
    )
    group_indices = {int(g): np.flatnonzero(inverse == g) for g in downsampled_groups}

    pools: dict[int, list[int]] = {g: [] for g in group_indices}
    result: list[np.ndarray] = []
    for _ in range(num_estimators):
        estimator_indices = [whole_indices]
        for g, rows in group_indices.items():
            slots, pools[g] = _draw_balanced_from_pool(
                pools[g], int(target_counts[g]), len(rows), rng
            )
            estimator_indices.append(rows[np.array(slots)])
        result.append(np.sort(np.concatenate(estimator_indices).astype(np.int64)))

    return result


def _resolve_sample_subsampling_method(
    method: SampleSubsamplingMethod,
    *,
    task_type: Literal["classifier", "regressor"],
) -> SampleSubsamplingMethod:
    """Resolve ``"auto"`` to a concrete row subsampling method.

    ``"auto"`` becomes ``"stratified"`` for classifiers and ``"balanced"`` for
    regressors. ``"stratified"`` is rejected for regressors, whose continuous
    target has no class proportions to preserve; ``"majority_downsample"`` works
    for both task types since it only groups rows by exact target value.
    """
    method = SampleSubsamplingMethod(method)
    if method == SampleSubsamplingMethod.AUTO:
        return (
            SampleSubsamplingMethod.STRATIFIED
            if task_type == "classifier"
            else SampleSubsamplingMethod.BALANCED
        )
    if task_type == "regressor" and method == SampleSubsamplingMethod.STRATIFIED:
        raise ValueError(
            "SAMPLE_SUBSAMPLING_METHOD='stratified' requires class labels and is "
            "only supported for classification. Use 'balanced', "
            "'majority_downsample', or 'auto' for regression."
        )
    return method


def _targets_to_numpy(y: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return the targets as a numpy array for label-only bookkeeping.

    Row indices and feature importance are non-differentiable metadata, so a
    tensor is detached here; the original tensor and its autograd graph stay
    intact for the preprocessing and inference paths. Reduced-precision floats
    such as bfloat16 have no numpy counterpart and are widened to float32 first.
    """
    if isinstance(y, np.ndarray):
        return y
    y = y.detach().cpu()
    if y.is_floating_point() and y.dtype not in (
        torch.float16,
        torch.float32,
        torch.float64,
    ):
        y = y.float()
    return y.numpy()


def _subsample_rows_by_method(
    *,
    method: SampleSubsamplingMethod,
    subsample_size: int,
    n_rows: int,
    num_estimators: int,
    rng: np.random.Generator,
    y: np.ndarray | torch.Tensor | None,
    task_type: Literal["classifier", "regressor"],
) -> list[np.ndarray] | None:
    """Dispatch to the row sampler for a concrete ``SampleSubsamplingMethod``."""
    method = SampleSubsamplingMethod(method)
    if method == SampleSubsamplingMethod.AUTO:
        raise ValueError(
            "SampleSubsamplingMethod.AUTO must be resolved via "
            "_resolve_sample_subsampling_method before drawing row indices."
        )
    if method == SampleSubsamplingMethod.BALANCED:
        return _subsample_rows_balanced(
            subsample_size=subsample_size,
            n_rows=n_rows,
            num_estimators=num_estimators,
            rng=rng,
        )
    if y is None:
        raise ValueError(
            f"Row subsampling method {method.value!r} requires the targets (y)."
        )
    y = _targets_to_numpy(y)
    if method == SampleSubsamplingMethod.STRATIFIED:
        return _subsample_rows_stratified(
            subsample_size=subsample_size,
            y=y,
            num_estimators=num_estimators,
            rng=rng,
        )
    return _subsample_rows_majority_downsample(
        subsample_size=subsample_size,
        y=y,
        num_estimators=num_estimators,
        rng=rng,
        task_type=task_type,
    )


def _get_subsample_indices_for_estimators(  # noqa: C901
    subsample_samples: int | float | list[np.ndarray] | None,
    num_estimators: int,
    n_samples: int,
    rng: np.random.Generator,
    method: SampleSubsamplingMethod = SampleSubsamplingMethod.BALANCED,
    y: np.ndarray | torch.Tensor | None = None,
    task_type: Literal["classifier", "regressor"] = "classifier",
) -> list[np.ndarray] | None:
    """Get the indices of the rows to subsample for each estimator.

    Args:
        subsample_samples: Method to subsample rows. If int, subsample that many
            samples. If float, subsample that fraction of samples. If a
            list of arrays of indices, use those indices directly (balanced across
            estimators). If `None`, no subsampling is done.
        num_estimators: Number of estimators to generate subsample indices for.
        n_samples: Total number of rows. Only used if subsample_samples is int/float.
        rng: Random number generator.
        method: Concrete row subsampling method ("balanced", "stratified", or
            "majority_downsample"). ``"auto"`` must be resolved by the caller via
            ``_resolve_sample_subsampling_method``. Only applies when
            subsample_samples is int or float.
        y: Targets. Required for the target-aware methods "stratified" and
            "majority_downsample"; ignored by "balanced".
        task_type: Determines the task-specific fallback used by
            "majority_downsample" when no unique majority target value exists.

    Returns:
        List of row-index arrays (one per estimator), or ``None`` entries when no
        subsampling is needed.
    """
    if isinstance(subsample_samples, (int, float)):
        if isinstance(subsample_samples, int):
            if subsample_samples < 1:
                raise ValueError(f"{subsample_samples=} must be >= 1 if int")
            size = min(subsample_samples, n_samples)
        else:
            if not (0 < subsample_samples < 1):
                raise ValueError(f"{subsample_samples=} must be in (0, 1) if float")
            size = int(subsample_samples * n_samples) + 1
        return _subsample_rows_by_method(
            method=method,
            subsample_size=size,
            n_rows=n_samples,
            num_estimators=num_estimators,
            rng=rng,
            y=y,
            task_type=task_type,
        )

    if isinstance(subsample_samples, list):
        if len(subsample_samples) > num_estimators:
            warnings.warn(
                f"Your list of subsample indices has more elements "
                f"(={len(subsample_samples)}) than the number of estimators "
                f"(={num_estimators}). The extra indices will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            subsample_samples = subsample_samples[:num_estimators]
        for subsample in subsample_samples:
            if len(subsample) == 0:
                raise ValueError("Length of subsampled indices must be larger than 0")
        balance_count = num_estimators // len(subsample_samples)
        subsample_indices = _balance(subsample_samples, balance_count)
        leftover = num_estimators % len(subsample_samples)
        if leftover > 0:
            subsample_indices += subsample_samples[:leftover]
        return [np.array(subsample) for subsample in subsample_indices]

    if subsample_samples is None:
        return None

    raise ValueError(f"Invalid subsample_samples: {subsample_samples}")


def _generate_class_permutations(
    *,
    num_estimators: int,
    class_shift_method: Literal["rotate", "shuffle"] | None,
    n_classes: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | list[None]:
    """Generate per-estimator permutations of class indices for an ensemble.

    Parameters
    ----------
    num_estimators:
        Number of ensemble members for which to generate permutations.
    class_shift_method:
        Strategy used to generate permutations of the class indices:
        * ``"rotate"`` - draw random circular shifts of ``np.arange(n_classes)``
          and sample from those shifts for each estimator.
        * ``"shuffle"`` - create random permutations of ``range(n_classes)``,
          deduplicate them, and balance their usage across estimators.
        * ``None`` - disable class permutation and return ``None`` entries.
    n_classes:
        Total number of distinct classes.
    rng:
        Numpy random generator used for reproducible permutations.

    Returns:
    -------
    list[np.ndarray] | list[None]
        A list of permutations (or ``None`` entries) with length ``num_estimators``.
    """
    if class_shift_method == "rotate":
        arange = np.arange(0, n_classes)
        shifts = rng.permutation(n_classes).tolist()
        class_permutations = [np.roll(arange, s) for s in shifts]
        return [class_permutations[c] for c in rng.choice(n_classes, num_estimators)]

    if class_shift_method == "shuffle":
        noise = rng.random(
            (num_estimators * CLASS_SHUFFLE_OVERESTIMATE_FACTOR, n_classes)
        )
        shufflings = np.argsort(noise, axis=1)
        uniqs = np.unique(shufflings, axis=0)
        balance_count = num_estimators // len(uniqs)
        class_permutations = _balance(uniqs, balance_count)
        rand_count = num_estimators % len(uniqs)
        if rand_count > 0:
            class_permutations += [
                uniqs[i] for i in rng.choice(len(uniqs), size=rand_count)
            ]
        return class_permutations

    if class_shift_method is None:
        return [None] * num_estimators  # type: ignore[return-value]

    raise ValueError(f"Unknown {class_shift_method=}")


def _find_max_input_features(
    pipeline: PreprocessingPipeline,
    n_samples: int,
    feature_schema: FeatureSchema,
    max_features_per_estimator: int,
) -> int:
    """Find the largest number of input features that fits within the budget.

    Decrements k until k + pipeline.num_added_features(...) <= max.

    TODO: The search always slices the *first* k features, so the budget
    estimate can be biased when transforms add features depending on feature
    type (e.g. one-hot for categoricals). Shuffling the schema indices before
    slicing would give a more representative estimate.
    """
    n_total = feature_schema.num_columns

    for k in range(min(n_total, max_features_per_estimator), -1, -1):
        if k == n_total:
            sliced_schema = feature_schema
        else:
            sliced_schema = feature_schema.slice_for_indices(list(range(k)))
        total = k + pipeline.num_added_features(n_samples, sliced_schema)
        if total <= max_features_per_estimator:
            return k

    return 0


def _get_subsample_feature_indices(
    pipelines: Sequence[PreprocessingPipeline],
    n_samples: int,
    feature_schema: FeatureSchema,
    max_features_per_estimator: Sequence[int],
    rng: np.random.Generator,
    feature_subsampling_method: FeatureSubsamplingMethod,
    constant_feature_count: int = 50,
    importance_feature_order: np.ndarray | None = None,
    importance_top_k_count: int = 150,
) -> list[np.ndarray | None]:
    """Get the indices of the features to subsample for each estimator.

    Args:
        pipelines: Preprocessing pipelines for each estimator.
        n_samples: Number of training samples.
        feature_schema: Feature schema of the dataset.
        max_features_per_estimator: Maximum number of features per estimator,
            one value per pipeline.
        rng: Random number generator.
        feature_subsampling_method: Method for subsampling features. One of
            "balanced", "random", or "constant_and_balanced".
        constant_feature_count: Number of leading features to always include
            when using the "constant_and_balanced" method.
        importance_feature_order: Feature indices sorted most->least important,
            shared by every estimator. Produced by
            ``_compute_feature_importance_order``.
        importance_top_k_count: Number of top features always included per estimator.
            Only used when feature_subsampling_method is "feature_importance".
    """
    if len(max_features_per_estimator) != len(pipelines):
        raise ValueError(
            f"max_features_per_estimator has {len(max_features_per_estimator)} "
            f"elements, but there are {len(pipelines)} pipelines"
        )
    n_total_features = feature_schema.num_columns

    # The feature subsampling will be done aware of the settings used in the
    # preprocessing pipelines because some steps add additional features
    # (SVD, append_original, fingerprint, one-hot).
    # For one-hot encoding, num_added_features returns 0 as an approximation
    # because the true count depends on data cardinality (see warning below).
    subsample_sizes = []
    for pipeline, max_feats in zip(pipelines, max_features_per_estimator, strict=True):
        subsample_sizes.append(
            _find_max_input_features(
                pipeline=pipeline,
                n_samples=n_samples,
                feature_schema=feature_schema,
                max_features_per_estimator=max_feats,
            )
        )

    # Warn when one-hot encoding and feature subsampling are both active.
    # The subsampling budget is computed assuming one-hot adds 0 extra columns,
    # so the actual post-expansion feature count may exceed max_features_per_estimator.
    if any(s < feature_schema.num_columns for s in subsample_sizes) and any(
        p.has_data_dependent_feature_expansion() for p in pipelines
    ):
        warnings.warn(
            "Feature subsampling is active, but at least one preprocessing "
            "pipeline uses data dependent feature exampnsion (for example "
            "one-hot encoding). The subsampling budget is computed "
            "without accounting for the additional columns created by this "
            "expansion (which depends on training data cardinality). The actual "
            "number of features per estimator may exceed `max_features_per_estimator` "
            "for those pipelines.",
            UserWarning,
            stacklevel=2,
        )

    if feature_subsampling_method == FeatureSubsamplingMethod.BALANCED:
        return _subsample_features_balanced(subsample_sizes, n_total_features, rng)
    if feature_subsampling_method == FeatureSubsamplingMethod.RANDOM:
        return _subsample_features_random(subsample_sizes, n_total_features, rng)
    if feature_subsampling_method == FeatureSubsamplingMethod.CONSTANT_AND_BALANCED:
        return _subsample_features_constant_and_balanced(
            subsample_sizes, n_total_features, rng, constant_feature_count
        )
    if feature_subsampling_method == FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE:
        if importance_feature_order is None:
            # top_k covers all features — importance ordering is irrelevant, fall back
            # to balanced subsampling for variety across estimators.
            return _subsample_features_balanced(subsample_sizes, n_total_features, rng)
        return _subsample_features_importance_based(
            subsample_sizes,
            n_total_features,
            importance_feature_order,
            importance_top_k_count,
            rng,
        )

    raise ValueError(
        f"Unsupported feature_subsampling_method={feature_subsampling_method!r}. "
        "If using AUTO, it must be resolved to a concrete method before calling "
        "_get_subsample_feature_indices."
    )


def _draw_balanced_from_pool(
    pool: list[int],
    size: int,
    pool_size: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Draw ``size`` slot indices via round-robin from a refillable pool.

    When the pool is exhausted it is refilled with ``range(pool_size)`` minus
    any slots already drawn for the current estimator (to avoid duplicates).
    If every slot has already been drawn (``size > pool_size``), duplicates are
    unavoidable and the pool is refilled with the full range instead.

    Returns:
        (drawn_slots, remaining_pool) so the caller can carry the pool across
        estimators for balanced coverage.
    """
    slots: list[int] = []
    remaining = size

    while remaining > 0:
        if len(pool) == 0:
            already_selected = set(slots)
            available = [i for i in range(pool_size) if i not in already_selected]
            if not available:
                available = list(range(pool_size))
            rng.shuffle(available)
            pool = available

        take = min(remaining, len(pool))
        slots.extend(pool[:take])
        pool = pool[take:]
        remaining -= take

    return slots, pool


def _subsample_features_balanced(
    subsample_sizes: list[int],
    n_total_features: int,
    rng: np.random.Generator,
) -> list[np.ndarray | None]:
    """Balanced round-robin sampling from a shared shuffled pool.

    Features are globally shuffled once so that consecutive pool positions
    correspond to unrelated original features. This prevents the round-robin
    from systematically grouping neighboring columns (which may be correlated)
    into the same estimator. Each feature appears approximately the same number
    of times across all estimators.
    """
    # Global shuffle: slot i -> original feature index shuffled_order[i].
    shuffled_order = rng.permutation(n_total_features)
    subsample_feature_indices: list[np.ndarray | None] = []
    pool: list[int] = []

    for size in subsample_sizes:
        if size >= n_total_features:
            subsample_feature_indices.append(None)
            continue

        slots, pool = _draw_balanced_from_pool(pool, size, n_total_features, rng)
        original_indices = shuffled_order[np.array(slots)]
        subsample_feature_indices.append(np.sort(original_indices))

    return subsample_feature_indices


def _subsample_features_random(
    subsample_sizes: list[int],
    n_total_features: int,
    rng: np.random.Generator,
) -> list[np.ndarray | None]:
    """Each estimator independently draws a random subset of features."""
    subsample_feature_indices: list[np.ndarray | None] = []

    for size in subsample_sizes:
        if size >= n_total_features:
            subsample_feature_indices.append(None)
        else:
            indices = rng.permutation(n_total_features)[:size]
            subsample_feature_indices.append(np.sort(indices))

    return subsample_feature_indices


def _subsample_features_constant_and_balanced(
    subsample_sizes: list[int],
    n_total_features: int,
    rng: np.random.Generator,
    constant_feature_count: int,
) -> list[np.ndarray | None]:
    """Always include the first N features, balanced round-robin for the rest.

    The constant features (indices 0..n_constant-1) are always included. The
    remaining budget is filled using balanced round-robin sampling from the
    non-constant features (indices n_constant..n_total-1). Non-constant features
    are globally shuffled once so that consecutive pool positions correspond to
    unrelated original features, preventing correlated neighboring columns from
    clustering in the same estimator.
    """
    n_constant = min(constant_feature_count, n_total_features)
    n_non_constant = n_total_features - n_constant

    # Global shuffle of non-constant features: slot i -> original feature index.
    non_constant_shuffled = rng.permutation(np.arange(n_constant, n_total_features))
    subsample_feature_indices: list[np.ndarray | None] = []
    pool: list[int] = []

    for size in subsample_sizes:
        if size >= n_total_features:
            subsample_feature_indices.append(None)
            continue

        if size <= n_constant:
            # Budget is less than constant count; just take the first `size` features.
            subsample_feature_indices.append(np.arange(size))
            continue

        # Always include the first n_constant features, fill rest via balanced pool.
        remaining_budget = size - n_constant
        slots, pool = _draw_balanced_from_pool(
            pool, remaining_budget, n_non_constant, rng
        )

        non_constant_indices = non_constant_shuffled[np.array(slots)]
        all_indices = np.concatenate([np.arange(n_constant), non_constant_indices])
        subsample_feature_indices.append(np.sort(all_indices))

    return subsample_feature_indices


def _subsample_features_importance_based(
    subsample_sizes: list[int],
    n_total_features: int,
    importance_feature_order: np.ndarray,
    top_k_count: int,
    rng: np.random.Generator,
) -> list[np.ndarray | None]:
    """Always include top-K important features; fill the rest from a balanced pool.

    All estimators share one ordering, so the non-top features are dealt from a
    single pool: each is drawn once before any is drawn twice.

    Args:
        subsample_sizes: Number of input features to select per estimator.
        n_total_features: Total number of features in the dataset.
        importance_feature_order: Feature indices sorted most->least important.
            Produced by ``_compute_feature_importance_order``.
        top_k_count: Number of top features always included per estimator.
        rng: Random number generator.
    """
    n_top = min(top_k_count, n_total_features)
    top_features = importance_feature_order[:n_top]
    remaining_features = importance_feature_order[n_top:]
    pool: list[int] = []

    result: list[np.ndarray | None] = []
    for size in subsample_sizes:
        if size >= n_total_features:
            result.append(None)
            continue
        if size <= n_top:
            # Budget only fits a portion of the top features; take the most important.
            result.append(np.sort(top_features[:size]))
            continue
        # Always include all top features, fill remaining budget via balanced pool.
        slots, pool = _draw_balanced_from_pool(
            pool, size - n_top, len(remaining_features), rng
        )
        sampled = remaining_features[np.array(slots)]
        result.append(np.sort(np.concatenate([top_features, sampled])))

    return result


def _get_lightgbm_model_cls(task_type: Literal["classifier", "regressor"]) -> type:
    # Lazy import: libomp (required by LightGBM on macOS) may not be present.
    # This path only runs for large datasets (>100k samples), which aren't
    # typically used on macOS anyway.
    import lightgbm  # noqa: PLC0415

    return (
        lightgbm.LGBMClassifier if task_type == "classifier" else lightgbm.LGBMRegressor
    )


def _fit_importance_ordering(
    X: np.ndarray,
    y: np.ndarray,
    task_type: Literal["classifier", "regressor"],
    max_samples: int,
    fit_ordering_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Fit one feature-importance ordering, on at most ``max_samples`` rows.

    The subsample is stratified for classification.
    """
    n_samples = len(X)

    if n_samples > max_samples:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        idx, _ = train_test_split(
            np.arange(n_samples),
            train_size=max_samples,
            stratify=y if task_type == "classifier" else None,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        X, y = X[idx], y[idx]

    # The importance model bins each feature from the values it is handed,
    # but `clean_data` no longer casts: not a no-op
    return fit_ordering_fn(np.asarray(X, dtype=np.float64), y)


def _compute_feature_importance_order(
    X: np.ndarray,
    y: np.ndarray,
    task_type: Literal["classifier", "regressor"],
    *,
    max_samples: int = FEATURE_IMPORTANCE_MAX_SAMPLES,
    n_tree_estimators: int = 50,
    categorical_feature_indices: list[int] | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rank features by LightGBM gain importance.

    Args:
        X: Training features, shape (n_samples, n_features).
        y: Training targets, shape (n_samples,).
        task_type: ``"classifier"`` or ``"regressor"`` (matches TabPFN estimator_type).
        max_samples: Row budget for the importance model fit.
        n_tree_estimators: Number of trees in LightGBM models.
        categorical_feature_indices: Column indices of categorical features
            passed natively to LightGBM.
        rng: Random number generator.

    Returns:
        Array of feature indices sorted from most to least important, shared by
        every estimator.
    """
    model_cls = _get_lightgbm_model_cls(task_type)
    cat_feature: list[int] | str = categorical_feature_indices or "auto"

    def _fit_ordering(X_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        model = model_cls(
            importance_type="gain",
            n_estimators=n_tree_estimators,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
        model.fit(X_fit, y_fit, categorical_feature=cat_feature)
        return np.argsort(model.feature_importances_)[::-1].copy()

    return _fit_importance_ordering(X, y, task_type, max_samples, _fit_ordering, rng)


def generate_classification_ensemble_configs(  # noqa: PLR0913
    *,
    num_estimators: int,
    add_fingerprint_feature: bool,
    polynomial_features: Literal["no", "all"] | int,
    feature_shift_decoder: Literal["shuffle", "rotate"] | None,
    preprocessor_configs: Sequence[PreprocessorConfig],
    class_shift_method: Literal["rotate", "shuffle"] | None,
    n_classes: int,
    random_state: int | np.random.Generator | None,
    num_models: int,
    outlier_removal_std: float | None,
    passthrough_inf: bool = False,
) -> list[ClassifierEnsembleConfig]:
    """Generate ensemble configurations for classification.

    Args:
        num_estimators: Number of ensemble configurations to generate.
        add_fingerprint_feature: Whether to add fingerprint features.
        polynomial_features: Maximum number of polynomial features to add, if any.
        feature_shift_decoder: How shift features
        preprocessor_configs: Preprocessor configurations to use on the data.
        class_shift_method: How to shift classes for classpermutation.
        n_classes: Number of classes.
        random_state: Random number generator.
        num_models: Number of models to use.
        outlier_removal_std: The standard deviation to remove outliers.
        passthrough_inf: Whether to pass infinite values through to the model.

    Returns:
        List of ensemble configurations.
    """
    _, rng = infer_random_state(random_state)
    start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
    featshifts = np.arange(start, start + num_estimators)
    featshifts = rng.choice(featshifts, size=num_estimators, replace=False)  # type: ignore[arg-type]

    class_permutations = _generate_class_permutations(
        num_estimators=num_estimators,
        class_shift_method=class_shift_method,
        n_classes=n_classes,
        rng=rng,
    )

    balance_count = num_estimators // len(preprocessor_configs)
    configs_ = _balance(preprocessor_configs, balance_count)
    leftover = num_estimators - len(configs_)
    if leftover > 0:
        configs_.extend(preprocessor_configs[:leftover])

    model_indices = [i % num_models for i in range(num_estimators)]

    return [
        ClassifierEnsembleConfig(
            preprocess_config=preprocesses_config,
            feature_shift_count=featshift,
            class_permutation=class_perm,
            add_fingerprint_feature=add_fingerprint_feature,
            polynomial_features=polynomial_features,
            feature_shift_decoder=feature_shift_decoder,
            _model_index=model_index,
            outlier_removal_std=outlier_removal_std,
            passthrough_inf=passthrough_inf,
        )
        for (
            featshift,
            preprocesses_config,
            class_perm,
            model_index,
        ) in zip(
            featshifts,
            configs_,
            class_permutations,
            model_indices,
            strict=True,
        )
    ]


def generate_regression_ensemble_configs(
    *,
    num_estimators: int,
    add_fingerprint_feature: bool,
    polynomial_features: Literal["no", "all"] | int,
    feature_shift_decoder: Literal["shuffle", "rotate"] | None,
    preprocessor_configs: Sequence[PreprocessorConfig],
    target_transforms: Sequence[TransformerMixin | Pipeline | None],
    random_state: int | np.random.Generator | None,
    num_models: int,
    outlier_removal_std: float | None,
    passthrough_inf: bool = False,
) -> list[RegressorEnsembleConfig]:
    """Generate ensemble configurations for regression.

    Args:
        num_estimators: Number of ensemble configurations to generate.
        add_fingerprint_feature: Whether to add fingerprint features.
        polynomial_features: Maximum number of polynomial features to add, if any.
        feature_shift_decoder: How shift features
        preprocessor_configs: Preprocessor configurations to use on the data.
        target_transforms: Target transformations to apply.
        random_state: Random number generator.
        num_models: Number of models to use.
        outlier_removal_std: The standard deviation to remove outliers.
        passthrough_inf: Whether to pass infinite values through to the model.

    Returns:
        List of ensemble configurations.
    """
    _, rng = infer_random_state(random_state)
    start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
    featshifts = np.arange(start, start + num_estimators)
    featshifts = rng.choice(featshifts, size=num_estimators, replace=False)  # type: ignore[arg-type]

    combos = list(product(preprocessor_configs, target_transforms))
    balance_count = num_estimators // len(combos)
    configs_ = _balance(combos, balance_count)
    leftover = num_estimators - len(configs_)
    if leftover > 0:
        configs_ += combos[:leftover]

    model_indices = [i % num_models for i in range(num_estimators)]

    return [
        RegressorEnsembleConfig(
            preprocess_config=preprocess_config,
            feature_shift_count=featshift,
            add_fingerprint_feature=add_fingerprint_feature,
            polynomial_features=polynomial_features,
            feature_shift_decoder=feature_shift_decoder,
            # Each config gets its own copy: the transform is later fitted in
            # place per ensemble member (see _transform_labels_one), so a
            # shared instance would end up with the last member's fitted state.
            target_transform=copy.deepcopy(target_transform),
            outlier_removal_std=outlier_removal_std,
            _model_index=model_index,
            passthrough_inf=passthrough_inf,
        )
        for featshift, (
            preprocess_config,
            target_transform,
        ), model_index in zip(
            featshifts,
            configs_,
            model_indices,
            strict=True,
        )
    ]


def _resolve_importance_top_k(
    importance_top_k_count: int | float | Literal["auto"],
    n_total_features: int,
    auto_top_k: int = AUTO_FEATURE_SUBSAMPLING_TOP_K,
    auto_min_features: int = AUTO_FEATURE_SUBSAMPLING_TOP_K_MIN_FEATURES,
) -> int:
    """Resolve importance_top_k_count to a concrete integer.

    - "auto": auto_top_k when n_features > auto_min_features,
      else n_total_features (no filtering).
    - float in (0, 1]: ceil(value * n_total_features).
    - int: used as-is.
    """
    if importance_top_k_count == "auto":
        if n_total_features > auto_min_features:
            return auto_top_k
        return n_total_features
    if isinstance(importance_top_k_count, float):
        return max(1, int(np.ceil(importance_top_k_count * n_total_features)))
    return importance_top_k_count


def _resolve_feature_subsampling_method(
    method: FeatureSubsamplingMethod,
    *,
    needs_subsampling: bool,
    n_samples: int,
    auto_min_samples: int = AUTO_FEATURE_SUBSAMPLING_IMPORTANCE_MIN_SAMPLES,
) -> FeatureSubsamplingMethod:
    """Resolve AUTO to a concrete subsampling method.

    Uses GINI_FEATURE_IMPORTANCE when subsampling is needed and the dataset is
    large enough that importance scoring is reliable (n_samples > auto_min_samples).
    Falls back to BALANCED otherwise. Non-AUTO values are returned unchanged.
    """
    if method is not FeatureSubsamplingMethod.AUTO:
        return method
    if needs_subsampling and n_samples > auto_min_samples:
        return FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE
    return FeatureSubsamplingMethod.BALANCED


MAX_AUTO_SCALED_N_ESTIMATORS = 32
"""Upper bound on the n_estimators value produced by feature-coverage scaling.

Very wide datasets would otherwise require an unbounded number of estimators to
cover every feature. We cap the auto-scaled value here; beyond this point some
features may never be sampled unless the user raises n_estimators explicitly.
"""


DEFAULT_N_ESTIMATORS = 8
"""The n_estimators value ``"auto"`` resolves to.

``"auto"`` can come from the user or from ``InferenceConfig.N_ESTIMATORS``, whose
default it is. This is the base value that feature-coverage scaling may then raise;
an explicit count from either source is used as given.
"""


def scale_n_estimators_for_feature_coverage(
    *,
    n_estimators: int | Literal["auto"],
    n_total_features: int,
    preprocessor_configs: Sequence[PreprocessorConfig],
    auto_scale_n_estimators: bool = True,
) -> int:
    """Scale up n_estimators so every feature is included in at least one estimator.

    Scaling only applies to ``n_estimators="auto"``; an explicit integer is always
    returned unchanged, so the package never overrides a value the user chose. An
    explicit value too small to cover every feature warns instead of being raised.

    With balanced feature subsampling each estimator sees at most
    ``max_features_per_estimator`` features. If
    ``n_estimators * max_features_per_estimator < n_total_features`` some features
    are never sampled. For ``"auto"`` this returns the smallest n_estimators that
    covers all features (using the smallest ``max_features_per_estimator`` across the
    supplied configs, which is the binding budget), at least
    ``DEFAULT_N_ESTIMATORS`` and capped at ``MAX_AUTO_SCALED_N_ESTIMATORS``. When
    the cap binds, full coverage is not reached and some features may never be
    sampled unless the user raises ``n_estimators`` explicitly.

    ``auto_scale_n_estimators`` (the deprecated constructor argument of the same
    name) is redundant now that scaling is opt-out by passing an explicit
    ``n_estimators``: it only affects ``"auto"``, which ``False`` resolves to
    ``DEFAULT_N_ESTIMATORS`` without scaling, exactly what passing that integer
    does. Passing ``False`` emits a ``FutureWarning``; the argument is removed in
    v9.
    """
    if not auto_scale_n_estimators:
        warnings.warn(
            "auto_scale_n_estimators is deprecated and will be removed in v9. It "
            'only affects n_estimators="auto", where False skips feature-coverage '
            "scaling; pass an explicit n_estimators instead, which also skips it.",
            FutureWarning,
            stacklevel=2,
        )
    min_max_features = (
        min(c.max_features_per_estimator for c in preprocessor_configs)
        if preprocessor_configs
        else 0
    )
    if n_estimators != "auto":
        # A count that was named explicitly -- by the user, or by the checkpoint it
        # was resolved from -- is never overridden, only warned about. The warning
        # names no source, since it cannot tell them apart and the remedy is the
        # same either way: an explicit `n_estimators` wins over a checkpoint's.
        n_covered = n_estimators * min_max_features
        if 0 < n_covered < n_total_features:
            warnings.warn(
                f"Running {n_estimators} estimators covers at most {n_covered} of "
                f"{n_total_features} features (max_features_per_estimator="
                f"{min_max_features}); the remaining features are never sampled by "
                f"any ensemble member. Pass n_estimators >= "
                f"{math.ceil(n_total_features / min_max_features)} to cover all "
                f"features.",
                UserWarning,
                stacklevel=2,
            )
        return n_estimators
    n_estimators = DEFAULT_N_ESTIMATORS
    if not auto_scale_n_estimators or min_max_features <= 0:
        return n_estimators
    min_required = math.ceil(n_total_features / min_max_features)
    target = min(min_required, MAX_AUTO_SCALED_N_ESTIMATORS)
    if n_estimators >= target:
        return n_estimators
    if min_required > MAX_AUTO_SCALED_N_ESTIMATORS:
        warnings.warn(
            f"Auto-scaling n_estimators from {n_estimators} to {target}, capped at "
            f"MAX_AUTO_SCALED_N_ESTIMATORS={MAX_AUTO_SCALED_N_ESTIMATORS}. Full "
            f"feature coverage would require {min_required} estimators "
            f"(n_total_features={n_total_features}, "
            f"max_features_per_estimator={min_max_features}); because of the cap "
            f"some features may never be sampled. Pass n_estimators >= "
            f"{min_required} to cover all features, or any explicit n_estimators "
            f"to disable scaling.",
            UserWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            f"Auto-scaling n_estimators from {n_estimators} to {target} so "
            f"every feature is included in at least one ensemble member "
            f"(n_total_features={n_total_features}, "
            f"max_features_per_estimator={min_max_features}). "
            f"Pass n_estimators >= {target} to silence this warning. "
            f"If this scaling is not desired, pass an explicit n_estimators in the "
            f"estimator constructor to disable it (note: some features may then "
            f"never be sampled).",
            UserWarning,
            stacklevel=2,
        )
    return target
