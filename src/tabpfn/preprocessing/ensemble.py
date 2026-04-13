"""Module for generating ensemble configurations."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain, product, repeat
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from tabpfn.constants import (
    CLASS_SHUFFLE_OVERESTIMATE_FACTOR,
    MAXIMUM_FEATURE_SHIFT,
)
from tabpfn.preprocessing.configs import (
    ClassifierEnsembleConfig,
    EnsembleConfig,
    FeatureSubsamplingMethod,
    RegressorEnsembleConfig,
)
from tabpfn.preprocessing.pipeline_factory import create_preprocessing_pipeline
from tabpfn.preprocessing.torch import (
    FeatureSchema,
    TorchPreprocessingPipeline,
    create_gpu_preprocessing_pipeline,
)
from tabpfn.preprocessing.transform import fit_preprocessing
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch
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

    def __init__(
        self,
        *,
        configs: list[ClassifierEnsembleConfig] | list[RegressorEnsembleConfig],
        n_samples: int,
        feature_schema: FeatureSchema,
        random_state: int | np.random.Generator,
        n_preprocessing_jobs: int,
        keep_fitted_cache: bool = False,
        feature_subsampling_method: FeatureSubsamplingMethod = FeatureSubsamplingMethod.RANDOM,  # noqa: E501
        constant_feature_count: int = 50,
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
            feature_subsampling_method: Method for subsampling features. One of
                "balanced", "random", or "constant_and_balanced".
            constant_feature_count: Number of leading features to always include
                when using the "constant_and_balanced" method.
        """
        super().__init__()
        self.configs = configs
        self.feature_schema = feature_schema
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.keep_fitted_cache = keep_fitted_cache

        self.random_state = random_state
        _, self.rng = infer_random_state(random_state)

        pipeline_seeds = self.rng.integers(0, np.iinfo(np.int32).max, len(self.configs))
        self.pipelines = [
            create_preprocessing_pipeline(config, random_state=int(seed))
            for config, seed in zip(self.configs, pipeline_seeds)
        ]

        max_features_per_estimator = [
            c.preprocess_config.max_features_per_estimator for c in self.configs
        ]
        self.subsample_feature_indices = _get_subsample_feature_indices(
            pipelines=self.pipelines,
            n_samples=n_samples,
            feature_schema=self.feature_schema,
            max_features_per_estimator=max_features_per_estimator,
            rng=self.rng,
            feature_subsampling_method=feature_subsampling_method,
            constant_feature_count=constant_feature_count,
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
            subsample_feature_indices=self.subsample_feature_indices,
            pipelines=self.pipelines,
        )

        gpu_preprocessors = []
        for config in self.configs:
            gpu_preprocessor = create_gpu_preprocessing_pipeline(
                config=config,
                keep_fitted_cache=self.keep_fitted_cache,
            )
            gpu_preprocessors.append(gpu_preprocessor)

        for (
            config_index,
            config,
            cpu_preprocessor,
            X_train_preprocessed,
            y_train_preprocessed,
            feature_schema_preprocessed,
        ) in preprocessed_data_iterator:
            yield TabPFNEnsembleMember(
                config=config,
                cpu_preprocessor=cpu_preprocessor,
                gpu_preprocessor=gpu_preprocessors[config_index],
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


def _generate_index_permutations(
    n: int,
    *,
    max_index: int,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
) -> list[npt.NDArray[np.int64]]:
    """Generate indices for subsampling from the data.

    Args:
        n: Number of indices to generate.
        max_index: Maximum index to generate.
        subsample:
            Number of indices to subsample. If `int`, subsample that many
            indices. If float, subsample that fraction of indices.
        random_state: Random number generator.

    Returns:
        List of indices to subsample.
    """
    _, rng = infer_random_state(random_state)
    if isinstance(subsample, int):
        if subsample < 1:
            raise ValueError(f"{subsample=} must be larger than 1 if int")
        subsample = min(subsample, max_index)

        return [rng.permutation(max_index)[:subsample] for _ in range(n)]

    if isinstance(subsample, float):
        if not (0 < subsample < 1):
            raise ValueError(f"{subsample=} must be in (0, 1) if float")
        subsample = int(subsample * max_index) + 1
        return [rng.permutation(max_index)[:subsample] for _ in range(n)]

    raise ValueError(f"{subsample=} must be int or float.")


def _get_subsample_indices_for_estimators(
    subsample_samples: int | float | list[np.ndarray] | None,
    num_estimators: int,
    max_index: int,
    static_seed: int | np.random.Generator | None,
) -> list[None] | list[np.ndarray]:
    """Get the indices of the rows to subsample for each estimator.

    Args:
        subsample_samples: Method to subsample rows. If int, subsample that many
            samples. If float, subsample that fraction of samples. If a
            list of lists of indices, subsample the indices for each estimator.
            If `None`, no subsampling is done.
        num_estimators: Number of estimators to generate subsample indices for.
        max_index: Maximum index to generate for. Only used if subsample_samples is an
            int or float.
        static_seed: Static seed to use for the random number generator.

    Returns:
        List of list of indices to subsample for each estimator.
    """
    if isinstance(subsample_samples, (int, float)):
        subsample_indices = _generate_index_permutations(
            n=num_estimators,
            max_index=max_index,
            subsample=subsample_samples,
            random_state=static_seed,
        )
    elif isinstance(subsample_samples, list):
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
        subsample_indices = [np.array(subsample) for subsample in subsample_indices]
    elif subsample_samples is None:
        subsample_indices = [None] * num_estimators
    else:
        raise ValueError(
            f"Invalid subsample_samples: {subsample_samples}",
        )

    return subsample_indices


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

    Decrements from n_total until k + pipeline.num_added_features(...) <= max.

    TODO: The search always slices the *first* k features, so the budget
    estimate can be biased when transforms add features depending on feature
    type (e.g. one-hot for categoricals). Shuffling the schema indices before
    slicing would give a more representative estimate.
    """
    n_total = feature_schema.num_columns

    for k in range(n_total, -1, -1):
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
    for pipeline, max_feats in zip(pipelines, max_features_per_estimator):
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

    if feature_subsampling_method is FeatureSubsamplingMethod.BALANCED:
        return _subsample_features_balanced(subsample_sizes, n_total_features, rng)
    if feature_subsampling_method is FeatureSubsamplingMethod.RANDOM:
        return _subsample_features_random(subsample_sizes, n_total_features, rng)
    if feature_subsampling_method is FeatureSubsamplingMethod.CONSTANT_AND_BALANCED:
        return _subsample_features_constant_and_balanced(
            subsample_sizes, n_total_features, rng, constant_feature_count
        )

    raise ValueError(
        f"Unknown feature subsampling method: {feature_subsampling_method}"
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


def generate_classification_ensemble_configs(  # noqa: PLR0913
    *,
    num_estimators: int,
    subsample_samples: int | float | list[np.ndarray] | None,
    max_index: int,
    add_fingerprint_feature: bool,
    polynomial_features: Literal["no", "all"] | int,
    feature_shift_decoder: Literal["shuffle", "rotate"] | None,
    preprocessor_configs: Sequence[PreprocessorConfig],
    class_shift_method: Literal["rotate", "shuffle"] | None,
    n_classes: int,
    random_state: int | np.random.Generator | None,
    num_models: int,
    outlier_removal_std: float | None,
) -> list[ClassifierEnsembleConfig]:
    """Generate ensemble configurations for classification.

    Args:
        num_estimators: Number of ensemble configurations to generate.
        subsample_samples: Method to subsample rows. If int, subsample that many
            samples. If float, subsample that fraction of samples. If a
            list of lists of indices, subsample the indices for each estimator.
            If `None`, no subsampling is done.
        max_index: Maximum index to generate for.
        add_fingerprint_feature: Whether to add fingerprint features.
        polynomial_features: Maximum number of polynomial features to add, if any.
        feature_shift_decoder: How shift features
        preprocessor_configs: Preprocessor configurations to use on the data.
        class_shift_method: How to shift classes for classpermutation.
        n_classes: Number of classes.
        random_state: Random number generator.
        num_models: Number of models to use.
        outlier_removal_std: The standard deviation to remove outliers.

    Returns:
        List of ensemble configurations.
    """
    static_seed, rng = infer_random_state(random_state)
    start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
    featshifts = np.arange(start, start + num_estimators)
    featshifts = rng.choice(featshifts, size=num_estimators, replace=False)  # type: ignore[arg-type]

    class_permutations = _generate_class_permutations(
        num_estimators=num_estimators,
        class_shift_method=class_shift_method,
        n_classes=n_classes,
        rng=rng,
    )

    subsample_indices: list[None] | list[np.ndarray] = (
        _get_subsample_indices_for_estimators(
            subsample_samples=subsample_samples,
            num_estimators=num_estimators,
            max_index=max_index,
            static_seed=static_seed,
        )
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
            subsample_ix=subsample_ix,
            _model_index=model_index,
            outlier_removal_std=outlier_removal_std,
        )
        for (
            featshift,
            preprocesses_config,
            subsample_ix,
            class_perm,
            model_index,
        ) in zip(
            featshifts,
            configs_,
            subsample_indices,
            class_permutations,
            model_indices,
        )
    ]


def generate_regression_ensemble_configs(  # noqa: PLR0913
    *,
    num_estimators: int,
    subsample_samples: int | float | list[np.ndarray] | None,
    max_index: int,
    add_fingerprint_feature: bool,
    polynomial_features: Literal["no", "all"] | int,
    feature_shift_decoder: Literal["shuffle", "rotate"] | None,
    preprocessor_configs: Sequence[PreprocessorConfig],
    target_transforms: Sequence[TransformerMixin | Pipeline | None],
    random_state: int | np.random.Generator | None,
    num_models: int,
    outlier_removal_std: float | None,
) -> list[RegressorEnsembleConfig]:
    """Generate ensemble configurations for regression.

    Args:
        num_estimators: Number of ensemble configurations to generate.
        subsample_samples: Method to subsample rows. If int, subsample that many
            samples. If float, subsample that fraction of samples. If a
            list of lists of indices, subsample the indices for each estimator.
            If `None`, no subsampling is done.
        max_index: Maximum index to generate for.
        add_fingerprint_feature: Whether to add fingerprint features.
        polynomial_features: Maximum number of polynomial features to add, if any.
        feature_shift_decoder: How shift features
        preprocessor_configs: Preprocessor configurations to use on the data.
        target_transforms: Target transformations to apply.
        random_state: Random number generator.
        num_models: Number of models to use.
        outlier_removal_std: The standard deviation to remove outliers.

    Returns:
        List of ensemble configurations.
    """
    static_seed, rng = infer_random_state(random_state)
    start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
    featshifts = np.arange(start, start + num_estimators)
    featshifts = rng.choice(featshifts, size=num_estimators, replace=False)  # type: ignore[arg-type]

    subsample_indices: list[None] | list[np.ndarray] = (
        _get_subsample_indices_for_estimators(
            subsample_samples=subsample_samples,
            num_estimators=num_estimators,
            max_index=max_index,
            static_seed=static_seed,
        )
    )

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
            subsample_ix=subsample_ix,
            target_transform=target_transform,
            outlier_removal_std=outlier_removal_std,
            _model_index=model_index,
        )
        for featshift, subsample_ix, (
            preprocess_config,
            target_transform,
        ), model_index in zip(
            featshifts,
            subsample_indices,
            configs_,
            model_indices,
        )
    ]
