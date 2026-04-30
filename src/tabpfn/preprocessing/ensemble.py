"""Module for generating ensemble configurations."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain, product, repeat
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import numpy.typing as npt

from tabpfn.constants import (
    CLASS_SHUFFLE_OVERESTIMATE_FACTOR,
    GINI_FEATURE_IMPORTANCE_MAX_SAMPLES,
    MAXIMUM_FEATURE_SHIFT,
    PERMUTATION_FEATURE_IMPORTANCE_MAX_SAMPLES,
    PERMUTATION_FEATURE_IMPORTANCE_N_FOLDS,
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
        importance_top_k_count: int | float = 50,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
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
            importance_top_k_count: Number of top-important features always included
                per estimator when feature_subsampling_method is "feature_importance".
                If a float in (0, 1], resolved as ceil(value * n_total_features).
            X_train: Training features used to compute feature importance. Required
                when feature_subsampling_method is "feature_importance".
            y_train: Training targets used to compute feature importance. Required
                when feature_subsampling_method is "feature_importance".
            task_type: ``"classifier"`` or ``"regressor"``, controls whether
                ExtraTreesClassifier or ExtraTreesRegressor is used.
                Only used when feature_subsampling_method is "feature_importance".
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
            for config, seed in zip(self.configs, self.pipeline_seeds)
        ]

        n_total_features = feature_schema.num_columns
        if isinstance(importance_top_k_count, float):
            resolved_top_k = max(
                1, int(np.ceil(importance_top_k_count * n_total_features))
            )
        else:
            resolved_top_k = importance_top_k_count

        is_feature_importance_subsampling = (
            feature_subsampling_method
            is FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE
            or feature_subsampling_method
            is FeatureSubsamplingMethod.PERMUTATION_FEATURE_IMPORTANCE
        )

        importance_feature_orders: list[np.ndarray] | None = None
        if is_feature_importance_subsampling and resolved_top_k < n_total_features:
            if X_train is None or y_train is None:
                raise ValueError(
                    "X_train and y_train must be provided when using a "
                    "feature_importance subsampling method."
                )
            importance_feature_orders = compute_feature_importance_order(
                X=X_train,
                y=y_train,
                task_type=task_type,
                method=feature_subsampling_method,
                n_estimators=len(self.configs),
                rng=rng_features,
            )

        max_features_per_estimator = [
            c.preprocess_config.max_features_per_estimator for c in self.configs
        ]
        self.subsample_feature_indices = _get_subsample_feature_indices(
            pipelines=self.pipelines,
            n_samples=n_samples,
            feature_schema=self.feature_schema,
            max_features_per_estimator=max_features_per_estimator,
            rng=rng_features,
            feature_subsampling_method=feature_subsampling_method,
            constant_feature_count=constant_feature_count,
            importance_feature_orders=importance_feature_orders,
            importance_top_k_count=resolved_top_k,
        )

        self.subsample_row_indices = _get_subsample_indices_for_estimators(
            subsample_samples=subsample_samples,
            num_estimators=len(self.configs),
            n_samples=n_samples,
            rng=rng_rows,
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
) -> list[npt.NDArray[np.int64]] | None:
    """Balanced round-robin row subsampling from a shared shuffled pool.

    Rows are globally shuffled once so consecutive pool positions correspond to
    unrelated original rows. Each estimator draws ``subsample_size`` slots from
    a shared pool that refills when exhausted. This ensures every row appears
    approximately the same number of times across all estimators.
    """
    if subsample_size >= n_rows:
        return None

    shuffled_order = rng.permutation(n_rows)
    result: list[npt.NDArray[np.int64] | None] = []
    pool: list[int] = []

    for _ in range(num_estimators):
        slots, pool = _draw_balanced_from_pool(pool, subsample_size, n_rows, rng)
        original_indices = shuffled_order[slots]
        result.append(np.sort(original_indices))

    return result


def _get_subsample_indices_for_estimators(  # noqa: C901
    subsample_samples: int | float | list[np.ndarray] | None,
    num_estimators: int,
    n_samples: int,
    rng: np.random.Generator,
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

    Returns:
        List of row-index arrays (one per estimator), or ``None`` entries when no
        subsampling is needed.
    """
    if isinstance(subsample_samples, int):
        if subsample_samples < 1:
            raise ValueError(f"{subsample_samples=} must be >= 1 if int")
        size = min(subsample_samples, n_samples)
        return _subsample_rows_balanced(
            subsample_size=size,
            n_rows=n_samples,
            num_estimators=num_estimators,
            rng=rng,
        )

    if isinstance(subsample_samples, float):
        if not (0 < subsample_samples < 1):
            raise ValueError(f"{subsample_samples=} must be in (0, 1) if float")
        size = int(subsample_samples * n_samples) + 1
        return _subsample_rows_balanced(
            subsample_size=size,
            n_rows=n_samples,
            num_estimators=num_estimators,
            rng=rng,
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
    importance_feature_orders: list[np.ndarray] | None = None,
    importance_top_k_count: int = 100,
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
        importance_feature_orders: Per-estimator feature indices sorted most->least
            important. Produced by ``compute_feature_importance_order``.
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
    if feature_subsampling_method in (
        FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE,
        FeatureSubsamplingMethod.PERMUTATION_FEATURE_IMPORTANCE,
    ):
        if importance_feature_orders is None:
            # top_k covers all features — importance ordering is irrelevant, fall back
            # to balanced subsampling for variety across estimators.
            return _subsample_features_balanced(subsample_sizes, n_total_features, rng)
        return _subsample_features_importance_based(
            subsample_sizes,
            n_total_features,
            importance_feature_orders,
            importance_top_k_count,
            rng,
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


def _subsample_features_importance_based(
    subsample_sizes: list[int],
    n_total_features: int,
    importance_feature_orders: list[np.ndarray],
    top_k_count: int | float,
    rng: np.random.Generator,
) -> list[np.ndarray | None]:
    """Always include top-K important features; randomly sample the rest.

    Each estimator uses its own importance ordering from ``importance_feature_orders``,
    cycling through the list when there are more estimators than orderings.

    Args:
        subsample_sizes: Number of input features to select per estimator.
        n_total_features: Total number of features in the dataset.
        importance_feature_orders: Per-estimator feature indices sorted most->least
            important. Produced by ``compute_feature_importance_order``.
        top_k_count: Number of top features always included per estimator.
            If a float in (0, 1], resolved as ``ceil(top_k_count * n_total_features)``.
        rng: Random number generator.
    """
    if isinstance(top_k_count, float):
        if not 0.0 < top_k_count <= 1.0:
            raise ValueError(
                f"top_k_count as float must be in (0, 1], got {top_k_count}"
            )
        n_top = max(1, int(np.ceil(top_k_count * n_total_features)))
    else:
        n_top = top_k_count
    n_top = min(n_top, n_total_features)

    n_orderings = len(importance_feature_orders)
    result: list[np.ndarray | None] = []
    for i, size in enumerate(subsample_sizes):
        importance_feature_order = importance_feature_orders[i % n_orderings]
        top_features = importance_feature_order[:n_top]
        remaining_features = importance_feature_order[n_top:]
        if size >= n_total_features:
            result.append(None)
            continue
        if size <= n_top:
            # Budget only fits a portion of the top features; take the most important.
            result.append(np.sort(top_features[:size]))
            continue
        # Always include all top features, fill remaining budget randomly.
        remaining_budget = size - n_top
        n_draw = min(remaining_budget, len(remaining_features))
        sampled = rng.choice(remaining_features, size=n_draw, replace=False)
        result.append(np.sort(np.concatenate([top_features, sampled])))

    return result


def _get_extra_trees_model_cls(
    task_type: Literal["classifier", "regressor"],
) -> type:
    if task_type == "classifier":
        from sklearn.ensemble import ExtraTreesClassifier  # noqa: PLC0415

        return ExtraTreesClassifier

    from sklearn.ensemble import ExtraTreesRegressor  # noqa: PLC0415

    return ExtraTreesRegressor


def _compute_gini_importance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: Literal["classifier", "regressor"],
    n_tree_estimators: int,
    n_estimators: int,
    max_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Return one feature-importance ordering per TabPFN ensemble estimator.

    When the dataset fits within ``max_samples`` a single ExtraTrees model is
    trained on all data and its ordering is repeated for every estimator.

    When the dataset is larger, ``n_subsamples = min(n_estimators,
    n_samples // max_samples + 1)`` independent subsamples of size ``max_samples``
    are drawn, each producing a different ordering.  The resulting orderings are
    then cycled to fill the full list of length ``n_estimators``.
    """
    model_cls = _get_extra_trees_model_cls(task_type)
    n_samples = len(X)

    def _fit_ordering(X_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        seed = int(rng.integers(0, 2**31))
        model = model_cls(n_estimators=n_tree_estimators, random_state=seed, n_jobs=-1)
        model.fit(X_fit, y_fit)
        return np.argsort(model.feature_importances_)[::-1].copy()

    if n_samples <= max_samples:
        ordering = _fit_ordering(X, y)
        return [ordering] * n_estimators

    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    n_subsamples = min(n_estimators, n_samples // max_samples + 1)
    stratify = y if task_type == "classifier" else None
    orderings = []
    for _ in range(n_subsamples):
        idx, _ = train_test_split(
            np.arange(n_samples),
            train_size=max_samples,
            stratify=stratify,
            random_state=int(rng.integers(0, 2**31)),
        )
        orderings.append(_fit_ordering(X[idx], y[idx]))
    return [orderings[i % n_subsamples] for i in range(n_estimators)]


def _compute_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    task_type: Literal["classifier", "regressor"],
    n_folds: int,
    n_tree_estimators: int,
    n_estimators: int,
    max_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Return one feature-importance ordering per fold, cycled to ``n_estimators``.

    The number of folds is capped at ``n_estimators``.

    When ``n_samples > max_samples`` standard k-fold is replaced by a custom
    scheme: for each fold a fixed-size validation set of
    ``max_samples // n_folds`` rows is sampled, and the remaining budget
    (``max_samples - val_size``) is used for training.  This keeps the
    validation set small and predictable regardless of dataset size.
    """
    from sklearn.inspection import permutation_importance  # noqa: PLC0415

    model_cls = _get_extra_trees_model_cls(task_type)
    n_folds = min(n_folds, n_estimators)
    n_samples = len(X)

    def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
        seed = int(rng.integers(0, 2**31))
        model = model_cls(n_estimators=n_tree_estimators, random_state=seed, n_jobs=-1)
        model.fit(X[train_idx], y[train_idx])
        result = permutation_importance(
            model,
            X[val_idx],
            y[val_idx],
            n_repeats=5,
            random_state=int(rng.integers(0, 2**31)),
            n_jobs=-1,
        )
        return np.argsort(result.importances_mean)[::-1].copy()

    orderings: list[np.ndarray] = []

    if n_samples <= max_samples:
        if task_type == "classifier":
            from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

            kf: object = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=int(rng.integers(0, 2**31)),
            )
        else:
            from sklearn.model_selection import KFold  # noqa: PLC0415

            kf = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=int(rng.integers(0, 2**31)),
            )
        for train_idx, val_idx in kf.split(X, y):  # type: ignore[union-attr]
            orderings.append(_fit_fold(train_idx, val_idx))
    else:
        # Custom folds: fixed val_size drawn in a stratified manner (for
        # classifiers), train fills the remaining budget with random rows.
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        val_size = max_samples // n_folds
        train_size = max_samples - val_size
        stratify = y if task_type == "classifier" else None
        all_idx = np.arange(n_samples)
        for _ in range(n_folds):
            val_idx, remaining = train_test_split(
                all_idx,
                train_size=val_size,
                stratify=stratify,
                random_state=int(rng.integers(0, 2**31)),
            )
            n_train = min(train_size, len(remaining))
            train_idx = rng.choice(remaining, n_train, replace=False)
            orderings.append(_fit_fold(train_idx, val_idx))

    return [orderings[i % n_folds] for i in range(n_estimators)]


def compute_feature_importance_order(
    X: np.ndarray,
    y: np.ndarray,
    task_type: Literal["classifier", "regressor"],
    *,
    method: FeatureSubsamplingMethod = FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE,
    n_estimators: int,
    n_folds: int | None = None,
    gini_max_samples: int = GINI_FEATURE_IMPORTANCE_MAX_SAMPLES,
    permutation_max_samples: int = PERMUTATION_FEATURE_IMPORTANCE_MAX_SAMPLES,
    n_tree_estimators: int = 50,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Rank features by importance, returning one ordering per TabPFN estimator.

    The returned list always has length ``n_estimators``.  When fewer distinct
    orderings are computed than there are estimators the list is filled by
    cycling through the available orderings.

    Args:
        X: Training features, shape (n_samples, n_features).
        y: Training targets, shape (n_samples,).
        task_type: ``"classifier"`` or ``"regressor"`` (matches TabPFN estimator_type).
        method: Feature-importance method to use.

            ``GINI_FEATURE_IMPORTANCE``: fits an ExtraTrees ensemble on the data.
            When ``n_samples <= gini_max_samples`` a single fit is performed and its
            ordering is repeated for all estimators.  When the data is larger,
            ``min(n_estimators, n_samples // gini_max_samples + 1)`` independent
            subsamples of size ``gini_max_samples`` are drawn, giving a different
            ordering per estimator (cycled when fewer subsamples than estimators).

            ``PERMUTATION_FEATURE_IMPORTANCE``: uses permutation importance over
            cross-validation folds.  The number of folds is
            ``min(n_folds, n_estimators)``.  When ``n_samples >
            permutation_max_samples`` a custom fold scheme is used: each fold draws
            a fixed-size validation set of ``permutation_max_samples // n_folds``
            rows and fills the rest of the budget with training data, keeping
            validation cost constant regardless of dataset size.

        n_estimators: Number of TabPFN ensemble estimators.  The returned list
            has exactly this length.
        n_folds: Number of cross-validation folds for
            ``PERMUTATION_FEATURE_IMPORTANCE``.  Defaults to
            ``PERMUTATION_FEATURE_IMPORTANCE_N_FOLDS``, further capped at
            ``n_estimators``.  Ignored for ``GINI_FEATURE_IMPORTANCE``.
        gini_max_samples: Row budget per ExtraTrees fit (gini method).
        permutation_max_samples: Total row budget for permutation-importance
            folds.  Controls both the subsampling threshold and the per-fold
            train/val split sizes.
        n_tree_estimators: Number of trees in the ExtraTrees model used by both
            methods.
        rng: Random number generator.

    Returns:
        List of length ``n_estimators``, each element an array of feature indices
        sorted from most to least important.
    """
    if method == FeatureSubsamplingMethod.GINI_FEATURE_IMPORTANCE:
        return _compute_gini_importance(
            X=X,
            y=y,
            task_type=task_type,
            n_tree_estimators=n_tree_estimators,
            n_estimators=n_estimators,
            max_samples=gini_max_samples,
            rng=rng,
        )

    if method == FeatureSubsamplingMethod.PERMUTATION_FEATURE_IMPORTANCE:
        resolved_folds = (
            n_folds if n_folds is not None else PERMUTATION_FEATURE_IMPORTANCE_N_FOLDS
        )
        return _compute_permutation_importance(
            X=X,
            y=y,
            task_type=task_type,
            n_folds=resolved_folds,
            n_tree_estimators=n_tree_estimators,
            n_estimators=n_estimators,
            max_samples=permutation_max_samples,
            rng=rng,
        )

    raise ValueError(f"Unsupported feature importance method: {method!r}")


def generate_classification_ensemble_configs(
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
            target_transform=target_transform,
            outlier_removal_std=outlier_removal_std,
            _model_index=model_index,
        )
        for featshift, (
            preprocess_config,
            target_transform,
        ), model_index in zip(
            featshifts,
            configs_,
            model_indices,
        )
    ]
