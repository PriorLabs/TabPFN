"""Defines the preprocessing configurations that define the ensembling of
different members.
"""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import chain, product, repeat
from typing import TYPE_CHECKING, Literal, TypeVar
from typing_extensions import override

import numpy as np
from sklearn.utils.validation import joblib

from tabpfn.constants import (
    CLASS_SHUFFLE_OVERESTIMATE_FACTOR,
    MAXIMUM_FEATURE_SHIFT,
    PARALLEL_MODE_TO_RETURN_AS,
    SUPPORTS_RETURN_AS,
)
from tabpfn.model.preprocessing import (
    AddFingerprintFeaturesStep,
    EncodeCategoricalFeaturesStep,
    FeaturePreprocessingTransformerStep,
    NanHandlingPolynomialFeaturesStep,
    RemoveConstantFeaturesStep,
    ReshapeFeatureDistributionsStep,
    SequentialFeatureTransformer,
    ShuffleFeaturesStep,
)
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline

T = TypeVar("T")


def balance(x: Iterable[T], n: int) -> list[T]:
    """Take a list of elements and make a new list where each appears `n` times."""
    return list(chain.from_iterable(repeat(elem, n) for elem in x))


@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessors.

    Attributes:
        name: Name of the preprocessor.
        categorical_name:
            Name of the categorical encoding method.
            Options: "none", "numeric", "onehot", "ordinal", "ordinal_shuffled", "none".
        append_original: Whether to append original features to the transformed features
        subsample_features: Fraction of features to subsample. -1 means no subsampling.
        global_transformer_name: Name of the global transformer to use.
    """

    name: Literal[
        "per_feature",  # a different transformation for each feature
        "power",  # a standard sklearn power transformer
        "safepower",  # a power transformer that prevents some numerical issues
        "power_box",
        "safepower_box",
        "quantile_uni_coarse",  # quantile transformations with few quantiles up to many
        "quantile_norm_coarse",
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_fine",
        "quantile_norm_fine",
        "robust",  # a standard sklearn robust scaler
        "kdi",
        "none",  # no transformation (only standardization in transformer)
        "kdi_random_alpha",
        "kdi_uni",
        "kdi_random_alpha_uni",
        "adaptive",
        "norm_and_kdi",
        # KDI with alpha collection
        "kdi_alpha_0.3_uni",
        "kdi_alpha_0.5_uni",
        "kdi_alpha_0.8_uni",
        "kdi_alpha_1.0_uni",
        "kdi_alpha_1.2_uni",
        "kdi_alpha_1.5_uni",
        "kdi_alpha_2.0_uni",
        "kdi_alpha_3.0_uni",
        "kdi_alpha_5.0_uni",
        "kdi_alpha_0.3",
        "kdi_alpha_0.5",
        "kdi_alpha_0.8",
        "kdi_alpha_1.0",
        "kdi_alpha_1.2",
        "kdi_alpha_1.5",
        "kdi_alpha_2.0",
        "kdi_alpha_3.0",
        "kdi_alpha_5.0",
    ]
    categorical_name: Literal[
        # categorical features are pretty much treated as ordinal, just not resorted
        "none",
        # categorical features are treated as numeric,
        # that means they are also power transformed for example
        "numeric",
        # "onehot": categorical features are onehot encoded
        "onehot",
        # "ordinal": categorical features are sorted and encoded as
        # integers from 0 to n_categories - 1
        "ordinal",
        # "ordinal_shuffled": categorical features are encoded as integers
        # from 0 to n_categories - 1 in a random order
        "ordinal_shuffled",
        "ordinal_very_common_categories_shuffled",
    ] = "none"
    append_original: bool = False
    subsample_features: float = -1
    global_transformer_name: str | None = None

    @override
    def __str__(self) -> str:
        return (
            f"{self.name}_cat:{self.categorical_name}"
            + ("_and_none" if self.append_original else "")
            + (
                f"_subsample_feats_{self.subsample_features}"
                if self.subsample_features > 0
                else ""
            )
            + (
                f"_global_transformer_{self.global_transformer_name}"
                if self.global_transformer_name is not None
                else ""
            )
        )


def default_classifier_preprocessor_configs() -> list[PreprocessorConfig]:
    """Default preprocessor configurations for classification."""
    return [
        PreprocessorConfig(
            "quantile_uni_coarse",
            append_original=True,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
            subsample_features=-1,
        ),
        PreprocessorConfig(
            "none",
            categorical_name="numeric",
            subsample_features=-1,
        ),
    ]


def default_regressor_preprocessor_configs() -> list[PreprocessorConfig]:
    """Default preprocessor configurations for regression."""
    return [
        PreprocessorConfig(
            "quantile_uni",
            append_original=True,
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
        ),
        PreprocessorConfig("safepower", categorical_name="onehot"),
    ]


def generate_index_permutations(
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
        random_state: Random number generator.

    Returns:
        List of indices to subsample.
    """
    _, rng = infer_random_state(random_state)
    if isinstance(subsample, int):
        if not (1 <= subsample <= max_index):
            raise ValueError(f"{subsample=} must be in [1, {max_index}] if int")

        return [rng.permutation(max_index)[:subsample] for _ in range(n)]

    if isinstance(subsample, float):
        if not (0 < subsample < 1):
            raise ValueError(f"{subsample=} must be in (0, 1) if float")
        subsample = int(subsample * max_index) + 1
        return [rng.permutation(max_index)[:subsample] for _ in range(n)]

    raise ValueError(f"{subsample=} must be int or float.")


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble member.

    Attributes:
        feature_shift_count: How much to shift the features columns.
        class_permutation: Permutation to apply to classes
        preprocess_config: Preprocessor configuration to use.
        subsample_ix: Indices of samples to use for this ensemble member.
            If `None`, no subsampling is done.
    """

    preprocess_config: PreprocessorConfig
    add_fingerprint_feature: bool
    polynomial_features: Literal["no", "all"] | int
    feature_shift_count: int
    feature_shift_decoder: Literal["shuffle", "rotate"] | None
    subsample_ix: npt.NDArray[np.int64] | None  # OPTIM: Could use uintp

    @classmethod
    def generate_for_classification(
        cls,
        *,
        n: int,
        subsample_size: int | float | None,
        max_index: int,
        add_fingerprint_feature: bool,
        polynomial_features: Literal["no", "all"] | int,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None,
        preprocessor_configs: Sequence[PreprocessorConfig],
        class_shift_method: Literal["rotate", "shuffle"] | None,
        n_classes: int,
        random_state: int | np.random.Generator | None,
    ) -> list[ClassifierEnsembleConfig]:
        """Generate ensemble configurations for classification.

        Args:
            n: Number of ensemble configurations to generate.
            subsample_size:
                Number of samples to subsample. If int, subsample that many
                samples. If float, subsample that fraction of samples. If `None`, no
                subsampling is done.
            max_index: Maximum index to generate for.
            add_fingerprint_feature: Whether to add fingerprint features.
            polynomial_features: Maximum number of polynomial features to add, if any.
            feature_shift_decoder: How shift features
            preprocessor_configs: Preprocessor configurations to use on the data.
            class_shift_method: How to shift classes for classpermutation.
            n_classes: Number of classes.
            random_state: Random number generator.

        Returns:
            List of ensemble configurations.
        """
        static_seed, rng = infer_random_state(random_state)
        start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
        featshifts = np.arange(start, start + n)
        featshifts = rng.choice(featshifts, size=n, replace=False)  # type: ignore

        if class_shift_method == "rotate":
            arange = np.arange(0, n_classes)
            shifts = rng.permutation(n_classes).tolist()
            class_permutations = [np.roll(arange, s) for s in shifts]
            class_permutations = [  # type: ignore
                class_permutations[c] for c in rng.choice(n_classes, n)
            ]
        elif class_shift_method == "shuffle":
            noise = rng.random((n * CLASS_SHUFFLE_OVERESTIMATE_FACTOR, n_classes))
            shufflings = np.argsort(noise, axis=1)
            uniqs = np.unique(shufflings, axis=0)
            balance_count = n // len(uniqs)
            class_permutations = balance(uniqs, balance_count)
            rand_count = n % len(uniqs)
            if rand_count > 0:
                class_permutations += [  # type: ignore
                    uniqs[i] for i in rng.choice(len(uniqs), size=rand_count)
                ]
        elif class_shift_method is None:
            class_permutations = [None] * n  # type: ignore
        else:
            raise ValueError(f"Unknown {class_shift_method=}")

        subsamples: list[None] | list[np.ndarray]
        if isinstance(subsample_size, (int, float)):
            subsamples = generate_index_permutations(
                n=n,
                max_index=max_index,
                subsample=subsample_size,
                random_state=static_seed,
            )
        elif subsample_size is None:
            subsamples = [None] * n  # type: ignore
        else:
            raise ValueError(
                f"Invalid subsample_samples: {subsample_size}",
            )

        balance_count = n // len(preprocessor_configs)

        # Replicate each config balance_count times
        configs_ = balance(preprocessor_configs, balance_count)

        # Number still needed to reach n
        leftover = n - len(configs_)

        if leftover > 0:
            # Randomly pick leftover items from *all* preprocessor configs
            picks = rng.choice(len(preprocessor_configs), size=leftover, replace=True)
            configs_.extend(preprocessor_configs[i] for i in picks)

        return [
            ClassifierEnsembleConfig(
                preprocess_config=preprocess_config,
                feature_shift_count=featshift,
                add_fingerprint_feature=add_fingerprint_feature,
                polynomial_features=polynomial_features,
                feature_shift_decoder=feature_shift_decoder,
                subsample_ix=subsample_ix,
                class_permutation=class_perm,
            )
            for featshift, preprocess_config, subsample_ix, class_perm in zip(
                featshifts,
                configs_,
                subsamples,
                class_permutations,
            )
        ]

    @classmethod
    def generate_for_regression(
        cls,
        *,
        n: int,
        subsample_size: int | float | None,
        max_index: int,
        add_fingerprint_feature: bool,
        polynomial_features: Literal["no", "all"] | int,
        feature_shift_decoder: Literal["shuffle", "rotate"] | None,
        preprocessor_configs: Sequence[PreprocessorConfig],
        target_transforms: Sequence[TransformerMixin | Pipeline | None],
        random_state: int | np.random.Generator | None,
    ) -> list[RegressorEnsembleConfig]:
        """Generate ensemble configurations for regression.

        Args:
            n: Number of ensemble configurations to generate.
            subsample_size:
                Number of samples to subsample. If int, subsample that many
                samples. If float, subsample that fraction of samples. If `None`, no
                subsampling is done.
            max_index: Maximum index to generate for.
            add_fingerprint_feature: Whether to add fingerprint features.
            polynomial_features: Maximum number of polynomial features to add, if any.
            feature_shift_decoder: How shift features
            preprocessor_configs: Preprocessor configurations to use on the data.
            target_transforms: Target transformations to apply.
            random_state: Random number generator.

        Returns:
            List of ensemble configurations.
        """
        static_seed, rng = infer_random_state(random_state)
        start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
        featshifts = np.arange(start, start + n)
        featshifts = rng.choice(featshifts, size=n, replace=False)  # type: ignore

        subsamples: list[None] | list[np.ndarray]
        if isinstance(subsample_size, (int, float)):
            subsamples = generate_index_permutations(
                n=n,
                max_index=max_index,
                subsample=subsample_size,
                random_state=static_seed,
            )
        elif subsample_size is None:
            subsamples = [None] * n
        else:
            raise ValueError(
                f"Invalid subsample_samples: {subsample_size}",
            )

        # Get equal representation of all preprocessor configs
        combos = list(product(preprocessor_configs, target_transforms))
        balance_count = n // len(combos)
        configs_ = balance(combos, balance_count)

        # Fill in the rest with random choices
        rand_count = n % len(combos)
        if rand_count > 0:
            configs_ += [combos[i] for i in rng.choice(len(combos), size=rand_count)]

        return [
            RegressorEnsembleConfig(
                preprocess_config=preprocess_config,
                feature_shift_count=featshift,
                add_fingerprint_feature=add_fingerprint_feature,
                polynomial_features=polynomial_features,
                feature_shift_decoder=feature_shift_decoder,
                subsample_ix=subsample_ix,
                target_transform=target_transform,
            )
            for featshift, subsample_ix, (preprocess_config, target_transform) in zip(
                featshifts,
                subsamples,
                configs_,
            )
        ]

    # TODO(eddiebergman): Make this sklearn pipeline
    def to_pipeline(
        self,
        *,
        random_state: int | np.random.Generator | None,
    ) -> SequentialFeatureTransformer:
        """Convert the ensemble configuration to a preprocessing pipeline."""
        steps: list[FeaturePreprocessingTransformerStep] = []

        if isinstance(self.polynomial_features, int):
            assert self.polynomial_features > 0, "Poly. features to add must be >0!"
            use_poly_features = True
            max_poly_features = self.polynomial_features
        elif self.polynomial_features == "all":
            use_poly_features = True
            max_poly_features = None
        elif self.polynomial_features == "no":
            use_poly_features = False
            max_poly_features = None
        else:
            raise ValueError(
                f"Invalid polynomial_features value: {self.polynomial_features}",
            )
        if use_poly_features:
            steps.append(
                NanHandlingPolynomialFeaturesStep(
                    max_features=max_poly_features,
                    random_state=random_state,
                ),
            )

        steps.extend(
            [
                RemoveConstantFeaturesStep(),
                ReshapeFeatureDistributionsStep(
                    transform_name=self.preprocess_config.name,
                    append_to_original=self.preprocess_config.append_original,
                    subsample_features=self.preprocess_config.subsample_features,
                    global_transformer_name=self.preprocess_config.global_transformer_name,
                    apply_to_categorical=(
                        self.preprocess_config.categorical_name == "numeric"
                    ),
                    random_state=random_state,
                ),
                EncodeCategoricalFeaturesStep(
                    self.preprocess_config.categorical_name,
                    random_state=random_state,
                ),
            ],
        )

        if self.add_fingerprint_feature:
            steps.append(AddFingerprintFeaturesStep(random_state=random_state))

        steps.append(
            ShuffleFeaturesStep(
                shuffle_method=self.feature_shift_decoder,
                shuffle_index=self.feature_shift_count,
                random_state=random_state,
            ),
        )
        return SequentialFeatureTransformer(steps)


@dataclass
class ClassifierEnsembleConfig(EnsembleConfig):
    """Configuration for a classifier ensemble member.

    See [EnsembleConfig][tabpfn.preprocessing.EnsembleConfig] for more details.
    """

    class_permutation: np.ndarray | None


@dataclass
class RegressorEnsembleConfig(EnsembleConfig):
    """Configuration for a regression ensemble member.

    See [EnsembleConfig][tabpfn.preprocessing.EnsembleConfig] for more details.
    """

    target_transform: TransformerMixin | Pipeline | None


def fit_preprocessing_one(
    config: EnsembleConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int | np.random.Generator | None = None,
    *,
    cat_ix: list[int],
) -> tuple[
    EnsembleConfig,
    SequentialFeatureTransformer,
    np.ndarray,
    np.ndarray,
    list[int],
]:
    """Fit preprocessing pipeline for a single ensemble configuration.

    Args:
        config: Ensemble configuration.
        X_train: Training data.
        y_train: Training target.
        random_state: Random seed.
        cat_ix: Indices of categorical features.

    Returns:
        Tuple containing the ensemble configuration, the fitted preprocessing pipeline,
        the transformed training data, the transformed target, and the indices of
        categorical features.
    """
    static_seed, _ = infer_random_state(random_state)
    if config.subsample_ix is not None:
        X_train = X_train[config.subsample_ix].copy()
        y_train = y_train[config.subsample_ix].copy()
    else:
        X_train = X_train.copy()
        y_train = y_train.copy()

    preprocessor = config.to_pipeline(random_state=static_seed)
    res = preprocessor.fit_transform(X_train, cat_ix)

    # TODO(eddiebergman): Not a fan of this, wish it was more transparent, but we want
    # to distuinguish what to do with the `ys` based on the ensemble config type
    if isinstance(config, RegressorEnsembleConfig):
        if config.target_transform is not None:
            # TODO(eddiebergman): Verify this transformer is fitted back in the main
            # process context, otherwise we need some way to return it, possibly
            # by just returning the config
            y_train = config.target_transform.fit_transform(
                y_train.reshape(-1, 1),
            ).ravel()
    elif isinstance(config, ClassifierEnsembleConfig):
        if config.class_permutation is not None:
            y_train = config.class_permutation[y_train]
    else:
        raise ValueError(f"Invalid ensemble config type: {type(config)}")

    return (config, preprocessor, res.X, y_train, res.categorical_features)


def fit_preprocessing(
    configs: Sequence[EnsembleConfig],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int | np.random.Generator | None,
    cat_ix: list[int],
    n_workers: int,  # noqa: ARG001
    parallel_mode: Literal["block", "as-ready", "in-order"],
) -> Iterator[
    tuple[
        EnsembleConfig,
        SequentialFeatureTransformer,
        np.ndarray,
        np.ndarray,
        list[int],
    ]
]:
    """Fit preprocessing pipelines in parallel.

    Args:
        configs: List of ensemble configurations.
        X_train: Training data.
        y_train: Training target.
        random_state: Random number generator.
        cat_ix: Indices of categorical features.
        n_workers: Number of workers to use.
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

    # TODO: It seems like we really don't benefit from much more than 1,2,4 workers,
    # even for the largest datasets from AutoMLBenchmark. Even then, the benefit is
    # marginal. For now, we stick with single worker.
    #
    # The parameters worth tuning are `batch_size` and `n_jobs`
    # * `n_jobs` - how many workers to spawn.
    # * `batch_size` - how many tasks to send to a worker at once.
    #
    # For small datasets (for which this model is built for), it's quite hard to tune
    # for increased performance and staying at 1 worker seems ideal. However for larger
    # datasets, at the limit of what we support, having `len(configs) // 2` workers
    # seemed good, with a `batch_size` of 2.
    # NOTE: By setting `n_jobs` = 1, it effectively doesn't spawn anything and runs
    # in-process
    if SUPPORTS_RETURN_AS:
        return_as = PARALLEL_MODE_TO_RETURN_AS[parallel_mode]
        executor = joblib.Parallel(
            n_jobs=1,
            return_as=return_as,
            batch_size="auto",  # type: ignore
        )
    else:
        executor = joblib.Parallel(
            n_jobs=1,
            batch_size="auto",  # type: ignore
        )
    func = partial(fit_preprocessing_one, cat_ix=cat_ix)
    worker_func = joblib.delayed(func)

    seeds = rng.integers(0, np.iinfo(np.int32).max, len(configs))
    yield from executor(  # type: ignore
        [
            worker_func(config, X_train, y_train, seed)
            for config, seed in zip(configs, seeds)
        ],
    )
