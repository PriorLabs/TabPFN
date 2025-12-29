from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from itertools import chain, product, repeat
from typing import TYPE_CHECKING, Callable, Literal, TypeVar

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn.constants import (
    CLASS_SHUFFLE_OVERESTIMATE_FACTOR,
    MAXIMUM_FEATURE_SHIFT,
    PARALLEL_MODE_TO_RETURN_AS,
    SUPPORTS_RETURN_AS,
)
from tabpfn.utils import infer_random_state

from .add_fingerprint_features_step import AddFingerprintFeaturesStep
from .definitions import (
    BaseDatasetConfig,
    ClassifierDatasetConfig,
    ClassifierEnsembleConfig,
    EnsembleConfig,
    PreprocessorConfig,
    RegressorDatasetConfig,
    RegressorEnsembleConfig,
)
from .differentiable_z_norm_step import DifferentiableZNormStep
from .encode_categorical_features_step import EncodeCategoricalFeaturesStep
from .nan_handling_polynomial_features_step import NanHandlingPolynomialFeaturesStep
from .preprocessing_helpers import FeaturePreprocessingTransformerStep, SequentialFeatureTransformer
from .remove_constant_features_step import RemoveConstantFeaturesStep
from .reshape_feature_distribution_step import ReshapeFeatureDistributionsStep
from .shuffle_features_step import ShuffleFeaturesStep

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline

T = TypeVar("T")

# --- helpers ---


def _balance(x: Iterable[T], n: int) -> list[T]:
    """Take a list of elements and make a new list where each appears `n` times."""

    return list(chain.from_iterable(repeat(elem, n) for elem in x))


# --- sampling ---


def generate_index_permutations(
    n: int,
    *,
    max_index: int,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
) -> list[npt.NDArray[np.int64]]:
    """Generate indices for subsampling from the data."""

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


def get_subsample_indices_for_estimators(
    subsample_samples: int | float | list[np.ndarray] | None,
    num_estimators: int,
    max_index: int,
    static_seed: int | np.random.Generator | None,
) -> list[None] | list[np.ndarray]:
    """Get the indices of the rows to subsample for each estimator."""

    if isinstance(subsample_samples, (int, float)):
        subsample_indices = generate_index_permutations(
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
            assert len(subsample) > 0, (
                "Length of subsampled indices must be larger than 0"
            )
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


# --- ensemble config generation ---


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
        * ``"rotate"`` – draw random circular shifts of ``np.arange(n_classes)``
          and sample from those shifts for each estimator.
        * ``"shuffle"`` – create random permutations of ``range(n_classes)``,
          deduplicate them, and balance their usage across estimators.
        * ``None`` – disable class permutation and return ``None`` entries.
    n_classes:
        Total number of distinct classes.
    rng:
        Numpy random generator used for reproducible permutations.

    Returns
    -------
    list[np.ndarray] | list[None]
        A list of permutations (or ``None`` entries) with length ``num_estimators``.
    """

    if class_shift_method == "rotate":
        arange = np.arange(0, n_classes)
        shifts = rng.permutation(n_classes).tolist()
        class_permutations = [np.roll(arange, s) for s in shifts]
        return [
            class_permutations[c] for c in rng.choice(n_classes, num_estimators)
        ]

    if class_shift_method == "shuffle":
        noise = rng.random((num_estimators * CLASS_SHUFFLE_OVERESTIMATE_FACTOR, n_classes))
        shufflings = np.argsort(noise, axis=1)
        uniqs = np.unique(shufflings, axis=0)
        balance_count = num_estimators // len(uniqs)
        class_permutations = _balance(uniqs, balance_count)
        rand_count = num_estimators % len(uniqs)
        if rand_count > 0:
            class_permutations += [uniqs[i] for i in rng.choice(len(uniqs), size=rand_count)]
        return class_permutations

    if class_shift_method is None:
        return [None] * num_estimators  # type: ignore[return-value]

    raise ValueError(f"Unknown {class_shift_method=}")


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
) -> list[ClassifierEnsembleConfig]:
    """Generate ensemble configurations for classification.

    Parameters
    ----------
    num_estimators:
        Number of ensemble configurations to generate.
    subsample_samples:
        Method to subsample rows. If ``int``, subsample that many samples. If
        ``float``, subsample that fraction of samples. If a list of lists of
        indices, subsample the indices for each estimator. If ``None``, no
        subsampling is done.
    max_index:
        Maximum index to generate for.
    add_fingerprint_feature:
        Whether to add fingerprint features.
    polynomial_features:
        Maximum number of polynomial features to add, if any.
    feature_shift_decoder:
        How to shift features.
    preprocessor_configs:
        Preprocessor configurations to use on the data.
    class_shift_method:
        How to shift classes for class permutation.
    n_classes:
        Number of classes.
    random_state:
        Random number generator or seed.
    num_models:
        Number of models to use.

    Returns
    -------
    list[ClassifierEnsembleConfig]
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
        get_subsample_indices_for_estimators(
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
            class_permutation=class_permutation,
            add_fingerprint_feature=add_fingerprint_feature,
            polynomial_features=polynomial_features,
            feature_shift_decoder=feature_shift_decoder,
            subsample_ix=subsample_ix,
            _model_index=model_index,
        )
        for featshift, subsample_ix, preprocesses_config, class_permutation, model_index in zip(
            featshifts,
            subsample_indices,
            configs_,
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
) -> list[RegressorEnsembleConfig]:
    """Generate ensemble configurations for regression.

    Parameters
    ----------
    num_estimators:
        Number of ensemble configurations to generate.
    subsample_samples:
        Method to subsample rows. If ``int``, subsample that many samples. If
        ``float``, subsample that fraction of samples. If a list of lists of
        indices, subsample the indices for each estimator. If ``None``, no
        subsampling is done.
    max_index:
        Maximum index to generate for.
    add_fingerprint_feature:
        Whether to add fingerprint features.
    polynomial_features:
        Maximum number of polynomial features to add, if any.
    feature_shift_decoder:
        How to shift features.
    preprocessor_configs:
        Preprocessor configurations to use on the data.
    target_transforms:
        Target transformations to apply.
    random_state:
        Random number generator or seed.
    num_models:
        Number of models to use.

    Returns
    -------
    list[RegressorEnsembleConfig]
        List of ensemble configurations.
    """

    static_seed, rng = infer_random_state(random_state)
    start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
    featshifts = np.arange(start, start + num_estimators)
    featshifts = rng.choice(featshifts, size=num_estimators, replace=False)  # type: ignore[arg-type]

    subsample_indices: list[None] | list[np.ndarray] = (
        get_subsample_indices_for_estimators(
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
            _model_index=model_index,
        )
        for featshift, subsample_ix, (preprocess_config, target_transform), model_index in zip(
            featshifts,
            subsample_indices,
            configs_,
            model_indices,
        )
    ]


# --- pipeline construction ---


def _polynomial_feature_settings(polynomial_features: Literal["no", "all"] | int) -> tuple[bool, int | None]:
    if isinstance(polynomial_features, int):
        assert polynomial_features > 0, "Poly. features to add must be >0!"
        return True, polynomial_features
    if polynomial_features == "all":
        return True, None
    if polynomial_features == "no":
        return False, None
    raise ValueError(f"Invalid polynomial_features value: {polynomial_features}")


def build_pipeline(
    config: EnsembleConfig,
    *,
    random_state: int | np.random.Generator | None,
) -> SequentialFeatureTransformer:
    """Convert the ensemble configuration to a preprocessing pipeline."""

    steps: list[FeaturePreprocessingTransformerStep] = []

    use_poly_features, max_poly_features = _polynomial_feature_settings(
        config.polynomial_features
    )
    if use_poly_features:
        steps.append(
            NanHandlingPolynomialFeaturesStep(
                max_features=max_poly_features,
                random_state=random_state,
            ),
        )

    steps.append(RemoveConstantFeaturesStep())

    if config.preprocess_config.differentiable:
        steps.append(DifferentiableZNormStep())
    else:
        steps.extend(
            [
                ReshapeFeatureDistributionsStep(
                    transform_name=config.preprocess_config.name,
                    append_to_original=config.preprocess_config.append_original,
                    max_features_per_estimator=config.preprocess_config.max_features_per_estimator,
                    global_transformer_name=config.preprocess_config.global_transformer_name,
                    apply_to_categorical=(
                        config.preprocess_config.categorical_name == "numeric"
                    ),
                    random_state=random_state,
                ),
                EncodeCategoricalFeaturesStep(
                    config.preprocess_config.categorical_name,
                    random_state=random_state,
                ),
            ],
        )

    if config.add_fingerprint_feature:
        steps.append(AddFingerprintFeaturesStep(random_state=random_state))

    steps.append(
        ShuffleFeaturesStep(
            shuffle_method=config.feature_shift_decoder,
            shuffle_index=config.feature_shift_count,
            random_state=random_state,
        ),
    )
    return SequentialFeatureTransformer(steps)


# --- fitting/orchestration ---


def fit_preprocessing_one(
    config: EnsembleConfig,
    X_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
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
    """Fit preprocessing pipeline for a single ensemble configuration."""

    static_seed, _ = infer_random_state(random_state)
    if config.subsample_ix is not None:
        X_train = X_train[config.subsample_ix]
        y_train = y_train[config.subsample_ix]
    if not isinstance(X_train, torch.Tensor):
        X_train = X_train.copy()
        y_train = y_train.copy()

    preprocessor = build_pipeline(config, random_state=static_seed)
    res = preprocessor.fit_transform(X_train, cat_ix)

    y_train_processed = transform_labels_one(config, y_train)

    return (config, preprocessor, res.X, y_train_processed, res.categorical_features)


def transform_labels_one(
    config: EnsembleConfig, y_train: np.ndarray | torch.Tensor
) -> np.ndarray:
    """Transform the labels for one ensemble config."""

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
    cat_ix: list[int],
    n_preprocessing_jobs: int,
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
    """Fit preprocessing pipelines in parallel."""

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
    func = partial(fit_preprocessing_one, cat_ix=cat_ix)
    worker_func = joblib.delayed(func)

    seeds = rng.integers(0, np.iinfo(np.int32).max, len(configs))
    yield from executor(  # type: ignore[misc]
        [
            worker_func(config, X_train, y_train, seed)
            for config, seed in zip(configs, seeds)
        ],
    )


# --- dataset wrapper ---


class DatasetCollectionWithPreprocessing(Dataset):
    """Manages a collection of dataset configurations for lazy processing."""

    def __init__(
        self,
        split_fn: Callable,
        rng: np.random.Generator,
        dataset_config_collection: Sequence[BaseDatasetConfig],
        n_preprocessing_jobs: int = 1,
    ) -> None:
        """Initialize the dataset collection wrapper.

        Parameters
        ----------
        split_fn:
            Function that splits features and targets into train/test parts.
        rng:
            Random number generator used for preprocessing reproducibility.
        dataset_config_collection:
            Sequence of dataset configurations that will be lazily processed.
        n_preprocessing_jobs:
            Degree of parallelism when fitting preprocessing pipelines.
        """
        super().__init__()
        self.configs = dataset_config_collection
        self.split_fn = split_fn
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.rng = rng

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, index: int):  # noqa: ANN204
        """Return the lazily preprocessed dataset elements for one configuration.

        This method selects the configuration at ``index``, splits the raw
        features/targets using :attr:`split_fn`, applies the configured
        preprocessing (including optional standardization for regression), and
        converts arrays to :class:`torch.Tensor` objects.

        Parameters
        ----------
        index:
            Position of the dataset configuration within :attr:`configs`.

        Returns
        -------
        tuple
            A tuple containing the preprocessed tensors and any auxiliary
            information required by the task. For classification tasks the tuple
            includes ``X_trains_preprocessed``, ``X_tests_preprocessed``,
            ``y_trains_preprocessed``, ``y_test_raw``, ``cat_ixs``, and
            ``conf``. For regression tasks, standardized targets and
            normalization artifacts are returned alongside the test tensors.

        Raises
        ------
        IndexError
            If ``index`` is out of bounds.
        TypeError
            If the stored configuration type is unsupported.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds.")

        config = self.configs[index]

        if isinstance(config, RegressorDatasetConfig):
            conf = config.config
            x_full_raw = config.X_raw
            y_full_raw = config.y_raw
            cat_ix = config.cat_ix
            znorm_space_bardist_ = config.znorm_space_bardist_
        elif isinstance(config, ClassifierDatasetConfig):
            conf = config.config
            x_full_raw = config.X_raw
            y_full_raw = config.y_raw
            cat_ix = config.cat_ix
        else:
            raise TypeError(f"Invalid dataset config type: {type(config)}")

        regression_task = isinstance(config, RegressorDatasetConfig)

        x_train_raw, x_test_raw, y_train_raw, y_test_raw = self.split_fn(
            x_full_raw, y_full_raw
        )

        if regression_task:
            train_mean = np.mean(y_train_raw)
            train_std = np.std(y_train_raw)
            y_test_standardized = (y_test_raw - train_mean) / train_std
            y_train_standardized = (y_train_raw - train_mean) / train_std
            raw_space_bardist_ = FullSupportBarDistribution(
                znorm_space_bardist_.borders * train_std
                + train_mean  # Inverse normalization back to raw space
            ).float()

        y_train = y_train_standardized if regression_task else y_train_raw

        itr = fit_preprocessing(
            configs=conf,
            X_train=x_train_raw,
            y_train=y_train,
            random_state=self.rng,
            cat_ix=cat_ix,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            parallel_mode="block",
        )
        (
            configs,
            preprocessors,
            X_trains_preprocessed,
            y_trains_preprocessed,
            cat_ixs,
        ) = list(zip(*itr))
        X_trains_preprocessed = list(X_trains_preprocessed)
        y_trains_preprocessed = list(y_trains_preprocessed)

        X_tests_preprocessed = []
        for _, estim_preprocessor in zip(configs, preprocessors):
            X_tests_preprocessed.append(estim_preprocessor.transform(x_test_raw).X)

        for i in range(len(X_trains_preprocessed)):
            if not isinstance(X_trains_preprocessed[i], torch.Tensor):
                X_trains_preprocessed[i] = torch.as_tensor(
                    X_trains_preprocessed[i], dtype=torch.float32
                )
            if not isinstance(X_tests_preprocessed[i], torch.Tensor):
                X_tests_preprocessed[i] = torch.as_tensor(
                    X_tests_preprocessed[i], dtype=torch.float32
                )
            if not isinstance(y_trains_preprocessed[i], torch.Tensor):
                y_trains_preprocessed[i] = torch.as_tensor(
                    y_trains_preprocessed[i], dtype=torch.float32
                )

        if regression_task and not isinstance(y_test_standardized, torch.Tensor):
            y_test_standardized = torch.from_numpy(y_test_standardized)
            if torch.is_floating_point(y_test_standardized):
                y_test_standardized = y_test_standardized.float()
            else:
                y_test_standardized = y_test_standardized.long()

        x_test_raw = torch.from_numpy(x_test_raw)
        y_test_raw = torch.from_numpy(y_test_raw)

        if regression_task:
            return (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_preprocessed,
                y_test_standardized,
                cat_ixs,
                conf,
                raw_space_bardist_,
                znorm_space_bardist_,
                x_test_raw,
                y_test_raw,
            )

        return (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_raw,
            cat_ixs,
            conf,
        )


__all__ = [
    "DatasetCollectionWithPreprocessing",
    "build_pipeline",
    "fit_preprocessing",
    "fit_preprocessing_one",
    "generate_classification_ensemble_configs",
    "generate_index_permutations",
    "generate_regression_ensemble_configs",
    "get_subsample_indices_for_estimators",
    "transform_labels_one",
]
