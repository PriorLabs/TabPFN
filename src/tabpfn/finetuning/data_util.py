"""Utilities for data preparation used in fine-tuning wrappers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal
from typing_extensions import override

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
from tabpfn.preprocessing import (
    EnsembleConfig,
    fit_preprocessing,
)
from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.utils import infer_random_state, pad_tensors

if TYPE_CHECKING:
    from tabpfn.constants import XType, YType


@dataclass
class ClassifierBatch:
    """Batch data for classifier fine-tuning.

    Attributes:
        X_context: Preprocessed training features (list per estimator).
        X_query: Preprocessed test features (list per estimator).
        y_context: Preprocessed training targets (list per estimator).
        y_query: Raw test target tensor.
        cat_indices: Categorical feature indices (list per estimator).
        configs: Preprocessing configurations used for this batch.
    """

    X_context: list[torch.Tensor]
    X_query: list[torch.Tensor]
    y_context: list[torch.Tensor]
    y_query: torch.Tensor
    # In a single dataset sample, this is "per-estimator":
    #   list[list[int] | None]
    # After collation (batch_size datasets), categorical indices must be batched:
    #   list[list[list[int] | None]]
    # The batched structure is required by InferenceEngineBatchedNoPreprocessing.
    cat_indices: list[list[int] | None] | list[list[list[int] | None]]
    configs: list[Any]


@dataclass
class RegressorBatch:
    """Batch data for regressor fine-tuning.

    Attributes:
        X_context: Preprocessed training features (list per estimator).
        X_query: Preprocessed test features (list per estimator).
        y_context: Preprocessed standardized training targets (list per estimator).
        y_query: Standardized test target tensor.
        cat_indices: Categorical feature indices (list per estimator).
        configs: Preprocessing configurations used for this batch.
        raw_space_bardist: Bar distribution in raw (original) target space.
        znorm_space_bardist: Bar distribution in z-normalized target space.
        X_query_raw: Original unprocessed test features.
        y_query_raw: Original unprocessed test targets.
    """

    X_context: list[torch.Tensor]
    X_query: list[torch.Tensor]
    y_context: list[torch.Tensor]
    y_query: torch.Tensor
    # See ClassifierBatch.cat_indices for the rationale of this union type.
    cat_indices: list[list[int] | None] | list[list[list[int] | None]]
    configs: list[Any]
    raw_space_bardist: FullSupportBarDistribution
    znorm_space_bardist: FullSupportBarDistribution
    X_query_raw: torch.Tensor
    y_query_raw: torch.Tensor


@dataclass
class BaseDatasetConfig:
    """Base configuration class for holding dataset specifics."""

    config: EnsembleConfig
    X_raw: np.ndarray | torch.Tensor
    y_raw: np.ndarray | torch.Tensor
    cat_ix: list[int]


@dataclass
class ClassifierDatasetConfig(BaseDatasetConfig):
    """Classification Dataset + Model Configuration class."""


@dataclass
class RegressorDatasetConfig(BaseDatasetConfig):
    """Regression Dataset + Model Configuration class."""

    znorm_space_bardist_: FullSupportBarDistribution | None = field(default=None)

    @property
    def bardist_(self) -> FullSupportBarDistribution:
        """DEPRECATED: Accessing `bardist_` is deprecated.
        Use `znorm_space_bardist_` instead.
        """
        warnings.warn(
            "`bardist_` is deprecated and will be removed in a future version. "
            "Please use `znorm_space_bardist_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.znorm_space_bardist_

    @bardist_.setter
    def bardist_(self, value: FullSupportBarDistribution) -> None:
        """DEPRECATED: Setting `bardist_` is deprecated.
        Use `znorm_space_bardist_`.
        """
        warnings.warn(
            "`bardist_` is deprecated and will be removed in a future version. "
            "Please use `znorm_space_bardist_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.znorm_space_bardist_ = value


def _take(obj: Any, idx: np.ndarray) -> Any:
    """Index obj by idx using .iloc when available (for pd.DataFrame), otherwise []."""
    return obj.iloc[idx] if hasattr(obj, "iloc") else obj[idx]


def _chunk_data_non_stratified(
    X_shuffled: XType,
    y_shuffled: YType,
    *,
    max_chunk_size: int,
    equal_split_size: bool,
    min_chunk_size: int,
) -> tuple[list[XType], list[YType]]:
    """Split shuffled data into chunks without stratification.

    Args:
        X_shuffled: Shuffled features.
        y_shuffled: Shuffled targets.
        max_chunk_size: Maximum size for any chunk.
        equal_split_size: If True, produce equally sized chunks (all
            <= max_chunk_size). Otherwise, make fixed-size chunks of
            max_chunk_size, keeping a remainder chunk only if its size
            >= min_chunk_size.
        min_chunk_size: Minimum size for any chunk when using remainder logic.

    Returns:
        Two lists with chunks of X and y respectively.
    """
    tot_size = len(X_shuffled)
    if equal_split_size:
        num_chunks = ((tot_size - 1) // max_chunk_size) + 1
        indices_per_chunk = np.array_split(np.arange(tot_size), num_chunks)

        X_chunks: list[XType] = [_take(X_shuffled, idx) for idx in indices_per_chunk]
        y_chunks: list[YType] = [_take(y_shuffled, idx) for idx in indices_per_chunk]
        return X_chunks, y_chunks

    full_chunks = tot_size // max_chunk_size
    remainder = tot_size % max_chunk_size
    if full_chunks == 0:
        if remainder >= min_chunk_size:
            return [X_shuffled], [y_shuffled]
        return [], []

    positions = np.arange(tot_size)
    pos_parts = [
        positions[i * max_chunk_size : (i + 1) * max_chunk_size]
        for i in range(full_chunks)
    ]

    if remainder >= min_chunk_size:
        pos_parts.append(positions[full_chunks * max_chunk_size :])

    X_chunks = [_take(X_shuffled, pos) for pos in pos_parts]
    y_chunks = [_take(y_shuffled, pos) for pos in pos_parts]

    return X_chunks, y_chunks


def _chunk_data_stratified(
    X_shuffled: XType,
    y_shuffled: YType,
    *,
    max_chunk_size: int,
    equal_split_size: bool,
    min_chunk_size: int,
    seed: int,
) -> tuple[list[XType], list[YType]]:
    """Split shuffled data into chunks using StratifiedKFold for classification.

    Falls back to non-stratified splitting if stratification is not feasible
    (e.g., some class has fewer samples than the required number of splits).

    Args:
        X_shuffled: Shuffled features.
        y_shuffled: Shuffled class labels.
        max_chunk_size: Maximum size for any chunk.
        equal_split_size: If True, produce equally sized chunks (all
            <= max_chunk_size). Otherwise, make fixed-size chunks of
            max_chunk_size and consider a remainder chunk only if its size
            >= min_chunk_size.
        min_chunk_size: Minimum size for any chunk when using remainder logic.
        seed: Random seed used by StratifiedKFold.

    Returns:
        Two lists with chunks of X and y respectively.
    """
    tot_size = len(X_shuffled)
    if equal_split_size:
        num_chunks = ((tot_size - 1) // max_chunk_size) + 1
    else:
        if tot_size < max_chunk_size:
            if tot_size >= min_chunk_size:
                return [X_shuffled], [y_shuffled]
            return [], []
        full_chunks = tot_size // max_chunk_size
        remainder = tot_size % max_chunk_size
        num_chunks = full_chunks + (1 if remainder >= min_chunk_size else 0)

    if num_chunks <= 1:
        return [X_shuffled], [y_shuffled]

    y_values = (
        y_shuffled.to_numpy() if isinstance(y_shuffled, pd.Series) else y_shuffled
    )
    _, counts = np.unique(y_values, return_counts=True)
    min_class_count = int(counts.min()) if len(counts) > 0 else 0

    if min_class_count >= num_chunks:
        skf = StratifiedKFold(
            n_splits=num_chunks,
            shuffle=True,
            random_state=seed,
        )
        folds = [test_idx for _, test_idx in skf.split(np.zeros(tot_size), y_values)]
        X_chunks: list[XType] = [_take(X_shuffled, idx) for idx in folds]
        y_chunks: list[YType] = [_take(y_shuffled, idx) for idx in folds]
        return X_chunks, y_chunks

    # Fallback if some classes are too small for the requested number of splits.
    return _chunk_data_non_stratified(
        X_shuffled,
        y_shuffled,
        max_chunk_size=max_chunk_size,
        equal_split_size=equal_split_size,
        min_chunk_size=min_chunk_size,
    )


class DatasetCollectionWithPreprocessing(torch.utils.data.Dataset):
    """Manages a collection of dataset configurations for lazy processing.

    This class acts as a meta-dataset where each item corresponds to a
    single, complete dataset configuration (e.g., raw features, raw labels,
    preprocessing details defined in `RegressorDatasetConfig` or
    `ClassifierDatasetConfig`). When an item is accessed via `__getitem__`,
    it performs the following steps on the fly:

    1.  Retrieves the specified dataset configuration.
    2.  Splits the raw data into training and testing sets using the provided
        `split_fn` and a random seed derived from `rng`. For regression,
        both raw and pre-standardized targets might be split.
    3.  Fits preprocessors (defined in the dataset configuration's `config`
        attribute) on the *training* data using the `fit_preprocessing`
        utility. This may result in multiple preprocessed versions
        if the configuration specifies an ensemble of preprocessing pipelines.
        For regression we also standardise the target variable.
    4.  Applies the fitted preprocessors to the *testing* features (`x_test_raw`).
    5.  Converts relevant outputs to `torch.Tensor` objects.
    6.  Returns the preprocessed data splits along with other relevant
        information (like raw test data, configs) as a tuple.

    This approach is memory-efficient, especially when dealing with many
    datasets or configurations, as it avoids loading and preprocessing
    everything simultaneously.

    Args:
        split_fn (Callable): A function compatible with scikit-learn's
            `train_test_split` signature (e.g.,
            `sklearn.model_selection.train_test_split`). It's used to split
            the raw data (X, y) into train and test sets. It will receive
            `X`, `y`, `random_state` and (optional) `stratify` as arguments.
        rng: A NumPy random number generator instance
            used for generating the split seed and potentially within the
            preprocessing steps defined in the configs.
        dataset_config_collection: A sequence containing dataset configuration objects.
            Each object must hold the raw data (`X_raw`, `y_raw`), categorical feature
            indices (`cat_ix`), and the specific preprocessing configurations
            (`config`) for that dataset. Regression configs require additional
            fields (`znorm_space_bardist_`).
        n_preprocessing_jobs: The number of workers to use for potentially parallelized
            preprocessing steps (passed to `fit_preprocessing`).
        stratify: Whether to stratify the data by the target variable.
            Only used for classification tasks.

    Attributes:
        configs (Sequence[Union[RegressorDatasetConfig, ClassifierDatasetConfig]]):
            Stores the input dataset configuration collection.
        split_fn (Callable): Stores the splitting function.
        random_state (int | np.random.Generator): Stores the random number generator.
            If int, the preprocessing will always use the same random seed.
        n_preprocessing_jobs (int): The number of worker processes that will be used for
            the preprocessing.
        stratify (bool): Whether to stratify the data when splitting with split_fn.
    """

    def __init__(
        self,
        split_fn: Callable,
        random_state: np.random.Generator | int,
        dataset_config_collection: Sequence[
            RegressorDatasetConfig | ClassifierDatasetConfig
        ],
        n_preprocessing_jobs: int = 1,
        *,
        stratify: bool = False,
    ) -> None:
        self.configs = dataset_config_collection
        self.split_fn = split_fn
        self.random_state = random_state
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.stratify = stratify

    def __len__(self) -> int:
        return len(self.configs)

    @override
    def __getitem__(self, index: int) -> ClassifierBatch | RegressorBatch:  # noqa: C901, PLR0912
        """Retrieves, splits, and preprocesses the dataset config at the index.

        Performs train/test splitting and applies potentially multiple
        preprocessing pipelines defined in the dataset's configuration.

        Args:
            index: The index of the dataset configuration in the
                `dataset_config_collection` to process.

        Returns:
            A ClassifierBatch or RegressorBatch dataclass containing the
            processed data and metadata. Each list field has length equal to
            the number of estimators in the TabPFN ensemble.

            For **Classification** tasks: Returns a ClassifierBatch with:
                - X_context: Preprocessed training features (per estimator)
                - X_query: Preprocessed test features (per estimator)
                - y_context: Preprocessed training targets (per estimator)
                - y_query: Raw test target tensor
                - cat_indices: Categorical feature indices (per estimator)
                - configs: Preprocessing configurations used

            For **Regression** tasks: Returns a RegressorBatch with all
            ClassifierBatch fields plus:
                - raw_space_bardist: Bar distribution in raw target space
                - znorm_space_bardist: Bar distribution in z-normalized space
                - X_query_raw: Original unprocessed test features
                - y_query_raw: Original unprocessed test targets

        Raises:
            IndexError: If the index is out of the bounds of the dataset
                collection.
            ValueError: If the dataset configuration type at the index is not
                recognized.
            AssertionError: If sanity checks during processing fail.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of bounds.")

        config = self.configs[index]

        is_regression_task = isinstance(config, RegressorDatasetConfig)

        # Check type of Dataset Config
        if is_regression_task:
            conf = config.config
            x_full_raw = config.X_raw
            y_full_raw = config.y_raw
            cat_ix = config.cat_ix
            znorm_space_bardist_ = config.znorm_space_bardist_
        else:
            assert isinstance(config, ClassifierDatasetConfig), (
                "Invalid dataset config type"
            )
            conf = config.config
            x_full_raw = config.X_raw
            y_full_raw = config.y_raw
            cat_ix = config.cat_ix

        stratify_y = y_full_raw if not is_regression_task and self.stratify else None
        x_train_raw, x_test_raw, y_train_raw, y_test_raw = self.split_fn(
            x_full_raw, y_full_raw, stratify=stratify_y
        )

        # Compute target variable Z-transform standardization
        # based on statistics of training set
        # Note: Since we compute raw_space_bardist_ here,
        # it is not set as an attribute of the Regressor class
        # This however makes also sense when considering that
        # this attribute changes on every dataset
        if is_regression_task:
            train_mean = np.mean(y_train_raw)
            train_std = np.std(y_train_raw)

            eps = 1e-8
            if train_std < eps:
                warnings.warn(
                    f"Target variable has constant or near-constant values "
                    f"(std={train_std:.2e}). Adding epsilon={eps} to prevent "
                    f"division by zero in standardization.",
                    UserWarning,
                    stacklevel=2,
                )
                train_std = eps

            y_test_standardized = (y_test_raw - train_mean) / train_std
            y_train_standardized = (y_train_raw - train_mean) / train_std
            raw_space_bardist_ = FullSupportBarDistribution(
                znorm_space_bardist_.borders * train_std
                + train_mean  # Inverse normalization back to raw space
            ).float()
            y_train = y_train_standardized
        else:
            y_train = y_train_raw

        num_columns = x_train_raw.shape[1]
        feature_schema = FeatureSchema.from_only_categorical_indices(
            cat_ix, num_columns
        )
        preprocessing_iterator = fit_preprocessing(
            configs=conf,
            X_train=x_train_raw,
            y_train=y_train,
            feature_schema=feature_schema,
            random_state=self.random_state,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            parallel_mode="block",
        )
        (
            configs,
            preprocessors,
            X_trains_preprocessed,
            y_trains_preprocessed,
            feature_schema_preprocessed,
        ) = list(zip(*preprocessing_iterator))
        X_trains_preprocessed = list(X_trains_preprocessed)
        y_trains_preprocessed = list(y_trains_preprocessed)

        ## Process test data for all ensemble estimators.
        X_tests_preprocessed = []
        for _, estim_preprocessor in zip(configs, preprocessors):
            X_tests_preprocessed.append(estim_preprocessor.transform(x_test_raw).X)

        ## Convert to tensors.
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

        if is_regression_task and not isinstance(y_test_standardized, torch.Tensor):
            y_test_standardized = torch.from_numpy(y_test_standardized)
            if torch.is_floating_point(y_test_standardized):
                y_test_standardized = y_test_standardized.float()
            else:
                y_test_standardized = y_test_standardized.long()

        x_test_raw = torch.from_numpy(x_test_raw)
        y_test_raw = torch.from_numpy(y_test_raw)

        cat_indices = [
            m.indices_for(FeatureModality.CATEGORICAL)
            for m in feature_schema_preprocessed
        ]
        # Return structured batch data using dataclasses for clarity
        if is_regression_task:
            return RegressorBatch(
                X_context=X_trains_preprocessed,
                X_query=X_tests_preprocessed,
                y_context=y_trains_preprocessed,
                y_query=y_test_standardized,
                cat_indices=cat_indices,
                configs=list(conf),
                raw_space_bardist=raw_space_bardist_,
                znorm_space_bardist=znorm_space_bardist_,
                X_query_raw=x_test_raw,
                y_query_raw=y_test_raw,
            )

        return ClassifierBatch(
            X_context=X_trains_preprocessed,
            X_query=X_tests_preprocessed,
            y_context=y_trains_preprocessed,
            y_query=y_test_raw,
            cat_indices=cat_indices,
            configs=list(conf),
        )


def _collate_list_field(
    batch: list,
    field_name: str,
    num_estimators: int,
    padding_val: float,
) -> list:
    """Collate a list field (per-estimator data) from batch items."""
    batch_sz = len(batch)
    field_values = [getattr(b, field_name) for b in batch]
    estim_list = []
    for estim_no in range(num_estimators):
        if isinstance(field_values[0][0], torch.Tensor):
            labels = field_values[0][0].ndim == 1
            estim_list.append(
                torch.stack(
                    pad_tensors(
                        [field_values[r][estim_no] for r in range(batch_sz)],
                        padding_val=padding_val,
                        labels=labels,
                    )
                )
            )
        else:
            estim_list.append(
                list(field_values[r][estim_no] for r in range(batch_sz))  # noqa: C400
            )
    return estim_list


def _collate_tensor_field(
    batch: list,
    field_name: str,
    padding_val: float,
) -> torch.Tensor:
    """Collate a tensor field from batch items."""
    batch_sz = len(batch)
    field_values = [getattr(b, field_name) for b in batch]
    labels = field_values[0].ndim == 1
    return torch.stack(
        pad_tensors(
            [field_values[r] for r in range(batch_sz)],
            padding_val=padding_val,
            labels=labels,
        )
    )


def _collate_cat_indices(
    batch: Sequence[ClassifierBatch | RegressorBatch],
) -> list[list[list[int] | None]]:
    """Collate cat indices into the batched shape expected by the batched executor.

    In fine-tuning, the batched inference engine expects categorical indices as:
        [dataset_batch][estimator][cat_index]

    Individual dataset samples carry categorical indices as:
        [estimator][cat_index]  (or None per estimator)
    """
    batched_cat_indices: list[list[list[int] | None]] = []
    for item in batch:
        cat_indices = item.cat_indices

        # Empty is unambiguous (no estimators / no categorical features).
        if len(cat_indices) == 0:
            batched_cat_indices.append([])
            continue

        # If the first element is a list of ints, it's the per-estimator form.
        first = cat_indices[0]
        if first is None:
            batched_cat_indices.append(cat_indices)  # type: ignore[arg-type]
            continue

        if len(first) == 0 or isinstance(first[0], int):
            batched_cat_indices.append(cat_indices)  # type: ignore[arg-type]
            continue

        # Otherwise it's already batched: [dataset_batch][estimator][...].
        # We only support batch_size=1 in this collator.
        assert len(cat_indices) == 1
        batched_cat_indices.append(cat_indices[0])  # type: ignore[index]

    return batched_cat_indices


def meta_dataset_collator(
    batch: list[ClassifierBatch | RegressorBatch],
    padding_val: float = 0.0,
) -> ClassifierBatch | RegressorBatch:
    """Collate function for torch.utils.data.DataLoader.

    Designed for batches from DatasetCollectionWithPreprocessing.
    Takes a list of dataset samples (the batch) and structures them
    into a single batch dataclass suitable for model input, often for
    fine-tuning using `fit_from_preprocessed`.

    Handles samples containing nested lists (e.g., for ensemble members)
    and tensors. Pads tensors to consistent shapes using `pad_tensors`
    before stacking. Non-tensor items are grouped into lists.

    Args:
        batch: A list of ClassifierBatch or RegressorBatch dataclass instances.
        padding_val: Value used for padding tensors to allow stacking across
            the batch dimension.

    Returns:
        A collated ClassifierBatch or RegressorBatch with stacked/padded data.

    Note:
        Currently only implemented and tested for `batch_size = 1`,
        as enforced by an internal assertion.
    """
    batch_sz = len(batch)
    assert batch_sz == 1, "Only Implemented and tested for batch size of 1"

    first_item = batch[0]
    num_estimators = len(first_item.X_context)

    if isinstance(first_item, ClassifierBatch):
        return ClassifierBatch(
            X_context=_collate_list_field(
                batch, "X_context", num_estimators, padding_val
            ),
            X_query=_collate_list_field(batch, "X_query", num_estimators, padding_val),
            y_context=_collate_list_field(
                batch, "y_context", num_estimators, padding_val
            ),
            y_query=_collate_tensor_field(batch, "y_query", padding_val),
            cat_indices=_collate_cat_indices(batch),
            configs=_collate_list_field(batch, "configs", num_estimators, padding_val),
        )

    # RegressorBatch - since batch_size=1, we extract the single item for bardist
    # first_item is already narrowed to RegressorBatch by the isinstance check above
    assert isinstance(first_item, RegressorBatch)
    return RegressorBatch(
        X_context=_collate_list_field(batch, "X_context", num_estimators, padding_val),
        X_query=_collate_list_field(batch, "X_query", num_estimators, padding_val),
        y_context=_collate_list_field(batch, "y_context", num_estimators, padding_val),
        y_query=_collate_tensor_field(batch, "y_query", padding_val),
        cat_indices=_collate_cat_indices(batch),
        configs=_collate_list_field(batch, "configs", num_estimators, padding_val),
        raw_space_bardist=first_item.raw_space_bardist,
        znorm_space_bardist=first_item.znorm_space_bardist,
        X_query_raw=_collate_tensor_field(batch, "X_query_raw", padding_val),
        y_query_raw=_collate_tensor_field(batch, "y_query_raw", padding_val),
    )


def shuffle_and_chunk_data(
    X: XType,
    y: YType,
    *,
    max_chunk_size: int,
    equal_split_size: bool,
    seed: int,
    min_chunk_size: int = 2_000,
    task: Literal["regression", "multiclass"] | None = None,
    shuffle: bool = True,
) -> tuple[list[XType], list[YType]]:
    """Shuffle X and y with the given seed, then split into chunks.

    Args:
        X: Features as a numpy array or pandas DataFrame.
        y: Targets as a numpy array or pandas Series.
        max_chunk_size: Maximum size for any chunk.
        equal_split_size: If True, produce equally sized chunks (all <= max_chunk_size);
            otherwise make chunks of size max_chunk_size, keeping a final remainder
            chunk only if it has at least 2 samples.
        seed: Random seed used to shuffle X and y before splitting.
        min_chunk_size: Minimum size for any chunk.
        task: If "multiclass", perform stratified splitting using StratifiedKFold so
            each chunk has roughly the same class proportions. If "regression" or
            None, use non-stratified splitting.
        shuffle: If True, shuffle the data before splitting.

    Returns:
        A tuple of two lists: (list of X chunks as XType, list of y chunks as YType).
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if len(X) == 0:
        return [], []

    if shuffle:
        _, rng = infer_random_state(seed)
        perm = rng.permutation(len(X))
        X = _take(X, perm)
        y = _take(y, perm)

    if task == "multiclass":
        return _chunk_data_stratified(
            X,
            y,
            max_chunk_size=max_chunk_size,
            equal_split_size=equal_split_size,
            min_chunk_size=min_chunk_size,
            seed=seed,
        )

    return _chunk_data_non_stratified(
        X,
        y,
        max_chunk_size=max_chunk_size,
        equal_split_size=equal_split_size,
        min_chunk_size=min_chunk_size,
    )


def get_preprocessed_dataset_chunks(  # noqa: PLR0913
    calling_instance: Any,
    X_raw: XType | list[XType],
    y_raw: YType | list[YType],
    split_fn: Callable,
    max_data_size: int | None,
    model_type: Literal["regressor", "classifier"],
    *,
    equal_split_size: bool,
    data_shuffle_seed: int,
    preprocessing_random_state: int | np.random.Generator,
    shuffle: bool = True,
    force_no_stratify: bool = False,
) -> DatasetCollectionWithPreprocessing:
    """Helper function to create a DatasetCollectionWithPreprocessing.

    Relies on methods from the calling_instance for specific initializations.
    Modularises Code for both Regressor and Classifier.

    Args:
        calling_instance: The instance of the TabPFNRegressor or TabPFNClassifier.
        X_raw: individual or list of input dataset features
        y_raw: individual or list of input dataset labels
        split_fn: A function to dissect a dataset into train and test partition.
        max_data_size: Maximum allowed number of samples within one dataset.
        If None, datasets are not splitted.
        model_type: The type of the model.
        equal_split_size: If True, splits data into equally sized chunks under
            max_data_size.
            If False, splits into chunks of size `max_data_size`, with
            the last chunk having the remainder samples but is dropped if its
            size is less than 2.
        data_shuffle_seed: int. Random seed to use for the data shuffling and splitting.
        preprocessing_random_state: Random state to use for the preprocessing.
        shuffle: If True, shuffle the data before splitting.
        force_no_stratify: If True, do not stratify the data even if the model
            type is classification. If None, use the model type to determine whether
            to stratify.
        use_fixed_preprocessing_seed: If True, use the same preprocessing seed for all
            dataset chunks. This also means the column permutations are the same
            for all dataset chunks.
    """
    # TODO: This will become very expensive for large datasets.
    # We need to change this strategy and do the preprocessing in a
    # streaming fashion.
    if not isinstance(X_raw, list):
        X_raw = [X_raw]
    if not isinstance(y_raw, list):
        y_raw = [y_raw]
    assert len(X_raw) == len(y_raw), "X and y lists must have the same length."

    calling_instance._initialize_model_variables()

    X_split, y_split = [], []
    for X_item, y_item in zip(X_raw, y_raw):
        if max_data_size is not None:
            Xparts, yparts = shuffle_and_chunk_data(
                X_item,
                y_item,
                max_chunk_size=max_data_size,
                equal_split_size=equal_split_size,
                seed=data_shuffle_seed,
                task=("multiclass" if model_type == "classifier" else "regression"),
                shuffle=shuffle,
            )
        else:
            Xparts, yparts = [X_item], [y_item]
        X_split.extend(Xparts)
        y_split.extend(yparts)

    dataset_config_collection: list[
        RegressorDatasetConfig | ClassifierDatasetConfig
    ] = []
    for X_item, y_item in zip(X_split, y_split):
        if model_type == "classifier":
            ensemble_configs, X_mod, y_mod = (
                calling_instance._initialize_dataset_preprocessing(
                    X=X_item,
                    y=y_item,
                    random_state=preprocessing_random_state,
                )
            )
            current_cat_ix = calling_instance.inferred_feature_schema_.indices_for(
                FeatureModality.CATEGORICAL
            )
            dataset_config = ClassifierDatasetConfig(
                config=ensemble_configs,
                X_raw=X_mod,
                y_raw=y_mod,
                cat_ix=current_cat_ix,
            )
        elif model_type == "regressor":
            ensemble_configs, X_mod, y_mod, bardist_ = (
                calling_instance._initialize_dataset_preprocessing(
                    X=X_item,
                    y=y_item,
                    random_state=preprocessing_random_state,
                )
            )
            current_cat_ix = calling_instance.inferred_feature_schema_.indices_for(
                FeatureModality.CATEGORICAL
            )
            dataset_config = RegressorDatasetConfig(
                config=ensemble_configs,
                X_raw=X_mod,
                y_raw=y_mod,
                cat_ix=current_cat_ix,
                znorm_space_bardist_=bardist_,
            )
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        dataset_config_collection.append(dataset_config)

    return DatasetCollectionWithPreprocessing(
        split_fn,
        random_state=preprocessing_random_state,
        dataset_config_collection=dataset_config_collection,
        stratify=False if force_no_stratify else (model_type == "classifier"),
    )
