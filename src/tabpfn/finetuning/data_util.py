"""Utilities for data preparation used in fine-tuning wrappers.

Note that the get_preprocessed_datasets_helper() method below is a copy
of the public package. Copied here for easier modification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tabpfn.base import (
    ClassifierDatasetConfig,
    RegressorDatasetConfig,
)
from tabpfn.preprocessing import DatasetCollectionWithPreprocessing
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    from tabpfn.constants import XType, YType


def _take(obj: Any, idx: np.ndarray) -> Any:
    """Index obj by idx using .iloc when available, otherwise []."""
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


def shuffle_and_chunk_data(
    X: XType,
    y: YType,
    *,
    max_chunk_size: int,
    equal_split_size: bool,
    seed: int,
    min_chunk_size: int = 2_000,
    task: Literal["regression", "multiclass"] | None = None,
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

    Returns:
        A tuple of two lists: (list of X chunks as XType, list of y chunks as YType).
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if len(X) == 0:
        return [], []

    _, rng = infer_random_state(seed)
    perm = rng.permutation(len(X))

    X_shuffled = X.iloc[perm] if isinstance(X, pd.DataFrame) else X[perm]
    y_shuffled = y.iloc[perm] if isinstance(y, pd.Series) else y[perm]

    if task == "multiclass":
        return _chunk_data_stratified(
            X_shuffled,
            y_shuffled,
            max_chunk_size=max_chunk_size,
            equal_split_size=equal_split_size,
            min_chunk_size=min_chunk_size,
            seed=seed,
        )

    return _chunk_data_non_stratified(
        X_shuffled,
        y_shuffled,
        max_chunk_size=max_chunk_size,
        equal_split_size=equal_split_size,
        min_chunk_size=min_chunk_size,
    )


# Copied from tabpfn.base.get_preprocessed_datasets_helper
# Most likely we can remove the function in the base class
# as well as the member functions of classifier and regressor
def get_preprocessed_datasets_helper(
    calling_instance: Any,
    X_raw: XType | list[XType],
    y_raw: YType | list[YType],
    split_fn: Callable,
    max_data_size: int | None,
    model_type: Literal["regressor", "classifier"],
    *,
    equal_split_size: bool,
    seed: int,
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
        seed: int. Random seed to use for the data shuffling and splitting.
    """
    # TODO: This will become very expensive for large datasets.
    # We need to change this strategy and do the preprocessing in a
    # streaming fashion.
    if not isinstance(X_raw, list):
        X_raw = [X_raw]
    if not isinstance(y_raw, list):
        y_raw = [y_raw]
    assert len(X_raw) == len(y_raw), "X and y lists must have the same length."

    if not hasattr(calling_instance, "models_") or calling_instance.models_ is None:
        _, rng = calling_instance._initialize_model_variables()
    else:
        _, rng = infer_random_state(calling_instance.random_state)

    X_split, y_split = [], []
    for X_item, y_item in zip(X_raw, y_raw):
        if max_data_size is not None:
            Xparts, yparts = shuffle_and_chunk_data(
                X_item,
                y_item,
                max_chunk_size=max_data_size,
                equal_split_size=equal_split_size,
                seed=seed,
                task=("multiclass" if model_type == "classifier" else "regression"),
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
                calling_instance._initialize_dataset_preprocessing(X_item, y_item, rng)
            )
            current_cat_ix = calling_instance.inferred_categorical_indices_

            dataset_config = ClassifierDatasetConfig(
                config=ensemble_configs,
                X_raw=X_mod,
                y_raw=y_mod,
                cat_ix=current_cat_ix,
            )
        elif model_type == "regressor":
            ensemble_configs, X_mod, y_mod, bardist_ = (
                calling_instance._initialize_dataset_preprocessing(X_item, y_item, rng)
            )
            current_cat_ix = calling_instance.inferred_categorical_indices_
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
        rng=rng,
        dataset_config_collection=dataset_config_collection,
        stratify=(model_type == "classifier"),
    )
