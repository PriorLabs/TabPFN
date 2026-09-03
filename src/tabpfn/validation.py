#  Copyright (c) Prior Labs GmbH 2026.

"""Module for validation logic.

This includes input validation with sklearn's methods,
as well as input format validation.
"""

from __future__ import annotations

import numbers
import typing
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd
import torch
from sklearn.base import is_classifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _get_feature_names, _num_features

from tabpfn.errors import TabPFNValidationError
from tabpfn.misc._sklearn_compat import (
    _check_feature_names,
    check_array,
    check_X_y,
)
from tabpfn.preprocessing.clean import coerce_nullable_dtypes_to_numpy
from tabpfn.settings import settings

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch
    from sklearn.base import BaseEstimator

    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import XType, YType

    T = TypeVar("T")


def extract_input_shape(X: XType) -> tuple[npt.NDArray[Any] | None, int | None]:
    """The `feature_names_in_` and `n_features_in_` a raw fit input implies.

    Read off the raw input, before date conversion (`DateTransformer`) or value
    validation run: both attributes have to describe what the caller passed, not
    TabPFN's internal, possibly wider, view of it. Only the shape and the column
    labels are read here, never the values, so an `X` still holding a genuine
    `datetime64` column is fine, unlike for the rest of validation.

    Returns:
        The column labels (`None` for an input that carries none, e.g. an array),
        and the column count (`None` when it cannot be determined, e.g. for a 1D
        array, which value validation then rejects with its clearer "Reshape your
        data" message).

    Raises:
        TabPFNValidationError: If the labels mix strings with other types, which
            sklearn rejects too.
    """
    try:
        feature_names = _get_feature_names(X)
    except (ValueError, TypeError) as e:
        raise TabPFNValidationError(str(e)) from e

    try:
        n_features = _num_features(X)
    except TypeError:
        n_features = None
    return feature_names, n_features


def check_input_shape_matches(X: XType, *, estimator: BaseEstimator) -> None:
    """Check a predict input against the `feature_names_in_`/`n_features_in_` of fit.

    The counterpart of `extract_input_shape`, and read-only likewise: it compares
    and raises, writing nothing back onto `estimator`. Call it on the raw input,
    before date conversion, for the same reason.

    The labels are checked before the count, so a frame that is both renamed and
    narrowed reports the name mismatch, as sklearn's own check would.

    Raises:
        TabPFNValidationError: If `X`'s column labels or count disagree with what
            fit recorded.
    """
    try:
        _check_feature_names(estimator, X, reset=False)
    except (ValueError, TypeError) as e:
        raise TabPFNValidationError(str(e)) from e

    try:
        n_features = _num_features(X)
    except TypeError:
        return
    expected = getattr(estimator, "n_features_in_", None)
    if expected is not None and n_features != expected:
        raise TabPFNValidationError(
            f"X has {n_features} features, but {estimator.__class__.__name__} "
            f"is expecting {expected} features as input."
        )


def validate_categorical_features_indices(indices: Sequence[int] | None) -> None:
    """Check that `categorical_features_indices` holds integer column positions.

    Every consumer works positionally, so a column label in the list would be
    ignored at best and crash the index arithmetic at worst.

    Raises:
        TabPFNValidationError: On an entry that is not an integer.
    """
    for entry in indices or ():
        if not isinstance(entry, numbers.Integral):
            raise TabPFNValidationError(
                "`categorical_features_indices` must hold integer column positions, "
                f"got {entry!r} ({type(entry).__name__}). Pass the position of each "
                "categorical column, not its label."
            )


def ensure_compatible_fit_inputs(
    X: XType,
    y: YType,
    *,
    estimator: TabPFNRegressor | TabPFNClassifier,
    max_num_samples: int,
    max_num_features: int,
    ignore_pretraining_limits: bool,
    ensure_y_numeric: bool = False,
    devices: tuple[torch.device, ...],
    max_cpu_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Validate the values of already-shape-captured, already-date-resolved inputs.

    `feature_names_in_`/`n_features_in_` were read off the raw `X` by
    `extract_input_shape`: nothing here touches them, so they are never taken from
    the shape of the (possibly wider) `X` handed over here.

    Args:
        X: The input data.
        y: The target data.
        estimator: The estimator to validate the data for.
        max_num_samples: The maximum number of samples to allow.
        max_num_features: The maximum number of features to allow.
        ignore_pretraining_limits: Whether to ignore the pretraining limits.
        ensure_y_numeric: Whether to ensure the target data is numeric, e.g. for
            regression tasks.
        devices: The devices to use for the input data.
        max_cpu_samples: Sample count above which CPU inference raises by default.

    Returns:
        A tuple of three elements:
        - the validated input data X as np.ndarray,
        - target data y as np.ndarray,
        - target name if the input was a Series, otherwise None
    """
    # Preserve the name of the target data, if it exists.
    original_y_name: str | None = str(y.name) if isinstance(y, pd.Series) else None

    X, y = ensure_compatible_fit_inputs_sklearn(
        X,
        y,
        estimator=estimator,
        ensure_y_numeric=ensure_y_numeric,
    )
    validate_dataset_size(
        X=X,
        y=y,
        max_num_samples=max_num_samples,
        max_num_features=max_num_features,
        max_cpu_samples=max_cpu_samples,
        devices=devices,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )
    return X, y, original_y_name


def ensure_compatible_predict_input_sklearn(
    X: XType,
    estimator: TabPFNRegressor | TabPFNClassifier,
) -> np.ndarray:
    """Validate the values of an already-shape-checked, already-date-resolved input.

    `check_input_shape_matches` must already have run, on the raw input, to check
    it against `feature_names_in_`/`n_features_in_`; this checks values only, via
    `check_array` rather than `validate_data`, so it never re-reads those two
    attributes off the (possibly wider) `X` handed over here.

    Note that this also changes the type of X to np.ndarray.
    """
    if isinstance(X, pd.DataFrame):
        X = coerce_nullable_dtypes_to_numpy(X)
    try:
        result = check_array(
            X,
            accept_sparse=False,
            dtype=None,
            ensure_all_finite=False
            if estimator.get_inference_config().PASSTHROUGH_INF
            else "allow-nan",
            estimator=estimator,
        )
    except (ValueError, TypeError) as e:
        raise TabPFNValidationError(str(e)) from e
    return typing.cast("np.ndarray", result)


def validate_dataset_size(
    X: pd.DataFrame | np.ndarray | torch.Tensor,
    y: pd.Series | np.ndarray | torch.Tensor,
    *,
    max_num_samples: int,
    max_num_features: int,
    devices: tuple[torch.device, ...],
    ignore_pretraining_limits: bool = False,
    max_cpu_samples: int = 1000,
) -> None:
    """Validate the dataset size."""
    if len(X) != len(y):
        raise ValueError(
            f"Number of samples in X ({len(X)}) and y ({len(y)}) do not match.",
        )
    if len(X.shape) != 2:
        raise ValueError(
            f"The input data X is not a 2D array. Got shape: {X.shape}",
        )
    num_samples, num_features = X.shape
    _validate_num_samples_and_features(
        num_features=num_features,
        num_samples=num_samples,
        max_num_samples=max_num_samples,
        max_num_features=max_num_features,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )
    _validate_num_samples_for_cpu(
        devices=devices,
        num_samples=num_samples,
        max_cpu_samples=max_cpu_samples,
        allow_cpu_override=ignore_pretraining_limits,
    )


def ensure_compatible_fit_inputs_sklearn(
    X: XType,
    y: YType,
    *,
    estimator: TabPFNRegressor | TabPFNClassifier,
    ensure_y_numeric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the values of already-shape-captured fit inputs.

    Note that this also changes the type of X and y to np.ndarray.

    Args:
        X: The input data, already date-resolved (`DateTransformer`).
        y: The target data.
        estimator: The estimator to validate the data for.
        ensure_y_numeric: Whether to ensure the target data is numeric.

    Returns:
        A tuple of the validated input data X and target data y.
    """
    if isinstance(X, pd.DataFrame):
        X = coerce_nullable_dtypes_to_numpy(X)
    try:
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=None,  # This is handled later in `fit()`
            ensure_all_finite=False
            if estimator.get_inference_config().PASSTHROUGH_INF
            else "allow-nan",
            ensure_min_samples=2,
            ensure_min_features=1,
            y_numeric=ensure_y_numeric,
            estimator=estimator,
        )

        if is_classifier(estimator):
            check_classification_targets(y)
            # Annoyingly, the `ensure_all_finite` above only applies to `X` and
            # there is no way to specify this for `y`. The validation check above
            # will also only check for NaNs in `y` if `multi_output=True` which is
            # something we don't want. Hence, we run another check on `y` here.
            # However, we also have to consider that if the dtype is a string type,
            # then we still want to run finite checks without forcing a numeric dtype.
            y = check_array(
                y,
                accept_sparse=False,
                ensure_all_finite=True,
                dtype=None,  # type: ignore
                ensure_2d=False,
            )
    except (ValueError, TypeError) as e:
        e_str = str(e)
        if "X contains infinity" in e_str:
            e_str += (
                "\nHint: TabPFN uses infinite values as missing completely at random "  # noqa: S608
                "(MCAR) markers. If this matches your use case, try using "
                f"{estimator.__class__.__name__}"
                '(inference_config={"PASSTHROUGH_INF": True}).\n'
                "Otherwise, replace your infinite values with NaN to indicate "
                "missingness."
            )
        raise TabPFNValidationError(e_str) from e

    return X, y


def validate_num_classes(
    num_classes: int,
    max_num_classes: int,
) -> None:
    """Validate the number of classes.

    Raises a TabPFNValidationError if the number of classes exceeds the maximum
    number of classes officially supported by TabPFN.
    """
    if num_classes > max_num_classes:
        raise TabPFNValidationError(
            f"Number of classes `{num_classes}` exceeds the maximum number of "
            f"classes `{max_num_classes}` officially supported by TabPFN.",
        )


def _validate_num_samples_and_features(
    num_features: int,
    num_samples: int,
    max_num_samples: int,
    max_num_features: int,
    *,
    ignore_pretraining_limits: bool = False,
) -> None:
    """Validate the dataset size.

    If `ignore_pretraining_limits` is True, the validation is skipped.

    Raises a TabPFNValidationError if the number of features or samples exceeds
    the maximum number of features or samples officially supported by TabPFN.
    """
    if ignore_pretraining_limits:
        return

    if num_samples > max_num_samples:
        raise TabPFNValidationError(
            f"Number of samples `{num_samples:,}` in the input data is greater than "
            f"the maximum number of samples `{max_num_samples:,}` officially supported"
            f" by TabPFN. Set `ignore_pretraining_limits=True` to override this "
            f"error!",
        )
    if num_features > max_num_features:
        raise TabPFNValidationError(
            f"Number of features `{num_features}` reaching the model is greater than "
            f"the maximum number of features `{max_num_features}` officially "
            "supported by the TabPFN model. Set `ignore_pretraining_limits=True` "
            "to override this error!",
        )


def _validate_num_samples_for_cpu(
    devices: Sequence[torch.device],
    num_samples: int,
    max_cpu_samples: int,
    *,
    allow_cpu_override: bool = False,
) -> None:
    """Check if using CPU with large datasets and warn or error appropriately.

    Args:
        devices: The torch devices being used
        num_samples: The number of samples in the input data
        max_cpu_samples: Sample count above which CPU inference raises by default.
        allow_cpu_override: If True, allow CPU usage with large datasets.
    """
    allow_cpu_override = allow_cpu_override or settings.tabpfn.allow_cpu_large_dataset

    if allow_cpu_override:
        return

    if any(device.type == "cpu" for device in devices):
        if num_samples > max_cpu_samples:
            raise RuntimeError(
                f"Running on CPU with more than {max_cpu_samples} samples is not "
                "allowed by default due to slow performance.\n"
                "To override this behavior, set the environment variable "
                "TABPFN_ALLOW_CPU_LARGE_DATASET=1 or "
                "set ignore_pretraining_limits=True.\n"
                "Alternatively, consider using a GPU or the tabpfn-client API: "
                "https://github.com/PriorLabs/tabpfn-client"
            )
        # Warn at a fifth of the hard limit, matching the pre-existing 200:1000 ratio.
        warn_threshold = max_cpu_samples // 5
        if num_samples > warn_threshold:
            warnings.warn(
                f"Running on CPU with more than {warn_threshold} samples may be slow.\n"
                "Consider using a GPU or the tabpfn-client API: "
                "https://github.com/PriorLabs/tabpfn-client",
                stacklevel=2,
            )
