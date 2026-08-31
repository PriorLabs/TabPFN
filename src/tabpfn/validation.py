#  Copyright (c) Prior Labs GmbH 2026.

"""Module for validation logic.

This includes input validation with sklearn's methods,
as well as input format validation.
"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
import torch
from sklearn.base import is_classifier
from sklearn.utils.multiclass import check_classification_targets

from tabpfn.errors import TabPFNValidationError
from tabpfn.misc._sklearn_compat import check_array, check_X_y, validate_data
from tabpfn.preprocessing.clean import coerce_nullable_dtypes_to_numpy
from tabpfn.settings import settings

if TYPE_CHECKING:
    import numpy as np
    import torch

    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import XType, YType


def original_target_name(y: YType) -> str | None:
    """The name of `y`, if it was passed as a `pandas.Series`, else `None`."""
    return str(y.name) if isinstance(y, pd.Series) else None


def capture_input_shape(
    X: XType,
    *,
    estimator: TabPFNRegressor | TabPFNClassifier,
    reset: bool,
) -> None:
    """Set (`reset=True`) or check (`reset=False`) `feature_names_in_`/`n_features_in_`.

    Call on the raw input, before date resolution (`resolve_date_columns`,
    date_encoding.py) or value validation run: those two attributes must
    describe what the caller actually passed to `fit`/`predict`, not
    TabPFN's internal (possibly wider, post-date-expansion) representation.
    `skip_check_array=True` means only the input's shape and column labels
    are inspected here, never its values, so this is safe to call on `X`
    that still holds a genuine `datetime64` column, unlike the rest of
    validation.
    """
    try:
        validate_data(estimator, X=X, reset=reset, skip_check_array=True)
    except (ValueError, TypeError) as e:
        raise TabPFNValidationError(str(e)) from e


def ensure_compatible_predict_input_sklearn(
    X: XType,
    estimator: TabPFNRegressor | TabPFNClassifier,
) -> np.ndarray:
    """Validate the values of an already-shape-checked, already-date-resolved
    predict input, converting it to a plain array.

    `capture_input_shape` must already have run (on the raw, pre-resolution
    input) to check `n_features_in_`/`feature_names_in_` consistency -- this
    only checks values (dtype coercion, NaN/Inf), via `check_array` directly
    rather than `validate_data`, so it never re-touches those two attributes
    from the shape of the (possibly wider) `X` handed to it here.
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
        raise TabPFNValidationError(
            f"Number of samples in X ({len(X)}) and y ({len(y)}) do not match.",
        )
    if len(X.shape) != 2:
        raise TabPFNValidationError(
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
    """Validate the values of already-shape-checked, already-date-resolved
    fit inputs, converting them to plain arrays.

    `capture_input_shape` must already have run (on the raw, pre-resolution
    `X`) to set `feature_names_in_`/`n_features_in_` -- this only checks
    values (dtype coercion, NaN/Inf, minimum samples, classification
    targets), via `check_X_y` directly rather than `validate_data`, so it
    never re-touches those two attributes from the shape of the (possibly
    wider) `X` handed to it here.

    Args:
        X: The input data, already resolved (`resolve_date_columns`).
        y: The target data.
        estimator: The estimator to validate the data for.
        ensure_y_numeric: Whether to ensure the target data is numeric.

    Returns:
        The validated input data X and target data y.
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
            f"Number of features `{num_features}` in the input data is greater than "
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
