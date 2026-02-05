"""Module for validation logic.

This includes input validation with sklearn's methods,
as well as input format validation.
"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from sklearn.base import is_classifier
from sklearn.utils.multiclass import check_classification_targets

from tabpfn.errors import TabPFNValidationError
from tabpfn.misc._sklearn_compat import check_array, validate_data
from tabpfn.settings import settings

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import XType, YType


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
) -> tuple[np.ndarray, np.ndarray, npt.NDArray[Any] | None, int, str | None]:
    """Validate and convert inputs to standardized format.

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

    Returns:
        A tuple of five elements:
        - the validated input data X as np.ndarray,
        - target data y as np.ndarray,
        - feature names as npt.NDArray[Any] | None,
        - number of features as int
        - target name if the input was a Series, otherwise None
    """
    # Preserve the name of the target data, if it exists.
    original_y_name: str | None = str(y.name) if isinstance(y, pd.Series) else None

    # Rely on sklearn's validation to return feature names to be consistent
    # with sklearn interfaces.
    X, y, feature_names_in, n_features_in = ensure_compatible_fit_inputs_sklearn(
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
        devices=devices,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )
    return X, y, feature_names_in, n_features_in, original_y_name


def ensure_compatible_predict_input_sklearn(
    X: XType,
    estimator: TabPFNRegressor | TabPFNClassifier,
) -> np.ndarray:
    """Validate and convert the input data for prediction.

    Note that this also changes the type of X to np.ndarray.
    """
    try:
        result = validate_data(
            estimator,
            X=X,
            # NOTE: Important that reset is False, i.e. doesn't reset estimator
            reset=False,
            # Parameters to `check_X_y()`
            accept_sparse=False,
            dtype=None,
            ensure_all_finite="allow-nan",
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
        allow_cpu_override=ignore_pretraining_limits,
    )


def ensure_compatible_fit_inputs_sklearn(
    X: XType,
    y: YType,
    *,
    estimator: TabPFNRegressor | TabPFNClassifier,
    ensure_y_numeric: bool = False,
) -> tuple[np.ndarray, np.ndarray, npt.NDArray[Any] | None, int]:
    """Validate the input data for fitting with standard input.

    Note that this also changes the type of X and y to np.ndarray.

    Args:
        X: The input data.
        y: The target data.
        estimator: The estimator to validate the data for.
        ensure_y_numeric: Whether to ensure the target data is numeric.

    Returns:
        A tuple of the validated input data X, target data y, feature names,
        and number of features.
    """
    try:
        X, y = validate_data(
            estimator,
            X=X,
            y=y,
            # Parameters to `check_X_y()`
            accept_sparse=False,
            dtype=None,  # This is handled later in `fit()`
            ensure_all_finite="allow-nan",
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
        raise TabPFNValidationError(str(e)) from e

    # NOTE: Theoretically we don't need to return the feature names and number,
    # but it makes it clearer in the calling code that these variables now exist
    # and can be set on the estimator.
    return X, y, getattr(estimator, "feature_names_in_", None), estimator.n_features_in_


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
    *,
    allow_cpu_override: bool = False,
) -> None:
    """Check if using CPU with large datasets and warn or error appropriately.

    Args:
        devices: The torch devices being used
        num_samples: The number of samples in the input data
        allow_cpu_override: If True, allow CPU usage with large datasets.
    """
    allow_cpu_override = allow_cpu_override or settings.tabpfn.allow_cpu_large_dataset

    if allow_cpu_override:
        return

    if any(device.type == "cpu" for device in devices):
        if num_samples > 1000:
            raise RuntimeError(
                "Running on CPU with more than 1000 samples is not allowed "
                "by default due to slow performance.\n"
                "To override this behavior, set the environment variable "
                "TABPFN_ALLOW_CPU_LARGE_DATASET=1 or "
                "set ignore_pretraining_limits=True.\n"
                "Alternatively, consider using a GPU or the tabpfn-client API: "
                "https://github.com/PriorLabs/tabpfn-client"
            )
        if num_samples > 200:
            warnings.warn(
                "Running on CPU with more than 200 samples may be slow.\n"
                "Consider using a GPU or the tabpfn-client API: "
                "https://github.com/PriorLabs/tabpfn-client",
                stacklevel=2,
            )
