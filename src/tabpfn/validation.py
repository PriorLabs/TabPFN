#  Copyright (c) Prior Labs GmbH 2026.

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
from sklearn.utils.validation import _get_feature_names, _num_features

from tabpfn.errors import TabPFNValidationError
from tabpfn.misc._sklearn_compat import _check_feature_names, check_array, check_X_y
from tabpfn.preprocessing.clean import coerce_nullable_dtypes_to_numpy
from tabpfn.settings import settings

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch
    from sklearn.base import BaseEstimator

    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import XType, YType


def original_target_name(y: YType) -> str | None:
    """The name of `y`, if it was passed as a `pandas.Series`, else `None`."""
    return str(y.name) if isinstance(y, pd.Series) else None


def extract_input_shape(X: XType) -> tuple[npt.NDArray[Any] | None, int | None]:
    """The `feature_names_in_`/`n_features_in_` that a raw fit input implies.

    Read from the raw input, before date expansion (`DateTimeExpander`,
    date_encoding.py) or value validation run: those two attributes must
    describe what the caller actually passed to `fit`, not TabPFN's internal
    (possibly wider, post-date-expansion) representation. Only the input's
    shape and column labels are inspected, never its values, so this is safe
    to call on an `X` that still holds a genuine `datetime64` column, unlike
    the rest of validation.

    Sets nothing: the caller assigns the two values it gets back, so what an
    estimator records about its input is written where it can be read.

    Returns:
        The column labels (`None` for an input that carries none, e.g. an
        array), and the column count -- `None` for an `X` whose count can't
        be determined at all (e.g. a 1D array), which value validation
        downstream (`check_array`/`check_X_y`) then rejects with its clearer,
        sklearn-standard message (e.g. "Reshape your data ...").
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

    The counterpart to `extract_input_shape`, and equally read-only: it
    compares and raises, but writes nothing back onto `estimator`. Call on the
    raw input, before date expansion or value validation, for the same reason.

    Names are checked before the column count, via sklearn's own helper, so a
    frame that is both renamed and narrowed reports the name mismatch --
    what sklearn's estimator checks expect.

    Raises:
        TabPFNValidationError: If `X`'s column labels or count disagree with
            what fit recorded.
    """
    try:
        _check_feature_names(estimator, X, reset=False)
    except (ValueError, TypeError) as e:
        raise TabPFNValidationError(str(e)) from e

    try:
        n_features = _num_features(X)
    except TypeError:
        return

    if hasattr(estimator, "n_features_in_") and n_features != estimator.n_features_in_:
        raise TabPFNValidationError(
            f"X has {n_features} features, but {estimator.__class__.__name__} "
            f"is expecting {estimator.n_features_in_} features as input."
        )


def ensure_compatible_predict_input_sklearn(
    X: XType,
    estimator: TabPFNRegressor | TabPFNClassifier,
) -> np.ndarray:
    """Validate the values of an already-shape-checked, already-date-expanded
    predict input, converting it to a plain array.

    `check_input_shape_matches` must already have run (on the raw,
    pre-expansion input) to check `n_features_in_`/`feature_names_in_`
    consistency -- this only checks values (dtype coercion, NaN/Inf), via
    `check_array` directly rather than `validate_data`, so it never touches
    those two attributes from the shape of the (possibly wider) `X` handed to
    it here.
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
    """Validate the values of already-shape-read, already-date-expanded
    fit inputs, converting them to plain arrays.

    `extract_input_shape` must already have run (on the raw, pre-expansion
    `X`) to give the caller `feature_names_in_`/`n_features_in_` -- this only
    checks values (dtype coercion, NaN/Inf, minimum samples, classification
    targets), via `check_X_y` directly rather than `validate_data`, so it
    never touches those two attributes from the shape of the (possibly
    wider) `X` handed to it here.

    Args:
        X: The input data, already expanded (`DateTimeExpander`).
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
