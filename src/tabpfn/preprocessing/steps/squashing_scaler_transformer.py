"""Implementation of the SquashingScaler, adapted from skrub.

Adapted from skrub: https://github.com/skrub-data/skrub
  reference: https://skrub-data.org/stable/reference/generated/skrub.SquashingScaler.html

Copyright (c) 2018-2023, The dirty_cat developers, 2023-2026 the skrub developers.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

This preprocessing is used e.g. in RealMLP, see https://arxiv.org/abs/2407.04491
"""

from __future__ import annotations

import numbers
from typing import Any
from typing_extensions import override

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

try:
    from sklearn.utils.validation import validate_data as sklearn_validate_data
except ImportError:
    sklearn_validate_data = None


def _validate_data(estimator: BaseEstimator, **kwargs: Any) -> Any:
    """Select the appropriate validate_data API and runs it."""
    if sklearn_validate_data is not None:
        return sklearn_validate_data(estimator, **kwargs)

    if "ensure_all_finite" in kwargs:
        force_all_finite = kwargs.pop("ensure_all_finite")
    else:
        force_all_finite = True
    return estimator._validate_data(**kwargs, force_all_finite=force_all_finite)


def _mask_inf(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace infinite values with NaN and return their sign."""
    if (mask_inf := np.isinf(X)).any():
        sign = np.sign(X)
        X = np.where(mask_inf, np.nan, X)
        # 0 when X is finite, 1 when X is +inf, -1 when X is -inf
        mask_inf = mask_inf.astype(X.dtype) * sign

    return X, mask_inf


def _set_zeros(X: np.ndarray, zero_cols: np.ndarray) -> np.ndarray:
    """Set the finite values of the specified columns to zero."""
    mask = np.isfinite(X)
    mask[:, ~zero_cols] = False
    X[mask] = 0.0
    return X


def _soft_clip(
    X: np.ndarray,
    max_absolute_value: float,
    mask_inf: np.ndarray,
) -> np.ndarray:
    """Apply a soft clipping to the data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be clipped.
    max_absolute_value : float, default=3.0
        Maximum absolute value that the transformed data can take.
    mask_inf : array-like, shape (n_samples, n_features)
        A mask indicating the positions of infinite values in the input data and their
        signs.

    Returns:
    -------
    X_clipped : array-like, shape (n_samples, n_features)
        The clipped version of the input.
    """
    X = X / np.sqrt(1 + (X / max_absolute_value) ** 2)
    X = np.where(mask_inf == 1, max_absolute_value, X)
    return np.where(mask_inf == -1, -max_absolute_value, X)


class SquashingScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    r"""Perform robust centering and scaling followed by soft clipping.

    When features have large outliers, smooth clipping prevents the outliers from
    affecting the result too strongly, while robust scaling prevents the outliers from
    affecting the inlier scaling. Infinite values are mapped to the corresponding
    boundaries of the interval. NaN values are preserved.

    Parameters
    ----------
    max_absolute_value : float, default=3.0
        Maximum absolute value that the transformed data can take.

    quantile_range : tuple of float, default=(25.0, 75.0)
        The quantiles (on a 0-100 scale) used to compute the scaling factor. The first
        value is the lower quantile and the second value is the upper quantile. The
        default values are the 25th and 75th percentiles, respectively. The quantiles are used to compute the
        scaling factor for the robust scaling step. The quantiles are computed from the
        finite values in the input column. If the two quantiles are equal, the scaling
        factor is computed from the 0th and 100th percentiles (i.e., the minimum and
        maximum values of the finite values in the input column).

    Notes:
    -----
    This transformer is applied to each column independently. It uses two stages:

    1. The first stage centers the median of the data to zero and multiplies the data by a
       scaling factor determined from quantiles of the distribution, using
       scikit-learn's :class:`~sklearn.preprocessing.RobustScaler`. It also handles
       edge-cases in which the two quantiles are equal by following-up with a
       :class:`~sklearn.preprocessing.MinMaxScaler`.
    2. The second stage applies a soft clipping to the transformed data to limit the
       data to the interval ``[-max_absolute_value, max_absolute_value]`` in an
       injective way.

    Infinite values will be mapped to the corresponding boundaries of the interval. NaN
    values will be preserved.

    The formula for the transform is:

    .. math::

        \begin{align*}
            a &:= \begin{cases}
                1/(q_{\beta} - q_{\alpha}) &\text{if} \quad q_{\beta} \neq q_{\alpha} \\
                2/(q_1 - q_0) &\text{if}\quad q_{\beta} = q_{\alpha} \text{ and } q_1
                \neq q_0 \\ 0 & \text{otherwise}
            \end{cases} \\ z &:= a.(x - q_{1/2}), \\ x_{\text{out}} &:= \frac{z}{\sqrt{1
            + (z/B)^2}},
        \end{align*}

    where:

    - :math:`x` is a value in the input column.
    - :math:`q_{\gamma}` is the :math:`\gamma`-quantile of the finite values in X,
    - :math:`B` is max_abs_value
    - :math:`\alpha` is the lower quantile
    - :math:`\beta` is the upper quantile.

    References:
    ----------
    This method has been introduced as the robust scaling and smooth clipping transform
    in `Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data
    (Holzmüller et al., 2024) <https://arxiv.org/abs/2407.04491>`_.

    Examples:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from tabpfn.preprocessing import SquashingScaler

    In the general case, this scale uses a RobustScaler:

    >>> X = pd.DataFrame(dict(col=[np.inf, -np.inf, 3, -1, np.nan, 2]))
    >>> SquashingScaler(max_absolute_value=3).fit_transform(X)
    array([[ 3.        ],
           [-3.        ],
           [ 0.49319696],
           [-1.34164079],
           [        nan],
           [ 0.        ]])

    When quantile ranges are equal, this scaler uses a customized MinMaxScaler:

    >>> X = pd.DataFrame(dict(col=[0, 1, 1, 1, 2, np.nan]))
    >>> SquashingScaler().fit_transform(X)
    array([[-0.9486833],
           [ 0.       ],
           [ 0.       ],
           [ 0.       ],
           [ 0.9486833],
           [       nan]])

    Finally, when the min and max are equal, this scaler fills the column with zeros:

    >>> X = pd.DataFrame(dict(col=[1, 1, 1, np.nan]))
    >>> SquashingScaler().fit_transform(X)
    array([[ 0.],
           [ 0.],
           [ 0.],
           [nan]])
    """  # noqa: E501

    robust_center_: np.ndarray
    robust_scale_: np.ndarray
    minmax_center_: np.ndarray
    minmax_scale_: np.ndarray
    robust_cols_: np.ndarray
    minmax_cols_: np.ndarray
    zero_cols_: np.ndarray

    def __init__(
        self,
        max_absolute_value: float = 3.0,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        super().__init__()
        self.max_absolute_value = max_absolute_value
        self.quantile_range = quantile_range

    def fit(self, X: np.ndarray, y: None | np.ndarray = None) -> SquashingScaler:
        """Fit the transformer to a column.

        Parameters
        ----------
        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
            The data to transform.
        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns:
        -------
        self : SquashingScaler
            The fitted transformer.
        """
        del y

        if not (
            isinstance(self.max_absolute_value, numbers.Number)
            and np.isfinite(self.max_absolute_value)
            and self.max_absolute_value > 0
        ):
            raise ValueError(
                f"Got max_absolute_value={self.max_absolute_value!r}, but expected a "
                "positive finite number."
            )

        X = _validate_data(
            self,
            X=X,  # type: ignore
            reset=True,
            dtype=FLOAT_DTYPES,
            accept_sparse=False,
            ensure_2d=True,
            ensure_all_finite=False,
        )
        # Convert np.inf to np.nan so the percentile/min/max routines below
        # operate only on the finite distribution of each column.
        X, _ = _mask_inf(X)

        # All column statistics are computed in a single pass: nanmin, nanmax, and
        # the (lower, median, upper) quantiles. Reusing these avoids the duplicate
        # nanpercentile call that previously came from constructing a sklearn
        # RobustScaler that re-derived the same quartiles internally.
        col_min = np.nanmin(X, axis=0)
        col_max = np.nanmax(X, axis=0)
        lower_q, upper_q = self.quantile_range
        q_lower, q_median, q_upper = np.nanpercentile(
            X, [lower_q, 50.0, upper_q], axis=0
        )

        # For each column, exactly one of three scaling strategies applies:
        # 1) zero_cols: max == min, so the column is constant; finite values
        #    are mapped to 0 in transform.
        # 2) minmax_cols: q_lower == q_upper but max != min; we fall back to a
        #    median-centered scaling using the full range.
        # 3) robust_cols: the general case, scaled by (q_upper - q_lower).
        zero_cols = col_max == col_min
        minmax_cols = (q_lower == q_upper) & ~zero_cols
        robust_cols = ~(minmax_cols | zero_cols)

        eps = np.finfo(X.dtype).tiny
        self.robust_center_ = q_median[robust_cols]
        self.robust_scale_ = q_upper[robust_cols] - q_lower[robust_cols]
        self.minmax_center_ = q_median[minmax_cols]
        self.minmax_scale_ = 2.0 / (col_max[minmax_cols] - col_min[minmax_cols] + eps)

        self.robust_cols_ = robust_cols
        self.minmax_cols_ = minmax_cols
        self.zero_cols_ = zero_cols

        return self

    @override
    def fit_transform(
        self,
        X: np.ndarray,
        y: None | np.ndarray = None,
        **fit_params: Any,
    ) -> np.ndarray:
        """Fit the transformer and transform a column.

        Parameters
        ----------
        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
            The data to transform.
        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns:
        -------
        X_out: numpy array, shape (n_samples, n_features)
            The transformed version of the input.
        """
        del y
        del fit_params

        self.fit(X)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform a column.

        Parameters
        ----------
        X : numpy array, Pandas or Polars DataFrame of shape (n_samples, n_features)
            The data to transform.

        Returns:
        -------
        X_out: numpy array of shape (n_samples, n_features)
            The transformed version of the input.
        """
        check_is_fitted(
            self,
            [
                "robust_center_",
                "robust_scale_",
                "minmax_center_",
                "minmax_scale_",
                "zero_cols_",
                "robust_cols_",
                "minmax_cols_",
            ],
        )

        X = _validate_data(
            self,
            X=X,  # type: ignore
            reset=False,
            dtype=FLOAT_DTYPES,
            accept_sparse=False,
            ensure_2d=True,
            ensure_all_finite=False,
        )
        # To replace the original ±np.inf with ±max_absolute_value in the final output.
        # mask_inf is a 2D array containing the sign of the np.inf in the input.
        X, mask_inf = _mask_inf(X)

        # copy the input since we change the values in place
        X_tr = X.copy()
        if self.robust_cols_.any():
            X_tr[:, self.robust_cols_] = (
                X[:, self.robust_cols_] - self.robust_center_
            ) / self.robust_scale_
        if self.minmax_cols_.any():
            X_tr[:, self.minmax_cols_] = (
                X[:, self.minmax_cols_] - self.minmax_center_
            ) * self.minmax_scale_
        if self.zero_cols_.any():
            # if the scale is 0, we set the values to 0
            X_tr = _set_zeros(X_tr, self.zero_cols_)

        return _soft_clip(X_tr, self.max_absolute_value, mask_inf)


__all__ = ["SquashingScaler"]
