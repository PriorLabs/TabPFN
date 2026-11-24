#  Copyright (c) Prior Labs GmbH 2026.

"""Safe Power Transformer."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar
from typing_extensions import override

import numpy as np
from scipy import optimize
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted


# this is taken from https://github.com/scipy/scipy/pull/18852
# which fix overflow issues
# we can directly import from scipy once we drop support for scipy < 1.12
def _yeojohnson(
    x: np.ndarray,
    lmbda: float | None = None,
) -> tuple[np.ndarray, float | None]:
    x = np.asarray(x)
    if x.size == 0:
        # changed from scipy from return x
        return (x, None) if lmbda is None else x  # type: ignore

    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError(
            "Yeo-Johnson transformation is not defined for complex numbers."
        )

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64, copy=False)

    if lmbda is not None:
        return _yeojohnson_transform(x, lmbda)  # type: ignore

    # if lmbda=None, find the lmbda that maximizes the log-likelihood function.
    lmax = _yeojohnson_normmax(x)
    y = _yeojohnson_transform(x, lmax)

    return y, lmax


def _yeojohnson_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Returns `x` transformed by the Yeo-Johnson power transform with given
    parameter `lmbda`.
    """
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    out = np.zeros_like(x, dtype=dtype)
    pos = x >= 0  # binary mask

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        out[pos] = np.log1p(x[pos])
    else:  # lmbda != 0
        # more stable version of: ((x + 1) ** lmbda - 1) / lmbda
        out[pos] = np.expm1(lmbda * np.log1p(x[pos])) / lmbda

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        out[~pos] = -np.expm1((2 - lmbda) * np.log1p(-x[~pos])) / (2 - lmbda)
    else:  # lmbda == 2
        out[~pos] = -np.log1p(-x[~pos])

    return out


def _yeojohnson_llf(lmb: float, data: np.ndarray) -> np.ndarray:
    r"""The yeojohnson log-likelihood function."""
    data = np.asarray(data)
    n_samples = data.shape[0]

    if n_samples == 0:
        return np.nan  # type: ignore

    trans = _yeojohnson_transform(data, lmb)
    trans_var = trans.var(axis=0)
    loglike = np.empty_like(trans_var)

    # Avoid RuntimeWarning raised by np.log when the variance is too low
    tiny_variance = trans_var < np.finfo(trans_var.dtype).tiny
    loglike[tiny_variance] = np.inf

    loglike[~tiny_variance] = -n_samples / 2 * np.log(trans_var[~tiny_variance])
    loglike[~tiny_variance] += (lmb - 1) * (np.sign(data) * np.log1p(np.abs(data))).sum(
        axis=0
    )
    return loglike


def _yeojohnson_normmax(
    x: np.ndarray,
    brack: float | None = None,
    log_likelihood: Callable[[float, np.ndarray], np.ndarray] = _yeojohnson_llf,
) -> float:
    """Compute optimal Yeo-Johnson transform parameter.
    Compute optimal Yeo-Johnson transform parameter for input data, using
    maximum likelihood estimation.

    """

    def _neg_llf(lmbda: float, data: np.ndarray) -> np.ndarray:
        llf = log_likelihood(lmbda, data)
        # reject likelihoods that are inf which are likely due to small
        # variance in the transformed space
        llf[np.isinf(llf)] = -np.inf
        return -llf

    with np.errstate(invalid="ignore"):
        if not np.all(np.isfinite(x)):
            raise ValueError("Yeo-Johnson input must be finite.")
        if np.all(x == 0):
            return 1.0
        if brack is not None:
            return optimize.brent(_neg_llf, brack=brack, args=(x,))  # type: ignore
        x = np.asarray(x)
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        # Allow values up to 20 times the maximum observed value to be safely
        # transformed without over- or underflow.
        log1p_max_x = np.log1p(20 * np.max(np.abs(x)))
        # Use half of floating point's exponent range to allow safe computation
        # of the variance of the transformed data.
        log_eps = np.log(np.finfo(dtype).eps)
        log_tiny_float = (np.log(np.finfo(dtype).tiny) - log_eps) / 2
        log_max_float = (np.log(np.finfo(dtype).max) + log_eps) / 2
        # Compute the bounds by approximating the inverse of the Yeo-Johnson
        # transform on the smallest and largest floating point exponents, given
        # the largest data we expect to observe. See [1] for further details.
        # [1] https://github.com/scipy/scipy/pull/18852#issuecomment-1630286174
        lb = log_tiny_float / log1p_max_x
        ub = log_max_float / log1p_max_x
        # Convert the bounds if all or some of the data is negative.
        if np.all(x < 0):
            lb, ub = 2 - ub, 2 - lb
        elif np.any(x < 0):
            lb, ub = max(2 - ub, lb), min(2 - lb, ub)
        # Match `optimize.brent`'s tolerance.
        tol_brent = 1.48e-08
        return optimize.fminbound(_neg_llf, lb, ub, args=(x,), xtol=tol_brent)  # type: ignore


# we created this inspired by the scipy change for transform
# https://github.com/scipy/scipy/pull/18852
# this is not in scipy even 1.12
def _yeojohnson_inverse_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Return inverse-transformed input x following Yeo-Johnson inverse
    transform with parameter lambda.
    """
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    x_inv = np.zeros_like(x, dtype=dtype)
    pos = x >= 0

    # Clip expm1 arguments to prevent overflow in the output dtype.
    # Leave a small margin below log(max): expm1's float64 round-trip on the
    # clipped value can land a few ULPs over dtype's max, which then overflows
    # when cast back into x_inv (dtype). Nudge the bound down to absorb that.
    log_max = np.log(np.finfo(dtype).max)
    max_arg = log_max - 4 * np.spacing(log_max)

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        x_inv[pos] = np.expm1(np.clip(x[pos], -max_arg, max_arg))
    else:  # lmbda != 0
        # more stable version of: (x * lmbda + 1) ** (1 / lmbda) - 1
        x_inv[pos] = np.expm1(
            np.clip(np.log1p(x[pos] * lmbda) / lmbda, -max_arg, max_arg)
        )

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        # more stable version of: 1 - (-(2 - lmbda) * x + 1) ** (1 / (2 - lmbda))
        x_inv[~pos] = -np.expm1(
            np.clip(np.log1p(-(2 - lmbda) * x[~pos]) / (2 - lmbda), -max_arg, max_arg)
        )
    else:  # lmbda == 2
        x_inv[~pos] = -np.expm1(np.clip(-x[~pos], -max_arg, max_arg))

    return x_inv


def _power_transform_from_log(
    log_y: np.ndarray | float, power: float
) -> np.ndarray | float:
    """Return ``(y^power - 1) / power`` given ``log(y)``, or ``log(y)`` for power 0.

    This is the Box-Cox transform of ``y``. Working from the log keeps it accurate
    for ``y`` close to 1.
    """
    if abs(power) < np.spacing(1.0):
        return log_y
    return np.expm1(power * log_y) / power


def _yeojohnson_centered_transform(
    x: np.ndarray, lmbda: float, log_mean: float
) -> np.ndarray:
    """Yeo-Johnson transform of ``x`` relative to the geometric mean, in float64.

    For x >= 0 the classic transform is::

        y = ((1 + x)^lambda - 1) / lambda

    In a column far from zero, such as a year, the power can be so small in every
    row that subtracting 1 rounds the differences between rows away. This function
    takes the power of a number close to 1 instead::

        ratio = (1 + x) / g
        y = (ratio^lambda - 1) / lambda

    Here g is the geometric mean of 1 + x over the fitted column. For a given
    lambda the result is the classic one shifted and scaled, so it is the same
    after standardization.

    Args:
        x: The values of one column.
        lmbda: The power, lambda.
        log_mean: log(g), stored by the fit as the mean of
            ``sign(x) * log(1 + |x|)``. With 0 the result is the classic transform.
    """
    x = np.asarray(x, dtype=np.float64)
    # Plain floats, so the result does not depend on the scalar types passed in.
    lmbda, log_mean = float(lmbda), float(log_mean)
    if log_mean < 0:
        # A mostly negative column is transformed as its mirror image. Yeo-Johnson
        # is symmetric: flipping the sign of x and replacing lambda by 2 - lambda
        # flips the sign of the result.
        return -_yeojohnson_centered_transform(-x, 2 - lmbda, -log_mean)

    nonnegative = x >= 0
    out = np.empty_like(x)
    out[nonnegative], out[~nonnegative] = _power_of_ratio(
        *_log_ratio(x, log_mean), lmbda, log_mean
    )
    return out


def _log_ratio(x: np.ndarray, log_mean: float) -> tuple[np.ndarray, np.ndarray]:
    """Step 1 of `_yeojohnson_centered_transform`, which does not depend on lambda.

    Returns the logs that step 2, `_power_of_ratio`, takes a power of: one array
    for the values x >= 0 and one for the other values. Expects ``log_mean >= 0``.
    """
    x_at_mean = np.expm1(log_mean)  # the x at which 1 + x equals g
    nonnegative = x >= 0

    # Values x >= 0:
    #
    # 1. ratio = (1 + x) / g, kept as its log.
    #    For a row near the mean, log(1 + x) and log(g) are almost equal, and
    #    subtracting them would lose the digits that tell rows apart. There the
    #    ratio is formed from x - x_at_mean instead, a subtraction that is exact
    #    when the two are within a factor of two (Sterbenz lemma). Far from the
    #    mean that route fails in turn, and the two logs are safe to subtract.
    x_pos = x[nonnegative]
    near_mean = (x_pos >= x_at_mean / 2) & (x_pos <= 2 * x_at_mean)
    distance = np.where(near_mean, (x_pos - x_at_mean) / (1 + x_at_mean), 0.0)
    log_ratio = np.where(near_mean, np.log1p(distance), np.log1p(x_pos) - log_mean)

    # Values x < 0 get the log that the classic formula takes a power of.
    return log_ratio, np.log1p(-x[~nonnegative])


def _power_of_ratio(
    log_ratio: np.ndarray, log_negative: np.ndarray, lmbda: float, log_mean: float
) -> tuple[np.ndarray, np.ndarray]:
    """Step 2 of `_yeojohnson_centered_transform`, from the two arrays of step 1."""
    # Values x >= 0:
    #
    # 2. y = (ratio^lambda - 1) / lambda.
    y_pos = _power_transform_from_log(log_ratio, lmbda)

    # Values x < 0. In a mostly positive column these lie across zero from the
    # rest, and the classic formula, -((1 - x)^(2 - lambda) - 1) / (2 - lambda),
    # is accurate for them. Compared with the classic result, step 1 scaled and
    # shifted the values above. The same scale and shift here make the two parts
    # join at x = 0.
    value_at_zero = _power_transform_from_log(-log_mean, lmbda)  # y at x = 0
    scale = np.exp(-lmbda * log_mean)
    classic = -_power_transform_from_log(log_negative, 2 - lmbda)
    return y_pos, value_at_zero + scale * classic


def _yeojohnson_centered_inverse_transform(
    z: np.ndarray, lmbda: float, log_mean: float
) -> np.ndarray:
    """Inverse of `_yeojohnson_centered_transform`: its steps undone in reverse.

    As `_yeojohnson_inverse_transform`, the result stays inside the range of the
    dtype of ``z``, so that extreme values neither overflow nor warn.
    """
    z = np.asarray(z)
    lmbda, log_mean = float(lmbda), float(log_mean)
    if log_mean < 0:
        # The mirror image, as in the transform.
        return -_yeojohnson_centered_inverse_transform(-z, 2 - lmbda, -log_mean)

    limit = np.finfo(z.dtype if np.issubdtype(z.dtype, np.floating) else float).max
    z = z.astype(np.float64)
    x_at_mean = np.expm1(log_mean)
    value_at_zero = _power_transform_from_log(-log_mean, lmbda)
    # The transform is increasing, so its value at x = 0 separates the two parts.
    nonnegative = z >= value_at_zero
    x = np.empty_like(z)

    # Values from x >= 0.
    #
    # Undo step 2: recover the log of the ratio from y.
    log_ratio = z[nonnegative]
    if abs(lmbda) >= np.spacing(1.0):
        log_ratio = np.log1p(lmbda * log_ratio) / lmbda
    # Undo step 1: recover x from the ratio. Written around x_at_mean, so that
    # rows near the mean keep their differences. The log of 1 + x is capped just
    # below the largest float, as in the classic inverse, so exp cannot overflow.
    log_max = np.log(np.finfo(np.float64).max)
    log_ratio = np.minimum(log_ratio, log_max - 4 * np.spacing(log_max) - log_mean)
    x[nonnegative] = x_at_mean + (1 + x_at_mean) * np.expm1(log_ratio)

    # Values from x < 0: undo the shift and the scale, then the classic formula.
    classic = (z[~nonnegative] - value_at_zero) * np.exp(lmbda * log_mean)
    x[~nonnegative] = _yeojohnson_inverse_transform(classic, lmbda)
    return np.clip(x, -limit, limit)


def _fit_yeojohnson_centered(x: np.ndarray) -> tuple[float, float]:
    """Fit one column: its maximum-likelihood lambda and the ``log_mean`` to store."""
    x = np.asarray(x[~np.isnan(x)], dtype=np.float64)  # missing values do not count
    if x.size == 0:
        return np.nan, 0.0
    # On a constant column the search would only fit rounding noise. Leave the
    # column unchanged: lambda = 1 with log_mean = 0 is the identity.
    if _is_constant_feature(x.var(), x.mean(), x.size):
        return 1.0, 0.0
    # The log of the geometric mean is the mean of the logs. Each log carries the
    # sign of its value, so that a mostly negative column gets a negative mean.
    log_mean = float(np.mean(np.sign(x) * np.log1p(np.abs(x))))

    # Step 1 of the transform does not depend on lambda, so the search computes it
    # once and repeats only step 2. A mostly negative column is searched as its
    # mirror image, as in the transform.
    mirrored = log_mean < 0
    x_mirrored = -x if mirrored else x
    logs = _log_ratio(x_mirrored, abs(log_mean))
    nonnegative = x_mirrored >= 0

    def log_likelihood(lmbda: float, data: np.ndarray) -> np.ndarray:
        # The Yeo-Johnson log-likelihood has two terms: one in the variance of the
        # transformed column and one in the sum of its logs (`_yeojohnson_llf`).
        # Dividing by the geometric mean makes the second term constant, so the
        # variance alone decides. A vanishing variance is reported as inf, as
        # there.
        y = np.empty_like(x)
        y[nonnegative], y[~nonnegative] = _power_of_ratio(
            *logs, 2 - lmbda if mirrored else lmbda, abs(log_mean)
        )
        var = y.var()
        if not var >= np.finfo(np.float64).tiny:
            return np.array(np.inf)
        return np.array(-data.size / 2 * np.log(var))

    # The same search, within the same bounds on lambda, as the classic fit.
    lmbda = _yeojohnson_normmax(x, log_likelihood=log_likelihood)
    return float(lmbda), log_mean


def _is_constant_feature(var: float, mean: float, n_samples: int) -> bool:
    """Near-zero variance test of sklearn's ``_is_constant_feature``."""
    eps = np.finfo(np.float64).eps
    return var <= n_samples * eps * var + (n_samples * mean * eps) ** 2


class SafePowerTransformer(PowerTransformer):
    """Variant of PowerTransformer that uses the scipy yeo-johnson functions
    which have been fixed to avoid overflow issues.

    Args:
        method: The power transform method, as in `PowerTransformer`.
        standardize: Whether to standardize the transformed output.
        copy: Whether to copy the input.
        relative_to_geometric_mean: Whether to compute the Yeo-Johnson transform
            relative to each column's geometric mean instead of relative to zero.
            This keeps columns that are far from zero compared with their spread,
            such as years, from collapsing to a constant. The output differs from
            the classic transform by a shift and a scale per column, so it is the
            same once standardized. Off by default, which keeps the
            unstandardized output equal to scikit-learn's.

    Attributes:
        log_means_: With ``relative_to_geometric_mean``, the mean of
            ``sign(x) * log(1 + |x|)`` of each fitted column.
    """

    _parameter_constraints: ClassVar[dict] = {
        **PowerTransformer._parameter_constraints,
        "relative_to_geometric_mean": ["boolean"],
    }

    def __init__(
        self,
        method: str = "yeo-johnson",
        *,
        standardize: bool = True,
        copy: bool = True,
        relative_to_geometric_mean: bool = False,
    ) -> None:
        super().__init__(method=method, standardize=standardize, copy=copy)
        self.relative_to_geometric_mean = relative_to_geometric_mean

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        # An instance pickled before the option existed lacks the attribute.
        self.__dict__.setdefault("relative_to_geometric_mean", False)

    def _is_centered(self) -> bool:
        return self.relative_to_geometric_mean and self.method == "yeo-johnson"

    @override
    def _fit(
        self, X: np.ndarray, y: object = None, force_transform: bool = False
    ) -> np.ndarray:
        if not self._is_centered():
            return super()._fit(X, y, force_transform=force_transform)
        X = self._check_input(X, in_fit=True)
        if not self.copy and not force_transform:
            X = X.copy()  # `fit` must not change its input

        self.lambdas_ = np.empty(X.shape[1], dtype=X.dtype)
        self.log_means_ = np.zeros(X.shape[1])
        with np.errstate(invalid="ignore"):  # hide NaN warnings
            for i in range(X.shape[1]):
                lmbda, log_mean = _fit_yeojohnson_centered(X[:, i])
                self.lambdas_[i], self.log_means_[i] = lmbda, log_mean
                if self.standardize or force_transform:
                    # With the stored lambda, as `transform` does.
                    X[:, i] = _yeojohnson_centered_transform(
                        X[:, i], self.lambdas_[i], log_mean
                    )

        if self.standardize:
            self._scaler = StandardScaler(copy=False).set_output(transform="default")
            X = self._scaler.fit_transform(X)
        return X

    @override
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_centered():
            return super().transform(X)
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)
        with np.errstate(invalid="ignore"):  # hide NaN warnings
            for i in range(X.shape[1]):
                X[:, i] = _yeojohnson_centered_transform(
                    X[:, i], self.lambdas_[i], self.log_means_[i]
                )
        return self._scaler.transform(X) if self.standardize else X

    @override
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_centered():
            return super().inverse_transform(X)
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)
        if self.standardize:
            X = self._scaler.inverse_transform(X)
        with np.errstate(invalid="ignore"):  # hide NaN warnings
            for i in range(X.shape[1]):
                X[:, i] = _yeojohnson_centered_inverse_transform(
                    X[:, i], self.lambdas_[i], self.log_means_[i]
                )
        return X

    # requires scipy >= 1.9
    # this is the default in scikit-learn main for versions > 1.7
    # see https://github.com/scikit-learn/scikit-learn/pull/31227
    @override
    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
        # sklearn >= 1.3 skips the search only for constant columns without NaNs.
        # On a constant column the search optimises rounding noise, so apply
        # sklearn's test to the observed values here, for every sklearn version.
        if x.size and _is_constant_feature(x.var(), x.mean(), x.size):
            return 1.0
        _, lmbda = _yeojohnson(x, lmbda=None)
        return lmbda  # type: ignore

    @override
    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        return _yeojohnson_transform(x, lmbda)

    @override
    def _yeo_johnson_inverse_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        return _yeojohnson_inverse_transform(x, lmbda)


__all__ = [
    "SafePowerTransformer",
]
