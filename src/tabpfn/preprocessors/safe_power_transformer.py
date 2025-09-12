"""Safe Power Transformer."""

from __future__ import annotations

from typing_extensions import override

import numpy as np
from scipy import optimize
from sklearn.preprocessing import PowerTransformer


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


def _yeojohnson_normmax(x: np.ndarray, brack: float | None = None) -> float:
    """Compute optimal Yeo-Johnson transform parameter.
    Compute optimal Yeo-Johnson transform parameter for input data, using
    maximum likelihood estimation.

    """

    def _neg_llf(lmbda: float, data: np.ndarray) -> np.ndarray:
        llf = _yeojohnson_llf(lmbda, data)
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

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        x_inv[pos] = np.expm1(x[pos])
    else:  # lmbda != 0
        # more stable version of: (x * lmbda + 1) ** (1 / lmbda) - 1
        x_inv[pos] = np.expm1(np.log(x[pos] * lmbda + 1) / lmbda)

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        # more stable version of: 1 - (-(2 - lmbda) * x + 1) ** (1 / (2 - lmbda))
        x_inv[~pos] = -np.expm1(np.log(-(2 - lmbda) * x[~pos] + 1) / (2 - lmbda))
    else:  # lmbda == 2
        x_inv[~pos] = -np.expm1(-x[~pos])

    return x_inv


class SafePowerTransformer(PowerTransformer):
    """Variant of PowerTransformer that uses the scipy yeo-johnson functions
    which have been fixed to avoid overflow issues.
    """

    def __init__(
        self,
        method: str = "yeo-johnson",
        *,
        standardize: bool = True,
        copy: bool = True,
    ) -> None:
        super().__init__(method=method, standardize=standardize, copy=copy)

    # requires scipy >= 1.9
    # this is the default in scikit-learn main for versions > 1.7
    # see https://github.com/scikit-learn/scikit-learn/pull/31227
    @override
    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
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
