#  Copyright (c) Prior Labs GmbH 2026.

"""KDI Transformer with NaN."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.interpolate as spip
import scipy.stats as spst
import torch
from scipy import integrate
from scipy.special import ndtri
from sklearn.preprocessing import PowerTransformer

try:
    from kditransform import KDITransformer
    from kditransform.kdi_transformer import BOUNDS_THRESHOLD
    from kditransform.ksum import betas_for_order, h_Gauss_to_K, ksum_numba

    # `norm.ppf(BOUNDS_THRESHOLD - spacing(1))` and its mirror, computed once.
    _NORMAL_CLIP_MIN = float(ndtri(BOUNDS_THRESHOLD - np.spacing(1)))
    _NORMAL_CLIP_MAX = float(ndtri(1 - (BOUNDS_THRESHOLD - np.spacing(1))))
    # This import fails on some systems, due to problems with numba
except ImportError:
    KDITransformer = PowerTransformer  # fallback to avoid error

# Track whether we've warned the user about missing kditransform
_warned_about_missing_kditransform = False

ALPHAS = (
    0.05,
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
    1.2,
    1.5,
    1.8,
    2.0,
    2.5,
    3.0,
    5.0,
)


class KDITransformerWithNaN(KDITransformer):
    """KDI transformer that can handle NaN values.

    It performs KDI with NaNs replaced by mean values and then fills the NaN values
    with NaNs after the transformation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        output_distribution: str = "uniform",
        *,
        standardize: bool = True,
        copy: bool = True,
    ) -> None:
        # ``kditransform`` exposes ``alpha`` and ``output_distribution`` but the
        # PowerTransformer fallback does not. To keep compatibility across both
        # backends, only pass the parameters that are supported by the active
        # base class.
        if KDITransformer is PowerTransformer:
            self.alpha = alpha
            self.output_distribution = output_distribution
            super().__init__(standardize=standardize, copy=copy)
        else:
            self.standardize = standardize
            super().__init__(
                alpha=alpha,
                output_distribution=output_distribution,
                copy=copy,
            )

    def _more_tags(self) -> dict:  # sklearn < 1.6
        return {"allow_nan": True}

    def __sklearn_tags__(self):  # sklearn >= 1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(
        self,
        X: torch.Tensor | np.ndarray,
        y: Any | None = None,
    ) -> KDITransformerWithNaN:
        """Fit the transformer."""
        global _warned_about_missing_kditransform  # noqa: PLW0603
        if (
            KDITransformer is PowerTransformer
            and not _warned_about_missing_kditransform
        ):
            warnings.warn(
                "Cannot use KDITransformer because `kditransform` is not installed. "
                "Using `PowerTransformer` as fallback. This warning is only shown "
                "once per Python interpreter instance.",
                UserWarning,
                stacklevel=2,
            )
            _warned_about_missing_kditransform = True

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # If all-nan or empty, nanmean returns nan.
        self.imputation_values_ = np.nan_to_num(np.nanmean(X, axis=0), nan=0)
        X = np.nan_to_num(X, nan=self.imputation_values_)

        return super().fit(X, y)  # type: ignore

    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        """Transform the data."""
        # if tensor convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Calculate the NaN mask for the current dataset
        nan_mask = np.isnan(X)

        # Replace NaNs with the mean of columns
        X = np.nan_to_num(X, nan=self.imputation_values_)

        # Apply the transformation
        X = super().transform(X)

        # Reintroduce NaN values based on the current dataset's mask
        X[nan_mask] = np.nan

        return X  # type: ignore

    def fit_transform(
        self, X: torch.Tensor | np.ndarray, y: Any | None = None
    ) -> np.ndarray:
        """Fit the transformer and transform the data."""
        self.fit(X, y)
        return self.transform(X)

    # The two methods below mirror `kditransform.KDITransformer`'s and must give the
    # same output to the last bit; `tests/test_preprocessing/test_kdi_transformer.py`
    # checks that against the upstream class. They differ only in how the numbers are
    # reached: the quantiles of the kernel density come from the column already sorted
    # for the kernel sum instead of a second `np.quantile` pass, the bandwidth
    # conversion constant is computed once instead of per column, and the normal
    # output distribution goes through `ndtri` directly instead of `norm.ppf`.

    def _polyexp_dense_fit(  # noqa: C901
        self, X: np.ndarray, alphas: list, random_state: np.random.RandomState
    ) -> None:
        n_samples, _n_features = X.shape
        wgts = np.ones(n_samples).astype(X.dtype)
        betas = betas_for_order(self.polyexp_order)
        # `h_Gauss_to_K(h, betas)` is `h` times a constant of `betas`.
        h_to_k = h_Gauss_to_K(1.0, betas)

        if self.polyexp_eval == "uniform":
            n_eval = n_samples if self.n_quantiles is None else self.n_quantiles
        elif self.polyexp_eval == "train":
            n_eval = n_samples + (n_samples - 1) * 1
        elif self.polyexp_eval == "auto":
            n_eval = (
                n_samples if self.n_quantiles is None else self.n_quantiles
            ) + self.n_quantiles_

        density_out = np.zeros(n_eval).astype(X.dtype)
        counts = np.zeros(n_eval).astype(np.int64)
        coefs = np.zeros_like(betas)
        Ly = np.zeros((self.polyexp_order + 1, n_samples), order="C")
        Ry = np.zeros((self.polyexp_order + 1, n_samples), order="C")

        quantiles_per_column = []
        for column, bandwidth in zip(X.T, alphas, strict=False):
            col = column
            alpha = bandwidth
            if self.subsample_ < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample_, replace=False
                )
                col = col.take(subsample_idx, mode="clip")
            if np.var(col) == 0:
                quantiles = col[0] * np.ones_like(self.references_)
            else:
                xmin = np.min(col)
                xmax = np.max(col)
                if alpha == "scott":
                    alpha = np.power(n_samples, (-1.0 / (1 + 4)))
                elif alpha == "silverman":
                    alpha = np.power(n_samples * (1 + 2.0) / 4.0, -1.0 / (1 + 4))
                h = alpha * np.std(col) * h_to_k
                col_mean = np.mean(col)
                col -= col_mean
                col_sort = np.sort(col)
                if self.polyexp_eval == "uniform":
                    col_eval = np.linspace(np.min(col), np.max(col), n_eval)
                elif self.polyexp_eval == "train":
                    midpts = col_sort[:-1] + 0.50 * np.diff(col_sort)
                    col_eval = np.sort(np.concatenate([col_sort, midpts]))
                elif self.polyexp_eval == "auto":
                    col_u = np.linspace(
                        np.min(col),
                        np.max(col),
                        n_samples if self.n_quantiles is None else self.n_quantiles,
                    )
                    col_s = _sorted_quantile_linear(col_sort, self.references_)
                    col_eval = np.sort(np.concatenate([col_u, col_s]))
                    assert len(col_eval) == n_eval

                ksum_numba(
                    col_sort,
                    wgts,
                    col_eval,
                    h,
                    betas,
                    density_out,
                    counts,
                    coefs,
                    Ly,
                    Ry,
                )
                density_out /= n_samples * h
                density_out[np.isnan(density_out)] = 1e-300
                density_out[~np.isfinite(density_out)] = 1e-300
                col += col_mean
                col_sort += col_mean
                col_eval += col_mean
                T = integrate.cumulative_trapezoid(density_out, col_eval, initial=0)
                intcx1 = 0.0
                intcxN = T[-1]
                m = 1.0 / (intcxN - intcx1)
                b = -m * intcx1
                T = m * T + b

                inverse_func = spip.interp1d(
                    T, col_eval, bounds_error=False, fill_value=(xmin, xmax)
                )
                quantiles = inverse_func(self.references_)
            quantiles_per_column.append(quantiles)
        self.quantiles_ = np.transpose(quantiles_per_column)
        # Make sure that quantiles are monotonically increasing
        self.quantiles_ = np.maximum.accumulate(self.quantiles_, axis=0)

    def _transform_col(
        self,
        X_col: np.ndarray,
        quantiles: np.ndarray,
        inverse: bool,  # noqa: FBT001  called positionally by the base class
    ) -> np.ndarray:
        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = spst.norm.cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if output_distribution == "normal":
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == "uniform":
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the mean, so
            # repeated values (and hence repeated quantiles) use both extremes.
            X_col[isfinite_mask] = 0.5 * (
                np.interp(X_col_finite, quantiles, self.references_)
                - np.interp(-X_col_finite, -quantiles[::-1], -self.references_[::-1])
            )
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = ndtri(X_col)
                    # clip so the inverse transform stays consistent instead of
                    # mapping to infinity
                    X_col = np.clip(X_col, _NORMAL_CLIP_MIN, _NORMAL_CLIP_MAX)
                # else output distribution is uniform and the ppf is the identity

        return X_col


def _sorted_quantile_linear(col_sort: np.ndarray, q: np.ndarray) -> np.ndarray:
    """`np.quantile(col, q)` (linear method) read off the already sorted column.

    Reproduces numpy's arithmetic, including its `t >= 0.5` branch of the
    interpolation, so the result is bit-identical.
    """
    n = len(col_sort)
    virtual = q * (n - 1)
    previous = np.floor(virtual)
    gamma = virtual - previous
    previous = previous.astype(np.intp)
    following = np.minimum(previous + 1, n - 1)
    lower = col_sort[previous]
    upper = col_sort[following]
    diff = upper - lower
    out = lower + diff * gamma
    np.subtract(upper, diff * (1 - gamma), out=out, where=gamma >= 0.5)
    return out


def get_all_kdi_transformers() -> dict[str, KDITransformerWithNaN]:
    """Get all KDI transformers."""
    try:
        all_preprocessors = {
            "kdi": KDITransformerWithNaN(alpha=1.0, output_distribution="normal"),
            "kdi_uni": KDITransformerWithNaN(
                alpha=1.0,
                output_distribution="uniform",
            ),
        }
        for alpha in ALPHAS:
            all_preprocessors[f"kdi_alpha_{alpha}"] = KDITransformerWithNaN(
                alpha=alpha,
                output_distribution="normal",
            )
            all_preprocessors[f"kdi_alpha_{alpha}_uni"] = KDITransformerWithNaN(
                alpha=alpha,
                output_distribution="uniform",
            )
        return all_preprocessors
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "KDITransformerWithNaN",
    "get_all_kdi_transformers",
]
