#  Copyright (c) Prior Labs GmbH 2026.

"""Torch implementation of the ``safepower`` preset."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

_INV_GOLDEN = (math.sqrt(5) - 1) / 2
# scipy.optimize.fminbound's xtol, as used by SafePowerTransformer.
_XTOL = 1.48e-8
# The lambda search holds about this many float64 copies of a column chunk.
_FIT_OVERHEAD_FACTOR = 8
_FIT_TARGET_BYTES = 2 * 1024**3


class TorchSafePowerTransformer:
    """Yeo-Johnson power transform followed by a NaN-safe standard scaler.

    Mirrors the CPU ``safepower`` preset: ``SafePowerTransformer(standardize=False)``
    then ``make_scaler_safe(StandardScaler())``. Each column's lambda maximises the
    Yeo-Johnson log-likelihood within scipy's overflow-safe bounds, found by a
    batched golden-section search. Computes in float64, so it cannot run on MPS.
    """

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit lambdas and scaler statistics on ``x`` of shape [n_rows, ...]."""
        x = x.reshape(x.shape[0], -1).to(torch.float64)
        lambdas = self._fit_lambdas(x)

        y = _yeo_johnson(x, lambdas)
        y = torch.where(torch.isinf(y), torch.nan, y)
        impute_mean = torch.nan_to_num(y.nanmean(dim=0), nan=0.0)
        y = torch.where(torch.isnan(y), impute_mean, y)

        mean = y.mean(dim=0)
        var = y.var(dim=0, correction=0)
        scale = torch.where(_is_constant(var, mean, x.shape[0]), 1.0, var.sqrt())
        return {
            "lambdas": lambdas,
            "impute_mean": impute_mean,
            "mean": mean,
            "scale": scale,
        }

    def transform(
        self, x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply the fitted transform; the output has no NaNs."""
        shape, dtype = x.shape, x.dtype
        x = x.reshape(shape[0], -1).to(torch.float64)
        y = _yeo_johnson(x, fitted_cache["lambdas"])
        y = torch.where(torch.isfinite(y), y, fitted_cache["impute_mean"])
        y = (y - fitted_cache["mean"]) / fitted_cache["scale"]
        return y.reshape(shape).to(dtype)

    def _fit_lambdas(self, x: torch.Tensor) -> torch.Tensor:
        bytes_per_col = 8 * _FIT_OVERHEAD_FACTOR * max(x.shape[0], 1)
        n_cols = max(1, _FIT_TARGET_BYTES // bytes_per_col)
        return torch.cat(
            [
                self._fit_lambdas_chunk(x[:, i : i + n_cols])
                for i in range(0, x.shape[1], n_cols)
            ]
        )

    def _fit_lambdas_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """Per-column ``_yeojohnson_normmax`` with sklearn's special cases."""
        valid = ~torch.isnan(x)
        n_valid = valid.sum(dim=0)
        abs_max = torch.where(valid, x.abs(), 0.0).amax(dim=0)
        any_neg = (x < 0).any(dim=0)
        all_neg = (x < 0).sum(dim=0) == n_valid

        finfo = torch.finfo(torch.float64)
        log_eps = math.log(finfo.eps)
        log_tiny = (math.log(finfo.tiny) - log_eps) / 2
        log_max = (math.log(finfo.max) + log_eps) / 2
        log1p_max_x = torch.log1p(20 * abs_max)
        lb, ub = log_tiny / log1p_max_x, log_max / log1p_max_x
        lb, ub = (
            torch.where(
                all_neg, 2 - ub, torch.where(any_neg, (2 - ub).maximum(lb), lb)
            ),
            torch.where(
                all_neg, 2 - lb, torch.where(any_neg, (2 - lb).minimum(ub), ub)
            ),
        )

        # Columns without a search get finite dummy bounds and are overwritten below.
        skip = (abs_max == 0) | (n_valid == 0)
        lb = torch.where(skip, 0.0, lb)
        ub = torch.where(skip, 1.0, ub)

        lambdas = _golden_section_search(lambda lmb: _neg_llf(x, valid, lmb), lb, ub)
        lambdas = torch.where(abs_max == 0, 1.0, lambdas)
        lambdas = torch.where(n_valid == 0, torch.nan, lambdas)
        # sklearn checks constant columns with NaN-propagating statistics.
        mean, var = x.mean(dim=0), x.var(dim=0, correction=0)
        return torch.where(_is_constant(var, mean, x.shape[0]), 1.0, lambdas)


def _yeo_johnson(x: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """Yeo-Johnson transform with one lambda per column, as in scipy."""
    eps = torch.finfo(torch.float64).eps
    pos = x >= 0
    log1p_pos = torch.log1p(torch.where(pos, x, 0.0))
    log1p_neg = torch.log1p(torch.where(pos, 0.0, -x))

    lmb_is_0 = lambdas.abs() < eps
    lmb_is_2 = (lambdas - 2).abs() <= eps
    lmb_pos = torch.where(lmb_is_0, 1.0, lambdas)
    lmb_neg = torch.where(lmb_is_2, 1.0, 2 - lambdas)

    out_pos = torch.where(
        lmb_is_0, log1p_pos, torch.expm1(lmb_pos * log1p_pos) / lmb_pos
    )
    out_neg = torch.where(
        lmb_is_2, -log1p_neg, -torch.expm1(lmb_neg * log1p_neg) / lmb_neg
    )
    return torch.where(pos, out_pos, torch.where(torch.isnan(x), x, out_neg))


def _neg_llf(
    x: torch.Tensor, valid: torch.Tensor, lambdas: torch.Tensor
) -> torch.Tensor:
    """Negative Yeo-Johnson log-likelihood per column, over non-NaN values."""
    n_valid = valid.sum(dim=0)
    y = torch.where(valid, _yeo_johnson(x, lambdas), 0.0)
    mean = y.sum(dim=0) / n_valid
    var = torch.where(valid, (y - mean) ** 2, 0.0).sum(dim=0) / n_valid
    sum_log = torch.where(valid, x.sign() * torch.log1p(x.abs()), 0.0).sum(dim=0)
    llf = -n_valid / 2 * var.log() + (lambdas - 1) * sum_log
    # scipy rejects near-zero variance as an infinite likelihood.
    return torch.where(var < torch.finfo(torch.float64).tiny, torch.inf, -llf)


def _golden_section_search(
    f: Callable[[torch.Tensor], torch.Tensor], lb: torch.Tensor, ub: torch.Tensor
) -> torch.Tensor:
    """Minimise ``f`` elementwise on ``[lb, ub]`` to within ``_XTOL``."""
    a, b = lb.clone(), ub.clone()
    width = float((b - a).max()) if a.numel() else 0.0
    n_iter = math.ceil(math.log(max(width, _XTOL) / _XTOL) / -math.log(_INV_GOLDEN))
    c = b - _INV_GOLDEN * (b - a)
    d = a + _INV_GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(n_iter):
        left = fc < fd
        a = torch.where(left, a, c)
        b = torch.where(left, d, b)
        new = torch.where(left, b - _INV_GOLDEN * (b - a), a + _INV_GOLDEN * (b - a))
        f_new = f(new)
        c, d = torch.where(left, new, d), torch.where(left, c, new)
        fc, fd = torch.where(left, f_new, fd), torch.where(left, fc, f_new)
    return (a + b) / 2


def _is_constant(var: torch.Tensor, mean: torch.Tensor, n: int) -> torch.Tensor:
    """Sklearn's ``_is_constant_feature`` near-zero variance test."""
    eps = torch.finfo(torch.float64).eps
    return var <= n * eps * var + (n * mean * eps) ** 2
