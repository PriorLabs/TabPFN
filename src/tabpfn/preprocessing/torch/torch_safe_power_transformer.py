#  Copyright (c) Prior Labs GmbH 2026.

"""Torch implementation of the ``safepower`` preset."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

_INV_GOLDEN = (math.sqrt(5) - 1) / 2
# scipy.optimize.fminbound's xtol, as used by SafePowerTransformer.
_XTOL = 1.48e-8
_EPS = torch.finfo(torch.float64).eps
# Float64 working set of one fit or transform chunk, including temporaries.
_CHUNK_TARGET_BYTES = 1024**3
# Float64 tensors alive per chunk element at peak: measured about 4.3 on CUDA
# for both fit and transform, rounded up for margin.
_OVERHEAD_FACTOR = 6


class TorchSafePowerTransformer:
    """Yeo-Johnson power transform followed by a NaN-safe standard scaler.

    Mirrors the CPU ``safepower`` preset: ``SafePowerTransformer(standardize=False)``
    then ``make_scaler_safe(StandardScaler())``. Each column's lambda maximises the
    Yeo-Johnson log-likelihood within scipy's overflow-safe bounds, found by a
    batched golden-section search. Computes in float64, so it cannot run on MPS.
    Fit runs on column chunks and transform on row chunks, so the float64
    working set stays near ``_CHUNK_TARGET_BYTES`` for any number of rows.
    """

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit lambdas and scaler statistics on ``x`` of shape [n_rows, ...]."""
        x = x.reshape(x.shape[0], -1)
        bytes_per_col = 8 * _OVERHEAD_FACTOR * max(x.shape[0], 1)
        n_cols = max(1, _CHUNK_TARGET_BYTES // bytes_per_col)
        caches = [
            _fit_chunk(x[:, i : i + n_cols].to(torch.float64))
            for i in range(0, x.shape[1], n_cols)
        ]
        return {k: torch.cat([c[k] for c in caches]) for k in caches[0]}

    def transform(
        self, x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply the fitted transform; the output has no NaNs."""
        shape = x.shape
        x = x.reshape(shape[0], -1)
        out = torch.empty_like(x)
        bytes_per_row = 8 * _OVERHEAD_FACTOR * max(x.shape[1], 1)
        n_rows = max(1, _CHUNK_TARGET_BYTES // bytes_per_row)
        for i in range(0, x.shape[0], n_rows):
            out[i : i + n_rows] = _transform_chunk(
                x[i : i + n_rows].to(torch.float64), fitted_cache
            )
        return out.reshape(shape)


def _fit_chunk(x: torch.Tensor) -> dict[str, torch.Tensor]:
    lambdas = _fit_lambdas(x)
    y = _yeo_johnson(x, torch.log1p(x.abs()), lambdas)
    y.masked_fill_(torch.isinf(y), torch.nan)
    impute_mean = torch.nan_to_num(y.nanmean(dim=0), nan=0.0)
    torch.where(torch.isnan(y), impute_mean, y, out=y)

    mean = y.mean(dim=0)
    var = y.var(dim=0, correction=0)
    scale = torch.where(_is_constant(var, mean, x.shape[0]), 1.0, var.sqrt())
    return {
        "lambdas": lambdas,
        "impute_mean": impute_mean,
        "mean": mean,
        "scale": scale,
    }


def _transform_chunk(
    x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]
) -> torch.Tensor:
    y = _yeo_johnson(x, torch.log1p(x.abs()), fitted_cache["lambdas"])
    torch.where(torch.isfinite(y), y, fitted_cache["impute_mean"], out=y)
    return y.sub_(fitted_cache["mean"]).div_(fitted_cache["scale"])


def _fit_lambdas(x: torch.Tensor) -> torch.Tensor:
    """Per-column ``_yeojohnson_normmax`` with sklearn's special cases."""
    invalid = torch.isnan(x)
    n_valid = x.shape[0] - invalid.sum(dim=0)
    abs_max = x.abs().nan_to_num_(nan=0.0).amax(dim=0)
    n_neg = (x < 0).sum(dim=0)
    any_neg, all_neg = n_neg > 0, n_neg == n_valid

    finfo = torch.finfo(torch.float64)
    log_tiny = (math.log(finfo.tiny) - math.log(_EPS)) / 2
    log_max = (math.log(finfo.max) + math.log(_EPS)) / 2
    log1p_max_x = torch.log1p(20 * abs_max)
    lb, ub = log_tiny / log1p_max_x, log_max / log1p_max_x
    neg_lb, neg_ub = 2 - ub, 2 - lb
    lb, ub = (
        torch.where(all_neg, neg_lb, torch.where(any_neg, neg_lb.maximum(lb), lb)),
        torch.where(all_neg, neg_ub, torch.where(any_neg, neg_ub.minimum(ub), ub)),
    )

    # Columns without a search get finite dummy bounds and are overwritten below.
    skip = (abs_max == 0) | (n_valid == 0)
    lb = torch.where(skip, 0.0, lb)
    ub = torch.where(skip, 1.0, ub)

    # Lambda-independent terms of the log-likelihood, computed once.
    log1p_abs = torch.log1p(x.abs())
    sum_log = log1p_abs.copysign(x).nansum(dim=0)

    def neg_llf(lambdas: torch.Tensor) -> torch.Tensor:
        y = _yeo_johnson(x, log1p_abs, lambdas).masked_fill_(invalid, 0.0)
        mean = y.sum(dim=0) / n_valid
        y.sub_(mean).masked_fill_(invalid, 0.0)
        var = y.square_().sum(dim=0) / n_valid
        llf = -n_valid / 2 * var.log() + (lambdas - 1) * sum_log
        # scipy rejects near-zero variance as an infinite likelihood.
        return torch.where(var < finfo.tiny, torch.inf, -llf)

    lambdas = _golden_section_search(neg_llf, lb, ub)
    lambdas = torch.where(abs_max == 0, 1.0, lambdas)
    lambdas = torch.where(n_valid == 0, torch.nan, lambdas)
    # sklearn checks constant columns with NaN-propagating statistics.
    mean, var = x.mean(dim=0), x.var(dim=0, correction=0)
    return torch.where(_is_constant(var, mean, x.shape[0]), 1.0, lambdas)


def _yeo_johnson(
    x: torch.Tensor, log1p_abs: torch.Tensor, lambdas: torch.Tensor
) -> torch.Tensor:
    """Yeo-Johnson transform with one lambda per column, as in scipy.

    Both branches equal ``sign(x) * expm1(m * log1p(|x|)) / m``, with
    ``m = lambda`` for ``x >= 0`` and ``m = 2 - lambda`` otherwise, and the
    ``m -> 0`` limit ``sign(x) * log1p(|x|)``.
    """
    pos = x >= 0
    m = torch.where(pos, lambdas, 2 - lambdas)
    m_abs = m.abs()
    # scipy's lambda == 0 test is strict, its lambda == 2 test is not.
    m_is_0 = torch.where(pos, m_abs < _EPS, m_abs <= _EPS)
    del m_abs, pos
    m.masked_fill_(m_is_0, 1.0)
    y = torch.mul(m, log1p_abs).expm1_().div_(m)
    del m
    torch.where(m_is_0, log1p_abs, y, out=y)
    return y.copysign_(x)


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
        # Ties move right, as in fminbound.
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
    return var <= n * _EPS * var + (n * mean * _EPS) ** 2
