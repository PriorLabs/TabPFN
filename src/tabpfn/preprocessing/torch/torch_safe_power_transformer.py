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

    Mirrors the CPU ``safepower`` preset:
    ``SafePowerTransformer(standardize=False, relative_to_geometric_mean=True)`` then
    ``make_scaler_safe(StandardScaler())``. Each column's lambda maximises the
    Yeo-Johnson log-likelihood within scipy's overflow-safe bounds, found by a
    batched golden-section search. The transform is computed relative to each
    column's geometric mean, which keeps columns far from zero, such as years,
    from collapsing to a constant. ``_yeojohnson_centered_transform`` describes
    its two steps, which ``_log_ratios`` and ``_yeo_johnson`` implement here.
    Computes in float64, so it cannot run on MPS. Fit runs on column chunks and
    transform on row chunks to keep the float64 working set near
    ``_CHUNK_TARGET_BYTES``. A fit chunk holds at least one whole column, so
    beyond about 22M rows the fit working set grows with the row count.
    """

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit lambdas and scaler statistics on ``x`` of shape [n_rows, ...]."""
        x = x.reshape(x.shape[0], -1)
        n_cols = fit_chunk_columns(x.shape[0])
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


def fit_chunk_columns(n_rows: int) -> int:
    """Columns ``TorchSafePowerTransformer.fit`` fits together, from the first.

    A column's lambda depends on the columns of its chunk: the search runs as many
    steps as the widest bracket among them needs.
    """
    bytes_per_col = 8 * _OVERHEAD_FACTOR * max(n_rows, 1)
    return max(1, _CHUNK_TARGET_BYTES // bytes_per_col)


def _fit_chunk(x: torch.Tensor) -> dict[str, torch.Tensor]:
    lambdas, log_means = _fit_lambdas(x)
    y = _yeo_johnson(*_log_ratios(x, log_means), lambdas, log_means)
    y.masked_fill_(torch.isinf(y), torch.nan)
    impute_mean = torch.nan_to_num(y.nanmean(dim=0), nan=0.0)
    torch.where(torch.isnan(y), impute_mean, y, out=y)

    mean = y.mean(dim=0)
    var = y.var(dim=0, correction=0)
    scale = torch.where(_is_constant(var, mean, x.shape[0]), 1.0, var.sqrt())
    return {
        "lambdas": lambdas,
        "log_means": log_means,
        "impute_mean": impute_mean,
        "mean": mean,
        "scale": scale,
    }


def _transform_chunk(
    x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]
) -> torch.Tensor:
    lambdas = fitted_cache["lambdas"]
    # A cache fitted before the log means existed has none. Zero gives the
    # classic transform it was fitted with.
    log_means = fitted_cache.get("log_means", torch.zeros_like(lambdas))
    y = _yeo_johnson(*_log_ratios(x, log_means), lambdas, log_means)
    torch.where(torch.isfinite(y), y, fitted_cache["impute_mean"], out=y)
    return y.sub_(fitted_cache["mean"]).div_(fitted_cache["scale"])


def _fit_lambdas(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit every column: its maximum-likelihood lambda and its ``log_mean``.

    The batched counterpart of ``_fit_yeojohnson_centered``.
    """
    invalid = torch.isnan(x)
    n_valid = x.shape[0] - invalid.sum(dim=0)
    abs_max = x.abs().nan_to_num_(nan=0.0).amax(dim=0)
    n_neg = (x < 0).sum(dim=0)
    # Near-constant observed values, as in SafePowerTransformer; False when all-NaN.
    obs_mean = x.nansum(dim=0) / n_valid
    obs_var = (x - obs_mean).square_().nansum(dim=0) / n_valid
    constant = _is_constant(obs_var, obs_mean, n_valid)
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
    skip = constant | (n_valid == 0)
    lb = torch.where(skip, 0.0, lb)
    ub = torch.where(skip, 1.0, ub)

    # The log of the geometric mean is the mean of the logs. Each log carries the
    # sign of its value, so that a mostly negative column gets a negative mean.
    log_means = torch.log1p(x.abs()).copysign_(x).nansum(dim=0) / n_valid
    # Zero for the columns without a search: with lambda = 1 it leaves them
    # unchanged.
    log_means = torch.where(skip, 0.0, log_means)
    # Step 1 of the transform does not depend on lambda, so the search computes
    # it once and repeats only step 2.
    log_ratios, nonnegative = _log_ratios(x, log_means)
    # Without missing values the two fills below change nothing and are skipped.
    any_invalid = bool(invalid.any())

    def neg_llf(lambdas: torch.Tensor) -> torch.Tensor:
        # The Yeo-Johnson log-likelihood has two terms: one in the variance of the
        # transformed column and one in the sum of its logs. Dividing by the
        # geometric mean makes the second term constant, so the variance alone
        # decides.
        y = _yeo_johnson(log_ratios, nonnegative, lambdas, log_means)
        if any_invalid:
            y.masked_fill_(invalid, 0.0)
        mean = y.sum(dim=0) / n_valid
        y.sub_(mean)
        if any_invalid:
            y.masked_fill_(invalid, 0.0)
        var = y.square_().sum(dim=0) / n_valid
        llf = -n_valid / 2 * var.log()
        # Reject a vanishing or undefined variance, as SafePowerTransformer does.
        return torch.where(var >= finfo.tiny, -llf, torch.inf)

    lambdas = _golden_section_search(neg_llf, lb, ub)
    lambdas = torch.where(constant, 1.0, lambdas)
    return torch.where(n_valid == 0, torch.nan, lambdas), log_means


def _log_ratios(
    x: torch.Tensor, log_means: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step 1 of ``_yeojohnson_centered_transform``, for all columns at once.

    Returns the log that step 2, ``_yeo_johnson``, takes a power of, and whether
    each value is nonnegative once mostly negative columns have been mirrored.
    """
    # A mostly negative column is transformed as its mirror image, which
    # ``_yeo_johnson`` completes.
    x = x * torch.where(log_means >= 0, 1.0, -1.0).to(x.dtype)
    nonnegative = x >= 0
    abs_x = x.abs_()
    x_at_mean = log_means.abs().expm1()  # the x at which 1 + x equals g

    # Values x >= 0:
    #
    # 1. ratio = (1 + x) / g, kept as its log.
    #    For a row near the mean, log(1 + x) and log(g) are almost equal, and
    #    subtracting them would lose the digits that tell rows apart. There the
    #    ratio is formed from x - x_at_mean instead, a subtraction that is exact
    #    when the two are within a factor of two (Sterbenz lemma). Far from the
    #    mean that route fails in turn, and the two logs are safe to subtract.
    #
    # Values x < 0 get the log that the classic formula uses, in the same pass.
    near_mean = nonnegative & (abs_x >= x_at_mean / 2) & (abs_x <= 2 * x_at_mean)
    distance = (abs_x - x_at_mean).div_(1 + x_at_mean)
    logs = torch.where(near_mean, distance, abs_x).log1p_()
    del distance, abs_x
    far_from_mean = nonnegative & ~near_mean
    del near_mean
    logs.sub_(torch.where(far_from_mean, log_means.abs(), 0.0))
    del far_from_mean
    # For x < 0 the classic formula uses the log with a negative sign.
    return logs.mul_(torch.where(nonnegative, 1.0, -1.0)), nonnegative


def _yeo_johnson(
    log_ratios: torch.Tensor,
    nonnegative: torch.Tensor,
    lambdas: torch.Tensor,
    log_means: torch.Tensor,
) -> torch.Tensor:
    """Step 2 of ``_yeojohnson_centered_transform``, for all columns at once.

    Takes ``log_ratios`` and ``nonnegative`` from ``_log_ratios``. With
    ``log_means = 0`` the result is the classic transform.
    """
    # A mostly negative column is transformed as its mirror image. Yeo-Johnson
    # is symmetric: flipping the sign of x and replacing lambda by 2 - lambda
    # flips the sign of the result. ``_log_ratios`` flipped x, and the last line
    # here flips the result back.
    mirrored = log_means < 0
    lambdas = torch.where(mirrored, 2 - lambdas, lambdas)
    log_means = log_means.abs()

    # The search calls this function once per step. Each ``if`` below skips
    # passes over the data that would change nothing.
    #
    # 2. y = (ratio^power - 1) / power, from the log of the ratio.
    #    The power is lambda for the values x >= 0. The values x < 0 go through
    #    the same line with the power of the classic formula, lambda - 2.
    power = torch.where(nonnegative, lambdas, lambdas - 2)
    y = torch.mul(power, log_ratios).expm1_().div_(power)
    # A power of 0 leaves NaN in y. The limit of the line above is the log itself.
    if ((lambdas.abs() < _EPS) | ((lambdas - 2).abs() < _EPS)).any():
        torch.where(power.abs() < _EPS, log_ratios, y, out=y)
    del power

    # Values x < 0 now hold the classic result, which is accurate for them.
    # Compared with it, step 1 scaled and shifted the values x >= 0. The same
    # scale and shift here make the two parts join at x = 0.
    if not nonnegative.all():
        lambda_is_0 = lambdas.abs() < _EPS
        safe_lambdas = lambdas.masked_fill(lambda_is_0, 1.0)
        value_at_zero = (-safe_lambdas * log_means).expm1_().div_(safe_lambdas)
        value_at_zero = torch.where(lambda_is_0, -log_means, value_at_zero)
        rescaled = (y * (-lambdas * log_means).exp_()).add_(value_at_zero)
        torch.where(nonnegative, y, rescaled, out=y)
        del rescaled
    if mirrored.any():
        y.mul_(torch.where(mirrored, -1.0, 1.0))
    return y


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


def _is_constant(
    var: torch.Tensor, mean: torch.Tensor, n: int | torch.Tensor
) -> torch.Tensor:
    """Sklearn's ``_is_constant_feature`` near-zero variance test."""
    return var <= n * _EPS * var + (n * mean * _EPS) ** 2
