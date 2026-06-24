#  Copyright (c) Prior Labs GmbH 2026.

"""Utility functions for preprocessing steps."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


def make_scaler_safe(
    name: str,
    scaler: TransformerMixin,
    categorical_features: list[int] | None = None,
) -> Pipeline:
    """Wrap a scaler with steps that ensure all input and output values are finite.

    Inserts inf-to-nan conversion and mean imputation both before and after the
    scaler to guard against edge cases such as division-by-zero during scaling
    or non-finite values in the input data.

    Args:
        name: Name for the scaler step in the resulting pipeline.
        scaler: The scaler / transformer to wrap.
        categorical_features: Column indices (into the scaler's input) to impute
            with the per-column mode instead of the mean. ``None``/empty keeps
            mean imputation for every column.

    Returns:
        A `Pipeline` with sanitization steps surrounding the scaler.
    """
    return Pipeline(
        steps=[
            *_make_finite_steps("pre", categorical_features),
            (name, scaler),
            *_make_finite_steps("post", categorical_features),
        ],
    )


def wrap_with_safe_standard_scaler(
    transformer: TransformerMixin,
    categorical_features: list[int] | None = None,
) -> Pipeline:
    """Wrap a transformer with a safely-guarded `StandardScaler`.

    Useful for transformers like `PowerTransformer` that can produce inf
    values in edge cases, which would crash a subsequent `StandardScaler`.

    Args:
        transformer: The transformer to apply before standard scaling.
        categorical_features: Column indices to impute with the per-column mode
            instead of the mean. Column order is preserved by ``transformer``
            (the registry transforms are all 1-to-1), so the same indices apply.

    Returns:
        A Pipeline of `[transformer_name, safe_standard_scaler]`.
    """
    return Pipeline(
        steps=[
            ("input_transformer", transformer),
            (
                "standard",
                make_scaler_safe("standard", StandardScaler(), categorical_features),
            ),
        ],
    )


def _mode_col(x_col: np.ndarray) -> float:
    finite = x_col[np.isfinite(x_col)]
    if finite.size == 0:
        return np.nan
    values, counts = np.unique(finite, return_counts=True)
    return values[np.argmax(counts)]


def mode(x: np.ndarray) -> np.ndarray | float:
    """Compute the mode of each column along the row dimension (axis 0).

    For every column, returns the most frequently occurring value, ignoring
    non-finite entries (NaN, +inf, -inf) so the result is unaffected by missing
    data. Ties are broken deterministically by choosing the smallest value.
    Columns with no finite values yield NaN.

    This is the categorical counterpart to ``np.nanmean(x, axis=0)``: it
    produces per-feature imputation values that respect the discrete nature of
    categorical data instead of averaging category codes.

    Args:
        x: Array of shape ``(n_samples, n_features)``.

    Returns:
        Array of shape ``(n_features,)`` with the per-column mode.
    """
    if x.ndim == 1:
        return _mode_col(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got {x.ndim}D.")

    return np.asarray([_mode_col(x[:, col]) for col in range(x.shape[1])])


class _NoInverseImputer(SimpleImputer):
    """SimpleImputer (mean) that imputes selected columns with the mode instead.

    Returns the input unchanged on ``inverse_transform``. ``categorical_features``
    lists the columns whose missing values should be filled with the per-column
    mode rather than the mean; all other columns keep the mean strategy.
    """

    def __init__(
        self,
        *,
        missing_values: float = np.nan,
        strategy: str = "mean",
        keep_empty_features: bool = False,
        categorical_features: list[int] | None = None,
    ) -> None:
        self.categorical_features = categorical_features
        super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            keep_empty_features=keep_empty_features,
        )

    @override
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X

    @override
    def fit(self, X: np.ndarray, y: object = None) -> _NoInverseImputer:
        super().fit(X, y)
        if self.categorical_features:
            cats = np.array(self.categorical_features)
            modes = np.asarray(mode(np.asarray(X, dtype=float)[:, cats]))
            valid = ~np.isnan(modes)
            self.statistics_[cats[valid]] = modes[valid]
        return self


def _identity(x: np.ndarray) -> np.ndarray:
    """Return input unchanged. Used as a no-op inverse for FunctionTransformer."""
    return x


def _replace_inf_with_nan(x: np.ndarray) -> np.ndarray:
    """Replace +inf and -inf with NaN, leaving existing NaN values unchanged."""
    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)


def _make_finite_steps(
    suffix: str,
    categorical_features: list[int] | None = None,
) -> list[tuple[str, TransformerMixin]]:
    """Create pipeline steps that replace non-finite values and impute NaNs.

    Args:
        suffix: Appended to step names to ensure uniqueness (e.g. "pre", "post").
        categorical_features: Column indices imputed with the mode rather than
            the mean. ``None``/empty keeps mean imputation everywhere.

    Returns:
        A list of `(name, transformer)` tuples for use as sklearn Pipeline steps.
    """
    return [
        (
            f"inf_to_nan_{suffix}",
            FunctionTransformer(
                func=_replace_inf_with_nan,
                inverse_func=_identity,
                check_inverse=False,
            ),
        ),
        (
            f"nan_impute_{suffix}",
            _NoInverseImputer(
                missing_values=np.nan,
                strategy="mean",
                # keep empty features so inverse_transform dimensions are consistent
                keep_empty_features=True,
                categorical_features=categorical_features,
            ),
        ),
    ]
