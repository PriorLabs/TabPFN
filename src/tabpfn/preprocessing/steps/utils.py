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


def is_identity_transformer(transformer: str | TransformerMixin) -> bool:
    """Whether `transformer` hands back what it is given, untouched.

    Used to decide that a transform need not be run at all -- so it answers only for the
    cases where that is provable, and declines everything else.

    `"passthrough"` says so outright. A `FunctionTransformer` qualifies when it has no
    `func` *and* does not validate: `validate=True` is not nothing, it sends the input
    through `check_array`, which coerces the dtype and refuses a NaN that would
    otherwise pass. Its remaining arguments cannot reach the values -- `kw_args` only
    applies to a `func`, `check_inverse` is skipped without one, `accept_sparse` only
    matters under `validate`, and `inverse_func` and `feature_names_out` are not on the
    forward path.

    The type is compared exactly rather than with `isinstance` because of what this
    decides. A subclass may override `transform` and do something else entirely with
    `func` still unset, and no inspection here would catch it. Refusing a harmless
    subclass costs whichever fallback the caller keeps for the general case; accepting
    an unfaithful one costs the result.
    """
    if isinstance(transformer, str):
        return transformer == "passthrough"
    return (
        type(transformer) is FunctionTransformer
        and transformer.func is None
        and not transformer.validate
    )


def make_scaler_safe(name: str, scaler: TransformerMixin) -> Pipeline:
    """Wrap a scaler with steps that ensure all input and output values are finite.

    Inserts inf-to-nan conversion and mean imputation both before and after the
    scaler to guard against edge cases such as division-by-zero during scaling
    or non-finite values in the input data.

    Args:
        name: Name for the scaler step in the resulting pipeline.
        scaler: The scaler / transformer to wrap.

    Returns:
        A `Pipeline` with sanitization steps surrounding the scaler.
    """
    return Pipeline(
        steps=[*_make_finite_steps("pre"), (name, scaler), *_make_finite_steps("post")],
    )


def wrap_with_safe_standard_scaler(
    transformer: TransformerMixin,
) -> Pipeline:
    """Wrap a transformer with a safely-guarded `StandardScaler`.

    Useful for transformers like `PowerTransformer` that can produce inf
    values in edge cases, which would crash a subsequent `StandardScaler`.

    Args:
        transformer: The transformer to apply before standard scaling.

    Returns:
        A Pipeline of `[transformer_name, safe_standard_scaler]`.
    """
    return Pipeline(
        steps=[
            ("input_transformer", transformer),
            ("standard", make_scaler_safe("standard", StandardScaler())),
        ],
    )


class _NoInverseImputer(SimpleImputer):
    """SimpleImputer that returns input unchanged on `inverse_transform`."""

    @override
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X


def _identity(x: np.ndarray) -> np.ndarray:
    """Return input unchanged. Used as a no-op inverse for FunctionTransformer."""
    return x


def _replace_inf_with_nan(x: np.ndarray) -> np.ndarray:
    """Replace +inf and -inf with NaN, leaving existing NaN values unchanged."""
    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)


def _make_finite_steps(suffix: str) -> list[tuple[str, TransformerMixin]]:
    """Create pipeline steps that replace non-finite values and impute NaNs.

    Args:
        suffix: Appended to step names to ensure uniqueness (e.g. "pre", "post").

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
            ),
        ),
    ]
