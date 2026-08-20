#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the regression target transform pipelines."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from tabpfn.preprocessing.steps import (
    get_all_reshape_feature_distribution_preprocessors,
)
from tabpfn.preprocessing.target_transform import (
    UNSTANDARDIZE_STEP,
    UnstandardizeTarget,
    rebind_target_transform_statistics,
    robust_target_scale,
    wrap_target_transform,
)

TRANSFORM_NAMES = ["1_plus_log", "log", "safepower", "quantile_norm", "robust", "none"]


def _skewed_positive_target(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.exp(rng.normal(5.0, 1.0, size=n))


def _znorm(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean, std = float(np.mean(y)), float(np.std(y)) + 1e-20
    return (y - mean) / std, mean, std


def _get_transform(name: str, n: int):  # noqa: ANN202
    return get_all_reshape_feature_distribution_preprocessors(
        num_examples=n, random_state=0
    )[name]


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test__wrap_target_transform__standardizes_the_transformed_raw_target(
    name: str,
) -> None:
    """The wrapped transform acts on the raw target and standardizes the result.

    This is the point of the wrapping: fed the z-normalised target, the
    pipeline must produce exactly what applying the transform to the target in
    its original units and standardizing afterwards would.
    """
    y = _skewed_positive_target()
    y_znorm, mean, std = _znorm(y)

    wrapped = wrap_target_transform(
        _get_transform(name, len(y)), mean=mean, std=std
    ).fit_transform(y_znorm.reshape(-1, 1))

    unwrapped = (
        _get_transform(name, len(y)).fit_transform(y.reshape(-1, 1)).astype(float)
    )
    expected = (unwrapped - unwrapped.mean()) / unwrapped.std()

    np.testing.assert_allclose(wrapped, expected, atol=1e-10)


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test__wrap_target_transform__inverse_transform_returns_znormalized_target(
    name: str,
) -> None:
    """``inverse_transform`` must land back in the z-normalised space.

    The regressor maps the model's bar-distribution borders through it, and
    those borders -- as well as the sanity limits applied to them -- live in
    the z-normalised space.
    """
    y = _skewed_positive_target()
    y_znorm, mean, std = _znorm(y)

    pipeline = wrap_target_transform(_get_transform(name, len(y)), mean=mean, std=std)
    transformed = pipeline.fit_transform(y_znorm.reshape(-1, 1))

    np.testing.assert_allclose(
        pipeline.inverse_transform(transformed).ravel(), y_znorm, atol=1e-8
    )


def test__wrap_target_transform__none_transform_is_the_identity() -> None:
    """Wrapping the identity transform leaves the z-normalised target alone.

    Keeps the ``None`` entry of ``REGRESSION_Y_PREPROCESS_TRANSFORMS`` and the
    ``"none"`` preset equivalent, so the wrapping cannot silently rescale the
    target of an estimator that is not supposed to transform it.
    """
    y = _skewed_positive_target()
    y_znorm, mean, std = _znorm(y)

    pipeline = wrap_target_transform(_get_transform("none", len(y)), mean=mean, std=std)

    np.testing.assert_allclose(
        pipeline.fit_transform(y_znorm.reshape(-1, 1)).ravel(), y_znorm, atol=1e-10
    )
    borders = np.linspace(-5.0, 5.0, 21)
    np.testing.assert_allclose(
        pipeline.inverse_transform(borders.reshape(-1, 1)).ravel(), borders, atol=1e-10
    )


def test__wrap_target_transform__is_picklable() -> None:
    """Fitted configs are pickled for joblib workers and for `save_fit_state`."""
    y = _skewed_positive_target()
    y_znorm, mean, std = _znorm(y)

    pipeline = wrap_target_transform(
        _get_transform("1_plus_log", len(y)), mean=mean, std=std
    )
    pipeline.fit(y_znorm.reshape(-1, 1))
    restored = pickle.loads(pickle.dumps(pipeline))  # noqa: S301

    np.testing.assert_allclose(
        restored.transform(y_znorm.reshape(-1, 1)),
        pipeline.transform(y_znorm.reshape(-1, 1)),
    )


def test__unstandardize_target__transform_and_inverse_are_inverses() -> None:
    x = np.array([[-1.0], [0.0], [2.5]])
    step = UnstandardizeTarget(mean=3.0, std=2.0).fit(x)

    assert step.scale_ == 1.0
    np.testing.assert_allclose(step.transform(x), x * 2.0 + 3.0)
    np.testing.assert_allclose(step.inverse_transform(step.transform(x)), x)


def test__unstandardize_target__scale_normalized__divides_by_the_robust_scale() -> None:
    """Scale-normalised, the step returns the target divided by its own scale."""
    x = np.array([[-1.0], [0.0], [2.5]])
    step = UnstandardizeTarget(mean=3.0, std=2.0, scale_normalized=True).fit(x)

    scale = robust_target_scale(x * 2.0 + 3.0)
    assert step.scale_ == scale
    np.testing.assert_allclose(step.transform(x), (x * 2.0 + 3.0) / scale)
    np.testing.assert_allclose(step.inverse_transform(step.transform(x)), x)


def test__robust_target_scale__falls_back_for_a_mostly_zero_target() -> None:
    """A zero-inflated target has a zero median, which cannot be a divisor."""
    y = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 6.0])

    assert robust_target_scale(y) == pytest.approx(np.mean(np.abs(y)))
    assert robust_target_scale(np.zeros(4)) == 1.0


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test__wrap_target_transform__scale_normalized__standardizes_scaled_target(
    name: str,
) -> None:
    """Scale-normalised, the transform sees the target divided by its scale.

    The zero point of the target is preserved, only its scale is normalised.
    """
    y = _skewed_positive_target()
    y_znorm, mean, std = _znorm(y)

    wrapped = wrap_target_transform(
        _get_transform(name, len(y)), mean=mean, std=std, scale_normalized=True
    ).fit_transform(y_znorm.reshape(-1, 1))

    scaled = y / robust_target_scale(y)
    unwrapped = (
        _get_transform(name, len(y)).fit_transform(scaled.reshape(-1, 1)).astype(float)
    )
    expected = (unwrapped - unwrapped.mean()) / unwrapped.std()

    np.testing.assert_allclose(wrapped, expected, atol=1e-10)


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test__wrap_target_transform__scale_normalized__inverse_returns_znormalized(
    name: str,
) -> None:
    """The z-normalised space of the borders must survive the scale normalisation."""
    y = _skewed_positive_target()
    y_znorm, mean, std = _znorm(y)

    pipeline = wrap_target_transform(
        _get_transform(name, len(y)), mean=mean, std=std, scale_normalized=True
    )
    transformed = pipeline.fit_transform(y_znorm.reshape(-1, 1))

    np.testing.assert_allclose(
        pipeline.inverse_transform(transformed).ravel(), y_znorm, atol=1e-8
    )


def test__wrap_target_transform__scale_normalized__is_invariant_to_target_units() -> (
    None
):
    """Scale normalisation is the point: the units of the target stop mattering.

    Without it, Yeo-Johnson on a target far from unit scale degenerates, since
    its bend sits at zero with a fixed unit offset.
    """
    y = _skewed_positive_target()

    def model_target(target: np.ndarray, *, scale_normalized: bool) -> np.ndarray:
        y_znorm, mean, std = _znorm(target)
        pipeline = wrap_target_transform(
            _get_transform("safepower", len(target)),
            mean=mean,
            std=std,
            scale_normalized=scale_normalized,
        )
        return pipeline.fit_transform(y_znorm.reshape(-1, 1)).ravel()

    np.testing.assert_allclose(
        model_target(y, scale_normalized=True),
        model_target(y * 1e-6, scale_normalized=True),
        atol=1e-6,
    )
    # The unnormalised composition is not invariant, which is what motivates
    # the scale normalisation in the first place.
    assert not np.allclose(
        model_target(y, scale_normalized=False),
        model_target(y * 1e-6, scale_normalized=False),
        atol=1e-6,
    )


def test__rebind_target_transform_statistics__updates_wrapped_transforms() -> None:
    """The fine-tuning pipeline re-normalises per split and rebinds the stats."""
    y = _skewed_positive_target()
    pipeline = wrap_target_transform(_get_transform("log", len(y)), mean=0.0, std=1.0)

    rebind_target_transform_statistics([None, pipeline], mean=7.0, std=3.0)

    step = pipeline.named_steps[UNSTANDARDIZE_STEP]
    assert (step.mean, step.std) == (7.0, 3.0)


def test__rebind_target_transform_statistics__ignores_other_pipelines() -> None:
    """Pipelines that do not come from `wrap_target_transform` are left alone."""
    other = Pipeline(steps=[("some_step", _get_transform("safepower", 100))])

    rebind_target_transform_statistics([other], mean=7.0, std=3.0)

    assert UNSTANDARDIZE_STEP not in other.named_steps
