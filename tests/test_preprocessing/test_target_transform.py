#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the regression target pipelines."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from tabpfn.preprocessing.steps import (
    get_all_reshape_feature_distribution_preprocessors,
)
from tabpfn.preprocessing.target_transform import (
    STANDARDIZE_STEP,
    TARGET_TRANSFORM_STEP,
    StandardizeTarget,
    make_target_transform,
)
from tabpfn.utils import transform_borders_one

TRANSFORM_NAMES = ["1_plus_log", "safepower", "quantile_norm", "robust", "none"]


def _target(n: int = 200) -> np.ndarray:
    """A strictly positive, right-skewed target."""
    rng = np.random.default_rng(0)
    return np.exp(rng.normal(5.0, 1.0, size=n))


def _get_transform(name: str, n: int):  # noqa: ANN202
    return get_all_reshape_feature_distribution_preprocessors(
        num_examples=n, random_state=0
    )[name]


def test__standardize_target__matches_the_estimators_own_znormalisation() -> None:
    """Bit-for-bit, so a member without a target transform is unaffected.

    The regressor used to z-normalise the target itself with exactly this
    arithmetic before handing it to the preprocessing.
    """
    y = _target()

    standardized = StandardizeTarget().fit_transform(y.reshape(-1, 1)).ravel()

    expected = (y - np.mean(y)) / (np.std(y) + StandardizeTarget.EPSILON)
    assert np.array_equal(standardized, expected)


def test__standardize_target__inverse_transform_round_trips() -> None:
    y = _target()
    step = StandardizeTarget().fit(y.reshape(-1, 1))

    np.testing.assert_allclose(
        step.inverse_transform(step.transform(y.reshape(-1, 1))).ravel(), y, rtol=1e-12
    )


def test__standardize_target__constant_target_does_not_divide_by_zero() -> None:
    """`fit` rejects a constant target, but the epsilon must hold regardless."""
    standardized = StandardizeTarget().fit_transform(np.full((10, 1), 3.0))

    assert np.isfinite(standardized).all()


def test__make_target_transform__without_a_transform_only_standardizes() -> None:
    y = _target()
    pipeline = make_target_transform(None)

    assert list(pipeline.named_steps) == [STANDARDIZE_STEP]
    expected = (y - np.mean(y)) / (np.std(y) + StandardizeTarget.EPSILON)
    assert np.array_equal(pipeline.fit_transform(y.reshape(-1, 1)).ravel(), expected)


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test__make_target_transform__transform_sees_the_standardized_target(
    name: str,
) -> None:
    """The preset still reshapes the standardized target.

    This is what RES-2639 goes on to change; pinning it here keeps the move of
    the standardization to a separate, reviewable step.
    """
    y = _target()
    transform = _get_transform(name, len(y))

    got = make_target_transform(transform).fit_transform(y.reshape(-1, 1)).astype(float)

    standardized = (y - np.mean(y)) / (np.std(y) + StandardizeTarget.EPSILON)
    expected = _get_transform(name, len(y)).fit_transform(standardized.reshape(-1, 1))
    np.testing.assert_allclose(got, np.asarray(expected, dtype=float), rtol=1e-12)


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test__make_target_transform__inverse_transform_returns_original_units(
    name: str,
) -> None:
    """The contract the regressor relies on to map the model's borders back.

    The pipeline knows nothing about the frame the ensemble is aggregated in;
    it returns the target's own units and the estimator takes it from there.
    """
    y = _target()
    pipeline = make_target_transform(_get_transform(name, len(y)))
    transformed = pipeline.fit_transform(y.reshape(-1, 1))

    np.testing.assert_allclose(
        pipeline.inverse_transform(transformed).ravel(), y, rtol=1e-6
    )


def test__make_target_transform__is_picklable() -> None:
    """Fitted configs are pickled for joblib workers and for `save_fit_state`."""
    y = _target()
    pipeline = make_target_transform(_get_transform("1_plus_log", len(y)))
    pipeline.fit(y.reshape(-1, 1))

    restored = pickle.loads(pickle.dumps(pipeline))  # noqa: S301

    np.testing.assert_array_equal(
        restored.transform(y.reshape(-1, 1)), pipeline.transform(y.reshape(-1, 1))
    )
    assert TARGET_TRANSFORM_STEP in restored.named_steps


def test__transform_borders_one__maps_borders_into_the_znorm_frame() -> None:
    """Without a preset the borders must come back where they started.

    `transform_borders_one` undoes the member's standardization and then maps
    into the frame the ensemble is aggregated in; for a member that only
    standardizes, those two steps cancel.
    """
    y = _target()
    pipeline = make_target_transform(None)
    pipeline.fit(y.reshape(-1, 1))
    borders = np.linspace(-5.0, 5.0, 101, dtype=np.float32)

    _, descending, borders_t = transform_borders_one(
        borders,
        target_transform=pipeline,
        znorm_mean=float(np.mean(y)),
        znorm_std=float(np.std(y)) + StandardizeTarget.EPSILON,
        repair_nan_borders_after_transform=True,
    )

    assert not descending
    np.testing.assert_allclose(borders_t, borders, atol=1e-6)


def test__transform_borders_one__guard_is_applied_in_the_znorm_frame() -> None:
    """The sanity limits mean standard deviations, not units of the target.

    A large-magnitude target must not have every one of its borders rejected,
    which is what an absolute limit in the target's own units would do.
    """
    y = _target() * 1e6
    pipeline = make_target_transform(None)
    pipeline.fit(y.reshape(-1, 1))
    borders = np.linspace(-5.0, 5.0, 101, dtype=np.float32)

    logit_cancel_mask, _, borders_t = transform_borders_one(
        borders,
        target_transform=pipeline,
        znorm_mean=float(np.mean(y)),
        znorm_std=float(np.std(y)) + StandardizeTarget.EPSILON,
        repair_nan_borders_after_transform=True,
    )

    assert logit_cancel_mask is None
    np.testing.assert_allclose(borders_t, borders, atol=1e-6)
