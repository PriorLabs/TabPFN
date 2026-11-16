#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the regression target pipelines."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.preprocessing import FunctionTransformer

from tabpfn.preprocessing.steps import (
    get_all_reshape_feature_distribution_preprocessors,
)
from tabpfn.preprocessing.target_transform import (
    STANDARDIZE_STEP,
    TARGET_TRANSFORM_STEP,
    StandardizeTarget,
    make_target_transform,
)
from tabpfn.regressor import _unflip_member
from tabpfn.utils import transform_borders_one, translate_probs_across_borders

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
    """Preserve the exact arithmetic of global target standardization."""
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
def test__make_target_transform__transform_sees_the_unnormalized_target(
    name: str,
) -> None:
    """The preset reshapes the raw target before standardization."""
    y = _target()
    transform = _get_transform(name, len(y))

    got = make_target_transform(transform).fit_transform(y.reshape(-1, 1)).astype(float)

    reshaped = np.asarray(
        _get_transform(name, len(y)).fit_transform(y.reshape(-1, 1)), dtype=float
    )
    expected = (reshaped - reshaped.mean()) / (
        reshaped.std() + StandardizeTarget.EPSILON
    )
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test__make_target_transform__log_of_the_target_is_defined() -> None:
    """Positive raw targets stay finite even when their z-scores fall below -1."""
    rng = np.random.default_rng(0)
    y = rng.normal(100.0, 10.0, size=500)  # symmetric, strictly positive

    standardized = (y - np.mean(y)) / np.std(y)
    assert not np.isfinite(np.log1p(standardized)).all()

    got = (
        make_target_transform(_get_transform("1_plus_log", len(y)))
        .fit_transform(y.reshape(-1, 1))
        .ravel()
    )

    assert np.isfinite(got).all()
    expected = np.log1p(y)
    np.testing.assert_allclose(
        got, (expected - expected.mean()) / expected.std(), rtol=1e-10
    )


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test__make_target_transform__inverse_transform_returns_original_units(
    name: str,
) -> None:
    """The inverse returns raw units, independently of the aggregation frame."""
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
    """Standardization and the aggregation-frame mapping cancel without a preset."""
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
    """Border sanity limits apply in standard deviations, not raw target units."""
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


def _odd_inverse(sign: float) -> FunctionTransformer:
    """An odd map from model space, increasing for sign=1, that blows up past +-2."""

    def inverse(z: np.ndarray) -> np.ndarray:
        return sign * np.sinh(4.0 * z)

    return FunctionTransformer(inverse_func=inverse, check_inverse=False)


def test__transform_borders_one__reversing_transform_with_broken_ends() -> None:
    """A reversing transform whose outer borders pass the guard limits.

    It must give the borders and the cancelled buckets of the increasing transform
    on the mirrored grid, reversed.
    """
    borders = np.linspace(-3.0, 3.0, 61)
    kwargs = {"znorm_mean": 0.0, "znorm_std": 1.0}

    mask, descending, borders_t = transform_borders_one(
        borders,
        target_transform=_odd_inverse(-1.0),
        repair_nan_borders_after_transform=True,
        **kwargs,
    )
    ref_mask, ref_descending, ref_borders = transform_borders_one(
        -borders[::-1],
        target_transform=_odd_inverse(1.0),
        repair_nan_borders_after_transform=True,
        **kwargs,
    )

    assert descending
    assert not ref_descending
    assert mask is not None
    assert mask.any()
    # Ascending; the cancelled buckets collapse to zero width.
    assert (np.diff(borders_t) >= 0).all()
    np.testing.assert_array_equal(mask, ref_mask)
    np.testing.assert_allclose(borders_t, ref_borders)


def test__unflip_member__mirrors_the_member_distribution() -> None:
    """A member that saw the negated target predicts the mirrored distribution.

    On a symmetric target grid its bucket masses are the plain member's reversed.
    The model's borders are asymmetric, so mirroring only the logits, and not the
    borders, would fail.
    """
    rng = np.random.default_rng(0)
    std_borders = np.sort(rng.normal(size=41)) + 0.3
    logits = torch.as_tensor(rng.normal(size=(3, 40)))
    target_grid = torch.linspace(-4.0, 4.0, 81, dtype=torch.float64)
    pipeline = make_target_transform(None).fit(np.array([[-1.0], [1.0]]))
    frame = {"znorm_mean": 0.0, "znorm_std": 1.0}

    def member_probs(flip: bool) -> torch.Tensor:
        config = SimpleNamespace(flip_target=flip)
        out, borders = _unflip_member(logits, std_borders, config)  # type: ignore[arg-type]
        mask, descending, borders_t = transform_borders_one(
            borders,
            target_transform=pipeline,
            repair_nan_borders_after_transform=True,
            **frame,
        )
        assert mask is None
        assert not descending
        return translate_probs_across_borders(
            out, frm=torch.as_tensor(borders_t), to=target_grid
        )

    torch.testing.assert_close(
        member_probs(flip=True), member_probs(flip=False).flip(-1)
    )
