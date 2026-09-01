#  Copyright (c) Prior Labs GmbH 2026.

"""End-to-end tests for TabPFNRegressor.predict_from_shared_pool."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn import TabPFNRegressor
from tabpfn.model_loading import resolve_model_path
from tabpfn.shared_pool import SharedPoolError


def _v3_regressor_path() -> str:
    """Locate the cached v3 regressor checkpoint, skipping if it is absent.

    Resolved by path rather than by the ``"v3"`` alias so the test never tries to
    re-authenticate against the model host.
    """
    paths, _, _, _ = resolve_model_path(None, "regressor", "v3")
    path = paths[0]
    if not path.exists():
        pytest.skip(f"v3 regressor checkpoint not cached at {path}")
    return str(path)


@pytest.fixture(scope="module")
def fitted() -> tuple[TabPFNRegressor, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_pool, n_test, n_features = 60, 8, 5
    x_pool = rng.normal(size=(n_pool, n_features))
    y_pool = x_pool[:, 0] * 2.0 + rng.normal(scale=0.1, size=n_pool)
    x_test = rng.normal(size=(n_test, n_features))

    reg = TabPFNRegressor(
        n_estimators=2, model_path=_v3_regressor_path(), random_state=0
    )
    reg.fit(x_pool, y_pool)
    return reg, x_test, y_pool


def test__whole_pool_as_context_reproduces_predict(fitted) -> None:
    """Giving every test row the whole pool must match the standard predict path.

    With the context equal to the pool, pool-fit statistics and context-fit
    statistics coincide, so the two paths are computing the same thing. This is the
    end-to-end check that the plumbing — which rows land in the pool, that the GPU
    pipeline is fitted on them, target alignment, and the decode — is correct.

    Not bit-exact: the standard path pushes pool and test rows through stages 0-2 in
    one pass while the pooled path embeds them separately, so the reductions run in a
    different order. Measured deviation is ~1e-3 under autocast.
    """
    reg, x_test, y_pool = fitted
    n_pool, n_test = len(y_pool), len(x_test)

    expected = reg.predict(x_test)
    whole_pool = np.tile(np.arange(n_pool), (n_test, 1))
    got = reg.predict_from_shared_pool(x_test, whole_pool)

    np.testing.assert_allclose(got, expected, rtol=5e-3, atol=5e-3)


def test__result_does_not_depend_on_chunk_size(fitted) -> None:
    """Nothing in the pooled path depends on which contexts share a forward."""
    reg, x_test, y_pool = fitted
    rng = np.random.default_rng(1)
    contexts = np.stack(
        [rng.choice(len(y_pool), size=16, replace=False) for _ in range(len(x_test))]
    )

    small = reg.predict_from_shared_pool(x_test, contexts, chunk_size=2)
    large = reg.predict_from_shared_pool(x_test, contexts, chunk_size=1000)

    np.testing.assert_allclose(small, large, rtol=1e-5, atol=1e-6)


def test__per_row_contexts_produce_finite_predictions(fitted) -> None:
    reg, x_test, y_pool = fitted
    rng = np.random.default_rng(2)
    contexts = np.stack(
        [rng.choice(len(y_pool), size=12, replace=False) for _ in range(len(x_test))]
    )

    got = reg.predict_from_shared_pool(x_test, contexts)

    assert got.shape == (len(x_test),)
    assert np.isfinite(got).all()


def test__quantile_output_is_supported(fitted) -> None:
    reg, x_test, y_pool = fitted
    rng = np.random.default_rng(3)
    contexts = np.stack(
        [rng.choice(len(y_pool), size=12, replace=False) for _ in range(len(x_test))]
    )

    got = reg.predict_from_shared_pool(
        x_test, contexts, output_type="quantiles", quantiles=[0.1, 0.9]
    )

    assert len(got) == 2
    assert all(np.isfinite(q).all() for q in got)
    assert (got[0] <= got[1] + 1e-6).all()


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (np.zeros((3, 4), dtype=int), "covers 3 test rows"),
        (np.zeros(8, dtype=int), r"must be \(n_test, k\)"),
    ],
)
def test__invalid_context_indices_are_rejected(fitted, bad, match) -> None:
    reg, x_test, _ = fitted
    with pytest.raises(SharedPoolError, match=match):
        reg.predict_from_shared_pool(x_test, bad)


def test__out_of_range_indices_are_rejected(fitted) -> None:
    reg, x_test, y_pool = fitted
    bad = np.full((len(x_test), 4), len(y_pool), dtype=int)
    with pytest.raises(SharedPoolError, match="must index the pool"):
        reg.predict_from_shared_pool(x_test, bad)


def test__per_estimator_contexts_are_accepted(fitted) -> None:
    """Each ensemble member may get its own context for each test row.

    This is the general form, and the replacement for per-estimator row
    subsampling: the classic subsample is the special case where all of a member's
    test rows share one context.
    """
    reg, x_test, y_pool = fitted
    n_members = len(reg.executor_.ensemble_members)
    rng = np.random.default_rng(4)
    per_estimator = np.stack(
        [
            np.stack(
                [
                    rng.choice(len(y_pool), size=12, replace=False)
                    for _ in range(len(x_test))
                ]
            )
            for _ in range(n_members)
        ]
    )
    assert per_estimator.shape == (n_members, len(x_test), 12)

    got = reg.predict_from_shared_pool(x_test, per_estimator)

    assert got.shape == (len(x_test),)
    assert np.isfinite(got).all()


def test__broadcast_form_matches_the_explicit_per_estimator_form(fitted) -> None:
    """(n_test, k) must equal (n_estimators, n_test, k) with the rows repeated."""
    reg, x_test, y_pool = fitted
    n_members = len(reg.executor_.ensemble_members)
    rng = np.random.default_rng(5)
    shared = np.stack(
        [rng.choice(len(y_pool), size=12, replace=False) for _ in range(len(x_test))]
    )
    repeated = np.repeat(shared[None], n_members, axis=0)

    np.testing.assert_allclose(
        reg.predict_from_shared_pool(x_test, shared),
        reg.predict_from_shared_pool(x_test, repeated),
        rtol=0,
        atol=0,
    )


def test__wrong_estimator_count_is_rejected(fitted) -> None:
    reg, x_test, _y_pool = fitted
    bad = np.zeros((len(reg.executor_.ensemble_members) + 1, len(x_test), 4), dtype=int)
    with pytest.raises(SharedPoolError, match="estimator slots"):
        reg.predict_from_shared_pool(x_test, bad)


def test__subsample_samples_is_rejected(fitted) -> None:
    """Per-estimator row subsampling makes a pool index ambiguous, so refuse it.

    Each member would hold a different subset, so the same index names a different
    row in every member and the row the caller meant is never read. Nothing
    downstream can detect that, which is why it fails here rather than silently.
    """
    _, x_test, y_pool = fitted
    rng = np.random.default_rng(6)
    x_pool = rng.normal(size=(len(y_pool), 5))

    reg = TabPFNRegressor(
        n_estimators=2,
        model_path=_v3_regressor_path(),
        random_state=0,
        inference_config={"SUBSAMPLE_SAMPLES": 30},
    )
    reg.fit(x_pool, y_pool)
    contexts = np.zeros((len(x_test), 4), dtype=int)

    with pytest.raises(SharedPoolError, match="SUBSAMPLE_SAMPLES is not supported"):
        reg.predict_from_shared_pool(x_test, contexts)
