#  Copyright (c) Prior Labs GmbH 2026.

"""End-to-end tests for TabPFNClassifier.predict_proba_from_shared_pool."""

from __future__ import annotations

import numpy as np
import pytest

from tabpfn import TabPFNClassifier
from tabpfn.model_loading import resolve_model_path
from tabpfn.shared_pool import SharedPoolError


def _v3_classifier_path() -> str:
    """Locate the cached v3 classifier checkpoint, skipping if it is absent.

    Resolved by path rather than by the version alias so the test never tries to
    re-authenticate against the model host.
    """
    paths, _, _, _ = resolve_model_path(None, "classifier", "v3")
    path = paths[0]
    if not path.exists():
        pytest.skip(f"v3 classifier checkpoint not cached at {path}")
    return str(path)


@pytest.fixture(scope="module")
def fitted() -> tuple[TabPFNClassifier, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_pool, n_test, n_features = 60, 8, 5
    x_pool = rng.normal(size=(n_pool, n_features))
    y_pool = (x_pool[:, 0] + rng.normal(scale=0.2, size=n_pool) > 0).astype(int)
    x_test = rng.normal(size=(n_test, n_features))

    clf = TabPFNClassifier(
        n_estimators=2, model_path=_v3_classifier_path(), random_state=0
    )
    clf.fit(x_pool, y_pool)
    return clf, x_test, y_pool


def test__whole_pool_as_context_reproduces_predict_proba(fitted) -> None:
    """Giving every test row the whole pool must match the standard predict path.

    With the context equal to the pool, pool-fit and context-fit statistics coincide,
    so both paths compute the same thing. This is the end-to-end check on the
    plumbing: pool membership, GPU-pipeline fitting, class permutation, and the
    softmax/averaging order.

    Not bit-exact — the standard path pushes pool and test rows through stages 0-2 in
    one pass while the pooled path embeds them separately, so reductions run in a
    different order.
    """
    clf, x_test, y_pool = fitted
    n_pool, n_test = len(y_pool), len(x_test)

    expected = clf.predict_proba(x_test)
    whole_pool = np.tile(np.arange(n_pool), (n_test, 1))
    got = clf.predict_proba_from_shared_pool(x_test, whole_pool)

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=5e-3, atol=5e-3)


def test__probabilities_are_normalised_and_finite(fitted) -> None:
    clf, x_test, y_pool = fitted
    rng = np.random.default_rng(1)
    contexts = np.stack(
        [rng.choice(len(y_pool), size=12, replace=False) for _ in range(len(x_test))]
    )

    got = clf.predict_proba_from_shared_pool(x_test, contexts)

    assert got.shape == (len(x_test), len(clf.classes_))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test__result_does_not_depend_on_chunk_size(fitted) -> None:
    """Nothing in the pooled path depends on which contexts share a forward."""
    clf, x_test, y_pool = fitted
    rng = np.random.default_rng(2)
    contexts = np.stack(
        [rng.choice(len(y_pool), size=16, replace=False) for _ in range(len(x_test))]
    )

    small = clf.predict_proba_from_shared_pool(x_test, contexts, chunk_size=2)
    large = clf.predict_proba_from_shared_pool(x_test, contexts, chunk_size=1000)

    np.testing.assert_allclose(small, large, rtol=1e-5, atol=1e-6)


def test__context_missing_a_class_still_scores_the_full_class_set(fitted) -> None:
    """``classes_`` comes from the pool, not from each context.

    This is the failure mode #1045 has: with independently-fitted datasets a context
    holding only one class would score against a single-class output. Here the pool
    defines the class set, so the shape is stable and the absent class simply gets
    little mass.
    """
    clf, x_test, y_pool = fitted
    single_class = np.flatnonzero(y_pool == y_pool[0])[:10]
    contexts = np.tile(single_class, (len(x_test), 1))

    got = clf.predict_proba_from_shared_pool(x_test, contexts)

    assert got.shape == (len(x_test), len(clf.classes_))
    assert np.isfinite(got).all()


def test__per_estimator_contexts_are_accepted(fitted) -> None:
    clf, x_test, y_pool = fitted
    n_members = len(clf.executor_.ensemble_members)
    rng = np.random.default_rng(3)
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

    got = clf.predict_proba_from_shared_pool(x_test, per_estimator)

    assert got.shape == (len(x_test), len(clf.classes_))
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test__broadcast_form_matches_the_explicit_per_estimator_form(fitted) -> None:
    clf, x_test, y_pool = fitted
    n_members = len(clf.executor_.ensemble_members)
    rng = np.random.default_rng(4)
    shared = np.stack(
        [rng.choice(len(y_pool), size=12, replace=False) for _ in range(len(x_test))]
    )
    repeated = np.repeat(shared[None], n_members, axis=0)

    np.testing.assert_allclose(
        clf.predict_proba_from_shared_pool(x_test, shared),
        clf.predict_proba_from_shared_pool(x_test, repeated),
        rtol=0,
        atol=0,
    )


def test__out_of_range_indices_are_rejected(fitted) -> None:
    clf, x_test, y_pool = fitted
    bad = np.full((len(x_test), 4), len(y_pool), dtype=int)
    with pytest.raises(SharedPoolError, match="must index the pool"):
        clf.predict_proba_from_shared_pool(x_test, bad)


def test__subsample_samples_is_rejected(fitted) -> None:
    """Per-estimator row subsampling makes a pool index ambiguous, so refuse it."""
    _, x_test, y_pool = fitted
    rng = np.random.default_rng(5)
    x_pool = rng.normal(size=(len(y_pool), 5))

    clf = TabPFNClassifier(
        n_estimators=2,
        model_path=_v3_classifier_path(),
        random_state=0,
        inference_config={"SUBSAMPLE_SAMPLES": 30},
    )
    clf.fit(x_pool, y_pool)

    with pytest.raises(SharedPoolError, match="SUBSAMPLE_SAMPLES is not supported"):
        clf.predict_proba_from_shared_pool(
            x_test, np.zeros((len(x_test), 4), dtype=int)
        )
