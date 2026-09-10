#  Copyright (c) Prior Labs GmbH 2026.
"""Tests for `prediction_scaling`: the factor sources and the estimator wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.prediction_scaling import (
    PredictionScalingMode,
    apply_class_weights,
    balanced_class_weights,
    context_class_prior,
    fit_holdout_class_weights,
    fit_holdout_level_scaling,
    majority_value_shares,
    resolve_prediction_scaling,
    sampler_bucket_log_weights,
    sampler_class_weights,
)
from tabpfn.utils import balance_probas_by_class_counts

# --------------------------------------------------------------------------- #
# Factor sources
# --------------------------------------------------------------------------- #


def test__resolve_prediction_scaling__auto_follows_sampler_shift():
    assert (
        resolve_prediction_scaling(
            "auto", task_type="classifier", sampler_shifted_prior=True
        )
        == PredictionScalingMode.SAMPLER
    )
    assert (
        resolve_prediction_scaling(
            "auto", task_type="regressor", sampler_shifted_prior=False
        )
        == PredictionScalingMode.NONE
    )


def test__resolve_prediction_scaling__balanced_rejected_for_regressor():
    with pytest.raises(ValueError, match="only defined for classification"):
        resolve_prediction_scaling(
            "balanced", task_type="regressor", sampler_shifted_prior=False
        )


def test__context_class_prior__averages_over_estimators():
    y = np.array([0] * 8 + [1] * 2)
    indices = [np.array([0, 1, 8]), np.array([2, 3, 4, 9])]
    prior = context_class_prior(y, indices, n_classes=2)
    np.testing.assert_allclose(prior, [(2 / 3 + 3 / 4) / 2, (1 / 3 + 1 / 4) / 2])
    np.testing.assert_allclose(context_class_prior(y, None, 2), [0.8, 0.2])


def test__sampler_class_weights__restore_training_prior_under_label_shift():
    """Probabilities calibrated to a shifted context are moved back to the
    training prior, and the correction is exact for a Bayes-optimal model.
    """
    rng = np.random.default_rng(0)
    train_counts = np.array([950, 50])
    context_prior = np.array([0.5, 0.5])
    # Class-conditional likelihoods for a batch of rows.
    likelihood = rng.random((1000, 2))
    p_context = likelihood * context_prior
    p_context /= p_context.sum(axis=1, keepdims=True)
    p_train = likelihood * (train_counts / train_counts.sum())
    p_train /= p_train.sum(axis=1, keepdims=True)

    weights = sampler_class_weights(train_counts, context_prior)
    corrected = apply_class_weights(torch.tensor(p_context), weights).numpy()
    np.testing.assert_allclose(corrected, p_train, atol=1e-12)


def test__sampler_class_weights__class_missing_from_context_keeps_weight_one():
    weights = sampler_class_weights(np.array([90, 10]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(weights, [0.9, 1.0])


def test__balanced_class_weights__match_legacy_balancing_without_subsampling():
    counts = np.array([700, 200, 100])
    probas = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    legacy = balance_probas_by_class_counts(probas, counts)
    weighted = apply_class_weights(probas, balanced_class_weights(counts, None))
    torch.testing.assert_close(weighted, legacy)


def test__balanced_class_weights__use_context_prior_when_shifted():
    weights = balanced_class_weights(np.array([900, 100]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(weights, [2.0, 2.0])


def test__apply_class_weights__renormalizes_and_keeps_binary_ranking():
    rng = np.random.default_rng(1)
    p1 = rng.random(50)
    probas = torch.tensor(np.stack([1 - p1, p1], axis=1))
    out = apply_class_weights(probas, np.array([0.2, 3.0]))
    torch.testing.assert_close(out.sum(dim=1), torch.ones(50, dtype=out.dtype))
    assert np.array_equal(np.argsort(out[:, 1].numpy()), np.argsort(p1))


def test__fit_holdout_class_weights__matches_mean_prediction_to_frequency():
    rng = np.random.default_rng(2)
    y = np.array([0] * 900 + [1] * 100)
    # Over-confident in the minority: mean predicted positive rate ~0.4.
    logits = rng.normal(size=(1000, 2))
    logits[:, 1] += 0.5
    probas = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    weights = fit_holdout_class_weights(probas, y, n_classes=2)
    scaled = probas * weights
    scaled /= scaled.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(scaled.mean(axis=0), [0.9, 0.1], atol=1e-6)


def test__majority_value_shares__zero_inflated_target():
    y = np.array([0.0] * 90 + list(range(1, 11)))
    indices = [np.array(list(range(10)) + list(range(90, 100)))]
    value, train_share, context_share = majority_value_shares(y, indices)
    assert value == 0.0
    assert train_share == 0.9
    assert context_share == 0.5
    assert majority_value_shares(y, None)[2] == 0.9


def test__sampler_bucket_log_weights__reweights_spike_bucket_only():
    bardist = FullSupportBarDistribution(torch.linspace(-1.0, 9.0, 11))
    log_w = sampler_bucket_log_weights(
        bardist, majority_value=0.0, train_share=0.9, context_share=0.5
    )
    spike = int(bardist.map_to_bucket_idx(torch.tensor([0.0])).item())
    assert log_w.shape == (10,)
    np.testing.assert_allclose(log_w[spike].item(), np.log(0.9 / 0.5))
    others = torch.cat([log_w[:spike], log_w[spike + 1 :]])
    np.testing.assert_allclose(others.numpy(), np.log(0.1 / 0.5))

    # A uniform prediction moves most of its mass onto the spike bucket, so
    # the predicted mean drops toward the majority value.
    uniform = torch.zeros(1, 10)
    assert bardist.mean(uniform + log_w).item() < bardist.mean(uniform).item()


def test__sampler_bucket_log_weights__rejects_degenerate_shares():
    bardist = FullSupportBarDistribution(torch.linspace(-1.0, 9.0, 11))
    with pytest.raises(ValueError, match="cover part of both"):
        sampler_bucket_log_weights(
            bardist, majority_value=0.0, train_share=0.9, context_share=1.0
        )


def test__fit_holdout_level_scaling__multiplicative_then_additive():
    assert fit_holdout_level_scaling(np.array([2.0, 4.0]), np.array([1.0, 2.0])) == (
        0.5,
        0.0,
    )
    scale, shift = fit_holdout_level_scaling(
        np.array([-1.0, 1.0]), np.array([1.0, 3.0])
    )
    assert (scale, shift) == (1.0, 2.0)


# --------------------------------------------------------------------------- #
# Classifier wiring
# --------------------------------------------------------------------------- #


def _imbalanced_classification(
    seed: int = 0, n_majority: int = 270, n_minority: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_majority + n_minority, 4))
    y = np.array([0] * n_majority + [1] * n_minority)
    X[y == 1] += 1.0
    return X, y


def _downsampling_classifier(**kwargs) -> TabPFNClassifier:
    return TabPFNClassifier(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={
            "SUBSAMPLE_SAMPLES": 90,
            "SAMPLE_SUBSAMPLING_METHOD": "majority_downsample",
        },
        **kwargs,
    )


def test__classifier__auto_resolves_to_sampler_under_majority_downsample():
    X, y = _imbalanced_classification()
    clf = _downsampling_classifier().fit(X, y)
    assert clf.prediction_scaling_ == PredictionScalingMode.SAMPLER
    assert clf.prediction_scaling_weights_ is not None
    # Minority rows are all in context (30 of 90), so its weight is its
    # training share over its context share: 0.1 / (1/3).
    np.testing.assert_allclose(clf.prediction_scaling_weights_[1], 0.1 / (30 / 90))
    np.testing.assert_allclose(clf.prediction_scaling_weights_[0], 0.9 / (60 / 90))


def test__classifier__sampler_scaling_restores_prior_and_keeps_ranking():
    X, y = _imbalanced_classification()
    scaled = _downsampling_classifier().fit(X, y).predict_proba(X)
    raw = _downsampling_classifier(prediction_scaling="none").fit(X, y).predict_proba(X)
    # Same ranking of the positive class, lower mean positive probability.
    assert np.array_equal(np.argsort(scaled[:, 1]), np.argsort(raw[:, 1]))
    assert scaled[:, 1].mean() < raw[:, 1].mean()
    assert abs(scaled[:, 1].mean() - 0.1) < abs(raw[:, 1].mean() - 0.1)


def test__classifier__auto_is_none_without_prior_shift():
    X, y = _imbalanced_classification()
    clf = TabPFNClassifier(n_estimators=2, device="cpu", random_state=0).fit(X, y)
    assert clf.prediction_scaling_ == PredictionScalingMode.NONE
    assert clf.prediction_scaling_weights_ is None


def test__classifier__predictions_are_batch_independent():
    X, y = _imbalanced_classification()
    clf = _downsampling_classifier().fit(X, y)
    full = clf.predict_proba(X[:20])
    single = clf.predict_proba(X[:1])
    np.testing.assert_allclose(single, full[:1], atol=1e-6)


def test__classifier__balance_probabilities_is_deprecated_alias():
    X, y = _imbalanced_classification()
    with pytest.warns(DeprecationWarning, match="prediction_scaling='balanced'"):
        legacy = TabPFNClassifier(
            n_estimators=2, device="cpu", random_state=0, balance_probabilities=True
        ).fit(X, y)
    new = TabPFNClassifier(
        n_estimators=2, device="cpu", random_state=0, prediction_scaling="balanced"
    ).fit(X, y)
    assert legacy.prediction_scaling_ == PredictionScalingMode.BALANCED
    np.testing.assert_allclose(legacy.predict_proba(X), new.predict_proba(X), atol=1e-6)


def test__classifier__balance_probabilities_conflicts_with_other_mode():
    X, y = _imbalanced_classification()
    clf = TabPFNClassifier(
        n_estimators=2,
        device="cpu",
        random_state=0,
        balance_probabilities=True,
        prediction_scaling="sampler",
    )
    with pytest.raises(ValueError, match="conflicts with"):
        clf.fit(X, y)


def test__classifier__balanced_matches_legacy_output_without_subsampling():
    X, y = _imbalanced_classification()
    clf = TabPFNClassifier(
        n_estimators=2, device="cpu", random_state=0, prediction_scaling="balanced"
    ).fit(X, y)
    raw = TabPFNClassifier(n_estimators=2, device="cpu", random_state=0).fit(X, y)
    expected = balance_probas_by_class_counts(
        torch.tensor(raw.predict_proba(X)), clf.class_counts_
    ).numpy()
    np.testing.assert_allclose(clf.predict_proba(X), expected, atol=1e-6)


def test__classifier__holdout_mode_fits_weights_without_tuning_config():
    X, y = _imbalanced_classification(n_majority=450, n_minority=50)
    clf = _downsampling_classifier(prediction_scaling="holdout").fit(X, y)
    assert clf.prediction_scaling_ == PredictionScalingMode.HOLDOUT
    assert clf.prediction_scaling_weights_ is not None
    assert clf.prediction_scaling_weights_.shape == (2,)
    # The holdout weights push the mean positive probability toward the base
    # rate; without them the downsampled context predicts far too many positives.
    raw = _downsampling_classifier(prediction_scaling="none").fit(X, y)
    assert abs(clf.predict_proba(X)[:, 1].mean() - 0.1) < abs(
        raw.predict_proba(X)[:, 1].mean() - 0.1
    )


def test__classifier__holdout_rejected_with_differentiable_input():
    X, y = _imbalanced_classification()
    clf = TabPFNClassifier(
        n_estimators=2,
        device="cpu",
        random_state=0,
        differentiable_input=True,
        prediction_scaling="holdout",
    )
    with pytest.raises(ValueError, match="not supported with"):
        clf.fit_with_differentiable_input(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y)
        )


def test__classifier__differentiable_input_applies_sampler_scaling():
    X, y = _imbalanced_classification()
    clf = _downsampling_classifier(differentiable_input=True)
    clf.fit_with_differentiable_input(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    )
    assert clf.prediction_scaling_ == PredictionScalingMode.SAMPLER
    assert clf.prediction_scaling_weights_ is not None


@pytest.mark.parametrize("mode", ["balanced", "holdout"])
def test__classifier__predict_proba_batched_rejects_active_scaling(mode: str):
    X, y = _imbalanced_classification()
    clf = TabPFNClassifier(
        n_estimators=2, device="cpu", random_state=0, prediction_scaling=mode
    )
    with pytest.raises(NotImplementedError, match="prediction_scaling"):
        clf.predict_proba_batched([X], [y], [X[:5]])


def test__classifier__predict_proba_batched_allows_inactive_auto():
    X, y = _imbalanced_classification()
    clf = TabPFNClassifier(n_estimators=2, device="cpu", random_state=0)
    out = clf.predict_proba_batched([X], [y], [X[:5]])
    assert len(out) == 1
    assert out[0].shape[0] == 5


# --------------------------------------------------------------------------- #
# Regressor wiring
# --------------------------------------------------------------------------- #


def _zero_inflated_regression(
    seed: int = 0, n_zeros: int = 270, n_nonzero: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_zeros + n_nonzero, 4))
    y = np.concatenate([np.zeros(n_zeros), rng.exponential(size=n_nonzero) + 0.5])
    X[y > 0] += 1.0
    return X, y


def _downsampling_regressor(**kwargs) -> TabPFNRegressor:
    return TabPFNRegressor(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={
            "SUBSAMPLE_SAMPLES": 90,
            "SAMPLE_SUBSAMPLING_METHOD": "majority_downsample",
        },
        **kwargs,
    )


def test__regressor__auto_resolves_to_sampler_under_majority_downsample():
    X, y = _zero_inflated_regression()
    reg = _downsampling_regressor().fit(X, y)
    assert reg.prediction_scaling_ == PredictionScalingMode.SAMPLER
    assert reg.prediction_scaling_log_weights_ is not None
    assert reg.prediction_scaling_log_weights_.shape == (
        reg.raw_space_bardist_.num_bars,
    )
    assert reg.prediction_scaling_affine_ == (1.0, 0.0)


def test__regressor__sampler_scaling_lowers_predicted_level():
    X, y = _zero_inflated_regression()
    scaled = _downsampling_regressor().fit(X, y).predict(X)
    raw = _downsampling_regressor(prediction_scaling="none").fit(X, y).predict(X)
    assert scaled.mean() < raw.mean()
    assert abs(scaled.mean() - y.mean()) < abs(raw.mean() - y.mean())


def test__regressor__sampler_scaling_applies_to_every_output_type():
    X, y = _zero_inflated_regression()
    scaled = _downsampling_regressor().fit(X, y)
    raw = _downsampling_regressor(prediction_scaling="none").fit(X, y)
    s_full = scaled.predict(X[:10], output_type="full")
    r_full = raw.predict(X[:10], output_type="full")
    assert not np.allclose(s_full["median"], r_full["median"])
    assert not np.allclose(s_full["quantiles"][0], r_full["quantiles"][0])


def test__regressor__auto_is_none_without_prior_shift():
    X, y = _zero_inflated_regression()
    reg = TabPFNRegressor(n_estimators=2, device="cpu", random_state=0).fit(X, y)
    assert reg.prediction_scaling_ == PredictionScalingMode.NONE
    assert reg.prediction_scaling_log_weights_ is None


def test__regressor__predictions_are_batch_independent():
    X, y = _zero_inflated_regression()
    reg = _downsampling_regressor().fit(X, y)
    np.testing.assert_allclose(reg.predict(X[:1]), reg.predict(X[:20])[:1], atol=1e-5)


def test__regressor__balanced_rejected():
    X, y = _zero_inflated_regression()
    with pytest.raises(ValueError, match="only defined for classification"):
        TabPFNRegressor(
            n_estimators=2, device="cpu", random_state=0, prediction_scaling="balanced"
        ).fit(X, y)


def test__regressor__holdout_mode_fits_affine_map_and_rescales_borders():
    X, y = _zero_inflated_regression(n_zeros=450, n_nonzero=50)
    reg = _downsampling_regressor(prediction_scaling="holdout").fit(X, y)
    assert reg.prediction_scaling_ == PredictionScalingMode.HOLDOUT
    scale, shift = reg.prediction_scaling_affine_
    assert (scale, shift) != (1.0, 0.0)
    assert reg.prediction_scaling_log_weights_ is None
    # The raw-space borders carry the affine map, so every output type follows.
    expected = (
        reg.znorm_space_bardist_.borders * reg.y_train_std_ + reg.y_train_mean_
    ) * scale + shift
    torch.testing.assert_close(
        reg.raw_space_bardist_.borders, expected.float(), atol=1e-4, rtol=1e-5
    )
    raw = _downsampling_regressor(prediction_scaling="none").fit(X, y)
    assert abs(reg.predict(X).mean() - y.mean()) < abs(raw.predict(X).mean() - y.mean())


def test__regressor__holdout_rejected_with_differentiable_input():
    X, y = _zero_inflated_regression()
    reg = TabPFNRegressor(
        n_estimators=2,
        device="cpu",
        random_state=0,
        differentiable_input=True,
        prediction_scaling="holdout",
    )
    with pytest.raises(ValueError, match="not supported with"):
        reg.fit_with_differentiable_input(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )


def test__regressor__predict_batched_rejects_active_scaling():
    X, y = _zero_inflated_regression()
    reg = _downsampling_regressor()
    with pytest.raises(NotImplementedError, match="prediction_scaling"):
        reg.predict_batched([X], [y], [X[:5]])


def test__regressor__predict_batched_allows_inactive_auto():
    X, y = _zero_inflated_regression()
    reg = TabPFNRegressor(n_estimators=2, device="cpu", random_state=0)
    out = reg.predict_batched([X], [y], [X[:5]])
    assert len(out) == 1
    assert len(out[0]) == 5
