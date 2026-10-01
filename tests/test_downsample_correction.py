#  Copyright (c) Prior Labs GmbH 2026.
"""Tests for the prior correction applied under `majority_downsample` row sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import tabpfn.inference_tuning as tuning
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.downsample_correction import (
    apply_class_weights,
    context_class_prior,
    downsample_bucket_log_weights,
    downsample_class_weights,
    temper_and_correct_logits,
)
from tabpfn.finetuning.data_util import (
    ClassifierBatch,
    RegressorBatch,
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)
from tabpfn.inference_tuning import (
    RegressorEvalMetrics,
    find_regression_optimal_temperature,
)
from tabpfn.preprocessing import (
    generate_regression_ensemble_configs,
)
from tabpfn.preprocessing.configs import PreprocessorConfig
from tabpfn.preprocessing.datamodel import Feature, FeatureModality
from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
from tabpfn.preprocessing.torch import FeatureSchema
from tabpfn.utils import balance_probas_by_class_counts

# --------------------------------------------------------------------------- #
# Weight math
# --------------------------------------------------------------------------- #


def test__context_class_prior__averages_over_estimators():
    y = np.array([0] * 8 + [1] * 2)
    indices = [np.array([0, 1, 8]), np.array([2, 3, 4, 9])]
    prior = context_class_prior(y, indices, n_classes=2)
    np.testing.assert_allclose(prior, [(2 / 3 + 3 / 4) / 2, (1 / 3 + 1 / 4) / 2])


def test__downsample_class_weights__recover_training_posterior_under_label_shift():
    """For a Bayes-optimal model the correction is exact."""
    rng = np.random.default_rng(0)
    train_counts = np.array([950, 50])
    context_prior = np.array([0.5, 0.5])
    likelihood = rng.random((1000, 2))
    p_context = likelihood * context_prior
    p_context /= p_context.sum(axis=1, keepdims=True)
    p_train = likelihood * (train_counts / train_counts.sum())
    p_train /= p_train.sum(axis=1, keepdims=True)

    weights = downsample_class_weights(train_counts, context_prior)
    corrected = apply_class_weights(torch.tensor(p_context), weights).numpy()
    np.testing.assert_allclose(corrected, p_train, atol=1e-12)


def test__downsample_class_weights__class_missing_from_context_keeps_weight_one():
    weights = downsample_class_weights(np.array([90, 10]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(weights, [0.9, 1.0])


def test__apply_class_weights__renormalizes_and_keeps_binary_ranking():
    rng = np.random.default_rng(1)
    p1 = rng.random(50)
    probas = torch.tensor(np.stack([1 - p1, p1], axis=1))
    out = apply_class_weights(probas, np.array([0.2, 3.0]))
    torch.testing.assert_close(out.sum(dim=1), torch.ones(50, dtype=out.dtype))
    assert np.array_equal(np.argsort(out[:, 1].numpy()), np.argsort(p1))


def test__downsample_bucket_log_weights__spike_bar_versus_rest():
    """With every non-majority row kept and the majority kept at rate q, the
    spike bar gets ratio mean(q)/q and every other bar mean(q)/1.
    """
    bardist = FullSupportBarDistribution(torch.linspace(-1.0, 9.0, 11))
    y = np.concatenate([np.zeros(90), np.linspace(1.0, 8.0, 10)])
    inclusion = np.where(y == 0, 0.5, 1.0)
    log_w = downsample_bucket_log_weights(
        bardist,
        y_raw=y,
        row_inclusion_probabilities=inclusion,
        unobserved_target_probability=1.0,
    )
    spike = int(bardist.map_to_bucket_idx(torch.tensor([0.0])).item())
    mean_inclusion = inclusion.mean()
    np.testing.assert_allclose(log_w[spike].item(), np.log(mean_inclusion / 0.5))
    others = torch.cat([log_w[:spike], log_w[spike + 1 :]])
    np.testing.assert_allclose(others.numpy(), np.log(mean_inclusion), rtol=1e-6)

    # Empty bars use the unobserved-target rate (one here), same as other bars.
    populated = np.bincount(
        bardist.map_to_bucket_idx(torch.tensor(y)).numpy(), minlength=10
    )
    assert (populated == 0).any()

    # Mass moves onto the spike, so a flat prediction's mean drops toward zero.
    uniform = torch.zeros(1, 10)
    assert bardist.mean(uniform + log_w).item() < bardist.mean(uniform).item()


def test__downsample_bucket_log_weights__uniform_inclusion_is_identity():
    bardist = FullSupportBarDistribution(torch.linspace(-1.0, 9.0, 11))
    y = np.linspace(0.0, 8.0, 50)
    log_w = downsample_bucket_log_weights(
        bardist,
        y_raw=y,
        row_inclusion_probabilities=np.full(50, 0.3),
        unobserved_target_probability=0.3,
    )
    assert torch.all(log_w == 0)


def test__downsample_bucket_log_weights__rejects_bad_inputs():
    bardist = FullSupportBarDistribution(torch.linspace(-1.0, 9.0, 11))
    with pytest.raises(ValueError, match="equal length"):
        downsample_bucket_log_weights(
            bardist,
            y_raw=np.zeros(3),
            row_inclusion_probabilities=np.ones(2),
            unobserved_target_probability=1.0,
        )
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        downsample_bucket_log_weights(
            bardist,
            y_raw=np.zeros(3),
            row_inclusion_probabilities=np.array([0.0, 1.0, 1.0]),
            unobserved_target_probability=1.0,
        )


# --------------------------------------------------------------------------- #
# Preprocessor plumbing
# --------------------------------------------------------------------------- #


def _schema(n: int) -> FeatureSchema:
    return FeatureSchema(
        [Feature(name=f"f{i}", modality=FeatureModality.NUMERICAL) for i in range(n)]
    )


def _preprocessor(
    y: np.ndarray, *, method: str, subsample: int | None
) -> TabPFNEnsemblePreprocessor:
    configs = generate_regression_ensemble_configs(
        num_estimators=2,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[PreprocessorConfig("none", categorical_name="numeric")],
        target_transforms=[None],
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )
    return TabPFNEnsemblePreprocessor(
        configs=configs,
        n_samples=len(y),
        feature_schema=_schema(3),
        random_state=0,
        n_preprocessing_jobs=1,
        subsample_samples=subsample,
        sample_subsampling_method=method,  # type: ignore[arg-type]
        y_train=y,
        task_type="regressor",
    )


def test__preprocessor__reports_inclusion_probabilities_under_majority_downsample():
    y = np.concatenate([np.zeros(90), np.arange(1, 11, dtype=float)])
    pre = _preprocessor(y, method="majority_downsample", subsample=40)
    assert pre.downsample_shifted_prior
    probabilities, unobserved = pre.row_sampling_distribution_
    assert unobserved == 1.0
    np.testing.assert_allclose(probabilities[y != 0], 1.0)
    # 40 - 10 kept non-majority rows = 30 of the 90 zeros per estimator.
    np.testing.assert_allclose(probabilities[y == 0], 30 / 90)


@pytest.mark.parametrize("method", ["balanced", "auto"])
def test__preprocessor__no_correction_for_other_samplers(method: str):
    y = np.concatenate([np.zeros(90), np.arange(1, 11, dtype=float)])
    pre = _preprocessor(y, method=method, subsample=40)
    assert not pre.downsample_shifted_prior
    assert pre.row_sampling_distribution_ is None


def test__preprocessor__no_correction_without_subsampling():
    y = np.concatenate([np.zeros(90), np.arange(1, 11, dtype=float)])
    pre = _preprocessor(y, method="majority_downsample", subsample=None)
    assert not pre.downsample_shifted_prior


def test__preprocessor__tied_majority_falls_back_without_correction():
    y = np.concatenate([np.zeros(50), np.ones(50)])
    with pytest.warns(UserWarning, match="unique majority"):
        pre = _preprocessor(y, method="majority_downsample", subsample=40)
    assert not pre.downsample_shifted_prior


# --------------------------------------------------------------------------- #
# Classifier
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


def test__classifier__computes_exact_ratio_weights():
    X, y = _imbalanced_classification()
    clf = _downsampling_classifier().fit(X, y)
    # Minority: 30 of 90 in context vs 30 of 300 in training.
    np.testing.assert_allclose(
        clf.downsample_correction_weights_, [0.9 / (60 / 90), 0.1 / (30 / 90)]
    )


def test__classifier__correction_restores_prior_and_keeps_ranking():
    """Compared with the uncorrected probabilities, the correction is a fixed
    per-class reweighting: same ranking, mean positive rate near the base rate.
    """
    X, y = _imbalanced_classification()
    clf = _downsampling_classifier().fit(X, y)
    corrected = clf.predict_proba(X)
    weights = clf.downsample_correction_weights_
    clf.downsample_correction_weights_ = None
    raw = clf.predict_proba(X)
    np.testing.assert_allclose(
        corrected, apply_class_weights(torch.tensor(raw), weights).numpy(), atol=1e-6
    )
    assert np.array_equal(np.argsort(corrected[:, 1]), np.argsort(raw[:, 1]))
    assert abs(corrected[:, 1].mean() - 0.1) < abs(raw[:, 1].mean() - 0.1)


def test__classifier__no_correction_without_majority_downsample():
    X, y = _imbalanced_classification()
    clf = TabPFNClassifier(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={"SUBSAMPLE_SAMPLES": 90},
    ).fit(X, y)
    assert clf.downsample_correction_weights_ is None
    plain = TabPFNClassifier(n_estimators=2, device="cpu", random_state=0).fit(X, y)
    assert plain.downsample_correction_weights_ is None


def test__classifier__correction_composes_with_balance_probabilities():
    """Balancing acts after the correction, so it divides probabilities that
    already express the training prior by that same prior.
    """
    X, y = _imbalanced_classification()
    balanced = _downsampling_classifier(balance_probabilities=True).fit(X, y)
    corrected = _downsampling_classifier().fit(X, y)
    expected = balance_probas_by_class_counts(
        torch.tensor(corrected.predict_proba(X)), corrected.class_counts_
    ).numpy()
    np.testing.assert_allclose(balanced.predict_proba(X), expected, atol=1e-6)


def test__classifier__temperature_calibration_sees_corrected_probabilities():
    X, y = _imbalanced_classification(n_majority=450, n_minority=50)
    clf = _downsampling_classifier(tuning_config={"calibrate_temperature": True}).fit(
        X, y
    )
    assert clf.downsample_correction_weights_ is not None
    assert np.isfinite(clf.softmax_temperature_)


def test__classifier__tuning_folds_carry_their_own_correction_weights():
    """With an absolute context budget a holdout fold has fewer minority rows
    than the full fit, so its correction weights differ from the parent's. The
    tuning objective must use the fold's weights, not the parent's.
    """
    X, y = _imbalanced_classification(n_majority=450, n_minority=50)
    clf = _downsampling_classifier().fit(X, y)
    _raw_logits, y_true, weights = clf._compute_holdout_validation_data(
        X, y, holdout_frac=0.5, n_folds=1
    )
    assert weights is not None
    assert weights.shape == (len(y_true), 2)
    # One clone per fold, so the weights are constant within the fold ...
    np.testing.assert_allclose(weights, np.broadcast_to(weights[:1], weights.shape))
    # ... and differ from the full model's: the fold keeps ~25 minority rows in a
    # 90-row context, the full fit keeps 50.
    assert not np.allclose(weights[0], clf.downsample_correction_weights_)
    expected_minority = (25 / 250) / (25 / 90)
    np.testing.assert_allclose(weights[0, 1], expected_minority, rtol=0.05)


def test__classifier__tuning_folds_without_downsampling_get_weight_one():
    """A budget that subsamples the full fit but not the smaller tuning folds:
    the folds must be scored uncorrected, not with the parent's weights.
    """
    X, y = _imbalanced_classification(n_majority=450, n_minority=50)
    clf = TabPFNClassifier(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={
            "SUBSAMPLE_SAMPLES": 300,
            "SAMPLE_SUBSAMPLING_METHOD": "majority_downsample",
        },
    ).fit(X, y)
    assert clf.downsample_correction_weights_ is not None
    raw_logits, y_true, weights = clf._compute_holdout_validation_data(
        X, y, holdout_frac=0.5, n_folds=1
    )
    np.testing.assert_array_equal(weights, np.ones((len(y_true), 2)))
    # And the tuning objective sees the uncorrected probabilities.
    scored = clf.logits_to_probabilities(
        raw_logits, downsample_correction_weights=weights
    ).numpy()
    uncorrected = clf.logits_to_probabilities(
        raw_logits, downsample_correction_weights=np.ones(2)
    ).numpy()
    parent_corrected = clf.logits_to_probabilities(raw_logits).numpy()
    np.testing.assert_allclose(scored, uncorrected, atol=1e-6)
    assert not np.allclose(scored, parent_corrected)


def test__classifier__logits_to_probabilities_applies_override_after_averaging():
    """Composition used by tuning: temperature -> softmax -> average -> fold
    weights -> renormalize, with per-sample weights.
    """
    X, y = _imbalanced_classification()
    clf = _downsampling_classifier().fit(X, y)
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(2, 7, 2))
    weights = np.tile(np.array([[1.5, 0.25]]), (7, 1))
    weights[3:] = [0.5, 4.0]
    out = clf.logits_to_probabilities(
        raw,
        softmax_temperature=2.0,
        average_before_softmax=False,
        downsample_correction_weights=weights,
    ).numpy()
    averaged = torch.tensor(raw / 2.0).softmax(-1).mean(0).numpy()
    expected = averaged * weights
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bfloat16])
def test__classifier__differentiable_input_applies_correction(dtype: torch.dtype):
    X, y = _imbalanced_classification()
    reference = _downsampling_classifier().fit(X, y)
    clf = _downsampling_classifier(differentiable_input=True)
    clf.fit_with_differentiable_input(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=dtype)
    )
    np.testing.assert_allclose(
        clf.downsample_correction_weights_, reference.downsample_correction_weights_
    )


def test__classifier__predict_proba_batched_rejects_majority_downsample():
    X, y = _imbalanced_classification()
    clf = _downsampling_classifier()
    with pytest.raises(NotImplementedError, match="majority_downsample"):
        clf.predict_proba_batched([X], [y], [X[:5]])


# --------------------------------------------------------------------------- #
# Regressor
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


def test__regressor__computes_bar_log_weights():
    X, y = _zero_inflated_regression()
    reg = _downsampling_regressor().fit(X, y)
    log_w = reg.downsample_correction_log_weights_
    assert log_w is not None
    assert log_w.shape == (reg.raw_space_bardist_.num_bars,)
    spike = int(reg.raw_space_bardist_.map_to_bucket_idx(torch.tensor([0.0])).item())
    assert log_w.argmax().item() == spike


def test__regressor__correction_lowers_level_toward_truth_and_moves_all_outputs():
    X, y = _zero_inflated_regression()
    reg = _downsampling_regressor().fit(X, y)
    corrected = reg.predict(X, output_type="full")
    reg.downsample_correction_log_weights_ = None
    raw = reg.predict(X, output_type="full")
    assert corrected["mean"].mean() < raw["mean"].mean()
    assert abs(corrected["mean"].mean() - y.mean()) < abs(raw["mean"].mean() - y.mean())
    assert not np.allclose(corrected["median"], raw["median"])
    assert not np.allclose(corrected["quantiles"][0], raw["quantiles"][0])


def test__regressor__no_correction_without_majority_downsample():
    X, y = _zero_inflated_regression()
    reg = TabPFNRegressor(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={"SUBSAMPLE_SAMPLES": 90},
    ).fit(X, y)
    assert reg.downsample_correction_log_weights_ is None


def test__regressor__continuous_target_falls_back_without_correction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = rng.normal(size=200)
    with pytest.warns(UserWarning, match="unique majority"):
        reg = _downsampling_regressor().fit(X, y)
    assert reg.downsample_correction_log_weights_ is None


def test__regressor__differentiable_input_matches_fit():
    X, y = _zero_inflated_regression()
    reference = _downsampling_regressor().fit(X, y)
    reg = _downsampling_regressor(differentiable_input=True)
    reg.fit_with_differentiable_input(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    torch.testing.assert_close(
        reg.downsample_correction_log_weights_,
        reference.downsample_correction_log_weights_,
    )


def test__regressor__predict_batched_rejects_majority_downsample():
    X, y = _zero_inflated_regression()
    reg = _downsampling_regressor()
    with pytest.raises(NotImplementedError, match="majority_downsample"):
        reg.predict_batched([X], [y], [X[:5]])


def test__regressor__prediction_composes_temperature_then_correction():
    """At predict time the aggregated logits are divided by the temperature and
    the correction is added afterwards, not scaled with them.
    """
    X, y = _zero_inflated_regression()
    reg = _downsampling_regressor().fit(X, y)
    log_w = reg.downsample_correction_log_weights_
    assert log_w is not None

    reg.downsample_correction_log_weights_ = None
    reg.ensemble_softmax_temperature_ = 1.0
    uncorrected_t1 = reg.predict(X[:10], output_type="full")["logits"]

    reg.downsample_correction_log_weights_ = log_w
    reg.ensemble_softmax_temperature_ = 2.0
    corrected_t2 = reg.predict(X[:10], output_type="full")["logits"]

    expected = temper_and_correct_logits(
        uncorrected_t1, temperature=2.0, log_weights=log_w
    )
    finite = torch.isfinite(expected) & torch.isfinite(corrected_t2)
    assert finite.float().mean() > 0.9
    torch.testing.assert_close(
        corrected_t2[finite], expected[finite], atol=1e-4, rtol=1e-4
    )


def test__regressor__tuning_folds_keep_logits_and_correction_separate():
    X, y = _zero_inflated_regression(n_zeros=450, n_nonzero=50)
    reg = _downsampling_regressor().fit(X, y)
    folds = reg._compute_holdout_validation_data(X, y, holdout_frac=0.5, n_folds=1)
    assert len(folds) == 1
    _logits, bardist, _y_true, log_w = folds[0]
    assert log_w is not None
    assert log_w.shape == (bardist.num_bars,)
    # The fold's clone trains on 250 rows with ~25 nonzero, so its inclusion
    # rate for zeros (65/225) differs from the full fit's (40/450): the fold
    # carries its own correction rather than the parent's.
    assert not torch.allclose(log_w, reg.downsample_correction_log_weights_)


def test__find_regression_optimal_temperature__composes_like_prediction(
    monkeypatch: pytest.MonkeyPatch,
):
    """The sweep must score `logits / T + log_w`, not `(logits + log_w) / T`.
    Intercept what the sweep hands to the metric and compare it, for every
    candidate temperature, with the predict-time composition.
    """
    bardist = FullSupportBarDistribution(torch.linspace(-1.0, 9.0, 11))
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(50, 10)), dtype=torch.float32)
    y_true = torch.tensor(rng.uniform(0.0, 8.0, size=50), dtype=torch.float32)
    log_w = torch.tensor(rng.normal(size=10), dtype=torch.float32)

    received: list[torch.Tensor] = []
    original = tuning.compute_regression_metric_to_minimize

    def spy(**kwargs):  # noqa: ANN202
        received.append(kwargs["logits"].clone())
        return original(**kwargs)

    monkeypatch.setattr(tuning, "compute_regression_metric_to_minimize", spy)
    find_regression_optimal_temperature(
        holdout_folds=[(logits, bardist, y_true, log_w)],
        metric_name=RegressorEvalMetrics.NLL,
        current_default_temperature=1.0,
    )

    temperatures = tuning.get_tuning_temperatures()
    assert len(received) == len(temperatures)
    for temperature, seen in zip(temperatures, received, strict=True):
        expected = temper_and_correct_logits(
            logits, temperature=float(temperature), log_weights=log_w
        )
        torch.testing.assert_close(seen, expected)
        # The wrong composition would scale the correction with the temperature.
        wrong = (logits + log_w) / float(temperature)
        assert temperature == 1.0 or not torch.allclose(seen, wrong)


def test__regressor__temperature_calibration_runs_with_correction():
    X, y = _zero_inflated_regression(n_zeros=450, n_nonzero=50)
    reg = _downsampling_regressor(tuning_config={"calibrate_temperature": True}).fit(
        X, y
    )
    assert reg.downsample_correction_log_weights_ is not None
    assert reg.ensemble_softmax_temperature_ > 0


# --------------------------------------------------------------------------- #
# Finetuning path: fit_from_preprocessed must not inherit a correction
# --------------------------------------------------------------------------- #


def _finetuning_batch(
    estimator: TabPFNClassifier | TabPFNRegressor,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
) -> object:
    chunks = get_preprocessed_dataset_chunks(
        estimator,
        X,
        y,
        train_test_split,
        100,
        model_type=model_type,
        equal_split_size=True,
        data_shuffle_seed=42,
        preprocessing_random_state=42,
    )
    return next(
        iter(DataLoader(chunks, batch_size=1, collate_fn=meta_dataset_collator))
    )


def test__classifier__fit_from_preprocessed_clears_stale_correction():
    """fit() under majority_downsample stores weights; a later finetuning batch
    has no prior shift, so forward() must match a never-corrected estimator.
    """
    X, y = _imbalanced_classification(n_majority=180, n_minority=20)
    stale = _downsampling_classifier().fit(X, y)
    assert stale.downsample_correction_weights_ is not None
    fresh = _downsampling_classifier()

    batch = _finetuning_batch(stale, X, y, "classifier")
    assert isinstance(batch, ClassifierBatch)
    cat_indices = batch.cat_indices
    outputs = []
    for clf in (stale, fresh):
        clf.fit_from_preprocessed(
            batch.X_context,
            batch.y_context,
            cat_indices,
            batch.configs,
            performance_options=PerformanceOptions(),
        )
        assert clf.downsample_correction_weights_ is None
        outputs.append(clf.forward(batch.X_query).detach())
    torch.testing.assert_close(outputs[0], outputs[1])


def test__regressor__fit_from_preprocessed_clears_stale_correction():
    X, y = _zero_inflated_regression(n_zeros=180, n_nonzero=20)
    stale = _downsampling_regressor().fit(X, y)
    assert stale.downsample_correction_log_weights_ is not None
    fresh = _downsampling_regressor()

    batch = _finetuning_batch(stale, X, y, "regressor")
    assert isinstance(batch, RegressorBatch)
    outputs = []
    for reg in (stale, fresh):
        reg.fit_from_preprocessed(
            batch.X_context,
            batch.y_context,
            batch.cat_indices,
            batch.configs,
            performance_options=PerformanceOptions(),
        )
        # The finetuning loop hands the batch's bar distributions and target frame
        # to the estimator.
        reg.znorm_space_bardist_ = batch.znorm_space_bardist
        reg.raw_space_bardist_ = batch.raw_space_bardist
        reg.y_train_mean_ = batch.y_train_mean
        reg.y_train_std_ = batch.y_train_std
        assert reg.downsample_correction_log_weights_ is None
        out = reg.forward(batch.X_query)
        outputs.append((out[0] if isinstance(out, tuple) else out).detach())
    torch.testing.assert_close(outputs[0], outputs[1])
