#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import copy
import warnings
from typing import Literal

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_tweedie_deviance

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.inference_tuning import (
    MEAN_CALIBRATION_MIN_MEAN_FRACTION,
    MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING,
    ClassifierEvalMetrics,
    ClassifierTuningConfig,
    RegressorEvalMetrics,
    RegressorTuningConfig,
    compute_regression_metric_to_minimize,
    find_mean_calibration_scale,
    find_optimal_classification_threshold_single_class,
    find_optimal_classification_thresholds,
    find_optimal_temperature,
    find_regression_optimal_temperature,
    get_tuning_feature_flags,
    get_tuning_split_indices,
    get_tuning_temperatures,
    resolve_tuning_config,
    select_robust_optimal_threshold,
)
from tabpfn.regression_metrics import (
    ranked_probability_score_loss_from_bar_logits,
)


@pytest.mark.parametrize(
    ("y_true", "y_pred_probs", "expected_interval"),
    [
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), (0.3, 0.7)),
        (np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.6]), (0.3, 0.7)),
    ],
)
def test__find_optimal_classification_threshold_single_class__threshold_in_interval(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    expected_interval: tuple[float, float],
) -> None:
    best_threshold = find_optimal_classification_threshold_single_class(
        metric_name=ClassifierEvalMetrics.F1,
        y_true=y_true,
        y_pred_probas=y_pred_probs,
    )
    lo, hi = expected_interval
    assert lo <= best_threshold <= hi


@pytest.mark.parametrize(
    ("thresholds_and_losses", "expected_threshold", "plateau_delta"),
    [
        ([(1, 0.4), (2, 0.3), (3, 0.301), (4, 0.3015), (5, 0.6)], 3.0, 0.0018),
        ([(1, 0.2), (2, 0.1), (3, 0.101), (4, 0.1015), (5, 0.05)], 5.0, 0.002),
        ([(1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2), (5, 0.2)], 3.0, 0.2),
        ([(1, 0.1), (2, 0.5), (3, 0.6), (4, 0.7), (5, 0.8)], 1.0, 0.001),
        ([(1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5), (5, 0.1)], 5.0, 0.001),
        ([(1, 0.3), (2, 0.1), (3, 0.11), (4, 0.5)], 2.0, 0.005),
        ([(1, 0.2), (2, 0.2), (3, 0.6), (4, 0.7)], 2.0, 0.0001),
        ([(1, 0.1), (2, 0.11), (3, 0.12), (4, 0.11), (5, 0.1)], 1.0, 0.002),
        ([(1, 0.1), (2, 0.11), (3, 0.12), (4, 0.11), (5, 0.1)], 3.0, 0.5),
        ([(1, 0.3), (2, 0.2), (3, 0.21), (4, 0.22), (5, 0.23)], 2.0, 0.01),
        ([(1, 0.5), (2, 0.3), (3, 0.2), (4, 0.21), (5, 0.21)], 4.0, 0.01),
        ([(1, 0.4), (2, 0.4), (3, 0.1), (4, 0.4), (5, 0.4)], 3.0, 0.001),
        ([(1, 0.1), (2, 0.101), (3, 0.102)], 2.0, 0.002),
    ],
)
def test__select_robust_optimal_threshold__works_as_expected(
    thresholds_and_losses: list[tuple[float, float]],
    expected_threshold: float,
    plateau_delta: float,
) -> None:
    assert (
        select_robust_optimal_threshold(
            thresholds_and_losses=thresholds_and_losses,
            plateau_delta=plateau_delta,
        )
        == expected_threshold
    )


@pytest.mark.parametrize(
    (
        "y_true",
        "y_pred_probas",
        "expected_thresholds",
    ),
    [
        (
            np.array([0, 1, 2, 0, 1, 2]),
            np.array(
                [
                    [0.9, 0.05, 0.05],
                    [0.05, 0.9, 0.05],
                    [0.05, 0.05, 0.9],
                    [0.9, 0.05, 0.05],
                    [0.05, 0.9, 0.05],
                    [0.05, 0.05, 0.9],
                ]
            ),
            [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95)],
        ),
        (
            np.array([0, 0, 0, 1, 1, 1, 2, 2]),
            np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.85, 0.08, 0.07],
                    [0.75, 0.15, 0.1],
                    [0.15, 0.7, 0.15],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.75, 0.05],
                    [0.1, 0.1, 0.8],
                    [0.05, 0.15, 0.8],
                ]
            ),
            [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95)],
        ),
        (
            np.array([0, 0, 1, 1, 2, 2]),
            np.array(
                [
                    [0.9, 0.05, 0.05],
                    [0.70, 0.25, 0.05],
                    [0.3, 0.6, 0.1],
                    [0.25, 0.65, 0.1],
                    [0.2, 0.1, 0.7],
                    [0.15, 0.15, 0.7],
                ]
            ),
            [(0.45, 0.95), (0.4, 0.75), (0.3, 0.8)],
        ),
        (
            np.array([0, 0, 0, 1, 1, 2]),
            np.array(
                [
                    [0.95, 0.03, 0.02],
                    [0.9, 0.05, 0.05],
                    [0.88, 0.07, 0.05],
                    [0.4, 0.55, 0.05],
                    [0.35, 0.6, 0.05],
                    [0.1, 0.1, 0.8],
                ]
            ),
            [(0.6, 0.95), (0.1, 0.5), (0.05, 0.95)],
        ),
    ],
)
def test__find_optimal_classification_thresholds__works_for_multiclass_f1(
    y_true: np.ndarray,
    y_pred_probas: np.ndarray,
    expected_thresholds: list[tuple[float, float]],
) -> None:
    thresholds = find_optimal_classification_thresholds(
        metric_name=ClassifierEvalMetrics.F1,
        y_true=y_true,
        y_pred_probas=y_pred_probas,
        n_classes=len(expected_thresholds),
    )

    assert thresholds.shape == (len(expected_thresholds),)
    for i, (lo, hi) in enumerate(expected_thresholds):
        assert lo <= thresholds[i] <= hi, (
            f"Threshold for class {i} is {thresholds[i]}, "
            f"expected to be in [{lo}, {hi}]"
        )


# Class 3 never appears, as happens when a rare class receives no holdout rows.
_Y_TRUE_ABSENT_CLASS = np.array([0, 0, 1, 1, 2, 2])
_PROBAS_ABSENT_CLASS = np.array(
    [
        [0.7, 0.1, 0.1, 0.1],
        [0.6, 0.2, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.6, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.2, 0.6, 0.1],
    ]
)


def test__find_optimal_classification_thresholds__no_warning_when_all_tunable() -> None:
    """Every class present in the holdout means nothing to warn about."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        find_optimal_classification_thresholds(
            metric_name=ClassifierEvalMetrics.ACCURACY,
            y_true=y_true,
            y_pred_probas=_PROBAS_ABSENT_CLASS[:, :3],
            n_classes=3,
        )


@pytest.mark.parametrize("metric_name", list(ClassifierEvalMetrics))
def test__find_optimal_classification_thresholds__absent_class_is_neutral(
    metric_name: ClassifierEvalMetrics,
) -> None:
    """A class with no holdout rows gets the geometric mean, not a searched value."""
    with pytest.warns(UserWarning, match=r"1 of 4 classes have no rows"):
        thresholds = find_optimal_classification_thresholds(
            metric_name=metric_name,
            y_true=_Y_TRUE_ABSENT_CLASS,
            y_pred_probas=_PROBAS_ABSENT_CLASS,
            n_classes=4,
        )

    assert thresholds.shape == (4,)
    assert np.isfinite(thresholds).all()
    # Pin the tuned classes to a direct search, so an implementation that also
    # skips a tunable class cannot pass by making the fill self-consistent.
    for i in range(3):
        assert thresholds[i] == find_optimal_classification_threshold_single_class(
            metric_name=metric_name,
            y_true=(i == _Y_TRUE_ABSENT_CLASS).astype(int),
            y_pred_probas=_PROBAS_ABSENT_CLASS[:, i],
        )
    geometric_mean = float(np.exp(np.mean(np.log(thresholds[:3]))))
    assert thresholds[3] == pytest.approx(geometric_mean)


def test__find_optimal_classification_thresholds__single_class_holdout_is_a_noop() -> (
    None
):
    """With no tunable class at all the thresholds must not change any prediction."""
    y_true = np.zeros(4, dtype=int)
    y_pred_probas = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])

    thresholds = find_optimal_classification_thresholds(
        metric_name=ClassifierEvalMetrics.F1,
        y_true=y_true,
        y_pred_probas=y_pred_probas,
        n_classes=2,
    )

    # Uniform thresholds cancel in the renormalisation the estimator applies.
    assert np.allclose(thresholds, thresholds[0])
    reweighted = y_pred_probas / thresholds
    reweighted = reweighted / reweighted.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(reweighted, y_pred_probas)


def test__select_robust_optimal_threshold__all_non_finite_losses_returns_midpoint() -> (
    None
):
    """An all-nan array makes argmin return 0, the most aggressive threshold."""
    thresholds_and_losses = [
        (float(t), float("nan")) for t in np.linspace(0.01, 0.99, 99)
    ]

    chosen = select_robust_optimal_threshold(
        thresholds_and_losses=thresholds_and_losses
    )

    assert chosen == pytest.approx(0.5)


def test__select_robust_optimal_threshold__ignores_non_finite_losses() -> None:
    """A nan must never win the minimum over a finite loss."""
    thresholds_and_losses = [(0.1, float("nan")), (0.5, -1.0), (0.9, 0.0)]

    chosen = select_robust_optimal_threshold(
        thresholds_and_losses=thresholds_and_losses
    )

    assert chosen == pytest.approx(0.5)


def test__select_robust_optimal_threshold__nan_loses_to_small_positive_losses() -> None:
    """A nan must lose even when every finite loss is positive.

    log_loss is the one metric whose losses are positive, so an internal fill
    value such as 0.0 would win its argmin outright. The finite losses here sit
    more than plateau_delta above 0.0 to pin that the fill can neither win the
    minimum nor join the plateau.
    """
    thresholds_and_losses = [(0.1, float("nan")), (0.5, 0.01), (0.9, 0.02)]

    chosen = select_robust_optimal_threshold(
        thresholds_and_losses=thresholds_and_losses
    )

    assert chosen == pytest.approx(0.5)


@pytest.mark.parametrize(
    (
        "X_train_shape",
        "tune_decision_thresholds",
        "calibrate_temperature",
        "expected_tuning_holdout_pct",
        "expected_tuning_holdout_n_splits",
    ),
    [
        ((1_000, 10), False, True, 0.1, 10),
        ((9_000, 10), False, True, 0.2, 3),
        ((9_000, 10), True, False, 0.2, 3),
        ((20_000, 10), True, False, 0.2, 2),
        ((21_000, 10), True, False, 0.3, 1),
    ],
)
def test__resolve_tuning_config__provides_expected_values_for_auto_config(
    X_train_shape: tuple[int, int],
    calibrate_temperature: bool,
    tune_decision_thresholds: bool,
    expected_tuning_holdout_pct: float,
    expected_tuning_holdout_n_splits: int,
) -> None:
    tuning_config = ClassifierTuningConfig(
        calibrate_temperature=calibrate_temperature,
        tune_decision_thresholds=tune_decision_thresholds,
        tuning_holdout_frac="auto",
        tuning_n_folds="auto",
    )
    resolved_tuning_config = resolve_tuning_config(
        tuning_config=tuning_config,
        num_samples=X_train_shape[0],
    )
    assert isinstance(resolved_tuning_config, ClassifierTuningConfig)

    assert resolved_tuning_config is not None
    assert resolved_tuning_config.calibrate_temperature == calibrate_temperature
    assert resolved_tuning_config.tune_decision_thresholds == tune_decision_thresholds
    assert resolved_tuning_config.tuning_holdout_frac == expected_tuning_holdout_pct
    assert resolved_tuning_config.tuning_n_folds == expected_tuning_holdout_n_splits


def _softmax_over_estimators(
    raw_logits: np.ndarray,
    softmax_temperature: float,
) -> np.ndarray:
    """Stands in for `TabPFNClassifier.logits_to_probabilities`."""
    scaled = raw_logits / softmax_temperature
    exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    probas = exp / exp.sum(axis=-1, keepdims=True)
    return probas.mean(axis=0)


def test__find_optimal_temperature__works_when_class_missing_from_holdout() -> None:
    rng = np.random.default_rng(0)
    n_estimators, n_samples, n_classes = 2, 50, 3
    raw_logits = rng.normal(size=(n_estimators, n_samples, n_classes))
    # Class 2 never appears in the holdout labels.
    y_true = rng.integers(0, 2, size=n_samples)

    temperature = find_optimal_temperature(
        raw_logits=raw_logits,
        y_true=y_true,
        logits_to_probabilities_fn=_softmax_over_estimators,
        current_default_temperature=1.0,
    )

    assert 0.6 <= temperature <= 1.4


def test__find_optimal_temperature__follows_the_weighted_rows() -> None:
    rng = np.random.default_rng(0)
    n_samples = 400
    y_true = rng.integers(0, 2, size=n_samples)
    signed = np.where(y_true == 1, 1.0, -1.0)
    # The first half is confidently right, the second half confidently wrong.
    signed[n_samples // 2 :] *= -1
    raw_logits = np.stack([np.zeros(n_samples), 2.0 * signed], axis=-1)[None]
    first_half = (np.arange(n_samples) < n_samples // 2).astype(float)

    def temperature(sample_weight: np.ndarray | None) -> float:
        return find_optimal_temperature(
            raw_logits=raw_logits,
            y_true=y_true,
            logits_to_probabilities_fn=_softmax_over_estimators,
            current_default_temperature=1.0,
            sample_weight=sample_weight,
        )

    temperatures = get_tuning_temperatures()
    assert temperature(first_half) == temperatures.min()
    assert temperature(1 - first_half) == temperatures.max()
    assert temperature(np.zeros(n_samples)) == 1.0


def test__find_optimal_classification_thresholds__follows_the_weighted_rows() -> None:
    rng = np.random.default_rng(0)
    n_samples = 400
    y_true = rng.integers(0, 2, size=n_samples)
    low = np.arange(n_samples) < n_samples // 2
    # Class 1 separates at 0.2 in the first half and at 0.8 in the second.
    centre = np.where(low, 0.2, 0.8)
    proba_1 = centre + np.where(y_true == 1, 0.1, -0.1)
    y_pred_probas = np.stack([1 - proba_1, proba_1], axis=-1)

    def class_1_threshold(sample_weight: np.ndarray) -> float:
        return find_optimal_classification_thresholds(
            metric_name=ClassifierEvalMetrics.ACCURACY,
            y_true=y_true,
            y_pred_probas=y_pred_probas,
            n_classes=2,
            sample_weight=sample_weight,
        )[1]

    assert 0.1 < class_1_threshold(low.astype(float)) < 0.3
    assert 0.7 < class_1_threshold((~low).astype(float)) < 0.9


def test__find_optimal_classification_thresholds__zero_weight_class_is_neutral() -> (
    None
):
    y_true = np.array([0, 0, 1, 1, 2, 2] * 10)
    y_pred_probas = np.random.default_rng(0).dirichlet(np.ones(3), size=len(y_true))
    sample_weight = (y_true != 2).astype(float)

    with pytest.warns(UserWarning, match="1 of 3 classes"):
        thresholds = find_optimal_classification_thresholds(
            metric_name=ClassifierEvalMetrics.F1,
            y_true=y_true,
            y_pred_probas=y_pred_probas,
            n_classes=3,
            sample_weight=sample_weight,
        )

    assert thresholds[2] == pytest.approx(np.sqrt(thresholds[0] * thresholds[1]))


def test__find_optimal_classification_threshold_single_class__all_negative() -> None:
    # One-vs-rest labels are all-negative when the class is absent from the holdout.
    y_true = np.zeros(20, dtype=int)
    y_pred_probas = np.linspace(0.05, 0.95, 20)

    threshold = find_optimal_classification_threshold_single_class(
        metric_name=ClassifierEvalMetrics.LOG_LOSS,
        y_true=y_true,
        y_pred_probas=y_pred_probas,
    )

    assert 0.0 < threshold < 1.0


@pytest.mark.parametrize("task_type", ["classifier", "regressor"])
def test__get_tuning_split_indices__accepts_numpy_generator_random_state(
    task_type: Literal["classifier", "regressor"],
) -> None:
    X = np.zeros((50, 2))
    y = np.array([0, 1] * 25)

    splits = get_tuning_split_indices(
        X=X,
        y=y,
        holdout_frac=0.2,
        n_splits=1,
        random_state=np.random.default_rng(0),
        task_type=task_type,
    )

    assert len(splits) == 1
    train_indices, holdout_indices = splits[0]
    assert len(train_indices) + len(holdout_indices) == len(X)


def test__get_tuning_split_indices__regression_accepts_continuous_target() -> None:
    # StratifiedKFold rejects a continuous target, which is exactly why regression
    # needs its own splitter. Assert both halves so the parameter is shown to matter.
    X = np.zeros((50, 2))
    y_continuous = np.random.default_rng(0).normal(size=50)

    with pytest.raises(ValueError, match="Supported target types"):
        get_tuning_split_indices(
            X=X,
            y=y_continuous,
            holdout_frac=0.2,
            n_splits=1,
            task_type="classifier",
        )

    splits = get_tuning_split_indices(
        X=X,
        y=y_continuous,
        holdout_frac=0.2,
        n_splits=1,
        task_type="regressor",
    )

    assert len(splits) == 1
    train_indices, holdout_indices = splits[0]
    assert len(train_indices) + len(holdout_indices) == len(X)


@pytest.mark.parametrize(
    ("holdout_frac", "n_splits", "expected_holdout_size"),
    [
        (0.2, 5, 40),
        (0.1, 3, 20),
        (0.5, 2, 100),
        (0.25, 1, 50),
    ],
)
def test__get_tuning_split_indices__regression_respects_frac_and_n_splits(
    holdout_frac: float,
    n_splits: int,
    expected_holdout_size: int,
) -> None:
    n_samples = 200
    X = np.zeros((n_samples, 2))
    y = np.random.default_rng(0).normal(size=n_samples)

    splits = get_tuning_split_indices(
        X=X,
        y=y,
        holdout_frac=holdout_frac,
        n_splits=n_splits,
        task_type="regressor",
    )

    assert len(splits) == n_splits
    for train_indices, holdout_indices in splits:
        assert len(holdout_indices) == expected_holdout_size
        assert len(train_indices) == n_samples - expected_holdout_size


def test__get_tuning_split_indices__regression_holdouts_are_disjoint_across_folds() -> (
    None
):
    # The K-fold structure exists so that no sample is held out twice; a plain
    # random shuffle split would not guarantee this.
    n_samples = 200
    X = np.zeros((n_samples, 2))
    y = np.random.default_rng(0).normal(size=n_samples)

    splits = get_tuning_split_indices(
        X=X,
        y=y,
        holdout_frac=0.2,
        n_splits=5,
        task_type="regressor",
    )

    all_holdout_indices = np.concatenate([holdout for _, holdout in splits])
    assert len(all_holdout_indices) == len(set(all_holdout_indices.tolist()))
    assert set(all_holdout_indices.tolist()) == set(range(n_samples))

    # Each fold's train and holdout parts must be complementary.
    for train_indices, holdout_indices in splits:
        train, holdout = set(train_indices.tolist()), set(holdout_indices.tolist())
        assert train.isdisjoint(holdout)
        assert train | holdout == set(range(n_samples))


def test__get_tuning_split_indices__regression_handles_constant_target() -> None:
    # A constant target has no variation to stratify or bin on.
    X = np.zeros((50, 2))
    y = np.full(50, 3.0)

    splits = get_tuning_split_indices(
        X=X,
        y=y,
        holdout_frac=0.2,
        n_splits=2,
        task_type="regressor",
    )

    assert len(splits) == 2
    for train_indices, holdout_indices in splits:
        assert len(train_indices) + len(holdout_indices) == len(X)


def test__get_tuning_split_indices__classification_is_stratified() -> None:
    n_samples = 200
    X = np.zeros((n_samples, 2))
    y = np.array([0] * 20 + [1] * 180)

    splits = get_tuning_split_indices(
        X=X, y=y, holdout_frac=0.2, n_splits=5, task_type="classifier"
    )

    assert len(splits) == 5
    # Stratification keeps the rare class present in every holdout.
    for _, holdout_indices in splits:
        assert (y[holdout_indices] == 0).sum() == 4


@pytest.mark.parametrize(
    ("num_samples", "expected_holdout_frac", "expected_n_folds"),
    [
        (1_000, 0.1, 10),
        (9_000, 0.2, 3),
        (20_000, 0.2, 2),
        (21_000, 0.3, 1),
    ],
)
def test__regressor_tuning_config__resolves_auto_values_like_classifier(
    num_samples: int,
    expected_holdout_frac: float,
    expected_n_folds: int,
) -> None:
    tuning_config = RegressorTuningConfig(
        calibrate_temperature=True,
        tuning_holdout_frac="auto",
        tuning_n_folds="auto",
    )

    resolved = tuning_config.resolve(num_samples=num_samples)

    # `resolve` must preserve the concrete subclass, not degrade to TuningConfig.
    assert isinstance(resolved, RegressorTuningConfig)
    assert resolved.calibrate_temperature is True
    assert resolved.tuning_holdout_frac == expected_holdout_frac
    assert resolved.tuning_n_folds == expected_n_folds
    # `resolve` returns a new instance and leaves the original untouched.
    assert tuning_config.tuning_holdout_frac == "auto"
    assert tuning_config.tuning_n_folds == "auto"


def test__regressor_tuning_config__keeps_explicit_values() -> None:
    tuning_config = RegressorTuningConfig(
        calibrate_temperature=True,
        tuning_holdout_frac=0.15,
        tuning_n_folds=4,
    )

    resolved = tuning_config.resolve(num_samples=1_000)

    assert resolved.tuning_holdout_frac == 0.15
    assert resolved.tuning_n_folds == 4


def test__regressor_tuning_config__has_no_classification_only_fields() -> None:
    # Threshold tuning is classification-only; a regression config must not expose it.
    assert not hasattr(RegressorTuningConfig(), "tune_decision_thresholds")
    assert RegressorTuningConfig().calibrate_temperature is False


def test__regressor_tuning_config__calibrate_mean_is_off_by_default() -> None:
    assert RegressorTuningConfig().calibrate_mean is False
    assert RegressorTuningConfig().mean_calibration_mode == "tweedie"


@pytest.mark.parametrize("power", [0.0, 0.99, 2.01])
def test__regressor_tuning_config__rejects_mean_calibration_power_outside_1_2(
    power: float,
) -> None:
    with pytest.raises(ValueError, match="must be in \\[1, 2\\]"):
        RegressorTuningConfig(calibrate_mean=True, mean_calibration_power=power)


@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
def test__find_mean_calibration_scale__recovers_a_known_scale(power: float) -> None:
    rng = np.random.default_rng(0)
    # Shifted by 1, so no mean falls below the floor of the mean calibration.
    mean = 1.0 + rng.gamma(shape=2.0, scale=3.0, size=10_000)
    # Zero-inflated, multiplicative noise with mean 1: E[y] = 2.5 * mean.
    y = 2.5 * mean * rng.binomial(1, 0.2, size=mean.size) * 5.0

    scale = find_mean_calibration_scale(y_true=y, mean=mean, power=power)

    assert scale == pytest.approx(2.5, rel=0.1)
    # Without noise every power recovers the scale exactly.
    assert find_mean_calibration_scale(
        y_true=2.5 * mean, mean=mean, power=power
    ) == pytest.approx(2.5, rel=1e-12)


@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
def test__find_mean_calibration_scale__minimizes_the_tweedie_deviance(
    power: float,
) -> None:
    rng = np.random.default_rng(1)
    # Shifted, so no mean falls below the floor of the mean calibration.
    mean = 0.5 + rng.gamma(shape=2.0, scale=1.0, size=500)
    # Not proportional to `mean`, so each power has a different minimizer.
    y = rng.poisson(mean**2) + 1.0

    scale = find_mean_calibration_scale(y_true=y, mean=mean, power=power)

    def deviance(c: float) -> float:
        return mean_tweedie_deviance(y, c * mean, power=power)

    assert deviance(scale) < deviance(scale * 0.99)
    assert deviance(scale) < deviance(scale * 1.01)


@pytest.mark.parametrize("power", [1.0, 1.5, 2.0])
def test__find_mean_calibration_scale__minimizes_the_weighted_tweedie_deviance(
    power: float,
) -> None:
    rng = np.random.default_rng(2)
    mean = rng.gamma(shape=2.0, scale=1.0, size=500)
    y = rng.poisson(mean**2) + 1.0
    # Tied to `mean`, so the weighted minimizer moves away from the plain one.
    weight = rng.uniform(0.0, 1.0, size=mean.size) * mean**2

    scale = find_mean_calibration_scale(
        y_true=y, mean=mean, power=power, sample_weight=weight
    )

    def deviance(c: float) -> float:
        return mean_tweedie_deviance(y, c * mean, power=power, sample_weight=weight)

    assert deviance(scale) < deviance(scale * 0.99)
    assert deviance(scale) < deviance(scale * 1.01)
    assert scale != pytest.approx(
        find_mean_calibration_scale(y_true=y, mean=mean, power=power), rel=1e-3
    )


def test__find_mean_calibration_scale__floors_non_positive_means() -> None:
    y = np.array([0.0, 0.0, 4.0])
    mean = np.array([-0.5, 0.0, 2.0])
    floor = MEAN_CALIBRATION_MIN_MEAN_FRACTION * y.mean()

    # p = 1 is sum(y) / sum(m) with the floored means.
    assert find_mean_calibration_scale(y_true=y, mean=mean, power=1.0) == pytest.approx(
        4.0 / (2.0 + 2 * floor)
    )
    assert np.isfinite(find_mean_calibration_scale(y_true=y, mean=mean, power=1.5))


@pytest.mark.parametrize("power", [1.0, 1.5])
def test__find_mean_calibration_scale__zero_weight_rows_do_not_move_the_floor(
    power: float,
) -> None:
    y = np.array([0.0, 0.0, 4.0, 1e6])
    mean = np.array([-0.5, 0.0, 2.0, 1.0])
    weight = np.array([1.0, 1.0, 1.0, 0.0])

    assert find_mean_calibration_scale(
        y_true=y, mean=mean, power=power, sample_weight=weight
    ) == pytest.approx(
        find_mean_calibration_scale(y_true=y[:3], mean=mean[:3], power=power),
        rel=1e-12,
    )


@pytest.mark.parametrize("power", [1.5, 2.0])
def test__find_mean_calibration_scale__floors_relative_to_the_targets(
    power: float,
) -> None:
    # 1000 rows with m = y = 100 give c = 1. One more row has y = 100 and m = 0.
    # With an absolute floor of 1e-6 this row moves c to about 11 at p = 1.5 and
    # about 1e5 at p = 2.
    y = np.full(1001, 100.0)
    mean = np.append(np.full(1000, 100.0), 0.0)

    scale = find_mean_calibration_scale(y_true=y, mean=mean, power=power)

    assert 1.0 < scale < 1.15
    # The floor scales with the targets, so c does not depend on their units.
    assert find_mean_calibration_scale(
        y_true=1e-4 * y, mean=1e-4 * mean, power=power
    ) == pytest.approx(scale, rel=1e-12)


def test__regressor_eval_metrics__round_trips_from_string() -> None:
    # Constructed from a string the same way `eval_metric` arguments will be.
    assert RegressorEvalMetrics("nll") is RegressorEvalMetrics.NLL
    assert RegressorEvalMetrics.NLL == "nll"
    assert RegressorEvalMetrics.NLL.value == "nll"

    with pytest.raises(ValueError, match="not a valid RegressorEvalMetrics"):
        RegressorEvalMetrics("rmse")


def test__resolve_tuning_config__builds_dict_into_the_requested_config_class() -> None:
    resolved = resolve_tuning_config(
        tuning_config={"calibrate_temperature": True},
        num_samples=1_000,
        config_cls=RegressorTuningConfig,
    )

    assert isinstance(resolved, RegressorTuningConfig)
    assert resolved.calibrate_temperature is True
    assert resolved.tuning_holdout_frac == 0.1
    assert resolved.tuning_n_folds == 10


def test__resolve_tuning_config__defaults_dicts_to_the_classifier_config() -> None:
    # Regression guard: a dict without `config_cls` must still become a
    # `ClassifierTuningConfig`, with its classification-only fields.
    resolved = resolve_tuning_config(
        tuning_config={"tune_decision_thresholds": True},
        num_samples=1_000,
    )

    assert isinstance(resolved, ClassifierTuningConfig)
    assert resolved.tune_decision_thresholds is True


def test__resolve_tuning_config__keeps_a_config_instance_over_config_cls() -> None:
    # An already-built config wins; `config_cls` only applies to dict inputs.
    resolved = resolve_tuning_config(
        tuning_config=ClassifierTuningConfig(tune_decision_thresholds=True),
        num_samples=1_000,
        config_cls=RegressorTuningConfig,
    )

    assert isinstance(resolved, ClassifierTuningConfig)


@pytest.mark.parametrize(
    ("tuning_config", "expected_options"),
    [
        (
            ClassifierTuningConfig(),
            "`calibrate_temperature=True` or `tune_decision_thresholds=True`",
        ),
        (
            RegressorTuningConfig(),
            "`calibrate_temperature=True` or `calibrate_mean=True`",
        ),
    ],
)
def test__resolve_tuning_config__warns_about_the_options_the_config_has(
    tuning_config: ClassifierTuningConfig | RegressorTuningConfig,
    expected_options: str,
) -> None:
    # The regression config has no `tune_decision_thresholds`, so suggesting it
    # would send the user after an argument that does not exist.
    with pytest.warns(UserWarning, match="no tuning features were enabled") as record:
        resolved = resolve_tuning_config(
            tuning_config=tuning_config,
            num_samples=1_000,
        )

    assert resolved is None
    assert f"Set {expected_options} to enable tuning." in str(record[0].message)


def test__resolve_tuning_config__warns_for_small_datasets_in_both_tasks() -> None:
    for tuning_config in (
        ClassifierTuningConfig(calibrate_temperature=True),
        RegressorTuningConfig(calibrate_temperature=True),
    ):
        with pytest.warns(UserWarning, match="samples in the training data"):
            resolved = resolve_tuning_config(
                tuning_config=tuning_config,
                num_samples=MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING - 1,
            )

        assert resolved is not None
        assert resolved.calibrate_temperature is True


def test__get_tuning_feature_flags__lists_only_the_boolean_feature_fields() -> None:
    # The holdout knobs configure how tuning runs, not whether it runs.
    assert get_tuning_feature_flags(
        ClassifierTuningConfig(calibrate_temperature=True, tuning_n_folds=3)
    ) == {"calibrate_temperature": True, "tune_decision_thresholds": False}
    assert get_tuning_feature_flags(RegressorTuningConfig()) == {
        "calibrate_temperature": False,
        "calibrate_mean": False,
    }


def _make_bardist(
    n_buckets: int = 40,
    *,
    scale: float = 1.0,
    shift: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> FullSupportBarDistribution:
    """A bar distribution over `[-6, 6] * scale + shift`, in the target's units.

    `dtype` exists for the exact-invariance tests below: in float32 the rescaled
    borders are not the exact rescaling of the originals, so the geometry itself
    shifts and swamps the property under test.
    """
    borders = torch.linspace(-6.0, 6.0, n_buckets + 1, dtype=dtype) * scale + shift
    return FullSupportBarDistribution(borders)


def _gaussian_logits(
    raw_space_bardist: FullSupportBarDistribution,
    means: np.ndarray,
    sd: float,
) -> torch.Tensor:
    """Logits of a Gaussian with width `sd` centred on each of `means`.

    Dividing these by T rescales the width to `sd * sqrt(T)`, so the temperature
    that best fits targets drawn with width `s_true` is `(s_true / sd) ** 2`.
    """
    centers = (raw_space_bardist.borders[:-1] + raw_space_bardist.borders[1:]) / 2
    means_t = torch.as_tensor(means, dtype=torch.float32)
    return -0.5 * ((centers[None, :] - means_t[:, None]) / sd) ** 2


def _miscalibrated_fold(
    *,
    n_samples: int,
    target_temperature: float,
    seed: int,
    scale: float = 1.0,
) -> tuple[torch.Tensor, FullSupportBarDistribution, torch.Tensor]:
    """A fold whose predictions are off by `target_temperature`.

    Targets are drawn with width `scale`, and predicted with width
    `scale / sqrt(target_temperature)`, so the sweep should recover roughly
    `target_temperature`.
    """
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.5 * scale, size=n_samples)
    y_true = means + rng.normal(0.0, scale, size=n_samples)

    raw_space_bardist = _make_bardist(scale=scale)
    logits = _gaussian_logits(
        raw_space_bardist, means, sd=scale / np.sqrt(target_temperature)
    )
    return logits, raw_space_bardist, torch.as_tensor(y_true, dtype=torch.float32)


def _pooled_nll(
    folds: list[tuple[torch.Tensor, FullSupportBarDistribution, torch.Tensor]],
    temperature: float,
) -> float:
    return float(
        torch.cat(
            [
                compute_regression_metric_to_minimize(
                    metric_name=RegressorEvalMetrics.NLL,
                    raw_space_bardist=copy.deepcopy(raw_space_bardist),
                    logits=logits / temperature,
                    y_true=y_true,
                )
                for logits, raw_space_bardist, y_true in folds
            ]
        ).mean()
    )


def test__find_regression_optimal_temperature__widens_overconfident() -> None:
    # Predictions sharper than the targets warrant: the sweep should widen them.
    folds = [_miscalibrated_fold(n_samples=400, target_temperature=1.2, seed=0)]

    temperature = find_regression_optimal_temperature(
        holdout_folds=folds,
        metric_name=RegressorEvalMetrics.NLL,
        current_default_temperature=1.0,
    )

    assert 1.0 < temperature <= 1.4
    assert _pooled_nll(folds, temperature) < _pooled_nll(folds, 1.0)


def test__find_regression_optimal_temperature__sharpens_underconfident() -> None:
    # Predictions wider than the targets warrant: the sweep should sharpen them.
    folds = [_miscalibrated_fold(n_samples=400, target_temperature=0.83, seed=1)]

    temperature = find_regression_optimal_temperature(
        holdout_folds=folds,
        metric_name=RegressorEvalMetrics.NLL,
        current_default_temperature=1.0,
    )

    assert 0.6 <= temperature < 1.0
    assert _pooled_nll(folds, temperature) < _pooled_nll(folds, 1.0)


def _optimal_temperature(
    folds: list[tuple[torch.Tensor, FullSupportBarDistribution, torch.Tensor]],
) -> float:
    return find_regression_optimal_temperature(
        holdout_folds=folds,
        metric_name=RegressorEvalMetrics.NLL,
        current_default_temperature=1.0,
    )


def test__find_regression_optimal_temperature__pools_differing_bardists() -> None:
    # Each fold's regressor normalises with its own statistics, so the folds carry
    # different bar distributions and cannot be concatenated into one. The pooled
    # answer must still land between what either fold would pick alone.
    sharp_fold = _miscalibrated_fold(n_samples=300, target_temperature=0.8, seed=2)
    wide_fold = _miscalibrated_fold(
        n_samples=300, target_temperature=1.3, seed=3, scale=50.0
    )
    assert not torch.equal(sharp_fold[1].borders, wide_fold[1].borders)

    sharp_only = _optimal_temperature([sharp_fold])
    wide_only = _optimal_temperature([wide_fold])
    pooled = _optimal_temperature([sharp_fold, wide_fold])

    assert sharp_only < pooled < wide_only


def test__find_regression_optimal_temperature__weights_folds_by_holdout_size() -> None:
    # Pooling happens per sample, not per fold, so the bigger holdout dominates.
    small_fold = _miscalibrated_fold(n_samples=40, target_temperature=0.8, seed=4)
    big_fold = _miscalibrated_fold(n_samples=1_200, target_temperature=1.3, seed=5)

    small_only = _optimal_temperature([small_fold])
    big_only = _optimal_temperature([big_fold])
    pooled = _optimal_temperature([small_fold, big_fold])

    assert abs(pooled - big_only) < abs(pooled - small_only)


def test__find_regression_optimal_temperature__follows_the_weighted_rows() -> None:
    sharp_fold = _miscalibrated_fold(n_samples=400, target_temperature=0.8, seed=4)
    wide_fold = _miscalibrated_fold(n_samples=400, target_temperature=1.3, seed=5)
    ones, zeros = torch.ones(400), torch.zeros(400)

    def temperature(sharp_weight: torch.Tensor, wide_weight: torch.Tensor) -> float:
        return _optimal_temperature(
            [(*sharp_fold, None, sharp_weight), (*wide_fold, None, wide_weight)]
        )

    assert temperature(ones, zeros) == _optimal_temperature([sharp_fold])
    assert temperature(zeros, ones) == _optimal_temperature([wide_fold])
    assert temperature(ones, ones) == _optimal_temperature([sharp_fold, wide_fold])
    # No weight anywhere leaves nothing to tune on.
    assert temperature(zeros, zeros) == 1.0


def test__find_regression_optimal_temperature__is_invariant_to_target_units() -> None:
    # Rescaling a fold's targets and its bar distribution together is a change of
    # units, so the chosen temperature must not move even though the NLL value does.
    in_units = _miscalibrated_fold(n_samples=300, target_temperature=1.25, seed=6)
    scaled = _miscalibrated_fold(
        n_samples=300, target_temperature=1.25, seed=6, scale=1_000.0
    )

    assert _optimal_temperature([in_units]) == _optimal_temperature([scaled])
    # A wider bucket means a lower density, hence a loss offset by log(1000).
    assert _pooled_nll([scaled], 1.0) == pytest.approx(
        _pooled_nll([in_units], 1.0) + np.log(1_000.0), abs=1e-3
    )


def test__find_regression_optimal_temperature__leaves_bardists_unmutated() -> None:
    # `FullSupportBarDistribution.forward` accumulates into `losses_per_bucket` on
    # every call; an 82-point sweep must not leak that into the caller's distribution.
    logits, raw_space_bardist, y_true = _miscalibrated_fold(
        n_samples=100, target_temperature=1.2, seed=7
    )
    before = raw_space_bardist.losses_per_bucket.clone()

    find_regression_optimal_temperature(
        holdout_folds=[(logits, raw_space_bardist, y_true)],
        metric_name=RegressorEvalMetrics.NLL,
        current_default_temperature=1.0,
    )

    assert torch.equal(raw_space_bardist.losses_per_bucket, before)
    assert not before.any()


@pytest.mark.parametrize("n_samples", [0])
def test__find_regression_optimal_temperature__falls_back_when_degenerate(
    n_samples: int,
) -> None:
    empty = _miscalibrated_fold(n_samples=n_samples, target_temperature=1.2, seed=8)

    assert (
        find_regression_optimal_temperature(
            holdout_folds=[],
            metric_name=RegressorEvalMetrics.NLL,
            current_default_temperature=0.9,
        )
        == 0.9
    )
    assert (
        find_regression_optimal_temperature(
            holdout_folds=[empty],
            metric_name=RegressorEvalMetrics.NLL,
            current_default_temperature=0.9,
        )
        == 0.9
    )


@pytest.mark.parametrize("metric", list(RegressorEvalMetrics))
def test__compute_regression_metric_to_minimize__returns_one_loss_per_sample(
    metric: RegressorEvalMetrics,
) -> None:
    # Every metric must expose per-row losses, so that the search can pool folds of
    # differing size without a metric-specific reduction.
    logits, raw_space_bardist, y_true = _miscalibrated_fold(
        n_samples=17, target_temperature=1.2, seed=9
    )

    losses = compute_regression_metric_to_minimize(
        metric_name=metric,
        raw_space_bardist=raw_space_bardist,
        logits=logits,
        y_true=y_true,
    )

    assert losses.shape == (17,)
    assert torch.isfinite(losses).all()


def test__compute_regression_metric_to_minimize__rejects_unsupported_metrics() -> None:
    logits, raw_space_bardist, y_true = _miscalibrated_fold(
        n_samples=5, target_temperature=1.0, seed=10
    )

    with pytest.raises(ValueError, match="Metric 'rmse' is not supported"):
        compute_regression_metric_to_minimize(
            metric_name="rmse",  # type: ignore[arg-type]
            raw_space_bardist=raw_space_bardist,
            logits=logits,
            y_true=y_true,
        )


def test__get_tuning_temperatures__straddles_the_no_op_temperature() -> None:
    temperatures = get_tuning_temperatures()

    assert temperatures.shape == (81,)
    assert temperatures[0] == pytest.approx(0.6)
    assert temperatures[-1] == pytest.approx(1.4)
    assert (np.diff(temperatures) > 0).all()
    np.testing.assert_allclose(np.diff(temperatures), 0.01)
    # A sharpening correction, a widening one, and leaving well alone must all be
    # reachable; see `test__get_tuning_temperatures__contains_exactly_one`.
    assert temperatures.min() < 1.0 < temperatures.max()


def test__temperature_searches__only_return_values_from_the_shared_grid() -> None:
    # Both searches sweep the same grid, so neither can return an off-grid value.
    temperatures = get_tuning_temperatures()

    regression_temperature = find_regression_optimal_temperature(
        holdout_folds=[
            _miscalibrated_fold(n_samples=200, target_temperature=1.2, seed=11)
        ],
        metric_name=RegressorEvalMetrics.NLL,
        current_default_temperature=1.0,
    )
    rng = np.random.default_rng(11)
    classification_temperature = find_optimal_temperature(
        raw_logits=rng.normal(size=(2, 60, 3)),
        y_true=rng.integers(0, 3, size=60),
        logits_to_probabilities_fn=_softmax_over_estimators,
        current_default_temperature=1.0,
    )

    assert regression_temperature in temperatures
    assert classification_temperature in temperatures


def _pooled(folds, metric, temperature) -> float:
    """Reproduces the search's pooling: fold means weighted by holdout size."""
    total = sum(len(y) for _, _, y in folds)
    return (
        sum(
            len(y)
            * float(
                compute_regression_metric_to_minimize(
                    metric_name=metric,
                    raw_space_bardist=copy.deepcopy(bardist),
                    logits=logits / temperature,
                    y_true=y,
                ).mean()
            )
            for logits, bardist, y in folds
        )
        / total
    )


@pytest.mark.parametrize(
    "metric",
    [RegressorEvalMetrics.NLL, RegressorEvalMetrics.CRPS],
)
def test__find_regression_optimal_temperature__every_metric_beats_no_calibration(
    metric: RegressorEvalMetrics,
) -> None:
    # Whichever metric drives the search, the temperature it picks must score at least
    # as well on that same metric as leaving the distribution alone -- 1.0 is on the
    # grid, so the sweep can always fall back to it and can never do worse.
    folds = [_miscalibrated_fold(n_samples=600, target_temperature=1.25, seed=7)]

    temperature = find_regression_optimal_temperature(
        holdout_folds=folds,
        metric_name=metric,
        current_default_temperature=1.0,
    )

    assert temperature in get_tuning_temperatures()
    assert _pooled(folds, metric, temperature) <= _pooled(folds, metric, 1.0)


@pytest.mark.parametrize(
    "metric", [RegressorEvalMetrics.NLL, RegressorEvalMetrics.CRPS]
)
def test__find_regression_optimal_temperature__distributional_metrics_widen(
    metric: RegressorEvalMetrics,
) -> None:
    # Predictions sharper than the targets warrant: both metrics should widen them.
    folds = [_miscalibrated_fold(n_samples=600, target_temperature=1.25, seed=7)]

    temperature = find_regression_optimal_temperature(
        holdout_folds=folds,
        metric_name=metric,
        current_default_temperature=1.0,
    )

    assert temperature > 1.0


def test__find_regression_optimal_temperature__metrics_share_the_search() -> None:
    # All metrics run through the identical code path and grid, so they can only
    # differ in *which* grid point wins -- never in the set of candidates.
    folds = [_miscalibrated_fold(n_samples=500, target_temperature=1.15, seed=8)]
    grid = get_tuning_temperatures()

    for metric in RegressorEvalMetrics:
        assert (
            find_regression_optimal_temperature(
                holdout_folds=folds,
                metric_name=metric,
                current_default_temperature=1.0,
            )
            in grid
        )


def test__compute_regression_metric_to_minimize__crps_uses_the_shared_metric() -> None:
    # The regression metric must be the *same* function the finetuning loss uses, so
    # the two cannot drift apart. Compared against a direct call to prove it.
    raw_space_bardist = _make_bardist()
    rng = np.random.default_rng(13)
    logits = torch.as_tensor(rng.normal(size=(9, 40)), dtype=torch.float32)
    y = torch.as_tensor(rng.normal(size=9), dtype=torch.float32)

    torch.testing.assert_close(
        compute_regression_metric_to_minimize(
            metric_name=RegressorEvalMetrics.CRPS,
            raw_space_bardist=raw_space_bardist,
            logits=logits,
            y_true=y,
        ),
        ranked_probability_score_loss_from_bar_logits(
            logits_BQL=logits.unsqueeze(0),
            targets_BQ=y.unsqueeze(0),
            bardist_loss_fn=raw_space_bardist,
            loss_type="crps",
        ).squeeze(0),
    )


def test__get_tuning_temperatures__contains_exactly_one() -> None:
    # calibration must be able to conclude "leave
    # this distribution alone". An approximate 1.0 would not do, because
    # `_compute_aggregated_logits` guards on `temperature != 1.0`.
    temperatures = get_tuning_temperatures()

    assert (temperatures == 1.0).any()
    assert float(temperatures[temperatures == 1.0][0]) == 1.0
    # A true no-op divisor, not merely a value that rounds to one.
    assert 3.14159 / float(temperatures[temperatures == 1.0][0]) == 3.14159


@pytest.mark.parametrize("seed", [9, 10, 11])
def test__find_regression_optimal_temperature__settles_near_the_no_op(
    seed: int,
) -> None:
    """A fold that is already honest must not be pushed far from 1.0.

    This is a statistical statement, not an exact one: even with 4000 holdout rows the
    empirical optimum wanders a few grid steps around the true one, so the tolerance is
    a noise floor rather than a claim about the grid. That 1.0 is *exactly* reachable
    is pinned separately by `test__get_tuning_temperatures__contains_exactly_one`;
    before the grid change the nearest candidates were 0.995 and 1.005 and neither was
    a true no-op.
    """
    folds = [_miscalibrated_fold(n_samples=4_000, target_temperature=1.0, seed=seed)]

    temperature = find_regression_optimal_temperature(
        holdout_folds=folds,
        metric_name=RegressorEvalMetrics.NLL,
        current_default_temperature=1.0,
    )

    assert abs(temperature - 1.0) <= 0.05
