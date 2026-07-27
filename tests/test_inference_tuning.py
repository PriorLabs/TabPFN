#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import numpy as np
import pytest

from tabpfn.inference_tuning import (
    ClassifierEvalMetrics,
    ClassifierTuningConfig,
    find_optimal_classification_threshold_single_class,
    find_optimal_classification_thresholds,
    find_optimal_temperature,
    get_tuning_splits,
    resolve_tuning_config,
    select_robust_optimal_threshold,
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


def test__find_optimal_temperature__works_when_class_missing_from_holdout() -> None:
    rng = np.random.default_rng(0)
    n_estimators, n_samples, n_classes = 2, 50, 3
    raw_logits = rng.normal(size=(n_estimators, n_samples, n_classes))
    # Class 2 never appears in the holdout labels.
    y_true = rng.integers(0, 2, size=n_samples)

    def logits_to_probabilities_fn(
        raw_logits: np.ndarray,
        softmax_temperature: float,
    ) -> np.ndarray:
        scaled = raw_logits / softmax_temperature
        exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        probas = exp / exp.sum(axis=-1, keepdims=True)
        return probas.mean(axis=0)

    temperature = find_optimal_temperature(
        raw_logits=raw_logits,
        y_true=y_true,
        logits_to_probabilities_fn=logits_to_probabilities_fn,
        current_default_temperature=1.0,
    )

    assert 0.6 <= temperature <= 1.4


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


def test__get_tuning_splits__accepts_numpy_generator_random_state() -> None:
    X = np.arange(100, dtype=np.float64).reshape(50, 2)
    y = np.array([0, 1] * 25)

    splits = get_tuning_splits(
        X=X,
        y=y,
        holdout_frac=0.2,
        n_splits=1,
        random_state=np.random.default_rng(0),
    )

    assert len(splits) == 1
    X_train, X_holdout, y_train, y_holdout = splits[0]
    assert len(X_train) + len(X_holdout) == len(X)
    assert len(y_train) + len(y_holdout) == len(y)
