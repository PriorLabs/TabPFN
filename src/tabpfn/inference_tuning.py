"""Inference tuning helpers for TabPFN fit/predict calls."""

from __future__ import annotations

import dataclasses
import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, Literal
from typing_extensions import Self

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

if TYPE_CHECKING:
    import torch


MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING = 500


@dataclasses.dataclass
class TuningConfig:
    """Configuration for tuning the model during fit/predict calls."""

    calibrate_temperature: bool = False
    """Whether to calibrate the softmax temperature. Set to True to enable."""

    tuning_holdout_frac: Literal["auto"] | float = "auto"
    """The percentage of the data to hold out for tuning per split. If "auto", a value
    is automatically chosen based on the dataset size, trading off between
    computational cost and accuracy."""

    tuning_n_folds: Literal["auto"] | int = "auto"
    """The number of cross-validation folds to use for tuning. If "auto", a value
    is automatically chosen based on the dataset size, trading off between
    computational cost and accuracy."""

    def resolve(self: Self, num_samples: int) -> Self:
        """Resolves 'auto' values based on the number of samples.

        Args:
            num_samples: The number of samples in the training data.

        Returns:
            A new TuningConfig instance with resolved values.
        """
        tuning_holdout_frac = (
            get_default_tuning_holdout_pct(n_samples=num_samples)
            if self.tuning_holdout_frac == "auto"
            else self.tuning_holdout_frac
        )
        tuning_n_folds = (
            get_default_tuning_holdout_n_splits(n_samples=num_samples)
            if self.tuning_n_folds == "auto"
            else self.tuning_n_folds
        )
        return dataclasses.replace(
            self,
            tuning_holdout_frac=tuning_holdout_frac,
            tuning_n_folds=tuning_n_folds,
        )


@dataclasses.dataclass
class ClassifierTuningConfig(TuningConfig):
    """Configuration for tuning the model during fit/predict calls
    for classification tasks.
    """

    tune_decision_thresholds: bool = False
    """Whether to tune decision thresholds for the specified `eval_metric`.
    Set to True to enable."""


class ClassificationEvalMetrics(str, Enum):
    """Metric by which predictions will be ultimately evaluated on test data."""

    F1 = "f1"
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"


METRIC_NAME_TO_OBJECTIVE = {
    "f1": lambda y_true, y_pred: -f1_score(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    ),
    "accuracy": lambda y_true, y_pred: -accuracy_score(
        y_true,
        y_pred,
    ),
    "balanced_accuracy": lambda y_true, y_pred: -balanced_accuracy_score(
        y_true,
        y_pred,
    ),
    "roc_auc": lambda y_true, y_pred: -roc_auc_score(
        y_true,
        y_pred,
    ),
    "log_loss": lambda y_true, y_pred: log_loss(
        y_true,
        y_pred,
    ),
}


def compute_metric_to_minimize(
    metric_name: ClassificationEvalMetrics,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Computes the metric.

    Adjusts the sign of the metric to frame the problem as a minimization
    problem.
    """
    if metric_name not in METRIC_NAME_TO_OBJECTIVE:
        raise ValueError(
            f"Metric '{metric_name}' is not supported. "
            f"Supported metrics are: {list(METRIC_NAME_TO_OBJECTIVE.keys())}"
        )
    return METRIC_NAME_TO_OBJECTIVE[metric_name](y_true, y_pred)


def get_tuning_splits(
    X: np.ndarray,
    y: np.ndarray,
    holdout_frac: float,
    n_splits: int = 1,
    random_state: int | np.random.RandomState | np.random.Generator | None = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Get stratified tuning split(s) for the given configuration.

    Args:
        X: The input data of shape [n_samples, n_features].
        y: The target labels of shape [n_samples].
        holdout_frac: The percentage of the data to hold out for tuning.
        n_splits: Number of stratified random splits to generate.
        random_state: The random state to use for the split(s).

    Returns:
        Returns a list of splits as tuples of
        (X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh).
        Shape suffixes: Nt=num train samples, F=num features, Nh=num holdout samples.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=holdout_frac,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_indices, holdout_indices in splitter.split(X, y):
        X_train_NtF = X[train_indices]
        X_holdout_NhF = X[holdout_indices]
        y_train_Nt = y[train_indices]
        y_holdout_Nh = y[holdout_indices]
        splits.append((X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh))

    return splits


def find_optimal_classification_thresholds(
    metric_name: ClassificationEvalMetrics,
    y_true: np.ndarray,
    y_pred_probas: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Finds the optimal thresholds for each class in a one-vs-rest (OvR) fashion.

    Args:
        metric_name: The name of the metric to optimize.
        y_true: The true labels of shape [n_samples].
        y_pred_probas: The predicted probabilities of shape [n_samples, n_classes].
        n_classes: The number of classes.

    Returns:
        The optimal thresholds of shape [n_classes].
    """
    optimal_thresholds = []

    # TODO: vectorize this loop lop and the one in
    # find_optimal_classification_threshold_single_class.
    for i in range(n_classes):
        y_true_ovr = (y_true == i).astype(int)
        y_pred_probas_ovr = y_pred_probas[:, i]
        best_thresh = find_optimal_classification_threshold_single_class(
            metric_name=metric_name,
            y_true=y_true_ovr,
            y_pred_probas=y_pred_probas_ovr,
        )

        optimal_thresholds.append(best_thresh)

    return np.array(optimal_thresholds)


def find_optimal_classification_threshold_single_class(
    metric_name: ClassificationEvalMetrics,
    y_true: np.ndarray,
    y_pred_probas: np.ndarray,
) -> float:
    """Finds the optimal classification threshold to maximize the specified metric.

    The true labels are binary, and the predicted probabilities are the probabilities of
    the positive class.

    Args:
        metric_name: The name of the metric to optimize.
        y_true: The true labels of shape [n_samples].
        y_pred_probas: The predicted probabilities of shape [n_samples].

    Returns:
        The optimal threshold.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    thresholds_and_losses: list[tuple[float, float]] = []  # (threshold, metric)

    for threshold in thresholds:
        y_pred_tuned = (y_pred_probas >= threshold).astype(int)
        current_loss = compute_metric_to_minimize(
            metric_name=metric_name,
            y_true=y_true,
            y_pred=y_pred_tuned,
        )
        thresholds_and_losses.append((float(threshold), current_loss))

    return select_robust_optimal_threshold(thresholds_and_losses=thresholds_and_losses)


def select_robust_optimal_threshold(
    thresholds_and_losses: list[tuple[float, float]],
    plateau_delta: float = 0.002,
) -> float:
    """Selects the robust optimal threshold for the given metric.

    This method avoids selecting a threshold that is at the edge
    of a plateau which may not generalize well.

    Args:
        thresholds_and_losses: The thresholds and losses as a list of tuples.
            The first element of the tuple is the threshold and the second element
            is the loss.
        plateau_delta: The delta to define a plateau around the best loss.

    Returns:
        The robust optimal threshold.
    """
    thresholds = np.array([t for t, _ in thresholds_and_losses], dtype=float)
    losses = np.array([f for _, f in thresholds_and_losses], dtype=float)
    best_loss = float(np.min(losses))
    close_mask = losses <= (best_loss + plateau_delta)

    # Find the contiguous region around the global minimum index
    max_index = int(np.argmin(losses))
    start = max_index
    while start - 1 >= 0 and close_mask[start - 1]:
        start -= 1
    end = max_index
    num_points = len(losses)
    while end + 1 < num_points and close_mask[end + 1]:
        end += 1
    mid_index = (start + end) // 2
    robust_threshold = float(thresholds[mid_index])

    # Edge guard: if chosen threshold is at exact endpoints and region has width,
    # pick the second point from edge
    if mid_index == 0 and end > 0:
        robust_threshold = float(thresholds[1])
    elif mid_index == num_points - 1 and start < num_points - 1:
        robust_threshold = float(thresholds[num_points - 2])

    return robust_threshold


def find_optimal_temperature(
    raw_logits: np.ndarray,
    y_true: np.ndarray,
    logits_to_probabilities_fn: Callable[
        [np.ndarray | torch.Tensor, float], np.ndarray
    ],
    current_default_temperature: float,
) -> float:
    """Finds the optimal temperature to maximize the specified metric.

    Args:
        raw_logits: The raw logits of shape [n_estimators, n_samples, n_classes].
        y_true: The true labels of shape [n_samples].
        logits_to_probabilities_fn: The function to convert logits to probabilities.
            The function should take an array of the shape of raw_logits and
            a softmax temperature as argument.
        current_default_temperature: The current default temperature.

    Returns:
        The temperature that minimizes the log loss.
    """
    temperatures = np.linspace(0.6, 1.4, 41)
    best_log_loss = float("inf")
    best_temperature = current_default_temperature

    if raw_logits.ndim == 3:
        raw_logits = raw_logits.mean(axis=0)

    # TODO: think about vectorizing this loop.
    for temperature in temperatures:
        probas = logits_to_probabilities_fn(raw_logits, temperature)
        current_log_loss = log_loss(y_true=y_true, y_pred=probas)

        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_temperature = temperature

    return best_temperature


def eval_metric_benefits_from_temperature_calibration(
    eval_metric: ClassificationEvalMetrics,
) -> bool:
    """Whether the evaluation metric benefits from calibrating the temperature."""
    return eval_metric in {
        ClassificationEvalMetrics.LOG_LOSS,  # directly benefits from calibration
        ClassificationEvalMetrics.F1,  # indirect benefit
        ClassificationEvalMetrics.ACCURACY,  # indirect benefit
        # Benefits indirectly from threshold tuning but we perform
        # a hard rescaling based on class counts. That's why we comment it here.
        # ClassificationEvalMetrics.BALANCED_ACCURACY,  # indirect benefit
    }


def eval_metric_benefits_from_threshold_tuning(
    eval_metric: ClassificationEvalMetrics,
) -> bool:
    """Whether the evaluation metric benefits from threshold tuning."""
    return eval_metric in {
        ClassificationEvalMetrics.F1,
        # Benefits directly from threshold tuning but we perform
        # a hard rescaling based on class counts. That's why we comment it here.
        # ClassificationEvalMetrics.BALANCED_ACCURACY,
        ClassificationEvalMetrics.ACCURACY,
    }


def get_default_tuning_holdout_pct(n_samples: int) -> float:
    """Gets the default tuning holdout percentage based on a heuristic.

    We aim to tradeoff between computational cost and accuracy.
    """
    n_samples_to_pct = {
        2_000: 0.1,
        5_000: 0.2,
        10_000: 0.2,
        20_000: 0.2,
        50_000: 0.3,
    }
    for n_samples_threshold, pct in n_samples_to_pct.items():
        if n_samples <= n_samples_threshold:
            return pct
    return 0.2


def get_default_tuning_holdout_n_splits(n_samples: int) -> int:
    """Gets the default tuning holdout number of splits based on a heuristic.

    We aim to tradeoff between computational cost and accuracy.
    """
    n_samples_to_splits = {
        2_000: 10,
        5_000: 5,
        10_000: 3,
        20_000: 2,
        50_000: 1,
    }
    for n_samples_threshold, n_splits in n_samples_to_splits.items():
        if n_samples <= n_samples_threshold:
            return n_splits
    return 1


def resolve_tuning_config(
    tuning_config: dict | TuningConfig | None,
    num_samples: int,
) -> TuningConfig | ClassifierTuningConfig | None:
    """Resolves the tuning configuration by checking if tuning is needed,
    resolving 'auto' values for holdout parameters, and returning the appropriate
    type of tuning configuration if the input is a dict.

    Args:
        tuning_config: The tuning configuration to use. If a dict is provided,
            the function will infer the appropriate config type based on the keys
            present (e.g., 'tune_decision_thresholds' indicates
            ClassificationTuningConfig).
        num_samples: The number of samples in the training data.

    Returns:
        The resolved tuning configuration or None if no tuning is needed.
        The returned type will be the same as the input type (or inferred from dict).
    """
    if tuning_config is None:
        return None

    if isinstance(tuning_config, dict):
        if "tune_decision_thresholds" in tuning_config:
            tuning_config = ClassifierTuningConfig(**tuning_config)
        else:
            tuning_config = TuningConfig(**tuning_config)

    compute_holdout_logits = bool(
        tuning_config.calibrate_temperature
        or (
            isinstance(tuning_config, ClassifierTuningConfig)
            and tuning_config.tune_decision_thresholds
        )
    )
    if not compute_holdout_logits:
        logging.warning(
            "You specified a tuning configuration but no tuning features were enabled. "
            "Set `calibrate_temperature=True` or `tune_decision_thresholds=True` to "
            "enable tuning."
        )
        return None

    if num_samples < MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING:
        logging.warning(
            f"You have `{num_samples}` samples in the training data and specifed "
            "a tuning configuration. "
            "We recommend tuning only for datasets with more than "
            f"{MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING} samples. "
        )

    return tuning_config.resolve(num_samples=num_samples)
