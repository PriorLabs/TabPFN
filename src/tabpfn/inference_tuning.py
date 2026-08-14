#  Copyright (c) Prior Labs GmbH 2026.

"""Inference tuning helpers for TabPFN fit/predict calls."""

from __future__ import annotations

import copy
import dataclasses
import warnings
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Literal
from typing_extensions import Self

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from tabpfn.regression_metrics import (
    ranked_probability_score_loss_from_bar_logits,
)
from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    from tabpfn.architectures.shared.bar_distribution import (
        FullSupportBarDistribution,
    )


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
            get_default_tuning_holdout_frac(n_samples=num_samples)
            if self.tuning_holdout_frac == "auto"
            else self.tuning_holdout_frac
        )
        tuning_n_folds = (
            get_default_tuning_n_folds(n_samples=num_samples)
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


@dataclasses.dataclass
class RegressorTuningConfig(TuningConfig):
    """Configuration for tuning the model during fit/predict calls
    for regression tasks.

    This currently adds no fields beyond the ones inherited from
    [TuningConfig][tabpfn.inference_tuning.TuningConfig]. It exists so that the
    task type can be recovered from the configuration alone, and so that
    regression-only tuning options have a place to live as they are added.
    """


class ClassifierEvalMetrics(str, Enum):
    """Metric by which predictions will be ultimately evaluated on test data."""

    F1 = "f1"
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"


class RegressorEvalMetrics(str, Enum):
    """Metric by which predictions will be ultimately evaluated on test data."""

    NLL = "nll"
    """Negative log-likelihood of the targets under the predicted bar
    distribution, evaluated in the raw target space."""

    CRPS = "crps"
    """Continuous ranked probability score of the targets under the predicted bar
    distribution, evaluated in the raw target space."""


# Objectives for the one-vs-rest threshold search, so the inputs are always
# binary: callers binarize y_true as ``(y_true == i)`` and pass thresholded 0/1
# predictions. ``f1`` (average="binary") and ``roc_auc`` (1-D scores) rely on
# this too, and raise on multiclass input.
BINARY_METRIC_NAME_TO_OBJECTIVE = {
    "f1": lambda y_true, y_pred: (
        -f1_score(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
    ),
    "accuracy": lambda y_true, y_pred: (
        -accuracy_score(
            y_true,
            y_pred,
        )
    ),
    "balanced_accuracy": lambda y_true, y_pred: (
        -balanced_accuracy_score(
            y_true,
            y_pred,
        )
    ),
    "roc_auc": lambda y_true, y_pred: (
        -roc_auc_score(
            y_true,
            y_pred,
        )
    ),
    # y_true is binarized one-vs-rest and may contain a single label.
    "log_loss": lambda y_true, y_pred: log_loss(y_true, y_pred, labels=[0, 1]),
}


def compute_metric_to_minimize(
    metric_name: ClassifierEvalMetrics,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Computes the metric for binary one-vs-rest inputs.

    Adjusts the sign of the metric to frame the problem as a minimization
    problem.
    """
    if metric_name not in BINARY_METRIC_NAME_TO_OBJECTIVE:
        raise ValueError(
            f"Metric '{metric_name}' is not supported. "
            f"Supported metrics are: {list(BINARY_METRIC_NAME_TO_OBJECTIVE.keys())}"
        )
    return BINARY_METRIC_NAME_TO_OBJECTIVE[metric_name](y_true, y_pred)


# Objectives for the regression temperature search. Each returns the loss of every
# holdout row, given logits of shape [n_samples, n_buckets] and targets of shape
# [n_samples].
REGRESSION_METRIC_NAME_TO_OBJECTIVE: dict[
    str,
    Callable[
        [FullSupportBarDistribution, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
] = {
    # Calling the bar distribution runs `FullSupportBarDistribution.forward`, which
    # returns the negative log density of each target.
    "nll": lambda raw_space_bardist, logits, y_true: raw_space_bardist(logits, y_true),
    "crps": lambda raw_space_bardist, logits, y_true: (
        ranked_probability_score_loss_from_bar_logits(
            logits_BQL=logits.unsqueeze(0),
            targets_BQ=y_true.unsqueeze(0),
            bardist_loss_fn=raw_space_bardist,
            loss_type="crps",
        ).squeeze(0)
    ),
}


def compute_regression_metric_to_minimize(
    metric_name: RegressorEvalMetrics,
    raw_space_bardist: FullSupportBarDistribution,
    logits: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Computes the per-sample losses of a regression metric, to be minimized.

    The metric is evaluated in the units of the original target: `raw_space_bardist`
    is expected to be the fold's `TabPFNRegressor.raw_space_bardist_`, which undoes
    the z-normalisation that its regressor applied, and `y_true` is expected to be
    the untransformed target.

    Args:
        metric_name: The name of the metric to compute.
        raw_space_bardist: The bar distribution that gives `logits` their support,
            with borders in the units of the original target.
        logits: The aggregated logits of shape [n_samples, n_buckets].
        y_true: The true targets of shape [n_samples], in the same units as the
            borders of `raw_space_bardist`, and on the same device.

    Returns:
        The losses to minimize, one per sample, of shape [n_samples].
    """
    if metric_name not in REGRESSION_METRIC_NAME_TO_OBJECTIVE:
        raise ValueError(
            f"Metric '{metric_name}' is not supported. Supported metrics are: "
            f"{list(REGRESSION_METRIC_NAME_TO_OBJECTIVE.keys())}"
        )
    return REGRESSION_METRIC_NAME_TO_OBJECTIVE[metric_name](
        raw_space_bardist, logits, y_true
    )


def get_tuning_splits(
    X: np.ndarray,
    y: np.ndarray,
    holdout_frac: float,
    n_splits: int = 1,
    random_state: int | np.random.RandomState | np.random.Generator | None = 0,
    task_type: Literal["classifier", "regressor"] = "classifier",
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Get tuning split(s) for the given configuration.

    Splits are stratified for classification and shuffled-but-unstratified for
    regression, where the continuous target cannot be stratified on.

    Args:
        X: The input data of shape [n_samples, n_features].
        y: The target labels of shape [n_samples].
        holdout_frac: The percentage of the data to hold out for tuning.
        n_splits: Number of random splits to generate.
        random_state: The random state to use for the split(s).
        task_type: Whether the splits are for a classifier or a regressor.

    Returns:
        Returns a list of splits as tuples of
        (X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh).
        Shape suffixes: Nt=num train samples, F=num features, Nh=num holdout samples.
    """
    # We want to use (Stratified)KFold to ensure that no train samples are used twice.
    # Therefore, we have to invert the holdout_frac to get the number of folds to
    # use for (Stratified)KFold. Round holdout_frac to 2 digits to avoid needing
    # more than 100 folds
    rounded_holdout_frac = round(holdout_frac, 2)
    n_folds = max(2, round(1 / rounded_holdout_frac))

    if isinstance(random_state, np.random.Generator):
        # (Stratified)KFold does not accept np.random.Generator.
        random_state, _ = infer_random_state(random_state)

    # A continuous target has no classes to stratify on, so regression uses a plain
    # shuffled K-fold. This keeps the guarantee that no sample is held out twice.
    splitter_cls = KFold if task_type == "regressor" else StratifiedKFold
    splitter = splitter_cls(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, (train_indices, holdout_indices) in enumerate(splitter.split(X, y)):
        if i >= n_splits:
            break
        X_train_NtF = X[train_indices]
        X_holdout_NhF = X[holdout_indices]
        y_train_Nt = y[train_indices]
        y_holdout_Nh = y[holdout_indices]
        splits.append((X_train_NtF, X_holdout_NhF, y_train_Nt, y_holdout_Nh))

    return splits


def find_optimal_classification_thresholds(
    metric_name: ClassifierEvalMetrics,
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

    # TODO: vectorize this loop loop and the one in
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
    metric_name: ClassifierEvalMetrics,
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
    thresholds = np.linspace(0.01, 0.99, 198)
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
    min_loss_index = int(np.argmin(losses))
    start = min_loss_index
    while start - 1 >= 0 and close_mask[start - 1]:
        start -= 1
    end = min_loss_index
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
    temperatures = get_tuning_temperatures()
    best_log_loss = float("inf")
    best_temperature = current_default_temperature

    # TODO: think about vectorizing this loop.
    for temperature in temperatures:
        probas = logits_to_probabilities_fn(raw_logits, temperature)
        # The holdout may lack a rare class, so labels cannot be inferred from y_true.
        current_log_loss = log_loss(
            y_true=y_true,
            y_pred=probas,
            labels=np.arange(probas.shape[-1]),
        )

        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_temperature = temperature

    return best_temperature


def find_regression_optimal_temperature(
    holdout_folds: list[tuple[torch.Tensor, FullSupportBarDistribution, torch.Tensor]],
    metric_name: RegressorEvalMetrics,
    current_default_temperature: float,
) -> float:
    """Finds the temperature to apply to the aggregated ensemble distribution.

    The temperature acts after ensemble aggregation, as `log -> /T -> softmax`.
    Because the bar distribution applies `log_softmax` itself, dividing the
    already-logged aggregated logits by T is exactly that operation. `T > 1` widens
    the predicted distribution and `T < 1` sharpens it.

    Each fold is kept separate because each fold's regressor normalises the target
    with its own mean and standard deviation, so each fold's bar distribution has its
    own borders.

    Args:
        holdout_folds: One `(logits, raw_space_bardist, y_true)` triple per fold,
            where `logits` has shape [n_holdout, n_buckets] and `y_true` shape
            [n_holdout]. Each triple must be internally consistent: the targets in
            the units of that bar distribution's borders, and all three on the same
            device.
        metric_name: The metric to minimize.
        current_default_temperature: The temperature to fall back to when the sweep
            has nothing usable to choose from.

    Returns:
        The temperature that minimizes the metric over the pooled holdout rows.
    """
    # `FullSupportBarDistribution.forward` accumulates into its `losses_per_bucket`
    # buffer on every call, so sweep on copies and leave the callers' bar
    # distributions clean.
    folds = [
        (logits, copy.deepcopy(raw_space_bardist), y_true)
        for logits, raw_space_bardist, y_true in holdout_folds
        if logits.shape[0] > 0
    ]
    if not folds:
        return current_default_temperature

    n_holdout_total = sum(logits.shape[0] for logits, _, _ in folds)
    temperatures = get_tuning_temperatures()
    best_loss = float("inf")
    best_temperature = current_default_temperature

    # TODO: think about vectorizing this loop.
    with torch.no_grad():
        for temperature in temperatures:
            current_loss = (
                sum(
                    logits.shape[0]
                    * float(
                        compute_regression_metric_to_minimize(
                            metric_name=metric_name,
                            raw_space_bardist=raw_space_bardist,
                            logits=logits / temperature,
                            y_true=y_true,
                        ).mean()
                    )
                    for logits, raw_space_bardist, y_true in folds
                )
                / n_holdout_total
            )

            if np.isfinite(current_loss) and current_loss < best_loss:
                best_loss = current_loss
                best_temperature = float(temperature)

    return best_temperature


def get_tuning_temperatures() -> np.ndarray:
    """Gets the candidate softmax temperatures to sweep when calibrating.

    Shared by the classification and regression searches.
    A temperature above 1 flattens the predicted distribution and one
    below 1 sharpens it. Contains the default value 1.0

    Returns:
        The candidate temperatures, in ascending order, in steps of 0.01.
    """
    return np.arange(60, 141) / 100.0


def get_default_tuning_holdout_frac(n_samples: int) -> float:
    """Gets the default tuning holdout percentage based on a heuristic.

    We aim to tradeoff between computational cost and accuracy.
    """
    n_samples_to_holdout_frac = {
        2_000: 0.1,
        5_000: 0.2,
        10_000: 0.2,
        20_000: 0.2,
        50_000: 0.3,
    }
    for n_samples_threshold, frac in n_samples_to_holdout_frac.items():
        if n_samples <= n_samples_threshold:
            return frac
    return 0.2


def get_default_tuning_n_folds(n_samples: int) -> int:
    """Gets the default tuning holdout number of splits based on a heuristic.

    We aim to tradeoff between computational cost and accuracy.
    """
    n_samples_to_n_folds = {
        2_000: 10,
        5_000: 5,
        10_000: 3,
        20_000: 2,
        50_000: 1,
    }
    for n_samples_threshold, n_folds in n_samples_to_n_folds.items():
        if n_samples <= n_samples_threshold:
            return n_folds
    return 1


def get_tuning_feature_flags(tuning_config: TuningConfig) -> dict[str, bool]:
    """Gets the tuning feature flags of a tuning configuration by field name.

    Every tuning feature is exposed as a boolean field (e.g.
    `calibrate_temperature`); the remaining fields configure how tuning is run
    rather than whether it happens at all. Reading them off the dataclass keeps
    this task-agnostic, so a config without `tune_decision_thresholds` works.

    Args:
        tuning_config: The tuning configuration to inspect.

    Returns:
        A mapping from feature flag name to whether it is enabled, in field
        declaration order.
    """
    return {
        field.name: value
        for field in dataclasses.fields(tuning_config)
        if isinstance(value := getattr(tuning_config, field.name), bool)
    }


def resolve_tuning_config(
    tuning_config: dict | TuningConfig | None,
    num_samples: int,
    config_cls: type[TuningConfig] = ClassifierTuningConfig,
) -> TuningConfig | None:
    """Resolves the tuning configuration by checking if tuning is needed,
    resolving 'auto' values for holdout parameters, and building the appropriate
    type of tuning configuration if the input is a dict.

    Args:
        tuning_config: The tuning configuration to use. If a dict is provided,
            it is used as the keyword arguments of `config_cls`.
        num_samples: The number of samples in the training data.
        config_cls: The tuning configuration class to build a dict input into,
            e.g. `ClassifierTuningConfig` or `RegressorTuningConfig`. Ignored
            when `tuning_config` is already a config instance.

    Returns:
        The resolved tuning configuration or None if no tuning is needed. It is
        the same instance type as the input (or `config_cls` for a dict input);
        callers needing task-specific fields should narrow it.
    """
    if tuning_config is None:
        return None

    tuning_config = (
        config_cls(**tuning_config)
        if isinstance(tuning_config, dict)
        else tuning_config
    )

    feature_flags = get_tuning_feature_flags(tuning_config)
    if not any(feature_flags.values()):
        enable_options = " or ".join(f"`{name}=True`" for name in feature_flags)
        warnings.warn(
            "You specified a tuning configuration but no tuning features were enabled. "
            f"Set {enable_options} to enable tuning.",
            UserWarning,
            stacklevel=3,
        )
        return None

    if num_samples < MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING:
        warnings.warn(
            f"You have `{num_samples}` samples in the training data and specified "
            "a tuning configuration. "
            "We recommend tuning only for datasets with more than "
            f"{MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING} samples. ",
            UserWarning,
            stacklevel=3,
        )

    return tuning_config.resolve(num_samples=num_samples)
