"""A TabPFN classifier that finetunes the underlying model for a single task.

This module provides the FinetunedTabPFNClassifier class, which wraps TabPFN
and allows fine-tuning on a specific dataset using the familiar scikit-learn
.fit() and .predict() API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import torch
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils.validation import check_is_fitted

from tabpfn import TabPFNClassifier
from tabpfn.finetuning.finetuned_base import EvalResult, FinetunedTabPFNBase
from tabpfn.finetuning.train_util import clone_model_for_evaluation

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tabpfn.constants import XType, YType
    from tabpfn.finetuning.data_util import ClassifierBatch


def compute_classification_loss(
    *,
    predictions_BLQ: torch.Tensor,
    targets_BQ: torch.Tensor,
) -> torch.Tensor:
    """Compute the cross-entropy training loss.

    Shapes suffixes:
        B=batch * estimators, L=logits, Q=n_queries.

    Args:
        predictions_BLQ: Raw logits of shape (B*E, L, Q).
        targets_BQ: Integer class targets of shape (B*E, Q).

    Returns:
        A scalar loss tensor.
    """
    return torch.nn.functional.cross_entropy(predictions_BLQ, targets_BQ)


class FinetunedTabPFNClassifier(FinetunedTabPFNBase, ClassifierMixin):
    """A scikit-learn compatible wrapper for fine-tuning the TabPFNClassifier.

    This class encapsulates the fine-tuning loop, allowing you to fine-tune
    TabPFN on a specific dataset using the familiar .fit() and .predict() API.

    Args:
        device: The device to run the model on. Defaults to "cuda".
        epochs: The total number of passes through the fine-tuning data.
            Defaults to 30.
        learning_rate: The learning rate for the AdamW optimizer. A small value
            is crucial for stable fine-tuning. Defaults to 1e-5.
        weight_decay: The weight decay for the AdamW optimizer. Defaults to 0.01.
        validation_split_ratio: Fraction of the original training data reserved
            as a validation set for early stopping and monitoring. Defaults to 0.1.
        n_finetune_ctx_plus_query_samples: The total number of samples per
            meta-dataset during fine-tuning (context plus query) before applying
            the `finetune_ctx_query_split_ratio`. Defaults to 10_000.
        finetune_ctx_query_split_ratio: The proportion of each fine-tuning
            meta-dataset to use as query samples for calculating the loss. The
            remainder is used as context. Defaults to 0.2.
        n_inference_subsample_samples: The total number of subsampled training
            samples per estimator during validation and final inference.
            Defaults to 50_000.
        random_state: Seed for reproducibility of data splitting and model
            initialization. Defaults to 0.
        early_stopping: Whether to use early stopping based on validation
            performance. Defaults to True.
        early_stopping_patience: Number of epochs to wait for improvement before
            early stopping. Defaults to 8.
        min_delta: Minimum change in metric to be considered as an improvement.
            Defaults to 1e-4.
        grad_clip_value: Maximum norm for gradient clipping. If None, gradient
            clipping is disabled. Gradient clipping helps stabilize training by
            preventing exploding gradients. Defaults to 1.0.
        use_lr_scheduler: Whether to use a learning rate scheduler (linear warmup
            with optional cosine decay) during fine-tuning. Defaults to True.
        lr_warmup_only: If True, only performs linear warmup to the base learning
            rate and then keeps it constant. If False, applies cosine decay after
            warmup. Defaults to False.
        n_estimators_finetune: If set, overrides `n_estimators` of the underlying
            estimator only during fine-tuning to control the number of
            estimators (ensemble size) used in the training loop. If None, the
            value from `kwargs` or the estimator default is used.
            Defaults to 2.
        n_estimators_validation: If set, overrides `n_estimators` only for
            validation-time evaluation during fine-tuning (early-stopping /
            monitoring). If None, the value from `kwargs` or the
            estimator default is used. Defaults to 2.
        n_estimators_final_inference: If set, overrides `n_estimators` only for
            the final fitted inference model that is used after fine-tuning. If
            None, the value from `kwargs` or the estimator default is used.
            Defaults to 8.
        use_activation_checkpointing: Whether to use activation checkpointing to
            reduce memory usage. Defaults to True.
        save_checkpoint_interval: Number of epochs between checkpoint saves. This
            only has an effect if `output_dir` is provided during the `fit()` call.
            If None, no intermediate checkpoints are saved. The best model checkpoint
            is always saved regardless of this setting. Defaults to 10.

        FinetunedTabPFNClassifier specific arguments:

        extra_classifier_kwargs: Additional keyword arguments to pass to the
            underlying `TabPFNClassifier`, such as `n_estimators`.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        device: str = "cuda",
        epochs: int = 30,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        validation_split_ratio: float = 0.1,
        n_finetune_ctx_plus_query_samples: int = 10_000,
        finetune_ctx_query_split_ratio: float = 0.2,
        n_inference_subsample_samples: int = 50_000,
        random_state: int = 0,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        min_delta: float = 1e-4,
        grad_clip_value: float | None = 1.0,
        use_lr_scheduler: bool = True,
        lr_warmup_only: bool = False,
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_final_inference: int = 8,
        use_activation_checkpointing: bool = True,
        save_checkpoint_interval: int | None = 10,
        extra_classifier_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_split_ratio=validation_split_ratio,
            n_finetune_ctx_plus_query_samples=n_finetune_ctx_plus_query_samples,
            finetune_ctx_query_split_ratio=finetune_ctx_query_split_ratio,
            n_inference_subsample_samples=n_inference_subsample_samples,
            random_state=random_state,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            grad_clip_value=grad_clip_value,
            use_lr_scheduler=use_lr_scheduler,
            lr_warmup_only=lr_warmup_only,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_final_inference=n_estimators_final_inference,
            use_activation_checkpointing=use_activation_checkpointing,
            save_checkpoint_interval=save_checkpoint_interval,
        )
        self.extra_classifier_kwargs = extra_classifier_kwargs

    @property
    @override
    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the classifier-specific kwargs."""
        return self.extra_classifier_kwargs or {}

    @property
    @override
    def _model_type(self) -> Literal["classifier", "regressor"]:
        """Return the model type string."""
        return "classifier"

    @property
    @override
    def _metric_name(self) -> str:
        """Return the name of the primary metric."""
        return "ROC AUC"

    @override
    def _create_estimator(self, config: dict[str, Any]) -> TabPFNClassifier:
        """Create the TabPFNClassifier with the given config."""
        return TabPFNClassifier(
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )

    @override
    def _setup_estimator(self) -> None:
        """Set up softmax temperature after estimator creation."""
        self.finetuned_estimator_.softmax_temperature_ = (
            self.finetuned_estimator_.softmax_temperature
        )

    @override
    def _setup_batch(self, batch: ClassifierBatch) -> None:  # type: ignore[override]
        """No batch-specific setup needed for classifier."""

    @override
    def _should_skip_batch(self, batch: ClassifierBatch) -> bool:  # type: ignore[override]
        """Check if the batch should be skipped."""
        ctx_unique = torch.unique(
            torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_context])
        )
        qry_unique = torch.unique(
            torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_query])
        )

        query_in_context = torch.isin(qry_unique, ctx_unique, assume_unique=True)
        if not bool(query_in_context.all()):
            missing_labels = qry_unique[~query_in_context].detach().cpu().numpy()
            context_labels = ctx_unique.detach().cpu().numpy()
            logger.warning(
                "Skipping batch: query labels %s are not a subset of context labels %s",
                missing_labels,
                context_labels,
            )
            return True
        return False

    @override
    def _forward_with_loss(self, batch: ClassifierBatch) -> torch.Tensor:  # type: ignore[override]
        """Perform forward pass and compute and return cross-entropy loss.

        Args:
            batch: The ClassifierBatch containing preprocessed context and
                query data.

        Returns:
            The computed cross-entropy loss tensor.
        """
        X_query_batch = batch.X_query
        y_query_batch = batch.y_query

        # shape suffix: Q=n_queries, B=batch(=1), E=estimators, L=logits
        predictions_QBEL = self.finetuned_estimator_.forward(
            X_query_batch,
            return_raw_logits=True,
        )

        Q, B, E, L = predictions_QBEL.shape
        assert y_query_batch.shape[1] == Q
        assert B == 1
        assert self.n_estimators_finetune == E
        assert self.finetuned_estimator_.n_classes_ == L

        # Reshape for CE loss: treat estimator dim as batch dim
        # permute to shape (B, E, L, Q) then reshape to (B*E, L, Q)
        predictions_BLQ = predictions_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
        targets_BQ = y_query_batch.repeat(B * self.n_estimators_finetune, 1).to(
            self.device
        )

        return compute_classification_loss(
            predictions_BLQ=predictions_BLQ,
            targets_BQ=targets_BQ,
        )

    @override
    def _evaluate_model(
        self,
        eval_config: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> EvalResult:
        """Evaluate the classifier using ROC AUC and log loss."""
        eval_classifier = clone_model_for_evaluation(
            self.finetuned_estimator_,
            eval_config,
            TabPFNClassifier,
        )
        eval_classifier.fit(X_train, y_train)

        try:
            probabilities = eval_classifier.predict_proba(X_val)  # type: ignore
            if probabilities.shape[1] > 2:
                roc_auc = roc_auc_score(y_val, probabilities, multi_class="ovr")
            else:
                roc_auc = roc_auc_score(y_val, probabilities[:, 1])
            log_loss_score = log_loss(y_val, probabilities)
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning(f"An error occurred during evaluation: {e}")
            roc_auc, log_loss_score = np.nan, np.nan

        return EvalResult(
            primary=roc_auc,  # pyright: ignore[reportArgumentType]
            secondary={"log_loss": log_loss_score},
        )

    @override
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current ROC AUC is better (higher) than best."""
        return current > best + self.min_delta

    @override
    def _get_initial_best_metric(self) -> float:
        """Return -inf for maximization."""
        return -np.inf

    @override
    def _get_checkpoint_metrics(self, eval_result: EvalResult) -> dict[str, float]:
        """Return metrics for checkpoint saving."""
        return {
            "roc_auc": eval_result.primary,
            "log_loss": eval_result.secondary.get("log_loss", np.nan),
        }

    @override
    def _log_epoch_evaluation(
        self, epoch: int, eval_result: EvalResult, mean_train_loss: float | None
    ) -> None:
        """Log evaluation results for classification."""
        log_loss_score = eval_result.secondary.get("log_loss", np.nan)
        logger.info(
            f"📊 Epoch {epoch + 1} Evaluation | Val ROC: {eval_result.primary:.4f}, "
            f"Val Log Loss: {log_loss_score:.4f}, Train Loss: {mean_train_loss:.4f}"
        )

    @override
    def _setup_inference_model(
        self, final_inference_eval_config: dict[str, Any]
    ) -> None:
        """Set up the final inference classifier."""
        finetuned_inference_classifier = clone_model_for_evaluation(
            self.finetuned_estimator_,
            final_inference_eval_config,
            TabPFNClassifier,
        )
        self.finetuned_inference_classifier_ = finetuned_inference_classifier
        self.finetuned_inference_classifier_.fit_mode = "fit_preprocessors"  # type: ignore
        self.finetuned_inference_classifier_.fit(self.X_, self.y_)  # type: ignore

    @override
    def fit(
        self, X: XType, y: YType, output_dir: Path | None = None
    ) -> FinetunedTabPFNClassifier:
        """Fine-tune the TabPFN model on the provided training data.

        Args:
            X: The training input samples of shape (n_samples, n_features).
            y: The target values of shape (n_samples,).
            output_dir: Directory path for saving checkpoints. If None, no
                checkpointing is performed and progress will be lost if
                training is interrupted.

        Returns:
            The fitted instance itself.
        """
        super().fit(X, y, output_dir)
        return self

    def predict_proba(self, X: XType) -> np.ndarray:
        """Predict class probabilities for X.

        Args:
            X: The input samples of shape (n_samples, n_features).

        Returns:
            The class probabilities of the input samples with shape
            (n_samples, n_classes).
        """
        check_is_fitted(self)

        return self.finetuned_inference_classifier_.predict_proba(X)  # type: ignore

    @override
    def predict(self, X: XType) -> np.ndarray:
        """Predict the class for X.

        Args:
            X: The input samples of shape (n_samples, n_features).

        Returns:
            The predicted classes with shape (n_samples,).
        """
        check_is_fitted(self)

        return self.finetuned_inference_classifier_.predict(X)  # type: ignore
