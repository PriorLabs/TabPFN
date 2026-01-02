"""A TabPFN regressor that finetunes the underlying model for a single task.

This module provides the FinetunedTabPFNRegressor class, which wraps TabPFN
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
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from tabpfn import TabPFNRegressor
from tabpfn.finetuning.finetuned_base import EvalResult, FinetunedTabPFNBase
from tabpfn.finetuning.train_util import clone_model_for_evaluation
from tabpfn.model_loading import get_n_out

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tabpfn.finetuning.data_util import RegressorBatch


def compute_regression_loss(
    *,
    predictions_BLQ: torch.Tensor,
    targets_BQ: torch.Tensor,
    bardist_loss_fn: Any,
    mse_loss_weight: float,
    mse_loss_clip: float | None,
) -> torch.Tensor:
    """Compute the regression training loss from bar distribution and auxiliary terms.

    Shapes suffixes:
        B=batch * estimators, L=logits, Q=n_queries.

    Args:
        predictions_BLQ: Bar distribution logits of shape (B*E, Q, L).
        targets_BQ: Regression targets of shape (B*E, Q).
        bardist_loss_fn: Bar distribution loss function (callable) which also
            exposes a `.mean()` method for converting bar logits to mean
            predictions.
        mse_loss_weight: Weight for an auxiliary MSE term. Set to 0.0 to disable.
        mse_loss_clip: Optional upper bound for the auxiliary MSE term.

    Returns:
        A scalar loss tensor.
    """
    losses = bardist_loss_fn(predictions_BLQ, targets_BQ)

    if mse_loss_weight > 0.0:
        predictions_mean = bardist_loss_fn.mean(predictions_BLQ)
        mse_aux_loss = ((predictions_mean - targets_BQ) ** 2).mean()
        if mse_loss_clip is not None:
            mse_aux_loss = mse_aux_loss.clamp(max=mse_loss_clip)
        losses = losses + mse_loss_weight * mse_aux_loss

    return losses.mean()


class FinetunedTabPFNRegressor(FinetunedTabPFNBase, RegressorMixin):
    """A scikit-learn compatible wrapper for fine-tuning the TabPFNRegressor.

    This class encapsulates the fine-tuning loop, allowing you to fine-tune
    TabPFN on a specific dataset using the familiar .fit() and .predict() API.

    Args:
        FinetunedTabPFNRegressor specific arguments:

        mse_loss_weight: Weight for an auxiliary MSE loss term added to the
            bar distribution loss. Set to 0.0 to disable. Defaults to 8.0.
        mse_loss_clip: Optional upper bound for the auxiliary MSE loss term.
            If None, no clipping is applied. Defaults to None.
        extra_regressor_kwargs: Additional keyword arguments to pass to the
            underlying `TabPFNRegressor`, such as `n_estimators`.

        See FinetunedTabPFNBase for details on common arguments.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        device: str = "cuda",
        epochs: int = 30,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.001,
        validation_split_ratio: float = 0.1,
        n_finetune_ctx_plus_query_samples: int = 20_000,
        finetune_ctx_query_split_ratio: float = 0.2,
        n_inference_subsample_samples: int = 50_000,
        meta_batch_size: int = 1,
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
        use_activation_checkpointing: bool = False,
        save_checkpoint_interval: int | None = 10,
        extra_regressor_kwargs: dict[str, Any] | None = None,
        mse_loss_weight: float = 8.0,
        mse_loss_clip: float | None = None,
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
            meta_batch_size=meta_batch_size,
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
        self.mse_loss_weight = mse_loss_weight
        self.mse_loss_clip = mse_loss_clip
        self.regressor_kwargs = extra_regressor_kwargs or {}
        # Store for sklearn get_params compatibility
        self.extra_regressor_kwargs = extra_regressor_kwargs

    @property
    @override
    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the regressor-specific kwargs."""
        return self.regressor_kwargs

    @property
    @override
    def _model_type(self) -> Literal["classifier", "regressor"]:
        """Return the model type string."""
        return "regressor"

    @property
    @override
    def _metric_name(self) -> str:
        """Return the name of the primary metric."""
        return "MSE"

    @override
    def _create_estimator(self, config: dict[str, Any]) -> TabPFNRegressor:
        """Create the TabPFNRegressor with the given config."""
        return TabPFNRegressor(
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )

    @override
    def _setup_estimator(self) -> None:
        """No additional setup needed for regressor at creation time."""

    @override
    def _get_train_val_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data without stratification for regression."""
        return train_test_split(  # type: ignore[return-value]
            X,
            y,
            test_size=self.validation_split_ratio,
            random_state=self.random_state,
        )

    @override
    def _setup_batch(self, batch: RegressorBatch) -> None:  # type: ignore[override]
        """Set up bar distribution for this batch."""
        self.finetuned_estimator_.raw_space_bardist_ = batch.raw_space_bardist
        self.finetuned_estimator_.bardist_ = batch.znorm_space_bardist
        self._bardist_loss = batch.znorm_space_bardist

    @override
    def _forward_with_loss(self, batch: RegressorBatch) -> torch.Tensor:  # type: ignore[override]
        """Perform forward pass and compute bar distribution loss with optional MSE.

        Args:
            batch: The RegressorBatch containing preprocessed context and query
                data plus bar distribution information.

        Returns:
            The computed loss tensor (bar distribution + optional MSE auxiliary).
        """
        X_query_batch = batch.X_query
        y_query_batch = batch.y_query
        bardist_loss_fn = self._bardist_loss

        _, per_estim_logits, _ = self.finetuned_estimator_.forward(X_query_batch)
        # per_estim_logits is a list (per estimator) of tensors with shape [Q, B(=1), L]

        # shape suffix: Q=n_queries, B=batch(=1), E=estimators, L=logits
        predictions_QBEL = torch.stack(per_estim_logits, dim=2)

        Q, B, E, L = predictions_QBEL.shape
        num_bars = get_n_out(self.finetuned_estimator_.configs_[0], bardist_loss_fn)
        assert y_query_batch.shape[1] == Q
        assert B == 1
        assert self.n_estimators_finetune == E
        assert num_bars == L

        # Reshape for bar distribution loss: treat estimator dim as batch dim
        # permute to shape (B, E, Q, L) then reshape to (B*E, Q, L)
        predictions_BLQ = predictions_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L)

        targets_BQ = y_query_batch.repeat(B * self.n_estimators_finetune, 1).to(
            self.device
        )

        return compute_regression_loss(
            predictions_BLQ=predictions_BLQ,
            targets_BQ=targets_BQ,
            bardist_loss_fn=bardist_loss_fn,
            mse_loss_weight=self.mse_loss_weight,
            mse_loss_clip=self.mse_loss_clip,
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
        """Evaluate the regressor using MSE."""
        eval_regressor = clone_model_for_evaluation(
            self.finetuned_estimator_,
            eval_config,
            TabPFNRegressor,
        )
        eval_regressor.fit(X_train, y_train)

        try:
            predictions = eval_regressor.predict(X_val)  # type: ignore
            mse = mean_squared_error(y_val, predictions)
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning(f"An error occurred during evaluation: {e}")
            mse = np.nan

        return EvalResult(primary=mse)  # pyright: ignore[reportArgumentType]

    @override
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current MSE is better (lower) than best."""
        return current < best - self.min_delta

    @override
    def _get_initial_best_metric(self) -> float:
        """Return inf for minimization."""
        return np.inf

    @override
    def _get_checkpoint_metrics(self, eval_result: EvalResult) -> dict[str, float]:
        """Return metrics for checkpoint saving."""
        return {"mse": eval_result.primary}

    @override
    def _log_epoch_evaluation(
        self, epoch: int, eval_result: EvalResult, mean_train_loss: float | None
    ) -> None:
        """Log evaluation results for regression."""
        logger.info(
            f"📊 Epoch {epoch + 1} Evaluation | Val MSE: {eval_result.primary:.4f}, "
            f"Train Loss: {mean_train_loss:.4f}"
        )

    @override
    def _setup_inference_model(
        self, final_inference_eval_config: dict[str, Any]
    ) -> None:
        """Set up the final inference regressor."""
        finetuned_inference_regressor = clone_model_for_evaluation(
            self.finetuned_estimator_,
            final_inference_eval_config,
            TabPFNRegressor,
        )
        self.finetuned_inference_regressor_ = finetuned_inference_regressor
        self.finetuned_inference_regressor_.fit_mode = "fit_preprocessors"  # type: ignore
        self.finetuned_inference_regressor_.fit(self.X_, self.y_)  # type: ignore

    @override
    def fit(
        self, X: np.ndarray, y: np.ndarray, output_dir: Path | None = None
    ) -> FinetunedTabPFNRegressor:
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

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X.

        Args:
            X: The input samples of shape (n_samples, n_features).

        Returns:
            The predicted target values with shape (n_samples,).
        """
        check_is_fitted(self)

        return self.finetuned_inference_regressor_.predict(X)  # type: ignore
