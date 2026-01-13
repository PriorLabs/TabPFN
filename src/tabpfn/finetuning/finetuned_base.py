"""Abstract base class for fine-tuning TabPFN models.

This module provides the FinetunedTabPFNBase class, which contains shared
functionality for fine-tuning TabPFN on a specific dataset using the familiar
scikit-learn .fit() and .predict() API.
"""

from __future__ import annotations

import copy
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tabpfn.finetuning._torch_compat import GradScaler, autocast, sdpa_kernel_context
from tabpfn.finetuning.data_util import (
    ClassifierBatch,
    RegressorBatch,
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)
from tabpfn.finetuning.train_util import (
    get_and_init_optimizer,
    get_checkpoint_path_and_epoch_from_output_dir,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
)
from tabpfn.utils import infer_devices
from tabpfn.validation import ensure_compatible_fit_inputs_sklearn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tabpfn.constants import XType, YType

# Currently, we only support a batch size of 1 for finetuning.
META_BATCH_SIZE = 1

# Hard limit on the number of samples to use for validation.
# This is used to avoid spending too much time on validation
# and prevent OOM issues for very large datasets.
MAX_VALIDATION_SAMPLES = 50_000


@dataclass
class EvalResult:
    """Container for evaluation results.

    Attributes:
        primary: The primary metric used for early stopping decisions.
        secondary: Additional metrics for logging purposes only.
    """

    primary: float
    secondary: dict[str, float] = field(default_factory=dict)


class FinetunedTabPFNBase(BaseEstimator, ABC):
    """Abstract base class for fine-tuning TabPFN models.

    This class encapsulates the shared fine-tuning logic, allowing you to
    fine-tune TabPFN on a specific dataset using the familiar .fit() and
    .predict() API.

    Args:
        device: The device to run the model on. Defaults to "cuda".
        epochs: The total number of passes through the fine-tuning data.
            Defaults to 30.
        time_limit: Time limit in seconds for fine-tuning.
            If None, no time limit is applied. Defaults to None.
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
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        device: str = "cuda",
        epochs: int = 30,
        time_limit: int | None = None,
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
    ):
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.time_limit = time_limit
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_split_ratio = validation_split_ratio
        self.n_finetune_ctx_plus_query_samples = n_finetune_ctx_plus_query_samples
        self.finetune_ctx_query_split_ratio = finetune_ctx_query_split_ratio
        self.n_inference_subsample_samples = n_inference_subsample_samples
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.grad_clip_value = grad_clip_value
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_warmup_only = lr_warmup_only
        self.n_estimators_finetune = n_estimators_finetune
        self.n_estimators_validation = n_estimators_validation
        self.n_estimators_final_inference = n_estimators_final_inference
        self.use_activation_checkpointing = use_activation_checkpointing
        self.save_checkpoint_interval = save_checkpoint_interval
        self.meta_batch_size = META_BATCH_SIZE

    def _build_estimator_config(
        self,
        base_config: dict[str, Any],
        n_estimators_override: int | None,
    ) -> dict[str, Any]:
        """Return a deep-copy of base_config with an optional n_estimators override."""
        config = copy.deepcopy(base_config)
        if n_estimators_override is not None:
            config["n_estimators"] = n_estimators_override
        return config

    def _build_eval_config(
        self,
        base_config: dict[str, Any],
        n_estimators_override: int | None,
    ) -> dict[str, Any]:
        """Return eval config with n_estimators override and subsample setting."""
        config = self._build_estimator_config(base_config, n_estimators_override)
        existing = dict(config.get("inference_config", {}) or {})
        existing["SUBSAMPLE_SAMPLES"] = self.n_inference_subsample_samples
        config["inference_config"] = existing
        return config

    @property
    @abstractmethod
    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the task-specific estimator kwargs."""
        ...

    @property
    @abstractmethod
    def _model_type(self) -> Literal["classifier", "regressor"]:
        """Return the model type string ('classifier' or 'regressor')."""
        ...

    @property
    @abstractmethod
    def _metric_name(self) -> str:
        """Return the name of the primary metric for logging."""
        ...

    @abstractmethod
    def _create_estimator(self, config: dict[str, Any]) -> Any:
        """Create and return the underlying TabPFN estimator with the given config."""
        ...

    @abstractmethod
    def _setup_estimator(self) -> None:
        """Perform any task-specific setup after estimator creation."""
        ...

    @abstractmethod
    def _setup_batch(self, batch: ClassifierBatch | RegressorBatch) -> None:
        """Perform any batch-specific setup before the forward pass."""
        ...

    @abstractmethod
    def _should_skip_batch(self, batch: ClassifierBatch | RegressorBatch) -> bool:
        """Check if the batch should be skipped."""
        ...

    @abstractmethod
    def _forward_with_loss(
        self,
        batch: ClassifierBatch | RegressorBatch,
    ) -> torch.Tensor:
        """Perform forward pass and compute loss for the given batch.

        Args:
            batch: The batch tuple from the dataloader.

        Returns:
            The computed loss tensor.
        """
        ...

    @abstractmethod
    def _evaluate_model(
        self,
        eval_config: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> EvalResult:
        """Evaluate the model on validation data and return metrics.

        Args:
            eval_config: Configuration dictionary for the evaluation estimator.
            X_train: Training input samples.
            y_train: Training target values.
            X_val: Validation input samples.
            y_val: Validation target values.

        Returns:
            EvalResult with primary metric for early stopping and secondary
            metrics for logging.
        """
        ...

    @abstractmethod
    def _is_improvement(self, current: float, best: float) -> bool:
        """Return True if current metric is an improvement over best.

        Args:
            current: The current metric value.
            best: The best metric value seen so far.

        Returns:
            True if current is better than best (accounting for min_delta).
        """
        ...

    @abstractmethod
    def _get_initial_best_metric(self) -> float:
        """Return initial 'best' metric (inf for min, -inf for max)."""
        ...

    @abstractmethod
    def _get_checkpoint_metrics(self, eval_result: EvalResult) -> dict[str, float]:
        """Return the metrics dict to save in checkpoints."""
        ...

    @abstractmethod
    def _log_epoch_evaluation(
        self, epoch: int, eval_result: EvalResult, mean_train_loss: float | None
    ) -> None:
        """Log the evaluation results for the current epoch."""
        ...

    @abstractmethod
    def _setup_inference_model(
        self, final_inference_eval_config: dict[str, Any]
    ) -> None:
        """Set up the final inference model after fine-tuning completes."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X."""
        ...

    def _get_train_val_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation sets with task-specific options."""
        n_samples = len(y)
        test_size = int(n_samples * self.validation_split_ratio)

        if test_size > MAX_VALIDATION_SAMPLES:
            warnings.warn(
                f"Validation set size would be {test_size:,} samples "
                f"based on validation_split_ratio="
                f"{self.validation_split_ratio:.2f}, but limiting to "
                f"{MAX_VALIDATION_SAMPLES:,} samples to avoid excessive "
                f"validation time and memory usage.",
                UserWarning,
                stacklevel=3,
            )
            test_size = MAX_VALIDATION_SAMPLES

        # test_size should be greater or equal to the number of classes
        if self._model_type == "classifier":
            n_classes = len(np.unique(y))
            test_size = max(test_size, n_classes)

        return train_test_split(  # type: ignore[return-value]
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if self._model_type == "classifier" else None,
        )

    @abstractmethod
    def _get_valid_finetuning_query_size(
        self, *, query_size: int, y_train: np.ndarray | None
    ) -> int:
        """Calculate a valid finetuning query size."""
        ...

    def fit(
        self,
        X: XType,
        y: YType,
        X_val: XType | None = None,
        y_val: YType | None = None,
        output_dir: Path | None = None,
    ) -> FinetunedTabPFNBase:
        """Fine-tune the TabPFN model on the provided training data.

        Args:
            X: The training input samples of shape (n_samples, n_features).
            y: The target values of shape (n_samples,).
            X_val: Optional validation input samples.
            y_val: Optional validation target values.
            output_dir: Directory path for saving checkpoints. If None, no
                checkpointing is performed and progress will be lost if
                training is interrupted.

        Returns:
            The fitted instance itself.
        """
        if output_dir is None:
            warnings.warn(
                "`output_dir` is not set. This means no checkpointing will be done and "
                "all progress will be lost if the training is interrupted.",
                UserWarning,
                stacklevel=2,
            )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        return self._fit(X=X, y=y, X_val=X_val, y_val=y_val, output_dir=output_dir)

    def _fit(  # noqa: C901,PLR0912
        self,
        X: XType,
        y: YType,
        X_val: XType | None = None,
        y_val: YType | None = None,
        output_dir: Path | None = None,
    ) -> FinetunedTabPFNBase:
        """Internal implementation of fit that runs the finetuning loop."""
        # Store the original training size for checkpoint naming
        train_size = X.shape[0]
        start_time = time.monotonic()

        _estimator_kwargs = copy.deepcopy(self._estimator_kwargs)
        model_path = _estimator_kwargs.pop("model_path", None)
        inference_config = copy.deepcopy(_estimator_kwargs.get("inference_config", {}))
        base_estimator_config: dict[str, Any] = {
            **_estimator_kwargs,
            "ignore_pretraining_limits": True,
            "device": self.device,
            "random_state": self.random_state,
            "inference_config": inference_config,
        }

        # Config used for the finetuning loop.
        finetuning_estimator_config = self._build_estimator_config(
            base_estimator_config,
            self.n_estimators_finetune,
        )
        if model_path is not None:
            finetuning_estimator_config["model_path"] = model_path

        # Configs used for validation-time evaluation and final inference.
        validation_eval_config = self._build_eval_config(
            base_estimator_config,
            self.n_estimators_validation,
        )
        final_inference_eval_config = self._build_eval_config(
            base_estimator_config,
            self.n_estimators_final_inference,
        )

        eval_devices = infer_devices(self.device)
        validation_eval_config["device"] = eval_devices
        final_inference_eval_config["device"] = eval_devices

        epoch_to_start_from = 0
        checkpoint_path = None
        if output_dir is not None:
            checkpoint_path, epoch_to_start_from = (
                get_checkpoint_path_and_epoch_from_output_dir(
                    output_dir=output_dir,
                    train_size=train_size,
                    get_best=False,
                )
            )
            if checkpoint_path is not None:
                logger.info(
                    f"Restarting training from checkpoint {checkpoint_path} at epoch "
                    f"{epoch_to_start_from}",
                )
                finetuning_estimator_config["model_path"] = checkpoint_path

        self.finetuned_estimator_ = self._create_estimator(finetuning_estimator_config)
        self._setup_estimator()

        X, y, _, _ = ensure_compatible_fit_inputs_sklearn(
            X,
            y,
            estimator=self.finetuned_estimator_,
            ensure_y_numeric=self._model_type == "regressor",
        )

        self.X_ = X
        self.y_ = y
        if X_val is not None and y_val is not None:
            X_train, y_train = X, y
            X_val, y_val, _, _ = ensure_compatible_fit_inputs_sklearn(
                X_val,
                y_val,
                estimator=self.finetuned_estimator_,
                ensure_y_numeric=self._model_type == "regressor",
            )
        else:
            X_train, X_val, y_train, y_val = self._get_train_val_split(X, y)

        # Calculate the context size used during finetuning.
        n_finetune_ctx_plus_query_samples = min(
            self.n_finetune_ctx_plus_query_samples,
            len(y_train),
        )

        self.finetuned_estimator_._initialize_model_variables()
        self.finetuned_estimator_.model_.to(self.device)

        if self.use_activation_checkpointing:
            self.finetuned_estimator_.model_.recompute_layer = True  # type: ignore

        optimizer = get_and_init_optimizer(
            model_parameters=self.finetuned_estimator_.model_.parameters(),  # type: ignore
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

        use_amp = self.device.startswith("cuda") and torch.cuda.is_available()
        scaler = GradScaler() if use_amp else None  # type: ignore

        logger.info("--- 🚀 Eval default model ---")
        eval_result = self._evaluate_model(
            validation_eval_config,
            X_train,  # pyright: ignore[reportArgumentType]
            y_train,  # pyright: ignore[reportArgumentType]
            X_val,  # pyright: ignore[reportArgumentType]
            y_val,  # pyright: ignore[reportArgumentType]
        )
        self._log_epoch_evaluation(-1, eval_result, mean_train_loss=None)
        best_metric: float = eval_result.primary

        logger.info("--- 🚀 Starting Fine-tuning ---")
        patience_counter = 0
        best_model = None

        scheduler: LambdaLR | None = None

        finetuning_query_size = self._get_valid_finetuning_query_size(
            query_size=int(
                n_finetune_ctx_plus_query_samples * self.finetune_ctx_query_split_ratio
            ),
            y_train=y_train,
        )
        for epoch in range(epoch_to_start_from, self.epochs):
            # Per-epoch aggregates for cleaner learning curves.
            epoch_loss_sum = 0.0
            epoch_batches = 0

            # Regenerate datasets each epoch with a different random_state
            training_splitter = partial(
                train_test_split,
                test_size=finetuning_query_size,
                random_state=self.random_state + epoch,
            )

            training_datasets = get_preprocessed_dataset_chunks(
                calling_instance=self.finetuned_estimator_,
                X_raw=X_train,
                y_raw=y_train,
                split_fn=training_splitter,
                max_data_size=n_finetune_ctx_plus_query_samples,
                model_type=self._model_type,
                equal_split_size=False,
                seed=self.random_state + epoch,
            )

            dataloader_generator = torch.Generator().manual_seed(
                self.random_state + epoch
            )
            finetuning_dataloader = DataLoader(
                training_datasets,
                batch_size=self.meta_batch_size,
                collate_fn=meta_dataset_collator,
                shuffle=True,
                generator=dataloader_generator,
            )

            # Instantiate the LR scheduler only once
            if self.use_lr_scheduler and scheduler is None:
                steps_per_epoch = len(finetuning_dataloader)
                if steps_per_epoch == 0:
                    logger.warning(
                        "No training batches available; ending training early.",
                    )
                    break

                total_steps = steps_per_epoch * self.epochs
                warmup_steps = int(total_steps * 0.1)

                lrate_schedule_fn = get_cosine_schedule_with_warmup(
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    warmup_only=self.lr_warmup_only,
                )
                scheduler = LambdaLR(optimizer, lr_lambda=lrate_schedule_fn)

                logger.info(
                    "Using LambdaLR %s schedule: total_steps=%d, warmup_steps=%d",
                    "warmup-only (constant LR after warmup)"
                    if self.lr_warmup_only
                    else "warmup+cosine",
                    total_steps,
                    warmup_steps,
                )

            progress_bar = tqdm(
                finetuning_dataloader,
                desc=f"Finetuning Epoch {epoch + 1}/{self.epochs}",
            )
            for batch in progress_bar:
                optimizer.zero_grad()

                if self._should_skip_batch(batch):
                    continue

                self._setup_batch(batch)

                self.finetuned_estimator_.fit_from_preprocessed(
                    batch.X_context,
                    batch.y_context,
                    batch.cat_indices,
                    batch.configs,
                )

                use_scaler = use_amp and scaler is not None

                with autocast(enabled=use_scaler), sdpa_kernel_context():  # type: ignore
                    loss = self._forward_with_loss(batch)

                if use_scaler:
                    with sdpa_kernel_context():
                        scaler.scale(loss).backward()  # type: ignore
                    scaler.unscale_(optimizer)  # type: ignore

                    if self.grad_clip_value is not None:
                        clip_grad_norm_(
                            self.finetuned_estimator_.model_.parameters(),  # type: ignore
                            self.grad_clip_value,
                        )

                    scaler.step(optimizer)  # type: ignore
                    scaler.update()  # type: ignore
                else:
                    with sdpa_kernel_context():
                        loss.backward()

                    if self.grad_clip_value is not None:
                        clip_grad_norm_(
                            self.finetuned_estimator_.model_.parameters(),  # type: ignore
                            self.grad_clip_value,
                        )

                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                loss_scalar = float(loss.detach().item())

                epoch_loss_sum += loss_scalar
                epoch_batches += 1

                progress_bar.set_postfix(
                    loss=f"{loss_scalar:.4f}",
                )

            mean_train_loss = (
                epoch_loss_sum / epoch_batches if epoch_batches > 0 else None
            )

            eval_result = self._evaluate_model(
                validation_eval_config,
                X_train,  # pyright: ignore[reportArgumentType]
                y_train,  # pyright: ignore[reportArgumentType]
                X_val,  # pyright: ignore[reportArgumentType]
                y_val,  # pyright: ignore[reportArgumentType]
            )

            self._log_epoch_evaluation(epoch, eval_result, mean_train_loss)

            primary_metric = eval_result.primary

            if output_dir is not None and not np.isnan(primary_metric):
                save_interval_checkpoint = (
                    self.save_checkpoint_interval is not None
                    and (epoch + 1) % self.save_checkpoint_interval == 0
                )

                is_best = self._is_improvement(primary_metric, best_metric)

                if save_interval_checkpoint or is_best:
                    save_checkpoint(
                        estimator=self.finetuned_estimator_,
                        output_dir=output_dir,
                        epoch=epoch + 1,
                        optimizer=optimizer,
                        metrics=self._get_checkpoint_metrics(eval_result),
                        train_size=train_size,
                        is_best=is_best,
                        save_interval_checkpoint=save_interval_checkpoint,
                    )

            if self.early_stopping and not np.isnan(primary_metric):
                if self._is_improvement(primary_metric, best_metric):
                    best_metric = primary_metric
                    patience_counter = 0
                    best_model = copy.deepcopy(self.finetuned_estimator_)
                else:
                    patience_counter += 1
                    logger.info(
                        "⚠️  No improvement for %s epochs. Best %s: %.4f",
                        patience_counter,
                        self._metric_name,
                        best_metric,
                    )

                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        "🛑 Early stopping triggered. Best %s: %.4f",
                        self._metric_name,
                        best_metric,
                    )
                    if best_model is not None:
                        self.finetuned_estimator_ = best_model
                    break

            if self.time_limit is not None:
                elapsed_time = time.monotonic() - start_time
                if elapsed_time > self.time_limit:
                    logger.info(
                        "🛑 Time limit of %d seconds reached. Stopping training.",
                        self.time_limit,
                    )
                    break

                n_epochs_run = epoch + 1 - epoch_to_start_from
                if elapsed_time + (elapsed_time / n_epochs_run) > self.time_limit:
                    logger.info(
                        "🛑 Not enough time remaining for another epoch. Stopping "
                        "training.",
                    )
                    break

        if self.early_stopping and best_model is not None:
            self.finetuned_estimator_ = best_model

        logger.info("--- ✅ Fine-tuning Finished ---")

        self._setup_inference_model(final_inference_eval_config)

        self.is_fitted_ = True
        return self
