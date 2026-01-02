"""A TabPFN classifier that finetunes the underlying model for a single task.

This module provides the FinetunedTabPFNClassifier class, which wraps TabPFN
and allows fine-tuning on a specific dataset using the familiar scikit-learn
.fit() and .predict() API.
"""

from __future__ import annotations

import copy
import logging
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.finetuning._torch_compat import GradScaler, autocast, sdpa_kernel_context
from tabpfn.finetuning.data_util import (
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)
from tabpfn.finetuning.train_util import (
    clone_model_for_evaluation,
    get_and_init_optimizer,
    get_checkpoint_path_and_epoch_from_output_dir,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def evaluate_model(
    classifier: TabPFNClassifier,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Evaluate the model's performance on the validation set.

    Args:
        classifier: The TabPFNClassifier instance to evaluate.
        eval_config: Configuration dictionary for the evaluation classifier.
        X_train: Training input samples of shape (n_samples, n_features).
        y_train: Training target values of shape (n_samples,).
        X_val: Validation input samples of shape (n_samples, n_features).
        y_val: Validation target values of shape (n_samples,).

    Returns:
        A tuple containing (roc_auc, log_loss_score). Returns (nan, nan) if
        evaluation fails due to an error.
    """
    eval_classifier = clone_model_for_evaluation(
        classifier,
        eval_config,
        TabPFNClassifier,
    )
    eval_classifier.fit(X_train, y_train)

    try:
        probabilities = eval_classifier.predict_proba(X_val)  # type: ignore
        roc_auc = (
            roc_auc_score(
                y_val,
                probabilities,
                multi_class="ovr",
            )
            if len(np.unique(y_val)) > 2
            else roc_auc_score(
                y_val,
                probabilities[:, 1],
            )
        )
        log_loss_score = log_loss(y_val, probabilities)
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.warning(f"An error occurred during evaluation: {e}")
        roc_auc, log_loss_score = np.nan, np.nan

    return roc_auc, log_loss_score  # pyright: ignore[reportReturnType]


class FinetunedTabPFNClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible wrapper for fine-tuning the TabPFNClassifier.

    This class encapsulates the fine-tuning loop, allowing you to fine-tune
    TabPFN on a specific dataset using the familiar .fit() and .predict() API.

    Args:
        device: The device to run the model on. Defaults to "cuda".
        epochs: The total number of passes through the fine-tuning data.
            Defaults to 30.
        learning_rate: The learning rate for the AdamW optimizer. A small value
            is crucial for stable fine-tuning. Defaults to 1e-5.
        weight_decay: The weight decay for the AdamW optimizer. Defaults to 1e-3.
        validation_split_ratio: Fraction of the original training data reserved
            as a validation set for early stopping and monitoring. Defaults to 0.1.
        n_finetune_ctx_plus_query_samples: The total number of samples per
            meta-dataset during fine-tuning (context plus query) before applying
            the `finetune_ctx_query_split_ratio`. Defaults to 20_000.
        finetune_ctx_query_split_ratio: The proportion of each fine-tuning
            meta-dataset to use as query samples for calculating the loss. The
            remainder is used as context. Defaults to 0.02.
        n_inference_subsample_samples: The total number of subsampled training
            samples per estimator during validation and final inference.
            Defaults to 50_000.
        meta_batch_size: The number of meta-datasets to process in a single
            optimization step. Currently, this should be kept at 1. Defaults to 1.
        random_state: Seed for reproducibility of data splitting and model
            initialization. Defaults to 0.
        early_stopping: Whether to use early stopping based on validation ROC AUC
            performance. Defaults to True.
        early_stopping_patience: Number of epochs to wait for improvement before
            early stopping. Defaults to 6.
        min_delta: Minimum change in ROC AUC to be considered as an improvement.
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
            `TabPFNClassifier` only during fine-tuning to control the number of
            estimators (ensemble size) used in the training loop. If None, the
            value from `kwargs` or the `TabPFNClassifier` default is used.
            Defaults to 2.
        n_estimators_validation: If set, overrides `n_estimators` only for
            validation-time evaluation during fine-tuning (early-stopping /
            monitoring). If None, the value from `kwargs` or the
            `TabPFNClassifier` default is used. Defaults to 4.
        n_estimators_final_inference: If set, overrides `n_estimators` only for
            the final fitted inference model that is used after fine-tuning. If
            None, the value from `kwargs` or the `TabPFNClassifier` default is
            used. Defaults to 8.
        use_activation_checkpointing: Whether to use activation checkpointing to
            reduce memory usage. Defaults to False.
        save_checkpoint_interval: Number of epochs between checkpoint saves. If
            None, no intermediate checkpoints are saved. The best model checkpoint
            is always saved regardless of this setting. Defaults to 10.
        **kwargs: Additional keyword arguments to pass to the underlying
            `TabPFNClassifier`, such as `n_estimators`.
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
        finetune_ctx_query_split_ratio: float = 0.02,
        n_inference_subsample_samples: int = 50_000,
        meta_batch_size: int = 1,
        random_state: int = 0,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        min_delta: float = 1e-4,
        grad_clip_value: float | None = 1.0,
        use_lr_scheduler: bool = True,
        lr_warmup_only: bool = False,
        n_estimators_finetune: int | None = 2,
        n_estimators_validation: int | None = 2,
        n_estimators_final_inference: int | None = 8,
        use_activation_checkpointing: bool = False,
        save_checkpoint_interval: int | None = 10,
        extra_classifier_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_split_ratio = validation_split_ratio
        self.n_finetune_ctx_plus_query_samples = n_finetune_ctx_plus_query_samples
        self.finetune_ctx_query_split_ratio = finetune_ctx_query_split_ratio
        self.n_inference_subsample_samples = n_inference_subsample_samples
        self.meta_batch_size = meta_batch_size
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
        self.classifier_kwargs = extra_classifier_kwargs or {}

        self.save_checkpoint_interval = save_checkpoint_interval

        assert self.meta_batch_size == 1, "meta_batch_size must be 1 for finetuning"

    def _build_classifier_config(
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
        """Return eval config sharing settings except for an optional n_estimators override."""  # noqa: E501
        config = self._build_classifier_config(base_config, n_estimators_override)
        existing = dict(config.get("inference_config", {}) or {})
        existing["SUBSAMPLE_SAMPLES"] = self.n_inference_subsample_samples
        config["inference_config"] = existing
        return config

    def fit(
        self, X: np.ndarray, y: np.ndarray, output_dir: Path | None = None
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
        if output_dir is None:
            warnings.warn(
                "`output_dir` is not set. This means no checkpointing will be done and "
                "all progress will be lost if the training is interrupted.",
                UserWarning,
                stacklevel=2,
            )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        self.X_ = X
        self.y_ = y

        return self._fit(X=X, y=y, output_dir=output_dir)

    def _fit(  # noqa: C901,PLR0912
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: Path | None = None,
    ) -> FinetunedTabPFNClassifier:
        """Internal implementation of fit that runs the finetuning loop."""
        # Store the original training size for checkpoint naming
        train_size = X.shape[0]

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.validation_split_ratio,
            random_state=self.random_state,
            stratify=y,
        )

        # Calculate the context size used during finetuning.
        n_finetune_ctx_plus_query_samples = min(
            self.n_finetune_ctx_plus_query_samples,
            len(y_train),
        )

        inference_config = self.classifier_kwargs.get("inference_config", {})
        base_classifier_config: dict[str, Any] = {
            **self.classifier_kwargs,
            "ignore_pretraining_limits": True,
            "device": self.device,
            "random_state": self.random_state,
            "inference_config": inference_config,
        }

        # Config used for the finetuning loop.
        finetuning_classifier_config = self._build_classifier_config(
            base_classifier_config,
            self.n_estimators_finetune,
        )

        # Configs used for validation-time evaluation and final inference. They
        # share all settings except for a potential `n_estimators` override, and
        # both use the same `SUBSAMPLE_SAMPLES` setting.
        validation_eval_config = self._build_eval_config(
            base_classifier_config,
            self.n_estimators_validation,
        )
        final_inference_eval_config = self._build_eval_config(
            base_classifier_config,
            self.n_estimators_final_inference,
        )

        if self.device.startswith("cuda") and torch.cuda.is_available():
            eval_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            eval_devices = ["cpu"]  # Used in tests

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
                finetuning_classifier_config["model_path"] = checkpoint_path

        self.finetuned_classifier_ = TabPFNClassifier(
            **finetuning_classifier_config,
            fit_mode="batched",
            differentiable_input=False,
        )
        self.finetuned_classifier_.softmax_temperature_ = (
            self.finetuned_classifier_.softmax_temperature
        )

        self.finetuned_classifier_._initialize_model_variables()

        self.finetuned_classifier_.model_.to(self.device)

        if self.use_activation_checkpointing:
            self.finetuned_classifier_.model_.recompute_layer = True  # type: ignore

        optimizer = get_and_init_optimizer(
            model_parameters=self.finetuned_classifier_.model_.parameters(),  # type: ignore
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )

        loss_function = torch.nn.CrossEntropyLoss()

        use_amp = self.device.startswith("cuda") and torch.cuda.is_available()
        scaler = GradScaler() if use_amp else None  # type: ignore

        logger.info("--- 🚀 Starting Fine-tuning ---")

        best_roc_auc = -np.inf
        patience_counter = 0
        best_model = None

        scheduler: LambdaLR | None = None

        for epoch in range(epoch_to_start_from, self.epochs):
            # Per-epoch aggregates for cleaner learning curves.
            epoch_loss_sum = 0.0
            epoch_batches = 0

            # Regenerate datasets each epoch with a different random_state to ensure
            # diversity in context/query pairs across epochs. This prevents the
            # model from seeing the exact same splits in every epoch, which could
            # reduce training signal diversity.
            training_splitter = partial(
                train_test_split,
                test_size=self.finetune_ctx_query_split_ratio,
                random_state=self.random_state + epoch,
            )

            training_datasets = get_preprocessed_dataset_chunks(
                calling_instance=self.finetuned_classifier_,
                X_raw=X_train,
                y_raw=y_train,
                split_fn=training_splitter,
                max_data_size=n_finetune_ctx_plus_query_samples,
                model_type="classifier",
                equal_split_size=False,
                seed=self.random_state + epoch,
            )

            finetuning_dataloader = DataLoader(
                training_datasets,
                batch_size=self.meta_batch_size,
                collate_fn=meta_dataset_collator,
                shuffle=True,
            )

            # Instantiate the LR scheduler only once so that the warmup and
            # cosine schedule run continuously across all epochs. scheduler is None
            # only in the first epoch.
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
            for (
                X_context_batch,
                X_query_batch,
                y_context_batch,
                y_query_batch,
                cat_ixs,
                confs,
            ) in progress_bar:
                ctx = set(
                    torch.cat([t.flatten() for t in y_context_batch]).unique().tolist()
                )
                qry = set(
                    torch.cat([t.flatten() for t in y_query_batch]).unique().tolist()
                )
                if not qry.issubset(ctx):
                    logger.warning(
                        "Skipping batch: query labels %s are not a subset of "
                        "context labels %s",
                        qry,
                        ctx,
                    )
                    continue

                optimizer.zero_grad()

                self.finetuned_classifier_.fit_from_preprocessed(
                    X_context_batch,
                    y_context_batch,
                    cat_ixs,
                    confs,
                )

                use_scaler = use_amp and scaler is not None

                with autocast(enabled=use_scaler):  # type: ignore
                    with sdpa_kernel_context():
                        # shape suffix: Q=n_queries, B=batch(=1), E=estimators, L=logits
                        predictions_QBEL = self.finetuned_classifier_.forward(
                            X_query_batch,
                            return_raw_logits=True,
                        )

                    Q, B, E, L = predictions_QBEL.shape
                    assert y_query_batch.shape[1] == Q
                    assert B == 1
                    assert self.n_estimators_finetune == E
                    assert self.finetuned_classifier_.n_classes_ == L

                    # For getting the loss using the CE loss, we need to reshape.
                    # We treat the estimator dim as batch dim and
                    # permute so that the shape is (B*E, L, Q). This way
                    # the loss is first calculated for each estimator and then
                    # the results are averaged. This is what we want. If we
                    # average each estimator first and then take the mean we
                    # don't improve the individual estimators but the sum of it,
                    # which is not ideal.
                    predictions_BLQ = predictions_QBEL.permute(1, 2, 3, 0).reshape(
                        B * E, L, Q
                    )

                    loss = loss_function(
                        predictions_BLQ,
                        y_query_batch.repeat(self.n_estimators_finetune, 1).to(
                            self.device
                        ),
                    )

                if use_scaler:
                    # When using activation checkpointing, we need to exclude the cuDNN
                    # backend also during the backward pass because checkpointing re-
                    # computes the forward pass during backward.
                    with sdpa_kernel_context():
                        scaler.scale(loss).backward()  # type: ignore
                    scaler.unscale_(optimizer)  # type: ignore

                    if self.grad_clip_value is not None:
                        clip_grad_norm_(
                            self.finetuned_classifier_.model_.parameters(),  # type: ignore
                            self.grad_clip_value,
                        )

                    scaler.step(optimizer)  # type: ignore
                    scaler.update()  # type: ignore
                else:
                    with sdpa_kernel_context():
                        loss.backward()

                    if self.grad_clip_value is not None:
                        clip_grad_norm_(
                            self.finetuned_classifier_.model_.parameters(),  # type: ignore
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

            roc_auc, log_loss_score = evaluate_model(
                self.finetuned_classifier_,
                validation_eval_config,
                X_train,  # pyright: ignore[reportArgumentType]
                y_train,  # pyright: ignore[reportArgumentType]
                X_val,  # pyright: ignore[reportArgumentType]
                y_val,  # pyright: ignore[reportArgumentType]
            )

            logger.info(
                f"📊 Epoch {epoch + 1} Evaluation | Val ROC: {roc_auc:.4f}, "
                f"Val Log Loss: {log_loss_score:.4f}, Train Loss: {mean_train_loss:.4f}"
            )

            if output_dir is not None and not np.isnan(roc_auc):
                save_interval_checkpoint = (
                    self.save_checkpoint_interval is not None
                    and (epoch + 1) % self.save_checkpoint_interval == 0
                )

                is_best = roc_auc > best_roc_auc + self.min_delta

                if save_interval_checkpoint or is_best:
                    save_checkpoint(
                        estimator=self.finetuned_classifier_,
                        output_dir=output_dir,
                        epoch=epoch + 1,
                        optimizer=optimizer,
                        metrics={"roc_auc": roc_auc, "log_loss": log_loss_score},
                        train_size=train_size,
                        is_best=is_best,
                        save_interval_checkpoint=save_interval_checkpoint,
                    )

            if self.early_stopping and not np.isnan(roc_auc):
                if roc_auc > best_roc_auc + self.min_delta:
                    best_roc_auc = roc_auc
                    patience_counter = 0
                    best_model = copy.deepcopy(self.finetuned_classifier_)
                else:
                    patience_counter += 1
                    logger.info(
                        "⚠️  No improvement for %s epochs. Best ROC AUC: %.4f",
                        patience_counter,
                        best_roc_auc,
                    )

                if patience_counter >= self.early_stopping_patience:
                    logger.info(
                        "🛑 Early stopping triggered. Best ROC AUC: %.4f",
                        best_roc_auc,
                    )
                    if best_model is not None:
                        self.finetuned_classifier_ = best_model
                    # Log one last set of epoch metrics before breaking.
                    break

        if self.early_stopping and best_model is not None:
            self.finetuned_classifier_ = best_model

        logger.info("--- ✅ Fine-tuning Finished ---")

        finetuned_inference_classifier = clone_model_for_evaluation(
            self.finetuned_classifier_,  # type: ignore
            final_inference_eval_config,
            TabPFNClassifier,
        )
        self.finetuned_inference_classifier_ = finetuned_inference_classifier
        self.finetuned_inference_classifier_.fit_mode = "fit_preprocessors"  # type: ignore
        self.finetuned_inference_classifier_.fit(self.X_, self.y_)  # type: ignore

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.

        Args:
            X: The input samples of shape (n_samples, n_features).

        Returns:
            The class probabilities of the input samples with shape
            (n_samples, n_classes).
        """
        check_is_fitted(self)

        return self.finetuned_inference_classifier_.predict_proba(X)  # type: ignore

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class for X.

        Args:
            X: The input samples of shape (n_samples, n_features).

        Returns:
            The predicted classes with shape (n_samples,).
        """
        check_is_fitted(self)

        return self.finetuned_inference_classifier_.predict(X)  # type: ignore
