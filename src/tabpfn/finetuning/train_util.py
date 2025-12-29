"""Some utility functions for training."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch.optim import AdamW

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tabpfn import TabPFNClassifier, TabPFNRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.getLogger().setLevel(logging.INFO)


def _format_train_size(train_size: int) -> str:
    """Format train size for filenames (e.g., 1000 -> '1K', 1500000 -> '1M500K')."""
    if train_size >= 1_000_000:
        millions = train_size // 1_000_000
        remainder = train_size % 1_000_000
        if remainder == 0:
            return f"{millions}M"
        thousands = remainder // 1_000
        return f"{millions}M{thousands}K"
    if train_size >= 1_000:
        thousands = train_size // 1_000
        return f"{thousands}K"
    return str(train_size)


def get_checkpoint_name(
    train_size: int,
    epoch: int | None = None,
    *,
    is_best: bool = False,
) -> str:
    """Get checkpoint filename with train size included.

    Args:
        train_size: Number of training samples.
        epoch: Epoch number (required if is_best=False).
        is_best: Whether this is the best checkpoint.

    Returns:
        Checkpoint filename like "checkpoint_1M_2.pth" or
        "checkpoint_1M_best.pth".
    """
    formatted_size = _format_train_size(train_size)
    if is_best:
        return f"checkpoint_{formatted_size}_best.pth"
    if epoch is None:
        raise ValueError("epoch must be provided when is_best=False")
    return f"checkpoint_{formatted_size}_{epoch}.pth"


# Copied from TabPFN-private/src/tabpfn/model_loading.py
# but allowing for saving additional fields
def save_tabpfn_model(
    model: TabPFNRegressor | TabPFNClassifier,
    save_path: Path | str | list[Path | str],
    additional_fields: dict[str, Any] | None = None,
) -> None:
    """Save the underlying TabPFN foundation model to ``save_path``.

    This writes only the base pre-trained weights and configuration. It does
    **not** store a fitted :class:`TabPFNRegressor`/``Classifier`` instance.
    The resulting file is merely a checkpoint consumed by
    :func:`load_model_criterion_config` to build a new estimator.

    Args:
        model:
            The internal model object of a ``TabPFN`` estimator.
        save_path:
            Path to save the checkpoint to.
        additional_fields:
            Additional fields to save to the checkpoint.
    """
    if len(model.models_) > 1 and (
        not isinstance(save_path, list) or len(save_path) != len(model.models_)
    ):
        raise ValueError(
            f"Your TabPFN estimator has multiple internal models ({len(model.models_)})"
            f", so you must provide a list of {len(model.models_)} save paths."
        )

    models = model.models_

    znorm_space_bardist = None
    if (
        hasattr(model, "znorm_space_bardist_")
        and model.znorm_space_bardist_ is not None  # type: ignore
    ):
        znorm_space_bardist = model.znorm_space_bardist_  # type: ignore

    configs = model.configs_
    save_paths = save_path if isinstance(save_path, list) else [save_path]

    for ens_model, config, path in zip(
        models,
        configs,
        save_paths,
    ):
        model_state = ens_model.state_dict()

        if znorm_space_bardist is not None:
            bardist_state = {
                f"criterion.{k}": v for k, v in znorm_space_bardist.state_dict().items()
            }
            state_dict = {**model_state, **bardist_state}
        else:
            state_dict = model_state

        checkpoint = {
            "state_dict": state_dict,
            "config": asdict(config),
        }

        if additional_fields is not None:
            checkpoint.update(additional_fields)

        torch.save(checkpoint, path)


def save_checkpoint(
    estimator: TabPFNClassifier | TabPFNRegressor,
    output_dir: Path,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    train_size: int,
    *,
    is_best: bool = False,
    save_interval_checkpoint: bool = False,
) -> None:
    """Save model checkpoint to disk.

    Saves both interval checkpoints (model_<epoch>.pth) and best checkpoint
    (checkpoint_best.pth) based on validation metrics.

    Args:
        estimator: The estimator to save.
        output_dir: Directory to save checkpoints to.
        epoch: Current epoch number.
        optimizer: Optimizer to save state dict from.
        metrics: Dictionary of validation metrics to save (e.g., {"mse": 0.5}
            or {"roc_auc": 0.9, "log_loss": 0.3}).
        train_size: Number of training samples.
        is_best: Whether this is the best model so far.
        save_interval_checkpoint: Whether to save an interval checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    additional_fields = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        **{key: float(value) for key, value in metrics.items()},
    }

    if save_interval_checkpoint:
        interval_checkpoint_name = get_checkpoint_name(train_size, epoch)
        interval_checkpoint_path = output_dir / interval_checkpoint_name
        save_tabpfn_model(
            model=estimator,
            save_path=interval_checkpoint_path,
            additional_fields=additional_fields,
        )
        logging.info(f"💾 Saved interval checkpoint: {interval_checkpoint_path}")

    if is_best:
        best_checkpoint_name = get_checkpoint_name(train_size, is_best=True)
        best_checkpoint_path = output_dir / best_checkpoint_name
        save_tabpfn_model(
            model=estimator,
            save_path=best_checkpoint_path,
            additional_fields=additional_fields,
        )
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logging.info(
            f"⭐ Saved best checkpoint: {best_checkpoint_path} "
            f"(epoch {epoch}, {metrics_str})"
        )


def get_checkpoint_path_and_epoch_from_output_dir(
    output_dir: Path,
    train_size: int,
    *,
    get_best: bool = True,
) -> tuple[Path | None, int]:
    """Get the best or latest checkpoint from the output directory.

    If `get_best` is True, prioritizes checkpoint_best.pth if it exists,
    otherwise returns the highest epoch checkpoint. If `get_best` is False,
    returns the latest checkpoint.
    Returns (None, 0) if no checkpoints are found.

    Args:
        output_dir: Directory containing checkpoints.
        train_size: Number of training samples to filter checkpoints by.
        get_best: Whether to prioritize best checkpoint.
    """
    formatted_size = _format_train_size(train_size)
    # Check for best model checkpoint first if requested
    if get_best:
        best_checkpoint_path = output_dir / f"checkpoint_{formatted_size}_best.pth"
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
            epoch = checkpoint.get("epoch", 0)
            return best_checkpoint_path, epoch

    # Fall back to numbered checkpoints with train_size
    pattern = f"checkpoint_{formatted_size}_[0-9]*.pth"
    checkpoint_files = sorted(output_dir.glob(pattern))
    if len(checkpoint_files) == 0:
        warnings.warn(
            f"Output dir present but no checkpoint file found for "
            f"train_size={train_size}. "
            "Starting from default checkpoint and epoch 0",
            UserWarning,
            stacklevel=2,
        )
        return None, 0

    checkpoint_path = checkpoint_files[-1]
    epoch = int(checkpoint_path.stem.split("_")[-1])
    return checkpoint_path, epoch


def get_and_init_optimizer(
    model_parameters: Iterator[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> AdamW:
    """Create and initialize AdamW optimizer, optionally loading from checkpoint.

    Args:
        model_parameters: Iterator of model parameters to optimize.
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        checkpoint_path: Path to checkpoint file containing optimizer state.
            If provided, loads optimizer state from this checkpoint.
        device: Device to load the optimizer state from.

    Returns:
        Initialized AdamW optimizer.
    """
    optimizer = AdamW(
        model_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info("Loaded optimizer state from checkpoint")

    return optimizer


def get_cosine_schedule_with_warmup(
    total_steps: int,
    warmup_steps: int,
    *,
    warmup_only: bool = False,
) -> Callable[[int], float]:
    """Return a lambda for a linear-warmup LR schedule with optional cosine decay.

    The returned function is intended for use with torch.optim.lr_scheduler.LambdaLR
    and therefore returns a multiplicative *factor* relative to the optimizer's
    base learning rate (which should be the user-provided max LR).

    If `warmup_only` is True, the schedule linearly warms up from 0 → 1 over
    `warmup_steps`` and then stays at 1.0 (i.e. no decay).
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(
            "warmup_steps must satisfy 0 <= warmup_steps < total_steps.",
        )

    def lrate_schedule_fn(curr_step: int) -> float:
        # Clamp to valid range for numerical stability.
        curr = max(0, min(curr_step, total_steps))

        if warmup_steps > 0 and curr < warmup_steps:
            # Linear warmup from 0 → 1 over warmup_steps.
            return float(curr) / float(max(1, warmup_steps))

        if warmup_only:
            # Keep LR constant at the base LR after warmup.
            return 1.0

        # Cosine decay from 1 → 0 over the remaining steps.
        decay_steps = max(1, total_steps - warmup_steps)
        decay_share = float(curr - warmup_steps) / float(decay_steps)

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * decay_share)))

    return lrate_schedule_fn
