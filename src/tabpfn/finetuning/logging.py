"""Protocol-based experiment logging for finetuning."""

from __future__ import annotations

from typing import Any, Protocol


class FinetuningLogger(Protocol):
    """Protocol for finetuning experiment loggers."""

    def setup(self, config: dict[str, Any]) -> None:
        """Initialize the logger with run configuration."""
        ...

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        """Log per-step metrics (e.g., batch loss, learning rate)."""
        ...

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        """Log per-epoch metrics (e.g., val metrics, mean loss)."""
        ...

    def finish(self) -> None:
        """Finalize the logger (e.g., close wandb run)."""
        ...


class NullLogger:
    """No-op logger used when no experiment tracking is configured."""

    def setup(self, config: dict[str, Any]) -> None:
        """No-op."""

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        """No-op."""

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        """No-op."""

    def finish(self) -> None:
        """No-op."""


class WandbLogger:
    """WandB experiment logger."""

    def __init__(
        self,
        project: str | None = None,
        run_name: str | None = None,
        **wandb_kwargs: Any,
    ):
        self.project = project
        self.run_name = run_name
        self.wandb_kwargs = wandb_kwargs
        self._run = None

    def setup(self, config: dict[str, Any]) -> None:
        """Initialize a new WandB run with the given config."""
        import wandb  # noqa: PLC0415

        init_kwargs = dict(self.wandb_kwargs)
        if self.project:
            init_kwargs.setdefault("project", self.project)
        if self.run_name:
            init_kwargs.setdefault("name", self.run_name)
        init_kwargs.setdefault("config", config)
        self._run = wandb.init(**init_kwargs)
        wandb.define_metric("val/*", step_metric="train/epoch")

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics for a single training step."""
        if self._run:
            self._run.log(metrics, step=step)

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics for a completed epoch."""
        if self._run:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        """Finish the WandB run."""
        if self._run:
            self._run.finish()
            self._run = None
