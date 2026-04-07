"""Tests for the finetuning experiment logging module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.logging import FinetuningLogger, NullLogger, WandbLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WANDB_SENTINEL = "_test_wandb_not_installed"


class TestNullLogger:
    def test_implements_protocol(self):
        logger: FinetuningLogger = NullLogger()
        assert isinstance(logger, NullLogger)

    def test_all_methods_are_noop(self):
        logger = NullLogger()
        logger.setup({"lr": 0.01, "epochs": 10})
        logger.log_step({"train/loss": 0.5}, step=1)
        logger.log_epoch({"val/accuracy": 0.9}, step=100)
        logger.finish()


@pytest.fixture
def mock_wandb(monkeypatch):
    """Inject a mock wandb module so WandbLogger.setup() picks it up."""
    mock = MagicMock()
    mock_run = MagicMock()
    mock.init.return_value = mock_run
    monkeypatch.setitem(sys.modules, "wandb", mock)
    return mock
    # monkeypatch automatically restores sys.modules on teardown


class TestWandbLogger:
    def test_init_stores_params(self):
        logger = WandbLogger(project="test-proj", run_name="run-1", entity="team")
        assert logger.project == "test-proj"
        assert logger.run_name == "run-1"
        assert logger.wandb_kwargs == {"entity": "team"}
        assert logger._run is None

    def test_setup_calls_wandb_init(self, mock_wandb):
        mock_run = mock_wandb.init.return_value

        logger = WandbLogger(project="my-proj", run_name="my-run")
        config = {"lr": 0.01, "epochs": 5}
        logger.setup(config)

        mock_wandb.init.assert_called_once_with(
            project="my-proj", name="my-run", config=config
        )
        mock_wandb.define_metric.assert_called_once_with(
            "val/*", step_metric="train/epoch"
        )
        assert logger._run is mock_run

    def test_setup_does_not_override_explicit_kwargs(self, mock_wandb):
        logger = WandbLogger(
            project="default-proj", run_name="default-run", config={"custom": True}
        )
        logger.setup({"lr": 0.01})

        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["config"] == {"custom": True}
        assert call_kwargs["project"] == "default-proj"
        assert call_kwargs["name"] == "default-run"

    def test_log_step_delegates_to_run(self, mock_wandb):
        mock_run = mock_wandb.init.return_value

        logger = WandbLogger()
        logger.setup({})
        logger.log_step({"train/loss": 0.5}, step=42)

        mock_run.log.assert_called_once_with({"train/loss": 0.5}, step=42)

    def test_log_epoch_delegates_to_run(self, mock_wandb):
        mock_run = mock_wandb.init.return_value

        logger = WandbLogger()
        logger.setup({})
        logger.log_epoch({"val/accuracy": 0.95}, step=100)

        mock_run.log.assert_called_once_with({"val/accuracy": 0.95}, step=100)

    def test_log_step_noop_before_setup(self):
        logger = WandbLogger()
        logger.log_step({"train/loss": 0.5}, step=1)

    def test_log_epoch_noop_before_setup(self):
        logger = WandbLogger()
        logger.log_epoch({"val/acc": 0.9}, step=1)

    def test_finish_closes_run(self, mock_wandb):
        mock_run = mock_wandb.init.return_value

        logger = WandbLogger()
        logger.setup({})
        logger.finish()

        mock_run.finish.assert_called_once()
        assert logger._run is None

    def test_finish_noop_before_setup(self):
        logger = WandbLogger()
        logger.finish()

    def test_double_finish_is_safe(self, mock_wandb):
        mock_run = mock_wandb.init.return_value

        logger = WandbLogger()
        logger.setup({})
        logger.finish()
        logger.finish()

        mock_run.finish.assert_called_once()

    def test_setup_raises_readable_error_when_wandb_missing(self, monkeypatch):
        """WandbLogger.setup() should raise when wandb is missing."""
        monkeypatch.setitem(sys.modules, "wandb", None)
        logger = WandbLogger(project="p")
        with pytest.raises(ModuleNotFoundError, match="wandb"):
            logger.setup({})


class TestClassifierMetricName:
    """Verify _metric_name reflects the chosen eval_metric."""

    def test_default_metric_is_roc_auc(self):
        clf = FinetunedTabPFNClassifier()
        # eval_metric defaults to None; _metric_name should return "ROC AUC"
        assert clf._metric_name == "ROC AUC"

    def test_roc_auc_metric_name(self):
        clf = FinetunedTabPFNClassifier(eval_metric="roc_auc")
        assert clf._metric_name == "ROC AUC"

    def test_log_loss_metric_name(self):
        clf = FinetunedTabPFNClassifier(eval_metric="log_loss")
        assert clf._metric_name == "log_loss"
