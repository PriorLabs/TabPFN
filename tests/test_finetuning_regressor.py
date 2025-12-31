"""Tests for TabPFN regressor finetuning functionality.

This module contains regressor-specific tests for:
- The FinetunedTabPFNRegressor wrapper class (.fit() / .predict()).
- Regression checkpoint metric fields (e.g. storing 'mse').

We intentionally avoid duplicating tests that primarily exercise common logic in
`FinetunedTabPFNBase`, since those are covered by the classifier finetuning tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

from .utils import get_pytest_devices

rng = np.random.default_rng(42)

devices = get_pytest_devices()


def create_mock_architecture_forward_regression() -> Callable[..., torch.Tensor]:
    """Return a side_effect for mocking the internal Architecture forward in regression.

    The Architecture.forward method signature is:
    forward(x, y, *, only_return_standard_out=True, categorical_inds=None)

    Where:
    - x has shape (train+test rows, batch_size, num_features)
    - y has shape (train rows, batch_size) or (train rows, batch_size, 1)
    - returns shape (test rows, batch_size, n_out), with n_out determined by the model.
    """

    def mock_forward(
        self: torch.nn.Module,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        **_kwargs: bool,
    ) -> torch.Tensor:
        """Mock forward pass that returns random logits with the correct shape."""
        if isinstance(x, dict):
            x = x["main"]

        if y is not None:
            y_tensor = y["main"] if isinstance(y, dict) else y
            num_train_rows = y_tensor.shape[0]
        else:
            num_train_rows = 0

        total_rows = x.shape[0]
        batch_size = x.shape[1]
        num_test_rows = total_rows - num_train_rows

        # Touch a model parameter so gradients flow during backward pass.
        # This mirrors the classifier tests and avoids GradScaler issues on CUDA.
        first_param = next(self.parameters())
        param_contribution = 0.0 * first_param.sum()

        n_out = int(getattr(self, "n_out", 1))
        return (
            torch.randn(
                num_test_rows,
                batch_size,
                n_out,
                requires_grad=True,
                device=x.device,
            )
            + param_contribution
        )

    return mock_forward


@pytest.fixture(scope="module")
def synthetic_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data for testing."""
    result = make_regression(
        n_samples=120,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=42,
        coef=False,
    )
    X = np.asarray(result[0], dtype=np.float32)
    y = np.asarray(result[1], dtype=np.float32)
    return X, y


@pytest.mark.parametrize(
    ("device", "early_stopping", "use_lr_scheduler"),
    [
        (device, early_stopping, use_lr_scheduler)
        for device in devices
        for early_stopping in [True, False]
        for use_lr_scheduler in [True, False]
    ],
)
def test_finetuned_tabpfn_regressor_fit_and_predict(
    device: str,
    early_stopping: bool,
    use_lr_scheduler: bool,
    synthetic_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test FinetunedTabPFNRegressor fit/predict with a mocked forward pass."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    X, y = synthetic_regression_data
    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)

    epochs = 4 if early_stopping else 2
    finetuned_reg = FinetunedTabPFNRegressor(
        device=device,
        epochs=epochs,
        learning_rate=1e-4,
        validation_split_ratio=0.2,
        n_finetune_ctx_plus_query_samples=60,
        finetune_ctx_query_split_ratio=0.2,
        n_inference_subsample_samples=120,
        random_state=42,
        early_stopping=early_stopping,
        early_stopping_patience=2,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        use_lr_scheduler=use_lr_scheduler,
        lr_warmup_only=False,
    )

    mock_forward = create_mock_architecture_forward_regression()
    with mock.patch(
        "tabpfn.architectures.base.transformer.PerFeatureTransformer.forward",
        autospec=True,
        side_effect=mock_forward,
    ):
        finetuned_reg.fit(X_train, y_train)

    assert finetuned_reg.is_fitted_
    assert hasattr(finetuned_reg, "finetuned_estimator_")
    assert hasattr(finetuned_reg, "finetuned_inference_regressor_")

    predictions = finetuned_reg.predict(X_test)
    assert predictions.shape == (X_test.shape[0],)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("device", devices)
def test_regressor_checkpoint_contains_mse_metric(
    device: str,
    tmp_path: Path,
    synthetic_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Ensure regressor checkpoints store regression metrics (mse).

    This also checks that classifier-only metric fields are not stored.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    X, y = synthetic_regression_data
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    output_folder = tmp_path / "checkpoints_regressor"

    finetuned_reg = FinetunedTabPFNRegressor(
        device=device,
        epochs=2,
        learning_rate=1e-4,
        validation_split_ratio=0.2,
        n_finetune_ctx_plus_query_samples=60,
        finetune_ctx_query_split_ratio=0.2,
        n_inference_subsample_samples=120,
        random_state=42,
        early_stopping=False,
        use_lr_scheduler=False,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        save_checkpoint_interval=1,
    )

    mock_forward = create_mock_architecture_forward_regression()
    with mock.patch(
        "tabpfn.architectures.base.transformer.PerFeatureTransformer.forward",
        autospec=True,
        side_effect=mock_forward,
    ):
        finetuned_reg.fit(X_train, y_train, output_dir=output_folder)

    best_checkpoint_candidates = list(output_folder.glob("checkpoint_*_best.pth"))
    assert len(best_checkpoint_candidates) == 1, "Expected exactly one best checkpoint."
    best_checkpoint_path = best_checkpoint_candidates[0]

    best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
    assert "state_dict" in best_checkpoint
    assert "config" in best_checkpoint
    assert "optimizer" in best_checkpoint
    assert "epoch" in best_checkpoint
    assert "mse" in best_checkpoint
    assert "roc_auc" not in best_checkpoint
    assert "log_loss" not in best_checkpoint
