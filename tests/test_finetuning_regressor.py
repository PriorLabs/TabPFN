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
from torch.utils.data import DataLoader

from tabpfn.architectures.base.bar_distribution import BarDistribution
from tabpfn.finetuning.data_util import (
    RegressorBatch,
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)
from tabpfn.finetuning.finetuned_regressor import (
    FinetunedTabPFNRegressor,
    _compute_regression_loss,
)
from tabpfn.preprocessing import RegressorEnsembleConfig
from tabpfn.regressor import TabPFNRegressor

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


@pytest.fixture(params=devices)
def regressor_instance(request: pytest.FixtureRequest) -> TabPFNRegressor:
    """Provide a basic regressor instance, parameterized by device."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")
    return TabPFNRegressor(
        n_estimators=2,
        device=device,
        random_state=42,
        inference_precision=torch.float32,
        fit_mode="batched",
        differentiable_input=False,
    )


@pytest.fixture
def variable_synthetic_regression_dataset_collection() -> list[
    tuple[np.ndarray, np.ndarray]
]:
    """Create a small collection of synthetic regression datasets with varying sizes."""
    datasets = []
    dataset_sizes = [10, 20, 30]
    num_features = 3
    for num_samples in dataset_sizes:
        X = rng.normal(size=(num_samples, num_features)).astype(np.float32)
        y = rng.normal(size=(num_samples,)).astype(np.float32)
        datasets.append((X, y))
    return datasets


@pytest.mark.parametrize(
    ("device", "early_stopping", "use_lr_scheduler"),
    [
        (device, early_stopping, use_lr_scheduler)
        for device in devices
        for early_stopping in [True, False]
        for use_lr_scheduler in [True, False]
    ],
)
def test__finetuned_tabpfn_regressor__fit_and_predict(
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
def test__regressor_checkpoint_contains_mse_metric(
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


def test__compute_regression_loss__correct_mse_and_mae_with_nan_targets() -> None:
    """Ensure NaN targets are ignored for auxiliary regression losses."""
    borders = torch.linspace(-2.0, 2.0, steps=9, dtype=torch.float32)
    bardist_loss_fn = BarDistribution(borders=borders, ignore_nan_targets=True)

    # Deterministic setup: zero logits -> uniform bar probabilities.
    # For symmetric borders, this yields an exact mean prediction of 0.0.
    logits_BQL = torch.zeros((1, 4, bardist_loss_fn.num_bars), dtype=torch.float32)
    predictions_mean_BQ = bardist_loss_fn.mean(logits_BQL)
    assert torch.equal(predictions_mean_BQ, torch.zeros_like(predictions_mean_BQ))

    targets_BQ = torch.tensor([[0.0, 0.5, float("nan"), -0.5]], dtype=torch.float32)
    assert torch.isnan(targets_BQ[0, 2]).item()

    mse_mae_only_loss = _compute_regression_loss(
        logits_BQL=logits_BQL,
        targets_BQ=targets_BQ,
        bardist_loss_fn=bardist_loss_fn,
        ce_loss_weight=0.0,
        mse_loss_weight=1.0,
        mse_loss_clip=None,
        mae_loss_weight=1.0,
        mae_loss_clip=None,
    )
    assert torch.isfinite(mse_mae_only_loss).item()
    # MSE terms (valid only): (0.0 - 0.0)^2 + (0.0 - 0.5)^2 + (0.0 - (-0.5))^2 = 0.5
    # Mean is over all 4 positions (NaN position contributes 0.0): 0.5 / 4 = 0.125
    expected_mse = 0.125
    # MAE terms (valid only): |0.0 - 0.0| + |0.0 - 0.5| + |0.0 - (-0.5)| = 1.0
    # Mean over all 4 positions: 1.0 / 4 = 0.25
    expected_mae = 0.25
    expected_total = expected_mse + expected_mae
    assert mse_mae_only_loss.item() == expected_total


def test__compute_regression_loss__masks_nan_targets() -> None:
    borders = torch.linspace(-2.0, 2.0, steps=9, dtype=torch.float32)
    bardist_loss_fn = BarDistribution(borders=borders, ignore_nan_targets=True)
    logits_BQL = torch.zeros((1, 4, bardist_loss_fn.num_bars), dtype=torch.float32)

    rps_only_loss = _compute_regression_loss(
        logits_BQL=logits_BQL,
        targets_BQ=torch.full((1, 4), float("nan"), dtype=torch.float32),
        bardist_loss_fn=bardist_loss_fn,
        ce_loss_weight=0.0,
        crps_loss_weight=1.0,
        mse_loss_weight=1.0,
        mse_loss_clip=None,
        mae_loss_weight=1.0,
        mae_loss_clip=None,
    )
    assert torch.isfinite(rps_only_loss).item()
    assert rps_only_loss.item() == 0.0

    rls_only_loss = _compute_regression_loss(
        logits_BQL=logits_BQL,
        targets_BQ=torch.full((1, 4), float("nan"), dtype=torch.float32),
        bardist_loss_fn=bardist_loss_fn,
        ce_loss_weight=0.0,
        crls_loss_weight=1.0,
        mse_loss_weight=1.0,
        mse_loss_clip=None,
        mae_loss_weight=1.0,
        mae_loss_clip=None,
    )
    assert torch.isfinite(rls_only_loss).item()
    assert rls_only_loss.item() == 0.0


def test__compute_regression_loss__rps_vs_rls_matches_expected_value() -> None:
    """Check RPS (squared) vs RLS (log) ranked probability loss.

    Uses a deterministic 2-bin toy example so expected values are easy to verify.
    """
    borders = torch.tensor([0.0, 1.0, 2.0])
    bardist_loss_fn = BarDistribution(borders=borders, ignore_nan_targets=True)

    # Two bins. Desired probs: [0.25, 0.75] -> pred CDF: [0.25, 1.0]
    probs_L = torch.tensor([0.25, 0.75])
    logits_L = probs_L.log()
    logits_BQL = logits_L.view(1, 1, -1)

    # Target y in bin 1 -> target CDF: [0.0, 1.0]
    targets_BQ = torch.tensor([[1.2]])

    rps_loss = _compute_regression_loss(
        logits_BQL=logits_BQL,
        targets_BQ=targets_BQ,
        bardist_loss_fn=bardist_loss_fn,
        ce_loss_weight=0.0,
        crps_loss_weight=1.0,
        mse_loss_weight=0.0,
        mse_loss_clip=None,
        mae_loss_weight=0.0,
        mae_loss_clip=None,
    )

    rls_loss = _compute_regression_loss(
        logits_BQL=logits_BQL,
        targets_BQ=targets_BQ,
        bardist_loss_fn=bardist_loss_fn,
        ce_loss_weight=0.0,
        crls_loss_weight=1.0,
        mse_loss_weight=0.0,
        mse_loss_clip=None,
        mae_loss_weight=0.0,
        mae_loss_clip=None,
    )

    # second term is zero but is added for clarity
    expected_rps = (0.25 - 0) ** 2 + (1 - 1) ** 2

    # second term is zero but just added here for clarity
    expected_rls = -np.log(abs(0.25 + 0 - 1)) - np.log(abs(1 + 1 - 1))

    assert rls_loss.item() == pytest.approx(expected_rls)
    assert rps_loss.item() == pytest.approx(expected_rps)

    combined_loss = _compute_regression_loss(
        logits_BQL=logits_BQL,
        targets_BQ=targets_BQ,
        bardist_loss_fn=bardist_loss_fn,
        ce_loss_weight=0.0,
        crps_loss_weight=1.0,
        crls_loss_weight=1.0,
        mse_loss_weight=0.0,
        mse_loss_clip=None,
        mae_loss_weight=0.0,
        mae_loss_clip=None,
    )
    assert combined_loss.item() == pytest.approx(expected_rps + expected_rls)


def test_regressor_dataset_and_collator_batches_type(
    variable_synthetic_regression_dataset_collection: list[
        tuple[np.ndarray, np.ndarray]
    ],
    regressor_instance: TabPFNRegressor,
) -> None:
    """Test that dataset and collator produce correctly-typed RegressorBatch objects."""
    X_list = [X for X, _ in variable_synthetic_regression_dataset_collection]
    y_list = [y for _, y in variable_synthetic_regression_dataset_collection]
    dataset_collection = get_preprocessed_dataset_chunks(
        regressor_instance,
        X_list,
        y_list,
        train_test_split,
        100,
        model_type="regressor",
        equal_split_size=True,
        seed=42,
    )

    dl = DataLoader(
        dataset_collection,
        batch_size=1,
        collate_fn=meta_dataset_collator,
    )
    for batch in dl:
        assert isinstance(batch, RegressorBatch)
        for est_tensor in batch.X_context:
            assert isinstance(est_tensor, torch.Tensor)
            assert est_tensor.shape[0] == 1
        for est_tensor in batch.y_context:
            assert isinstance(est_tensor, torch.Tensor)
            assert est_tensor.shape[0] == 1

        assert isinstance(batch.cat_indices, list)
        for conf in batch.configs:
            for c in conf:
                assert isinstance(c, RegressorEnsembleConfig)

        assert isinstance(batch.X_query_raw, torch.Tensor)
        assert isinstance(batch.y_query_raw, torch.Tensor)
        assert batch.X_query_raw.shape[0] == 1
        assert batch.y_query_raw.shape[0] == 1
        assert batch.y_query.shape[0] == 1
        break
