from __future__ import annotations

import unittest
from functools import partial
from typing import Any, Literal
from unittest.mock import patch

import numpy as np
import pytest
import sklearn
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from tabpfn import TabPFNRegressor
from tabpfn.model.bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
)
from tabpfn.preprocessing import RegressorEnsembleConfig
from tabpfn.utils import meta_dataset_collator

rng = np.random.default_rng(42)

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

fit_modes = [
    "batched",
    "fit_preprocessors",
]
inference_precision_methods: list[torch.types._dtype | Literal["autocast", "auto"]] = [
    "auto",
    torch.float64,
]
estimators = [1, 2]
optimization_spaces_values = ["raw_label_space", "preprocessed"]

param_order = [
    "n_estimators",
    "device",
    "fit_mode",
    "inference_precision",
    "optimization_space",
]

default_config = {
    "n_estimators": 1,
    "device": "cpu",
    "fit_mode": "batched",
    "inference_precision": "auto",
    "optimization_space": "raw_label_space",
}

param_values: dict[str, list[Any]] = {
    "n_estimators": estimators,
    "device": devices,
    "fit_mode": fit_modes,
    "inference_precision": inference_precision_methods,
    "optimization_space": optimization_spaces_values,
}

combinations = [tuple(default_config[p] for p in param_order)]
for param_name in param_order:
    for value in param_values[param_name]:
        if value != default_config[param_name]:
            current_config = default_config.copy()
            current_config[param_name] = value
            combinations.append(tuple(current_config[p] for p in param_order))


@pytest.fixture(scope="module")
def synthetic_regression_data():
    """Generate synthetic regression data."""
    X = rng.normal(size=(30, 4)).astype(np.float32)
    # Generate continuous target variable
    y = (X @ rng.normal(size=4)).astype(np.float32)
    # Add to previous as line too long otherwise
    y += rng.normal(size=30).astype(np.float32) * 0.1
    return X, y


@pytest.fixture(params=devices)
def ft_regressor_instance(request) -> TabPFNRegressor:
    """Provides a basic regressor instance, parameterized by device."""
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


@pytest.fixture(params=devices)
def std_regressor_instance(request) -> TabPFNRegressor:
    """Provides a basic regressor instance, parameterized by device."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")
    return TabPFNRegressor(
        n_estimators=2,
        device=device,
        random_state=42,
        inference_precision=torch.float32,
        fit_mode="low_memory",
        differentiable_input=False,
    )


def create_regressor(
    n_estimators: int,
    device: str,
    fit_mode: str,
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    **kwargs,
) -> TabPFNRegressor:
    """Instantiates regressor with common parameters."""
    if device == "cpu" and inference_precision == "autocast":
        pytest.skip("Unsupported combination: CPU with 'autocast'")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    default_kwargs = {"random_state": 42}
    default_kwargs.update(kwargs)

    return TabPFNRegressor(
        n_estimators=n_estimators,
        device=device,
        fit_mode=fit_mode,
        inference_precision=inference_precision,
        **default_kwargs,
    )


# --- Tests ---


def test_regressor_dataset_and_collator_batches_type(
    synthetic_regression_data, ft_regressor_instance
):
    """Test that the batches returned by the dataset and collator
    are of the correct type.
    """
    X, y = synthetic_regression_data
    dataset_collection = ft_regressor_instance.get_preprocessed_datasets(
        X, y, train_test_split, 100
    )
    batch_size = 1
    dl = DataLoader(
        dataset_collection,
        batch_size=batch_size,
        collate_fn=meta_dataset_collator,
    )
    for batch in dl:
        assert isinstance(batch, tuple)
        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_standardized,
            cat_ixs,
            confs,
            normalized_bardist_,
            bar_distribution,
            x_test_raw,
            y_test_raw,
        ) = batch
        for est_tensor in X_trains_preprocessed:
            assert isinstance(est_tensor, torch.Tensor)
            assert est_tensor.shape[0] == batch_size
        for est_tensor in y_trains_preprocessed:
            assert isinstance(est_tensor, torch.Tensor)
            assert est_tensor.shape[0] == batch_size
        assert isinstance(cat_ixs, list)
        for conf in confs:
            for c in conf:
                assert isinstance(c, RegressorEnsembleConfig)
        for ren_crit in normalized_bardist_:
            assert isinstance(ren_crit, FullSupportBarDistribution)
        for bar_dist in bar_distribution:
            assert isinstance(bar_dist, BarDistribution)
        break


@pytest.mark.parametrize(param_order, combinations)
def test_tabpfn_regressor_finetuning_loop(
    n_estimators,
    device,
    fit_mode,
    inference_precision,
    optimization_space,
    synthetic_regression_data,
) -> None:
    X, y = synthetic_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    reg = create_regressor(
        n_estimators,
        device,
        fit_mode,
        inference_precision,
        random_state=2,
        differentiable_input=False,
    )

    datasets_list = reg.get_preprocessed_datasets(
        X_train, y_train, train_test_split, 100
    )

    batch_size = 1
    my_dl_train = DataLoader(
        datasets_list, batch_size=batch_size, collate_fn=meta_dataset_collator
    )

    optim_impl = Adam(reg.model_.parameters(), lr=1e-5)

    if inference_precision == torch.float64:
        pass
        # TODO: check that it fails with the right error

    elif fit_mode in [
        "fit_preprocessors",
        "fit_with_cache",
        "low_memory",
    ]:
        # TODO: check that it fails with the right error
        pass
    else:
        for data_batch in my_dl_train:
            optim_impl.zero_grad()

            (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_preprocessed,
                y_test_standardized,
                cat_ixs,
                confs,
                normalized_bardist_,
                bar_distribution,
                batch_x_test_raw,
                batch_y_test_raw,
            ) = data_batch

            reg.fit_from_preprocessed(
                X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
            )

            reg.normalized_bardist_ = normalized_bardist_[0]

            averaged_pred_logits, _, _ = reg.forward(X_tests_preprocessed)

            # --- Basic Shape Checks ---
            assert (
                averaged_pred_logits.ndim == 3
            ), f"Expected 3D output, got {averaged_pred_logits.shape}"

            # Batch Size
            assert averaged_pred_logits.shape[0] == batch_y_test_raw.shape[0]
            assert averaged_pred_logits.shape[0] == batch_size
            assert averaged_pred_logits.shape[0] == X_tests_preprocessed[0].shape[0]
            assert averaged_pred_logits.shape[0] == y_test_standardized.shape[0]

            # N_samples
            assert averaged_pred_logits.shape[1] == batch_y_test_raw.shape[1]
            assert averaged_pred_logits.shape[1] == y_test_standardized.shape[1]

            # N_bins
            n_borders_bardist = reg.bardist_.borders.shape[0]
            assert averaged_pred_logits.shape[2] == n_borders_bardist - 1
            n_borders_norm_crit = reg.normalized_bardist_.borders.shape[0]
            assert averaged_pred_logits.shape[2] == n_borders_norm_crit - 1

            assert len(X_tests_preprocessed) == reg.n_estimators
            assert len(X_trains_preprocessed) == reg.n_estimators
            assert len(y_trains_preprocessed) == reg.n_estimators
            assert reg.model_ is not None, "Model not initialized after fit"
            assert hasattr(
                reg, "bardist_"
            ), "Regressor missing 'bardist_' attribute after fit"
            assert hasattr(
                reg, "normalized_bardist_"
            ), "Regressor missing 'normalized_bardist_' attribute after fit"
            assert reg.bardist_ is not None, "reg.bardist_ is None"

            lossfn = None
            if optimization_space == "raw_label_space":
                lossfn = reg.bardist_
            elif optimization_space == "preprocessed":
                lossfn = reg.normalized_bardist_
            else:
                raise ValueError("Need to define optimization space")

            nll_loss_per_sample = lossfn(
                averaged_pred_logits, batch_y_test_raw.to(device)
            )
            loss = nll_loss_per_sample.mean()

            # --- Gradient Check ---
            loss.backward()
            optim_impl.step()

            assert torch.isfinite(loss).all(), f"Loss is not finite: {loss.item()}"

            gradients_found = False
            for param in reg.model_.parameters():
                if (
                    param.requires_grad
                    and param.grad is not None
                    and param.grad.abs().sum().item() > 1e-12
                ):
                    gradients_found = True
                    break
            assert gradients_found, "No non-zero gradients found."

            reg.model_.zero_grad()
            break  # Only test one batch


def test_finetuning_consistency_bar_distribution(
    std_regressor_instance, ft_regressor_instance, synthetic_regression_data
):
    """Tests if predict() output matches the output derived from
    get_preprocessed_datasets -> fit_from_preprocessed -> forward() -> post-processing,
    when no actual fine-tuning occurs.
    """
    common_seed = 10
    test_set_size = 0.2

    reg_standard = std_regressor_instance
    reg_batched = ft_regressor_instance

    if reg_standard.device != reg_batched.device:
        pytest.skip("Devices do not match.")

    x_full_raw, y_full_raw = synthetic_regression_data

    splitfn = partial(
        train_test_split,
        test_size=test_set_size,
        random_state=common_seed,
        shuffle=False,
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = splitfn(x_full_raw, y_full_raw)

    reg_standard.fit(X_train_raw, y_train_raw)
    reg_standard.predict(X_test_raw, output_type="mean")

    datasets_list = reg_batched.get_preprocessed_datasets(
        x_full_raw, y_full_raw, splitfn, max_data_size=1000
    )

    batch_size = 1
    dataloader = DataLoader(
        datasets_list,
        batch_size=batch_size,
        collate_fn=meta_dataset_collator,
        shuffle=False,
    )
    data_batch = next(iter(dataloader))
    (
        X_trains_preprocessed,
        X_tests_preprocessed,
        y_trains_preprocessed,
        y_test_standardized,
        cat_ixs,
        confs,
        normalized_bardist_,
        bar_distribution,
        batch_x_test_raw,
        batch_y_test_raw,
    ) = data_batch

    np.testing.assert_allclose(
        batch_y_test_raw.flatten().detach().cpu().numpy(),
        y_test_raw,
        rtol=1e-5,
        atol=1e-5,
    )

    reg_batched.fit_from_preprocessed(
        X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
    )

    mean = np.mean(y_train_raw)
    std = np.std(y_train_raw)
    y_train_std_ = std.item() + 1e-20
    y_train_mean_ = mean.item()
    y_standardised_investigated = (y_test_raw - y_train_mean_) / y_train_std_

    np.testing.assert_allclose(
        y_test_standardized[0].flatten().detach().cpu().numpy(),
        y_standardised_investigated,
        rtol=1e-5,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        batch_x_test_raw[0].detach().cpu().numpy(),
        X_test_raw,
        rtol=1e-5,
        atol=1e-5,
    )

    normalized_bardist_ = normalized_bardist_[0]
    reg_batched.normalized_bardist_ = normalized_bardist_

    torch.testing.assert_close(
        normalized_bardist_.borders,
        reg_batched.normalized_bardist_.borders,
        rtol=1e-5,
        atol=1e-5,
        msg="Renormalized criterion borders do not match.",
    )

    torch.testing.assert_close(
        normalized_bardist_.borders,
        reg_standard.normalized_bardist_.borders,
        rtol=1e-5,  # Standard float tolerance
        atol=1e-5,
        msg="Renormalized criterion borders do not match.",
    )

    torch.testing.assert_close(
        reg_standard.normalized_bardist_.borders,
        reg_batched.normalized_bardist_.borders,
        rtol=1e-5,  # Standard float tolerance
        atol=1e-5,
        msg="Renormalized criterion borders do not match.",
    )

    torch.testing.assert_close(
        reg_standard.bardist_.borders,
        reg_batched.bardist_.borders,
        rtol=1e-5,  # Standard float tolerance
        atol=1e-5,
        msg="Bar distribution borders do not match.",
    )


# ----------------


class TestTabPFNPreprocessingInspection(unittest.TestCase):
    def test_finetuning_consistency_preprocessing_regressor(self):
        """In order to test the consistency of our FineTuning code
        and the preprocessing code, we will test the consistency
        of the preprocessed datasets. We do this by checking
        comparing the tensors that enter the internal transformer
        model.
        """
        test_set_size = 0.3
        common_seed = 42
        n_total = 20
        n_features = 10
        n_estimators = 1

        X, y = sklearn.datasets.make_regression(
            n_samples=n_total, n_features=n_features, random_state=common_seed
        )
        splitfn = partial(
            train_test_split,
            test_size=test_set_size,
            random_state=common_seed,
            shuffle=False,  # Keep False for consistent results if slicing were needed
        )
        X_train_raw, X_test_raw, y_train_raw, _ = splitfn(X, y)

        # Initialize two regressors with the inference and FineTuning
        reg_standard = TabPFNRegressor(
            n_estimators=n_estimators,
            device="cpu",
            random_state=common_seed,
            fit_mode="fit_preprocessors",  # Example standard mode
        )
        reg_batched = TabPFNRegressor(
            n_estimators=n_estimators,
            device="cpu",
            random_state=common_seed,
            fit_mode="batched",  # Mode compatible with get_preprocessed_datasets
        )

        # --- 2. Path 1: Standard fit -> predict -> Capture Tensor ---
        reg_standard.fit(X_train_raw, y_train_raw)
        assert hasattr(reg_standard, "model_")
        assert hasattr(reg_standard.model_, "forward")

        tensor_p1_full = None
        # Patch the standard regressor's internal model's forward method
        with patch.object(
            reg_standard.model_, "forward", wraps=reg_standard.model_.forward
        ) as mock_forward_p1:
            _ = reg_standard.predict(X_test_raw)  # Trigger the patched method
            assert mock_forward_p1.called
            # Capture the tensor input to the internal model
            tensor_p1_full = mock_forward_p1.call_args.args[1]

        assert tensor_p1_full is not None
        # Standard path's internal model receives the combined train+test sequence
        assert tensor_p1_full.shape[0] == n_total

        # --- 3. Path 3: FT Full Workflow ---
        # (get_prep -> fit_prep -> forward -> Capture Tensor)

        datasets_list = reg_batched.get_preprocessed_datasets(
            X, y, splitfn, max_data_size=1000
        )

        # Fit FT regressor
        dataloader = DataLoader(
            datasets_list,
            batch_size=1,
            collate_fn=meta_dataset_collator,
            shuffle=False,
        )
        data_batch = next(iter(dataloader))
        (
            X_trains_p2,
            X_tests_p2,
            y_trains_p2,
            _,
            cat_ixs_p2,
            confs_p2,
            _,
            _,
            _,
            _,
        ) = data_batch
        reg_batched.fit_from_preprocessed(
            X_trains_p2, y_trains_p2, cat_ixs_p2, confs_p2
        )
        assert hasattr(reg_batched, "model_")
        assert hasattr(reg_batched.model_, "forward")

        # Step 3c: Call forward and capture the input tensor to the *internal model*
        tensor_p3_full = None
        # Patch the *batched* regressor's internal model's forward method
        with patch.object(
            reg_batched.model_, "forward", wraps=reg_batched.model_.forward
        ) as mock_forward_p3:
            # Pass the list of preprocessed test tensors obtained earlier
            _ = reg_batched.forward(X_tests_p2)
            assert mock_forward_p3.called
            # Capture the tensor input to the internal model
            tensor_p3_full = mock_forward_p3.call_args.args[1]

        assert tensor_p3_full is not None
        # As confirmed before, the internal model in this path
        # also receives the full sequence
        assert tensor_p3_full.shape[0] == n_total

        # --- 4. Comparison (Path 1 vs Path 3) ---

        # Compare the two full tensors captured from the input to model_.forward
        # Squeeze dimensions of size 1 for direct comparison
        # shapes should be [N_Total, Features+1]
        p1_squeezed = tensor_p1_full.squeeze()
        p3_squeezed = tensor_p3_full.squeeze()

        assert (
            p1_squeezed.shape == p3_squeezed.shape
        ), "Shapes of final model input tensors mismatch."

        atol = 1e-6
        tensors_match = torch.allclose(p1_squeezed, p3_squeezed, atol=atol)

        assert tensors_match, "Mismatch between preprocessed model input tensors."
