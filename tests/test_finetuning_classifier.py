"""Tests for TabPFN classifier finetuning functionality.

This module contains tests for:
- The FinetunedTabPFNClassifier wrapper class
- Dataset preprocessing utilities for finetuning
- Data collation and batching for training loops
- Checkpoint saving/loading functionality
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
import sklearn
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tabpfn import TabPFNClassifier
from tabpfn.finetuning.data_util import (
    DatasetCollectionWithPreprocessing,
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)
from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.train_util import get_checkpoint_path_and_epoch_from_output_dir
from tabpfn.preprocessing import ClassifierEnsembleConfig

from .utils import get_pytest_devices

rng = np.random.default_rng(42)

devices = get_pytest_devices()

FitMode = Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"]

fit_modes: list[FitMode] = [
    "batched",
    "fit_preprocessors",
]

inference_precision_methods: list[torch.types._dtype | Literal["autocast", "auto"]] = [
    "auto",
    torch.float64,
]
estimators = [1, 2]

param_order = [
    "device",
    "n_estimators",
    "fit_mode",
    "inference_precision",
]

default_config: dict[str, Any] = {
    "n_estimators": 1,
    "device": "auto",
    "fit_mode": "batched",
    "inference_precision": "auto",
}

param_values: dict[str, list] = {
    "n_estimators": estimators,
    "device": devices,
    "fit_mode": fit_modes,
    "inference_precision": inference_precision_methods,
}

combinations = [tuple(default_config[param] for param in param_order)]
for param in param_order:
    for value in param_values[param]:
        if value != default_config[param]:
            config = default_config.copy()
            config[param] = value
            combinations.append(tuple(config[param] for param in param_order))


# =============================================================================
# Parametrization for FinetunedTabPFNClassifier tests
# =============================================================================

finetuned_param_order = [
    "device",
    "early_stopping",
    "use_lr_scheduler",
]

finetuned_default_config: dict[str, Any] = {
    "device": "cpu",
    "early_stopping": False,
    "use_lr_scheduler": False,
}

finetuned_param_values: dict[str, list] = {
    "device": devices,
    "early_stopping": [True, False],
    "use_lr_scheduler": [True, False],
}

# Generate combinations varying one parameter at a time from defaults
finetuned_combinations = [
    tuple(finetuned_default_config[param] for param in finetuned_param_order)
]
for param in finetuned_param_order:
    for value in finetuned_param_values[param]:
        if value != finetuned_default_config[param]:
            config = finetuned_default_config.copy()
            config[param] = value
            finetuned_combinations.append(
                tuple(config[param] for param in finetuned_param_order)
            )


def create_mock_architecture_forward(
    n_classes: int,
) -> Callable[..., torch.Tensor]:
    """Create a side_effect function for mocking the Architecture forward method.

    The Architecture.forward method signature is:
    forward(x, y, *, only_return_standard_out=True, categorical_inds=None)

    Where:
    - x has shape (train+test rows, batch_size, num_features)
    - y has shape (train rows, batch_size) or (train rows, batch_size, 1)
    - returns shape (test rows, batch_size, num_classes)

    Args:
        n_classes: Number of classes for the classification task.

    Returns:
        A mock forward function that returns random logits with correct shape.
    """

    def mock_forward(
        self: torch.nn.Module,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        **_kwargs: bool,
    ) -> torch.Tensor:
        """Mock forward pass that returns random logits."""
        if isinstance(x, dict):
            x = x["main"]

        if y is not None:
            y_tensor = y["main"] if isinstance(y, dict) else y
            num_train_rows = y_tensor.shape[0]
        else:
            num_train_rows = 0

        # x has shape (train+test rows, batch_size, num_features)
        total_rows = x.shape[0]
        batch_size = x.shape[1]
        num_test_rows = total_rows - num_train_rows

        # Touch a model parameter so gradients flow during backward pass.
        # This is needed for GradScaler to record inf checks on CUDA
        # on older torch versions.
        first_param = next(self.parameters())
        param_contribution = 0.0 * first_param.sum()

        # Return shape (test_rows, batch_size, num_classes)
        return (
            torch.randn(
                num_test_rows,
                batch_size,
                n_classes,
                requires_grad=True,
                device=x.device,
            )
            + param_contribution
        )

    return mock_forward


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing."""
    X = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=100).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def uniform_synthetic_dataset_collection() -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixture: synthetic collection of datasets (list of (X, y) tuples)."""
    datasets = []
    for _ in range(3):
        X = rng.normal(size=(30, 3)).astype(np.float32)
        y = rng.integers(0, 3, size=30)
        datasets.append((X, y))
    return datasets


@pytest.fixture(scope="module")
def classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate simple classification data."""
    X, y = make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    y = np.asarray(y)
    y = y - y.min()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
    )


@pytest.fixture(params=devices)
def classifier_instance(request: pytest.FixtureRequest) -> TabPFNClassifier:
    """Provides a basic classifier instance, parameterized by device."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")
    return TabPFNClassifier(
        n_estimators=2,
        device=device,
        random_state=42,
        inference_precision=torch.float32,
        fit_mode="batched",
        differentiable_input=False,
    )


def create_classifier(
    n_estimators: int,
    device: str,
    fit_mode: FitMode,
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    **kwargs: Any,
) -> TabPFNClassifier:
    """Instantiates classifier with common parameters."""
    if device == "cpu" and inference_precision == "autocast":
        pytest.skip("Unsupported combination: CPU with 'autocast'")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    default_kwargs: dict[str, Any] = {"random_state": 42}
    default_kwargs.update(kwargs)

    return TabPFNClassifier(
        n_estimators=n_estimators,
        device=device,
        fit_mode=fit_mode,
        inference_precision=inference_precision,
        **default_kwargs,
    )


@pytest.fixture
def variable_synthetic_dataset_collection() -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixture: synthetic collection of datasets with varying sizes and classes."""
    datasets = []
    dataset_sizes = [10, 20, 30]
    class_counts = [2, 4, 6]
    n_features = 3
    for size, n_classes in zip(dataset_sizes, class_counts):
        X = rng.normal(size=(size, n_features)).astype(np.float32)
        y = rng.integers(0, n_classes, size=size)
        datasets.append((X, y))
    return datasets


# =============================================================================
# Tests for FinetunedTabPFNClassifier (Training Loop)
# =============================================================================


@pytest.mark.parametrize(
    ("device", "early_stopping", "use_lr_scheduler"),
    finetuned_combinations,
)
def test_finetuned_tabpfn_classifier_fit_and_predict(
    device: str,
    early_stopping: bool,
    use_lr_scheduler: bool,
    synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test FinetunedTabPFNClassifier with various parameter combinations.

    Uses mocked Architecture.forward for fast test execution.
    Tests combinations of device, early_stopping, use_lr_scheduler,
    and n_estimators_finetune.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    X, y = synthetic_data
    n_classes = len(np.unique(y))
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)

    epochs = 4 if early_stopping else 2

    finetuned_clf = FinetunedTabPFNClassifier(
        device=device,
        epochs=epochs,
        learning_rate=1e-4,
        validation_split_ratio=0.2,
        n_finetune_ctx_plus_query_samples=50,
        finetune_ctx_query_split_ratio=0.1,
        n_inference_subsample_samples=100,
        random_state=42,
        early_stopping=early_stopping,
        early_stopping_patience=2,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        use_lr_scheduler=use_lr_scheduler,
        lr_warmup_only=False,
    )

    mock_forward = create_mock_architecture_forward(n_classes=n_classes)

    with mock.patch(
        "tabpfn.architectures.base.transformer.PerFeatureTransformer.forward",
        autospec=True,
        side_effect=mock_forward,
    ):
        finetuned_clf.fit(X_train, y_train)

    assert finetuned_clf.is_fitted_
    assert hasattr(finetuned_clf, "finetuned_classifier_")
    assert hasattr(finetuned_clf, "finetuned_inference_classifier_")

    probabilities = finetuned_clf.predict_proba(X_test)
    assert probabilities.shape[0] == X_test.shape[0]
    assert probabilities.shape[1] == n_classes
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    predictions = finetuned_clf.predict(X_test)
    assert predictions.shape[0] == X_test.shape[0]
    assert all(pred in np.unique(y_train) for pred in predictions)


# =============================================================================
# Tests for Checkpoint Saving and Loading
# =============================================================================


@pytest.mark.parametrize("device", devices)
def test_checkpoint_saving_and_loading(
    device: str,
    tmp_path: Path,
    synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that checkpoints are saved and can be loaded correctly."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    X, y = synthetic_data
    n_classes = len(np.unique(y))
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    output_folder = tmp_path / "checkpoints"

    finetuned_clf = FinetunedTabPFNClassifier(
        device=device,
        epochs=4,
        learning_rate=1e-5,
        validation_split_ratio=0.2,
        n_finetune_ctx_plus_query_samples=50,
        finetune_ctx_query_split_ratio=0.1,
        n_inference_subsample_samples=100,
        random_state=42,
        early_stopping=False,
        use_lr_scheduler=False,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        save_checkpoint_interval=2,
    )

    mock_forward = create_mock_architecture_forward(n_classes=n_classes)

    with mock.patch(
        "tabpfn.architectures.base.transformer.PerFeatureTransformer.forward",
        autospec=True,
        side_effect=mock_forward,
    ):
        finetuned_clf.fit(X_train, y_train, output_dir=output_folder)

    # Verify interval checkpoints exist
    checkpoint_files = sorted(output_folder.glob("checkpoint_*_[0-9]*.pth"))
    assert len(checkpoint_files) >= 2, (
        "At least two interval checkpoints should be saved"
    )

    # Check specific epoch checkpoints
    epoch_2_checkpoint = output_folder / "checkpoint_70_2.pth"
    assert epoch_2_checkpoint.exists(), "Checkpoint at epoch 2 should exist"

    epoch_4_checkpoint = output_folder / "checkpoint_70_4.pth"
    assert epoch_4_checkpoint.exists(), "Checkpoint at epoch 4 should exist"

    # Verify best checkpoint exists
    best_checkpoint_path = output_folder / "checkpoint_70_best.pth"
    assert best_checkpoint_path.exists(), "Best checkpoint should exist"

    # Verify checkpoint structure
    checkpoint = torch.load(epoch_2_checkpoint, weights_only=False)
    assert "state_dict" in checkpoint, "Checkpoint should contain state_dict"
    assert "config" in checkpoint, "Checkpoint should contain config"
    assert "optimizer" in checkpoint, "Checkpoint should contain optimizer state"
    assert "epoch" in checkpoint, "Checkpoint should contain epoch number"
    assert "roc_auc" in checkpoint, "Checkpoint should contain roc_auc"
    assert "log_loss" in checkpoint, "Checkpoint should contain log_loss"
    assert checkpoint["epoch"] == 2, "Checkpoint should be from epoch 2"

    # Verify best checkpoint structure
    best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
    assert "state_dict" in best_checkpoint
    assert "epoch" in best_checkpoint
    assert isinstance(best_checkpoint["epoch"], int)
    assert best_checkpoint["epoch"] > 0


@pytest.mark.parametrize("device", devices)
def test_checkpoint_resumption(
    device: str,
    tmp_path: Path,
    synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that training can be resumed from a checkpoint."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    X, y = synthetic_data
    n_classes = len(np.unique(y))
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)

    output_folder = tmp_path / "checkpoints_resume"

    # Train for 2 epochs with checkpoint_interval=2
    finetuned_clf = FinetunedTabPFNClassifier(
        device=device,
        epochs=2,
        learning_rate=1e-5,
        validation_split_ratio=0.2,
        n_finetune_ctx_plus_query_samples=50,
        finetune_ctx_query_split_ratio=0.1,
        n_inference_subsample_samples=100,
        random_state=42,
        early_stopping=False,
        use_lr_scheduler=False,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        save_checkpoint_interval=2,
    )

    mock_forward = create_mock_architecture_forward(n_classes=n_classes)

    with mock.patch(
        "tabpfn.architectures.base.transformer.PerFeatureTransformer.forward",
        autospec=True,
        side_effect=mock_forward,
    ):
        finetuned_clf.fit(X_train, y_train, output_dir=output_folder)

    epoch_2_checkpoint = output_folder / "checkpoint_70_2.pth"
    assert epoch_2_checkpoint.exists(), "Checkpoint at epoch 2 should exist"

    best_checkpoint_path = output_folder / "checkpoint_70_best.pth"
    assert best_checkpoint_path.exists(), (
        "Best checkpoint should exist after first training"
    )

    # Resume training for another 2 epochs (total 4)
    finetuned_clf_resumed = FinetunedTabPFNClassifier(
        device=device,
        epochs=4,  # Total epochs = 4
        learning_rate=1e-5,
        validation_split_ratio=0.2,
        n_finetune_ctx_plus_query_samples=50,
        finetune_ctx_query_split_ratio=0.1,
        n_inference_subsample_samples=100,
        random_state=42,
        early_stopping=False,
        use_lr_scheduler=False,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        save_checkpoint_interval=2,
    )

    with mock.patch(
        "tabpfn.architectures.base.transformer.PerFeatureTransformer.forward",
        autospec=True,
        side_effect=mock_forward,
    ):
        finetuned_clf_resumed.fit(X_train, y_train, output_dir=output_folder)

    epoch_4_checkpoint = output_folder / "checkpoint_70_4.pth"
    assert epoch_4_checkpoint.exists(), (
        "Checkpoint at epoch 4 should exist after resumption"
    )

    assert best_checkpoint_path.exists()

    # Verify predictions work
    y_pred_proba = finetuned_clf_resumed.predict_proba(X_test)
    y_pred = finetuned_clf_resumed.predict(X_test)

    assert y_pred_proba.shape == (len(X_test), n_classes)
    assert y_pred.shape == (len(X_test),)
    assert finetuned_clf_resumed.is_fitted_


def test_checkpoint_epoch_offset_extraction(tmp_path: Path) -> None:
    """Test that epoch offset is correctly extracted from checkpoint filenames."""
    # Test with no checkpoints
    output_folder = tmp_path / "no_checkpoints"
    output_folder.mkdir(parents=True, exist_ok=True)
    model_path, epoch = get_checkpoint_path_and_epoch_from_output_dir(
        output_folder, train_size=100
    )
    assert model_path is None
    assert epoch == 0

    # Test with one checkpoint file
    output_folder_with_checkpoint = tmp_path / "with_checkpoint"
    output_folder_with_checkpoint.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_folder_with_checkpoint / "checkpoint_100_10.pth"
    torch.save({"dummy": "data"}, checkpoint_path)

    model_path, epoch = get_checkpoint_path_and_epoch_from_output_dir(
        output_folder_with_checkpoint, train_size=100, get_best=False
    )
    assert model_path == checkpoint_path
    assert epoch == 10

    # Test with multiple checkpoint files (should return the last one)
    checkpoint_path_20 = output_folder_with_checkpoint / "checkpoint_100_20.pth"
    torch.save({"dummy": "data"}, checkpoint_path_20)

    model_path, epoch = get_checkpoint_path_and_epoch_from_output_dir(
        output_folder_with_checkpoint, train_size=100, get_best=False
    )
    assert model_path == checkpoint_path_20
    assert epoch == 20

    # Test with best checkpoint (should prioritize best over numbered)
    best_checkpoint_path = output_folder_with_checkpoint / "checkpoint_100_best.pth"
    torch.save({"epoch": 15, "dummy": "data"}, best_checkpoint_path)

    model_path, epoch = get_checkpoint_path_and_epoch_from_output_dir(
        output_folder_with_checkpoint, train_size=100, get_best=True
    )
    assert model_path == best_checkpoint_path
    assert epoch == 15


@pytest.mark.parametrize("device", devices)
def test_checkpoint_interval_configuration(
    device: str,
    tmp_path: Path,
    synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that checkpoint_interval parameter works correctly."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    X, y = synthetic_data
    n_classes = len(np.unique(y))
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    output_folder = tmp_path / "checkpoint_interval_test"

    # Train for 6 epochs with checkpoint_interval=3
    finetuned_clf = FinetunedTabPFNClassifier(
        device=device,
        epochs=6,
        learning_rate=1e-5,
        validation_split_ratio=0.2,
        n_finetune_ctx_plus_query_samples=50,
        finetune_ctx_query_split_ratio=0.1,
        n_inference_subsample_samples=100,
        random_state=42,
        early_stopping=False,
        use_lr_scheduler=False,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        save_checkpoint_interval=3,
    )

    mock_forward = create_mock_architecture_forward(n_classes=n_classes)

    with mock.patch(
        "tabpfn.architectures.base.transformer.PerFeatureTransformer.forward",
        autospec=True,
        side_effect=mock_forward,
    ):
        finetuned_clf.fit(X_train, y_train, output_dir=output_folder)

    # Verify checkpoints saved at epochs 3, 6
    epoch_3_checkpoint = output_folder / "checkpoint_70_3.pth"
    epoch_6_checkpoint = output_folder / "checkpoint_70_6.pth"

    assert epoch_3_checkpoint.exists(), "Checkpoint at epoch 3 should exist"
    assert epoch_6_checkpoint.exists(), "Checkpoint at epoch 6 should exist"

    # Verify no checkpoints at epochs 1, 2, 4, 5
    for epoch in [1, 2, 4, 5]:
        checkpoint_path = output_folder / f"checkpoint_70_{epoch}.pth"
        assert not checkpoint_path.exists(), (
            f"Checkpoint at epoch {epoch} should not exist"
        )

    # Verify best checkpoint exists
    best_checkpoint_path = output_folder / "checkpoint_70_best.pth"
    assert best_checkpoint_path.exists(), "Best checkpoint should exist"


@pytest.mark.parametrize("device", devices)
def test_best_checkpoint_saving(
    device: str,
    tmp_path: Path,
    synthetic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that best checkpoint is saved correctly."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    X, y = synthetic_data
    n_classes = len(np.unique(y))
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    output_folder = tmp_path / "best_checkpoint_test"

    # Train for 3 epochs with checkpoint_interval=None (no interval saves)
    finetuned_clf = FinetunedTabPFNClassifier(
        device=device,
        epochs=3,
        learning_rate=1e-5,
        validation_split_ratio=0.2,
        n_finetune_ctx_plus_query_samples=50,
        finetune_ctx_query_split_ratio=0.1,
        n_inference_subsample_samples=100,
        random_state=42,
        early_stopping=False,
        use_lr_scheduler=False,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        save_checkpoint_interval=None,  # Disable interval checkpoints
    )

    mock_forward = create_mock_architecture_forward(n_classes=n_classes)

    with mock.patch(
        "tabpfn.architectures.base.transformer.PerFeatureTransformer.forward",
        autospec=True,
        side_effect=mock_forward,
    ):
        finetuned_clf.fit(X_train, y_train, output_dir=output_folder)

    # Verify that best checkpoint exists
    best_checkpoint_path = output_folder / "checkpoint_70_best.pth"
    assert best_checkpoint_path.exists(), "Best checkpoint should exist"

    # Verify no interval checkpoints exist (since checkpoint_interval=None)
    interval_checkpoints = list(output_folder.glob("checkpoint_70_[0-9]*.pth"))
    assert len(interval_checkpoints) == 0, (
        "No interval checkpoints should exist when checkpoint_interval=None"
    )


# =============================================================================
# Tests for Dataset Preprocessing Utilities
# =============================================================================


def test_get_preprocessed_datasets_basic() -> None:
    """Test basic functionality of get_preprocessed_datasets_helper."""
    X = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=100)

    clf = TabPFNClassifier()
    dataset = get_preprocessed_dataset_chunks(
        clf,
        X,
        y,
        train_test_split,
        100,
        model_type="classifier",
        equal_split_size=True,
        seed=42,
    )
    assert hasattr(dataset, "__getitem__")
    assert hasattr(dataset, "__len__")
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 6


def test_datasetcollectionwithpreprocessing_classification_single_dataset(
    synthetic_data: tuple[np.ndarray, np.ndarray],
    classifier_instance: TabPFNClassifier,
) -> None:
    """Test DatasetCollectionWithPreprocessing with a single dataset."""
    X_raw, y_raw = synthetic_data
    clf = classifier_instance
    n_estimators = clf.n_estimators
    test_size = 0.3

    split_fn = partial(train_test_split, test_size=test_size, shuffle=True)
    dataset_collection = get_preprocessed_dataset_chunks(
        classifier_instance,
        X_raw,
        y_raw,
        split_fn=split_fn,
        max_data_size=None,
        model_type="classifier",
        equal_split_size=True,
        seed=42,
    )

    assert isinstance(dataset_collection, DatasetCollectionWithPreprocessing)
    assert len(dataset_collection) == 1, "Collection should contain one dataset config"

    item_index = 0
    processed_dataset_item = dataset_collection[item_index]

    assert isinstance(processed_dataset_item, tuple)
    assert len(processed_dataset_item) == 6, (
        "Item tuple should have 6 elements for classification"
    )

    (
        X_trains_preprocessed,
        _X_tests_preprocessed,
        _y_trains_preprocessed,
        y_test_raw_tensor,
        _cat_ixs,
        _returned_ensemble_configs,
    ) = processed_dataset_item

    assert isinstance(X_trains_preprocessed, list)
    assert len(X_trains_preprocessed) == n_estimators
    n_samples_total = X_raw.shape[0]
    expected_n_test = int(np.floor(n_samples_total * test_size))
    expected_n_train = n_samples_total - expected_n_test
    assert y_test_raw_tensor.shape == (expected_n_test,)
    assert X_trains_preprocessed[0].shape[0] == expected_n_train


def test_datasetcollectionwithpreprocessing_classification_multiple_datasets(
    uniform_synthetic_dataset_collection: list[tuple[np.ndarray, np.ndarray]],
    classifier_instance: TabPFNClassifier,
) -> None:
    """Test DatasetCollectionWithPreprocessing using multiple synthetic datasets."""
    datasets = uniform_synthetic_dataset_collection
    clf = classifier_instance
    n_estimators = clf.n_estimators
    test_size = 0.3
    split_fn = partial(train_test_split, test_size=test_size, shuffle=True)

    X_list = [X for X, _ in datasets]
    y_list = [y for _, y in datasets]

    dataset_collection = get_preprocessed_dataset_chunks(
        clf,
        X_list,
        y_list,
        split_fn=split_fn,
        max_data_size=None,
        model_type="classifier",
        equal_split_size=True,
        seed=42,
    )

    assert isinstance(dataset_collection, DatasetCollectionWithPreprocessing)
    assert len(dataset_collection) == len(datasets), (
        "Collection should contain one item per dataset"
    )

    for item_index in range(len(datasets)):
        processed_dataset_item = dataset_collection[item_index]
        assert isinstance(processed_dataset_item, tuple)
        assert len(processed_dataset_item) == 6, (
            "Item tuple should have 6 elements for classification"
        )
        (
            X_trains_preprocessed,
            _X_tests_preprocessed,
            _y_trains_preprocessed,
            y_test_raw_tensor,
            _cat_ixs,
            _returned_ensemble_configs,
        ) = processed_dataset_item
        assert isinstance(X_trains_preprocessed, list)
        assert len(X_trains_preprocessed) == n_estimators
        n_samples_total = X_list[item_index].shape[0]
        expected_n_test = int(np.floor(n_samples_total * test_size))
        expected_n_train = n_samples_total - expected_n_test
        assert y_test_raw_tensor.shape == (expected_n_test,)
        assert X_trains_preprocessed[0].shape[0] == expected_n_train


def test_dataset_and_collator_with_dataloader_uniform(
    uniform_synthetic_dataset_collection: list[tuple[np.ndarray, np.ndarray]],
    classifier_instance: TabPFNClassifier,
) -> None:
    """Test dataset and collator integration with DataLoader using uniform data."""
    X_list = [X for X, _ in uniform_synthetic_dataset_collection]
    y_list = [y for _, y in uniform_synthetic_dataset_collection]
    dataset_collection = get_preprocessed_dataset_chunks(
        classifier_instance,
        X_list,
        y_list,
        train_test_split,
        100,
        model_type="classifier",
        equal_split_size=True,
        seed=42,
    )
    batch_size = 1
    dl = DataLoader(
        dataset_collection,
        batch_size=batch_size,
        collate_fn=meta_dataset_collator,
    )
    for batch in dl:
        # Should be tuple with X_trains, X_tests, y_trains, y_tests, cat_ixs, confs
        assert isinstance(batch, tuple)
        X_trains, _X_tests, y_trains, _y_tests, _cat_ixs, _confs = batch
        for est_tensor in X_trains:
            assert isinstance(est_tensor, torch.Tensor), (
                "Each estimator's batch should be a tensor."
            )
            assert est_tensor.shape[0] == batch_size
        for est_tensor in y_trains:
            assert isinstance(est_tensor, torch.Tensor), (
                "Each estimator's batch should be a tensor for labels."
            )
            assert est_tensor.shape[0] == batch_size
        break  # Only check one batch


def test_classifier_dataset_and_collator_batches_type(
    variable_synthetic_dataset_collection: list[tuple[np.ndarray, np.ndarray]],
    classifier_instance: TabPFNClassifier,
) -> None:
    """Test that batches from dataset and collator have correct types."""
    X_list = [X for X, _ in variable_synthetic_dataset_collection]
    y_list = [y for _, y in variable_synthetic_dataset_collection]
    dataset_collection = get_preprocessed_dataset_chunks(
        classifier_instance,
        X_list,
        y_list,
        train_test_split,
        100,
        model_type="classifier",
        equal_split_size=True,
        seed=42,
    )
    batch_size = 1
    dl = DataLoader(
        dataset_collection,
        batch_size=batch_size,
        collate_fn=meta_dataset_collator,
    )
    for batch in dl:
        assert isinstance(batch, tuple)
        X_trains, _X_tests, y_trains, _y_tests, cat_ixs, confs = batch
        for est_tensor in X_trains:
            assert isinstance(est_tensor, torch.Tensor)
            assert est_tensor.shape[0] == batch_size
        for est_tensor in y_trains:
            assert isinstance(est_tensor, torch.Tensor)
            assert est_tensor.shape[0] == batch_size
        assert isinstance(cat_ixs, list)
        for conf in confs:
            for c in conf:
                assert isinstance(c, ClassifierEnsembleConfig)
        break


def test_get_preprocessed_datasets_multiple_datasets(
    classifier_instance: TabPFNClassifier,
) -> None:
    """Test get_preprocessed_datasets_helper with multiple datasets."""
    X1 = rng.standard_normal((10, 4))
    y1 = rng.integers(0, 2, size=10)
    X2 = rng.standard_normal((8, 4))
    y2 = rng.integers(0, 2, size=8)
    datasets = get_preprocessed_dataset_chunks(
        classifier_instance,
        [X1, X2],
        [y1, y2],
        split_fn=lambda x, y: (x, y),
        max_data_size=10_000,
        model_type="classifier",
        equal_split_size=True,
        seed=42,
    )
    assert hasattr(datasets, "__getitem__")
    assert len(datasets) == 2


def test_get_preprocessed_datasets_categorical_features(
    classifier_instance: TabPFNClassifier,
) -> None:
    """Test get_preprocessed_datasets_helper with categorical features."""
    X = np.array([[0, 1.2, 3.4], [1, 2.3, 4.5], [0, 0.1, 2.2], [2, 1.1, 3.3]])
    y = np.array([0, 1, 0, 1])
    classifier_instance.categorical_features_indices = [0]
    datasets = get_preprocessed_dataset_chunks(
        classifier_instance,
        X,
        y,
        split_fn=lambda x, y: (x, y),
        model_type="classifier",
        max_data_size=None,
        equal_split_size=True,
        seed=42,
    )
    assert hasattr(datasets, "__getitem__")


def test_forward_runs(
    classifier_instance: TabPFNClassifier,
    classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Ensure forward runs OK after standard fit."""
    X_train, X_test, y_train, _y_test = classification_data
    clf = classifier_instance
    clf.fit_mode = "low_memory"
    clf.fit(X_train, y_train)
    preds = clf.forward(
        torch.tensor(X_test, dtype=torch.float32), use_inference_mode=True
    )
    assert preds.ndim == 2, f"Expected 2D output, got {preds.shape}"
    assert preds.shape[0] == X_test.shape[0], "Mismatch in test sample count"
    assert preds.shape[1] == clf.n_classes_, "Mismatch in class count"
    probs_sum = preds.sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), (
        "Probabilities do not sum to 1"
    )


def test_fit_from_preprocessed_runs(
    classifier_instance: TabPFNClassifier,
    classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify fit_from_preprocessed runs and produces valid predictions."""
    X_train, _X_test, y_train, _y_test = classification_data
    clf = classifier_instance

    split_fn = partial(train_test_split, test_size=0.3, random_state=42)

    datasets_list = get_preprocessed_dataset_chunks(
        clf,
        X_train,
        y_train,
        split_fn,
        100,
        model_type="classifier",
        equal_split_size=True,
        seed=42,
    )
    batch_size = 1
    dl = DataLoader(
        datasets_list, batch_size=batch_size, collate_fn=meta_dataset_collator
    )

    for data_batch in dl:
        X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
        clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
        preds = clf.forward(X_tests)
        assert preds.ndim == 3, f"Expected 3D output, got {preds.shape}"
        assert preds.shape[0] == X_tests[0].shape[0]
        assert preds.shape[0] == y_tests.shape[0]
        assert preds.shape[1] == clf.n_classes_

        probs_sum = preds.sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), (
            "Probabilities do not sum to 1"
        )
        break


def test_finetuning_consistency_preprocessing_classifier() -> None:
    """Test consistency between standard and finetuning preprocessing pipelines.

    Compares tensors entering the internal model for:
    - Standard fit -> predict_proba path
    - Batched fit_from_preprocessed -> forward path
    """
    test_set_size = 0.3
    common_seed = 42
    n_total = 50  # Increased slightly for more robust testing
    n_features = 8
    n_classes = 2  # Use a specific number of classes
    n_informative = 5  # For make_classification
    n_estimators = 1  # Keep N=1 for easier direct comparison of tensors

    # --- 1. Setup ---
    X, y = sklearn.datasets.make_classification(
        n_samples=n_total,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=common_seed,
    )
    splitfn = partial(
        train_test_split,
        test_size=test_set_size,
        random_state=common_seed,
        shuffle=False,  # Keep False for consistent splitting
    )
    X_train_raw, X_test_raw, y_train_raw, _y_test_raw = splitfn(X, y)

    # Initialize two classifiers with the necessary modes
    clf_standard = TabPFNClassifier(
        n_estimators=n_estimators,
        device="auto",
        random_state=common_seed,
        fit_mode="fit_preprocessors",  # A standard mode that preprocesses on fit
    )
    # 'batched' mode is required for get_preprocessed_datasets
    #  and fit_from_preprocessed
    clf_batched = TabPFNClassifier(
        n_estimators=n_estimators,
        device="auto",
        random_state=common_seed,
        fit_mode="batched",
    )

    # --- 2. Path 1: Standard fit -> predict_proba -> Capture Tensor ---

    clf_standard.fit(X_train_raw, y_train_raw)
    # Ensure the internal model attribute exists after fit
    assert hasattr(clf_standard, "models_"), (
        "Standard classifier models_ not found after fit."
    )
    assert hasattr(clf_standard.models_[0], "forward"), (
        "Standard classifier models_[0].forward not found after fit."
    )

    tensor_p1_full = None
    # Patch the standard classifier's *internal model's* forward method
    # The internal model typically receives the combined train+test sequence
    with patch.object(
        clf_standard.models_[0], "forward", wraps=clf_standard.models_[0].forward
    ) as mock_forward_p1:
        _ = clf_standard.predict_proba(X_test_raw)
        assert mock_forward_p1.called, "Standard models_[0].forward was not called."

        # Capture the tensor input 'x' (usually the second positional argument)
        call_args_list = mock_forward_p1.call_args_list
        assert len(call_args_list) > 0, (
            "No calls recorded for standard models_[0].forward."
        )
        assert len(call_args_list[0].args) > 1, (
            f"Standard models_[0].forward call had "
            f"unexpected arguments: {call_args_list[0].args}"
        )
        tensor_p1_full = mock_forward_p1.call_args.args[0]

    assert tensor_p1_full is not None, "Failed to capture tensor from standard path."
    # Shape might be [1, N_Total, Features+1] or similar. Check the actual shape.
    # Example assertion: Check if the sequence length matches n_total
    assert tensor_p1_full.shape[0] == n_total, (
        f"Path 1 tensor sequence length ({tensor_p1_full.shape[0]})"
        f"does not match n_total ({n_total}). Shape was {tensor_p1_full.shape}"
    )

    # FT Workflow (get_prep -> fit_prep -> predict_prep -> Capture Tensor) ---
    # Step 3a: Get preprocessed datasets using the *full* dataset
    # Requires fit_mode='batched' on clf_batched
    # Make sure default max_data_size is large enough.
    datasets_list = get_preprocessed_dataset_chunks(
        clf_batched,
        X,
        y,
        split_fn=splitfn,  # Use the full X, y and the split function
        max_data_size=10_000,
        model_type="classifier",
        equal_split_size=True,
        seed=42,
        force_no_stratify=True,
        shuffle=False,
    )
    assert len(datasets_list) > 0, "get_preprocessed_datasets returned empty list."

    dataloader = DataLoader(
        datasets_list,
        batch_size=1,
        collate_fn=meta_dataset_collator,
        shuffle=False,
    )
    data_batch = next(iter(dataloader), None)
    assert data_batch is not None, "DataLoader yielded no batches."

    X_trains_p2, X_tests_p2, y_trains_p2, _, cat_ixs_p2, confs_p2, *_ = data_batch

    clf_batched.fit_from_preprocessed(X_trains_p2, y_trains_p2, cat_ixs_p2, confs_p2)
    assert hasattr(clf_batched, "models_"), (
        "Batched classifier models_ not found after fit_from_preprocessed."
    )
    assert hasattr(clf_batched.models_[0], "forward"), (
        "Batched classifier models_[0].forward not found after fit_from_preprocessed."
    )

    # Step 3c: Call forward and capture the input tensor
    # to the *internal transformer model*
    tensor_p2_full = None
    # Patch the *batched* classifier's internal model's forward method
    with patch.object(
        clf_batched.models_[0], "forward", wraps=clf_batched.models_[0].forward
    ) as mock_forward_p2:
        _ = clf_batched.forward(X_tests_p2)
        assert mock_forward_p2.called, "Batched models_[0].forward was not called."

        # Capture the tensor input 'x' (assuming same argument position as Path 1)
        call_args_list = mock_forward_p2.call_args_list
        assert len(call_args_list) > 0, (
            "No calls recorded for batched models_[0].forward."
        )
        assert len(call_args_list[0].args) > 1, (
            f"Batched models_[0].forward call had "
            f"unexpected arguments: {call_args_list[0].args}"
        )
        tensor_p2_full = mock_forward_p2.call_args.args[0]

    assert tensor_p2_full is not None, "Failed to capture tensor from batched path."
    # The internal model in this path should
    # also receive the full sequence if n_estimators=1
    # and the dataloader yielded the full split.
    assert tensor_p2_full.shape[0] == n_total, (
        f"Path 2 tensor sequence length ({tensor_p2_full.shape[0]}) "
        f"does not match n_total ({n_total}). Shape was {tensor_p2_full.shape}"
    )

    # --- 4. Comparison (Path 1 vs Path 2) ---

    # Ensure tensors are on the same device (CPU) for comparison
    tensor_p1_full = tensor_p1_full.cpu().squeeze(0)
    tensor_p2_full = tensor_p2_full.cpu().squeeze(0)

    # Final check of shapes after potential squeeze
    assert tensor_p1_full.shape == tensor_p2_full.shape, (
        "Shapes of final model input tensors mismatch after squeeze."
    )

    # Perform numerical comparison using torch.allclose
    # Use a reasonably small tolerance. Preprocessing should be near-identical.
    # Floating point ops might introduce tiny differences.
    atol = 1e-6
    rtol = 1e-5
    tensors_match = torch.allclose(tensor_p1_full, tensor_p2_full, atol=atol, rtol=rtol)

    if not tensors_match:
        diff = torch.abs(tensor_p1_full - tensor_p2_full)
        # Find where they differ most
        _max_diff_val, max_diff_idx = torch.max(diff.flatten(), dim=0)
        np.unravel_index(max_diff_idx.item(), tensor_p1_full.shape)

    assert tensors_match, "Mismatch between final model input tensors."
