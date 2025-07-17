from __future__ import annotations

from functools import partial
from typing import Any, Literal

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tabpfn import TabPFNClassifier
from tabpfn.preprocessing import (
    ClassifierEnsembleConfig,
    DatasetCollectionWithPreprocessing,
)
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

param_order = [
    "n_estimators",
    "device",
    "fit_mode",
    "inference_precision",
]

default_config = {
    "n_estimators": 1,
    "device": "cpu",
    "fit_mode": "batched",
    "inference_precision": "auto",
}

param_values: dict[str, list[Any]] = {
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


@pytest.fixture(scope="module")
def synthetic_data():
    X = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=100).astype(np.float32)
    return X, y


# Fixture: synthetic collection of datasets (list of (X, y) tuples)
@pytest.fixture(scope="module")
def uniform_synthetic_dataset_collection():
    datasets = []
    for _ in range(3):
        X = rng.normal(size=(30, 3)).astype(np.float32)
        y = rng.integers(0, 3, size=30)
        datasets.append((X, y))
    return datasets


@pytest.fixture(scope="module")
def classification_data():
    """Generate simple classification data."""
    X, y = make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    y = y - y.min()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(params=devices)
def classifier_instance(request) -> TabPFNClassifier:
    """Provides a basic classifier instance, parameterized by device."""
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")
    # Uses defaults similar to the original fixture, but with device param
    return TabPFNClassifier(
        n_estimators=2,
        device=device,
        random_state=42,
        inference_precision=torch.float32,
        fit_mode="batched",
        differentiable_input=False,
    )


# --- Helper to Create Classifier (for fully parameterized tests) ---
def create_classifier(
    n_estimators: int,
    device: str,
    fit_mode: str,
    inference_precision: torch.types._dtype | Literal["autocast", "auto"],
    **kwargs,
) -> TabPFNClassifier:
    """Instantiates classifier with common parameters."""
    if device == "cpu" and inference_precision == "autocast":
        pytest.skip("Unsupported combination: CPU with 'autocast'")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device requested but not available.")

    default_kwargs = {"random_state": 42}
    default_kwargs.update(kwargs)

    return TabPFNClassifier(
        n_estimators=n_estimators,
        device=device,
        fit_mode=fit_mode,
        inference_precision=inference_precision,
        **default_kwargs,
    )


# Fixture: synthetic collection of datasets (list of (X, y) tuples)
# where dataset size and number of classes vary.
@pytest.fixture
def variable_synthetic_dataset_collection():
    """Fixture: synthetic collection of datasets
    (list of (X, y) tuples)
    where dataset size and number of classes vary.
    """
    datasets = []
    dataset_sizes = [10, 20, 30]
    class_counts = [2, 4, 6]
    n_features = 3
    for size, n_classes in zip(dataset_sizes, class_counts):
        X = rng.normal(size=(size, n_features)).astype(np.float32)
        y = rng.integers(0, n_classes, size=size)
        datasets.append((X, y))
    return datasets


@pytest.mark.parametrize(param_order, combinations)
def test_tabpfn_classifier_finetuning_loop(
    n_estimators,
    device,
    fit_mode,
    inference_precision,
    synthetic_data,
) -> None:
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = create_classifier(
        n_estimators,
        device,
        fit_mode,
        inference_precision,
        random_state=2,
        differentiable_input=False,
    )

    datasets_list = clf.get_preprocessed_datasets(
        X_train, y_train, train_test_split, 100
    )
    lossfn = torch.nn.NLLLoss()
    batch_size = 1
    my_dl_train = DataLoader(
        datasets_list, batch_size=batch_size, collate_fn=meta_dataset_collator
    )

    if inference_precision == torch.float64:
        # Expect to raise a ValueError in
        # TODO: check that it fails with the right error
        pass

    elif fit_mode in [
        "fit_preprocessors",
        "fit_with_cache",
        "low_memory",
    ]:
        # TODO: check that it fails with the right error
        pass

    else:
        for data_batch in my_dl_train:
            X_tr, X_te, y_tr, y_te_raw, cat_ixs, confs = data_batch

            # --- Fit and Predict ---
            clf.fit_from_preprocessed(X_tr, y_tr, cat_ixs, confs)
            preds = clf.forward(X_te)

            # --- Basic Shape Checks ---
            assert preds.ndim == 3, f"Expected 3D output, got {preds.shape}"
            assert preds.shape[0] == X_te[0].shape[0]
            assert preds.shape[0] == y_te_raw.shape[0]
            assert preds.shape[1] == clf.n_classes_, "Class count mismatch"
            assert len(X_te) == clf.n_estimators
            assert len(X_tr) == clf.n_estimators
            assert len(y_tr) == clf.n_estimators

            # --- Loss Calculation and Backward Pass ---
            log_preds = torch.log(preds + 1e-12)
            # Target shape needs adjustment for NLLLoss (B, N) -> (B*N)
            # Prediction shape (B, Cls, N) -> (B*N, Cls)
            target = y_te_raw.to(preds.device).long()
            target = target.view(-1)  # Flatten target to (B*N)
            # Permute and flatten preds: (B, Cls, N) -> (B, N, Cls) -> (B*N, Cls)
            log_preds_permuted = log_preds.permute(0, 2, 1).contiguous()
            log_preds_flat = log_preds_permuted.view(-1, clf.n_classes_)

            assert log_preds_flat.ndim == 2
            assert target.ndim == 1
            assert log_preds_flat.shape[0] == target.shape[0]  # Total samples match

            loss = lossfn(log_preds_flat, target)
            assert torch.isfinite(loss).all(), f"Loss is not finite: {loss.item()}"

            # --- Gradient Check ---
            assert hasattr(clf, "model_"), "Classifier missing 'model_'"
            assert clf.model_ is not None, "Classifier model is None"
            clf.model_.zero_grad()
            loss.backward()

            gradients_found = False
            for param in clf.model_.parameters():
                if (
                    param.requires_grad
                    and param.grad is not None
                    and param.grad.abs().sum().item() > 1e-12
                ):
                    gradients_found = True
                    break
            assert gradients_found, "No non-zero gradients found."
            break  # Only test one batch


def test_get_preprocessed_datasets_basic():
    X = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=100)

    clf = TabPFNClassifier()
    # This should return a DatasetCollectionWithPreprocessing
    dataset = clf.get_preprocessed_datasets(X, y, split_fn=train_test_split)
    assert hasattr(dataset, "__getitem__")
    assert hasattr(dataset, "__len__")
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 6


def test_datasetcollectionwithpreprocessing_classification_single_dataset(
    synthetic_data, classifier_instance: TabPFNClassifier
) -> None:
    X_raw, y_raw = synthetic_data
    clf = classifier_instance
    n_estimators = clf.n_estimators
    test_size = 0.3

    split_fn = partial(train_test_split, test_size=test_size, shuffle=True)
    dataset_collection = clf.get_preprocessed_datasets(
        X_raw, y_raw, split_fn=split_fn, max_data_size=None
    )

    assert isinstance(dataset_collection, DatasetCollectionWithPreprocessing)
    assert len(dataset_collection) == 1, "Collection should contain one dataset config"

    item_index = 0
    processed_dataset_item = dataset_collection[item_index]

    assert isinstance(processed_dataset_item, tuple)
    assert (
        len(processed_dataset_item) == 6
    ), "Item tuple should have 4 elements for classification"

    (
        X_trains_preprocessed,
        X_tests_preprocessed,
        y_trains_preprocessed,
        y_test_raw_tensor,
        cat_ixs,
        returned_ensemble_configs,
    ) = processed_dataset_item

    assert isinstance(X_trains_preprocessed, list)
    assert len(X_trains_preprocessed) == n_estimators
    n_samples_total = X_raw.shape[0]
    expected_n_test = int(np.floor(n_samples_total * test_size))
    expected_n_train = n_samples_total - expected_n_test
    assert y_test_raw_tensor.shape == (expected_n_test,)
    assert X_trains_preprocessed[0].shape[0] == expected_n_train


def test_datasetcollectionwithpreprocessing_classification_multiple_datasets(
    uniform_synthetic_dataset_collection, classifier_instance: TabPFNClassifier
) -> None:
    """Test DatasetCollectionWithPreprocessing
    using a collection of multiple synthetic datasets.
    """
    datasets = uniform_synthetic_dataset_collection
    clf = classifier_instance
    n_estimators = clf.n_estimators
    test_size = 0.3
    split_fn = partial(train_test_split, test_size=test_size, shuffle=True)

    X_list = [X for X, _ in datasets]
    y_list = [y for _, y in datasets]

    dataset_collection = clf.get_preprocessed_datasets(
        X_list, y_list, split_fn=split_fn, max_data_size=None
    )

    assert isinstance(dataset_collection, DatasetCollectionWithPreprocessing)
    assert len(dataset_collection) == len(
        datasets
    ), "Collection should contain one item per dataset"

    for item_index in range(len(datasets)):
        processed_dataset_item = dataset_collection[item_index]
        assert isinstance(processed_dataset_item, tuple)
        assert (
            len(processed_dataset_item) == 6
        ), "Item tuple should have 6 elements for classification"
        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_raw_tensor,
            cat_ixs,
            returned_ensemble_configs,
        ) = processed_dataset_item
        assert isinstance(X_trains_preprocessed, list)
        assert len(X_trains_preprocessed) == n_estimators
        n_samples_total = X_list[item_index].shape[0]
        expected_n_test = int(np.floor(n_samples_total * test_size))
        expected_n_train = n_samples_total - expected_n_test
        assert y_test_raw_tensor.shape == (expected_n_test,)
        assert X_trains_preprocessed[0].shape[0] == expected_n_train


def test_dataset_and_collator_with_dataloader_uniform(
    uniform_synthetic_dataset_collection, classifier_instance
) -> None:
    X_list = [X for X, _ in uniform_synthetic_dataset_collection]
    y_list = [y for _, y in uniform_synthetic_dataset_collection]
    dataset_collection = classifier_instance.get_preprocessed_datasets(
        X_list, y_list, train_test_split, 100
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
        X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = batch
        for est_tensor in X_trains:
            assert isinstance(
                est_tensor, torch.Tensor
            ), "Each estimator's batch should be a tensor."
            assert est_tensor.shape[0] == batch_size
        for est_tensor in y_trains:
            assert isinstance(
                est_tensor, torch.Tensor
            ), "Each estimator's batch should be a tensor for labels."
            assert est_tensor.shape[0] == batch_size
        break  # Only check one batch


def test_classifier_dataset_and_collator_batches_type(
    variable_synthetic_dataset_collection, classifier_instance
):
    """Test that the batches returned by the dataset and collator
    are of the correct type.
    """
    X_list = [X for X, _ in variable_synthetic_dataset_collection]
    y_list = [y for _, y in variable_synthetic_dataset_collection]
    dataset_collection = classifier_instance.get_preprocessed_datasets(
        X_list, y_list, train_test_split, 100
    )
    batch_size = 1
    dl = DataLoader(
        dataset_collection,
        batch_size=batch_size,
        collate_fn=meta_dataset_collator,
    )
    for batch in dl:
        assert isinstance(batch, tuple)
        X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = batch
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


def test_get_preprocessed_datasets_multiple_datasets(classifier_instance):
    # Provide lists of datasets
    X1 = rng.standard_normal((10, 4))
    y1 = rng.integers(0, 2, size=10)
    X2 = rng.standard_normal((8, 4))
    y2 = rng.integers(0, 2, size=8)
    datasets = classifier_instance.get_preprocessed_datasets(
        [X1, X2], [y1, y2], split_fn=None
    )
    assert hasattr(datasets, "__getitem__")
    assert len(datasets) == 2


def test_get_preprocessed_datasets_categorical_features(classifier_instance):
    # One categorical column (e.g., int-coded)
    X = np.array([[0, 1.2, 3.4], [1, 2.3, 4.5], [0, 0.1, 2.2], [2, 1.1, 3.3]])
    y = np.array([0, 1, 0, 1])
    # Specify categorical_features_indices
    classifier_instance.categorical_features_indices = [0]
    datasets = classifier_instance.get_preprocessed_datasets(X, y, split_fn=None)
    # Should not raise, and should process categorical features
    assert hasattr(datasets, "__getitem__")
    # Optionally, check that categorical indices are stored or used


def test_forward_runs(classifier_instance, classification_data):
    """Ensure predict_proba_tensor runs OK after standard fit."""
    X_train, X_test, y_train, y_test = classification_data
    clf = classifier_instance
    clf.fit_mode = "low_memory"
    clf.fit(X_train, y_train)
    preds = clf.forward(
        torch.tensor(X_test, dtype=torch.float32), use_inference_mode=True
    )
    # Check output shape and probability sum
    assert preds.ndim == 2, f"Expected 2D output, got {preds.shape}"
    assert preds.shape[0] == X_test.shape[0], "Mismatch in test sample count"
    assert preds.shape[1] == clf.n_classes_, "Mismatch in class count"
    probs_sum = preds.sum(dim=1)
    assert torch.allclose(
        probs_sum, torch.ones_like(probs_sum), atol=1e-5
    ), "Probabilities do not sum to 1"


def test_fit_from_preprocessed_runs(classifier_instance, classification_data) -> None:
    """Verify fit_from_preprocessed runs
    using prepared data and produces
    valid predictions.
    """
    X_train, X_test, y_train, y_test = classification_data
    clf = classifier_instance

    split_fn = partial(train_test_split, test_size=0.3, random_state=42)

    datasets_list = clf.get_preprocessed_datasets(X_train, y_train, split_fn, 100)
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

        # TODO: verify number of classes, "Mismatch in class count"
        probs_sum = preds.sum(dim=1)
        assert torch.allclose(
            probs_sum, torch.ones_like(probs_sum), atol=1e-5
        ), "Probabilities do not sum to 1"
        break  # Only need to check one batch for this test
