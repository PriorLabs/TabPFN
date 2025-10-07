# use get_total_memory and compare it against result from psutils
# run it only if the it is windows os.name == "nt"
from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier
from tabpfn.config import ModelInterfaceConfig
from tabpfn.constants import NA_PLACEHOLDER
from tabpfn.utils import (
    fix_dtypes,
    get_ordinal_encoder,
    infer_categorical_features,
    infer_devices,
    process_text_na_dataframe,
    validate_Xy_fit,
)


def test_internal_windows_total_memory():
    if os.name != "nt":
        pytest.skip("Windows specific test")
    import psutil

    from tabpfn.utils import get_total_memory_windows

    utils_result = get_total_memory_windows()
    psutil_result = psutil.virtual_memory().total / 1e9
    assert utils_result == psutil_result


def test_internal_windows_total_memory_multithreaded():
    # collect results from multiple threads
    if os.name != "nt":
        pytest.skip("Windows specific test")
    import threading

    import psutil

    from tabpfn.utils import get_total_memory_windows

    results = []

    def get_memory() -> None:
        results.append(get_total_memory_windows())

    threads = []
    for _ in range(10):
        t = threading.Thread(target=get_memory)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    psutil_result = psutil.virtual_memory().total / 1e9
    assert all(result == psutil_result for result in results)


def test_infer_categorical_with_str_and_nan_provided_included():
    X = np.array([[np.nan, "NA"]], dtype=object).reshape(-1, 1)
    out = infer_categorical_features(
        X,
        provided=[0],
        min_samples_for_inference=0,
        max_unique_for_category=2,
        min_unique_for_numerical=5,
    )
    assert out == [0]


def test_infer_categorical_with_str_and_nan_multiple_rows_provided_included():
    X = np.array([[np.nan], ["NA"], ["NA"]], dtype=object)
    out = infer_categorical_features(
        X,
        provided=[0],
        min_samples_for_inference=0,
        max_unique_for_category=2,
        min_unique_for_numerical=5,
    )
    assert out == [0]


def test_infer_categorical_auto_inference_blocked_when_not_enough_samples():
    X = np.array([[1.0], [1.0], [np.nan]])
    out = infer_categorical_features(
        X,
        provided=None,
        min_samples_for_inference=3,
        max_unique_for_category=2,
        min_unique_for_numerical=4,
    )
    assert out == []


def test_infer_categorical_auto_inference_enabled_with_enough_samples():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0], [np.nan, 9.0]])
    out = infer_categorical_features(
        X,
        provided=None,
        min_samples_for_inference=3,
        max_unique_for_category=3,
        min_unique_for_numerical=4,
    )
    assert out == [0]


def test_infer_categorical_provided_column_excluded_if_exceeds_max_unique():
    X = np.array([[0], [1], [2], [3], [np.nan]], dtype=float)
    out = infer_categorical_features(
        X,
        provided=[0],
        min_samples_for_inference=0,
        max_unique_for_category=3,
        min_unique_for_numerical=2,
    )
    assert out == []


def test_infer_categorical_with_dict_raises_error():
    X = np.array([[{"a": 1}], [{"b": 2}]], dtype=object)
    with pytest.raises(TypeError):
        infer_categorical_features(
            X,
            provided=None,
            min_samples_for_inference=0,
            max_unique_for_category=2,
            min_unique_for_numerical=2,
        )


def test__infer_devices__auto__cuda_and_mps_not_available__selects_cpu(
    mocker: MagicMock,
) -> None:
    mocker.patch("torch.cuda").is_available.return_value = False
    mocker.patch("torch.backends.mps").is_available.return_value = False
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__auto__single_cuda_gpu_available__selects_it(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cuda:0"),)


def test__infer_devices__auto__multiple_cuda_gpus_available__selects_first(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 3
    mocker.patch("torch.backends.mps").is_available.return_value = True

    assert infer_devices(devices="auto") == (torch.device("cuda:0"),)


def test__infer_devices__auto__cuda_and_mps_available_but_excluded__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "mps,cuda")
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__device_specified__selects_it(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2
    mocker.patch("torch.backends.mps").is_available.return_value = True

    assert infer_devices(devices="cuda:0") == (torch.device("cuda:0"),)


def test__infer_devices__multiple_devices_specified___selects_them(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 3
    mocker.patch("torch.backends.mps").is_available.return_value = False

    inferred = set(infer_devices(devices=["cuda:0", "cuda:1", "cuda:4"]))
    expected = {torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:4")}
    assert inferred == expected


def test__infer_devices__device_selected_twice__raises() -> None:
    with pytest.raises(
        ValueError,
        match="The list of devices for inference cannot contain the same device more ",
    ):
        infer_devices(devices=["cpu", "cpu"])


# --- Test Data for the "test_process_text_na_dataframe" test ---
test_cases = [
    pd.DataFrame(
        {
            "ratio": [0.4, 0.5, 0.6],
            "risk": ["High", "Medium", "Low"],
            "height": ["Low", "Low", "Low"],
            "amount": [10.2, 20.4, 20.5],
            "type": ["guest", "member", "__MISSING__"],
        }
    ),
    pd.DataFrame(
        {
            "risk": ["High", "Medium", "Low"],
            "height": ["Low", "Medium", "High"],
            "type": ["guest", "member", "__MISSING__"],
        }
    ),
    pd.DataFrame(
        {
            "ratio": [0.4, 0.5, 0.6],
            "risk": ["High", "Medium", "Low"],
            "height": ["Low", "Medium", "High"],
            "amount": [10.2, 20.4, 20.5],
            "type": ["guest", "member", "vip"],
        }
    ),
    pd.DataFrame(
        {
            "ratio": [0.1, 0.2, 0.3],
            "risk": ["High", None, "Low"],
            "height": ["Low", "Medium", None],
            "amount": [5.0, 15.5, 25.0],
            "type": ["guest", None, "member"],
        }
    ),
    pd.DataFrame(
        {
            "ratio": [0.7, 0.8, 0.9],
            "risk": ["High", "High", "High"],
            "height": ["Low", "Low", "Low"],
            "amount": [30, 40, 50],
            "type": ["guest", "guest", "guest"],
        }
    ),
    pd.DataFrame(
        {"ratio": [0.1, 0.2, 0.3], "amount": [10, 20, 30], "score": [5.0, 6.5, 7.2]}
    ),
]


# --- Fixture for the "test_process_text_na_dataframe" test ---
# prepare the DataFrame
@pytest.fixture(params=test_cases)
def prepared_tabpfn_data(request):
    temp_df = request.param.copy()
    y = np.array([0, 1, 0])  # Dummy target

    cls = TabPFNClassifier()
    # Validate X and y
    X, y, feature_names_in, n_features_in = validate_Xy_fit(
        temp_df,
        y,
        estimator=cls,
        ensure_y_numeric=False,
        max_num_samples=ModelInterfaceConfig.MAX_NUMBER_OF_SAMPLES,
        max_num_features=ModelInterfaceConfig.MAX_NUMBER_OF_FEATURES,
        ignore_pretraining_limits=False,
    )

    if feature_names_in is not None:
        cls.feature_names_in_ = feature_names_in
    cls.n_features_in_ = n_features_in

    # Encode classes
    if not cls.differentiable_input:
        _, counts = np.unique(y, return_counts=True)
        cls.class_counts_ = counts
        cls.label_encoder_ = LabelEncoder()
        y = cls.label_encoder_.fit_transform(y)
        cls.classes_ = cls.label_encoder_.classes_
        cls.n_classes_ = len(cls.classes_)
    else:
        cls.label_encoder_ = None
        if not hasattr(cls, "n_classes_"):
            cls.n_classes_ = int(torch.max(torch.tensor(y)).item()) + 1
        cls.classes_ = torch.arange(cls.n_classes_)

    # Infer categorical features
    cls.inferred_categorical_indices_ = infer_categorical_features(
        X=X,
        provided=getattr(cls, "categorical_features_indices", None),
        min_samples_for_inference=ModelInterfaceConfig.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
        max_unique_for_category=ModelInterfaceConfig.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
        min_unique_for_numerical=ModelInterfaceConfig.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
    )

    # Fix dtypes
    return fix_dtypes(X, cat_indices=cls.inferred_categorical_indices_)


# --- Actual test ---
def test_process_text_na_dataframe(prepared_tabpfn_data):
    X = prepared_tabpfn_data  # use the fixture

    ord_encoder = get_ordinal_encoder()
    X_out = process_text_na_dataframe(
        X,
        placeholder=NA_PLACEHOLDER,
        ord_encoder=ord_encoder,
        fit_encoder=True,
    )

    # Output should have same shape
    assert X_out.shape[0] == X.shape[0]
    assert X_out.shape[1] == X.shape[1]

    # Ensure no extra features
    assert X_out.shape[1] == len(X.columns)

    # Column order: verify string/object columns
    string_cols = X.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        col_idx = X.columns.get_loc(col)
        mask = X[col] == NA_PLACEHOLDER
        np.testing.assert_array_equal(pd.isna(X_out[:, col_idx]), mask)
