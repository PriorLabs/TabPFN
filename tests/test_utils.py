# use get_total_memory and compare it against result from psutils
# run it only if the it is windows os.name == "nt"
from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from tabpfn.utils import (
    fix_dtypes,
    get_ordinal_encoder,
    infer_categorical_features,
    infer_devices,
    process_text_na_dataframe,
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


def test__process_text_na_dataframe__only_numeric_columns__converts_to_float64():
    """Test process_text_na_dataframe with only numeric columns."""
    df = pd.DataFrame(
        {
            "numeric1": [1.0, 2.0, 3.0],
            "numeric2": [4.0, 5.0, 6.0],
            "numeric3": [7.0, 8.0, 9.0],
        }
    )

    # No categorical indices
    df = fix_dtypes(df, cat_indices=None)

    result = process_text_na_dataframe(df, ord_encoder=None, fit_encoder=False)

    # Check shape is preserved
    assert result.shape == (3, 3)
    assert result.dtype == np.float64

    # Check numeric columns are preserved
    np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result[:, 1], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(result[:, 2], [7.0, 8.0, 9.0])


def test__process_text_na_dataframe__string_columns__replaces_placeholder_with_nan():
    """Test process_text_na_dataframe with encoder and string column names."""
    df = pd.DataFrame(
        {
            "numeric1": [1.0, 2.0, 3.0],
            "string1": ["a", "b", pd.NA],
            "numeric2": [4.0, 5.0, 6.0],
            "string2": ["x", pd.NA, "z"],
        }
    )

    # fix_dtypes to set proper dtypes (categoricals as needed)
    df = fix_dtypes(df, cat_indices=["string1", "string2"])

    encoder = get_ordinal_encoder()
    result = process_text_na_dataframe(df, ord_encoder=encoder, fit_encoder=True)

    # Check shape is preserved
    assert result.shape == (3, 4)
    assert result.dtype == np.float64

    # After ColumnTransformer, string columns come first, then numeric columns
    # Order should be: string1, string2, numeric1, numeric2

    # Check that NA placeholders in string columns are converted to NaN
    # string1 had pd.NA at row 2
    assert np.isnan(result[2, 0])
    # string2 had pd.NA at row 1
    assert np.isnan(result[1, 1])

    # Check numeric columns are preserved (now at positions 2 and 3)
    np.testing.assert_array_equal(result[:, 2], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result[:, 3], [4.0, 5.0, 6.0])

    # Check that non-NA string values are encoded (not NaN)
    assert not np.isnan(result[0, 0])  # 'a' in string1
    assert not np.isnan(result[1, 0])  # 'b' in string1
    assert not np.isnan(result[0, 1])  # 'x' in string2
    assert not np.isnan(result[2, 1])  # 'z' in string2


def test__process_text_na_dataframe__integer_columns__replaces_placeholder_with_nan():
    """Test process_text_na_dataframe with encoder and integer column names.

    This is the critical test case that catches the bug where string_cols_ix
    don't match the encoded column positions when column names are integers.
    """
    # DataFrame with integer column names (like after certain transformations)
    df = pd.DataFrame(
        {
            0: [0.4, 0.5, 0.6],  # numeric
            1: ["High", "Medium", "Low"],  # categorical
            2: ["High", "Low", "Medium"],  # categorical
            3: [10.2, 20.4, 20.5],  # numeric
            4: ["guest", "guest", pd.NA],  # categorical with NA
        }
    )

    # fix_dtypes with integer column indices
    df = fix_dtypes(df, cat_indices=[1, 2, 4])

    encoder = get_ordinal_encoder()
    result = process_text_na_dataframe(df, ord_encoder=encoder, fit_encoder=True)

    # Check shape is preserved
    assert result.shape == (3, 5)
    assert result.dtype == np.float64

    # After ColumnTransformer with integer column names:
    # sklearn converts them to 'x0', 'x1', etc.
    # String columns (1, 2, 4) come first, then numeric columns (0, 3)
    # Order should be: x1, x2, x4, x0, x3

    # Check that NA in column 4 (now at position 2) is converted to NaN
    assert np.isnan(result[2, 2])  # row 2, column that was originally 4

    # Check that non-NA string values are encoded (not NaN)
    assert not np.isnan(result[0, 0])  # 'High' in original column 1
    assert not np.isnan(result[1, 0])  # 'Medium' in original column 1
    assert not np.isnan(result[2, 0])  # 'Low' in original column 1

    assert not np.isnan(result[0, 1])  # 'High' in original column 2
    assert not np.isnan(result[1, 1])  # 'Low' in original column 2
    assert not np.isnan(result[2, 1])  # 'Medium' in original column 2

    assert not np.isnan(result[0, 2])  # 'guest' in original column 4
    assert not np.isnan(result[1, 2])  # 'guest' in original column 4

    # Check numeric columns are preserved (now at the end)
    np.testing.assert_array_equal(result[:, 3], [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(result[:, 4], [10.2, 20.4, 20.5])


def test__process_text_na_dataframe__all_numeric__passes_through_unchanged():
    """Test process_text_na_dataframe with only numeric columns."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )

    # No categorical indices
    df = fix_dtypes(df, cat_indices=None)

    encoder = get_ordinal_encoder()
    result = process_text_na_dataframe(df, ord_encoder=encoder, fit_encoder=True)

    # Should pass through unchanged
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result[:, 1], [4.0, 5.0, 6.0])


def test__process_text_na_dataframe__all_string__replaces_placeholder_with_nan():
    """Test process_text_na_dataframe with only string columns."""
    df = pd.DataFrame(
        {
            "a": ["x", "y", pd.NA],
            "b": [pd.NA, "p", "q"],
        }
    )

    # Set both as categorical
    df = fix_dtypes(df, cat_indices=["a", "b"])

    encoder = get_ordinal_encoder()
    result = process_text_na_dataframe(df, ord_encoder=encoder, fit_encoder=True)

    # Check shape is preserved
    assert result.shape == (3, 2)

    # Check that NA placeholders are converted to NaN
    assert np.isnan(result[2, 0])  # column 'a', row 2
    assert np.isnan(result[0, 1])  # column 'b', row 0

    # Check that non-NA values are encoded (not NaN)
    assert not np.isnan(result[0, 0])  # 'x'
    assert not np.isnan(result[1, 0])  # 'y'
    assert not np.isnan(result[1, 1])  # 'p'
    assert not np.isnan(result[2, 1])  # 'q'


def test__process_text_na_dataframe__transform_after_fit__works_correctly():
    """Test that transform works correctly after fit."""
    df_train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "string": ["a", "b", pd.NA],
        }
    )

    df_train = fix_dtypes(df_train, cat_indices=["string"])

    encoder = get_ordinal_encoder()
    # Fit
    _ = process_text_na_dataframe(df_train, ord_encoder=encoder, fit_encoder=True)

    # Transform new data
    df_test = pd.DataFrame(
        {
            "numeric": [4.0, 5.0],
            "string": ["a", pd.NA],
        }
    )

    df_test = fix_dtypes(df_test, cat_indices=["string"])

    result = process_text_na_dataframe(df_test, ord_encoder=encoder, fit_encoder=False)

    assert result.shape == (2, 2)
    # Check that NA is converted to NaN
    assert np.isnan(result[1, 0])
    # Check numeric column
    np.testing.assert_array_equal(result[:, 1], [4.0, 5.0])
