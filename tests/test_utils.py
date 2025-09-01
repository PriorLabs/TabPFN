# use get_total_memory and compare it against result from psutils
# run it only if the it is windows os.name == "nt"
from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tabpfn.utils import infer_categorical_features, infer_devices


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


def test__infer_devices__auto__multiple_cuda_gpus_available__selects_all(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 3
    mocker.patch("torch.backends.mps").is_available.return_value = True

    inferred = set(infer_devices(devices="auto"))
    expected = {torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")}
    assert inferred == expected


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
    mock_cuda.device_count.return_value = 3
    mocker.patch("torch.backends.mps").is_available.return_value = True

    inferred = set(infer_devices(devices="auto"))
    expected = {torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")}
    assert inferred == expected


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
