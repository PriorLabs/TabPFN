from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.torch_version import TorchVersion

from tabpfn.utils import (
    balance_probas_by_class_counts,
    infer_devices,
)


def test__infer_devices__auto__cuda_and_mps_not_available__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
    mocker.patch("torch.cuda").is_available.return_value = False
    mocker.patch("torch.backends.mps").is_available.return_value = False
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__auto__single_cuda_gpu_available__selects_it(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cuda:0"),)


def test__infer_devices__auto__multiple_cuda_gpus_available__selects_first(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
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


def test__infer_devices__auto__mps_available_but_torch_too_old__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", TorchVersion("2.4.0"))
    mocker.patch("torch.cuda").is_available.return_value = False
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


def test__infer_devices__device_specified_twice__raises() -> None:
    with pytest.raises(
        ValueError,
        match="The list of devices for inference cannot contain the same device more ",
    ):
        infer_devices(devices=["cpu", "cpu"])


def test__infer_devices__mps_specified_but_torch_too_old__raises(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", TorchVersion("2.4.0"))
    mocker.patch("torch.backends.mps").is_available.return_value = True
    with pytest.raises(ValueError, match="The MPS device was selected"):
        infer_devices(devices="mps")


# --- Test Data for the "test_process_text_na_dataframe" test ---
test_cases = [
    {
        # Mixed dtypes & None / pd.Na
        "df": pd.DataFrame(
            {
                "ratio": [0.4, 0.5, 0.6],
                "risk": ["High", None, "Low"],
                "height": ["Low", "Low", "Low"],
                "amount": [10.2, 20.4, 20.5],
                "type": ["guest", "member", pd.NA],
            }
        ),
        "categorical_indices": [1, 2, 4],
        "ground_truth": np.array(
            [
                [0.4, 0, 0, 10.2, 0],
                [0.5, np.nan, 0, 20.4, 1],
                [0.6, 1, 0, 20.5, np.nan],
            ]
        ),
    },
    {
        # Mixed dtypes & no missing values
        "df": pd.DataFrame(
            {
                "ratio": [0.4, 0.5, 0.6],
                "risk": ["High", "Medium", "Low"],
                "height": ["Low", "Low", "High"],
                "amount": [10.2, 20.4, np.nan],
                "type": ["guest", "member", "vip"],
            }
        ),
        "categorical_indices": [1, 2, 4],
        "ground_truth": np.array(
            [
                [0.4, 0, 1, 10.2, 0],
                [0.5, 2, 1, 20.4, 1],
                [0.6, 1, 0, np.nan, 2],
            ]
        ),
    },
    {
        # All numerical no nan
        "df": pd.DataFrame(
            {
                "ratio": [0.1, 0.2, 0.3],
                "amount": [5.0, 15.5, 25.0],
                "score": [1.0, 2.5, 3.5],
            }
        ),
        "categorical_indices": [],
        "ground_truth": np.array(
            [
                [0.1, 5.0, 1.0],
                [0.2, 15.5, 2.5],
                [0.3, 25.0, 3.5],
            ]
        ),
    },
    {
        # all categorical no nan
        "df": pd.DataFrame(
            {
                "risk": ["High", "High", "High"],
                "height": ["Low", "Low", "Low"],
                "type": ["guest", "guest", "guest"],
            }
        ),
        "categorical_indices": [0, 1, 2],
        "ground_truth": np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    },
]


def test_balance_probas_by_class_counts():
    """Test balancing probabilities by class counts."""
    probas = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
    class_counts = np.array([1, 2])

    balanced = balance_probas_by_class_counts(probas, class_counts)

    # Check that each row sums to one
    sums = balanced.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), rtol=1e-5, atol=1e-5)

    expected_balanced = torch.tensor([[1 / 3, 2 / 3], [0.75, 0.25], [2 / 3, 1 / 3]])
    assert torch.allclose(balanced, expected_balanced, rtol=1e-4, atol=1e-4)
