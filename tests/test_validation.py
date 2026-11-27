#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the error types raised by dataset shape validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.errors import TabPFNUserError, TabPFNValidationError
from tabpfn.validation import validate_dataset_size


def _cpu_devices() -> tuple[torch.device, ...]:
    return (torch.device("cpu"),)


def test__length_mismatch__raises_validation_error() -> None:
    X = np.zeros((10, 3))
    y = np.zeros(9)
    with pytest.raises(TabPFNValidationError, match="do not match"):
        validate_dataset_size(
            X,
            y,
            max_num_samples=100,
            max_num_features=10,
            devices=_cpu_devices(),
        )


def test__non_2d_input__raises_validation_error() -> None:
    X = np.zeros(10)
    y = np.zeros(10)
    with pytest.raises(TabPFNValidationError, match="not a 2D array"):
        validate_dataset_size(
            X,
            y,
            max_num_samples=100,
            max_num_features=10,
            devices=_cpu_devices(),
        )


def test__shape_errors__stay_value_errors_and_become_user_errors() -> None:
    """The fix keeps ``ValueError`` compatibility while adding user mapping."""
    cases = [(np.zeros((10, 3)), np.zeros(9)), (np.zeros(10), np.zeros(10))]
    for X, y in cases:
        with pytest.raises(TabPFNValidationError) as caught:
            validate_dataset_size(
                X,
                y,
                max_num_samples=100,
                max_num_features=10,
                devices=_cpu_devices(),
            )
        assert isinstance(caught.value, ValueError)
        assert isinstance(caught.value, TabPFNUserError)
