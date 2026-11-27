#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the error helpers in `tabpfn.errors`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.errors import TabPFNCUDAOutOfMemoryError, handle_oom_errors


class TestHandleOomErrors:
    """The OOM message describes the data the model actually ran on.

    Both counts come off the `X` handed to the context manager, never off
    `n_features_in_`: date expansion can make the frame the transformer runs on
    wider than the one the caller passed, and memory follows the wider one.
    """

    devices = (torch.device("cpu"),)

    def _raise_oom(self, X: np.ndarray) -> str:
        with (
            pytest.raises(TabPFNCUDAOutOfMemoryError) as caught,
            handle_oom_errors(
                self.devices, X, model_type="classifier", n_train_samples=100
            ),
        ):
            raise torch.OutOfMemoryError
        return str(caught.value)

    def test__oom_on_2d_input__message_names_both_sizes(self) -> None:
        message = self._raise_oom(np.zeros((7, 13)))

        assert "100 train / 7 test samples, 13 features." in message

    def test__oom_on_1d_input__omits_the_feature_count(self) -> None:
        message = self._raise_oom(np.zeros(7))

        assert "7 test samples." in message
        assert "features" not in message

    def test__non_oom_runtime_error__propagates_unchanged(self) -> None:
        with (
            pytest.raises(RuntimeError, match="something else"),
            handle_oom_errors(self.devices, np.zeros((2, 2)), model_type="classifier"),
        ):
            raise RuntimeError("something else")
