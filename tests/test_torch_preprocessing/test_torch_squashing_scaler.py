#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for TorchSquashingScaler."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.preprocessing.steps.squashing_scaler_transformer import SquashingScaler
from tabpfn.preprocessing.torch import TorchSquashingScaler


class TestTorchSquashingScaler:
    """Tests for TorchSquashingScaler class."""

    def test__call__shape_preservation(self) -> None:
        """Output shape matches input shape, 2D and 3D."""
        scaler = TorchSquashingScaler()
        for shape in [(50, 4), (100, 3, 7), (200, 2, 1)]:
            x = torch.randn(*shape)
            out = scaler(x)
            assert out.shape == x.shape

    def test__call__nan_preserved(self) -> None:
        """NaNs in the input remain NaN at the same positions in the output."""
        scaler = TorchSquashingScaler()
        x = torch.tensor(
            [
                [1.0, float("nan"), 3.0],
                [2.0, 4.0, float("nan")],
                [3.0, 5.0, 5.0],
                [4.0, 6.0, 7.0],
            ]
        )
        out = scaler(x)

        assert torch.isnan(out[0, 1])
        assert torch.isnan(out[1, 2])
        assert not torch.isnan(out[~torch.isnan(x)]).any()

    def test__call__inf_clamps_to_max_absolute_value(self) -> None:
        """+inf maps to +B and -inf maps to -B."""
        b = 3.0
        scaler = TorchSquashingScaler(max_absolute_value=b)
        x = torch.tensor(
            [
                [float("inf"), 1.0],
                [float("-inf"), 2.0],
                [3.0, 3.0],
                [-1.0, 4.0],
                [float("nan"), 5.0],
                [2.0, 6.0],
            ]
        )
        out = scaler(x)

        assert out[0, 0] == b
        assert out[1, 0] == -b
        # Sanity: column 1 stays in (-B, B)
        finite_col1 = out[:, 1]
        assert (finite_col1.abs() <= b).all()

    def test__call__constant_column_yields_zero_for_finite(self) -> None:
        """Columns with max == min produce 0 for finite values, NaN preserved."""
        scaler = TorchSquashingScaler()
        x = torch.tensor(
            [
                [1.0, 5.0],
                [2.0, 5.0],
                [3.0, 5.0],
                [4.0, float("nan")],
            ]
        )
        out = scaler(x)

        assert torch.allclose(out[:3, 1], torch.zeros(3))
        assert torch.isnan(out[3, 1])

    def test__call__minmax_path_matches_docstring(self) -> None:
        """q25 == q75 column should match the SquashingScaler docstring values."""
        scaler = TorchSquashingScaler()
        x = torch.tensor([[0.0], [1.0], [1.0], [1.0], [2.0], [float("nan")]])
        out = scaler(x).squeeze(-1)
        expected = torch.tensor([-0.9486833, 0.0, 0.0, 0.0, 0.9486833, float("nan")])
        # NaN positions
        assert torch.isnan(out[5])
        # Finite positions
        assert torch.allclose(out[:5], expected[:5], atol=1e-6)

    def test__call__robust_path_matches_docstring(self) -> None:
        """General-case docstring example reproduces exactly."""
        scaler = TorchSquashingScaler(max_absolute_value=3.0)
        x = torch.tensor(
            [[float("inf")], [float("-inf")], [3.0], [-1.0], [float("nan")], [2.0]],
            dtype=torch.float64,
        )
        out = scaler(x).squeeze(-1)
        expected = torch.tensor(
            [3.0, -3.0, 0.49319696, -1.34164079, float("nan"), 0.0],
            dtype=torch.float64,
        )
        assert out[0] == 3.0
        assert out[1] == -3.0
        assert torch.isnan(out[4])
        finite_idx = torch.tensor([2, 3, 5])
        assert torch.allclose(out[finite_idx], expected[finite_idx], atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test__call__dtype_preserved(self, dtype: torch.dtype) -> None:
        """Output keeps the input dtype."""
        scaler = TorchSquashingScaler()
        x = torch.randn(60, 5, dtype=dtype)
        out = scaler(x)
        assert out.dtype == dtype

    def test__call__device_preserved(self) -> None:
        """Output stays on the input device (CPU smoke test)."""
        scaler = TorchSquashingScaler()
        x = torch.randn(60, 5, device="cpu")
        out = scaler(x)
        assert out.device == x.device

    def test__call__num_train_rows_clips_test_outlier(self) -> None:
        """Stats fit on train rows; test rows pass through the same scaling."""
        scaler = TorchSquashingScaler(max_absolute_value=3.0)
        # Train rows tightly around 0; test row is a huge outlier.
        train = torch.linspace(-1.0, 1.0, 100).unsqueeze(-1)
        test = torch.tensor([[1e6]])
        x = torch.cat([train, test], dim=0)

        out = scaler(x, num_train_rows=100)
        # Outlier should be soft-clipped near +max_absolute_value.
        assert out[-1, 0] > 2.5
        assert out[-1, 0] <= 3.0

    def test__call__batch_dim_independent_per_batch(self) -> None:
        """Different per-batch distributions produce different transforms."""
        torch.manual_seed(0)
        scaler = TorchSquashingScaler()
        # Two batches with different scales.
        b0 = torch.randn(200, 3)
        b1 = torch.randn(200, 3) * 100
        x = torch.stack([b0, b1], dim=1)  # [T=200, batch=2, n_cols=3]
        out = scaler(x)
        # Both batches should be in [-B, B] (apart from NaNs we didn't add)
        assert (out.abs() <= 3.0 + 1e-6).all()
        # The scalings differ => the outputs differ even though the *raw*
        # b1 values are 100x b0 — after scaling they should be similar in
        # magnitude, but bitwise different.
        assert not torch.allclose(out[:, 0, :], out[:, 1, :])

    def test__transform__missing_keys_raises(self) -> None:
        """Missing keys in the cache surface a clear error."""
        scaler = TorchSquashingScaler()
        x = torch.randn(10, 3)
        with pytest.raises(ValueError, match="Invalid fitted cache"):
            scaler.transform(x, fitted_cache={"center": torch.zeros(3)})

    def test__init__rejects_invalid_args(self) -> None:
        with pytest.raises(ValueError, match="quantile_range"):
            TorchSquashingScaler(quantile_range=(0.5, 0.5))
        with pytest.raises(ValueError, match="max_absolute_value"):
            TorchSquashingScaler(max_absolute_value=0.0)

    def test__call__matches_sklearn_squashing_scaler_float64(self) -> None:
        """Numerical equivalence with the CPU SquashingScaler in float64."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(size=(200, 50))
        # Sprinkle in NaNs, infs, and a constant column to exercise all branches.
        x_np[0, 5] = np.inf
        x_np[1, 5] = -np.inf
        x_np[2, 5] = np.nan
        x_np[:, 10] = 7.0
        # q25 == q75 column (clustered values with extremes only).
        x_np[:, 20] = 1.0
        x_np[0, 20] = 0.0
        x_np[-1, 20] = 2.0

        cpu_out = SquashingScaler().fit_transform(x_np.copy())

        torch_scaler = TorchSquashingScaler()
        x_t = torch.from_numpy(x_np)
        torch_out = torch_scaler(x_t).numpy()

        assert np.allclose(cpu_out, torch_out, atol=1e-12, equal_nan=True), (
            f"max abs diff = {np.nanmax(np.abs(cpu_out - torch_out))}"
        )

    def test__call__matches_sklearn_squashing_scaler_float32(self) -> None:
        """Float32 matches the CPU SquashingScaler within float32 tolerance."""
        rng = np.random.default_rng(7)
        x_np = rng.standard_normal(size=(150, 20)).astype(np.float64)
        x_np[5, 0] = np.nan
        x_np[:, 4] = 3.0  # constant
        cpu_out = SquashingScaler().fit_transform(x_np.copy())

        torch_scaler = TorchSquashingScaler()
        x_t = torch.from_numpy(x_np.astype(np.float32))
        torch_out = torch_scaler(x_t).numpy().astype(np.float64)

        assert np.allclose(cpu_out, torch_out, atol=1e-5, equal_nan=True), (
            f"max abs diff = {np.nanmax(np.abs(cpu_out - torch_out))}"
        )
