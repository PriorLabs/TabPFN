#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for TorchSquashingScaler."""

from __future__ import annotations

import importlib

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


# The blocked reduction in `fit` and the blocked map in `transform` only engage on
# inputs far larger than anything the tests above build, so at their real budgets those
# paths are never taken here. These tests force the budgets instead: the same small
# input is run once with blocking off and once with it as fine as it goes -- a column
# per block, a row per block -- and the two must agree bit for bit.
#
# Bit for bit rather than `allclose`, because the failures worth catching are exactly
# the ones a tolerance hides: a NaN payload that changes, a zero that comes back
# negative, a quantile that lands one representable step away because a column was
# grouped differently.

# Captured once, at import, so a test that installs a counting wrapper still calls the
# real implementation rather than another test's wrapper.
_REAL_COLUMN_STATISTICS_BLOCK = TorchSquashingScaler._column_statistics_block
_REAL_TRANSFORM_BLOCK = TorchSquashingScaler._transform_block

# Budgets that force every block boundary the code can express, and budgets no input in
# this file can exceed. Both are set explicitly rather than leaning on the defaults, so
# the test keeps testing what it says even if the real constants are retuned.
_FORCE_BLOCKED = {
    "_FIT_BLOCK_BYTES": 1,
    "_FIT_UNBLOCKED_BYTES": 0,
    "_TRANSFORM_BLOCK_BYTES": 1,
}
_FORCE_UNBLOCKED = {
    "_FIT_BLOCK_BYTES": 1 << 40,
    "_FIT_UNBLOCKED_BYTES": 1 << 40,
    "_TRANSFORM_BLOCK_BYTES": 1 << 40,
}

_BIT_DTYPE = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


def _bits(t: torch.Tensor) -> torch.Tensor:
    """An integer view of a tensor's bytes.

    A float comparison calls two NaNs unequal and two zeros of opposite sign equal;
    neither is what "identical" means here.
    """
    if t.dtype == torch.bool or not t.dtype.is_floating_point:
        return t
    return t.contiguous().view(_BIT_DTYPE[t.dtype])


def _assert_bit_identical(
    what: str, blocked: torch.Tensor, plain: torch.Tensor
) -> None:
    """Assert two tensors are byte-for-byte equal, naming the first cell that is not."""
    assert blocked.shape == plain.shape, f"{what}: {blocked.shape} != {plain.shape}"
    assert blocked.dtype == plain.dtype, f"{what}: {blocked.dtype} != {plain.dtype}"
    unequal = _bits(blocked) != _bits(plain)
    if not unequal.any():
        return
    position = tuple(int(v) for v in torch.nonzero(unequal)[0])
    raise AssertionError(
        f"{what} differs at {position}: blocked {blocked[position].item()!r} vs "
        f"unblocked {plain[position].item()!r} "
        f"({int(unequal.sum())} of {unequal.numel()} entries differ)"
    )


def _run_with_budgets(
    monkeypatch: pytest.MonkeyPatch,
    budgets: dict[str, int],
    x: torch.Tensor,
    num_train_rows: int,
    max_absolute_value: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, int]]:
    """Fit and transform under the given budgets, counting the blocks processed.

    The counts are what stop this being a tautology: without them a run that quietly
    took the unblocked path both times would pass while testing nothing.
    """
    module = importlib.import_module(
        "tabpfn.preprocessing.torch.torch_squashing_scaler"
    )
    for name, value in budgets.items():
        monkeypatch.setattr(module, name, value)

    counts = {"fit_blocks": 0, "transform_blocks": 0}

    def counting_column_statistics_block(self, *args, **kwargs):  # noqa: ANN202
        counts["fit_blocks"] += 1
        return _REAL_COLUMN_STATISTICS_BLOCK(self, *args, **kwargs)

    def counting_transform_block(self, *args, **kwargs):  # noqa: ANN202
        counts["transform_blocks"] += 1
        return _REAL_TRANSFORM_BLOCK(self, *args, **kwargs)

    monkeypatch.setattr(
        module.TorchSquashingScaler,
        "_column_statistics_block",
        counting_column_statistics_block,
    )
    monkeypatch.setattr(
        module.TorchSquashingScaler, "_transform_block", counting_transform_block
    )

    scaler = module.TorchSquashingScaler(max_absolute_value=max_absolute_value)
    cache = scaler.fit(x[:num_train_rows])
    out = scaler.transform(x, fitted_cache=cache)
    return cache, out, counts


def _blocking_cases() -> list[pytest.param]:
    """Inputs covering every branch the blocked and unblocked paths share."""
    generator = torch.Generator().manual_seed(0)

    def normal(shape: tuple[int, ...], dtype: torch.dtype = torch.float32):  # noqa: ANN202
        return torch.randn(shape, generator=generator, dtype=dtype)

    cases: list[tuple[str, torch.Tensor, int]] = []

    # The pipeline's own layout, and the 2-D one `fit` documents.
    cases.append(("robust 3d", normal((96, 1, 12)), 64))
    cases.append(("robust 2d", normal((96, 12)), 64))
    # A batch dimension above 1: blocking splits the last axis, so a middle axis that
    # is not 1 is where an indexing slip would show up.
    cases.append(("batched", normal((96, 3, 8)), 64))

    # Every dtype the pipeline can force. float16 matters most: nanquantile refuses it,
    # so the implementation upcasts, and the cast happens per block.
    cases.append(("float64", normal((96, 1, 8), torch.float64), 64))
    cases.append(("float16", normal((96, 1, 8), torch.float16), 64))

    # The branch picks: a constant column (zero_mask), one whose quartiles collapse
    # while its range does not (minmax), and ordinary columns beside them.
    mixed = normal((96, 1, 8))
    mixed[:, :, 0] = 3.25
    mixed[:, :, 1] = torch.where(normal((96, 1)) < 1.0, 0.0, 9.0)
    mixed[:, :, 2] = 0.0
    cases.append(("zero and minmax columns", mixed, 64))

    # NaN and inf, including columns that are nothing else.
    nasty = normal((96, 1, 8))
    nasty[:, :, 0] = float("nan")
    nasty[:, :, 1] = float("inf")
    nasty[:, :, 2] = float("-inf")
    nasty[::7, :, 3] = float("nan")
    nasty[::11, :, 4] = float("inf")
    nasty[::13, :, 5] = float("-inf")
    cases.append(("nan and inf", nasty, 64))

    # +/-inf inside a zero column and inside a minmax one: the zero-column branch and
    # the +/-inf branch both want to write to those cells.
    overlap = normal((96, 1, 6))
    overlap[:, :, 0] = 2.5
    overlap[::9, :, 0] = float("inf")
    overlap[::10, :, 0] = float("-inf")
    overlap[:, :, 1] = torch.where(normal((96, 1)) < 1.0, 0.0, 4.0)
    overlap[::8, :, 1] = float("-inf")
    cases.append(("inf inside zero and minmax columns", overlap, 64))

    # Signed zero, which a value comparison would call equal to zero.
    zeros = normal((96, 1, 4))
    zeros[:, :, 0] = -0.0
    zeros[:, :, 1] = 0.0
    cases.append(("signed zero", zeros, 64))

    # Fitting on rows that are entirely NaN while the transformed rows are not.
    split = normal((96, 1, 6))
    split[:64] = float("nan")
    cases.append(("all-nan training rows", split, 64))

    # The early return in `fit`, which no amount of blocking should reach.
    cases.append(("single train row", normal((96, 1, 6)), 1))

    # Degenerate shapes: one column to block, and many rows to map.
    cases.append(("one column", normal((96, 1, 1)), 64))
    cases.append(("tall and thin", normal((512, 1, 2)), 341))

    return [pytest.param(x, rows, id=name) for name, x, rows in cases]


class TestBlockingEquivalence:
    """Blocking must not change what the scaler returns."""

    @pytest.mark.parametrize(("x", "num_train_rows"), _blocking_cases())
    @pytest.mark.parametrize("max_absolute_value", [3.0, 10.0])
    def test__blocked_and_unblocked_are_bit_identical(
        self,
        monkeypatch: pytest.MonkeyPatch,
        x: torch.Tensor,
        num_train_rows: int,
        max_absolute_value: float,
    ) -> None:
        """The fitted cache and the transformed output agree byte for byte."""
        with monkeypatch.context() as unblocked_patch:
            plain_cache, plain_out, plain_counts = _run_with_budgets(
                unblocked_patch, _FORCE_UNBLOCKED, x, num_train_rows, max_absolute_value
            )
        with monkeypatch.context() as blocked_patch:
            blocked_cache, blocked_out, blocked_counts = _run_with_budgets(
                blocked_patch, _FORCE_BLOCKED, x, num_train_rows, max_absolute_value
            )

        # Without this the test could pass while both runs took the same path.
        assert plain_counts["transform_blocks"] == 1
        assert blocked_counts["transform_blocks"] == x.shape[0]
        if num_train_rows > 1:
            assert plain_counts["fit_blocks"] == 1
            assert blocked_counts["fit_blocks"] == x.shape[-1]
        else:
            # `fit` returns before reducing anything, so there are no blocks either way.
            assert plain_counts["fit_blocks"] == 0
            assert blocked_counts["fit_blocks"] == 0

        assert set(blocked_cache) == set(plain_cache)
        for key in sorted(plain_cache):
            _assert_bit_identical(
                f"fitted cache[{key!r}]", blocked_cache[key], plain_cache[key]
            )
        _assert_bit_identical("transform output", blocked_out, plain_out)

    def test__blocked_transform_does_not_write_to_its_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`transform` leaves `x` alone; the pipeline reuses the tensor it passes in."""
        x = torch.randn(96, 1, 8, generator=torch.Generator().manual_seed(1))
        x[::5, :, 0] = float("inf")
        x[::7, :, 1] = float("nan")
        before = x.clone()

        with monkeypatch.context() as patch:
            _run_with_budgets(patch, _FORCE_BLOCKED, x, 64, 3.0)

        _assert_bit_identical("input tensor", x, before)
