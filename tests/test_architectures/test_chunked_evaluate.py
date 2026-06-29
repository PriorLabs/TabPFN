#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import pytest
import torch

from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace


@pytest.mark.parametrize("save_peak_memory_factor", [None, 3])
def test__residual_false__output_equal_to_normal_evaluation(
    save_peak_memory_factor: int | None,
) -> None:
    x1 = torch.randn(size=(100, 5), generator=torch.Generator().manual_seed(0))
    x2 = x1.clone()

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    output = chunked_evaluate_maybe_inplace(
        f, x1, save_peak_memory_factor, residual=False, batch_dims=1
    )
    assert torch.equal(output, f(x2))


@pytest.mark.parametrize("save_peak_memory_factor", [None, 4])
def test__residual_true__output_equal_to_normal_evaluation(
    save_peak_memory_factor: int | None,
) -> None:
    x1 = torch.randn(size=(100, 5), generator=torch.Generator().manual_seed(0))
    x2 = x1.clone()

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    output = chunked_evaluate_maybe_inplace(
        f, x1, save_peak_memory_factor, residual=True, batch_dims=1
    )
    assert torch.equal(output, x2 + f(x2))


@pytest.mark.parametrize("residual", [False, True])
def test__factor_not_none__is_inplace(residual: bool) -> None:
    x = torch.randn(size=(100, 5), generator=torch.Generator().manual_seed(0))

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    output = chunked_evaluate_maybe_inplace(
        f, x, save_peak_memory_factor=5, residual=residual, batch_dims=1
    )
    assert output.data_ptr() == x.data_ptr()


@pytest.mark.parametrize("residual", [False, True])
def test__non_contiguous_input__output_equal_to_normal_evaluation(
    residual: bool,
) -> None:
    # A non-contiguous input previously caused a silent no-op: `x.flatten(...)` returned
    # a copy rather than a view, so the in-place chunk writes never reached `x`. Mimic a
    # transposed activation tensor and check the in-place path still computes f.
    x_contig = torch.randn(
        size=(2, 7, 3, 5), generator=torch.Generator().manual_seed(0)
    )
    x = x_contig.transpose(1, 2)  # (2, 3, 7, 5), non-contiguous
    assert not x.is_contiguous()

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    expected = x + f(x) if residual else f(x)
    output = chunked_evaluate_maybe_inplace(
        f, x, save_peak_memory_factor=2, residual=residual, batch_dims=2
    )
    assert torch.equal(output, expected)


def test__chunks_have_correct_size_and_kwargs_passed() -> None:
    x = torch.randn(size=(100, 5), generator=torch.Generator().manual_seed(0))

    def f(x: torch.Tensor, my_arg: str) -> torch.Tensor:
        assert x.shape[0] == 20
        assert x.shape[1] == 5
        assert my_arg == "3"
        return x

    chunked_evaluate_maybe_inplace(
        f, x, save_peak_memory_factor=5, residual=False, batch_dims=1, my_arg="3"
    )
