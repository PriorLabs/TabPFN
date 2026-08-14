#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for tabpfn.architectures.shared.scaled_dot_product_attention."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.nn.attention import SDPBackend

from tabpfn.architectures.shared.scaled_dot_product_attention import (
    _FLASH_BACKWARD_MAX_ELEMENTS,
    _FLASH_QUERY_BLOCK,
    _torch_sdpa,
)

# batch * heads * round_up(seq_q, 128) * head_dim = 2,149,580,800, just over the
# 2**31 element range of FlashAttention's backward index. Without chunking this
# faults in backward with an illegal memory access.
_OVER_RANGE_SHAPE = {"batch": 400, "seq_q": 10_496, "heads": 4, "head_dim": 128}
_SEQ_KV = 128


def _padded_elements(batch: int, heads: int, seq_q: int, head_dim: int) -> int:
    block = _FLASH_QUERY_BLOCK
    return batch * heads * (((seq_q + block - 1) // block) * block) * head_dim


def _record_sdpa_calls(monkeypatch: pytest.MonkeyPatch) -> list[torch.Size]:
    """Replace SDPA with a shape-preserving stub and record the query shapes."""
    shapes: list[torch.Size] = []

    def fake_sdpa(
        query: torch.Tensor,
        _key: torch.Tensor,
        value: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        shapes.append(query.shape)
        return torch.empty(
            (*query.shape[:-1], value.shape[-1]),
            dtype=query.dtype,
            device=query.device,
        )

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    return shapes


def _run_on_meta(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch: int,
    seq_q: int,
    heads: int,
    head_dim: int,
    backends: list[SDPBackend] | None,
) -> list[torch.Size]:
    """Drive `_torch_sdpa` on meta tensors, which carry shape but no storage."""
    shapes = _record_sdpa_calls(monkeypatch)
    kwargs = {"device": "meta", "dtype": torch.bfloat16}
    q = torch.empty((batch, seq_q, heads, head_dim), **kwargs)
    k = torch.empty((batch, _SEQ_KV, heads, head_dim), **kwargs)
    v = torch.empty((batch, _SEQ_KV, heads, head_dim), **kwargs)
    _torch_sdpa(q, k, v, backends)
    return shapes


def test__torch_sdpa__query_over_flash_index_range__chunks_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each FlashAttention call must stay inside the backward's 32-bit range."""
    shapes = _run_on_meta(
        monkeypatch, **_OVER_RANGE_SHAPE, backends=[SDPBackend.FLASH_ATTENTION]
    )

    assert len(shapes) > 1, "expected the batch to be chunked"
    for shape in shapes:
        sub_batch, heads, seq_q, head_dim = shape
        assert (
            _padded_elements(sub_batch, heads, seq_q, head_dim)
            <= _FLASH_BACKWARD_MAX_ELEMENTS
        )
    assert sum(shape[0] for shape in shapes) == _OVER_RANGE_SHAPE["batch"]


def test__torch_sdpa__query_within_flash_index_range__single_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shape inside the range must not be chunked, so nothing slows down."""
    # 8192 needs no padding, giving exactly 2**31 elements, which is in range.
    shapes = _run_on_meta(
        monkeypatch,
        batch=512,
        seq_q=8_192,
        heads=4,
        head_dim=128,
        backends=[SDPBackend.FLASH_ATTENTION],
    )

    assert len(shapes) == 1


def test__torch_sdpa__flash_not_selectable__does_not_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limit is FlashAttention's, so other backends should not be chunked."""
    shapes = _run_on_meta(
        monkeypatch, **_OVER_RANGE_SHAPE, backends=[SDPBackend.EFFICIENT_ATTENTION]
    )

    assert len(shapes) == 1


def _cuda_bytes_available() -> int:
    if not torch.cuda.is_available():
        return 0
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes


# The query alone is 4 GiB; the backward additionally holds its gradient and an
# fp32 dQ accumulator, so leave room for roughly 24 GiB.
_NEEDED_BYTES = 24 * 1024**3


@pytest.mark.slow
@pytest.mark.skipif(
    _cuda_bytes_available() < _NEEDED_BYTES,
    reason="needs a CUDA device with ~24 GiB free",
)
def test__torch_sdpa__query_over_flash_index_range__backward_runs_on_cuda() -> None:
    """The shape that faults in an unchunked FlashAttention backward now runs."""
    shape = _OVER_RANGE_SHAPE
    assert (
        _padded_elements(
            shape["batch"], shape["heads"], shape["seq_q"], shape["head_dim"]
        )
        > _FLASH_BACKWARD_MAX_ELEMENTS
    ), "shape no longer exceeds the range this test is about"

    kwargs = {"device": "cuda", "dtype": torch.bfloat16, "requires_grad": True}
    q = torch.randn(
        (shape["batch"], shape["seq_q"], shape["heads"], shape["head_dim"]), **kwargs
    )
    k = torch.randn(
        (shape["batch"], _SEQ_KV, shape["heads"], shape["head_dim"]), **kwargs
    )
    v = torch.randn(
        (shape["batch"], _SEQ_KV, shape["heads"], shape["head_dim"]), **kwargs
    )

    out = _torch_sdpa(q, k, v, [SDPBackend.FLASH_ATTENTION])
    out.sum().backward()
    torch.cuda.synchronize()

    assert q.grad is not None
