"""Function for evaluating a function on a Tensor in chunks.

This reduces the memory required for inference.
"""

from __future__ import annotations

from typing import Callable
from typing_extensions import Concatenate, ParamSpec

import torch

P = ParamSpec("P")


def chunked_evaluate_maybe_inplace(
    f: Callable[Concatenate[torch.Tensor, P], torch.Tensor],
    x: torch.Tensor,
    save_peak_memory_factor: int | None,
    residual: bool,  # noqa: FBT001
    batch_dims: int,
    *args: P.args,
    **kwargs: P.kwargs,
) -> torch.Tensor:
    """Split x along the batch dimension(s), and sequentially evaluate f on the chunks.

    Computes either f(x) or x+f(x), depending on `residual`.
    If `save_peak_memory_factor` is an integer, the result is written in-place to the
    input tensor `x`, and `x` is returned. Otherwise, the result is returned as a new
    tensor.

    This is useful for reducing memory usage in the case where f() creates large tensors
    internally.

    Args:
        f: Function to evaluate.
        x: Input to chunk.
        save_peak_memory_factor: If set, splits x into that many chunks. If None,
            disables chunking.
        residual: Return x + f(x). Otherwise, return f(x).
        batch_dims: Indicates how many of the leading dimensions of x should be treated
            as batch dimensions.
        *args: Passed to f.
        **kwargs: Passed to f.
    """
    if save_peak_memory_factor is None:
        result = f(x.flatten(0, batch_dims - 1), *args, **kwargs).view(x.shape)
        return x + result if residual else result

    msg = "Gradients cannot be tracked save_peak_memory_factor is not None."
    assert not x.requires_grad, msg

    x_flat_batch = x.flatten(0, batch_dims - 1)
    split_size = (
        x_flat_batch.shape[0] + save_peak_memory_factor - 1
    ) // save_peak_memory_factor
    for x_chunk in torch.split(x_flat_batch, split_size):
        if residual:
            x_chunk.add_(f(x_chunk, *args, **kwargs))
        else:
            x_chunk[:] = f(x_chunk, *args, **kwargs)

    return x
