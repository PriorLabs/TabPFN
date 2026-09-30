#  Copyright (c) Prior Labs GmbH 2026.
"""Opt-in, process-local v3.5 aggregation compilation experiment.

Install once in a dedicated inference process before creating the estimator.
This patches architecture classes for every v3.5 model in that process; it is not
a public estimator option or a validated training recipe. ICL stays eager.
"""

import torch

from tabpfn.architectures import tabpfn_v3_5 as architecture
from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace

_installed = False


def install() -> None:
    """Compile tensor regions, retaining Python chunking and memory management."""
    global _installed  # noqa: PLW0603
    if _installed:
        return

    def attention_delta(
        block: architecture.TransformerBlock,
        x: torch.Tensor,
        rope: architecture.RotaryEmbedding,
    ) -> torch.Tensor:
        return block.attention(block.layernorm(x), rope=rope)

    def mlp_delta(
        block: architecture.TransformerBlock, x: torch.Tensor
    ) -> torch.Tensor:
        return block.mlp(block.layernorm_mlp(x))

    attention = torch.compile(attention_delta, dynamic=True, fullgraph=True)
    mlp = torch.compile(mlp_delta, dynamic=True, fullgraph=True)

    def forward(
        self: architecture.TransformerBlock,
        x_BRCE: torch.Tensor,
        rope: architecture.RotaryEmbedding,
        save_peak_memory_factor: int | None = None,
    ) -> torch.Tensor:
        x_BRCE = chunked_evaluate_maybe_inplace(
            lambda x, rope: attention(self, x, rope),
            x_BRCE,
            save_peak_memory_factor=save_peak_memory_factor,
            residual=True,
            batch_dims=2,
            rope=rope,
        )
        return chunked_evaluate_maybe_inplace(
            lambda x: mlp(self, x),
            x_BRCE,
            save_peak_memory_factor=save_peak_memory_factor,
            residual=True,
            batch_dims=3,
        )

    architecture.CrossAttentionBlock.forward = torch.compile(
        architecture.CrossAttentionBlock.forward, dynamic=True, fullgraph=True
    )
    architecture.TransformerBlock.forward = forward
    architecture.TransformerBlock.forward_cross = torch.compile(
        architecture.TransformerBlock.forward_cross, dynamic=True, fullgraph=True
    )
    _installed = True
