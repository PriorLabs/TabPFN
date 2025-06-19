from __future__ import annotations

from abc import ABC, abstractmethod
from typing_extensions import override

import torch
from torch import nn


class Attention(ABC, nn.Module):
    """Base class for attention layers."""

    @override
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        cache_kv: bool = False,
        use_cached_kv: bool = False,
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
        use_second_set_of_queries: bool = False,
        save_peak_mem_factor: int | None = None,
        add_input: bool = False,
        allow_inplace: bool = False,
    ) -> torch.Tensor:
        """Performs the attention layer.

        Args:
            x: Input sequence of embeddings with shape
                [batch... x query seq len x embedding dim].
                If `x_kv` is None, this is used to compute the queries, keys, and
                values.
                If `x_kv` is not None, this is used to compute the queries only.
            x_kv: If not None, an input sequence of embeddings with shape
                [batch... x kv seq len x embedding dim].
                It will be used to compute the keys and the values, with `x` used only
                to compute the queries. This is useful to avoid some sequence positions
                attending to others.
            cache_kv: If True, replaces the current key-value cache with the keys and
                values computed during this forward pass. Otherwise, the KV cache is
                left unchanged. If True, `use_cached_kv` must be False.
            use_cached_kv: If True, uses the keys and values cached during a previous
                forward pass when `cache_kv` was True. If True, `cache_kv` must be
                False.
            reuse_first_head_kv: If True, then uses the keys and values projected in the
                first head for all the heads.
            only_cache_first_head_kv: When True, only caches the keys and values of the
                first head.
            save_peak_mem_factor: Loops over the batch dimension rather than processing
                it in parallel, to reduce memory usage.
            use_second_set_of_queries: About to be removed in
                https://github.com/PriorLabs/TabPFN-private/pull/7
            add_input: If True, returns (x+the output of attention), i.e. enables a
                residual connection. If False, just returns the output of attention.
            allow_inplace: By setting this to True, the caller indicates that 'x' is not
                used after the call and its buffer can be reused for the output.
                The operation is not guaranteed to be inplace.

        Returns:
            Output hidden states of shape [batch x query seq len x embedding dim]
        """
        ...
