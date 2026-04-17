"""KV cache data structures for explicit cache passing through architectures.

Provides cache containers for storing key-value projections from attention
layers, enabling efficient inference by reusing computed values across
different test sets without storing state inside the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@dataclass
class KVCacheEntry:
    """A single key-value cache entry for one attention layer.

    Attributes:
        key: Cached key projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
        value: Cached value projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
    """

    key: Tensor | None = None
    value: Tensor | None = None

    def is_valid(self) -> bool:
        """Check if this cache entry contains valid data."""
        return self.key is not None and self.value is not None

    def to(self, device: torch.device | str) -> KVCacheEntry:
        """Move this entry to the given device. Returns a new KVCacheEntry."""
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(key=self.key.to(device), value=self.value.to(device))


@dataclass
class KVCache:
    """Maps layer indices to KVCacheEntry objects.

    Attributes:
        kv: Maps layer/block index to cached key-value projections.
    """

    kv: dict[int, KVCacheEntry] = field(default_factory=dict)

    def is_populated(self) -> bool:
        """True when the cache contains valid data."""
        return any(entry.is_valid() for entry in self.kv.values())

    def to(self, device: torch.device | str) -> KVCache:
        """Move all entries to the given device. Returns a new KVCache."""
        return KVCache(kv={idx: entry.to(device) for idx, entry in self.kv.items()})
