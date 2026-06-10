#  Copyright (c) Prior Labs GmbH 2026.

"""KV cache data structures for explicit cache passing through architectures.

Provides cache containers for storing key-value projections from attention
layers, enabling efficient inference by reusing computed values across
different test sets without storing state inside the model.

Includes optional integer quantization (e.g. int8) via
:class:`QuantizedKVCacheEntry` for reduced memory footprint with per-tensor
symmetric quantization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

# Low, high, max-magnitude value for each dtype.
# int8 uses the symmetric range [-127, 127] (one code below the full int8
# range) so that ``-max * scale`` equals ``+max * scale`` and dequantization
# cannot exceed the original absmax in magnitude.
_QUANTIZATION_RANGES: dict[torch.dtype, tuple[int, int, int]] = {
    torch.int8: (-127, 127, 127),
}


def _quantize_tensor(
    t: Tensor, dtype: torch.dtype = torch.int8
) -> tuple[Tensor, Tensor]:
    """Per-tensor symmetric quantization to the given integer *dtype*.

    Returns ``(quantized, scale)`` where
    ``scale = absmax / max_val`` and ``quantized = round(t / scale)``.
    """
    if dtype not in _QUANTIZATION_RANGES:
        raise ValueError(
            f"Unsupported quantization dtype {dtype}. "
            f"Supported: {list(_QUANTIZATION_RANGES)}"
        )
    lo, hi, max_val = _QUANTIZATION_RANGES[dtype]
    absmax = t.abs().amax()
    scale = absmax / float(max_val)
    # Avoid division by zero for all-zero tensors; floor at scale.dtype's
    # smallest positive normal so the clamp is representable in any dtype.
    scale = torch.clamp(scale, min=torch.finfo(scale.dtype).tiny)
    quantized = (t / scale).round().clamp(lo, hi).to(dtype)
    return quantized, scale


def _dequantize_tensor(t: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """Dequantize an integer tensor back to floating-point *dtype*."""
    return t.to(dtype) * scale.to(dtype)


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

    def quantize(self, dtype: torch.dtype = torch.int8) -> QuantizedKVCacheEntry:
        """Quantize this entry with per-tensor symmetric scaling.

        Args:
            dtype: Target integer dtype (default ``torch.int8``).
        """
        assert self.is_valid()
        k_q, k_s = _quantize_tensor(self.key, dtype)
        v_q, v_s = _quantize_tensor(self.value, dtype)
        return QuantizedKVCacheEntry(key=k_q, value=v_q, key_scale=k_s, value_scale=v_s)


@dataclass
class QuantizedKVCacheEntry:
    """Quantized key-value cache entry with per-tensor scale factors.

    Stores K/V as integer tensors alongside scalar scale factors for
    symmetric quantization: ``float_value = int_value * scale``. The
    integer dtype is implicit in the stored tensors (see ``self.key.dtype``);
    the scale already encodes the dtype's quantization range, so dequantizing
    requires no extra dtype bookkeeping.

    Attributes:
        key: Quantized key projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
        value: Quantized value projections, shape ``(B, N_train, num_kv_heads,
        head_dim)``.
        key_scale: Scalar scale factor for keys.
        value_scale: Scalar scale factor for values.
    """

    key: Tensor | None = None
    value: Tensor | None = None
    key_scale: Tensor | None = None
    value_scale: Tensor | None = None

    def is_valid(self) -> bool:
        """Check if this cache entry contains valid data."""
        return (
            self.key is not None
            and self.value is not None
            and self.key_scale is not None
            and self.value_scale is not None
        )

    def to(self, device: torch.device | str) -> QuantizedKVCacheEntry:
        """Move this entry to the given device."""
        if not self.is_valid():
            return QuantizedKVCacheEntry()
        return QuantizedKVCacheEntry(
            key=self.key.to(device),
            value=self.value.to(device),
            key_scale=self.key_scale.to(device),
            value_scale=self.value_scale.to(device),
        )

    def dequantize(self, dtype: torch.dtype) -> KVCacheEntry:
        """Dequantize back to a full-precision :class:`KVCacheEntry`."""
        assert self.is_valid()
        return KVCacheEntry(
            key=_dequantize_tensor(self.key, self.key_scale, dtype),
            value=_dequantize_tensor(self.value, self.value_scale, dtype),
        )


@dataclass
class KVCache(ABC):
    """Maps layer indices to KVCacheEntry or QuantizedKVCacheEntry objects.

    This is the base class for the architecture-specific caches. These
    store the per-layer key/value projections in ``kv`` and add their own fitted
    preprocessing / embedding state as extra fields.

    Attributes:
        kv: Maps layer/block index to cached key-value projections.
    """

    kv: dict[int, KVCacheEntry | QuantizedKVCacheEntry] = field(default_factory=dict)

    def is_populated(self) -> bool:
        """True when the cache contains valid data."""
        return any(entry.is_valid() for entry in self.kv.values())

    def is_empty(self) -> bool:
        """True when the cache has not been populated yet."""
        return not self.is_populated()

    @abstractmethod
    def to(self, device: torch.device | str) -> KVCache:
        """Move all entries to the given device. Returns a new KVCache."""
        return KVCache(kv=self._kv_to(device))

    def _kv_to(
        self, device: torch.device | str
    ) -> dict[int, KVCacheEntry | QuantizedKVCacheEntry]:
        """Move the per-layer KV entries to the given device."""
        return {idx: entry.to(device) for idx, entry in self.kv.items()}

    @staticmethod
    def _dict_of_tensors_to(
        state: dict[str, Tensor] | None, device: torch.device | str
    ) -> dict[str, Tensor] | None:
        """Move a dict of tensors to the given device (passing through ``None``)."""
        if state is None:
            return None
        return {k: v.to(device) for k, v in state.items()}

    @staticmethod
    def _list_of_tensors_to(
        tensors: list[Tensor] | None, device: torch.device | str
    ) -> list[Tensor] | None:
        """Move a list of tensors to the given device (passing through ``None``)."""
        if tensors is None:
            return None
        return [t.to(device) for t in tensors]
