#  Copyright (c) Prior Labs GmbH 2026.

"""KV cache data structures for explicit cache passing through architectures.

Provides cache containers for storing key-value projections from attention
layers, enabling efficient inference by reusing computed values across
different test sets without storing state inside the model.

Includes optional quantization (int8 or fp8) via
:class:`QuantizedKVCacheEntry` for reduced memory footprint with per-tensor
symmetric quantization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

QUANTIZED_KV_DTYPE: torch.dtype = torch.int8  # default
FP8_KV_DTYPE: torch.dtype = torch.float8_e4m3fn

#: Storage dtype for each quantized ``kv_cache_precision`` value.
KV_CACHE_PRECISION_DTYPES: dict[str, torch.dtype] = {
    "int8": QUANTIZED_KV_DTYPE,
    "fp8": FP8_KV_DTYPE,
    # The dtype when no attention backend left the keys and values on its grid.
    "adaptive": QUANTIZED_KV_DTYPE,
}

# Low, high, max-magnitude value for each integer dtype.
# int8 uses the symmetric range [-127, 127] (one code below the full int8
# range) so that ``-max * scale`` equals ``+max * scale`` and dequantization
# cannot exceed the original absmax in magnitude.
_QUANTIZATION_RANGES: dict[torch.dtype, tuple[int, int, int]] = {
    torch.int8: (-127, 127, 127),
}

# Float dtypes are scaled onto the representable range and rounded by the
# dtype cast. e4m3fn has no infinity, so the clamp is required: casting a
# value above the max would produce NaN.
_FLOAT_QUANTIZATION_DTYPES = (torch.float8_e4m3fn,)


def _quantize_tensor(
    t: Tensor, dtype: torch.dtype = torch.int8, *, per_batch_element: bool = False
) -> tuple[Tensor, Tensor]:
    """Symmetric quantization to the given *dtype*.

    Returns ``(quantized, scale)`` where ``scale = absmax / max_val`` and
    ``quantized ~= t / scale``, so ``float = quantized * scale``. The scale is a
    scalar over the whole tensor, or, with ``per_batch_element`` and a leading
    dimension above one, one scalar per leading index shaped ``(B, 1, ..., 1)`` so
    that every batch element keeps its own range, as it would quantized alone.
    """
    if dtype in _QUANTIZATION_RANGES:
        lo, hi, max_val = _QUANTIZATION_RANGES[dtype]
    elif dtype in _FLOAT_QUANTIZATION_DTYPES:
        max_val = torch.finfo(dtype).max
        lo, hi = -max_val, max_val
    else:
        raise ValueError(
            f"Unsupported quantization dtype {dtype}. Supported: "
            f"{list(_QUANTIZATION_RANGES) + list(_FLOAT_QUANTIZATION_DTYPES)}"
        )
    if per_batch_element and t.dim() > 1 and t.shape[0] > 1:
        absmax = t.abs().amax(dim=tuple(range(1, t.dim())), keepdim=True)
    else:
        absmax = t.abs().amax()
    scale = absmax / float(max_val)
    # Avoid division by zero for all-zero tensors; floor at scale.dtype's
    # smallest positive normal so the clamp is representable in any dtype.
    scale = torch.clamp(scale, min=torch.finfo(scale.dtype).tiny)
    scaled = t / scale
    if dtype in _QUANTIZATION_RANGES:
        scaled = scaled.round()
    quantized = scaled.clamp(lo, hi).to(dtype)
    return quantized, scale


def _dequantize_tensor(t: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """Dequantize a quantized tensor back to floating-point *dtype*."""
    return t.to(dtype) * scale.to(dtype)


@dataclass
class KVCacheEntry:
    """A single key-value cache entry for one attention layer.

    Attributes:
        key: Cached key projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
        value: Cached value projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
        grid_dtype: The lower-precision dtype whose grid ``key`` and ``value``
            already lie on, when the attention call that produced them rounded
            them there. Storing the entry at this dtype loses nothing.
    """

    key: Tensor | None = None
    value: Tensor | None = None
    grid_dtype: torch.dtype | None = None

    def is_valid(self) -> bool:
        """Check if this cache entry contains valid data."""
        return self.key is not None and self.value is not None

    def to(self, device: torch.device | str) -> KVCacheEntry:
        """Move this entry to the given device. Returns a new KVCacheEntry."""
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(
            key=self.key.to(device),
            value=self.value.to(device),
            grid_dtype=self.grid_dtype,
        )

    def at_storage_dtype(
        self, dtype: torch.dtype | None, *, follow_grid: bool = False
    ) -> KVCacheEntry | QuantizedKVCacheEntry:
        """This entry as the cache stores it.

        Args:
            dtype: The storage dtype, or ``None`` to keep the computed dtype.
            follow_grid: Store at ``grid_dtype`` instead, when it is set
                (``kv_cache_precision="adaptive"``).
        """
        if follow_grid and self.grid_dtype is not None:
            dtype = self.grid_dtype
        return self if dtype is None else self.quantize(dtype)

    @staticmethod
    def concatenate(entries: Sequence[KVCacheEntry]) -> KVCacheEntry:
        """One entry holding the batch elements of ``entries`` in order."""
        assert all(entry.is_valid() for entry in entries)
        return KVCacheEntry(
            key=torch.cat([entry.key for entry in entries]),
            value=torch.cat([entry.value for entry in entries]),
        )

    def quantize(
        self, dtype: torch.dtype = QUANTIZED_KV_DTYPE
    ) -> QuantizedKVCacheEntry:
        """Quantize this entry with per-tensor symmetric scaling.

        Args:
            dtype: Target quantization dtype (default `QUANTIZED_KV_DTYPE`).
        """
        assert self.is_valid()
        k_q, k_s = _quantize_tensor(self.key, dtype, per_batch_element=True)
        v_q, v_s = _quantize_tensor(self.value, dtype, per_batch_element=True)
        return QuantizedKVCacheEntry(key=k_q, value=v_q, key_scale=k_s, value_scale=v_s)


@dataclass
class QuantizedKVCacheEntry:
    """Quantized key-value cache entry with scale factors.

    Stores K/V as quantized tensors alongside scale factors for symmetric
    quantization: ``float_value = quantized_value * scale``. The quantized
    dtype is implicit in the stored tensors (see ``self.key.dtype``);
    the scale already encodes the dtype's quantization range, so dequantizing
    requires no extra dtype bookkeeping.

    Attributes:
        key: Quantized key projections, shape ``(B, N_train, num_kv_heads, head_dim)``.
        value: Quantized value projections, shape ``(B, N_train, num_kv_heads,
        head_dim)``.
        key_scale: Scale factor for keys: a scalar, or ``(B, 1, 1, 1)`` when the
            entry holds several batch elements, each with its own scale.
        value_scale: Scale factor for values, shaped like ``key_scale``.
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

    @staticmethod
    def concatenate(
        entries: Sequence[QuantizedKVCacheEntry],
    ) -> QuantizedKVCacheEntry:
        """One entry holding the batch elements of ``entries`` in order.

        A scalar scale is expanded to one per batch element first, so the result
        always scales per batch element.
        """
        assert all(entry.is_valid() for entry in entries)

        def per_element(scale: Tensor, batch: int) -> Tensor:
            return scale.reshape(-1, 1, 1, 1).expand(batch, 1, 1, 1)

        return QuantizedKVCacheEntry(
            key=torch.cat([entry.key for entry in entries]),
            value=torch.cat([entry.value for entry in entries]),
            key_scale=torch.cat(
                [per_element(e.key_scale, e.key.shape[0]) for e in entries]
            ),
            value_scale=torch.cat(
                [per_element(e.value_scale, e.value.shape[0]) for e in entries]
            ),
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

    @classmethod
    def concatenate(cls, caches: Sequence[KVCache]) -> KVCache:
        """One cache holding the batch elements of ``caches`` in order.

        Consumes ``caches``: their per-layer entries are released as the result is
        assembled, so the transient memory is one layer's worth rather than a
        second copy of the whole cache.
        """
        raise NotImplementedError(f"{cls.__name__} cannot concatenate caches.")

    @staticmethod
    def _kv_concatenate(
        caches: Sequence[KVCache],
    ) -> dict[int, KVCacheEntry | QuantizedKVCacheEntry]:
        """Concatenate the per-layer KV entries along the batch, layer by layer.

        Pops each layer from the source caches once it is copied.
        """
        layers = list(caches[0].kv)
        assert all(list(cache.kv) == layers for cache in caches)
        kv: dict[int, KVCacheEntry | QuantizedKVCacheEntry] = {}
        for idx in layers:
            entries = [cache.kv.pop(idx) for cache in caches]
            if isinstance(entries[0], QuantizedKVCacheEntry):
                kv[idx] = QuantizedKVCacheEntry.concatenate(entries)  # type: ignore[arg-type]
            else:
                kv[idx] = KVCacheEntry.concatenate(entries)  # type: ignore[arg-type]
        return kv

    @staticmethod
    def _cat_tensors(tensors: Sequence[Tensor | None], dim: int = 0) -> Tensor | None:
        """Concatenate along ``dim`` (passing through ``None``)."""
        if tensors[0] is None:
            assert all(t is None for t in tensors)
            return None
        return torch.cat(tensors, dim=dim)  # type: ignore[arg-type]

    @staticmethod
    def _cat_dicts_of_tensors(
        states: Sequence[dict[str, Tensor] | None],
    ) -> dict[str, Tensor] | None:
        """Concatenate each key's tensors along the batch (passing through ``None``)."""
        if states[0] is None:
            assert all(state is None for state in states)
            return None
        keys = list(states[0])
        assert all(list(state) == keys for state in states)  # type: ignore[arg-type]
        return {k: torch.cat([state[k] for state in states]) for k in keys}  # type: ignore[index]

    @staticmethod
    def _cat_lists_of_tensors(
        lists: Sequence[list[Tensor] | None],
    ) -> list[Tensor] | None:
        """Concatenate the tensors at each position along the batch."""
        if lists[0] is None:
            assert all(tensors is None for tensors in lists)
            return None
        return [torch.cat(tensors) for tensors in zip(*lists, strict=True)]  # type: ignore[arg-type]

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
