#  Copyright (c) Prior Labs GmbH 2026.
"""Registry for swappable attention backends.

Attention calls in the architectures funnel through the shared SDPA wrapper,
``scaled_dot_product_attention``, which asks this registry who should take each
one — so an alternative kernel does not have to live in this package.

Registration is **explicit**, with no discovery: the shared SDPA module
registers the in-tree backends at import, and external packages call
:func:`register_attention_backend` from their own setup function.

A backend is offered an :class:`AttentionSpec` — the call's geometry, not its
tensors — and cannot tell which stage of which architecture it belongs to.
Backends are a *performance* seam and fail open: if none prefers the call,
the ordinary SDPA path runs.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from tabpfn.architectures.kv_cache import QuantizedKVCacheEntry


@dataclasses.dataclass(frozen=True)
class AttentionSpec:
    """The shapes of one attention call, without its tensors.

    Built by the SDPA wrapper from the live tensors, so the numbers describe
    exactly the call about to run. A field is ``None`` only when the call
    does not have it (e.g. no keys of their own on the quantized-cache
    path); a backend whose decision depends on such a field must decline.
    """

    seq_len_q: int | None
    """Query sequence length."""

    seq_len_kv: int | None
    """Key/value sequence length; ``None`` if the call has no keys."""

    num_heads: int
    """Number of query heads."""

    num_kv_heads: int
    """Number of key/value heads (< ``num_heads`` under GQA/MQA)."""

    head_dim: int
    """Dimensionality of each head."""

    dtype: torch.dtype
    """Dtype of q/k/v at the call."""

    device: torch.device
    """Device the call runs on."""

    batch_size: int | None
    """SDPA batch of this call, including any dimensions folded into it."""

    quantized_kv_dtype: torch.dtype | None = None
    """Storage dtype of the quantized KV cache entry, or ``None`` when keys/
    values arrive as regular tensors."""

    is_grad_enabled: bool = False
    """Whether grad mode is on, i.e. whether a backward may follow."""


@runtime_checkable
class AttentionBackend(Protocol):
    """A drop-in replacement for one attention call.

    All tensors use the ``(B, S, H, D)`` layout of
    ``shared.scaled_dot_product_attention``. Backends receive dense
    ``k``/``v`` — the SDPA wrapper dequantizes a quantized KV cache entry
    exactly once, before dispatch. A backend that wants the entry *as
    stored* (to feed its kernel without materializing a dense copy) sets
    the class attribute ``consumes_quantized_kv = True``; it then receives
    ``quantized_kv`` with ``k``/``v`` as ``None`` whenever the call carries
    a quantized cache.

    Implementations should name only the keyword arguments they consume and
    absorb the rest with ``**kwargs``: informational context may grow over
    time, and an open signature keeps existing backends compatible.

    ``is_preferred`` sees only the :class:`AttentionSpec`, never the tensors,
    and must be cheap: it runs for every attention call. A backend that can
    be selected inside a ``torch.compile``-d region must have a
    Dynamo-traceable ``run`` — the registry does not check this; it is the
    registrant's responsibility.
    """

    name: str
    """Unique backend name."""

    def is_preferred(self, spec: AttentionSpec) -> bool:
        """Whether this backend should take calls matching *spec*."""
        ...

    def run(
        self,
        q_BSHD: torch.Tensor,
        k_BSJD: torch.Tensor | None,
        v_BSJD: torch.Tensor | None,
        *,
        quantized_kv: QuantizedKVCacheEntry | None = None,
    ) -> torch.Tensor:
        """Compute attention, returning ``(B, S_q, H, D)`` in a float dtype."""
        ...


_logger = logging.getLogger(__name__)

_registry: dict[str, AttentionBackend] = {}
_consult_order: tuple[AttentionBackend, ...] = ()
_lock = threading.Lock()


def register_attention_backend(*backends: AttentionBackend) -> None:
    """Register *backends*, consulted in the given order, before all others.

    The new backends go to the front of the consult order (in the order
    given), ahead of anything registered earlier. Re-entrant: registering an
    already-registered object again is a no-op that keeps its position, so
    setup hooks may run repeatedly (e.g. once per worker process bootstrap).

    Raises:
        ValueError: If a *different* backend is already registered under the
            same name.
    """
    global _consult_order  # noqa: PLW0603
    with _lock:
        new: list[AttentionBackend] = []
        for backend in backends:
            existing = _registry.get(backend.name)
            if existing is not None and existing is not backend:
                raise ValueError(
                    f"A different attention backend is already registered under "
                    f"{backend.name!r}: {existing!r}"
                )
            if existing is None:
                _registry[backend.name] = backend
                new.append(backend)
        _consult_order = (*new, *_consult_order)
        _logger.debug(
            "registered attention backends, consult order: %s (registration "
            "only makes a backend available — whether it can run here is "
            "decided per call by its is_preferred)",
            [b.name for b in _consult_order],
        )


def unregister_attention_backend(name: str) -> None:
    """Remove the backend registered under *name*, if any."""
    global _consult_order  # noqa: PLW0603
    with _lock:
        if _registry.pop(name, None) is not None:
            _consult_order = tuple(b for b in _consult_order if b.name != name)
        _logger.debug(
            "unregistered attention backend %r, consult order: %s",
            name,
            [b.name for b in _consult_order],
        )


def registered_attention_backends() -> tuple[AttentionBackend, ...]:
    """All registered backends in consult order (first wins)."""
    return _consult_order


def _spec_from_tensors(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor | None,
    v_BSJD: torch.Tensor | None,  # noqa: ARG001  (symmetry with run())
    *,
    quantized_kv: QuantizedKVCacheEntry | None = None,
) -> AttentionSpec:
    """Describe the call about to run, for the registry to match against."""
    k = quantized_kv.key if quantized_kv is not None else k_BSJD
    return AttentionSpec(
        seq_len_q=q_BSHD.shape[1],
        seq_len_kv=k.shape[1] if k is not None else None,
        num_heads=q_BSHD.shape[2],
        num_kv_heads=k.shape[2] if k is not None else q_BSHD.shape[2],
        head_dim=q_BSHD.shape[3],
        dtype=q_BSHD.dtype,
        device=q_BSHD.device,
        batch_size=q_BSHD.shape[0],
        quantized_kv_dtype=(
            quantized_kv.key.dtype
            if quantized_kv is not None and quantized_kv.key is not None
            else None
        ),
        is_grad_enabled=torch.is_grad_enabled(),
    )


def find_attention_backend(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor | None,
    v_BSJD: torch.Tensor | None,
    *,
    quantized_kv: QuantizedKVCacheEntry | None = None,
) -> AttentionBackend | None:
    """The backend for this call, or ``None`` for the standard SDPA path.

    Describes the call, then takes the first registered backend that prefers
    it. This is the one selection routine: the architectures call it (through
    the SDPA wrapper) and own no selection logic themselves.
    """
    spec = _spec_from_tensors(q_BSHD, k_BSJD, v_BSJD, quantized_kv=quantized_kv)
    backend: AttentionBackend | None = None
    for candidate in _consult_order:
        if candidate.is_preferred(spec):
            backend = candidate
            break
    # Not while tracing: logging is not traceable (it breaks the graph).
    if not torch.compiler.is_compiling() and _logger.isEnabledFor(logging.DEBUG):
        _log_selection(spec, backend)
    return backend


def _log_selection(spec: AttentionSpec, backend: AttentionBackend | None) -> None:
    """Debug-log one selection (the shapes make the stage recognisable)."""
    _logger.debug(
        "attention q=%s kv=%s heads=%s/%s head_dim=%s -> %s",
        spec.seq_len_q,
        spec.seq_len_kv,
        spec.num_heads,
        spec.num_kv_heads,
        spec.head_dim,
        "sdpa" if backend is None else backend.name,
    )
