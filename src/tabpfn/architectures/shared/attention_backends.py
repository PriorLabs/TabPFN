#  Copyright (c) Prior Labs GmbH 2026.
"""Registry for swappable attention backends.

Attention calls in the architectures funnel through the shared
``scaled_dot_product_attention`` chokepoint. This registry lets a backend
take over such a call with its own kernel — the in-tree FA3/MLX dispatch pattern,
generalized so the implementation does not have to live in this package.

Registration is **explicit**: the shared SDPA module registers the in-tree
backends in one call at import, and external packages call
:func:`register_attention_backend` from their own setup/enable function, once
per process. There is no automatic discovery. Each call places its backends
at the *front* of the consult order, in the order given — so the in-tree
defaults are consulted in their listed order, and anything registered
afterwards (e.g. an external backend at its enable call) is consulted before
them. Registration is re-entrant (re-registering the same object is a no-op,
keeping its position).

Selection is **shape-based, not tensor-based**: a backend's
:meth:`AttentionBackend.is_preferred` receives an :class:`AttentionSpec` —
the static description of an attention call (sequence lengths, head
geometry, dtype, device) — so architectures can resolve the backend for each
of their attention stages *upfront*, once per forward pass, when the input
shapes are known and before any layer runs. The spec is deliberately pure
geometry: backends cannot tell *which* stage of *which* architecture a call
belongs to, only what the call looks like. The resolved backend (or ``None``)
is then passed down to the attention chokepoints. Callers that have no plan
pass nothing and get per-call resolution from the live tensors instead.

Backends are a *performance* seam and therefore fail open: no registered
backend, or none preferred, means the ordinary SDPA path runs. Any policy
about *when* a backend should engage belongs to the backend and whoever
registers it.
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
    """Static description of one attention call, known before any tensor exists.

    Built by an architecture's planner from the forward-pass input shapes (or
    from live tensors for unplanned callers). Fields typed ``| None`` mean
    *unknown at plan time* (e.g. chunk-dependent): a backend whose decision
    depends on such a field must decline when it is ``None``.
    """

    seq_len_q: int | None
    """Query sequence length; an upper bound for chunk-dependent stages
    (in-forward chunking only shrinks it); ``None`` when unknown."""

    seq_len_kv: int | None
    """Key/value sequence length; an upper bound for chunk-dependent stages
    (in-forward chunking only shrinks it); ``None`` when unknown."""

    num_heads: int
    """Number of query heads."""

    num_kv_heads: int
    """Number of key/value heads (< ``num_heads`` under GQA/MQA)."""

    head_dim: int
    """Dimensionality of each head."""

    dtype: torch.dtype
    """Effective dtype of q/k/v at the call (autocast-aware)."""

    device: torch.device
    """Device the call runs on."""

    batch_size: int | None
    """Effective SDPA batch (with any folded dims); ``None`` when
    chunk-dependent. Where known upfront it is an upper bound — in-forward
    chunking only shrinks it."""

    quantized_kv_dtype: torch.dtype | None = None
    """Storage dtype of the quantized KV cache entry, or ``None`` when keys/
    values arrive as regular tensors."""

    is_grad_enabled: bool = False
    """Grad mode the call runs under, snapshot at plan time."""


@runtime_checkable
class AttentionBackend(Protocol):
    """A drop-in replacement for one attention call.

    All tensors use the ``(B, S, H, D)`` layout of
    ``shared.scaled_dot_product_attention``. Backends receive dense
    ``k``/``v`` — the chokepoint dequantizes a quantized KV cache entry
    exactly once, before dispatch. A backend that wants the entry *as
    stored* (to feed its kernel without materializing a dense copy) sets
    the class attribute ``consumes_quantized_kv = True``; it then receives
    ``quantized_kv`` with ``k``/``v`` as ``None`` whenever the call carries
    a quantized cache.

    Implementations should name only the keyword arguments they consume and
    absorb the rest with ``**kwargs``: informational context may grow over
    time, and an open signature keeps existing backends compatible.

    ``is_preferred`` sees only the :class:`AttentionSpec`, never tensors: it
    runs at plan time, before the tensors of the call exist. A backend that
    can be planned into a ``torch.compile``-d region must have a
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


def resolve_attention_backend(spec: AttentionSpec) -> AttentionBackend | None:
    """The first backend in consult order preferring *spec*, or ``None``.

    This is the single selection routine, used both by architectures planning
    their stages upfront and by the per-call fallback for unplanned callers.
    """
    for backend in _consult_order:
        if backend.is_preferred(spec):
            return backend
    return None


def effective_attention_dtype(
    device: torch.device, fallback: torch.dtype
) -> torch.dtype:
    """The dtype q/k/v will actually have, accounting for autocast.

    Inference commonly runs the forward under ``torch.autocast``, in which
    case the projections produce the autocast dtype regardless of the input
    dtype — planning from the input dtype would mis-describe the call.
    """
    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    return fallback


def spec_from_tensors(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor | None,
    v_BSJD: torch.Tensor | None,  # noqa: ARG001  (symmetry with run())
    *,
    quantized_kv: QuantizedKVCacheEntry | None = None,
) -> AttentionSpec:
    """Describe a live attention call, for per-call (unplanned) resolution."""
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
    """Per-call resolution from live tensors, for callers without a plan."""
    return resolve_attention_backend(
        spec_from_tensors(q_BSHD, k_BSJD, v_BSJD, quantized_kv=quantized_kv)
    )
