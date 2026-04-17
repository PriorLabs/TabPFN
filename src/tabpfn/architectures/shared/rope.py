"""Rotary positional embedding.

Adapted from https://github.com/lucidrains/rotary-embedding-torch
"""

from __future__ import annotations

from math import pi
from typing import Any, Literal, cast

import torch
from einops import rearrange
from torch import Tensor, einsum, nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for use in transformer models.

    Rotary embeddings encode positional information that allows continuous
    rotation of embeddings, enhancing the model's ability to capture
    long-range dependencies and positional relations.

    Args:
        dim: Dimension of the embeddings.
        interleaved: If True, rotates adjacent pairs (0,1), (2,3), etc.
            If False, splits into first half [0:d//2] and second half [d//2:d].
        custom_freqs: Custom frequency tensor. If provided, overrides the
            frequencies computed from `freqs_for`.
        freqs_for: Type of default frequencies: 'lang' for language models,
            'pixel' for image data, or 'constant' for a fixed frequency.
        theta: Base scaling factor for the rotary frequencies.
        max_freq: Maximum frequency for pixel-based embeddings.
        num_freqs: Number of frequencies when `freqs_for='constant'`.
        learned_freq: If True, frequencies are learnable parameters.
        use_xpos: If True, applies XPOS extrapolatable position scaling.
        xpos_scale_base: Base used to compute XPOS position scales.
        interpolate_factor: Factor by which sequence positions are scaled
            down to support longer contexts without fine-tuning.
        theta_rescale_factor: Multiplier applied to `theta` to rescale
            rotary embeddings for longer sequence lengths (NTK-aware scaling).
        seq_before_head_dim: If True, the sequence dimension precedes the
            head dimension in input tensors.
        cache_if_possible: If True, computed frequencies and scales are
            cached across calls.

    Attributes:
        freqs_for: Type of frequencies in use.
        cache_if_possible: Whether frequency caching is enabled.
        freqs: Frequency parameters, optionally learnable.
        learned_freq: Whether frequencies are learnable.
        seq_before_head_dim: Whether sequence precedes head dimension.
        default_seq_dim: Default sequence dimension (-2 or -3).
        interpolate_factor: Position interpolation factor.
        interleaved: Whether interleaved rotation mode is active.
        use_xpos: Whether XPOS scaling is active.
        scale_base: Base for XPOS scale computation.
    """

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        *,
        interleaved: bool = True,
        custom_freqs: Tensor | None = None,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta: float = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        learned_freq: bool = False,
        use_xpos: bool = False,
        xpos_scale_base: int = 512,
        interpolate_factor: float = 1.0,
        theta_rescale_factor: float = 1.0,
        seq_before_head_dim: bool = False,
        cache_if_possible: bool = True,
    ) -> None:
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings for longer
        # sequence lengths without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if _exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self._tmp_store("cached_freqs", None)
        self._tmp_store("cached_scales", None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)  # type: ignore

        self.learned_freq = learned_freq

        # dummy for device
        self._tmp_store("dummy", torch.tensor(0))

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        self.interleaved = interleaved

        self.use_xpos = use_xpos
        if not use_xpos:
            self._tmp_store("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self._tmp_store("scale", scale)

    @property
    def device(self) -> torch.device:
        """The device on which the embedding parameters reside."""
        return cast("torch.device", self.dummy.device)

    def rotate_queries_or_keys(
        self,
        t: Tensor,
        seq_dim: int | None = None,
        offset: int = 0,
        scale: float | Tensor | None = None,
    ) -> Tensor:
        """Apply rotary embeddings to a queries or keys tensor.

        Args:
            t: Queries or keys tensor to rotate.
            seq_dim: Sequence dimension of `t`. Defaults to
                `self.default_seq_dim`.
            offset: Position offset added to sequence indices.
            scale: Explicit XPOS scale. Must be provided when `use_xpos=True`;
                use `rotate_queries_and_keys` instead in that case.

        Returns:
            Tensor of the same shape as `t` with rotary embeddings applied.
        """
        seq_dim_int: int = _default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or _exists(scale), (
            "you must use `.rotate_queries_and_keys` method instead and pass in both "
            "queries and keys, for length extrapolatable rotary embeddings"
        )

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim_int]

        seq = self._get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        if self.interleaved:
            freqs = self.forward(seq, seq_len=seq_len, offset=offset)
        else:
            # For non-interleaved mode, compute freqs without repetition
            # freqs shape: (seq_len, half) instead of (seq_len, dim)
            freqs = einsum("..., f -> ... f", seq.type(self.freqs.dtype), self.freqs)

        if seq_dim_int == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")

        return _apply_rotary_emb(
            freqs,
            t,
            scale=_default(scale, 1.0),
            seq_dim=seq_dim_int,
            interleaved=self.interleaved,
        )

    def rotate_queries_and_keys(
        self, q: Tensor, k: Tensor, seq_dim: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary embeddings with XPOS scale to both queries and keys.

        Requires `use_xpos=True`. Queries receive the forward scale and keys
        receive the inverse scale.

        Args:
            q: Query tensor.
            k: Key tensor.
            seq_dim: Sequence dimension. Defaults to `self.default_seq_dim`.

        Returns:
            Tuple of (rotated_q, rotated_k) with the same shapes as the inputs.
        """
        seq_dim_int: int = _default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim_int]

        seq = self._get_seq_pos(seq_len, dtype=dtype, device=device)

        if self.interleaved:
            freqs = self.forward(seq, seq_len=seq_len)
        else:
            # For non-interleaved mode, compute freqs without repetition
            freqs = einsum("..., f -> ... f", seq.type(self.freqs.dtype), self.freqs)
        scale = cast("Tensor", self._get_scale(seq, seq_len=seq_len)).to(dtype)

        if seq_dim_int == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = _apply_rotary_emb(
            freqs, q, scale=scale, seq_dim=seq_dim_int, interleaved=self.interleaved
        )
        rotated_k = _apply_rotary_emb(
            freqs, k, scale=scale**-1, seq_dim=seq_dim_int, interleaved=self.interleaved
        )

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def _tmp_store(self, key: str, value: Tensor | None) -> None:
        """Register a non-persistent buffer.

        Args:
            key: Name of the buffer attribute.
            value: Tensor to store, or None to clear the buffer.
        """
        self.register_buffer(key, value, persistent=False)

    def _get_seq_pos(
        self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0
    ) -> Tensor:
        """Compute sequence position indices scaled by the interpolation factor.

        Args:
            seq_len: Length of the sequence.
            device: Target device for the returned tensor.
            dtype: Target dtype for the returned tensor.
            offset: Starting offset added to the position indices.

        Returns:
            1-D position tensor of length `seq_len` scaled by
            1/interpolate_factor.
        """
        return (
            torch.arange(seq_len, device=device, dtype=dtype) + offset
        ) / self.interpolate_factor

    def _get_scale(
        self, t: Tensor, seq_len: int | None = None, offset: int = 0
    ) -> float | Tensor:
        """Compute the XPOS position scale for a sequence of positions.

        Args:
            t: Position tensor.
            seq_len: Sequence length used as the cache key. Cache lookup
                is skipped when None.
            offset: Starting offset into the cached scale buffer.

        Returns:
            Scale tensor of shape [seq_len, dim], or 1.0 as a fallback
            before the scale has been initialised.
        """
        assert self.use_xpos

        should_cache = self.cache_if_possible and _exists(seq_len)

        if should_cache and _exists(self.cached_scales):
            assert seq_len is not None
            if (seq_len + offset) <= self.cached_scales.shape[0]:
                return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self._tmp_store("cached_scales", cast("Tensor", scale))

        return scale

    def forward(self, t: Tensor, seq_len: int | None = None, offset: int = 0) -> Tensor:  # type: ignore[override]
        """Compute interleaved rotary frequency tensor with optional caching.

        Args:
            t: Position tensor of shape [seq_len].
            seq_len: Sequence length used as the cache key. When provided and
                caching is enabled, the result is stored/retrieved from
                `cached_freqs`.
            offset: Starting offset into the cached frequency buffer.

        Returns:
            Frequency tensor of shape [seq_len, dim] with interleaved pairs,
            i.e. each frequency f is repeated as (f, f) in adjacent positions.
        """
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and _exists(seq_len)
            and self.freqs_for != "pixel"
        )

        if should_cache and _exists(self.cached_freqs):
            assert seq_len is not None
            if (offset + seq_len) <= self.cached_freqs.shape[0]:
                return self.cached_freqs[offset : (offset + seq_len)].detach()

        # Disable autocast so frequencies are always computed in full precision,
        # regardless of the active autocast dtype on any device type.
        with torch.autocast(t.device.type, enabled=False):
            freqs = self.freqs
            freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
            freqs = freqs.repeat_interleave(2, dim=-1)

        if should_cache:
            self._tmp_store("cached_freqs", freqs.detach())

        return freqs


def _exists(val: Any) -> bool:
    """Return True if `val` is not None."""
    return val is not None


def _default(val: Any, d: Any) -> Any:
    """Return `val` if it is not None, otherwise return `d`."""
    return val if _exists(val) else d


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    """Rotate pairs of adjacent dimensions: (0,1), (2,3), (4,5), etc.

    Rearranges input [..., d] to [..., d/2, 2] and rotates each adjacent
    pair by 90 degrees. Used by default in most RoPE implementations
    (e.g. LLaMA).

    Args:
        x: Input tensor of shape [..., d] where d is even.

    Returns:
        Rotated tensor of the same shape as `x`.
    """
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def _rotate_half_contiguous(x: Tensor) -> Tensor:
    """Rotate by splitting the last dimension into contiguous halves.

    Splits input [..., d] into [..., :d/2] (x1) and [..., d/2:] (x2),
    then returns [-x2, x1] concatenated along the last dimension.

    Args:
        x: Input tensor of shape [..., d] where d is even.

    Returns:
        Rotated tensor of the same shape as `x`.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(
    freqs: Tensor,
    t: Tensor,
    start_index: int = 0,
    scale: float | Tensor = 1.0,
    seq_dim: int = -2,
    *,
    interleaved: bool = True,
) -> Tensor:
    """Apply rotary embeddings to a tensor.

    Computes x * cos(θ) + rotate(x) * sin(θ) for the portion of the input
    that overlaps with the frequency tensor.

    Args:
        freqs: Frequency tensor. For interleaved mode, shape [..., dim] where
            dim = 2 * half. For non-interleaved mode, shape [..., half].
        t: Input tensor to rotate.
        start_index: Starting index for rotation in the last dimension.
        scale: Scaling factor applied to the rotation.
        seq_dim: Sequence dimension of `t`.
        interleaved: If True, rotates adjacent pairs (0,1), (2,3), etc.
            If False, splits into first half [0:d//2] and second half
            [d//2:d].

    Returns:
        Rotated tensor with the same shape as `t`.
    """
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    # interleaved: freqs has shape (L, dim); non-interleaved: (L, dim//2)
    rot_dim = freqs.shape[-1] if interleaved else freqs.shape[-1] * 2

    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], (
        f"feature dimension {t.shape[-1]} is not of sufficient size "
        f"to rotate in all the positions {rot_dim}"
    )

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    # Formula: x * cos + rotate(x) * sin
    if interleaved:
        # Interleaved mode: dimension pairs are (0,1), (2,3), etc.
        # freqs has shape (..., dim) where dim = 2 * half
        cos_freq, sin_freq = freqs.cos(), freqs.sin()
        t_transformed = (t_middle * cos_freq * scale) + (
            _rotate_half_interleaved(t_middle) * sin_freq * scale
        )
    else:
        # Non-interleaved mode: split embedding into first half [0:d//2] and second half
        # [d//2:d]
        # freqs has shape (..., half) for non-interleaved mode
        # Expand to full dim by concatenating: [f0, f1, ..., f0, f1, ...]
        c, s = freqs.cos(), freqs.sin()
        cos_freq = torch.cat([c, c], dim=-1) * scale
        sin_freq = torch.cat([s, s], dim=-1) * scale
        t_transformed = (t_middle * cos_freq) + (
            _rotate_half_contiguous(t_middle) * sin_freq
        )

    if start_index == 0 and end_index == t.shape[-1]:
        return t_transformed.type(dtype)

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)
    return out.type(dtype)
