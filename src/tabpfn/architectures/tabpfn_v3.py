# ruff: noqa: PLR0912, C901
"""TabPFN v3 architecture.


Shape suffix convention:

B: batch size
R: total rows (train + test)
Ri: input rows, could be either train + test or test.
Rj: Chunked rows (<= R)
N: train rows
M: test rows
C: total columns
Cj: Chunked columns (<= C)
E: embedding dimension
T: Target dim (e.g. number of classes).
Cl: number of CLS tokens

D: head dimension
H: num heads
S: sequence length
"""

from __future__ import annotations

import dataclasses
import logging as _logging
import math
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import numpy as np
import pydantic
import torch
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    PerformanceOptions,
)
from tabpfn.architectures.kv_cache import KVCache, KVCacheEntry
from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler

if TYPE_CHECKING:
    from tabpfn.constants import TaskType


_logger = _logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass
class TabPFNV3Config(ArchitectureConfig):
    """Configuration for the single-file TabPFN v3 architecture."""

    name: str = "TabPFN-v3"

    # ---- Distribution embedder (per-column induced self-attention) ----
    embed_dim: int = 128
    """Base embedding dimension used throughout the model."""

    dist_embed_num_blocks: int = 3
    """Number of induced-self-attention blocks in the distribution embedder."""

    dist_embed_num_heads: int = 8
    """Number of attention heads in the distribution embedder."""

    dist_embed_num_inducing_points: int = 128
    """Number of inducing points in the distribution embedder."""

    feature_group_size: int = 3
    """Number of features per circular-shift group in the distribution embedder."""

    # ---- Feature aggregation (cross-feature interaction via CLS tokens) ----
    feat_agg_num_blocks: int = 3
    """Number of transformer blocks in the feature aggregation stage."""

    feat_agg_num_heads: int = 8
    """Number of attention heads in the feature aggregation stage."""

    feat_agg_num_cls_tokens: int = 4
    """Number of CLS tokens used to aggregate per-row feature information."""

    feat_agg_rope_base: float = 100_000
    """RoPE base in the feature aggregation transformer."""

    use_rope: bool = True
    """If True, use RoPE in the attention layers."""

    # ---- ICL transformer ----
    nlayers: int = 24
    """Number of transformer blocks in the ICL stage."""

    icl_num_heads: int = 8
    """Number of attention heads in the ICL stage."""

    icl_num_kv_heads: int | None = None
    """GQA: number of KV heads in the ICL stage. None = standard MHA.
    Must divide icl_num_heads."""

    icl_num_kv_heads_test: int | None = None
    """Number of KV heads used by test rows in the ICL stage.
    None = same as train rows (i.e. icl_num_kv_heads / standard MHA).
    Any value that divides icl_num_heads is valid (1 = MQA, other = GQA)."""

    # ---- Output decoder (many-class for multiclass, MLP for regression) ----
    decoder_head_dim: int = 64
    """Head dimension for the many-class decoder attention."""

    decoder_num_heads: int = 6
    """Number of attention heads for the many-class decoder."""

    decoder_use_softmax_scaling: bool = False
    """If True, apply softmax scaling in the many-class decoder."""

    # ---- Shared ----
    ff_factor: int = 2
    """Feed-forward expansion factor used throughout the model."""

    dropout: float = 0.0
    """Dropout probability (currently unused, kept for config compatibility)."""

    softmax_scaling_mlp_hidden_dim: int = 64
    """Number of hidden units in the MLPs for the SoftmaxScalingMLP layer."""

    # ---- Norm ----
    layernorm_elementwise_affine: bool = True
    """Whether the normalization layers use learnable affine parameters."""

    use_nan_indicators: bool = True
    """If True, concatenate NaN/Inf indicator features to the cell values before
    embedding, matching the TabPFN v2.5 preprocessing. Doubles the input size
    of the cell embedder."""

    # ---- Memory-efficient inference ----
    inference_row_chunk_size: int = 2048
    """Max rows per Stage 0-2 chunk during inference."""

    inference_col_chunk_size: int = 4
    """Max output groups per chunk for inducing hidden state computation."""

    def __post_init__(self) -> None:
        """Validate config constraints."""
        if self.icl_num_kv_heads is not None and (
            self.icl_num_heads % self.icl_num_kv_heads != 0
        ):
            raise ValueError(
                f"icl_num_heads ({self.icl_num_heads}) must be divisible by "
                f"icl_num_kv_heads ({self.icl_num_kv_heads})"
            )
        if self.icl_num_kv_heads_test is not None:
            if self.icl_num_heads % self.icl_num_kv_heads_test != 0:
                raise ValueError(
                    f"icl_num_heads ({self.icl_num_heads}) must be divisible by "
                    f"icl_num_kv_heads_test ({self.icl_num_kv_heads_test})"
                )
            effective_kv = (
                self.icl_num_kv_heads
                if self.icl_num_kv_heads is not None
                else self.icl_num_heads
            )
            if self.icl_num_kv_heads_test > effective_kv:
                raise ValueError(
                    f"icl_num_kv_heads_test ({self.icl_num_kv_heads_test}) must be "
                    f"<= the number of train KV heads ({effective_kv})"
                )


# ---------------------------------------------------------------------------
# TabPFN v3 KV cache
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TabPFNV3Cache:
    """Top-level cache container for TabPFN v3 explicit KV cache.

    Stores everything needed to skip stages 0-2 for train rows and reuse
    cached K/V in the ICL transformer.

    Attributes:
        icl_cache: Per-layer KV cache for the ICL transformer blocks.
        train_embeddings: Post-ICL, post-norm train embeddings of shape
            ``(B, N_train, D)``. Needed by the multiclass decoder.
        train_shape: ``(batch_size, num_train)`` for validation.
        scaler_cache: Fitted standard-scaler statistics (``mean``, ``std``).
            Allows standardising test-only data without train rows present.
        inducing_hidden: Per-block inducing hidden states from the
            distribution embedder, each of shape ``(B*C_out, n_ind, E)``.
            Allows running ``cross_attn_block2`` on test rows without
            recomputing ``cross_attn_block1`` from train rows.
    """

    icl_cache: KVCache = dataclasses.field(default_factory=KVCache)
    train_embeddings: torch.Tensor | None = None
    train_shape: tuple[int, int] = (0, 0)
    scaler_cache: dict[str, torch.Tensor] | None = None
    inducing_hidden: list[torch.Tensor] | None = None

    def is_empty(self) -> bool:
        """Check if the cache is empty."""
        return not self.icl_cache.is_populated()

    def to(self, device: torch.device | str) -> TabPFNV3Cache:
        """Move all cached tensors to the given device."""
        return TabPFNV3Cache(
            icl_cache=self.icl_cache.to(device),
            train_embeddings=(
                self.train_embeddings.to(device)
                if self.train_embeddings is not None
                else None
            ),
            train_shape=self.train_shape,
            scaler_cache=(
                {k: v.to(device) for k, v in self.scaler_cache.items()}
                if self.scaler_cache is not None
                else None
            ),
            inducing_hidden=(
                [h.to(device) for h in self.inducing_hidden]
                if self.inducing_hidden is not None
                else None
            ),
        )

    def cache_size_mb(self) -> int:
        """Return the memory occupied by cached tensors in MB."""
        total = 0
        for entry in self.icl_cache.kv.values():
            if entry.key is not None:
                total += entry.key.numel() * entry.key.element_size()
            if entry.value is not None:
                total += entry.value.numel() * entry.value.element_size()
        if self.train_embeddings is not None:
            total += (
                self.train_embeddings.numel() * self.train_embeddings.element_size()
            )
        if self.scaler_cache is not None:
            for v in self.scaler_cache.values():
                total += v.numel() * v.element_size()
        if self.inducing_hidden is not None:
            for h in self.inducing_hidden:
                total += h.numel() * h.element_size()
        return total // (1024 * 1024)


_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]
_SDPA_BACKENDS_CPU = [*_SDPA_BACKENDS, SDPBackend.MATH]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE) — compile-friendly, no einops
# ---------------------------------------------------------------------------
# We don't cache cos/sin, since this blocks torch.compile.


def apply_rope(
    t: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply rotary positional embeddings to ``t`` along seq_dim=-2.

    All intermediate math is done in ``inv_freq.dtype`` (fp32 by
    construction) and the result is cast back to ``t.dtype``.

    Args:
        t: Tensor of shape ``(..., S, D)`` where the head dim ``D`` is
            even. The sequence dim is the second-to-last axis.
        inv_freq: ``(D // 2,)`` inverse frequencies (typically
            ``1 / theta ** (2i / D)``).
        interleaved: When ``True``, rotates dimension pairs
            ``(0, 1), (2, 3), …`` (LLaMA/HF interleaved layout). When
            ``False`` (default), splits the last dim into two contiguous
            halves and rotates them against each other.
    """
    dtype = t.dtype
    seq_len = t.shape[-2]
    positions = torch.arange(seq_len, device=t.device, dtype=inv_freq.dtype)
    freqs = positions[:, None] * inv_freq[None, :]  # (S, D/2)
    cos = freqs.cos()
    sin = freqs.sin()
    if interleaved:
        cos = cos.repeat_interleave(2, dim=-1)  # (S, D)
        sin = sin.repeat_interleave(2, dim=-1)
        t_even = t[..., 0::2]
        t_odd = t[..., 1::2]
        # stack → (..., D/2, 2) rows (-t_odd, t_even); flatten → (-t1, t0, -t3, t2, …)
        t_rotated = torch.stack((-t_odd, t_even), dim=-1).flatten(-2)
    else:
        cos = torch.cat((cos, cos), dim=-1)  # (S, D)
        sin = torch.cat((sin, sin), dim=-1)
        half = t.shape[-1] // 2
        t_rotated = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
    return (t * cos + t_rotated * sin).to(dtype)


class RotaryEmbedding(nn.Module):
    """Compile-friendly rotary positional embedding.

    Args:
        dim: Per-head rotation dimension. Must be even.
        theta: Base for the rotary frequencies (10_000 in the original
            paper, 100_000 in our configs).
        interleaved: See :func:`apply_rope`.
    """

    def __init__(
        self,
        dim: int,
        *,
        theta: float = 10_000.0,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, f"RoPE head dim must be even, got {dim}"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        # Store as a non-learnable nn.Parameter (not a buffer) to match the
        # upstream RotaryEmbedding which has ``self.freqs = nn.Parameter(...,
        # requires_grad=False)``. This preserves the parameter count seen by
        # the optimizer, avoiding subtle numerical drift in training due to
        # Adam state ordering changes.
        self.freqs = nn.Parameter(inv_freq, requires_grad=False)
        self.interleaved = interleaved

    def rotate_queries_or_keys(self, t_BHSD: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to t_BSHD."""
        return apply_rope(t_BHSD, self.freqs, interleaved=self.interleaved)


class _DtypeMatchingRMSNorm(nn.RMSNorm):
    """RMSNorm that casts weight to match the input dtype.

    Fused CUDA kernels require matching dtypes; casting the tiny weight/bias per-call
    avoids unfused fallbacks under autocast.
    """

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and self.weight.dtype != input.dtype:
            return F.rms_norm(
                input,
                self.normalized_shape,
                self.weight.to(input.dtype),
                self.eps,
            )
        return super().forward(input)


class ManyClassDecoder(nn.Module):
    """Attention-based retrieval decoder for many-class classification.

    Computes weighted (by attention score) average over one-hot encoded
    train targets, then takes the log to obtain logits.  Supports arbitrary
    class counts by chunking the value (one-hot) dimension into head_dim-sized
    pieces and folding them into the batch dimension for a single flash-attention
    call.
    """

    def __init__(
        self,
        max_num_classes: int,
        input_size: int,
        head_dim: int = 64,
        num_heads: int = 6,
        softmax_scaling_layer: nn.Module | None = None,
    ):
        """Init."""
        super().__init__()
        self.max_num_classes = max_num_classes
        self.input_size = input_size
        self.attention_size = head_dim * num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.q_projection = nn.Linear(self.input_size, self.attention_size)
        self.k_projection = nn.Linear(self.input_size, self.attention_size)
        self.softmax_scaling_layer = softmax_scaling_layer

    @override
    def forward(
        self,
        train_embeddings: torch.Tensor,  # (B, N, E)
        test_embeddings: torch.Tensor,  # (B, M, E)
        targets: torch.Tensor,  # (B, N) - class indices
    ) -> torch.Tensor:
        """Perform a forward pass."""
        B, M, _ = test_embeddings.shape
        q_BME = self.q_projection(test_embeddings)
        k_BNE = self.k_projection(train_embeddings)

        if M == 0:
            # OOM checks at training start run with no test rows. Flash attention
            # rejects a query sequence of length 0, so we return early.
            # Both dummy terms keep the output in the computation graph so that
            # gradients flow through both projections during memory estimation.
            empty = test_embeddings.new_empty((0, B, self.max_num_classes))
            return empty + (q_BME.sum() + k_BNE.sum()) * 0.0

        one_hot_targets_BNT = (
            F.one_hot(targets.long(), num_classes=self.max_num_classes)
            .to(dtype=q_BME.dtype)
            .contiguous()
        )

        q_BMHD = q_BME.view(B, M, self.num_heads, self.head_dim).contiguous()
        k_BNHD = k_BNE.view(B, -1, self.num_heads, self.head_dim).contiguous()
        one_hot_targets_BNHT = (
            one_hot_targets_BNT.unsqueeze(2)
            .expand(-1, -1, self.num_heads, -1)
            .contiguous()
        )
        test_output_BMHT = _chunked_class_attention(
            q_BMHD,
            k_BNHD,
            one_hot_targets_BNHT,
            softmax_scaling_layer=self.softmax_scaling_layer,
        )
        test_output_BMT = test_output_BMHT.mean(2)  # average over heads

        test_output_MBT = test_output_BMT.transpose(0, 1)
        # convert to logits:
        return torch.log(torch.clamp(test_output_MBT, min=1e-5) + 3e-5)


def _chunked_class_attention(
    q_BSHD: torch.Tensor,
    k_BJHD: torch.Tensor,
    v_BJHT: torch.Tensor,
    softmax_scaling_layer: nn.Module | None = None,
) -> torch.Tensor:
    """Run retrieval attention where the value dimension C may exceed head_dim D.

    Splits V into head_dim-sized chunks along the class axis, folds the chunk
    index into the batch dimension, and dispatches a single flash-attention call.
    This avoids the O(N*M) memory cost of the math backend for any class count.

    Args:
        q_BSHD: Query tensor of shape (B, S, H, D) for test points.
        k_BJHD: Key tensor of shape (B, J, H, D) for train points.
        v_BJHT: Value tensor of shape (B, J, H, T) holding one-hot class
            encodings; T may be larger than D.
        softmax_scaling_layer: Optional scaling module to scale queries before SDPA.

    Returns:
        Output tensor of shape (B, S, H, T).
    """
    B, S, H, D = q_BSHD.shape
    T = v_BJHT.shape[-1]
    num_chunks = math.ceil(T / D)

    # Pad V to a multiple of D along the class axis
    pad = num_chunks * D - T
    if pad > 0:
        v_BJHT = F.pad(v_BJHT, (0, pad))

    # Fold chunk index into batch dimension
    J = v_BJHT.shape[1]
    v_folded = (
        v_BJHT.reshape(B, J, H, num_chunks, D)
        .permute(0, 3, 1, 2, 4)
        .reshape(B * num_chunks, J, H, D)
        .contiguous()
    )
    q_folded = (
        q_BSHD.unsqueeze(1)
        .expand(-1, num_chunks, -1, -1, -1)
        .reshape(B * num_chunks, S, H, D)
        .contiguous()
    )
    k_folded = (
        k_BJHD.unsqueeze(1)
        .expand(-1, num_chunks, -1, -1, -1)
        .reshape(B * num_chunks, J, H, D)
        .contiguous()
    )

    # Single flash-attention call across all chunks
    out_folded = _batched_scaled_dot_product_attention(
        q_folded, k_folded, v_folded, softmax_scaling_layer=softmax_scaling_layer
    )

    # Unfold and trim padding: (B*K, S, H, D) -> (B, S, H, T)
    return (
        out_folded.reshape(B, num_chunks, S, H, D)
        .permute(0, 2, 3, 1, 4)
        .reshape(B, S, H, num_chunks * D)[..., :T]
    )


class TrainableOrthogonalEmbedding(nn.Module):
    """Trainable class embeddings initialized with orthogonal initialization."""

    def __init__(self, num_classes: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self._init()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map integer labels (B, T) -> embeddings (B, T, embed_dim)."""
        return self.embedding(x.long())

    def _init(self) -> None:
        """Initialize embedding weight rows orthogonally in-place.

        The first ``min(num_classes, embed_dim)`` rows are set to orthonormal
        vectors via QR decomposition; remaining rows (when ``num_classes >
        embed_dim``) are unit-normalized random vectors.
        """
        weight = self.embedding.weight
        num_classes, embed_dim = weight.shape
        k = min(num_classes, embed_dim)
        q, _ = torch.linalg.qr(torch.randn(embed_dim, k))
        ortho_rows = q.T  # (k, embed_dim)
        with torch.no_grad():
            weight[:k].copy_(ortho_rows)
            if num_classes > embed_dim:
                extra = torch.randn(num_classes - k, embed_dim)
                extra = extra / extra.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                weight[k:].copy_(extra)


class MLP(nn.Sequential):
    """Two-layer GELU feed-forward network with zero-initialized output."""

    def __init__(
        self,
        emsize: int,
        dim_feedforward: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        linear2 = nn.Linear(dim_feedforward, emsize, bias=False, **kw)
        nn.init.zeros_(linear2.weight)
        super().__init__(
            nn.Linear(emsize, dim_feedforward, bias=False, **kw),
            nn.GELU(),
            linear2,
        )


class SoftmaxScalingMLP(nn.Module):
    """Query-aware attention scaling using MLPs to compute scaling factors.

    Applies scaling to queries:

    q_scaled = q * base_mlp(logn) * (1 + tanh(query_mlp(q))),

    where the base MLP learns length-dependent scaling and the query MLP
    learns query-dependent modulation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        n_hidden: int = 64,
    ):
        """Initializes the SoftmaxScalingMLP module.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension of each attention head.
            n_hidden: Number of hidden units in the MLPs.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        base_out_dim = num_heads * head_dim
        query_out_dim = head_dim

        self.base_mlp = nn.Sequential(
            nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, base_out_dim)
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, query_out_dim)
        )
        # ensures initial modulation is zero
        nn.init.zeros_(self.query_mlp[-1].weight)  # type: ignore
        nn.init.zeros_(self.query_mlp[-1].bias)  # type: ignore

    @override
    def forward(self, q_BHSD: torch.Tensor, n: int) -> torch.Tensor:
        """Applies scalable attention scaling to queries.

        Args:
            q_BHSD: Query tensor after projection, shape `[B, H, S, D]`.
                B: Batch size.
                H: Number of heads.
                S: Sequence length.
                D: Head dimension.
            n: Number of elements for log-n scaling.

        Returns:
            Scaled query tensor, same shape as `q_BHSD`.
        """
        logn_11 = _safe_log_seqlen(n, q_BHSD.device, q_BHSD.dtype).reshape(1, 1)
        base_scales = self.base_mlp(logn_11).view(1, self.num_heads, 1, self.head_dim)
        modulation = 1 + torch.tanh(self.query_mlp(q_BHSD))
        scales = base_scales * modulation
        return q_BHSD * scales


def _batched_scaled_dot_product_attention(
    q_BSHD: torch.Tensor,
    k_BSJD: torch.Tensor,
    v_BSJD: torch.Tensor,
    softmax_scaling_layer: nn.Module | None = None,
    _backends_override: list[SDPBackend] | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention chunked over the batch dimension.

    For large batch sizes (> CUDA max-grid), we split into sub-batches.
    Softmax scaling is applied to queries before the SDPA call when provided.
    """
    q_BHSD = q_BSHD.permute(0, 2, 1, 3)
    k_BJSD = k_BSJD.permute(0, 2, 1, 3)
    v_BJSD = v_BSJD.permute(0, 2, 1, 3)

    if softmax_scaling_layer is not None:
        src_len = k_BJSD.shape[-2]
        q_BHSD = softmax_scaling_layer(q_BHSD, src_len)

    num_q_heads = q_BHSD.shape[-3]
    num_kv_heads = k_BJSD.shape[-3]
    dtype_supports_gqa = q_BHSD.dtype in {torch.float16, torch.bfloat16}
    if num_q_heads == num_kv_heads:
        keys = k_BJSD
        values = v_BJSD
        enable_gqa = {}
    elif gqa_is_supported() and dtype_supports_gqa:
        keys = k_BJSD
        values = v_BJSD
        enable_gqa = {"enable_gqa": True}
    else:
        repeat = num_q_heads // num_kv_heads
        keys = k_BJSD.repeat_interleave(repeat, dim=-3)
        values = v_BJSD.repeat_interleave(repeat, dim=-3)
        enable_gqa = {}

    if _backends_override is not None:
        backends = _backends_override
    else:
        backends = (
            _SDPA_BACKENDS_CPU if not torch.cuda.is_available() else _SDPA_BACKENDS
        )

    num_parallel_calls = q_BHSD.shape[:2].numel()
    torch._check(num_parallel_calls >= 1)  # These checks help torch.compile.
    torch._check(q_BHSD.shape[0] >= 1)
    CUDA_MAX_GRID = 65536
    num_iterations = (num_parallel_calls + CUDA_MAX_GRID - 1) // CUDA_MAX_GRID
    sub_batch = (q_BHSD.shape[0] + num_iterations - 1) // num_iterations

    with sdpa_kernel(backends=backends):
        outputs = []
        for i in range(num_iterations):
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_BHSD[i * sub_batch : (i + 1) * sub_batch],
                    keys[i * sub_batch : (i + 1) * sub_batch],
                    values[i * sub_batch : (i + 1) * sub_batch],
                    attn_mask=None,
                    **enable_gqa,
                )
            )
    output_BHSD = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output_BHSD.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention (optionally with RoPE)."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        kw = {"device": device, "dtype": dtype, "bias": False}

        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)

        torch.nn.init.xavier_uniform_(self.q_projection.weight)
        torch.nn.init.xavier_uniform_(self.k_projection.weight)
        torch.nn.init.xavier_uniform_(self.v_projection.weight)
        torch.nn.init.zeros_(self.out_projection.weight)

    @override
    def forward(
        self,
        x_BSE: torch.Tensor,
        rope: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        B, S, _ = x_BSE.shape
        q = self.q_projection(x_BSE).view(B, S, -1, self.head_dim)
        k = self.k_projection(x_BSE).view(B, S, -1, self.head_dim)
        v = self.v_projection(x_BSE).view(B, S, -1, self.head_dim)

        if rope is not None:
            q = rope.rotate_queries_or_keys(q.transpose(1, 2)).transpose(1, 2)
            k = rope.rotate_queries_or_keys(k.transpose(1, 2)).transpose(1, 2)

        out = _batched_scaled_dot_product_attention(q, k, v).reshape(
            B, S, self.head_dim * self.num_heads
        )
        return self.out_projection(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention (query attends to key/value sequence)."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling_layer: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scaling_layer = softmax_scaling_layer
        kw = {"device": device, "dtype": dtype, "bias": False}

        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)

        torch.nn.init.xavier_uniform_(self.q_projection.weight)
        torch.nn.init.xavier_uniform_(self.k_projection.weight)
        torch.nn.init.xavier_uniform_(self.v_projection.weight)
        torch.nn.init.zeros_(self.out_projection.weight)

    @override
    def forward(
        self,
        x_for_query_BQE: torch.Tensor,
        x_for_key_and_value_BVE: torch.Tensor,
    ) -> torch.Tensor:
        B, Q, _ = x_for_query_BQE.shape
        _, V, _ = x_for_key_and_value_BVE.shape
        q = self.q_projection(x_for_query_BQE).view(B, Q, -1, self.head_dim)
        k = self.k_projection(x_for_key_and_value_BVE).view(B, V, -1, self.head_dim)
        v = self.v_projection(x_for_key_and_value_BVE).view(B, V, -1, self.head_dim)

        out = _batched_scaled_dot_product_attention(
            q, k, v, softmax_scaling_layer=self.softmax_scaling_layer
        )

        return self.out_projection(out.reshape(B, Q, self.head_dim * self.num_heads))


class ICLAttention(nn.Module):
    """ICL attention: all rows attend to train-only keys/values.

    In v2, the ICL transformer restricts keys/values to training rows so that
    test rows cannot attend to each other or to future labels.

    When ``num_kv_heads_test`` is set, test rows use fewer KV heads than train
    rows (GQA / MQA for the test partition only), reducing the KV-cache at
    inference time.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling_layer: nn.Module | None = None,
        num_kv_heads: int | None = None,
        num_kv_heads_test: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scaling_layer = softmax_scaling_layer
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads_test = num_kv_heads_test
        kw = {"device": device, "dtype": dtype, "bias": False}

        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, **kw)

        torch.nn.init.xavier_uniform_(self.q_projection.weight)
        torch.nn.init.zeros_(self.out_projection.weight)

        if num_kv_heads is not None:
            # GQA: smaller K/V projections
            kv_dim = num_kv_heads * head_dim
            self.k_projection = nn.Linear(embedding_size, kv_dim, **kw)
            self.v_projection = nn.Linear(embedding_size, kv_dim, **kw)
        else:
            self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
            self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, **kw)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.xavier_uniform_(self.v_projection.weight)

    @override
    def forward(
        self,
        x_BRE: torch.Tensor,
        single_eval_pos: int,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        """Self-attention where k/v are restricted to train rows.

        Args:
            x_BRE: (B, R, E) all rows (train + test), or test-only when
                ``cached_kv`` is provided.
            single_eval_pos: Number of training rows; positions after this index
                are test rows. Should be 0 when using ``cached_kv``.
            cached_kv: Pre-computed K/V from a previous forward pass. When
                provided, K/V projection is skipped and these values are used
                directly.
            return_kv: If True, also return the computed K/V as a
                :class:`KVCacheEntry`.

        Returns:
            ``(output, kv_entry)`` where ``kv_entry`` is ``None`` unless
            ``return_kv`` is True.
        """
        B, R, _ = x_BRE.shape

        q = self.q_projection(x_BRE).view(B, R, self.num_heads, self.head_dim)

        if cached_kv is not None:
            # Use pre-computed K/V from cache (test-only path)
            k = cached_kv.key
            v = cached_kv.value
            assert k is not None, "cached key is None"
            assert v is not None, "cached value is None"
            # Match dtype in case of autocast (e.g. fp32 cache under fp16)
            # TODO: Add kv (de-)quantization here
            if k.dtype != q.dtype:
                k = k.to(q.dtype)
                v = v.to(q.dtype)
            # The cache already stores only the test KV heads (sliced at
            # cache-build time), so no slicing is needed here.
            if self.num_kv_heads_test is not None:
                nh_test_heads = self.num_kv_heads_test
                assert k.shape[2] == nh_test_heads, "cached key has wrong num heads"
                assert v.shape[2] == nh_test_heads, "cached value has wrong num heads"
            out = _batched_scaled_dot_product_attention(
                q, k, v, softmax_scaling_layer=self.softmax_scaling_layer
            )
        else:
            N = R if single_eval_pos is None else single_eval_pos
            x_train = x_BRE[:, :N]
            k = self.k_projection(x_train).view(B, N, self.num_kv_heads, self.head_dim)
            v = self.v_projection(x_train).view(B, N, self.num_kv_heads, self.head_dim)

            if (
                self.num_kv_heads_test is not None
                and single_eval_pos is not None
                and N < R
            ):
                # Train rows: full KV heads
                out_train = _batched_scaled_dot_product_attention(
                    q[:, :N], k, v, softmax_scaling_layer=self.softmax_scaling_layer
                )
                # Test rows: fewer KV heads (GQA / MQA)
                nh_test_heads = self.num_kv_heads_test
                out_test = _batched_scaled_dot_product_attention(
                    q[:, N:],
                    k[:, :, :nh_test_heads],
                    v[:, :, :nh_test_heads],
                    softmax_scaling_layer=self.softmax_scaling_layer,
                )
                out = torch.cat([out_train, out_test], dim=1)
            else:
                out = _batched_scaled_dot_product_attention(
                    q, k, v, softmax_scaling_layer=self.softmax_scaling_layer
                )

        result = self.out_projection(out.reshape(B, R, self.head_dim * self.num_heads))

        kv_entry: KVCacheEntry | None = None
        if return_kv:
            # Only cache the KV heads used for test<-train attention to save
            # memory. When num_kv_heads_test is set, test rows use fewer heads.
            k_cache, v_cache = k, v
            if self.num_kv_heads_test is not None:
                nh_test_heads = self.num_kv_heads_test
                k_cache = k_cache[:, :, :nh_test_heads]
                v_cache = v_cache[:, :, :nh_test_heads]
            kv_entry = KVCacheEntry(key=k_cache.detach(), value=v_cache.detach())
        return result, kv_entry


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with pre-norm and MLP."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        assert emsize % nhead == 0
        kw = {"device": device, "dtype": dtype}

        self.attn = CrossAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            softmax_scaling_layer=softmax_scaling_layer,
            **kw,
        )
        self.mlp = MLP(emsize, dim_feedforward, **kw)
        self.layernorm_q = norm_factory(emsize)
        self.layernorm_kv = norm_factory(emsize)
        self.layernorm2 = norm_factory(emsize)

    @override
    def forward(
        self,
        x_BQE: torch.Tensor,
        context_BVE: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.layernorm_q(x_BQE),
            self.layernorm_kv(context_BVE),
        )
        x_BQE = x_BQE + attn_out
        mlp_out = self.mlp(self.layernorm2(x_BQE))
        return x_BQE + mlp_out


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block used in ColumnAggregator."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        assert emsize % nhead == 0
        self.attention = Attention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            **kw,
        )
        self.layernorm = norm_factory(emsize)
        self.layernorm_mlp = norm_factory(emsize)
        self.mlp = MLP(emsize, dim_feedforward, **kw)

    @override
    def forward(
        self,
        x_BRCE: torch.Tensor,
        rope: RotaryEmbedding | None = None,
        save_peak_memory_factor: int | None = None,
    ) -> torch.Tensor:
        x_BRCE = chunked_evaluate_maybe_inplace(
            lambda x, rope=None: self.attention(self.layernorm(x), rope=rope),
            x_BRCE,
            save_peak_memory_factor=save_peak_memory_factor,
            residual=True,
            batch_dims=2,
            rope=rope,
        )
        return chunked_evaluate_maybe_inplace(
            lambda x: self.mlp(self.layernorm_mlp(x)),
            x_BRCE,
            save_peak_memory_factor=save_peak_memory_factor,
            residual=True,
            batch_dims=3,
        )

    def forward_cross(
        self,
        query_BRQE: torch.Tensor,
        context_BRCE: torch.Tensor,
        rope: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        """Cross-attention variant: query attends to context.

        Used in ColumnAggregator for the last CLS-readout block.
        """
        B, R, Q, _ = query_BRQE.shape
        _, _, V, E = context_BRCE.shape

        # Fold rows into batch for attention (per-row cross-attn over features)
        norm_q = self.layernorm(query_BRQE)
        q_flat = norm_q.view(B * R, Q, E)
        c_flat = self.layernorm(context_BRCE).view(B * R, V, E)
        q_proj = self.attention.q_projection(q_flat).view(
            B * R, Q, -1, self.attention.head_dim
        )
        k_flat = self.attention.k_projection(c_flat).view(
            B * R, V, -1, self.attention.head_dim
        )
        v_flat = self.attention.v_projection(c_flat).view(
            B * R, V, -1, self.attention.head_dim
        )

        if rope is not None:
            q_proj = rope.rotate_queries_or_keys(q_proj.transpose(1, 2)).transpose(1, 2)
            k_flat = rope.rotate_queries_or_keys(k_flat.transpose(1, 2)).transpose(1, 2)

        attn_out = _batched_scaled_dot_product_attention(q_proj, k_flat, v_flat)
        attn_out = attn_out.reshape(
            B * R, Q, self.attention.head_dim * self.attention.num_heads
        )
        attn_out = self.attention.out_projection(attn_out).view(B, R, Q, E)

        x_out = query_BRQE + attn_out
        mlp_out = self.mlp(self.layernorm_mlp(x_out))
        return x_out + mlp_out


class ICLTransformerBlock(nn.Module):
    """ICL transformer block with train-only keys and optional softmax scaling."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer: nn.Module | None = None,
        num_kv_heads: int | None = None,
        num_kv_heads_test: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        assert emsize % nhead == 0
        self.icl_attention = ICLAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            softmax_scaling_layer=softmax_scaling_layer,
            num_kv_heads=num_kv_heads,
            num_kv_heads_test=num_kv_heads_test,
            **kw,
        )
        self.layernorm = norm_factory(emsize)
        self.layernorm_mlp = norm_factory(emsize)
        self.mlp = MLP(emsize, dim_feedforward, **kw)

    @override
    def forward(
        self,
        x_BRE: torch.Tensor,
        single_eval_pos: int,
        save_peak_memory_factor: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        """Forward pass with optional KV cache support.

        Args:
            x_BRE: (B, R, E) all rows, or test-only when ``cached_kv`` is set.
            single_eval_pos: Number of training rows.
            save_peak_memory_factor: Chunking factor for memory saving.
            cached_kv: Pre-computed K/V for this layer.
            return_kv: If True, also return the K/V cache entry.

        Returns:
            ``(output, kv_entry)`` where ``kv_entry`` is ``None`` unless
            ``return_kv`` is True.
        """
        kv_entry: KVCacheEntry | None = None

        if return_kv:
            # Run attention without chunking so we can capture the KV entry
            attn_out, kv_entry = self.icl_attention(
                self.layernorm(x_BRE),
                single_eval_pos=single_eval_pos,
                return_kv=True,
            )
            x_BRE = x_BRE + attn_out
        elif cached_kv is not None:
            # Use cached KV -- chunking over test batch is fine
            # TODO: Performance test this as it might not be needed.
            def _attn_fn_cached(
                x: torch.Tensor,
                single_eval_pos: int | None = None,
            ) -> torch.Tensor:
                out, _ = self.icl_attention(
                    self.layernorm(x),
                    single_eval_pos=single_eval_pos,
                    cached_kv=cached_kv,
                )
                return out

            x_BRE = chunked_evaluate_maybe_inplace(
                _attn_fn_cached,
                x_BRE,
                save_peak_memory_factor=save_peak_memory_factor,
                residual=True,
                batch_dims=1,
                single_eval_pos=single_eval_pos,
            )
        else:
            # Default path -- no cache
            def _attn_fn(
                x: torch.Tensor,
                single_eval_pos: int | None = None,
            ) -> torch.Tensor:
                out, _ = self.icl_attention(
                    self.layernorm(x),
                    single_eval_pos=single_eval_pos,
                )
                return out

            x_BRE = chunked_evaluate_maybe_inplace(
                _attn_fn,
                x_BRE,
                save_peak_memory_factor=save_peak_memory_factor,
                residual=True,
                batch_dims=1,
                single_eval_pos=single_eval_pos,
            )

        # MLP (always the same regardless of cache mode)
        x_BRE = chunked_evaluate_maybe_inplace(
            lambda x: self.mlp(self.layernorm_mlp(x)),
            x_BRE,
            save_peak_memory_factor=save_peak_memory_factor,
            residual=True,
            batch_dims=2,
        )

        return x_BRE, kv_entry


# ---------------------------------------------------------------------------
# Induced self-attention block (v2 style, no affine output)
# ---------------------------------------------------------------------------


class InducedSelfAttentionBlock(nn.Module):
    """Induced self-attention (SetTransformer-style) for efficient O(n) attention.

    Two-stage mechanism:
    1. Inducing points attend to train rows uses softmax scaling when provided.
    2. All rows attend to the inducing-point hidden states.
    """

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        num_inducing_points: int,
        dim_feedforward: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        block_kw = {
            "emsize": emsize,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "norm_factory": norm_factory,
            **kw,
        }

        self.cross_attn_block1 = CrossAttentionBlock(
            **block_kw, softmax_scaling_layer=softmax_scaling_layer
        )
        self.cross_attn_block2 = CrossAttentionBlock(**block_kw)

        self.num_inducing_points = num_inducing_points
        self.inducing_vectors = nn.Parameter(torch.empty(num_inducing_points, emsize))
        nn.init.trunc_normal_(self.inducing_vectors, std=0.02)

    def _induced_attention(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        cached_hidden: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Induced self-attention with optional hidden-state return.

        When `return_hidden` is True, returns ``(output, hidden_detached)``
        so the caller can cache the inducing hidden states. Here, we opt for
        different output types depending on return_hidden, so that this function
        can be used in `chunked_evaluate_maybe_inplace` without any additional logic.
        """
        if cached_hidden is not None:
            hidden = cached_hidden.to(x_BcRE.dtype)
        else:
            Bc, R, _ = x_BcRE.shape
            N = R if single_eval_pos is None else single_eval_pos
            ind = self.inducing_vectors.unsqueeze(0).expand(Bc, -1, -1)
            hidden = self.cross_attn_block1(ind, x_BcRE[:, :N])
        out = self.cross_attn_block2(x_BcRE, hidden)
        if return_hidden:
            return out, hidden.detach()
        return out

    @override
    def forward(
        self,
        x_BRCE: torch.Tensor,
        single_eval_pos: int | None = None,
        save_peak_memory_factor: int | None = None,
        *,
        cached_hidden: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward with optional inducing hidden-state caching.

        Returns:
            ``(output, hidden)`` where ``hidden`` is ``None`` unless
            ``return_hidden`` is True.
        """
        B, R, C, E = x_BRCE.shape
        x_BCRE = x_BRCE.transpose(1, 2).contiguous()
        x_BcRE = x_BCRE.reshape(B * C, R, E)

        if return_hidden:
            out_BcRE, hidden = self._induced_attention(
                x_BcRE,
                single_eval_pos=single_eval_pos,
                return_hidden=True,
            )
        else:
            out_BcRE = chunked_evaluate_maybe_inplace(
                self._induced_attention,
                x_BcRE,
                save_peak_memory_factor,
                residual=False,
                batch_dims=1,
                single_eval_pos=single_eval_pos,
                cached_hidden=cached_hidden,
            )
            hidden = None

        out_BCRE = out_BcRE.reshape(B, C, R, E)
        return out_BCRE.transpose(1, 2).contiguous(), hidden


# ---------------------------------------------------------------------------
# Feature distribution embedder
# ---------------------------------------------------------------------------


class FeatureDistributionEmbedder(nn.Module):
    """Stack of InducedSelfAttentionBlock layers applied per column."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        num_inducing_points: int,
        dim_feedforward: int,
        num_layers: int,
        norm_factory: Callable[[int], nn.Module],
        softmax_scaling_layer_factory: Callable[[], nn.Module] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            InducedSelfAttentionBlock(
                emsize=emsize,
                nhead=nhead,
                num_inducing_points=num_inducing_points,
                dim_feedforward=dim_feedforward,
                norm_factory=norm_factory,
                softmax_scaling_layer=(
                    softmax_scaling_layer_factory()
                    if softmax_scaling_layer_factory is not None
                    else None
                ),
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        )

    @override
    def forward(
        self,
        x_BRiCE: torch.Tensor,
        num_train_rows: int | None = None,
        save_peak_memory_factor: int | None = None,
        *,
        force_recompute_layer: bool = False,
        cached_hidden: list[torch.Tensor] | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Forward pass through all induced self-attention blocks.

        Returns:
            ``(output, hidden_states)`` where ``hidden_states`` is ``None``
            unless ``return_hidden`` is True.
        """
        hidden_states: list[torch.Tensor] | None = [] if return_hidden else None
        assert not (return_hidden and force_recompute_layer), (
            "return_hidden is incompatible with force_recompute_layer"
        )
        for i, layer in enumerate(self.layers):
            if force_recompute_layer:
                x_BRiCE, _ = torch.utils.checkpoint.checkpoint(  # type: ignore
                    layer,
                    x_BRiCE,
                    num_train_rows,
                    use_reentrant=False,
                    save_peak_memory_factor=save_peak_memory_factor,
                )
            else:
                layer_cached = cached_hidden[i] if cached_hidden is not None else None
                x_BRiCE, h = layer(
                    x_BRiCE,
                    single_eval_pos=num_train_rows,
                    save_peak_memory_factor=save_peak_memory_factor,
                    cached_hidden=layer_cached,
                    return_hidden=return_hidden,
                )
                if hidden_states is not None:
                    hidden_states.append(h)
        return x_BRiCE, hidden_states


# ---------------------------------------------------------------------------
# Cross-feature interaction (Row interaction / v2 RowInteraction)
# ---------------------------------------------------------------------------


class ColumnAggregator(nn.Module):
    """Context-aware cross-feature interaction that aggregates column information.

    CLS tokens are prepended, the sequence passes through transformer blocks,
    and the last block performs CLS-only readout (q=CLS, k/v=all).
    An output normalization is applied before the CLS tokens are returned.
    """

    def __init__(
        self,
        emsize: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_cls_tokens: int,
        *,
        norm_factory: Callable[[int], nn.Module],
        use_rope: bool = True,
        rope_base: float = 100_000,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = emsize
        self.num_cls_tokens = num_cls_tokens
        kw = {"device": device, "dtype": dtype}

        self.blocks = nn.ModuleList(
            TransformerBlock(
                emsize=emsize,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                norm_factory=norm_factory,
                **kw,
            )
            for _ in range(num_layers)
        )
        self.rope = (
            RotaryEmbedding(
                dim=emsize // nhead,
                theta=int(rope_base),
                interleaved=False,
            )
            if use_rope
            else None
        )
        self.cls_tokens = nn.Parameter(torch.empty(num_cls_tokens, emsize))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        # Output norm applied to CLS tokens after the last block (v2 out_ln)
        self.out_ln = norm_factory(emsize)

    @override
    def forward(
        self,
        x_BRiCE: torch.Tensor,
        save_peak_memory_factor: int | None = None,
        force_recompute_layer: bool = False,
    ) -> torch.Tensor:
        """Transform feature embeddings into per-row CLS representations.

        Args:
            x_BRiCE: (B, Ri, C, E)
            save_peak_memory_factor: If set, chunk the evaluation to save memory.
            force_recompute_layer: If True, force gradient checkpointing.

        Returns:
            (B, Ri, num_cls_tokens, E)
        """
        B, Ri, _, E = x_BRiCE.shape
        cls = self.cls_tokens.expand(B, Ri, self.num_cls_tokens, E).to(x_BRiCE.device)
        # Prepend CLS tokens: (B, Ri, num_cls + C, E)
        x = torch.cat((cls, x_BRiCE), dim=2)

        # Run all blocks except the last
        for block in self.blocks[:-1]:
            if force_recompute_layer:
                x = torch.utils.checkpoint.checkpoint(  # type: ignore
                    block, x, self.rope, save_peak_memory_factor
                )
            else:
                x = block(
                    x, rope=self.rope, save_peak_memory_factor=save_peak_memory_factor
                )

        # Last block: CLS tokens as query, full sequence as key/value (v2 readout)
        last_block = cast("TransformerBlock", self.blocks[-1])
        x_full: torch.Tensor = x  # type: ignore[assignment]
        cls_part = x_full[..., : self.num_cls_tokens, :]
        if force_recompute_layer:
            cls_out = torch.utils.checkpoint.checkpoint(  # type: ignore
                last_block.forward_cross, cls_part, x_full, self.rope
            )
        else:
            cls_out = last_block.forward_cross(cls_part, x_full, self.rope)

        del x
        return self.out_ln(cls_out)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class TabPFNV3(Architecture):
    """Single-file TabPFNV3 architecture.

    Pipeline:
    1. Preprocessing: standard scaling + NaN encoding
    2. Feature grouping: circular shifts applied before embedding
    3. Cell embedding: feature_group_size scalar values → embed_dim
    4. Target-aware column embedding: add y_encoder(y_train) to train rows
    5. Feature distribution embedder: InducedSelfAttentionBlock x dist_embed_num_blocks
    6. Feature aggregator with feature interaction: transformer with aggregation tokens
    7. ICL transformer: y_encoder + standard attention (train-keys only) + decoder
    """

    def __init__(
        self,
        *,
        config: TabPFNV3Config,
        task_type: TaskType,
        n_out: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        super().__init__()
        self.ff_factor = config.ff_factor
        self.icl_emsize = config.embed_dim * config.feat_agg_num_cls_tokens
        self.n_out = n_out
        self.task_type: TaskType = task_type
        self.feature_group_size = config.feature_group_size
        self.use_nan_indicators = config.use_nan_indicators
        kw = {"device": device, "dtype": dtype}

        norm_factory = partial(
            _DtypeMatchingRMSNorm,
            elementwise_affine=config.layernorm_elementwise_affine,
            device=device,
            dtype=dtype,
        )

        # ---- Cell embedding (ordinal: grouped raw values → E) ----
        in_features = config.feature_group_size
        if self.use_nan_indicators:
            in_features *= 2
        self.x_embed = nn.Linear(in_features, config.embed_dim, **kw)

        # ---- Target-aware col embedding ----
        if task_type == "multiclass":
            self.col_y_encoder = TrainableOrthogonalEmbedding(
                config.max_num_classes,
                config.embed_dim,
            )
        else:
            self.col_y_encoder = nn.Linear(1, config.embed_dim, **kw)

        # ---- Distribution embedder (SetTransformer per feature column) ----
        self.feature_distribution_embedder = FeatureDistributionEmbedder(
            emsize=config.embed_dim,
            nhead=config.dist_embed_num_heads,
            num_layers=config.dist_embed_num_blocks,
            num_inducing_points=config.dist_embed_num_inducing_points,
            dim_feedforward=config.embed_dim * config.ff_factor,
            norm_factory=norm_factory,
            softmax_scaling_layer_factory=lambda: SoftmaxScalingMLP(
                num_heads=config.dist_embed_num_heads,
                head_dim=config.embed_dim // config.dist_embed_num_heads,
                n_hidden=config.softmax_scaling_mlp_hidden_dim,
            ),
            **kw,
        )

        # ---- Cross-feature interaction (RowInteraction) ----
        self.column_aggregator = ColumnAggregator(
            emsize=config.embed_dim,
            nhead=config.feat_agg_num_heads,
            num_layers=config.feat_agg_num_blocks,
            num_cls_tokens=config.feat_agg_num_cls_tokens,
            dim_feedforward=config.embed_dim * config.ff_factor,
            norm_factory=norm_factory,
            use_rope=config.use_rope,
            rope_base=config.feat_agg_rope_base,
            **kw,
        )

        # ---- ICL target encoder ----
        if task_type == "multiclass":
            self.icl_y_encoder: nn.Module = TrainableOrthogonalEmbedding(
                config.max_num_classes,
                self.icl_emsize,
            )
        else:
            self.icl_y_encoder = nn.Linear(1, self.icl_emsize, **kw)

        # ---- ICL transformer ----
        self.icl_blocks = nn.ModuleList(
            ICLTransformerBlock(
                emsize=self.icl_emsize,
                nhead=config.icl_num_heads,
                dim_feedforward=self.icl_emsize * config.ff_factor,
                norm_factory=norm_factory,
                num_kv_heads=config.icl_num_kv_heads,
                num_kv_heads_test=config.icl_num_kv_heads_test,
                softmax_scaling_layer=SoftmaxScalingMLP(
                    num_heads=config.icl_num_heads,
                    head_dim=self.icl_emsize // config.icl_num_heads,
                    n_hidden=config.softmax_scaling_mlp_hidden_dim,
                ),
                **kw,
            )
            for _ in range(config.nlayers)
        )

        # ---- Output norm + decoder ----
        self.output_norm = norm_factory(self.icl_emsize)
        if task_type == "multiclass":
            decoder_softmax_scaling = (
                SoftmaxScalingMLP(
                    num_heads=config.decoder_num_heads,
                    head_dim=config.decoder_head_dim,
                    n_hidden=config.softmax_scaling_mlp_hidden_dim,
                )
                if config.decoder_use_softmax_scaling
                else None
            )
            self.many_class_decoder = ManyClassDecoder(
                max_num_classes=config.max_num_classes,
                input_size=self.icl_emsize,
                head_dim=config.decoder_head_dim,
                num_heads=config.decoder_num_heads,
                softmax_scaling_layer=decoder_softmax_scaling,
            )
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(self.icl_emsize, self.icl_emsize * config.ff_factor, **kw),
                nn.GELU(),
                nn.Linear(self.icl_emsize * config.ff_factor, n_out, **kw),
            )

        self.register_buffer(
            "regression_borders",
            _spline_based_regression_borders(config.num_buckets),
        )
        self.standard_scaler = TorchStandardScaler()
        self._nan_safe_output = True
        self.ninp = config.embed_dim
        self.inference_row_chunk_size = config.inference_row_chunk_size
        self.inference_col_chunk_size = config.inference_col_chunk_size

    @override
    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
        kv_cache: TabPFNV3Cache | None = None,
        return_kv_cache: bool = False,
        x_is_test_only: bool = False,
        # TODO: test_targets_MB needed because model_loading has a condition
        # on its presence. Clean this up.
        test_targets_MB: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[torch.Tensor | dict[str, torch.Tensor], TabPFNV3Cache | None]
    ):
        """Main forward pass for TabPFN v3.

        When a KV cache is provided, ``x_is_test_only=True`` lets the
        caller pass only the test rows (shape ``(num_test, 1, D)``) instead
        of padding with train-row placeholders. ``y`` still carries the
        train labels — the decoder reads ``y[:num_train]`` for the
        many-class head. Outside the cache path, ``x`` is always the full
        dataset and this flag is ignored.
        """
        del task_type
        del test_targets_MB
        if isinstance(x, dict):
            x = x["main"]
        if isinstance(y, dict):
            y = y["main"]
        if y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)
        if y.dim() == 3 and y.shape[-1] == 1:
            y = y.squeeze(-1)

        if performance_options is None:
            performance_options = self.get_default_performance_options()

        if performance_options.enable_torch_compile:
            # We increase the limit, since we compile a couple of subgraphs for
            # chunking and different batched_sdpa configs.
            torch._dynamo.config.cache_size_limit = max(
                32, torch._dynamo.config.cache_size_limit
            )

        if x_is_test_only and (kv_cache is None or kv_cache.is_empty()):
            raise ValueError(
                "x_is_test_only=True requires kv_cache to be provided; "
                "the non-cache forward needs the full train+test tensor."
            )

        del categorical_inds

        if (
            not self.training
            and self.task_type == "multiclass"
            and (y > self.n_out - 1).any()
        ):
            raise ValueError(
                "Target is out of range. Make sure to use an ordinal encoded target. "
                f"Expected target values between 0 and {self.n_out - 1}, but got "
                f"values greater than {self.n_out - 1}."
            )
        x_RiBC = x
        B = x_RiBC.shape[1]
        num_train = y.shape[0]
        torch._dynamo.mark_dynamic(x_RiBC, index=0)
        torch._dynamo.mark_dynamic(x_RiBC, index=1)
        torch._dynamo.mark_dynamic(x_RiBC, index=2)

        x_BRiClE, inducing_hidden = self._stages_0_to_2(
            x_RiBC,
            y,
            performance_options=performance_options,
            return_inducing_hidden=return_kv_cache,
            kv_cache=kv_cache,
            x_is_test_only=x_is_test_only,
        )

        # ---- Stage 3: ICL ----
        x_BRiD = x_BRiClE.flatten(-2)
        del x_BRiClE

        icl_cache_out: KVCache | None = None  # Populated if return_kv_cache is True.

        if kv_cache is not None and not kv_cache.is_empty():
            # Cache path: no y_icl embedding; use cached K/V pairs
            for layer_idx, block in enumerate(self.icl_blocks):
                x_BRiD, _ = block(
                    x_BRiD,
                    0,
                    performance_options.save_peak_memory_factor,
                    cached_kv=kv_cache.icl_cache.kv[layer_idx],
                )
        else:
            if num_train > 0:
                y_icl = self._prepare_y(y, num_train, B)
                y_icl_emb = self._embed_icl_y(y_icl)
                x_BRiD[:, :num_train] = x_BRiD[:, :num_train] + y_icl_emb

            if return_kv_cache:
                icl_cache_out = KVCache()
                for layer_idx, block in enumerate(self.icl_blocks):
                    x_BRiD, kv_entry = block(
                        x_BRiD,
                        num_train,
                        performance_options.save_peak_memory_factor,
                        return_kv=True,
                    )
                    icl_cache_out.kv[layer_idx] = kv_entry
            else:
                for block in self.icl_blocks:
                    if performance_options.force_recompute_layer:
                        x_BRiD, _ = torch.utils.checkpoint.checkpoint(
                            block,
                            x_BRiD,
                            num_train,
                            use_reentrant=False,
                            save_peak_memory_factor=performance_options.save_peak_memory_factor,
                        )
                    else:
                        x_BRiD, _ = block(
                            x_BRiD,
                            num_train,
                            performance_options.save_peak_memory_factor,
                        )

        x_BRiD = self.output_norm(x_BRiD)

        # ---- Split embeddings --------------------------------------------------
        if kv_cache is not None and not kv_cache.is_empty():
            test_emb = x_BRiD
            train_emb = kv_cache.train_embeddings
        else:
            test_emb = x_BRiD[:, num_train:]
            train_emb = x_BRiD[:, :num_train]

        # ---- Build KV cache output ---------------------------------------------
        built_cache: TabPFNV3Cache | None = None
        if return_kv_cache:
            if kv_cache is not None and not kv_cache.is_empty():
                built_cache = kv_cache  # pass through unchanged
            else:
                scaler_stats = self.standard_scaler.fit(x_RiBC[:num_train])
                built_cache = TabPFNV3Cache(
                    icl_cache=icl_cache_out,
                    train_embeddings=train_emb.detach(),
                    train_shape=(B, num_train),
                    scaler_cache={k: v.detach() for k, v in scaler_stats.items()},
                    inducing_hidden=(
                        [h.detach() for h in inducing_hidden]
                        if inducing_hidden is not None
                        else None
                    ),
                )

        # ---- Decoder -----------------------------------------------------------
        if self.task_type == "multiclass":
            y_BN = y.transpose(0, 1) if y.dim() == 2 else y.unsqueeze(0)
            test_out: torch.Tensor = self.many_class_decoder(
                train_emb,
                test_emb,
                y_BN[:, :num_train],
            )
        else:
            test_out = self.output_projection(test_emb.transpose(0, 1))

        if self._nan_safe_output:
            test_out = torch.nan_to_num(test_out, nan=0.0)

        if only_return_standard_out:
            output = test_out
        else:
            output = {
                "standard": test_out,
                "train_embeddings": train_emb.transpose(0, 1),
                "test_embeddings": test_emb.transpose(0, 1),
            }
        if return_kv_cache:
            return output, built_cache
        return output

    @override
    def get_default_performance_options(self) -> PerformanceOptions:
        options = super().get_default_performance_options()
        return dataclasses.replace(
            options,
            use_chunkwise_inference=True,
        )

    def _prepare_y(
        self,
        y: torch.Tensor,
        num_train: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Prepare y_train for either target-embedding stage.

        Returns:
            Clean y_train of shape (B, train_size), or None if no train rows.
        """
        if num_train == 0:
            raise ValueError("No training rows available for target embedding.")

        y_NB1 = _prepare_targets(y, num_train, batch_size)[:num_train]
        y_NB1 = _impute_target_nan_and_inf(
            y_NB1=y_NB1,
            task_type=self.task_type,
            num_train_rows=num_train,
        )
        return y_NB1.squeeze(-1).transpose(0, 1)  # (B, train_size)

    def _embed_col_y(self, y_BN: torch.Tensor) -> torch.Tensor:
        """Embed y_train for the col stage → (B, T, E)."""
        if self.task_type == "multiclass":
            return self.col_y_encoder(y_BN)
        return self.col_y_encoder(y_BN.unsqueeze(-1))

    def _embed_icl_y(self, y_BN: torch.Tensor) -> torch.Tensor:
        """Embed y_train for the ICL stage → (B, T, D)."""
        if self.task_type == "multiclass":
            return self.icl_y_encoder(y_BN)
        return self.icl_y_encoder(y_BN.unsqueeze(-1))

    def _preprocess_raw(
        self,
        x_RiBC: torch.Tensor,
        num_train: int,
        scaler_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """NaN indicator capture → imputation → standardisation → transpose.

        When *scaler_cache* is provided the scaler is applied without refitting
        (inference mode); otherwise it is fitted on the first *num_train* rows.

        Returns ``(x_BRiC, nan_ind_BRiC)`` both of shape ``(B, Ri, C)``.
        """
        nan_ind_BRiC: torch.Tensor | None = None
        if self.use_nan_indicators:
            # Note: Indicators need to be computed before imputation.
            nan_indicator_RiBC = _generate_nan_and_inf_indicator(x_RiBC)
            nan_ind_BRiC = nan_indicator_RiBC.transpose(0, 1)

        x_RiBC, _ = _impute_nan_and_inf_with_mean(x_RiBC, num_train, scaler_cache)
        if scaler_cache is not None:
            x_RiBC = self.standard_scaler.transform(x_RiBC, fitted_cache=scaler_cache)
        else:
            x_RiBC = self.standard_scaler(x=x_RiBC, num_train_rows=num_train)
        x_BRiC = x_RiBC.transpose(0, 1)

        return x_BRiC, nan_ind_BRiC

    def _group_features(
        self,
        x_BRiC: torch.Tensor,
        nan_ind_BRiC: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the full grouped + indicator-concatenated tensor."""
        size = self.feature_group_size
        x_grouped = [torch.roll(x_BRiC, shifts=-(2**i), dims=2) for i in range(size)]
        x_grouped = torch.stack(x_grouped, dim=-1)
        if nan_ind_BRiC is not None:
            ind_grouped = [
                torch.roll(nan_ind_BRiC, shifts=-(2**i), dims=2) for i in range(size)
            ]
            ind_grouped = torch.stack(ind_grouped, dim=-1)
            x_grouped = torch.cat([x_grouped, ind_grouped], dim=-1)
        return x_grouped

    def _compute_all_inducing_hidden(
        self,
        dist_embedder_layers: nn.ModuleList,
        x_grouped_BRiCG: torch.Tensor,
        num_train: int,
        y_col_emb_BNE: torch.Tensor | None,
        col_chunk_size: int,
        *,
        enable_torch_compile: bool,
    ) -> list[torch.Tensor]:
        """Pre-compute inducing hidden states for every dist-embedder block.

        Processes columns in chunks of *col_chunk_size* to avoid
        materialising ``(B*C_out, N_train, embedding_size)`` all at once.

        Returns one ``(B*C, num_inducing, embedding_size)`` tensor per block.
        """
        num_columns = x_grouped_BRiCG.shape[2]
        num_blocks = len(dist_embedder_layers)
        # I: num inducing vectors.
        # Collect (B, Cj, I, E) per column-chunk, per block
        hidden_per_block: list[list[torch.Tensor]] = [[] for _ in range(num_blocks)]

        x_grouped_train_BNCG = x_grouped_BRiCG[:, :num_train]
        process_col_fn = (
            self._process_col_chunk_compiled
            if enable_torch_compile
            else self._process_col_chunk
        )

        for c0 in range(0, num_columns, col_chunk_size):
            c1 = min(c0 + col_chunk_size, num_columns)
            x_grouped_chunk_BNCjG = x_grouped_train_BNCG[:, :, c0:c1]
            torch._dynamo.mark_dynamic(x_grouped_chunk_BNCjG, index=0)
            torch._dynamo.mark_dynamic(x_grouped_chunk_BNCjG, index=1)
            # Will compile two versions: one with cols dynamic and one with cols
            # static for the fixed chunk size.
            if (c1 - c0) != col_chunk_size:
                torch._dynamo.mark_dynamic(x_grouped_chunk_BNCjG, index=2)

            chunk_outputs_BCjIE = process_col_fn(
                x_grouped_chunk_BNCjG=x_grouped_chunk_BNCjG,
                y_col_emb_BNE=y_col_emb_BNE,
                num_train=num_train,
            )
            for blk_idx, h in enumerate(chunk_outputs_BCjIE):
                hidden_per_block[blk_idx].append(h)

        # Concatenate and flatten column chunks (B * C_out, I, E) per block.
        return [torch.cat(chunks, dim=1).flatten(0, 1) for chunks in hidden_per_block]

    def _preprocess_and_group(
        self,
        rows_RiBC: torch.Tensor,
        y: torch.Tensor,
        num_train: int,
        scaler_cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Preprocess rows, embed y for the col stage, and group features.

        Combines the three pre-chunk-loop steps into one compiled pass.
        Returns the grouped x of shape `(B, Ri, C, G)` tensor and optionally
        the `(B, N_train, E)` y embedding.
        """
        B = rows_RiBC.shape[1]
        x_BRiC, nan_ind_BRiC = self._preprocess_raw(rows_RiBC, num_train, scaler_cache)
        y_col_emb_BNE: torch.Tensor | None = None
        if scaler_cache is None and num_train > 0:
            y_col_BN = self._prepare_y(y, num_train, B)
            y_col_emb_BNE = self._embed_col_y(y_col_BN)

        x_grouped_BRiCG = self._group_features(x_BRiC, nan_ind_BRiC)
        return x_grouped_BRiCG, y_col_emb_BNE

    @torch.compile(dynamic=True)
    def _preprocess_and_group_compiled(
        self,
        rows_RiBC: torch.Tensor,
        y: torch.Tensor,
        num_train: int,
        scaler_cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._preprocess_and_group(rows_RiBC, y, num_train, scaler_cache)

    def _stages_0_to_2(
        self,
        x_RiBC: torch.Tensor,
        y: torch.Tensor,
        *,
        performance_options: PerformanceOptions,
        return_inducing_hidden: bool,
        kv_cache: TabPFNV3Cache | None,
        x_is_test_only: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Stages 0-2: feature embedding, distribution embedding, column aggregation.

        Handles all three computation paths (cache / chunked / full) and returns
        ``(x_BRiClE, inducing_hidden)``.  ``inducing_hidden`` is ``None`` unless
        ``return_inducing_hidden`` is True (full path) or row-chunking is active
        (chunked path, where it is always computed as an intermediate).
        """
        num_train = y.shape[0]
        if performance_options.use_chunkwise_inference:
            row_chunk_size = self.inference_row_chunk_size
            col_chunk_size = self.inference_col_chunk_size
        else:
            row_chunk_size = None
            col_chunk_size = None
        force_recompute_layer = performance_options.force_recompute_layer
        save_peak_memory_factor = performance_options.save_peak_memory_factor

        if kv_cache is not None and not kv_cache.is_empty():
            rows_RiBC = x_RiBC if x_is_test_only else x_RiBC[num_train:]
            scaler_cache = kv_cache.scaler_cache
            precomputed_hidden: list[torch.Tensor] | None = kv_cache.inducing_hidden
            effective_num_train = 0
        else:
            rows_RiBC = x_RiBC
            scaler_cache = None
            precomputed_hidden = None
            effective_num_train = num_train

        # --- Preprocess + y col-embed + feature grouping (single compiled pass). ---
        preprocess_fn = (
            self._preprocess_and_group_compiled
            if performance_options.enable_torch_compile
            else self._preprocess_and_group
        )
        x_grouped_BRiCG, y_col_emb_BNE = preprocess_fn(
            rows_RiBC, y, num_train, scaler_cache
        )
        num_rows, C = x_grouped_BRiCG.shape[1], x_grouped_BRiCG.shape[2]

        # --- Phase 1: compute inducing hidden when chunking w/o a pre-built cache. ---
        use_chunks = row_chunk_size is not None and row_chunk_size < num_rows
        if use_chunks and precomputed_hidden is None:
            eff_col_chunk = col_chunk_size if col_chunk_size is not None else C
            while True:
                try:
                    precomputed_hidden = self._compute_all_inducing_hidden(
                        self.feature_distribution_embedder.layers,
                        x_grouped_BRiCG,
                        num_train,
                        y_col_emb_BNE,
                        eff_col_chunk,
                        enable_torch_compile=performance_options.enable_torch_compile,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    if eff_col_chunk <= 1:
                        raise
                    torch.cuda.empty_cache()
                    eff_col_chunk //= 2
                    _logger.warning(
                        "CUDA OOM during Phase 1; halving col_chunk_size to %d",
                        eff_col_chunk,
                    )

        # --- Shared per-chunk loop: embed → dist-embedder → column-aggregator ---
        # When not chunking, the single iteration covers all rows. force_recompute_layer
        # and return_hidden only apply on the full path (see below).
        is_full_path = not use_chunks and precomputed_hidden is None
        effective_chunk_size = row_chunk_size if use_chunks else num_rows

        process_row_chunk = (
            self._process_row_chunk_compiled
            if performance_options.enable_torch_compile
            else self._process_row_chunk
        )
        while True:
            parts: list[torch.Tensor] = []
            inducing_hidden: list[torch.Tensor] | None = None
            try:
                for row_chunk_start in range(0, num_rows, effective_chunk_size):
                    row_chunk_end = min(
                        row_chunk_start + effective_chunk_size, num_rows
                    )
                    x_grouped_chunk = x_grouped_BRiCG[:, row_chunk_start:row_chunk_end]
                    torch._dynamo.mark_dynamic(x_grouped_chunk, index=0)
                    torch._dynamo.mark_dynamic(x_grouped_chunk, index=2)
                    # Will compile two versions: One with dynamic rows and one with
                    # static rows for the fixed chunk size.
                    if (row_chunk_end - row_chunk_start) != row_chunk_size:
                        torch._dynamo.mark_dynamic(x_grouped_chunk, index=1)

                    row_embedding_chunk, chunk_hidden = process_row_chunk(
                        x_grouped_chunk_BRjCG=x_grouped_chunk,
                        y_col_emb=y_col_emb_BNE,
                        chunk_start=row_chunk_start,
                        chunk_end=row_chunk_end,
                        effective_num_train=effective_num_train,
                        precomputed_hidden=precomputed_hidden,
                        save_peak_memory_factor=save_peak_memory_factor,
                        force_recompute_layer=force_recompute_layer,
                        return_inducing_hidden=return_inducing_hidden,
                        is_full_path=is_full_path,
                    )
                    if chunk_hidden is not None:
                        inducing_hidden = chunk_hidden
                    parts.append(row_embedding_chunk)
                break
            except torch.cuda.OutOfMemoryError:
                if not use_chunks or effective_chunk_size <= 1:
                    raise
                parts.clear()
                torch.cuda.empty_cache()
                effective_chunk_size //= 2
                _logger.warning(
                    "CUDA OOM during Phase 2; halving row_chunk_size to %d",
                    effective_chunk_size,
                )

        if use_chunks:
            inducing_hidden = precomputed_hidden
        x_BRiClE = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return x_BRiClE, inducing_hidden

    def _process_col_chunk(
        self,
        *,
        x_grouped_chunk_BNCjG: torch.Tensor,
        y_col_emb_BNE: torch.Tensor | None,
        num_train: int,
    ) -> list[torch.Tensor]:
        """Compute inducing hidden for one column chunk across all dist-embedder blocks.

        ``x_grouped_chunk_BNCjG`` has shape ``(B, train rows, Cj, G)`` — a slice of the
        pre-grouped tensor with Cj << C, so the chunked op never sees the full ``C``
        dim. Returns one ``(B, Cj, n_ind, embedding_size)`` tensor per block.
        """
        B, _, Cj, _ = x_grouped_chunk_BNCjG.shape

        # Embed this column chunk → (B, Rt, Cj, E)
        x_emb_BNCjE = self.x_embed(x_grouped_chunk_BNCjG)
        E = x_emb_BNCjE.shape[-1]

        # Target-aware y (broadcasts over the Cj columns)
        if y_col_emb_BNE is not None and num_train > 0:
            x_emb_BNCjE = x_emb_BNCjE + y_col_emb_BNE.unsqueeze(2)

        # (B, Rt, Cj, E) → (B*Cj, Rt, E)
        x_flat = x_emb_BNCjE.transpose(1, 2).contiguous().reshape(B * Cj, num_train, E)

        layers = self.feature_distribution_embedder.layers
        num_blocks = len(layers)
        chunk_outputs: list[torch.Tensor] = []
        for blk_idx, blk in enumerate(layers):
            ind = blk.inducing_vectors.unsqueeze(0).expand(B * Cj, -1, -1)
            hidden = blk.cross_attn_block1(ind, x_flat)  # (B*cc, n_ind, E)
            # Reshape for correct batch-column ordering when concatenated
            chunk_outputs.append(hidden.reshape(B, Cj, -1, E))
            # Update train embeddings for next block's Step 1
            if blk_idx < num_blocks - 1:
                x_flat = blk.cross_attn_block2(x_flat, hidden)

        return chunk_outputs

    @torch.compile(dynamic=True)
    def _process_col_chunk_compiled(
        self,
        *,
        x_grouped_chunk_BNCjG: torch.Tensor,
        y_col_emb_BNE: torch.Tensor | None,
        num_train: int,
    ) -> list[torch.Tensor]:
        return self._process_col_chunk(
            x_grouped_chunk_BNCjG=x_grouped_chunk_BNCjG,
            y_col_emb_BNE=y_col_emb_BNE,
            num_train=num_train,
        )

    def _process_row_chunk(
        self,
        x_grouped_chunk_BRjCG: torch.Tensor,
        y_col_emb: torch.Tensor | None,
        chunk_start: int,
        chunk_end: int,
        effective_num_train: int,
        precomputed_hidden: list[torch.Tensor] | None,
        save_peak_memory_factor: int | None,
        *,
        force_recompute_layer: bool,
        return_inducing_hidden: bool,
        is_full_path: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run one row chunk through dist-embedder and column-aggregator.

        `x_grouped_chunk` has shape `(B, row_chunk_range, C, G)` — a slice
        of the pre-grouped tensor.
        Returns `(row_embedding_chunk, chunk_hidden)`. `chunk_hidden` is
        only non-None when `return_inducing_hidden` is True on the full path.
        """
        row_chunk_range = chunk_end - chunk_start
        # Number of train rows in this chunk, not overall dataset.
        num_train_rows = max(0, min(effective_num_train - chunk_start, row_chunk_range))

        x_emb = self.x_embed(x_grouped_chunk_BRjCG)

        if y_col_emb is not None and num_train_rows > 0:
            y_emb = y_col_emb[:, chunk_start : chunk_start + num_train_rows]
            x_emb[:, :num_train_rows] = x_emb[:, :num_train_rows] + y_emb.unsqueeze(2)

        x_emb, chunk_hidden = self.feature_distribution_embedder(
            x_BRiCE=x_emb,
            num_train_rows=num_train_rows,
            cached_hidden=precomputed_hidden,
            save_peak_memory_factor=(save_peak_memory_factor if is_full_path else None),
            force_recompute_layer=force_recompute_layer and is_full_path,
            return_hidden=return_inducing_hidden and is_full_path,
        )
        row_embedding_chunk = self.column_aggregator(
            x_BRiCE=x_emb,
            save_peak_memory_factor=save_peak_memory_factor,
            force_recompute_layer=force_recompute_layer and is_full_path,
        )
        return row_embedding_chunk, chunk_hidden

    @torch.compile(dynamic=True)
    def _process_row_chunk_compiled(
        self,
        x_grouped_chunk_BRjCG: torch.Tensor,
        y_col_emb: torch.Tensor | None,
        chunk_start: int,
        chunk_end: int,
        effective_num_train: int,
        precomputed_hidden: list[torch.Tensor] | None,
        save_peak_memory_factor: int | None,
        *,
        force_recompute_layer: bool,
        return_inducing_hidden: bool,
        is_full_path: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        return self._process_row_chunk(
            x_grouped_chunk_BRjCG=x_grouped_chunk_BRjCG,
            y_col_emb=y_col_emb,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            effective_num_train=effective_num_train,
            precomputed_hidden=precomputed_hidden,
            save_peak_memory_factor=save_peak_memory_factor,
            force_recompute_layer=force_recompute_layer,
            return_inducing_hidden=return_inducing_hidden,
            is_full_path=is_full_path,
        )


# ---------------------------------------------------------------------------
# Module interface
# ---------------------------------------------------------------------------


def parse_config(
    config: dict[str, Any],
) -> tuple[TabPFNV3Config, dict[str, Any]]:
    """Parse the config dict into a TabPFNV3Config, return unused keys."""
    parsed_config = TabPFNV3Config(**config)
    return parsed_config, parsed_config.get_unused_config(config)


def get_architecture(
    config: ArchitectureConfig,
    *,
    cache_trainset_representation: bool = False,
) -> TabPFNV3:
    """Construct TabPFN v3 from the given config."""
    del cache_trainset_representation
    assert isinstance(config, TabPFNV3Config)
    # cache_trainset_representation is accepted for interface compatibility but
    # is a no-op: v3 uses explicit KV cache passing via forward() parameters
    # (kv_cache / return_kv_cache) instead of model-internal caching.
    task_type = "multiclass" if config.max_num_classes >= 2 else "regression"
    n_out = config.max_num_classes if task_type == "multiclass" else config.num_buckets
    return TabPFNV3(
        config=config,
        task_type=task_type,
        n_out=n_out,
    )


# ---------------------------------------------------------------------------
# Private data utilities (unchanged from v1)
# ---------------------------------------------------------------------------


def _prepare_targets(
    y: torch.Tensor,
    num_train_and_test_rows: int,
    batch_size: int,
) -> torch.Tensor:
    """Pad y to match num_train_and_test_rows and ensure shape (Ri, B, 1)."""
    num_train_labels = y.shape[0]
    if num_train_labels > num_train_and_test_rows:
        raise ValueError("No test rows provided.")
    target_RBT = y.view(num_train_labels, 1 if y.ndim == 1 else batch_size, -1)
    return F.pad(
        target_RBT,
        (0, 0, 0, 0, 0, num_train_and_test_rows - num_train_labels),
        value=float("nan"),
    )


def _impute_nan_and_inf_with_mean(
    x: torch.Tensor,
    num_train_rows: int,
    scaler_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Impute the nan and inf with the mean of the feature.

    Returns:
        A tuple of (imputed tensor, is_finite mask).
    """
    is_finite = torch.isfinite(x)
    if num_train_rows == 0 and scaler_cache is None:
        _logging.warning("No training rows or scaler cache provided, imputing with 0.")
    if scaler_cache is not None:
        feature_means = scaler_cache["mean"]
    else:
        x_train = torch.where(is_finite[:num_train_rows], x[:num_train_rows], torch.nan)
        feature_means = torch.nan_to_num(torch.nanmean(x_train, dim=0), 0)
    return torch.where(is_finite, x, feature_means.unsqueeze(0).expand_as(x)), is_finite


def _impute_target_nan_and_inf(
    y_NB1: torch.Tensor,
    task_type: TaskType,
    num_train_rows: int,
) -> torch.Tensor:
    # The class imputation for is performed for backwards compatibility.
    # We impute the mean and then do a ceil() operation.
    # Only apply ceil() to imputed positions to preserve differentiability for
    # original values (e.g. during prompt tuning).
    y_NB1, is_finite = _impute_nan_and_inf_with_mean(y_NB1, num_train_rows)
    if task_type == "regression":
        return y_NB1
    return torch.where(is_finite, y_NB1, y_NB1.ceil())


_NAN_INDICATOR = -2.0
_INFINITY_INDICATOR = 2.0
_NEG_INFINITY_INDICATOR = 4.0


def _generate_nan_and_inf_indicator(x: torch.Tensor) -> torch.Tensor:
    """Generate NaN/Inf indicator features (matches TabPFN v2.5)."""
    return (
        torch.isnan(x) * _NAN_INDICATOR
        + torch.isposinf(x) * _INFINITY_INDICATOR
        + torch.isneginf(x) * _NEG_INFINITY_INDICATOR
    ).to(x.dtype)


def _safe_log_seqlen(
    n: int | torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Compute log(n) safely, avoiding fp16 overflow for large `n`."""
    if isinstance(n, torch.Tensor):
        return n.to(torch.float32).clamp(min=1).log().to(dtype)
    # Materialise `n` via arithmetic on a 0-d tensor rather than
    # `torch.as_tensor(n, ...)`. The latter bakes `n` into the graph as a constant and
    # emits a `n == <value>` guard, triggering a recompile on every new value
    # `one * n` keeps the value symbolic when `n` is a SymInt.
    one = torch.ones((), dtype=torch.float32, device=device)
    return (one * n).clamp(min=1).log().to(dtype)


def _spline_based_regression_borders(num_buckets: int) -> torch.Tensor:
    """Generate hardcoded regression bin borders based on the v2.5 checkpoint.

    Note: Borders are num_buckets + 1!
    Border reference points are derived from tabpfn-v2.5-regressor-v2.5_default.ckpt.
    For visual comparison of the original buckets vs approx, see
    https://www.notion.so/priorlabs/Regression-bucket-approx-3125be1f3b4980f0924bc7bcb6b72bbd


    Returns:
        An array of shape (num_buckets + 1,) containing the bucket borders.
    """
    border_reference_points = [
        (0, -128),
        (5, -16.9),
        (20, -13),
        (100, -9.9),
        (200, -8.47),
        (500, -6.48),
        (1000, -4.40),
    ]
    # The original model had 5000 buckets.
    border_reference_points = (
        border_reference_points
        + [(2500, 0)]
        + [(5000 - x, -y) for x, y in border_reference_points[::-1]]
    )
    x_scale = num_buckets / 5000
    xp = np.array([x for x, _ in border_reference_points]) * x_scale
    yp = np.array([y for _, y in border_reference_points])
    return torch.tensor(
        np.interp(x=np.arange(num_buckets + 1), xp=xp, fp=yp), dtype=torch.float32
    )
