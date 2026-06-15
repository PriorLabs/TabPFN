"""The TabPFN v2 model, implemented as a single file.

This is a single-file reimplementation of v2, replacing the multi-file ``base``
architecture. Like the base architecture, it loads the positional embeddings from disk
for the production seed 42 (see :meth:`TabPFNV2.add_column_embeddings`, so their values
are consistent across devices) and supports a KV cache for fast inference (an explicit
cache passed through :meth:`TabPFNV2.forward`, see :class:`TabPFNV2Cache`).

Although the transformer layers have been renamed/restructured relative to the base
architecture, checkpoints trained with the base architecture can still be loaded: see
:meth:`TabPFNV2.load_state_dict`, which translates the old key names on the fly.

Copyright (c) Prior Labs GmbH 2025.
"""

from __future__ import annotations

import dataclasses
from abc import ABC
from collections.abc import Mapping
from typing import Any, Literal, cast
from typing_extensions import override

import pydantic
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from tabpfn.architectures.encoders.steps._ops import (
    select_features,
    torch_nanmean,
)
from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    PerformanceOptions,
)
from tabpfn.architectures.kv_cache import (
    KVCache,
    KVCacheEntry,
)
from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace
from tabpfn.architectures.shared.column_embeddings import load_column_embeddings
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler

# Indicator values appended to the feature/target encodings to flag the original
# location of NaN / +Inf / -Inf cells (these match the base architecture's encoder).
NAN_INDICATOR = -2.0
INFINITY_INDICATOR = 2.0
NEG_INFINITY_INDICATOR = 4.0

# The feature/target encodings are concatenated with their NaN/Inf indicators before
# the linear projection, doubling the number of features fed into the projection.
ENCODING_SIZE_MULTIPLIER = 2


@pydantic.dataclasses.dataclass
class TabPFNV2Config(ArchitectureConfig):
    """Configuration for the single-file TabPFN v2 architecture."""

    name: str = "TabPFN-v2"
    emsize: int = 192
    nlayers: int = 12
    nhead: int = 6
    """Number of key/value heads to use for per-column-inter-row attention."""

    features_per_group: Literal[1, 2] = 2
    """If > 1, the features will be grouped into groups of this size and the attention
    is across groups."""

    seed: int = 42
    """Seed used to generate the per-column positional embeddings.

    When ``seed == 42`` and the embedding subspace size is 48 (i.e. ``emsize == 192``),
    the first 2000 columns use the pre-generated embeddings loaded from disk, matching
    the production checkpoints. For any other seed the embeddings are generated purely
    from the random generator, which is what the comparison tests rely on."""


class Attention(nn.Module, ABC):
    """Base class for the between-features and between-rows attention layers."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        """Construct a new instance.

        Args:
            embedding_size: The size of the input embedding.
            num_heads: The number of heads to use.
            head_dim: The dimensionality of the query, key and value vectors.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        device_and_dtype_no_bias = {"device": device, "dtype": dtype, "bias": False}

        self.q_projection = nn.Linear(
            embedding_size, head_dim * num_heads, **device_and_dtype_no_bias
        )
        self.k_projection = nn.Linear(
            embedding_size, head_dim * num_heads, **device_and_dtype_no_bias
        )
        self.v_projection = nn.Linear(
            embedding_size, head_dim * num_heads, **device_and_dtype_no_bias
        )

        self.out_projection = nn.Linear(
            head_dim * num_heads, embedding_size, **device_and_dtype_no_bias
        )

        torch.nn.init.xavier_uniform_(self.q_projection.weight)
        torch.nn.init.xavier_uniform_(self.k_projection.weight)
        torch.nn.init.xavier_uniform_(self.v_projection.weight)
        torch.nn.init.zeros_(self.out_projection.weight)


class AlongRowAttention(Attention):
    """Computes the attention between features of a single row.

    This is standard multi-head self-attention, where all features attend to each other.
    """

    @override
    def forward(self, x_BrSE: torch.Tensor) -> torch.Tensor:
        """Forward pass for along-row attention between features.

        Args:
            x_BrSE: The input tensor of shape (Br, C, E), where:
                - Br: Batch size * num rows.
                - C: Number of feature groups.
                - E: Embedding size.
        """
        # H: number of heads.
        # D: head dimension.
        # F: head_dimension * number of heads.
        Br, C, _ = x_BrSE.shape
        q_flat_BrCHF = self.q_projection(x_BrSE)
        k_flat_BrCHF = self.k_projection(x_BrSE)
        v_flat_BrCHF = self.v_projection(x_BrSE)
        q_BrCHD = q_flat_BrCHF.view(Br, C, -1, self.head_dim)
        k_BrCHD = k_flat_BrCHF.view(Br, C, -1, self.head_dim)
        v_BrCHD = v_flat_BrCHF.view(Br, C, -1, self.head_dim)

        output_BrHCD = _batched_scaled_dot_product_attention(q_BrCHD, k_BrCHD, v_BrCHD)
        output_BrCF = output_BrHCD.reshape(Br, C, self.head_dim * self.num_heads)
        return self.out_projection(output_BrCF)


class AlongColumnAttention(Attention):
    """Computes the attention between cells of a single column.

    This is multi-head attention featuring:
    - An implicit mask: The training rows attend to each other and themselves, but not
        the test rows. The test rows only attend to the training rows, and not
        themselves. By not attending to themselves, this avoids the requirement for an
        explicit mask.
    - Multi-query attention for the test rows: All the query heads for the test rows
        attend to the first key-value head. This is a further optimisation that only
        requires including one head in the key-value cache.
    """

    @override
    def forward(
        self,
        x_BcRE: torch.Tensor,
        single_eval_pos: int | None = None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        """Forward pass for attention between cells of a single column.

        Args:
            x_BcRE: The input tensor of shape (Bc, R, E), where:
                - Bc: Batch size * number of columns
                - R: Total rows (test + train), or test-only rows when ``cached_kv``
                  is provided.
                - E: Embedding size.
            single_eval_pos: The position from which on everything is treated as test
                set. If None, no mask is applied and all positions are attended to. If
                given, each query after single_eval_pos will only attend to positions
                before single_eval_pos. Should be 0 when ``cached_kv`` is provided.
            cached_kv: Pre-computed train key/value projections from a previous forward
                pass. When provided, the K/V projections are skipped and every query
                (all rows are test rows) attends to these cached values via the single
                multi-query-attention head.
            return_kv: If True, also return the train key/value projections as a
                :class:`KVCacheEntry`. Only the first key/value head is cached, since
                test rows attend to the first head only (multi-query attention).

        Returns:
            ``(output, kv_entry)`` where ``kv_entry`` is ``None`` unless ``return_kv``
            is True.
        """
        # H: number of heads.
        # D: head dimension.
        # F: head_dimension * number of heads.
        # N: number of train points = single_eval_pos
        # M: number of test points
        Bc, R, _ = x_BcRE.shape

        q_BcRHD = self.q_projection(x_BcRE).view(Bc, R, -1, self.head_dim)

        kv_entry: KVCacheEntry | None = None
        if cached_kv is not None:
            # Cache path: every row is a test row attending to the cached train K/V via
            # the single multi-query-attention head.
            k_Bc1 = cached_kv.key
            v_Bc1 = cached_kv.value
            assert k_Bc1 is not None
            assert v_Bc1 is not None
            if k_Bc1.dtype != q_BcRHD.dtype:
                k_Bc1 = k_Bc1.to(q_BcRHD.dtype)
                v_Bc1 = v_Bc1.to(q_BcRHD.dtype)
            output_BcSHD = _batched_scaled_dot_product_attention(q_BcRHD, k_Bc1, v_Bc1)
        else:
            # If no single_eval_pos was specified, then the whole input is training.
            N = R if single_eval_pos is None else single_eval_pos
            k_BcNHD = self.k_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)
            v_BcNHD = self.v_projection(x_BcRE[:, :N]).view(Bc, N, -1, self.head_dim)

            if single_eval_pos == R:
                output_BcSHD = _batched_scaled_dot_product_attention(
                    q_BcRHD, k_BcNHD, v_BcNHD
                )
            else:
                out_train_BcNHD = _batched_scaled_dot_product_attention(
                    q_BcRHD[:, :N], k_BcNHD, v_BcNHD
                )
                out_test_BcMHD = _batched_scaled_dot_product_attention(
                    q_BcRHD[:, N:], k_BcNHD[:, :, :1], v_BcNHD[:, :, :1]
                )
                output_BcSHD = torch.cat([out_train_BcNHD, out_test_BcMHD], dim=1)

            if return_kv:
                # Only cache the first K/V head, since test rows attend to the first
                # head only (multi-query attention). .contiguous() so the cache owns its
                # storage and the full-projection tensor can be freed.
                kv_entry = KVCacheEntry(
                    key=k_BcNHD[:, :, :1].contiguous().detach(),
                    value=v_BcNHD[:, :, :1].contiguous().detach(),
                )

        output_BcSF = output_BcSHD.reshape(Bc, R, self.head_dim * self.num_heads)
        return self.out_projection(output_BcSF), kv_entry


def _batched_scaled_dot_product_attention(
    q_BSHD: torch.Tensor, k_BSJD: torch.Tensor, v_BSJD: torch.Tensor
) -> torch.Tensor:
    """Execute scaled dot product attention, chunked over the batch dimension.

    Our between-feature attention can have a large batch size.
    E.g., for 2048 datapoints, a batch size of 32, and 6 heads,
    we compute 2048 * 32 * 6 = 393216 attentions.
    This is larger than the maximum launch grid size of cuda and will raise an error.
    Thus, we split the inputs into chunks of the maximum batch size, and execute these
    sequentially.
    """
    q_BHSD = q_BSHD.permute(0, 2, 1, 3)
    k_BJSD = k_BSJD.permute(0, 2, 1, 3)
    v_BJSD = v_BSJD.permute(0, 2, 1, 3)

    # In the case of multi-query attention, the keys and values will have only one head.
    # GQA is only supported with fp16/bf16 dtypes - the fused attention kernels
    # don't support GQA with float32.
    dtype_supports_gqa = q_BHSD.dtype in {torch.float16, torch.bfloat16}
    if gqa_is_supported() and dtype_supports_gqa:
        keys = k_BJSD
        values = v_BJSD
        enable_gqa = {"enable_gqa": True}
    else:
        # On older GPUs or with float32 dtype, the fused attention kernels don't
        # support broadcasting, so we manually expand the keys and values to the
        # same number of heads as the queries.
        keys = k_BJSD.expand(-1, q_BHSD.shape[-3], -1, -1)
        values = v_BJSD.expand(-1, q_BHSD.shape[-3], -1, -1)
        enable_gqa = {}

    # Enable backends explicitly to ensure we don't silently fall back to
    # the math backend, which requires a lot of memory as attention scores
    # are stored explicitly.
    backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ]
    if not torch.cuda.is_available():
        backends.append(SDPBackend.MATH)
    num_parallel_calls = q_BHSD.shape[:2].numel()
    num_iterations = 1
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


class LowerPrecisionLayerNorm(torch.nn.LayerNorm):
    """LayerNorm that maintains FP16 precision in autocast mode.

    PyTorch autocast runs LayerNorm in FP32, which has bad effects on our performance
    (we observed 2x slower) and uses more memory. This layer instead disabled autocast
    for the layer norm, so FP16 is maintained if this is the input format.

    WARNING: this could lead to instabilities for larger hidden sizes, so we only enable
    it for hidden sizes of <512.
    """

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype == torch.float16 and sum(self.normalized_shape) < 512:
            with torch.amp.autocast(input.device.type, enabled=False):
                return super().forward(input)

        return super().forward(input)


class TabPFNBlock(nn.Module):
    """A block of one column-wise, one row-wise attention layer and an MLP layer."""

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        """TabPFNBlock constructor.

        Args:
            emsize: The input embedding size.
            nhead: The number of query attention heads to use.
            dim_feedforward: The dimensionality of the feedforward network.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
        """
        super().__init__()
        device_and_dtype = {"device": device, "dtype": dtype}
        assert emsize % nhead == 0
        # The features of a single sample attend to each other.
        self.per_sample_attention_between_features = AlongRowAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            **device_and_dtype,
        )

        # The cells of a single column attend to each other.
        self.per_column_attention_between_cells = AlongColumnAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            **device_and_dtype,
        )

        layer_norm_args = {**device_and_dtype, "elementwise_affine": False}
        self.layernorm_mha1 = LowerPrecisionLayerNorm(emsize, **layer_norm_args)
        self.layernorm_mha2 = LowerPrecisionLayerNorm(emsize, **layer_norm_args)
        self.layernorm_mlp = LowerPrecisionLayerNorm(emsize, **layer_norm_args)

        self.mlp = nn.Sequential(
            torch.nn.Linear(emsize, dim_feedforward, bias=False, **device_and_dtype),
            torch.nn.GELU(),
            torch.nn.Linear(dim_feedforward, emsize, bias=False, **device_and_dtype),
        )
        torch.nn.init.zeros_(cast("torch.nn.Linear", self.mlp[2]).weight)

    def forward(
        self,
        x_BRCE: torch.Tensor,
        single_eval_pos: int,
        save_peak_memory_factor: int | None,
        *,
        cached_kv: KVCacheEntry | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None]:
        """Compute one column-wise, one row-wise attention, and an MLP layer.

        Uses post-norm.

        B: Batch size
        R: Number of rows / items
        C: Number of columns / features
        E: The embedding size of each cell.

        Args:
            x_BRCE:
                The transformer state passed as input to the layer of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos:
                The position from which on everything is treated as test set.
            save_peak_memory_factor:
                If not None, switch to the inference-optimised forward pass which
                reduces memory by chunking the evaluation of each layer over the batch
                dimension.
                If None, use the standard forward pass compatible with gradient
                computation.
            cached_kv:
                Pre-computed train key/value projections for the between-cells
                attention. When provided, ``x_BRCE`` holds test rows only.
            return_kv:
                If True, also return the between-cells attention's train key/value
                projections as a :class:`KVCacheEntry`.

        Returns:
            ``(transformed_state, kv_entry)`` where ``kv_entry`` is ``None`` unless
            ``return_kv`` is True.
        """
        # -- First Block: Attention between features.
        # The row attention has no train/test distinction and is not cached.
        x_BRCE = chunked_evaluate_maybe_inplace(
            self.per_sample_attention_between_features,
            x_BRCE,
            save_peak_memory_factor,
            residual=True,
            # The rows are folded into the batch, so computing attention over the column
            # here is per sample.
            batch_dims=2,
        )
        x_BRCE = chunked_evaluate_maybe_inplace(
            self.layernorm_mha1,
            x_BRCE,
            save_peak_memory_factor,
            residual=False,
            # The batch norm treats every token independently, so the batch includes
            # both the rows and the columns.
            batch_dims=3,
        )

        # -- Second Block: Attention between cells.
        # Call .contiguous() so that _chunk() can operate on x_BCRE in-place, when
        # memory saving is enabled.
        x_BCRE = x_BRCE.transpose(1, 2).contiguous()
        del x_BRCE
        kv_entry: KVCacheEntry | None = None
        if return_kv or cached_kv is not None:
            # Build / cache paths: bypass chunking. This is consistent with the
            # original behaviour.
            B, C = x_BCRE.shape[:2]
            attn_out, kv_entry = self.per_column_attention_between_cells(
                x_BCRE.flatten(0, 1),
                single_eval_pos=single_eval_pos,
                cached_kv=cached_kv,
                return_kv=return_kv,
            )
            x_BCRE = x_BCRE + attn_out.unflatten(0, (B, C))
        else:
            # The columns are flattened into the batch, so we compute attention over the
            # cells of each column independently.
            x_BCRE = chunked_evaluate_maybe_inplace(
                lambda x, single_eval_pos=None: self.per_column_attention_between_cells(
                    x, single_eval_pos=single_eval_pos
                )[0],
                x_BCRE,
                save_peak_memory_factor,
                residual=True,
                batch_dims=2,
                single_eval_pos=single_eval_pos,
            )
        x_BCRE = chunked_evaluate_maybe_inplace(
            self.layernorm_mha2,
            x_BCRE,
            save_peak_memory_factor,
            residual=False,
            batch_dims=3,
        )
        # Again, call .contiguous() so that _chunk() can operate on x_BCRE in-place.
        x_BRCE = x_BCRE.transpose(1, 2).contiguous()
        del x_BCRE

        # -- Third Block: MLP layer.
        x_BRCE = chunked_evaluate_maybe_inplace(
            self.mlp,
            x_BRCE,
            save_peak_memory_factor,
            residual=True,
            # The MLP also treats every token independently, so the batch includes both
            # the rows and the columns.
            batch_dims=3,
        )
        x_BRCE = chunked_evaluate_maybe_inplace(
            self.layernorm_mlp,
            x_BRCE,
            save_peak_memory_factor,
            residual=False,
            batch_dims=3,
        )
        return x_BRCE, kv_entry


@dataclasses.dataclass
class TabPFNV2Cache(KVCache):
    """Explicit KV cache for the single-file TabPFN v2 architecture.

    Attributes:
        kv: Per-block train key/value projections for the between-cells attention
            (only the first multi-query-attention head is stored).
        feature_cache: The fitted feature-preprocessing statistics (constant-feature
            mask, imputation means, standard-scaler mean/std and feature-group scaling),
            so test rows can be preprocessed using the training statistics without
            refitting. See :meth:`TabPFNV2._embed_features`.
        test_y_embedding: The embedded all-NaN target column for a single test row,
            of shape ``(batch_size, num_targets)``. Broadcast across all test rows to
            form the target column of the transformer input.
        train_shape: ``(batch_size, num_train)``.
    """

    feature_cache: dict[str, torch.Tensor] | None = None
    test_y_embedding: torch.Tensor | None = None
    train_shape: tuple[int, int] = (0, 0)

    @override
    def to(self, device: torch.device | str) -> TabPFNV2Cache:
        """Move all cached tensors to the given device. Returns a new cache."""
        return TabPFNV2Cache(
            kv=self._kv_to(device),
            feature_cache=self._dict_of_tensors_to(self.feature_cache, device),
            test_y_embedding=(
                None
                if self.test_y_embedding is None
                else self.test_y_embedding.to(device)
            ),
            train_shape=self.train_shape,
        )


class TabPFNV2(Architecture):
    """TabPFN V2 with post-layernorm and self-attention on test-items."""

    def __init__(
        self,
        *,
        config: TabPFNV2Config,
        n_out: int = 1,
        feature_positional_embedding: Literal["subspace"] | None = "subspace",
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        """Initializes the PerFeatureTransformer module.

        The feature and target preprocessing (NaN/Inf handling, standard scaling,
        constant-feature removal, feature-group normalization and the optional
        multiclass target densification) is implemented functionally in
        :meth:`_embed_features` / :meth:`_embed_targets`, matching the base
        architecture's encoder pipeline. The only learnable parts are the two linear
        projections into the embedding space (``feature_group_embedder`` and
        ``target_embedder``).

        Args:
            config: The model hyperparameters.
            n_out: The number of outputs the model should produce.
            feature_positional_embedding: The positional embedding type to use.
                The  positional embedding is added to the features to help the model
                distinguish them. Currently, only "subspace" is supported.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
        """
        super().__init__()
        if feature_positional_embedding != "subspace":
            raise ValueError("Currently only 'subspace' is supported.")
        self.input_size = config.emsize
        self.hidden_size = self.input_size * 4
        self.features_per_group = config.features_per_group
        self.n_out = n_out
        # The (ordinal) classification targets are densified with the unique training
        # labels iff the base architecture would have added a multiclass target encoder
        # step, i.e. for classification (max_num_classes >= 2).
        self.use_multiclass_target_encoding = config.max_num_classes >= 2

        device_and_dtype = {"device": device, "dtype": dtype}
        self.feature_group_embedder = nn.Linear(
            ENCODING_SIZE_MULTIPLIER * config.features_per_group,
            config.emsize,
            bias=False,
            **device_and_dtype,
        )
        self.target_embedder = nn.Linear(
            ENCODING_SIZE_MULTIPLIER, config.emsize, **device_and_dtype
        )
        self.standard_scaler = TorchStandardScaler()
        self.blocks = nn.ModuleList(
            TabPFNBlock(
                emsize=config.emsize,
                nhead=config.nhead,
                dim_feedforward=self.hidden_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.nlayers)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, n_out),
        )
        self.feature_positional_embedding_embeddings = nn.Linear(
            self.input_size // 4, self.input_size
        )
        self.seed = config.seed
        # Pre-generated column embeddings, persisted to disk so they are consistent
        # across platforms (random number generation varies between devices even with a
        # fixed seed). Used for the first 2000 columns of the production checkpoints.
        self.pre_generated_column_embeddings = load_column_embeddings()
        self._do_encoder_nan_check = True
        # TODO(Phil): This is here to not fail the memory computation. We should make
        # this a proper API.
        self.ninp = config.emsize
        self.emsize = config.emsize

    @property
    @override
    def embedding_dim(self) -> int:
        return self.emsize

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load a state dict, translating old base-architecture key names if needed.

        Checkpoints trained with the multi-file ``base`` architecture use a different
        naming convention (and layout) for the transformer blocks and the decoder.
        """
        has_base_keys = any(
            k.startswith(("transformer_encoder.", "decoder_dict.")) for k in state_dict
        )
        if has_base_keys:
            state_dict = _replace_keys_from_base_architecture(dict(state_dict))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def muon_compatible_params(self) -> set[nn.Parameter]:
        """Return parameters suitable for the Muon optimizer.

        These are the 2D weight matrices inside the transformer blocks
        (attention projections and MLP layers). All other parameters
        (embeddings, output head, biases) should use AdamW.
        """
        return {p for p in self.blocks.parameters() if p.ndim == 2}

    def add_column_embeddings(self, x_BRCX: torch.Tensor) -> torch.Tensor:
        """Add a random embedding to each column to prevent feature collapse."""
        generator = torch.Generator(device=x_BRCX.device).manual_seed(self.seed)
        num_cols, encoding_size = x_BRCX.shape[2], x_BRCX.shape[3]
        embs = torch.randn(
            (num_cols, encoding_size // 4),
            device=x_BRCX.device,
            dtype=x_BRCX.dtype,
            generator=generator,
        )
        # Random numbers vary between devices, even with a fixed seed. Thus, for the
        # production checkpoints (seed 42, embedding subspace size 48 == 192 // 4), we
        # overwrite the first 2000 columns with the pre-generated embeddings loaded from
        # disk to ensure they are consistent between pretraining and inference. Some
        # tests use a smaller embedding size or a different seed, in which case we use
        # the random embeddings.
        if embs.shape[1] == 48 and self.seed == 42:
            embs[:2000] = self.pre_generated_column_embeddings[: embs.shape[0]].to(
                device=embs.device, dtype=embs.dtype
            )
        embs = self.feature_positional_embedding_embeddings(embs)
        return x_BRCX + embs[None, None]

    def _pad_and_group_features(
        self, x_RBC: torch.Tensor
    ) -> tuple[torch.Tensor, int, int, int]:
        """Pad and group the raw feature tensor for preprocessing.

        Returns ``(x_RSF, num_feature_groups, num_rows, batch_size)`` where
        ``S = batch_size * num_feature_groups`` and ``F = features_per_group``.
        """
        num_rows, batch_size = x_RBC.shape[:2]
        x_RSF, num_feature_groups = _pad_and_reshape_feature_groups(
            x_RBC, self.features_per_group
        )
        return x_RSF, num_feature_groups, num_rows, batch_size

    def _embed_features(
        self,
        x_RSF: torch.Tensor,
        *,
        num_train_labels: int,
        num_feature_groups: int,
        batch_size: int,
        feature_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Preprocess and embed the (grouped) features, adding column embeddings.

        Feature encoder pipeline:
        remove-constant-features -> NaN/Inf indicator -> impute -> standard-scale ->
        feature-group normalization -> linear projection. The constant-feature mask and
        the feature-group statistics are fitted over all rows; the imputation means and
        the standard-scaler statistics are fitted on the training rows only (both
        matching the original architecture).

        When ``feature_cache`` is provided (the cached-inference path), the fitted
        statistics are reused instead of being refitted, so test rows are preprocessed
        with the training statistics.

        Returns ``(embedded_x_BRGX, feature_cache)``.
        """
        fitting = feature_cache is None

        column_selection_mask = (
            _constant_feature_mask(x_RSF)
            if fitting
            else feature_cache["column_selection_mask"]
        )
        x_RSF = _remove_constant_features(x_RSF, column_selection_mask)

        # The NaN/Inf indicators must be captured before imputation.
        nan_indicator_RSF = _generate_nan_and_inf_indicator(x_RSF)

        # Impute, then standard-scale. The imputation mean ignores NaN/Inf, while the
        # standard-scaler statistics are computed on the (imputed) training rows.
        feature_means = None if fitting else feature_cache["feature_means"]
        x_RSF, feature_means = _impute_nan_and_inf_with_mean(
            x_RSF, num_train_labels, feature_means
        )
        if fitting:
            scaler_stats = self.standard_scaler.fit(x_RSF[:num_train_labels])
        else:
            scaler_stats = {
                "mean": feature_cache["scaler_mean"],
                "std": feature_cache["scaler_std"],
            }
        x_RSF = self.standard_scaler.transform(x_RSF, fitted_cache=scaler_stats)

        # Feature-group normalization (fit over all rows, matching the base).
        if fitting:
            ng_non_constant_mask, ng_num_used_features = _fit_feature_group_scaling(
                x_RSF
            )
        else:
            ng_non_constant_mask = feature_cache["ng_non_constant_mask"]
            ng_num_used_features = feature_cache["ng_num_used_features"]
        x_RSF = _normalize_feature_groups(
            x_RSF,
            self.features_per_group,
            ng_non_constant_mask,
            ng_num_used_features,
        )

        x_concat_RSF = torch.cat([x_RSF, nan_indicator_RSF], dim=-1)
        x_concat_RSF = x_concat_RSF.to(self.feature_group_embedder.weight.dtype)
        embedded_x_RSX = self.feature_group_embedder(x_concat_RSF)
        embedded_x_RBGX = embedded_x_RSX.unflatten(1, [batch_size, num_feature_groups])
        embedded_x_BRGX = self.add_column_embeddings(embedded_x_RBGX.transpose(0, 1))

        if fitting:
            feature_cache = {
                "column_selection_mask": column_selection_mask,
                "feature_means": feature_means,
                "scaler_mean": scaler_stats["mean"],
                "scaler_std": scaler_stats["std"],
                "ng_non_constant_mask": ng_non_constant_mask,
                "ng_num_used_features": ng_num_used_features,
            }
        return embedded_x_BRGX, feature_cache

    def _embed_targets(
        self,
        y: torch.Tensor,
        *,
        num_rows: int,
        num_train_labels: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Preprocess and embed the targets (matches the base architecture's y-encoder).

        NaN-pads the test rows, captures NaN/Inf indicators, imputes with the training
        mean, optionally densifies the (multiclass) labels and projects into the
        embedding space.

        Returns ``(embedded_y_BRX, feature_means, unique_ys)`` where ``feature_means``
        and ``unique_ys`` are the fitted statistics needed to embed the all-NaN target
        column for the KV cache (see `_embed_nan_target`).
        """
        y_RB1 = _prepare_targets(y, num_rows, batch_size)
        nan_indicator_RB1 = _generate_nan_and_inf_indicator(y_RB1)
        y_RB1, feature_means = _impute_nan_and_inf_with_mean(y_RB1, num_train_labels)

        unique_ys: list[torch.Tensor] = []
        if self.use_multiclass_target_encoding:
            unique_ys = [
                torch.unique(y_RB1[:num_train_labels, b]) for b in range(batch_size)
            ]
            y_RB1 = _flatten_multiclass_targets(y_RB1, unique_ys)

        y_concat_RB1 = torch.cat([y_RB1, nan_indicator_RB1], dim=-1)
        y_concat_RB1 = y_concat_RB1.to(self.target_embedder.weight.dtype)
        embedded_y_RBX = self.target_embedder(y_concat_RB1)
        return embedded_y_RBX.transpose(0, 1), feature_means, unique_ys

    def _embed_nan_target(
        self,
        feature_means_B1: torch.Tensor,
        unique_ys: list[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Embed the all-NaN target column for the KV cache.

        Every test row carries an all-NaN target (it is what we predict), so after
        imputation/densification its embedding is identical across test rows. This
        reproduces ``y_encoder(all_nan_target)`` with the fitted training statistics.

        Returns the embedded target of shape ``(batch_size, num_targets)``.
        """
        # An all-NaN target imputes to the training mean and carries a NaN indicator.
        y_1B1 = feature_means_B1.unsqueeze(0)
        if self.use_multiclass_target_encoding:
            y_1B1 = _flatten_multiclass_targets(y_1B1, unique_ys)
        nan_indicator_1B1 = torch.full(
            (1, batch_size, 1),
            NAN_INDICATOR,
            device=y_1B1.device,
            dtype=y_1B1.dtype,
        )
        y_concat_1B1 = torch.cat([y_1B1, nan_indicator_1B1], dim=-1)
        y_concat_1B1 = y_concat_1B1.to(self.target_embedder.weight.dtype)
        return self.target_embedder(y_concat_1B1)[0]

    def _decode(
        self, x_BRCD: torch.Tensor, start: int, *, only_return_standard_out: bool
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Project the per-row embeddings of the target column to outputs.

        ``start`` is the first row treated as a test row.
        """
        test_embeddings_TBE = x_BRCD[:, start:, -1].transpose(0, 1)
        test_output_TB1 = self.output_projection(test_embeddings_TBE)
        if only_return_standard_out:
            return test_output_TB1
        return {
            "standard": test_output_TB1,
            "train_embeddings": x_BRCD[:, :start, -1].transpose(0, 1),
            "test_embeddings": test_embeddings_TBE,
        }

    def forward(  # noqa: C901, PLR0912
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
        kv_cache: TabPFNV2Cache | None = None,
        return_kv_cache: bool = False,
        x_is_test_only: bool = False,
    ) -> (
        torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[torch.Tensor | dict[str, torch.Tensor], TabPFNV2Cache | None]
    ):
        """Perform a forward pass.

        See ModelInterface.forward() for the full docstring of the shared arguments.

        In addition to those, this architecture supports an explicit KV cache:
        ``kv_cache``, when provided and populated, makes predictions for the rows in
        ``x`` (all treated as test rows) by attending to the cached training key/value
        projections, without the training rows being present. ``return_kv_cache``, when
        True, also builds and returns a :class:`TabPFNV2Cache`. ``x_is_test_only``, when
        True, signals that ``x`` contains only test rows and requires a populated
        ``kv_cache``.
        """
        if performance_options is None:
            performance_options = self.get_default_performance_options()
        force_recompute_layer = performance_options.force_recompute_layer
        save_peak_memory_factor = performance_options.save_peak_memory_factor
        del categorical_inds
        del task_type

        using_cache = kv_cache is not None and not kv_cache.is_empty()
        if x_is_test_only and not using_cache:
            raise ValueError(
                "x_is_test_only=True requires a populated kv_cache; the standard "
                "forward needs the full train+test tensor."
            )

        if isinstance(x, dict):
            if len(x) != 1:
                raise NotImplementedError(
                    f"Multiple keys in x not implemented yet ({x.keys()})."
                )
            x = x["main"]
        if isinstance(y, dict):
            if len(y) != 1:
                raise NotImplementedError(
                    f"Multiple keys in y not implemented yet ({y.keys()})."
                )
            y = y["main"]
        elif y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)

        x_RSF, num_feature_groups, num_rows, batch_size = self._pad_and_group_features(
            x
        )

        # Populated on the cache-build path so we can store the fitted feature-
        # preprocessing statistics and the embedded all-NaN target in the cache.
        feature_cache: dict[str, torch.Tensor] | None = None
        test_y_embedding: torch.Tensor | None = None

        if using_cache:
            # Cache path: every row in x is a test row attending to the cached train
            # K/V. Reuse the fitted feature-preprocessing statistics from the cache.
            embedded_x_BRGX, _ = self._embed_features(
                x_RSF,
                num_train_labels=0,
                num_feature_groups=num_feature_groups,
                batch_size=batch_size,
                feature_cache=kv_cache.feature_cache,
            )
            # The target column is the embedded all-NaN target, broadcast to all rows.
            test_y_BY = kv_cache.test_y_embedding.to(
                device=embedded_x_BRGX.device, dtype=embedded_x_BRGX.dtype
            )
            test_y_BRY = test_y_BY[:, None, :].expand(batch_size, num_rows, -1)
            x_BRCD = torch.cat((embedded_x_BRGX, test_y_BRY[:, :, None]), dim=2)
            del embedded_x_BRGX
            num_train_labels = kv_cache.train_shape[1]
            block_single_eval_pos = 0
        else:
            # Standard / cache-build path: x contains train (+ optionally test) rows.
            num_train_labels = y.shape[0]
            embedded_y_BRY, y_feature_means, unique_ys = self._embed_targets(
                y,
                num_rows=num_rows,
                num_train_labels=num_train_labels,
                batch_size=batch_size,
            )

            embedded_x_BRGX, feature_cache = self._embed_features(
                x_RSF,
                num_train_labels=num_train_labels,
                num_feature_groups=num_feature_groups,
                batch_size=batch_size,
            )

            # Add the targets as an additional column.
            x_BRCD = torch.cat((embedded_x_BRGX, embedded_y_BRY[:, :, None]), dim=2)
            # Drop refs so the (full-size) embedded feature/target tensors are freed
            # before the transformer blocks run.
            del embedded_x_BRGX, embedded_y_BRY
            # This check results in a graph break with torch compile, so we only run it
            # once in the beginning and then disable it.
            if self._do_encoder_nan_check:
                if torch.isnan(x_BRCD).any():
                    raise ValueError(
                        "Found NaNs in the encoded x and y. Make sure to use "
                        "a NaN-handling encoder."
                    )
                self._do_encoder_nan_check = False
            block_single_eval_pos = num_train_labels

            if return_kv_cache:
                # The all-NaN target column is shared by every test row; embed it once
                # here so the cache path can broadcast it without re-running the
                # (fitted) target preprocessing.
                test_y_embedding = self._embed_nan_target(
                    y_feature_means, unique_ys, batch_size
                ).detach()

        # This model is really heavy on memory but light on compute. On an A100,
        # we are completely CPU-bound. Using checkpointing, we can save a lot of
        # memory, which we can invest into increasing the compute via increased batch
        # size.
        kv_out: dict[int, KVCacheEntry] | None = (
            {} if (return_kv_cache and not using_cache) else None
        )
        for layer_idx, block in enumerate(self.blocks):
            if return_kv_cache and not using_cache:
                x_BRCD, kv_entry = block(
                    x_BRCD,
                    block_single_eval_pos,
                    save_peak_memory_factor,
                    return_kv=True,
                )
                kv_out[layer_idx] = kv_entry
            elif using_cache:
                x_BRCD, _ = block(
                    x_BRCD,
                    block_single_eval_pos,
                    save_peak_memory_factor,
                    cached_kv=kv_cache.kv[layer_idx],
                )
            elif force_recompute_layer:
                x_BRCD = torch.utils.checkpoint.checkpoint(
                    block, x_BRCD, block_single_eval_pos, save_peak_memory_factor
                )[0]
            else:
                x_BRCD, _ = block(
                    x_BRCD, block_single_eval_pos, save_peak_memory_factor
                )

        # In the cache path every row is a test row; otherwise test rows start after the
        # training rows.
        output = self._decode(
            x_BRCD,
            0 if using_cache else num_train_labels,
            only_return_standard_out=only_return_standard_out,
        )

        if not return_kv_cache:
            return output

        if using_cache:
            return output, kv_cache

        # Build the cache for later test-only inference.
        assert kv_out is not None
        built_cache = TabPFNV2Cache(
            kv=kv_out,
            feature_cache=feature_cache,
            test_y_embedding=test_y_embedding,
            train_shape=(batch_size, num_train_labels),
        )
        return output, built_cache


def _replace_keys_from_base_architecture(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Translate a base-architecture state dict to the single-file v2 key names.

    The transformer blocks and the decoder are renamed/restructured, and the base
    encoder's / y-encoder's final linear projection (which is now implemented directly
    on the model as ``feature_group_embedder`` / ``target_embedder``) is renamed. The
    remaining base encoder steps only held non-persistent buffers, so they contribute no
    keys. Any keys not in the mapping are passed through unchanged.
    """
    n_layers = sum(k.endswith("self_attn_between_features._w_qkv") for k in state_dict)

    base_to_v2_mapping = [
        # The base encoder's final linear projection (``encoder.5.layer``) and target
        # projection (``y_encoder.2.layer``) are now plain ``nn.Linear`` attributes.
        ("encoder.5.layer.weight", "feature_group_embedder.weight"),
        # regression: target linear projection was at position 1
        ("y_encoder.1.layer.weight", "target_embedder.weight"),
        ("y_encoder.1.layer.bias", "target_embedder.bias"),
        # multiclass: target linear projection was at position 2
        ("y_encoder.2.layer.weight", "target_embedder.weight"),
        ("y_encoder.2.layer.bias", "target_embedder.bias"),
        ("decoder_dict.standard.0.weight", "output_projection.0.weight"),
        ("decoder_dict.standard.0.bias", "output_projection.0.bias"),
        ("decoder_dict.standard.2.weight", "output_projection.2.weight"),
        ("decoder_dict.standard.2.bias", "output_projection.2.bias"),
    ]
    for i in range(n_layers):
        base_to_v2_mapping.extend(
            [
                (
                    f"transformer_encoder.layers.{i}.mlp.linear1.weight",
                    f"blocks.{i}.mlp.0.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.mlp.linear2.weight",
                    f"blocks.{i}.mlp.2.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_features._w_qkv",
                    f"blocks.{i}.per_sample_attention_between_features.qkv_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_features._w_out",
                    f"blocks.{i}.per_sample_attention_between_features.out_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_items._w_qkv",
                    f"blocks.{i}.per_column_attention_between_cells.qkv_projection.weight",
                ),
                (
                    f"transformer_encoder.layers.{i}.self_attn_between_items._w_out",
                    f"blocks.{i}.per_column_attention_between_cells.out_projection.weight",
                ),
            ]
        )

    new_state_dict: dict[str, torch.Tensor] = {}
    known_base_keys: set[str] = set()
    for base_key, v2_key in base_to_v2_mapping:
        known_base_keys.add(base_key)
        if base_key not in state_dict:
            continue

        # The base QKV weight has shape (3, num_heads, head_size, input_size). Split it
        # into separate q/k/v weights of shape (num_heads * head_size, input_size).
        if "qkv_projection.weight" in v2_key:
            q_key = v2_key.replace("qkv_projection", "q_projection")
            k_key = v2_key.replace("qkv_projection", "k_projection")
            v_key = v2_key.replace("qkv_projection", "v_projection")
            new_state_dict[q_key] = state_dict[base_key][0].flatten(0, 1)
            new_state_dict[k_key] = state_dict[base_key][1].flatten(0, 1)
            new_state_dict[v_key] = state_dict[base_key][2].flatten(0, 1)
            continue

        # The base out-projection weight has shape (num_heads, head_size, output_size).
        # Flatten the heads and transpose to the nn.Linear convention (output, input).
        if "out_projection.weight" in v2_key:
            new_state_dict[v2_key] = state_dict[base_key].flatten(0, 1).T
            continue

        new_state_dict[v2_key] = state_dict[base_key]

    for key, value in state_dict.items():
        if key not in known_base_keys:
            new_state_dict[key] = value

    return new_state_dict


def parse_config(config: dict[str, Any]) -> tuple[TabPFNV2Config, dict[str, Any]]:
    """Parse the config dict into a TabPFNV2Config, return unused keys.

    Args:
        config: Config dict to parse. This function should use Pydantic to
            verify that it matches the expected schema.

    Returns:
        A tuple, (parsed config, dict containing unused config items).

    Raises:
        pydantic.ValidationError: one or more of the values have the wrong type
    """
    allowed_keys = [field.name for field in dataclasses.fields(TabPFNV2Config)]
    usable_config = {k: v for k, v in config.items() if k in allowed_keys}
    unused_config = {k: v for k, v in config.items() if k not in allowed_keys}
    parsed_config = TabPFNV2Config(**usable_config)
    return parsed_config, unused_config


def _pad_and_reshape_feature_groups(
    x_RiBC: torch.Tensor, num_features_per_group: int
) -> tuple[torch.Tensor, int]:
    """Pad the columns to a multiple of the group size and fold groups into the batch.

    Returns:
        A tuple of the reshaped tensor of shape ``(Ri, B * G, F)`` and the number of
        feature groups ``G``, where:
        - Ri = number of input rows (train + test),
        - B = batch size,
        - G = number of feature groups,
        - F = number of features per group.
    """
    num_columns = x_RiBC.shape[-1]
    num_padding_features = -num_columns % num_features_per_group
    x_RiBC = torch.nn.functional.pad(x_RiBC, pad=(0, num_padding_features), value=0)
    num_rows, batch_size, num_padded_columns = x_RiBC.shape
    num_feature_groups = num_padded_columns // num_features_per_group
    x_RiBgF = x_RiBC.reshape(
        num_rows, batch_size * num_feature_groups, num_features_per_group
    )
    return x_RiBgF, num_feature_groups


def _constant_feature_mask(x_RSF: torch.Tensor) -> torch.Tensor:
    """Return a ``(S, F)`` mask that is True for non-constant features.

    A feature is constant if every row equals the first row. Padding columns (all
    zeros) are therefore flagged as constant. Matches the base architecture's
    ``RemoveEmptyFeaturesEncoderStep`` (the mask is computed over all rows).
    """
    return (x_RSF[1:] == x_RSF[0]).sum(0) != (x_RSF.shape[0] - 1)


def _remove_constant_features(
    x_RSF: torch.Tensor, non_constant_mask: torch.Tensor
) -> torch.Tensor:
    """Move non-constant features to the front and zero-pad back to the group size.

    Matches the base architecture's ``RemoveEmptyFeaturesEncoderStep._transform``:
    constant features are dropped (then re-padded with zeros so the group size stays
    constant), which also re-orders the features within each group.
    """
    orig_num_features = x_RSF.shape[-1]
    x_RSF = select_features(x_RSF, non_constant_mask.to(torch.bool))
    padding = -x_RSF.shape[-1] % orig_num_features
    return torch.nn.functional.pad(x_RSF, pad=(0, padding), value=0)


def _generate_nan_and_inf_indicator(x: torch.Tensor) -> torch.Tensor:
    """Generate the NaN/+Inf/-Inf indicator features for ``x`` (matches the base)."""
    return (
        torch.isnan(x) * NAN_INDICATOR
        + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * INFINITY_INDICATOR
        + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
        * NEG_INFINITY_INDICATOR
    ).to(x.dtype)


def _impute_nan_and_inf_with_mean(
    x: torch.Tensor,
    num_train_rows: int,
    feature_means: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace NaN/Inf cells with the per-feature mean of the (finite) training rows.

    When ``feature_means`` is given (the cached-inference path) it is used directly
    instead of being recomputed from the training rows.

    Returns:
        A tuple of ``(imputed tensor, feature_means)``.
    """
    if feature_means is None:
        feature_means = torch_nanmean(x[:num_train_rows], axis=0, include_inf=True)
    nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
    imputed = torch.where(nan_mask, feature_means.unsqueeze(0).expand_as(x), x)
    return imputed, feature_means


def _fit_feature_group_scaling(
    x_RSF: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-group scaling statistics used by ``_normalize_feature_groups``.

    Returns:
        A tuple of ``(non_constant_mask, number_of_used_features)`` of shapes ``(S, F)``
        and ``(S, 1)`` respectively.
    """
    non_constant_mask = _constant_feature_mask(x_RSF)
    number_of_used_features = torch.clip(non_constant_mask.sum(-1, keepdim=True), min=1)
    return non_constant_mask, number_of_used_features


def _normalize_feature_groups(
    x_RSF: torch.Tensor,
    num_features_per_group: int,
    non_constant_mask: torch.Tensor,
    number_of_used_features: torch.Tensor,
) -> torch.Tensor:
    """Scale feature groups so padding/constant columns do not change the variance.

    Each group is scaled by ``sqrt(num_features_per_group / num_used_features)`` and
    constant features are zeroed out. Matches the base architecture's
    ``NormalizeFeatureGroupsEncoderStep`` (with ``normalize_by_sqrt=True``).
    """
    scale = num_features_per_group / number_of_used_features.to(x_RSF.device)
    x_RSF = x_RSF * torch.sqrt(scale)
    return torch.where(
        non_constant_mask.unsqueeze(0).expand_as(x_RSF),
        x_RSF,
        torch.zeros_like(x_RSF),
    )


def _flatten_multiclass_targets(
    y_RB1: torch.Tensor, unique_ys: list[torch.Tensor]
) -> torch.Tensor:
    """Map each target to the count of smaller unique training values, per batch item.

    Matches the base architecture's ``MulticlassClassificationTargetEncoderStep``: it
    densifies the (ordinal) class labels using the unique training labels.
    """
    y_new = y_RB1.clone()
    for b in range(y_RB1.shape[1]):
        y_new[:, b, :] = (y_RB1[:, b, :].unsqueeze(-1) > unique_ys[b]).sum(dim=-1)
    return y_new


def _prepare_targets(y: torch.Tensor, num_rows: int, batch_size: int) -> torch.Tensor:
    """Reshape ``y`` to ``(num_rows, B, 1)`` and NaN-pad the missing (test) rows.

    Args:
        y: Target values of shape ``[Rt]``, ``[Rt, B]``, or ``[Rt, B, 1]`` where
            ``Rt`` is the number of train rows.
        num_rows: The total number of rows in ``x`` (train + test).
        batch_size: The batch size used to reshape ``y``.

    Returns:
        A tensor of shape ``(num_rows, B, 1)`` with the test rows set to NaN.
    """
    num_train_labels = y.shape[0]
    # Note: we allow `num_train_labels == num_rows` (i.e., no test data) to support
    # use cases like KV-caching and for consistency with the OOM check script
    # (`src/fomo_fitting/scripts/check_oom.py`).
    if num_train_labels > num_rows:
        raise ValueError("No test rows provided.")
    target_RB1 = y.view(num_train_labels, 1 if y.ndim == 1 else batch_size, -1)
    return torch.nn.functional.pad(
        target_RB1,
        (0, 0, 0, 0, 0, num_rows - num_train_labels),
        value=float("nan"),
    )


def get_architecture(
    config: ArchitectureConfig,
    *,
    cache_trainset_representation: bool = False,
) -> TabPFNV2:
    """Construct TabPFNV2 based on the given config.

    This factory method implements the interface defined in
    tabpfn.architectures.interface.ArchitectureModule.get_architecture().

    Args:
        config: The config returned by parse_config(). This method should use a
            runtime isinstance() check to downcast the config to this architecture's
            specific config class.
        cache_trainset_representation: Accepted for interface compatibility but
            ignored. This architecture uses an explicit KV cache passed through
            forward() (``kv_cache`` / ``return_kv_cache``) rather than model-internal
            caching, so no special construction is required.

    Returns: the constructed architecture
    """
    assert isinstance(config, TabPFNV2Config)
    # The explicit KV cache is selected at call time via forward()'s kv_cache /
    # return_kv_cache arguments, so the model does not need configuring here.
    del cache_trainset_representation
    n_out = config.max_num_classes or config.num_buckets
    return TabPFNV2(config=config, n_out=n_out)
