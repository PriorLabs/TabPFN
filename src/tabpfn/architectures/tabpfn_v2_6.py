"""The TabPFN v2.6 architecture.

Compared to v2.5, this uses RMSNorm instead of LayerNorm.

Copyright (c) Prior Labs GmbH 2026.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override

import pydantic
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

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
from tabpfn.preprocessing.torch.ops import select_features, torch_nanmean
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler

NAN_INDICATOR = -2.0
INFINITY_INDICATOR = 2.0
NEG_INFINITY_INDICATOR = 4.0

# Feature group size multiplier to compute the size of the encodings projected
# to the embedding dimension. Since we add nan indicators we multiply the
# number of features groups by 2.
ENCODING_SIZE_MULTIPLIER = 2

if TYPE_CHECKING:
    from tabpfn.constants import TaskType


@pydantic.dataclasses.dataclass
class TabPFNV2p6Config(ArchitectureConfig):
    """Configuration for the single-file TabPFN v2.6 architecture."""

    name: str = "TabPFN-v2.6"
    emsize: int = 192
    nlayers: int = 24
    nhead: int = 3
    """Number of key/value heads to use for per-column-inter-row attention."""

    features_per_group: int = 3
    """If > 1, the features will be grouped into groups of this size and the attention
    is across groups."""

    num_thinking_rows: int = 64
    """Number of "thinking rows" to prepend to each dataset, see AddThinkingRows."""

    encoder_type: Literal["linear", "mlp"] = "linear"
    """Whether to use a linear or MLP encoder in the input encoder."""

    encoder_mlp_hidden_dim: int = 1024
    """Hidden dimension for the MLP embedder."""


class AddThinkingRows(nn.Module):
    """Takes the embedded input and prepends "thinking rows" to it.

    Adjusts the single_eval_pos appropriately to account for the new, longer input.

    The thinking tokens give the model more computational capacity to perform in-context
    learning.
    """

    def __init__(self, num_thinking_rows: int, embedding_size: int) -> None:
        super().__init__()
        self.num_thinking_rows = num_thinking_rows
        # We have to work with variable numbers of features, so we use the same token
        # for each feature.
        self.row_token_values_TE = nn.Parameter(
            torch.empty(num_thinking_rows, embedding_size)
        )
        self.reset_parameters()

    @override
    def forward(
        self,
        x_BRiCE: torch.Tensor,
        single_eval_pos: int,
    ) -> tuple[torch.Tensor, int]:
        """Prepends the thinking tokens to the embedded input.

        Args:
            x_BRiCE: Input tensor of shape (B, Ri, C, E), where:
                - B: batch size
                - Ri: number of input rows (train and test)
                - C: number of feature groups
                - E: Model embedding size.
            single_eval_pos: Rows after this index are treated as evaluation rows.

        Returns:
            A tuple, (x_BRCE, updated single_eval_pos), where where x_BRCE has added
            rows, now having shape (B, R = Ri + num thinking rows, C, E).
        """
        # T: num thinking rows
        # R: Ri + T, total rows including thinking rows
        batch_size, _, num_features, _ = x_BRiCE.shape
        thinking_rows_BTCE = (
            self.row_token_values_TE.unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, -1, num_features, -1)
        )
        x_BRCE = torch.cat([thinking_rows_BTCE, x_BRiCE], dim=1)
        single_eval_pos += self.num_thinking_rows
        return x_BRCE, single_eval_pos

    def reset_parameters(self) -> None:
        """Set the tokens to randomly-sampled values."""
        # This is the initialisation used in torch.nn.Embedding, so hopefully a
        # reasonable choice for our application.
        torch.nn.init.normal_(self.row_token_values_TE)


class Attention(nn.Module):
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
                - R: Total rows (thinking + train + test), or test-only rows when
                  ``cached_kv`` is provided.
                - E: Embedding size.
            single_eval_pos: The position from which on everything is treated as test
                set. If None, no mask is applied and all positions are attended to. If
                given, each query after single_eval_pos will only attend to positions
                before single_eval_pos. Should be 0 when ``cached_kv`` is provided.
            cached_kv: Pre-computed key/value projections from a previous forward pass
                (thinking + train rows). When provided, the K/V projections are skipped
                and every query (all rows are test rows) attends to these cached values
                via the single multi-query-attention head.
            return_kv: If True, also return the thinking+train key/value projections as
                a :class:`KVCacheEntry`. Only the first key/value head is cached, since
                test rows attend to the first head only (multi-query attention).

        Returns:
            ``(output, kv_entry)`` where ``kv_entry`` is ``None`` unless ``return_kv``
            is True.
        """
        # H: number of heads.
        # D: head dimension.
        # F: head_dimension * number of heads.
        # N: number of thinking and train rows = single_eval_pos
        # M: number of test rows
        Bc, R, _ = x_BcRE.shape

        q_BcRHD = self.q_projection(x_BcRE).view(Bc, R, -1, self.head_dim)

        kv_entry: KVCacheEntry | None = None
        if cached_kv is not None:
            # Cache path: every row is a test row attending to the cached thinking+train
            # K/V via the single multi-query-attention head.
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

    # Enable backends explicitly. MATH is included as a last resort since it
    # stores attention scores explicitly and uses more memory, but we prefer
    # a slow fallback over a hard crash (e.g. on T4 GPUs on github runners where
    # flash/efficient attention kernels may not support all configurations).
    backends = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    num_parallel_calls = q_BHSD.shape[:2].numel()
    CUDA_MAX_GRID = 65536
    num_iterations = (num_parallel_calls + CUDA_MAX_GRID - 1) // CUDA_MAX_GRID
    sub_batch = (q_BHSD.shape[0] + num_iterations - 1) // num_iterations

    # MPS's scaled_dot_product_attention silently returns wrong values when given
    # non-contiguous inputs (the permute above and, for multi-query attention, the
    # expand below both produce non-contiguous tensors). PyTorch bug:
    # https://github.com/pytorch/pytorch/issues/181133
    # We apply .contiguous() to the per-chunk slices rather than the full tensors,
    # to avoid doubling peak memory.
    on_mps = q_BHSD.device.type == "mps"

    with sdpa_kernel(backends=backends):
        outputs = []
        for i in range(num_iterations):
            q_chunk = q_BHSD[i * sub_batch : (i + 1) * sub_batch]
            k_chunk = keys[i * sub_batch : (i + 1) * sub_batch]
            v_chunk = values[i * sub_batch : (i + 1) * sub_batch]
            if on_mps:
                q_chunk = q_chunk.contiguous()
                k_chunk = k_chunk.contiguous()
                v_chunk = v_chunk.contiguous()
            outputs.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attn_mask=None,
                    **enable_gqa,
                )
            )
    output_BHSD = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output_BHSD.permute(0, 2, 1, 3)


class LowerPrecisionRMSNorm(torch.nn.RMSNorm):
    """RMSNorm that maintains FP16/BF16 precision in autocast mode.

    PyTorch autocast runs RMSNorm in FP32, which has bad effects on our performance
    (we observed 2x slower) and uses more memory. This layer instead disabled autocast
    for the layer norm, so FP16 is maintained if this is the input format.

    WARNING: this could lead to instabilities for larger hidden sizes, so we only enable
    it for hidden sizes of <512.
    """

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (
            input.dtype in (torch.float16, torch.bfloat16)
            and sum(self.normalized_shape) < 512
        ):
            with torch.amp.autocast("cuda" if input.is_cuda else "cpu", enabled=False):
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

        layer_norm_args = {**device_and_dtype, "elementwise_affine": True}
        self.layernorm_mha1 = LowerPrecisionRMSNorm(emsize, **layer_norm_args)
        self.layernorm_mha2 = LowerPrecisionRMSNorm(emsize, **layer_norm_args)
        self.layernorm_mlp = LowerPrecisionRMSNorm(emsize, **layer_norm_args)

        self.mlp = nn.Sequential(
            torch.nn.Linear(emsize, dim_feedforward, bias=False, **device_and_dtype),
            torch.nn.GELU(),
            torch.nn.Linear(dim_feedforward, emsize, bias=False, **device_and_dtype),
        )
        torch.nn.init.zeros_(cast("torch.nn.Linear", self.mlp[2]).weight)

    @override
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
                Pre-computed key/value projections for the between-cells attention.
                When provided, ``x_BRCE`` holds test rows only.
            return_kv:
                If True, also return the between-cells attention's key/value
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
            # original behaviour when memory saving is disabled.
            B, C = x_BCRE.shape[:2]
            attn_out, kv_entry = self.per_column_attention_between_cells(
                x_BCRE.flatten(0, 1),
                single_eval_pos=single_eval_pos,
                cached_kv=cached_kv,
                return_kv=return_kv,
            )
            x_BCRE = x_BCRE + attn_out.unflatten(0, (B, C))
        else:
            x_BCRE = chunked_evaluate_maybe_inplace(
                lambda x, single_eval_pos=None: self.per_column_attention_between_cells(
                    x, single_eval_pos=single_eval_pos
                )[0],
                x_BCRE,
                save_peak_memory_factor,
                residual=True,
                # The columns are flattened into the batch, so we compute attention over
                # the cells of each column independently.
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
        # Again, call .contiguous() so that _chunk() can operate on x_BRCE in-place.
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
class TabPFNV2p6Cache(KVCache):
    """Explicit KV cache for the TabPFN v2.6 architecture.

    Stores everything derived from the training data that is needed to make predictions
    for test rows without the training (or thinking) rows being present:

    Attributes:
        kv: Per-block key/value projections for the between-cells attention
            (thinking + train rows, only the first multi-query-attention head is
            stored).
        scaler_cache: Fitted standard-scaler statistics (``mean``, ``std``). Reused for
            both imputation and standardisation of test-only data.
        feature_state: The remaining train-derived feature preprocessing parameters
            that depend on the full train+test input: the constant-feature column mask
            and the feature-group normalisation parameters.
        test_y_embedding: The embedded all-NaN target column for a single test row, of
            shape ``(batch_size, emsize)``. Broadcast across all test rows to form the
            target column of the transformer input.
        train_shape: ``(batch_size, num_train_labels)``.
    """

    scaler_cache: dict[str, torch.Tensor] | None = None
    feature_state: dict[str, torch.Tensor] | None = None
    test_y_embedding: torch.Tensor | None = None
    train_shape: tuple[int, int] = (0, 0)

    @override
    def to(self, device: torch.device | str) -> TabPFNV2p6Cache:
        """Move all cached tensors to the given device. Returns a new cache."""
        return TabPFNV2p6Cache(
            kv=self._kv_to(device),
            scaler_cache=self._dict_of_tensors_to(self.scaler_cache, device),
            feature_state=self._dict_of_tensors_to(self.feature_state, device),
            test_y_embedding=(
                None
                if self.test_y_embedding is None
                else self.test_y_embedding.to(device)
            ),
            train_shape=self.train_shape,
        )


class TabPFNV2p6(Architecture):
    """TabPFN V2.6 with post-layernorm and self-attention on test-items."""

    def __init__(
        self,
        *,
        config: TabPFNV2p6Config,
        task_type: TaskType,
        n_out: int = 1,
        feature_positional_embedding: Literal["subspace"] | None = "subspace",
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        """Initializes the PerFeatureTransformer module.

        Args:
            config: The model hyperparameters.
            encoder: An InputEncoder, which takes a dictionary with tensors of shape
                [num_rows, batch_size, num_cols, features] and returns a single tensor
                of shape [num_rows, batch_size, input_size].
            task_type: The type of task the model should perform.
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
        self.hidden_size = self.input_size * 2
        self.features_per_group = config.features_per_group
        self.n_out = n_out
        self.task_type: TaskType = task_type

        self.feature_group_embedder = self._get_feature_group_embedder(config)
        self.target_embedder = nn.Linear(ENCODING_SIZE_MULTIPLIER, config.emsize)
        self.add_thinking_rows = AddThinkingRows(
            num_thinking_rows=config.num_thinking_rows,
            embedding_size=config.emsize,
        )
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
        self.standard_scaler = TorchStandardScaler()

        self.pre_generated_column_embeddings = load_column_embeddings()
        self.feature_positional_embedding_embeddings = nn.Linear(
            self.input_size // 4, self.input_size
        )
        self._do_encoder_nan_check = True
        self.emsize = config.emsize

    @property
    @override
    def embedding_dim(self) -> int:
        return self.emsize

    def _get_feature_group_embedder(self, config: TabPFNV2p6Config) -> nn.Module:
        """Get the feature group embedder."""
        encoding_size = config.features_per_group * ENCODING_SIZE_MULTIPLIER
        hidden_dim = config.encoder_mlp_hidden_dim
        emsize = config.emsize
        if config.encoder_type == "mlp":
            embedder = nn.Sequential(
                nn.Linear(encoding_size, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, emsize, bias=False),
            )
        else:
            embedder = nn.Linear(encoding_size, emsize, bias=False)
        return embedder

    def _add_column_embeddings(self, x_BRGX: torch.Tensor) -> torch.Tensor:
        """Add a random embedding to each column to prevent feature collapse."""
        # Tracing for Onnx export can't trace the Generator below, so we use the default
        # RNG.
        if torch.jit.is_tracing():
            generator = None
        else:
            generator = torch.Generator(device=x_BRGX.device).manual_seed(42)
        num_cols, encoding_size = x_BRGX.shape[2], x_BRGX.shape[3]
        embs = torch.randn(
            (num_cols, encoding_size // 4),
            device=x_BRGX.device,
            dtype=x_BRGX.dtype,
            generator=generator,
        )
        # Random numbers vary between different devices, even with a fixed seed. Thus we
        # use the pre-generated column embeddings where possible to ensure they are
        # consistent between pretraining and inference.
        # Some tests use a smaller embedding size, in which case we do not use the saved
        # embeddings.
        if embs.shape[1] == self.pre_generated_column_embeddings.shape[1]:
            embs[:2000] = self.pre_generated_column_embeddings[: embs.shape[0]].to(
                device=embs.device, dtype=embs.dtype
            )
        embs = self.feature_positional_embedding_embeddings(embs)
        x_BRGX += embs[None, None]

        return x_BRGX

    def _decode(
        self,
        x_BRCD: torch.Tensor,
        *,
        test_start: int,
        train_start: int,
        train_end: int,
        only_return_standard_out: bool,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Project the per-row embeddings of the target column to outputs.

        ``test_start`` is the first row treated as a test row. ``train_start`` and
        ``train_end`` delimit the (non-thinking) training rows for the embeddings dict.
        In the cache path all three are 0, so there are no training rows to return.
        """
        # M: number of test samples, B: batch size, D: embedding size.
        test_embeddings_BMD = x_BRCD[:, test_start:, -1]
        test_embeddings_MBD = test_embeddings_BMD.transpose(0, 1)
        test_output_MB1 = self.output_projection(test_embeddings_MBD)

        if only_return_standard_out:
            return test_output_MB1

        train_embeddings_BND = x_BRCD[:, train_start:train_end, -1]
        return {
            "standard": test_output_MB1,
            "train_embeddings": train_embeddings_BND.transpose(0, 1),
            "test_embeddings": test_embeddings_MBD,
        }

    @override
    def forward(  # noqa: C901, PLR0912
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
        kv_cache: TabPFNV2p6Cache | None = None,
        return_kv_cache: bool = False,
        x_is_test_only: bool = False,
    ) -> (
        torch.Tensor
        | dict[str, torch.Tensor]
        | tuple[torch.Tensor | dict[str, torch.Tensor], TabPFNV2p6Cache | None]
    ):
        """Perform a forward pass.

        See ModelInterface.forward() for the full docstring of the shared arguments.

        In addition to those, this architecture supports an explicit KV cache:
        ``kv_cache``, when provided and populated, makes predictions for the rows in
        ``x`` (all treated as test rows) by attending to the cached thinking+train
        key/value projections, without those rows being present. ``return_kv_cache``,
        when True, also builds and returns a :class:`TabPFNV2p6Cache`.
        ``x_is_test_only``, when True, signals that ``x`` contains only test rows and
        requires a populated ``kv_cache``.
        """
        if performance_options is None:
            performance_options = self.get_default_performance_options()
        force_recompute_layer = performance_options.force_recompute_layer
        save_peak_memory_factor = performance_options.save_peak_memory_factor
        del categorical_inds

        using_cache = kv_cache is not None and not kv_cache.is_empty()
        if x_is_test_only and not using_cache:
            raise ValueError(
                "x_is_test_only=True requires a populated kv_cache; the standard "
                "forward needs the full train+test tensor."
            )

        if isinstance(x, dict):
            x = x["main"]
        if isinstance(y, dict):
            y = y["main"]
        if y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)

        if (
            not self.training
            and self.task_type == "multiclass"
            and (y > self.n_out - 1).any()
        ):
            raise ValueError(
                "Target is out of range. Make sure to use an ordinal encoded target. "
                f"Expected target values between 0 and {self.n_out - 1}, but got values"
                f" greater than {self.n_out - 1}."
            )

        # Ri = number of input rows. In the standard / build paths these are the
        # train (+ optionally test) rows; in the cache path they are test-only rows.
        # B = batch size, C = number of columns before grouping.
        x_RiBC = x
        num_input_rows, batch_size, *_ = x_RiBC.shape
        # Populated in the build path (return_kv_cache and not using_cache) and read
        # when constructing the cache below.
        scaler_cache: dict[str, torch.Tensor] | None = None
        feature_state: dict[str, torch.Tensor] | None = None

        if using_cache:
            # Cache path: every row is a test row attending to the cached thinking+train
            # K/V. The caller may pass only the test rows (``x_is_test_only=True``) or
            # the full train+test tensor, in which case the train rows are sliced off
            # here. Reuse the train-derived feature preprocessing.
            num_train_labels = kv_cache.train_shape[1]
            rows_RiBC = x_RiBC if x_is_test_only else x_RiBC[num_train_labels:]
            num_test_rows = rows_RiBC.shape[0]
            embedded_x_BRiGX, _, _ = self._preprocess_and_embed_features(
                x_RiBC=rows_RiBC,
                num_train_labels=0,
                batch_size=batch_size,
                scaler_cache=kv_cache.scaler_cache,
                feature_state=kv_cache.feature_state,
            )
            # The target column is the embedded all-NaN target, broadcast to all rows.
            test_y_BX = kv_cache.test_y_embedding.to(
                device=embedded_x_BRiGX.device, dtype=embedded_x_BRiGX.dtype
            )
            test_y_BRiX = test_y_BX[:, None, :].expand(batch_size, num_test_rows, -1)
            x_BRCD = torch.cat((embedded_x_BRiGX, test_y_BRiX[:, :, None]), dim=2)
            block_single_eval_pos = 0
        else:
            # Standard / cache-build path: x contains train (+ optionally test) rows.
            num_train_labels = y.shape[0]
            (
                embedded_x_BRiGX,
                scaler_cache,
                feature_state,
            ) = self._preprocess_and_embed_features(
                x_RiBC=x_RiBC,
                num_train_labels=num_train_labels,
                batch_size=batch_size,
                return_state=return_kv_cache,
            )
            embedded_y_BRiX = self._preprocess_and_embed_targets(
                y=y,
                num_train_rows=num_input_rows,
                num_train_labels=num_train_labels,
                batch_size=batch_size,
            )

            # Add the targets as an additional column.
            x_BRiCD = torch.cat((embedded_x_BRiGX, embedded_y_BRiX[:, :, None]), dim=2)
            # Drop refs so the (full-size) embedded feature/target tensors are freed
            # before the transformer blocks run.
            del embedded_x_BRiGX, embedded_y_BRiX
            # This check results in a graph break with torch compile, so we only run it
            # once in the beginning and then disable it.
            if self._do_encoder_nan_check:
                if torch.isnan(x_BRiCD).any():
                    raise ValueError(
                        "Found NaNs in the encoded x and y. Make sure to use "
                        "a NaN-handling encoder."
                    )
                self._do_encoder_nan_check = False

            # R = num rows + num thinking rows
            x_BRCD, block_single_eval_pos = self.add_thinking_rows(
                x_BRiCD, single_eval_pos=num_train_labels
            )

        kv_out: dict[int, KVCacheEntry] = {}
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
                    block,
                    x_BRCD,
                    block_single_eval_pos,
                    save_peak_memory_factor,
                    use_reentrant=False,
                )[0]
            else:
                x_BRCD, _ = block(
                    x_BRCD, block_single_eval_pos, save_peak_memory_factor
                )

        # In the cache path every row is a test row; otherwise the test rows start
        # after the thinking and training rows, and the training rows (excluding the
        # leading thinking rows) are returned in the embeddings dict.
        if using_cache:
            output = self._decode(
                x_BRCD,
                test_start=0,
                train_start=0,
                train_end=0,
                only_return_standard_out=only_return_standard_out,
            )
        else:
            output = self._decode(
                x_BRCD,
                test_start=block_single_eval_pos,
                train_start=self.add_thinking_rows.num_thinking_rows,
                train_end=block_single_eval_pos,
                only_return_standard_out=only_return_standard_out,
            )

        if not return_kv_cache:
            return output

        if using_cache:
            return output, kv_cache

        # Build the cache for later test-only inference. The embedded target column of
        # a test row is the same for every test row, so we embed a single all-NaN
        # target row using the train-derived imputation statistics.
        test_y_embedding_BX = self._preprocess_and_embed_targets(
            y=y,
            num_train_rows=num_train_labels + 1,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
        )[:, -1].detach()
        assert kv_out
        built_cache = TabPFNV2p6Cache(
            kv=kv_out,
            scaler_cache=scaler_cache,
            feature_state=feature_state,
            test_y_embedding=test_y_embedding_BX,
            train_shape=(batch_size, num_train_labels),
        )
        return output, built_cache

    def _preprocess_and_embed_features(
        self,
        x_RiBC: torch.Tensor,
        num_train_labels: int,
        batch_size: int,
        *,
        scaler_cache: dict[str, torch.Tensor] | None = None,
        feature_state: dict[str, torch.Tensor] | None = None,
        return_state: bool = False,
    ) -> tuple[
        torch.Tensor, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None
    ]:
        """Preprocess and embed input features and add column embeddings.

        When ``scaler_cache``/``feature_state`` are provided (cache path), ``x_RiBC``
        holds test-only rows and the train-derived preprocessing parameters are reused
        instead of being recomputed, so the result matches the standard forward.

        Args:
            x_RiBC: Input features of shape (Ri, B, C) where:
                - Ri: number of input rows (train + test, or test-only in the cache
                  path)
                - B: batch size
                - C: number of columns before grouping
            num_train_labels: Number of training labels for scaling (ignored when
                ``scaler_cache`` is provided).
            batch_size: Batch size for reshaping
            scaler_cache: Fitted standard-scaler statistics to reuse (cache path).
            feature_state: Constant-feature mask and feature-group normalisation
                parameters to reuse (cache path).
            return_state: If True, also return the (newly computed) ``scaler_cache``
                and ``feature_state`` for building a :class:`TabPFNV2p6Cache`.

        Returns:
            A tuple ``(embedded_x_BRiGX, scaler_cache, feature_state)`` where the
            embedded features have shape (B, Ri, G, X) with G feature groups and
            embedding size X. The two state dicts are ``None`` unless ``return_state``
            is True.
        """
        using_cache = feature_state is not None

        # Remove constant features. The mask depends on the full train+test input, so
        # it is captured and reused for test-only data.
        column_mask = feature_state["column_mask"] if using_cache else None
        x_RiBC, column_mask = _remove_constant_features(
            x_RiBC=x_RiBC, column_mask=column_mask
        )
        # Bg = folded batch size (B * G) and number of feature groups (G)
        x_RiBgF, num_feature_groups = _pad_and_reshape_feature_groups(
            x_RiBC=x_RiBC,
            num_features_per_group=self.features_per_group,
        )
        nan_and_inf_indicator_RiBgF = _generate_nan_and_inf_indicator(x=x_RiBgF)
        # For consistency with old base implementation, the imputation is done before
        # the standard scaling. Imputing with the train mean preserves that mean, so the
        # standard scaler's fitted mean equals the imputation mean and is reused here
        # for the cache path.
        x_RiBgF, _ = _impute_nan_and_inf_with_mean(
            x=x_RiBgF, num_train_rows=num_train_labels, scaler_cache=scaler_cache
        )
        if using_cache:
            x_RiBgF = self.standard_scaler.transform(x_RiBgF, fitted_cache=scaler_cache)
        else:
            fit_data = x_RiBgF[:num_train_labels] if num_train_labels > 0 else x_RiBgF
            scaler_cache = self.standard_scaler.fit(fit_data)
            x_RiBgF = self.standard_scaler.transform(x_RiBgF, fitted_cache=scaler_cache)

        # Normalize feature groups. The mask and feature counts depend on the full input
        # and are captured and reused for test-only data.
        normalize_params = (
            (feature_state["normalize_mask"], feature_state["normalize_used"])
            if using_cache
            else None
        )
        x_RiBgF, normalize_params = _normalize_feature_groups(
            x_RiBF=x_RiBgF,
            num_features_per_group=self.features_per_group,
            params=normalize_params,
        )

        x_RiBgF_concat = torch.cat([x_RiBgF, nan_and_inf_indicator_RiBgF], dim=-1)
        # X = embedding size
        embedded_x_RiBgX = self.feature_group_embedder(x_RiBgF_concat)
        # G = number of feature groups (the number of columns the model will see).
        embedded_x_RiBGX = embedded_x_RiBgX.unflatten(
            1, [batch_size, num_feature_groups]
        )
        embedded_x_BRiGX = embedded_x_RiBGX.transpose(0, 1)
        embedded_x_BRiGX = self._add_column_embeddings(embedded_x_BRiGX)

        if not return_state:
            return embedded_x_BRiGX, None, None
        return (
            embedded_x_BRiGX,
            {k: v.detach() for k, v in scaler_cache.items()},
            {
                "column_mask": column_mask.detach(),
                "normalize_mask": normalize_params[0].detach(),
                "normalize_used": normalize_params[1].detach(),
            },
        )

    def _preprocess_and_embed_targets(
        self,
        y: torch.Tensor,
        num_train_rows: int,
        num_train_labels: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Preprocess and embed the target values.

        Args:
            y: Target values of shape [Rt], [Rt, B], or [Rt, B, 1] where:
                - Rt: number of train rows
                - B: batch size
            num_train_rows: Total number of rows in x
            num_train_labels: Number of training labels for imputation
            batch_size: Batch size for reshaping

        Returns:
            Embedded targets of shape (B, Ri, X) where:
                - Ri: number of input rows (padded)
                - X: embedding size
        """
        y_RiB1 = _prepare_targets(
            y=y,
            num_train_rows=num_train_rows,
            batch_size=batch_size,
        )
        y_nan_and_inf_indicator_RiB1 = _generate_nan_and_inf_indicator(x=y_RiB1)
        y_RiB1 = _impute_target_nan_and_inf(
            y_RiB1=y_RiB1,
            task_type=self.task_type,
            num_train_rows=num_train_labels,
        )

        y_RiB1_concat = torch.cat([y_RiB1, y_nan_and_inf_indicator_RiB1], dim=-1)
        embedded_y_RiBX = self.target_embedder(y_RiB1_concat)

        return embedded_y_RiBX.transpose(0, 1)


def parse_config(config: dict[str, Any]) -> tuple[TabPFNV2p6Config, dict[str, Any]]:
    """Parse the config dict into a TabPFNV2Config, return unused keys.

    Args:
        config: Config dict to parse. This function should use Pydantic to
            verify that it matches the expected schema.

    Returns:
        A tuple, (parsed config, dict containing unused config items).

    Raises:
        pydantic.ValidationError: one or more of the values have the wrong type
    """
    parsed_config = TabPFNV2p6Config(**config)
    return parsed_config, parsed_config.get_unused_config(config)


def get_architecture(
    config: ArchitectureConfig,
    *,
    cache_trainset_representation: bool = False,
) -> TabPFNV2p6:
    """Construct TabPFNV2.6 based on the given config.

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
    assert isinstance(config, TabPFNV2p6Config)
    # The explicit KV cache is selected at call time via forward()'s kv_cache /
    # return_kv_cache arguments, so the model does not need configuring here.
    del cache_trainset_representation
    task_type = "multiclass" if config.max_num_classes > 0 else "regression"
    n_out = config.max_num_classes if task_type == "multiclass" else config.num_buckets
    return TabPFNV2p6(config=config, n_out=n_out, task_type=task_type)


def _prepare_targets(
    y: torch.Tensor,
    num_train_rows: int,
    batch_size: int,
) -> torch.Tensor:
    """Pad the targets to the number of rows in x and validate inputs.

    More specifically, we ensure that:
    1. there are test rows (i.e. y_RtB is shorter than num_train_rows),
    2. y is nan-padded to the number of rows in x.
    3. Make sure the target is 3-dimensional.

    Args:
        y: Target values of shape [Rt], [Rt, B], or [Rt, B, 1] where:
            - Rt: number of train rows
            - B: batch size
        num_train_rows: The number of rows in x.
        batch_size: Batch size for reshaping

    Returns:
        A tensor with the shape (Ri, B, 1), where
            - Ri = number of input rows (train + test)
    """
    num_train_labels = y.shape[0]

    # Check that the number of training labels is not greater than
    # the total number of rows.
    # Note: we allow `num_train_labels == num_train_rows` (i.e., no test data) to
    # support use cases like KV-caching and for consistency with the OOM check script
    # (`src/fomo_fitting/scripts/check_oom.py`).
    if num_train_labels > num_train_rows:
        raise ValueError("No test rows provided.")

    # Make sure the target is 3-dimensional.
    target_RBY = y.view(num_train_labels, 1 if y.ndim == 1 else batch_size, -1)
    return torch.nn.functional.pad(
        target_RBY,
        (0, 0, 0, 0, 0, num_train_rows - num_train_labels),
        value=float("nan"),
    )


def _remove_constant_features(
    x_RiBC: torch.Tensor, *, column_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Removes constant features from the input data.

    The selection mask depends on the full train+test input, so when a previously
    computed ``column_mask`` is provided (cache path) it is reused instead of being
    recomputed, ensuring the same columns are kept for test-only data.

    Returns ``(selected_x, column_mask)``.
    """
    if column_mask is None:
        if x_RiBC.shape[0] <= 1:
            column_mask = torch.ones(
                x_RiBC.shape[1],
                x_RiBC.shape[2],
                dtype=torch.bool,
                device=x_RiBC.device,
            )
        else:
            column_mask = ~(x_RiBC[1:] == x_RiBC[0]).all(0)
    sel = column_mask.to(device=x_RiBC.device, dtype=torch.bool)
    return select_features(x_RiBC, sel), column_mask


def _pad_and_reshape_feature_groups(
    x_RiBC: torch.Tensor, num_features_per_group: int
) -> tuple[torch.Tensor, int]:
    """Pad the feature groups and reshape so the feature group dimension is last.

    Returns:
        A tuple of padded and reshaped tensor with shape [Ri, B * G, F] and
        the number of feature groups.
        where:
        - Ri = number of input rows (train + test, before adding thinking rows)
        - B = batch size
        - G = number of feature groups
        - F = number of features per group
    """
    num_columns = x_RiBC.shape[-1]
    num_padding_features = -num_columns % num_features_per_group
    x_RiBC_padded = torch.nn.functional.pad(
        x_RiBC, pad=(0, num_padding_features), value=0
    )
    num_rows, B, num_padded_columns = x_RiBC_padded.shape
    num_feature_groups = num_padded_columns // num_features_per_group

    x_RiBgF = x_RiBC_padded.reshape(
        num_rows, B * num_feature_groups, num_features_per_group
    )
    return x_RiBgF, num_feature_groups


def _impute_nan_and_inf_with_mean(
    x: torch.Tensor,
    num_train_rows: int,
    scaler_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Impute the nan and inf with the mean of the feature.

    When ``scaler_cache`` is provided (cache path) the cached standard-scaler mean is
    reused instead of recomputing it from ``x[:num_train_rows]``. This is exact:
    imputing NaN/Inf with the train mean preserves that mean, so the scaler's fitted
    mean equals the imputation mean.

    Args:
        x: Tensor of shape [R, ...], where
            R = number of rows
        num_train_rows: The position to use for single evaluation.
        scaler_cache: Fitted standard-scaler statistics to reuse (cache path).

    Returns:
        A tuple of (imputed tensor, nan/inf mask).
    """
    if scaler_cache is not None:
        feature_means = scaler_cache["mean"].to(device=x.device, dtype=x.dtype)
    else:
        feature_means = torch_nanmean(x=x[:num_train_rows], axis=0, include_inf=True)
    nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
    return torch.where(nan_mask, feature_means.unsqueeze(0).expand_as(x), x), nan_mask


def _impute_target_nan_and_inf(
    y_RiB1: torch.Tensor,
    task_type: TaskType,
    num_train_rows: int,
) -> torch.Tensor:
    """Impute NaN/Inf values in the target tensor.

    Args:
        y_RiB1: Tensor of shape [Ri, B, 1], where Ri is the number of train+test rows.
        task_type: The task type ("regression" or "multiclass").
        num_train_rows: Number of training rows.

    """
    if task_type == "regression":
        y_RiB1, _ = _impute_nan_and_inf_with_mean(
            x=y_RiB1, num_train_rows=num_train_rows
        )
        return y_RiB1

    # The following class imputation is performed for backwards compatibility.
    # We impute the mean and then do a ceil() operation.
    # Only apply ceil() to imputed positions to preserve differentiability for
    # original values (e.g. during prompt tuning).
    y_RiB1, nan_inf_mask = _impute_nan_and_inf_with_mean(
        x=y_RiB1, num_train_rows=num_train_rows
    )
    return torch.where(nan_inf_mask, y_RiB1.ceil(), y_RiB1)


def _normalize_feature_groups(
    x_RiBF: torch.Tensor,
    num_features_per_group: int,
    *,
    params: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Normalize the feature groups.

    The non-constant mask and used-feature counts depend on the full train+test input,
    so when previously computed ``params`` are provided (cache path) they are reused
    instead of being recomputed from test-only data.

    Returns ``(normalized_x, (non_constant_mask, number_of_used_features))``.
    """
    if params is None:
        Ri = x_RiBF.shape[0]
        non_constant_mask = (x_RiBF[1:] == x_RiBF[0]).sum(0) != (Ri - 1)
        number_of_used_features = torch.clip(
            non_constant_mask.sum(-1).unsqueeze(-1),
            min=1,
        ).to(x_RiBF.device)
    else:
        non_constant_mask, number_of_used_features = params
        non_constant_mask = non_constant_mask.to(x_RiBF.device)
        number_of_used_features = number_of_used_features.to(x_RiBF.device)
    scale = num_features_per_group / number_of_used_features.to(x_RiBF.dtype)
    x_RiBF = x_RiBF * torch.sqrt(scale)

    return (
        torch.where(
            non_constant_mask.unsqueeze(0).expand_as(x_RiBF),
            x_RiBF,
            torch.zeros_like(x_RiBF),
        ),
        (non_constant_mask, number_of_used_features),
    )


def _generate_nan_and_inf_indicator(x: torch.Tensor) -> torch.Tensor:
    """Generate the nan and inf indicators for a generic input tensor."""
    return (
        torch.isnan(x) * NAN_INDICATOR
        + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * INFINITY_INDICATOR
        + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
        * NEG_INFINITY_INDICATOR
    ).to(x.dtype)
