"""The TabPFN v2.6 architecture.

Compared to v2.5, this uses RMSNorm instead of LayerNorm.

Copyright (c) Prior Labs GmbH 2025.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
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
from tabpfn.architectures.interface import Architecture, ArchitectureConfig
from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace
from tabpfn.architectures.shared.column_embeddings import load_column_embeddings
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
    ) -> torch.Tensor:
        """Forward pass for attention between cells of a single column.

        Args:
            x_BcRE: The input tensor of shape (Bc, R, E), where:
                - Bc: Batch size * number of columns
                - R: Total rows (thinking + train + test).
                - E: Embedding size.
            single_eval_pos: The position from which on everything is treated as test
                set. If None, no mask is applied and all positions are attended to. If
                given, each query after single_eval_pos will only attend to positions
                before single_eval_pos.
        """
        # H: number of heads.
        # D: head dimension.
        # F: head_dimension * number of heads.
        # N: number of thinking and train rows = single_eval_pos
        # M: number of test rows
        Bc, R, _ = x_BcRE.shape
        # If no single_eval_pos was specified, then the whole input is training.
        N = R if single_eval_pos is None else single_eval_pos

        q_flat_BcSHF = self.q_projection(x_BcRE)
        k_flat_BcNHF = self.k_projection(x_BcRE[:, :N])
        v_flat_BcNHF = self.v_projection(x_BcRE[:, :N])
        q_BcRHD = q_flat_BcSHF.view(Bc, R, -1, self.head_dim)
        k_BcNHD = k_flat_BcNHF.view(Bc, N, -1, self.head_dim)
        v_BcNHD = v_flat_BcNHF.view(Bc, N, -1, self.head_dim)

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

        output_BcSF = output_BcSHD.reshape(Bc, R, self.head_dim * self.num_heads)
        return self.out_projection(output_BcSF)


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
    ) -> torch.Tensor:
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

        Returns:
            The transformed state
        """
        # -- First Block: Attention between features.
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
        x_BCRE = chunked_evaluate_maybe_inplace(
            self.per_column_attention_between_cells,
            x_BCRE,
            save_peak_memory_factor,
            residual=True,
            # The columns are flattened into the batch, so we compute attention over the
            # cells of each column independently.
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
        return chunked_evaluate_maybe_inplace(
            self.layernorm_mlp,
            x_BRCE,
            save_peak_memory_factor,
            residual=False,
            batch_dims=3,
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
        # TODO(Phil): This is here to not fail the memory computation. We should make
        # this a proper API.
        self.ninp = config.emsize

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

    @override
    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        force_recompute_layer: bool = False,
        save_peak_memory_factor: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Perform a forward pass.

        See ModelInterface.forward() for the full docstring.
        """
        del categorical_inds
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

        # Ri = number of input rows (train + test, before adding thinking rows)
        # B = batch size
        # C = number of columns before grouping
        x_RiBC = x
        num_train_rows, batch_size, *_ = x_RiBC.shape
        num_train_labels = y.shape[0]

        embedded_x_BRiGX = self._preprocess_and_embed_features(
            x_RiBC=x_RiBC,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
        )

        embedded_y_BRiX = self._preprocess_and_embed_targets(
            y=y,
            num_train_rows=num_train_rows,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
        )

        # Add the targets as an additional column.
        x_BRiCD = torch.cat((embedded_x_BRiGX, embedded_y_BRiX[:, :, None]), dim=2)
        # This check results in a graph break with torch compile, so we only run it once
        # in the beginning and then disable it.
        if self._do_encoder_nan_check:
            if torch.isnan(x_BRiCD).any():
                raise ValueError(
                    "Found NaNs in the encoded x and y. Make sure to use "
                    "a NaN-handling encoder."
                )
            self._do_encoder_nan_check = False

        # R = num rows + num thinking rows
        x_BRCD, num_train_and_thinking_rows = self.add_thinking_rows(
            x_BRiCD, single_eval_pos=num_train_labels
        )

        for block in self.blocks:
            if force_recompute_layer:
                x_BRCD = torch.utils.checkpoint.checkpoint(
                    block,
                    x_BRCD,
                    num_train_and_thinking_rows,
                    save_peak_memory_factor,
                    use_reentrant=False,
                )
            else:
                x_BRCD = block(
                    x_BRCD, num_train_and_thinking_rows, save_peak_memory_factor
                )

        # M: number of test samples
        # B: batch size
        test_embeddings_BMD = x_BRCD[:, num_train_and_thinking_rows:, -1]
        test_embeddings_MBD = test_embeddings_BMD.transpose(0, 1)
        test_output_MB1 = self.output_projection(test_embeddings_MBD)

        if only_return_standard_out:
            return test_output_MB1

        # N: number of training rows
        train_rows_start = self.add_thinking_rows.num_thinking_rows
        train_rows_end = num_train_and_thinking_rows
        train_embeddings_BND = x_BRCD[:, train_rows_start:train_rows_end, -1]
        train_embeddings_NBD = train_embeddings_BND.transpose(0, 1)

        output = {"standard": test_output_MB1}
        output["train_embeddings"] = train_embeddings_NBD
        output["test_embeddings"] = test_embeddings_MBD

        return output

    def _preprocess_and_embed_features(
        self,
        x_RiBC: torch.Tensor,
        num_train_labels: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Preprocess and embed input features and add column embeddings.

        Args:
            x_RiBC: Input features of shape (Ri, B, C) where:
                - Ri: number of input rows (train + test)
                - B: batch size
                - C: number of columns before grouping
            num_train_labels: Number of training labels for scaling
            batch_size: Batch size for reshaping

        Returns:
            Embedded features of shape (B, Ri, G, X) where:
                - G: number of feature groups
                - X: embedding size
        """
        x_RiBC = _remove_constant_features(x_RiBC=x_RiBC)
        # Bg = folded batch size (B * G) and number of feature groups (G)
        x_RiBgF, num_feature_groups = _pad_and_reshape_feature_groups(
            x_RiBC=x_RiBC,
            num_features_per_group=self.features_per_group,
        )
        nan_and_inf_indicator_RiBgF = _generate_nan_and_inf_indicator(x=x_RiBgF)
        # For consistency with old base implementation, the imputation is done
        # before the standard scaling
        x_RiBgF = _impute_nan_and_inf_with_mean(
            x=x_RiBgF, num_train_rows=num_train_labels
        )
        x_RiBgF = self.standard_scaler(x=x_RiBgF, num_train_rows=num_train_labels)
        x_RiBgF = _normalize_feature_groups(
            x_RiBF=x_RiBgF, num_features_per_group=self.features_per_group
        )

        x_RiBgF_concat = torch.cat([x_RiBgF, nan_and_inf_indicator_RiBgF], dim=-1)
        # X = embedding size
        embedded_x_RiBgX = self.feature_group_embedder(x_RiBgF_concat)
        # G = number of feature groups (the number of columns the model will see).
        embedded_x_RiBGX = embedded_x_RiBgX.unflatten(
            1, [batch_size, num_feature_groups]
        )
        embedded_x_BRiGX = embedded_x_RiBGX.transpose(0, 1)

        return self._add_column_embeddings(embedded_x_BRiGX)

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
    n_out: int,
    cache_trainset_representation: bool = False,
) -> TabPFNV2p6:
    """Construct TabPFNV2.6 based on the given config.

    This factory method implements the interface defined in
    tabpfn.architectures.interface.ArchitectureModule.get_architecture().

    Args:
        config: The config returned by parse_config(). This method should use a
            runtime isinstance() check to downcast the config to this architecture's
            specific config class.
        n_out: The number of output classes that the model should predict.
        cache_trainset_representation: If True, the model should be configured to
            cache the training data during inference to improve speed.

    Returns: the constructed architecture
    """
    assert isinstance(config, TabPFNV2p6Config)
    if cache_trainset_representation:
        raise NotImplementedError("TabPFNV2.6 does not support kv cache yet.")
    return TabPFNV2p6(
        config=config,
        n_out=n_out,
        task_type="multiclass" if config.max_num_classes > 2 else "regression",
    )


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


def _remove_constant_features(x_RiBC: torch.Tensor) -> torch.Tensor:
    """Removes constant features from the input data."""
    if x_RiBC.shape[0] <= 1:
        return x_RiBC
    column_selection_mask = ~(x_RiBC[1:] == x_RiBC[0]).all(0)
    return select_features(x_RiBC, column_selection_mask.type(torch.bool))


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


def _impute_nan_and_inf_with_mean(x: torch.Tensor, num_train_rows: int) -> torch.Tensor:
    """Impute the nan and inf with the mean of the feature.

    Args:
        x: Tensor of shape [R, ...], where
            R = number of rows
        num_train_rows: The position to use for single evaluation.
    """
    feature_means = torch_nanmean(x=x[:num_train_rows], axis=0, include_inf=True)
    nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
    return torch.where(nan_mask, feature_means.unsqueeze(0).expand_as(x), x)


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
        return _impute_nan_and_inf_with_mean(x=y_RiB1, num_train_rows=num_train_rows)

    # The following class imputation is performed for backwards compatibility.
    # We impute the mean and then do a ceil() operation.
    # This is what happened in the previously used "encoder pipeline"
    # that ran a pipeline of NanHandlingEncoderStep and
    # MulticlassClassificationTargetEncoderStep.
    y_RiB1 = _impute_nan_and_inf_with_mean(x=y_RiB1, num_train_rows=num_train_rows)
    return y_RiB1.ceil()


def _normalize_feature_groups(
    x_RiBF: torch.Tensor,
    num_features_per_group: int,
) -> torch.Tensor:
    """Normalize the feature groups."""
    Ri = x_RiBF.shape[0]
    non_constant_mask = (x_RiBF[1:] == x_RiBF[0]).sum(0) != (Ri - 1)
    number_of_used_features = torch.clip(
        non_constant_mask.sum(-1).unsqueeze(-1),
        min=1,
    ).to(x_RiBF.device)
    scale = num_features_per_group / number_of_used_features
    x_RiBF = x_RiBF * torch.sqrt(scale)

    return torch.where(
        non_constant_mask.unsqueeze(0).expand_as(x_RiBF),
        x_RiBF,
        torch.zeros_like(x_RiBF),
    )


def _generate_nan_and_inf_indicator(x: torch.Tensor) -> torch.Tensor:
    """Generate the nan and inf indicators for a generic input tensor."""
    return (
        torch.isnan(x) * NAN_INDICATOR
        + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * INFINITY_INDICATOR
        + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
        * NEG_INFINITY_INDICATOR
    ).to(x.dtype)
