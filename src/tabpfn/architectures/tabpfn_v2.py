"""The TabPFN v2 model.

Compared to v2 as implemented by the base architecture, this version is missing the
following features which are required for public deployment:
- The base architecture loads the positional embeddings from disk, whereas this one
  generates them using a fixed random seed. Thus, their values may vary from device to
  device.
- This implementation does not support the KV cache.

Note that this version is not compatible with the original checkpoints as layers
have been renamed/restructured.

Copyright (c) Prior Labs GmbH 2025.
"""

from __future__ import annotations

import dataclasses
from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from tabpfn.architectures.base import get_encoder, get_y_encoder
from tabpfn.architectures.interface import Architecture, ArchitectureConfig
from tabpfn.architectures.shared.attention_gqa_check import gqa_is_supported
from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace

if TYPE_CHECKING:
    from tabpfn.architectures.encoders import TorchPreprocessingPipeline


@dataclasses.dataclass
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
    ) -> torch.Tensor:
        """Forward pass for attention between cells of a single column.

        Args:
            x_BcRE: The input tensor of shape (Bc, R, E), where:
                - Bc: Batch size * number of columns
                - R: Total rows (test + train).
                - E: Embedding size.
            single_eval_pos: The position from which on everything is treated as test
                set. If None, no mask is applied and all positions are attended to. If
                given, each query after single_eval_pos will only attend to positions
                before single_eval_pos.
        """
        # H: number of heads.
        # D: head dimension.
        # F: head_dimension * number of heads.
        # N: number of train points = single_eval_pos
        # M: number of test points
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
            with torch.amp.autocast("cuda" if input.is_cuda else "cpu", enabled=False):
                return super().forward(input)

        return super().forward(input)


def prepare_targets(
    y: dict[str, torch.Tensor], num_rows: int, batch_size: int
) -> dict[str, torch.Tensor]:
    """Verify target shapes and nan-pad to the number of rows in x if necessary.

    More specifically, we ensure that:
    1. there are test rows (i.e. y["main"] is shorter than num_rows),
    2. other keys in y are of length num_train_labels or num_rows,
    3. all keys are nan-padded to the number of rows in x["main"].

    Args:
        y: A dictionary containing the target data, where each key corresponds to a
            different target variable. Must contain the key "main".
        num_rows: The number of rows in x["main"].
        batch_size: The batch size of the input data.

    Returns:
        A dictionary with the same keys as `y`, but with all tensors nan-padded to
        the number of rows in `x["main"]`.
    """
    num_train_labels = y["main"].shape[0]

    # Check that the number of training labels is not greater than
    # the total number of rows.
    # Note: we allow `num_train_labels == num_rows` (i.e., no test data) to support
    # use cases like KV-caching and for consistency with the OOM check script
    # (`src/fomo_fitting/scripts/check_oom.py`).
    if num_train_labels > num_rows:
        raise ValueError("No test rows provided.")

    prepared_y = {}
    for key, target in y.items():
        target_rows = target.shape[0]
        if target_rows not in (num_train_labels, num_rows):
            raise ValueError(
                f"y[{key}] must have either {num_train_labels} or "
                f"{num_rows} rows, but has {target_rows}."
            )
        # Make sure the target is 3-dimensional.
        target_RBY = target.view(target_rows, 1 if target.ndim == 1 else batch_size, -1)
        # Pad the rows to the number of rows in x["main"].
        target_RBY = torch.nn.functional.pad(
            target_RBY, (0, 0, 0, 0, 0, num_rows - target_rows), value=float("nan")
        )
        prepared_y[key] = target_RBY

    return prepared_y


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


class TabPFNV2(Architecture):
    """TabPFN V2 with post-layernorm and self-attention on test-items."""

    def __init__(
        self,
        *,
        config: TabPFNV2Config,
        encoder: TorchPreprocessingPipeline,
        y_encoder: TorchPreprocessingPipeline,
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
            y_encoder:
                An InputEncoder, which takes a dictionary with tensors of shape
                [num_rows, batch_size, num_cols, features] and returns a single tensor
                of shape [num_rows, batch_size, num_targets].
            n_out: The number of outputs the model should produce.
            feature_positional_embedding: The positional embedding type to use.
                The  positional embedding is added to the features to help the model
                distinguish them. Currently, only "subspace" is supported.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
        """
        super().__init__()
        self.encoder = encoder
        self.y_encoder = y_encoder
        if feature_positional_embedding != "subspace":
            raise ValueError("Currently only 'subspace' is supported.")
        self.input_size = config.emsize
        self.hidden_size = self.input_size * 4
        self.features_per_group = config.features_per_group
        self.n_out = n_out
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
        self._do_encoder_nan_check = True
        # TODO(Phil): This is here to not fail the memory computation. We should make
        # this a proper API.
        self.ninp = config.emsize

    def muon_compatible_params(self) -> set[nn.Parameter]:
        """Return parameters suitable for the Muon optimizer.

        These are the 2D weight matrices inside the transformer blocks
        (attention projections and MLP layers). All other parameters
        (embeddings, output head, biases) should use AdamW.
        """
        return {p for p in self.blocks.parameters() if p.ndim == 2}

    def add_column_embeddings(self, x_BRCX: torch.Tensor) -> torch.Tensor:
        """Add a random embedding to each column to prevent feature collapse."""
        # Note: Don't use 42 as seed here. Otherwise we cannot compare this
        # implementation to the old multi-file implementation, because the
        # multi-file implementation has a special treatment for seed=42.
        generator = torch.Generator(device=x_BRCX.device).manual_seed(420)
        num_cols, encoding_size = x_BRCX.shape[2], x_BRCX.shape[3]
        embs = torch.randn(
            (num_cols, encoding_size // 4),
            device=x_BRCX.device,
            dtype=x_BRCX.dtype,
            generator=generator,
        )
        embs = self.feature_positional_embedding_embeddings(embs)
        x_BRCX += embs[None, None]

        return x_BRCX

    def forward(  # noqa: C901
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
        if isinstance(x, torch.Tensor):
            x = {"main": x}
        if isinstance(y, torch.Tensor):
            y = {"main": y}
        elif y is None:
            y = {"main": torch.zeros(0, device=x["main"].device, dtype=x["main"].dtype)}

        # TODO(Phil): Remove.
        if len(x) != 1:
            raise NotImplementedError(
                f"Reshaping for multiple keys in x not implemented yet. ({x.keys()})"
            )
        if len(y) != 1:
            raise NotImplementedError(
                f"Reshaping for multiple keys in y not implemented yet. ({y.keys()})"
            )

        x_RBC = x["main"]
        num_features = x_RBC.shape[2]  # Number of columns.
        padding = -num_features % self.features_per_group
        # C is now padded.
        x_RBC = torch.nn.functional.pad(x_RBC, (0, padding), value=0)
        num_features += padding
        # num_feature_groups corresponds to the number of columns the model will see.
        num_feature_groups = num_features // self.features_per_group
        num_rows, batch_size = x_RBC.shape[:2]
        # Fold the feature groups into the batch size for pre-processing.
        # S = batch size * number of feature groups.
        unflattened_shape = [num_feature_groups, self.features_per_group]
        x_RSF = x_RBC.unflatten(-1, unflattened_shape).flatten(1, 2)
        x["main"] = x_RSF

        num_train_labels = y["main"].shape[0]
        # All tensors in y targets now have shape (num_rows, batch_size, 1).
        y = prepare_targets(y, num_rows=num_rows, batch_size=batch_size)
        # The encoder converts the y dict to a single target tensor.
        embedded_y_RBY = self.y_encoder(y, single_eval_pos=num_train_labels)
        embedded_y_BRY = embedded_y_RBY.transpose(0, 1)

        # Unfold S = B * G into batch and feature groups.
        # G: number of feature groups (the number of columns the model will see).
        embedded_x_RSX = self.encoder(x, single_eval_pos=num_train_labels)
        embedded_x_RBGX = embedded_x_RSX.unflatten(1, [batch_size, num_feature_groups])
        embedded_x_BRGX = embedded_x_RBGX.transpose(0, 1)
        embedded_x_BRGX = self.add_column_embeddings(embedded_x_BRGX)

        # Add the targets as an additional column.
        x_BRCD = torch.cat((embedded_x_BRGX, embedded_y_BRY[:, :, None]), dim=2)
        # This check results in a graph break with torch compile, so we only run it once
        # in the beginning and then disable it.
        if self._do_encoder_nan_check:
            if torch.isnan(x_BRCD).any():
                raise ValueError(
                    "Found NaNs in the encoded x and y. Make sure to use "
                    "a NaN-handling encoder."
                )
            self._do_encoder_nan_check = False

        # This model is really heavy on memory but light on compute. On an A100,
        # we are completely CPU-bound. Using checkpointing, we can save a lot of
        # memory, which we can invest into increasing the compute via increased batch
        # size.
        for block in self.blocks:
            if force_recompute_layer:
                x_BRCD = torch.utils.checkpoint.checkpoint(
                    block, x_BRCD, num_train_labels, save_peak_memory_factor
                )
            else:
                x_BRCD = block(x_BRCD, num_train_labels, save_peak_memory_factor)

        # T: number of test samples
        # B: batch size
        # E: embedding size
        test_embeddings_TBE = x_BRCD[:, num_train_labels:, -1].transpose(0, 1)
        test_output_TB1 = self.output_projection(test_embeddings_TBE)

        if only_return_standard_out:
            return test_output_TB1

        output = {"standard": test_output_TB1}
        output["train_embeddings"] = x_BRCD[:, :num_train_labels, -1].transpose(0, 1)
        output["test_embeddings"] = test_embeddings_TBE

        return output


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


def get_architecture(
    config: ArchitectureConfig,
    *,
    n_out: int,
    cache_trainset_representation: bool = False,
) -> TabPFNV2:
    """Construct TabPFNV2 based on the given config.

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
    assert isinstance(config, TabPFNV2Config)
    if cache_trainset_representation:
        raise NotImplementedError("TabPFNV2 does not support kv cache yet.")
    return TabPFNV2(
        config=config,
        # Note: The encoders currently break torch.compile.
        encoder=get_encoder(
            num_features_per_group=config.features_per_group,
            embedding_size=config.emsize,
            remove_empty_features=True,
            remove_duplicate_features=False,
            nan_handling_enabled=True,
            normalize_on_train_only=True,
            normalize_to_ranking=False,
            normalize_x=True,
            remove_outliers=False,
            normalize_by_used_features=True,
            encoder_use_bias=False,
        ),
        y_encoder=get_y_encoder(
            num_inputs=1,
            embedding_size=config.emsize,
            nan_handling_y_encoder=True,
            max_num_classes=config.max_num_classes,
        ),
        n_out=n_out,
    )
