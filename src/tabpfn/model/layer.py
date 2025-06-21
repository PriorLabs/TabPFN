#  Copyright (c) Prior Labs GmbH 2025.

# TODO: Seems like there's a lot in this file that is over-parametrized for regular
# usage. Could likely just remove it.
from __future__ import annotations

import functools
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn
from torch.nn.modules.transformer import Module, Tensor

from tabpfn.model.attention.full_attention import MultiHeadAttention
from tabpfn.model.memory import support_save_peak_mem_factor
from tabpfn.model.mlp import MLP

if TYPE_CHECKING:
    from tabpfn.model.attention import Attention
    from tabpfn.model.config import ModelConfig

if TYPE_CHECKING:
    from tabpfn.model.config import ModelConfig

HIDDEN_SIZE_LIMIT = 512
MLP_SAVE_PEAK_MEM_FACTOR = 32


class LayerNorm(torch.nn.LayerNorm):
    """Custom LayerNorm module that supports saving peak memory factor.

    This module extends the PyTorch LayerNorm implementation to handle FP16 inputs
    efficiently and support saving peak memory factor.

    Args:
        *args: Positional arguments passed to the base LayerNorm class.
        **kwargs: Keyword arguments passed to the base LayerNorm class.
    """

    # TODO: Not sure why this needs to be wrapped with functools.wraps
    @functools.wraps(torch.nn.LayerNorm.__init__)  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any):  # type: ignore
        super().__init__(*args, **kwargs)

    @support_save_peak_mem_factor  # type: ignore
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the layer normalization.

        If the input is FP16 and the normalized shape is less than 512, the computation
        is forced to be in FP16 to improve performance.

        Args:
            x: The input tensor.

        Returns:
            The layer normalized tensor.
        """
        # this if statement has the following function:
        # torch.amp.autocast wants to run layer_norm in fp32
        # but that has very bad effects for our performance (up to 2x slower)
        # thus we force fp16, if the input to the module is fp16, which is the case
        # if autocast is used.
        # WARNING: this could lead to instabilities for higher hidden sizes (> 512),
        # thus we only do this for smaller hidden sizes

        if x.dtype == torch.float16 and sum(self.normalized_shape) < HIDDEN_SIZE_LIMIT:
            with torch.amp.autocast("cuda" if x.is_cuda else "cpu", enabled=False):
                return super().forward(x)

        return super().forward(x)

    def forward(
        self,
        input: torch.Tensor,
        *,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
    ) -> torch.Tensor:
        """Perform layer normalization on the input tensor.

        Args:
            input: The input tensor.
            allow_inplace: Whether to allow in-place operations. Default is False.
            save_peak_mem_factor: The factor to save peak memory. Default is None.

        Returns:
            The layer normalized tensor.
        """
        x = input
        input_shape = x.shape

        x = x.reshape(-1, *self.normalized_shape)
        x = self._compute(
            x,
            allow_inplace=allow_inplace,
            save_peak_mem_factor=save_peak_mem_factor,
        )
        return x.reshape(input_shape)


class PerFeatureEncoderLayer(Module):
    """Transformer encoder layer that processes each feature block separately.

    This layer consists of multi-head attention between features, multi-head
    attention between items, and feedforward neural networks (MLPs).

    It supports various configurations and optimization options.

    Args:
        dim_feedforward: The dimensionality of the feedforward network.
        activation: The activation function to use in the MLPs.
        layer_norm_eps: The epsilon value for layer normalization.
        pre_norm:
            Whether to apply layer normalization before or after the attention
            and MLPs.
        device: The device to use for the layer parameters.
        dtype: The data type to use for the layer parameters.
        second_mlp: Whether to include a second MLP in the layer.
        layer_norm_with_elementwise_affine:
            Whether to use elementwise affine parameters in layer normalization.
        zero_init: Whether to initialize the output of the MLPs to zero.
        save_peak_mem_factor:
            The factor to save peak memory, only effective with post-norm.
        attention_between_features: Whether to apply attention between feature blocks.
        d_k:
            The dimensionality of the query and key vectors.
            Default is (config.emsize // config.nhead).
        d_v:
            The dimensionality of the value vectors.
            Default is (config.emsize // config.nhead).
        precomputed_kv: Precomputed key-value pairs for attention.
    """

    __constants__: ClassVar = ["batch_first"]

    def __init__(  # noqa: PLR0913
        self,
        *,
        config: ModelConfig,
        dim_feedforward: int,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        second_mlp: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        zero_init: bool = False,
        save_peak_mem_factor: int | None = None,
        attention_between_features: bool = True,
        d_k: int | None = None,
        d_v: int | None = None,
        precomputed_kv: None | torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert config.emsize % config.nhead == 0 or (
            d_k is not None and d_v is not None
        )
        if (
            config.multiquery_item_attention_for_test_set
            and config.multiquery_item_attention
        ):
            raise ValueError(
                "Cannot use both multiquery_item_attention_for_test_set"
                "and multiquery_item_attention",
            )
        if (
            config.two_sets_of_queries
            and not config.multiquery_item_attention_for_test_set
        ):
            raise ValueError(
                "two_sets_of_queries requires multiquery_item_attention_for_test_set",
            )

        if d_k is None:
            d_k = config.emsize // config.nhead
        if d_v is None:
            d_v = config.emsize // config.nhead

        self.self_attn_between_features: Attention | None = None
        if attention_between_features:
            self.self_attn_between_features = _get_feature_attn_constructor(config)(
                d_k=d_k,
                d_v=d_v,
                device=device,
                config=config,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
            )

        if isinstance(precomputed_kv, tuple):
            precomputed_k, precomputed_v = precomputed_kv
            precomputed_kv = None
        else:
            precomputed_k = precomputed_v = None

        self.self_attn_between_items = _get_item_attn_constructor(config)(
            d_k=d_k,
            d_v=d_v,
            device=device,
            dtype=dtype,
            config=config,
            share_kv_across_n_heads=config.nhead
            if config.multiquery_item_attention
            else 1,
            initialize_output_to_zero=zero_init,
            precomputed_k=precomputed_k,
            precomputed_v=precomputed_v,
            precomputed_kv=precomputed_kv,
            two_sets_of_queries=(
                config.multiquery_item_attention_for_test_set
                and config.two_sets_of_queries
            ),
        )

        self.mlp = MLP(
            size=config.emsize,
            hidden_size=dim_feedforward,
            activation=activation,
            device=device,
            dtype=dtype,
            initialize_output_to_zero=zero_init,
            recompute=config.recompute_attn,
        )

        self.layer_norms = nn.ModuleList(
            [
                LayerNorm(
                    config.emsize,  # type: ignore
                    layer_norm_eps,
                    elementwise_affine=layer_norm_with_elementwise_affine,
                    **factory_kwargs,
                )
                for _ in range(4 if second_mlp else 3)
            ],
        )

        self.second_mlp: MLP | None = None
        if second_mlp:
            self.second_mlp = MLP(
                size=config.emsize,
                hidden_size=dim_feedforward,
                activation=activation,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=config.recompute_attn,
            )

        self.pre_norm = pre_norm
        self.save_peak_mem_factor = save_peak_mem_factor
        self.multiquery_item_attention_for_test_set = (
            config.multiquery_item_attention_for_test_set
        )
        self.two_sets_of_queries = config.two_sets_of_queries

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("save_peak_mem_factor", None)
        super().__setstate__(state)

    def forward(  # noqa: C901
        self,
        state: Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,
        att_src: Tensor | None = None,
    ) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            state:
                The transformer state passed as input to the layer of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos:
                The position from which on everything is treated as test
                set.
            cache_trainset_representation:
                Whether to cache the trainset representation.
                If single_eval_pos is set (> 0 and not None), create a cache of the
                trainset KV. This may require a lot of memory. Otherwise, use
                cached KV representations for inference.
            att_src:
                The tensor to attend to from the final layer of the encoder.
                It has a shape of
                (batch_size, num_train_items, num_feature_blocks, d_model).
                This does not work with multiquery_item_attention_for_test_set and
                cache_trainset_representation at this point.

        Returns:
            The transformer state passed through the encoder layer.
        """
        assert (
            len(state.shape) == 4
        ), "src must be of shape (batch_size, num_items, num feature blocks, d_model)"
        if single_eval_pos is None:
            single_eval_pos = 0

        save_peak_mem_factor = self.save_peak_mem_factor
        if cache_trainset_representation and not single_eval_pos:
            assert self.self_attn_between_items.has_cached_kv
            save_peak_mem_factor = None

        if att_src is not None:
            assert (
                not self.multiquery_item_attention_for_test_set
            ), "Not implemented yet."
            assert not cache_trainset_representation, "Not implemented yet."
            assert not single_eval_pos, (
                "single_eval_pos should not be set, as the train representation"
                " is in att_src"
            )

        if self.self_attn_between_features is None:
            assert not cache_trainset_representation, "Not implemented yet."
            assert state.shape[2] == 1, (
                f"One group architecture expects one feature group, "
                f"but got {state.shape[2]} feature groups."
            )

        def attn_between_features(x: torch.Tensor) -> torch.Tensor:
            assert self.self_attn_between_features is not None
            return self.self_attn_between_features(
                x,
                save_peak_mem_factor=save_peak_mem_factor,
                add_input=True,
                allow_inplace=True,
            )

        def attn_between_items(x: torch.Tensor) -> torch.Tensor:
            # we need to transpose as self attention always treats
            # dim -2 as the sequence dimension
            if self.multiquery_item_attention_for_test_set:
                if single_eval_pos < x.shape[1]:
                    new_x_test = self.self_attn_between_items(
                        x[:, single_eval_pos:].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2)
                        if single_eval_pos
                        else None,
                        save_peak_mem_factor=save_peak_mem_factor,
                        cache_kv=False,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=not single_eval_pos,
                        reuse_first_head_kv=True,
                        use_second_set_of_queries=self.two_sets_of_queries,
                    ).transpose(1, 2)
                else:
                    new_x_test = None

                if single_eval_pos:
                    new_x_train = self.self_attn_between_items(
                        x[:, :single_eval_pos].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2),
                        save_peak_mem_factor=save_peak_mem_factor,
                        cache_kv=cache_trainset_representation,
                        only_cache_first_head_kv=True,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=False,
                    ).transpose(1, 2)
                else:
                    new_x_train = None

                return torch.cat(
                    [x_ for x_ in [new_x_train, new_x_test] if x_ is not None],
                    dim=1,
                )

            attention_src_x = None
            if att_src is not None:
                attention_src_x = att_src.transpose(1, 2)
            elif single_eval_pos:
                attention_src_x = x[:, :single_eval_pos].transpose(1, 2)

            return self.self_attn_between_items(
                x.transpose(1, 2),
                attention_src_x,
                save_peak_mem_factor=save_peak_mem_factor,
                cache_kv=cache_trainset_representation and single_eval_pos,
                add_input=True,
                allow_inplace=True,
                use_cached_kv=cache_trainset_representation and not single_eval_pos,
            ).transpose(1, 2)

        mlp_save_peak_mem_factor = (
            save_peak_mem_factor * 8 if save_peak_mem_factor is not None else None
        )

        sublayers = []
        if self.self_attn_between_features is not None:
            sublayers.append(attn_between_features)
        else:
            assert state.shape[2] == 1, (
                "If there is no attention between features, the number of feature"
                " blocks must be 1."
            )

        sublayers += [
            attn_between_items,
            partial(
                self.mlp.__call__,
                save_peak_mem_factor=mlp_save_peak_mem_factor
                if (
                    mlp_save_peak_mem_factor is not None
                    and state.numel() // state.shape[-1] // mlp_save_peak_mem_factor
                )
                > 32
                else None,
                add_input=True,
                allow_inplace=True,
            ),
        ]

        if self.second_mlp is not None:
            sublayers.insert(
                1,
                partial(
                    self.second_mlp.__call__,
                    save_peak_mem_factor=mlp_save_peak_mem_factor,
                    add_input=True,
                    allow_inplace=True,
                ),
            )

        for sublayer, layer_norm in zip(sublayers, self.layer_norms):
            if self.pre_norm:
                raise AssertionError(
                    "Pre-norm implementation is wrong, as the residual should never"
                    " be layer normed here.",
                )
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=save_peak_mem_factor,
                )

            state = sublayer(state)
            if not self.pre_norm:
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=save_peak_mem_factor,
                )

        return state

    def empty_trainset_representation_cache(self) -> None:
        """Empty the trainset representation cache."""
        self.self_attn_between_items.empty_kv_cache()

        # TODO: This could be None but this just ignored that fact here.
        assert self.self_attn_between_features is not None
        self.self_attn_between_features.empty_kv_cache()  # not necessary, just in case


def _get_feature_attn_constructor(config: ModelConfig) -> type[Attention]:
    if config.feature_attention_type == "full":
        return MultiHeadAttention
    raise ValueError(f"Unknown attention type: {config.feature_attention_type}")


def _get_item_attn_constructor(config: ModelConfig) -> type[Attention]:
    if config.item_attention_type == "full":
        return MultiHeadAttention
    raise ValueError(f"Unknown attention type: {config.item_attention_type}")
