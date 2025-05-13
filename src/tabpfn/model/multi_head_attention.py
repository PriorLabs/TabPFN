#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import math
from functools import partial
from typing_extensions import override

import torch
from torch.utils.checkpoint import checkpoint

from tabpfn.model.memory import support_save_peak_mem_factor

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_func,
        flash_attn_unpadded_kvpacked_func,
        flash_attn_unpadded_qkvpacked_func,
    )

    HAVE_FLASH_ATTN = True
except (ModuleNotFoundError, ImportError):
    HAVE_FLASH_ATTN = False


class MultiHeadAttention(torch.nn.Module):
    _input_size: int
    _output_size: int
    _nhead: int
    _nhead_kv: int
    _d_k: int
    _d_v: int
    _share_kv_across_n_heads: int
    dropout_p: float | None
    softmax_scale: float | None
    _k_cache: torch.Tensor | None
    _v_cache: torch.Tensor | None
    _kv_cache: torch.Tensor | None
    _w_q: torch.nn.Parameter | None
    _w_k: torch.nn.Parameter | None
    _w_v: torch.nn.Parameter | None
    _w_kv: torch.nn.Parameter | None
    _w_qkv: torch.nn.Parameter | None
    _w_out: torch.nn.Parameter

    @property
    def w_q(self) -> torch.nn.Parameter | None:
        return self._w_q

    @property
    def w_k(self) -> torch.nn.Parameter | None:
        return self._w_k

    @property
    def w_v(self) -> torch.nn.Parameter | None:
        return self._w_v

    @property
    def w_qkv(self) -> torch.nn.Parameter | None:
        return self._w_qkv

    @property
    def w_kv(self) -> torch.nn.Parameter | None:
        return self._w_kv

    @property
    def w_out(self) -> torch.nn.Parameter:
        return self._w_out

    @property
    def has_cached_kv(self) -> bool:
        assert (self._k_cache is None) == (self._v_cache is None)
        assert self._kv_cache is None or (
            self._k_cache is None and self._v_cache is None
        )
        return (
            self._k_cache is not None and self._v_cache is not None
        ) or self._kv_cache is not None

    def empty_kv_cache(self):
        self._k_cache = None
        self._v_cache = None
        self._kv_cache = None

    def set_parameters(
        self,
        w_out: torch.nn.Parameter,
        w_q: torch.nn.Parameter | None = None,
        w_k: torch.nn.Parameter | None = None,
        w_v: torch.nn.Parameter | None = None,
        w_kv: torch.nn.Parameter | None = None,
        w_qkv: torch.nn.Parameter | None = None,
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
    ):
        assert (precomputed_k is None) == (precomputed_v is None)
        assert (precomputed_kv is None) or (precomputed_k is None)
        assert (precomputed_kv is None and precomputed_k is None) != (
            w_qkv is None and w_kv is None and w_k is None and w_v is None
        )
        assert (w_qkv is None) != (w_q is None)
        assert (w_qkv is None) or (w_kv is None and w_k is None and w_v is None)
        assert w_kv is None or (w_k is None and w_v is None)
        assert (w_k is None) == (w_v is None)

        def assert_tensor_shape(
            tensor: torch.Tensor | None,
            expected_shape: list[int | None],
        ):
            if tensor is None:
                return
            actual_shape = tensor.size()
            err = f"Tensor {actual_shape=} does not match {expected_shape=}."
            assert len(actual_shape) == len(expected_shape), err
            for actual_dim, expected_dim in zip(actual_shape, expected_shape):
                if expected_dim is not None:
                    assert actual_dim == expected_dim, err

        assert_tensor_shape(precomputed_k, [None, None, self._nhead_kv, self._d_k])
        assert_tensor_shape(precomputed_v, [None, None, self._nhead_kv, self._d_v])
        assert_tensor_shape(precomputed_kv, [None, None, 2, self._nhead_kv, self._d_k])
        assert_tensor_shape(
            w_q,
            [
                1 + int(bool(self.two_sets_of_queries)),
                self._nhead,
                self._d_k,
                self._input_size,
            ],
        )
        assert_tensor_shape(w_k, [self._nhead_kv, self._d_k, self._input_size])
        assert_tensor_shape(w_v, [self._nhead_kv, self._d_v, self._input_size])
        assert_tensor_shape(w_kv, [2, self._nhead_kv, self._d_k, self._input_size])
        assert_tensor_shape(w_qkv, [3, self._nhead, self._d_k, self._input_size])
        assert_tensor_shape(w_out, [self._nhead, self._d_v, self._output_size])

        self.register_parameter("_w_out", w_out)
        self.register_parameter("_w_q", w_q)
        self.register_parameter("_w_k", w_k)
        self.register_parameter("_w_v", w_v)
        self.register_parameter("_w_kv", w_kv)
        self.register_parameter("_w_qkv", w_qkv)

        self.register_buffer("_k_cache", precomputed_k)
        self.register_buffer("_v_cache", precomputed_v)
        self.register_buffer("_kv_cache", precomputed_kv)

    def newly_initialized_input_weight(
        self,
        dims: list[int],
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> torch.nn.Parameter:
        assert 3 <= len(dims) <= 4  # ([stack,] nhead_, d, input_size)
        w = torch.nn.Parameter(torch.empty(*dims, device=device, dtype=dtype))
        d, input_size = dims[-2:]
        std = math.sqrt(2.0 / float(nhead * d + input_size)) * self.init_gain
        a = math.sqrt(3.0) * std
        torch.nn.init.uniform_(w, -a, a)
        return w

    def __init__(  # noqa: PLR0913
        self,
        *,
        input_size: int,
        output_size: int,
        d_k: int,
        d_v: int,
        nhead: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
        share_kv_across_n_heads: int = 1,
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
        initialize_output_to_zero: bool = False,
        precomputed_k: torch.Tensor | None = None,
        precomputed_v: torch.Tensor | None = None,
        precomputed_kv: torch.Tensor | None = None,
        recompute: bool = False,
        init_gain: float = 1.0,
        two_sets_of_queries: bool = False,
    ):
        super().__init__()
        assert nhead % share_kv_across_n_heads == 0
        self._input_size = input_size
        self._output_size = output_size
        self._d_k = d_k
        self._d_v = d_v
        self._nhead = nhead
        self._nhead_kv = nhead // share_kv_across_n_heads
        self._device = device
        self._dtype = dtype
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.recompute = recompute
        self.init_gain = init_gain
        self.two_sets_of_queries = two_sets_of_queries

        w_out = torch.nn.Parameter(
            torch.empty(nhead, d_v, output_size, device=device, dtype=dtype),
        )
        if initialize_output_to_zero:
            torch.nn.init.zeros_(w_out)
        else:
            torch.nn.init.xavier_uniform_(w_out)

        assert precomputed_k is None == precomputed_v is None
        has_precomputed_kv = precomputed_kv is not None or precomputed_k is not None
        w_q = None
        w_k = None
        w_v = None
        w_kv = None
        w_qkv = None
        if (
            d_k == d_v
            and self._nhead == self._nhead_kv
            and not has_precomputed_kv
            and not two_sets_of_queries
        ):
            w_qkv = self.newly_initialized_input_weight(
                [3, self._nhead, self._d_k, self._input_size],
                nhead=self._nhead,
                device=device,
                dtype=dtype,
            )
        else:
            w_q = self.newly_initialized_input_weight(
                [
                    1 + int(bool(two_sets_of_queries)),
                    self._nhead,
                    self._d_k,
                    self._input_size,
                ],
                nhead=self._nhead,
                device=device,
                dtype=dtype,
            )
            if not has_precomputed_kv:
                if d_k == d_v:
                    w_kv = self.newly_initialized_input_weight(
                        [2, self._nhead_kv, self._d_k, self._input_size],
                        nhead=self._nhead, # Should be self._nhead_kv for consistency? Or does init logic handle it?
                        device=device,
                        dtype=dtype,
                    )
                else:
                    w_k = self.newly_initialized_input_weight(
                        [self._nhead_kv, self._d_k, self._input_size],
                        nhead=self._nhead_kv,
                        device=device,
                        dtype=dtype,
                    )
                    w_v = self.newly_initialized_input_weight(
                        [self._nhead_kv, self._d_v, self._input_size],
                        nhead=self._nhead_kv,
                        device=device,
                        dtype=dtype,
                    )
        self.set_parameters(
            w_out,
            w_q,
            w_k,
            w_v,
            w_kv,
            w_qkv,
            precomputed_k,
            precomputed_v,
            precomputed_kv,
        )
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False) # type: ignore


    @override
    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        cache_kv: bool = False,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
        reuse_first_head_kv: bool = False,
        only_cache_first_head_kv: bool = False,
        use_cached_kv: bool = False,
        use_second_set_of_queries: bool = False,
    ):
        """X is the current hidden and has a shape of [batch, ..., seq_len, input_size].
        If keys and values are present in the cache and 'freeze_kv' is not set, they
        are obtained from there and 'x_kv' has to be None.
        Else, if 'x_kv' is not None, keys and values are obtained by applying the
        respective linear transformations to 'x_kv'.
        Else, keys and values are attained by applying the respective linear
        transformations to 'x' (self attention).
        """
        assert not (
            cache_kv and use_cached_kv
        ), "Cannot cache and use cached keys and values at the same time."
        if use_second_set_of_queries:
            assert self.two_sets_of_queries, (
                "Two sets of queries are not supported."
                "Please set 'two_sets_of_queries' to True."
            )
        # assert not x.requires_grad or ( # This assertion might be too strict if we allow caching during validation passes even if x has requires_grad=False
        #     not self.has_cached_kv and not cache_kv 
        # ), "Saving keys and values is only supported during inference (no_grad mode or x.requires_grad=False)."
        if x.requires_grad and (self.has_cached_kv or cache_kv):
             import warnings
             warnings.warn("MultiHeadAttention: KV caching is typically used during inference (no_grad mode or when input does not require gradients). Using it with inputs requiring gradients might lead to unexpected behavior if not handled carefully in the overall model design.", UserWarning, stacklevel=2)


        x, x_kv, x_shape_after_transpose = self._rearrange_inputs_to_flat_batch(x, x_kv)

        nhead_kv_to_use = 1 if reuse_first_head_kv else self._nhead_kv

        if cache_kv:
            self._k_cache = self._v_cache = self._kv_cache = None # Free memory first

            batch_size_flat, seqlen_kv_eff = (x_kv if x_kv is not None else x).shape[:2]
            
            num_kv_heads_to_cache = 1 if only_cache_first_head_kv else nhead_kv_to_use

            if self._w_kv is not None or self._w_qkv is not None: # Using combined KV projection
                self._kv_cache = torch.empty(
                    batch_size_flat,
                    seqlen_kv_eff,
                    2, # For K and V
                    num_kv_heads_to_cache,
                    self._d_k, # Assuming d_k for KV cache if combined
                    device=x.device,
                    dtype=x.dtype,
                )
            else: # Using separate K and V projections
                self._k_cache = torch.empty(
                    batch_size_flat,
                    seqlen_kv_eff,
                    num_kv_heads_to_cache,
                    self._d_k,
                    device=x.device,
                    dtype=x.dtype,
                )
                self._v_cache = torch.empty(
                    batch_size_flat,
                    seqlen_kv_eff,
                    num_kv_heads_to_cache,
                    self._d_v,
                    device=x.device,
                    dtype=x.dtype,
                )

        output: torch.Tensor = self._compute(
            x,
            x_kv,
            self._k_cache,
            self._v_cache,
            self._kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            add_input=add_input, # Note: add_input is for the decorator, not used inside _compute
            allow_inplace=allow_inplace, # Note: allow_inplace is for the decorator
            save_peak_mem_factor=save_peak_mem_factor,
            reuse_first_head_kv=reuse_first_head_kv,
            only_cache_first_head_kv=only_cache_first_head_kv, # Pass this down
            use_second_set_of_queries=use_second_set_of_queries,
        )
        return output.reshape(x_shape_after_transpose[:-1] + (self._output_size,))


    def compute_qkv(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor,
        x_kv_orig: torch.Tensor | None, # Renamed to avoid confusion with 'kv' variable
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        reuse_first_head_kv: bool,
        only_cache_first_head_kv: bool, # Added this parameter
        use_second_set_of_queries: bool,
    ) -> tuple[
        torch.Tensor, # q
        torch.Tensor, # k
        torch.Tensor, # v
        torch.Tensor | None, # kv_combined (if used)
        torch.Tensor | None, # qkv_combined (if used)
    ]:
        assert not (cache_kv and use_cached_kv)
        if reuse_first_head_kv:
            assert x is not x_kv_orig, "reuse_first_head_kv requires x and x_kv to be different (cross-attention)."
        
        x_kv_eff = x_kv_orig if x_kv_orig is not None else x

        k_out = v_out = kv_combined_out = qkv_combined_out = None
        
        if use_cached_kv:
            assert self.has_cached_kv, "Attempting to use cached KV but cache is empty."
            if kv_cache is not None:
                kv_combined_out = kv_cache
                # If only first head was cached, expand it for use if needed by all heads
                if kv_combined_out.shape[-2] == 1 and self._nhead_kv > 1 and not reuse_first_head_kv: # Ensure nhead_kv is for current use
                    kv_combined_out = kv_combined_out.expand(-1, -1, -1, self._nhead_kv, -1)
            else:
                k_out = k_cache
                v_out = v_cache
                if k_out is not None and k_out.shape[-2] == 1 and self._nhead_kv > 1 and not reuse_first_head_kv:
                    k_out = k_out.expand(-1, -1, self._nhead_kv, -1)
                if v_out is not None and v_out.shape[-2] == 1 and self._nhead_kv > 1 and not reuse_first_head_kv:
                    v_out = v_out.expand(-1, -1, self._nhead_kv, -1)


        # Project Q
        if self._w_qkv is None: # Separate Q, K, V projections
            w_q_selected = self._w_q[1] if use_second_set_of_queries and self.two_sets_of_queries and self._w_q.shape[0] > 1 else self._w_q[0]
            q_out = torch.einsum("...s,hds->...hd", x, w_q_selected)
        else: # Combined QKV projection
            # If QKV is combined, q_out will be derived from qkv_combined_out later if needed
            # Or if x is x_kv_eff and no caching, it's computed together
            pass 


        # Project K, V if not using cache
        if not use_cached_kv:
            if self._w_qkv is not None and x is x_kv_eff : # Self-attention with combined QKV
                qkv_combined_out = torch.einsum("...s,jhds->...jhd", x, self._w_qkv)
                # q_out, k_out, v_out will be sliced from this by compute_attention_heads
                q_out = qkv_combined_out[:,:,0,:,:] # Placeholder, actual slicing in compute_attention_heads
            elif self._w_kv is not None: # Separate Q, combined KV
                w_kv_to_use = self._w_kv[:, :1] if reuse_first_head_kv and self._w_kv.shape[1] > 1 else self._w_kv
                kv_combined_out = torch.einsum("...s,jhds->...jhd", x_kv_eff, w_kv_to_use)
                if reuse_first_head_kv and self._w_kv.shape[1] > 1 and kv_combined_out.shape[-2] == 1: # Check if actual heads > 1
                    kv_combined_out = kv_combined_out.expand(-1, -1, -1, self._nhead_kv, -1)
                # k_out, v_out will be sliced from this
            else: # Separate Q, K, V
                w_k_to_use = self._w_k[:1] if reuse_first_head_kv and self._w_k.shape[0] > 1 else self._w_k
                w_v_to_use = self._w_v[:1] if reuse_first_head_kv and self._w_v.shape[0] > 1 else self._w_v
                k_out = torch.einsum("...s,hds->...hd", x_kv_eff, w_k_to_use)
                v_out = torch.einsum("...s,hds->...hd", x_kv_eff, w_v_to_use)
                if reuse_first_head_kv and self._w_k.shape[0] > 1 : # Check if actual heads > 1
                    if k_out.shape[-2] == 1: k_out = k_out.expand(-1, -1, self._nhead_kv, -1)
                    if v_out.shape[-2] == 1: v_out = v_out.expand(-1, -1, self._nhead_kv, -1)
            
            if cache_kv: # Store newly computed K,V (or combined KV)
                num_kv_heads_to_cache = 1 if only_cache_first_head_kv else self._nhead_kv
                if kv_cache is not None: # Storing combined KV
                    if kv_combined_out is not None:
                         kv_cache[..., :num_kv_heads_to_cache, :] = kv_combined_out[..., :num_kv_heads_to_cache, :]
                else: # Storing separate K, V
                    if k_cache is not None and k_out is not None:
                        k_cache[..., :num_kv_heads_to_cache, :] = k_out[..., :num_kv_heads_to_cache, :]
                    if v_cache is not None and v_out is not None:
                        v_cache[..., :num_kv_heads_to_cache, :] = v_out[..., :num_kv_heads_to_cache, :]
        
        # If QKV was projected together and not using cache, q_out needs to be explicitly set for return type hint
        # (it's used implicitly by compute_attention_heads if qkv_combined_out is passed)
        if self._w_qkv is not None and qkv_combined_out is not None and q_out is None:
             q_out = qkv_combined_out[:,:,0,:,:] # This is just for type hint, not used if qkv_combined_out is passed

        return q_out, k_out, v_out, kv_combined_out, qkv_combined_out


    @support_save_peak_mem_factor # type: ignore
    def _compute(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        kv_cache: torch.Tensor | None,
        *,
        cache_kv: bool,
        use_cached_kv: bool,
        add_input: bool, # For decorator
        allow_inplace: bool, # For decorator
        save_peak_mem_factor: int | None,
        reuse_first_head_kv: bool,
        only_cache_first_head_kv: bool,
        use_second_set_of_queries: bool,
    ) -> torch.Tensor:
        """Attention computation.
        Called by 'forward', potentially on shards, once shapes have been normalized.
        """
        q, k, v, kv_combined, qkv_combined = self.compute_qkv(
            x,
            x_kv,
            k_cache,
            v_cache,
            kv_cache,
            cache_kv=cache_kv,
            use_cached_kv=use_cached_kv,
            reuse_first_head_kv=reuse_first_head_kv,
            only_cache_first_head_kv=only_cache_first_head_kv,
            use_second_set_of_queries=use_second_set_of_queries,
        )
        
        attention_head_outputs = MultiHeadAttention.compute_attention_heads(
            q, # q is always explicitly computed or passed for type hints
            k,
            v,
            kv_combined,
            qkv_combined,
            self.dropout_p,
            self.softmax_scale,
        )
        return torch.einsum(
            "...hd,hds->...s", # Applying output projection
            attention_head_outputs,
            self._w_out,
        )

    def _rearrange_inputs_to_flat_batch(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Size]:
        x_shape_original = x.shape # Keep original shape for reshaping output
        if x.ndim > 2: # if not already [batch_flat, seq_len, dim]
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
            if x_kv is not None:
                assert x_shape_original[:-2] == x_kv.shape[:-2], "Batch dimensions must match for x and x_kv"
                x_kv = x_kv.reshape(-1, x_kv.shape[-2], x_kv.shape[-1])
        return x, x_kv, x_shape_original

    @staticmethod
    def broadcast_kv_across_heads(
        kv_tensor: torch.Tensor, # Can be K, V, or one slice of combined KV
        num_total_q_heads: int, # Total number of query heads
        num_actual_kv_heads: int # Number of distinct K/V heads present in kv_tensor
    ) -> torch.Tensor:
        if num_actual_kv_heads == num_total_q_heads:
            return kv_tensor
        
        # Example kv_tensor shape: (batch_size_flat, seq_len_kv, num_actual_kv_heads, d_head)
        # We need to repeat/expand num_actual_kv_heads to num_total_q_heads
        # share_factor = num_total_q_heads // num_actual_kv_heads
        # No, this is simpler: if nhead_kv < nhead, it means GQA.
        # kv_tensor has nhead_kv. We need to make it look like nhead for scaled_dot_product_attention
        # if it doesn't support GQA natively.
        
        # This function is typically called when the underlying attention mechanism
        # (like basic einsum path or older PyTorch SDPA) doesn't natively handle GQA.
        # It expects K and V to have the same number of heads as Q.
        # kv_tensor has shape (..., num_actual_kv_heads, dim_per_head)
        
        # If num_actual_kv_heads is 1 (common for some GQA variants where KVs are shared across ALL Q heads)
        if num_actual_kv_heads == 1 and num_total_q_heads > 1:
            return kv_tensor.expand(*kv_tensor.shape[:-2], num_total_q_heads, kv_tensor.shape[-1])

        # If num_actual_kv_heads > 1 but < num_total_q_heads (Grouped Query Attention)
        # Each KV head is repeated `share_factor` times.
        if num_actual_kv_heads < num_total_q_heads:
            share_factor = num_total_q_heads // num_actual_kv_heads
            if num_total_q_heads % num_actual_kv_heads != 0:
                 raise ValueError("Total Q heads must be a multiple of actual KV heads for GQA broadcast.")
            # (..., H_kv, D) -> (..., H_kv, 1, D) -> (..., H_kv, S, D) -> (..., H_kv * S, D)
            # where H_kv * S = H_q
            return kv_tensor.unsqueeze(-2).expand(*kv_tensor.shape[:-1], share_factor, kv_tensor.shape[-1]).reshape(*kv_tensor.shape[:-2], num_total_q_heads, kv_tensor.shape[-1])
            
        return kv_tensor # Should not be reached if logic is correct, or num_actual_kv_heads == num_total_q_heads

    @staticmethod
    def compute_attention_heads(  # noqa: C901, PLR0912
        q: torch.Tensor | None, # Explicit Q, always available if qkv_combined is None
        k: torch.Tensor | None, # Explicit K, if qkv_combined and kv_combined are None
        v: torch.Tensor | None, # Explicit V, if qkv_combined and kv_combined are None
        kv_combined: torch.Tensor | None, # Combined K and V, if qkv_combined is None
        qkv_combined: torch.Tensor | None, # Combined Q, K, and V
        dropout_p: float | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        # Priority: qkv_combined > kv_combined > separate k, v
        if qkv_combined is not None:
            # Shape: (batch_flat, seq_len, 3, nhead, d_k) -> unbind to Q, K, V parts
            # Assuming Q, K, V are stacked on dim 2
            q_eff, k_eff, v_eff = qkv_combined.unbind(dim=2)
        elif kv_combined is not None:
            # Shape: (batch_flat, seq_len_kv, 2, nhead_kv, d_k) -> unbind to K, V parts
            k_eff, v_eff = kv_combined.unbind(dim=2)
            q_eff = q # q must be provided if kv_combined is used
            if q_eff is None:
                 raise ValueError("Query (q) must be provided when using combined KV (kv_combined).")
        elif k is not None and v is not None:
            k_eff, v_eff = k, v
            q_eff = q # q must be provided
            if q_eff is None:
                 raise ValueError("Query (q) must be provided when using separate K and V.")
        else:
            raise ValueError("Invalid combination of Q, K, V, KV, QKV inputs for attention.")

        batch_size_flat, seqlen_q, nhead, d_k_q = q_eff.shape
        _, seqlen_kv, nhead_kv, d_v_eff = v_eff.shape # d_v_eff is the value dim per head

        if dropout_p is None:
            dropout_p = 0.0

        # Ensure FlashAttention is only considered for CUDA devices and compatible dtypes
        use_flash_attention = (
            HAVE_FLASH_ATTN
            and q_eff.device.type == 'cuda'
            and torch.cuda.is_available() # Good to have, though device.type == 'cuda' implies it
            and q_eff.dtype == torch.float16
            and k_eff.dtype == torch.float16
            and v_eff.dtype == torch.float16
        )

        TORCH_2_ATTENTION_POSSIBLE = torch.__version__ >= "2" # scaled_dot_product_attention is XLA compatible
        USE_TORCH_2_GQA = False # Flag for PyTorch's native GQA optimization

        if TORCH_2_ATTENTION_POSSIBLE and q_eff.device.type == 'cuda' and torch.cuda.is_available():
            # GQA specific optimizations in PyTorch are typically CUDA-focused
            try:
                # Test if enable_gqa is supported, usually on CUDA for specific hardware
                _ = torch.nn.functional.scaled_dot_product_attention(
                    torch.empty(1, 1, 1, 1, device='cuda'),
                    torch.empty(1, 1, 1, 1, device='cuda'),
                    torch.empty(1, 1, 1, 1, device='cuda'),
                    enable_gqa=True, # This parameter enables GQA path if supported
                )
                TORCH_2_SUPPORTS_GQ = True
            except (TypeError, RuntimeError): # Catches if enable_gqa is not a valid kwarg
                TORCH_2_SUPPORTS_GQ = False

            if TORCH_2_SUPPORTS_GQ:
                # Check NVIDIA Compute Capability for GQA hardware acceleration (e.g., Ampere+)
                # This is a heuristic; actual PyTorch internal dispatch might be more nuanced.
                device_props = torch.cuda.get_device_properties(q_eff.device)
                nvidia_compute_capability = float(f"{device_props.major}.{device_props.minor}")
                if nvidia_compute_capability >= 8.0: # Ampere and newer support GQA more efficiently
                    USE_TORCH_2_GQA = True
        
        current_softmax_scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(d_k_q))


        if use_flash_attention:
            # FlashAttention expects inputs usually in shape (total_tokens, num_heads, head_dim)
            # and uses cu_seqlens for unpadded variable length sequences.
            # Here, assuming fixed length sequences (seqlen_q, seqlen_kv) for simplicity of example.
            # Reshaping for flash_attn_unpadded_... functions:
            q_flash = q_eff.reshape(batch_size_flat * seqlen_q, nhead, d_k_q)
            
            # Helper for cu_seqlens
            def get_cu_seqlens(bs, sl, dev):
                return torch.arange(0, (bs + 1) * sl, step=sl, dtype=torch.int32, device=dev)

            cu_seqlens_q = get_cu_seqlens(batch_size_flat, seqlen_q, q_eff.device)
            cu_seqlens_kv = get_cu_seqlens(batch_size_flat, seqlen_kv, k_eff.device)

            if qkv_combined is not None: # Using QKV packed input for FlashAttention
                # qkv_combined original: (batch_flat, seq_len, 3, nhead, d_k)
                # Needs to be (total_tokens, 3, nhead, d_k_q)
                # Assuming seqlen_q == seqlen_kv for QKV packed case.
                qkv_flash = qkv_combined.reshape(batch_size_flat * seqlen_q, 3, nhead, d_k_q)
                attention_head_outputs = flash_attn_unpadded_qkvpacked_func(
                    qkv_flash,
                    cu_seqlens_q,
                    seqlen_q, # max_seqlen_q
                    dropout_p=dropout_p,
                    softmax_scale=current_softmax_scale, 
                    causal=False,
                )
            elif kv_combined is not None: # Using KV packed input for FlashAttention
                # kv_combined original: (batch_flat, seq_len_kv, 2, nhead_kv, d_k)
                # Needs to be (total_tokens_kv, 2, nhead_kv_broadcasted_to_q, d_k_q)
                # FlashAttention's kvpacked expects K and V to have same num_heads as Q after broadcast
                kv_flash = MultiHeadAttention.broadcast_kv_across_heads(kv_combined, nhead, nhead_kv)
                kv_flash = kv_flash.reshape(batch_size_flat * seqlen_kv, 2, nhead, d_k_q)
                attention_head_outputs = flash_attn_unpadded_kvpacked_func(
                    q_flash,
                    kv_flash,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    seqlen_q, # max_seqlen_q
                    seqlen_kv, # max_seqlen_kv
                    dropout_p=dropout_p,
                    softmax_scale=current_softmax_scale,
                    causal=False,
                )
            else: # Separate Q, K, V for FlashAttention
                # Reshape K and V for flash_attn_unpadded_func
                # K, V need to be broadcasted to match Q's head count if GQA
                k_flash = MultiHeadAttention.broadcast_kv_across_heads(k_eff, nhead, nhead_kv)
                v_flash = MultiHeadAttention.broadcast_kv_across_heads(v_eff, nhead, nhead_kv) # v_eff has d_v_eff
                
                k_flash = k_flash.reshape(batch_size_flat * seqlen_kv, nhead, d_k_q)
                v_flash = v_flash.reshape(batch_size_flat * seqlen_kv, nhead, d_v_eff)

                attention_head_outputs = flash_attn_unpadded_func(
                    q_flash,
                    k_flash,
                    v_flash,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    seqlen_q, # max_seqlen_q
                    seqlen_kv, # max_seqlen_kv
                    dropout_p=dropout_p,
                    softmax_scale=current_softmax_scale,
                    causal=False,
                )
            # Reshape output back to (batch_flat, seqlen_q, nhead, d_v_eff)
            attention_head_outputs = attention_head_outputs.reshape(batch_size_flat, seqlen_q, nhead, d_v_eff)

        elif TORCH_2_ATTENTION_POSSIBLE:
            # PyTorch 2.0 scaled_dot_product_attention
            # Expects (batch, num_heads, seq_len, dim_per_head)
            q_sdpa = q_eff.transpose(1, 2) 
            k_sdpa = k_eff.transpose(1, 2)
            v_sdpa = v_eff.transpose(1, 2)
            
            extra_sdpa_kwargs = {}
            if current_softmax_scale != (1.0 / math.sqrt(d_k_q)): # Only pass scale if not default
                 extra_sdpa_kwargs["scale"] = current_softmax_scale

            if nhead_kv < nhead and USE_TORCH_2_GQA: # Native GQA support
                extra_sdpa_kwargs["enable_gqa"] = True
            elif nhead_kv < nhead: # Manual broadcast for GQA if native not used/available
                k_sdpa = MultiHeadAttention.broadcast_kv_across_heads(k_eff, nhead, nhead_kv).transpose(1,2)
                v_sdpa = MultiHeadAttention.broadcast_kv_across_heads(v_eff, nhead, nhead_kv).transpose(1,2)
            
            attention_head_outputs = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=dropout_p if dropout_p > 0 else 0.0, # Pass 0.0 if None, SDPA expects float
                is_causal=False, # Assuming not causal for TabPFN context
                **extra_sdpa_kwargs
            )
            attention_head_outputs = attention_head_outputs.transpose(1, 2) # Back to (batch_flat, seqlen_q, nhead, d_v_eff)
        else:
            # Manual fallback implementation (e.g., for CPU, older PyTorch, or non-CUDA XLA)
            k_manual = MultiHeadAttention.broadcast_kv_across_heads(k_eff, nhead, nhead_kv)
            v_manual = MultiHeadAttention.broadcast_kv_across_heads(v_eff, nhead, nhead_kv)

            logits = torch.einsum("bqhd,bkhd->bqkh", q_eff, k_manual) # QK^T
            logits = logits * current_softmax_scale
            
            attn_weights = torch.softmax(logits, dim=-1)
            if dropout_p > 0.0:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=True) #
            
            attention_head_outputs = torch.einsum("bqkh,bkhd->bqhd", attn_weights, v_manual) # Multiply by V

        # Final reshape to match expected output format if it was altered by attention mechanism
        # (batch_size_flat, seqlen_q, nhead, d_v_eff)
        return attention_head_outputs.reshape(batch_size_flat, seqlen_q, nhead, d_v_eff)


    @staticmethod
    def convert_torch_nn_multihead_attention_state_dict(
        state_dict: dict,
        nhead: int,
        *,
        disable_stacked_w_qkv: bool = False,
    ) -> dict:
        in_proj_weight = state_dict["in_proj_weight"]
        out_proj_weight = state_dict["out_proj.weight"]

        embed_dim = in_proj_weight.shape[1]
        assert embed_dim % nhead == 0
        assert in_proj_weight.shape[0] == 3 * embed_dim
        assert out_proj_weight.shape == (embed_dim, embed_dim)
        
        # Assuming in_proj_weight is stacked as (Q_dim + K_dim + V_dim, embed_dim)
        # And Q_dim = K_dim = V_dim = embed_dim for standard nn.MultiheadAttention
        # It's stacked as [W_q^T, W_k^T, W_v^T] column-wise if bias is used, or row-wise if no bias...
        # PyTorch's nn.MultiheadAttention `in_proj_weight` is (3 * embed_dim, embed_dim)
        # It internally splits it into three parts for Q, K, V.
        # W_q: (embed_dim, embed_dim), W_k: (embed_dim, embed_dim), W_v: (embed_dim, embed_dim)
        # These are then applied to the input.
        # Our format for _w_qkv is (3, nhead, d_head, input_size) where input_size = embed_dim, d_head = embed_dim / nhead

        # Reshape out_proj_weight: (embed_dim, embed_dim) -> (nhead, d_v_head, output_size)
        # output_size is embed_dim, d_v_head = embed_dim / nhead
        # So, (embed_dim, embed_dim) -> transpose -> (embed_dim, embed_dim) -> reshape -> (nhead, embed_dim/nhead, embed_dim)
        new_state_dict = {}
        new_state_dict["_w_out"] = out_proj_weight.T.reshape(nhead, embed_dim // nhead, embed_dim)

        # Reshape in_proj_weight: (3 * embed_dim, embed_dim)
        # Split into q, k, v weights, each (embed_dim, embed_dim)
        wq, wk, wv = torch.split(in_proj_weight, embed_dim, dim=0)

        if disable_stacked_w_qkv: # For separate _w_q, _w_k, _w_v or _w_q, _w_kv
            # _w_q: (1, nhead, d_k_head, input_size)
            new_state_dict["_w_q"] = wq.reshape(1, nhead, embed_dim // nhead, embed_dim)
            
            # Assuming d_k == d_v for this conversion from standard MHA
            # _w_kv: (2, nhead_kv, d_k_head, input_size)
            # Here, nhead_kv = nhead for standard MHA
            wk_reshaped = wk.reshape(nhead, embed_dim//nhead, embed_dim)
            wv_reshaped = wv.reshape(nhead, embed_dim//nhead, embed_dim)
            new_state_dict["_w_kv"] = torch.stack([wk_reshaped, wv_reshaped], dim=0)
        else: # For combined _w_qkv
            # _w_qkv: (3, nhead, d_k_head, input_size)
            wq_r = wq.reshape(nhead, embed_dim // nhead, embed_dim)
            wk_r = wk.reshape(nhead, embed_dim // nhead, embed_dim)
            wv_r = wv.reshape(nhead, embed_dim // nhead, embed_dim)
            new_state_dict["_w_qkv"] = torch.stack([wq_r, wk_r, wv_r], dim=0)
            
        return new_state_dict