from __future__ import annotations

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
from tabpfn.architectures.base.config import ModelConfig


def test_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_batch = 7
    nhead = 4
    n_seq_q = 534
    n_seq_kv = 316
    embed_dim = 128
    config = ModelConfig(
        emsize=embed_dim,
        nhead=nhead,
        features_per_group=1,
        max_num_classes=1,
        remove_duplicate_features=False,
        num_buckets=1000,
        nlayers=2,
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    x_q = torch.normal(
        torch.tensor(0.0), torch.tensor(1.0), size=(n_batch, n_seq_q, embed_dim)
    )
    x_kv = torch.normal(
        torch.tensor(0.0), torch.tensor(1.0), size=(n_batch, n_seq_kv, embed_dim)
    )
    x_q = x_q.to(device, dtype)
    x_kv = x_kv.to(device, dtype)

    att_ref = torch.nn.MultiheadAttention(
        embed_dim, nhead, batch_first=True, bias=False, device=device, dtype=dtype
    )
    att_test = MultiHeadAttention(
        config=config,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        device=device,
        dtype=dtype,
    )

    att_test.load_state_dict(
        MultiHeadAttention.convert_torch_nn_multihead_attention_state_dict(
            att_ref.state_dict(), nhead
        )
    )

    y, _ = att_ref(x_q, x_kv, x_kv)
    y_ = att_test(x_q, x_kv)
    assert torch.sqrt(torch.nn.functional.mse_loss(y, y_)) < 5e-5

    x_q_ = x_q.clone()
    y__ = att_test(x_q_, x_kv, add_input=True)
    assert torch.sqrt(torch.nn.functional.mse_loss(y + x_q, y__)) < 5e-5

    x_q_ = x_q.clone()
    with torch.no_grad():
        y__ = att_test(
            x_q_, x_kv, add_input=True, allow_inplace=True, save_peak_mem_factor=7
        )
    assert torch.sqrt(torch.nn.functional.mse_loss(y + x_q, y__)) < 5e-5

    # Multiquery.
    share_kv_across_n_heads = 2
    att_multi_test = MultiHeadAttention(
        config=config,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        device=device,
        dtype=dtype,
        share_kv_across_n_heads=share_kv_across_n_heads,
    )
    w_kv = (
        att_multi_test.w_kv.unsqueeze(2)
        .expand(-1, -1, share_kv_across_n_heads, -1, -1)
        .reshape(2, nhead, embed_dim // nhead, embed_dim)
    )
    state_dict_to_load = {
        "_w_qkv": torch.cat([att_multi_test.w_q, w_kv], dim=0),
        "_w_out": att_multi_test.w_out,
    }
    att_multi_ref = MultiHeadAttention(
        config=config,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        device=device,
        dtype=dtype,
    )
    att_multi_ref.load_state_dict(state_dict_to_load)
    y = att_multi_ref(x_q, x_kv)
    y_ = att_multi_test(x_q, x_kv)
    assert torch.sqrt(torch.nn.functional.mse_loss(y, y_)) < 5e-5

    # Caching.
    att_test = MultiHeadAttention(
        config=config,
        d_k=embed_dim // nhead,
        d_v=embed_dim // nhead,
        device=device,
        dtype=dtype,
    )
    y = att_test(x_q, x_kv, cache_kv=True)
    y_ = att_test(x_q, use_cached_kv=True)
    assert torch.sqrt(torch.nn.functional.mse_loss(y, y_)) < 5e-5


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_heads", "head_dim"),
    [
        (100, 64, 8, 32),
        (1100, 16, 2, 8),  # Large batch (will be chunked)
    ],
)
def test_scaled_dot_product_attention_chunked(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int
) -> None:
    """Test that scaled_dot_product_attention_chunked is
    equivalent to torch scaled_dot_product_attention.
    """
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Test with dropout disabled for deterministic comparison
    dropout_p = 0.0

    torch.manual_seed(42)
    original_output = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    torch.manual_seed(42)
    chunked_output = MultiHeadAttention.scaled_dot_product_attention_chunked(
        q, k, v, dropout_p=dropout_p, max_batch_size=500
    )

    torch.testing.assert_close(original_output, chunked_output, rtol=1e-5, atol=1e-6)
