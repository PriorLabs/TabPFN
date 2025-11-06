from __future__ import annotations

import torch

from tabpfn.architectures.base.thinking_tokens import AddThinkingTokens


def test__forward__output_has_correct_shape() -> None:
    emsize = 8
    module = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)

    batch_size = 2
    rows = 10
    features = 3
    embedded_input = torch.randn(batch_size, rows, features, emsize)
    single_eval_pos = 7

    output, new_single_eval_pos = module(embedded_input, single_eval_pos)

    assert output.shape == (
        batch_size,
        15,  # rows + num_thinking_rows
        features,
        emsize,
    )
    assert new_single_eval_pos == 12  # original + num_thinking_rows


def test__forward__tokens_equal_for_each_feature() -> None:
    emsize = 8
    module = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)

    batch_size = 2
    n_rows = 10
    n_features = 3
    embedded_input = torch.randn(batch_size, n_rows, n_features, emsize)
    single_eval_pos = 7

    output, _ = module(embedded_input, single_eval_pos)

    assert output[0, 0, 0, 0] == output[0, 0, 1, 0]
    assert output[0, 0, 0, 0] == output[0, 0, 2, 0]
    assert output[0, 1, 0, 0] == output[0, 1, 1, 0]
    assert output[0, 1, 0, 0] == output[0, 1, 2, 0]


def test__forward__tokens_different_for_each_row() -> None:
    emsize = 8
    module = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)

    batch_size = 2
    n_rows = 3
    n_features = 3
    embedded_input = torch.randn(batch_size, n_rows, n_features, emsize)
    single_eval_pos = 7

    output, _ = module(embedded_input, single_eval_pos)

    assert not torch.allclose(output[0, 0, 0, 0], output[0, 1, 0, 0])
    assert not torch.allclose(output[0, 0, 0, 0], output[0, 2, 0, 0])
    assert not torch.allclose(output[0, 0, 0, 0], output[0, 1, 0, 1])
    assert not torch.allclose(output[0, 0, 0, 0], output[0, 2, 0, 1])


def test__save_and_load__output_has_same_value() -> None:
    emsize = 16
    embedded_input = torch.randn(2, 10, 3, emsize)
    single_eval_pos = 7

    module_1 = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)
    module_2 = AddThinkingTokens(emsize=emsize, num_thinking_rows=5)

    output_1, new_pos_1 = module_1(embedded_input, single_eval_pos)
    state = module_1.state_dict()
    module_2.load_state_dict(state)
    output_2, new_pos_2 = module_2(embedded_input, single_eval_pos)

    assert new_pos_1 == new_pos_2
    assert torch.allclose(output_1, output_2)
