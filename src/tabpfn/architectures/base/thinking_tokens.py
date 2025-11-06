from __future__ import annotations

from typing_extensions import override

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class AddThinkingTokens(Module):
    """Takes the embedded input and prepends "thinking tokens" to it.

    Adjusts the single_eval_pos appropriately to account for the new, longer input.

    We hope that the thinking tokens give the model more computational capacity to
    perform in-context learning, particuarly on small datasets. This is inspired by LLM
    results such as
    - Think Before You Speak, Goyal et al. 2024:
        https://openreview.net/forum?id=ph04CRkPdC
    - Exact Expressive Power of Transformers with Padding, Merrill & Sabharwal 2025:
        https://arxiv.org/abs/2505.18948
    """

    def __init__(self, num_thinking_rows: int, emsize: int) -> None:
        super().__init__()
        self.num_thinking_rows = num_thinking_rows
        # We have to work with variable numbers of features, so we use the same token
        # for each feature.
        self.row_token_values = Parameter(torch.empty(num_thinking_rows, emsize))
        self.reset_parameters()

    @override
    def forward(
        self, embedded_input: Tensor, single_eval_pos: int
    ) -> tuple[Tensor, int]:
        """Prepends the thinking tokens to the embedded input.

        Args:
            embedded_input: [batch x train+eval rows x feature groups x emsize]
            single_eval_pos: Rows after this index are treated as evaluation rows.

        Returns:
            (
                embedded_input with added rows
                    [batch size x thinking+train+eval rows x feature groups x emsize],
                updated single_eval_pos
            )
        """
        batch_size, _, num_features, _ = embedded_input.shape
        thinking_tokens_base = self.row_token_values.unsqueeze(0).unsqueeze(2)
        thinking_tokens = thinking_tokens_base.expand(batch_size, -1, num_features, -1)

        embedded_input = torch.cat([thinking_tokens, embedded_input], dim=1)
        single_eval_pos += self.num_thinking_rows
        return embedded_input, single_eval_pos

    def reset_parameters(self) -> None:
        # This is the initialisation used in torch.nn.Embedding, so hopefully a
        # reasonable choice for our application.
        torch.nn.init.normal_(self.row_token_values)
