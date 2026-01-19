"""Encoder step to encode multiclass classification targets."""

from __future__ import annotations

from typing import Any, Callable
from typing_extensions import override

import torch

from tabpfn.architectures.encoders import TorchPreprocessingStep


class MulticlassClassificationTargetEncoderStep(TorchPreprocessingStep):
    """Encoder step to encode multiclass classification targets."""

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        assert len(in_keys) == len(out_keys) == 1, (
            f"{self.__class__.__name__} expects a single input and output key."
        )

        super().__init__(in_keys, out_keys)
        self.unique_ys_: list[torch.Tensor] | None = None

    @override
    def _apply(self, fn: Callable) -> MulticlassClassificationTargetEncoderStep:
        super()._apply(fn)
        # As unique_ys_ is a variable-length list, the easiest way to correctly move it
        # with device moves is to override the _apply function.
        if self.unique_ys_ is not None:
            self.unique_ys_ = [fn(t) for t in self.unique_ys_]
        return self

    # torch.unique breaks the graph, so we disable compilation for this method.
    @torch.compiler.disable
    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if single_eval_pos is None:
            raise ValueError(
                f"single_eval_pos must be provided for {self.__class__.__name__}"
            )

        y = state[self.in_keys[0]]
        assert len(y.shape) == 3, "y must have 3 dimensions"
        assert y.shape[-1] == 1, "y must be of shape (T, B, 1)"
        self.unique_ys_ = [
            torch.unique(y[:single_eval_pos, b_i]) for b_i in range(y.shape[1])
        ]

    @staticmethod
    def flatten_targets(
        y: torch.Tensor, unique_ys: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Flatten targets."""
        if unique_ys is None:
            unique_ys = torch.unique(y)
        return (y.unsqueeze(-1) > unique_ys).sum(axis=-1)  # type: ignore

    # data-dependent control flow in .any() breaks the graph, so we disable compilation
    # for this method.
    @torch.compiler.disable
    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        y = state[self.in_keys[0]]
        assert len(y.shape) == 3, "y must have 3 dimensions"
        assert y.shape[-1] == 1, "y must be of shape (T, B, 1)"
        assert not (y.isnan().any() and self.training), (
            "NaNs are not allowed in the target at this point during training "
            "(set to model.eval() if not in training)"
        )
        y_new = y.clone()
        for B in range(y.shape[1]):
            y_new[:, B, :] = self.flatten_targets(y[:, B, :], self.unique_ys_[B])
        return {self.out_keys[0]: y_new}
