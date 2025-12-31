"""Shuffle Features Step."""

from __future__ import annotations

from typing import Literal
from typing_extensions import override

import numpy as np
import torch

from tabpfn.preprocessing.steps.preprocessing_helpers import (
    FeaturePreprocessingTransformerStep,
)
from tabpfn.utils import infer_random_state


class ShuffleFeaturesStep(FeaturePreprocessingTransformerStep):
    """Shuffle the features in the data."""

    def __init__(
        self,
        shuffle_method: Literal["shuffle", "rotate"] | None = "rotate",
        shuffle_index: int = 0,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.random_state = random_state
        self.shuffle_method = shuffle_method
        self.shuffle_index = shuffle_index

        self.index_permutation_: list[int] | torch.Tensor | None = None

    @override
    def _fit(
        self,
        X: np.ndarray | torch.Tensor,
        categorical_features: list[int],
    ) -> list[int]:
        _, rng = infer_random_state(self.random_state)
        if self.shuffle_method == "rotate":
            index_permutation = np.roll(
                np.arange(X.shape[1]),
                self.shuffle_index,
            ).tolist()
        elif self.shuffle_method == "shuffle":
            index_permutation = rng.permutation(X.shape[1]).tolist()
        elif self.shuffle_method is None:
            index_permutation = np.arange(X.shape[1]).tolist()
        else:
            raise ValueError(f"Unknown shuffle method {self.shuffle_method}")
        if isinstance(X, torch.Tensor):
            self.index_permutation_ = torch.tensor(index_permutation, dtype=torch.long)
        else:
            self.index_permutation_ = index_permutation

        return [
            new_idx
            for new_idx, idx in enumerate(index_permutation)
            if idx in categorical_features
        ]

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert self.index_permutation_ is not None, "You must call fit first"
        assert len(self.index_permutation_) == X.shape[1], (
            "The number of features must not change after fit"
        )
        return X[:, self.index_permutation_]
