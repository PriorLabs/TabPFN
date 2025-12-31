"""Add Fingerprint Features Step."""

from __future__ import annotations

import hashlib
from typing_extensions import override

import numpy as np
import torch

from tabpfn.preprocessing.steps.preprocessing_helpers import (
    FeaturePreprocessingTransformerStep,
)
from tabpfn.utils import infer_random_state

_CONSTANT = 10**12


def _float_hash_arr(arr: np.ndarray) -> float:
    _hash = int(hashlib.sha256(arr.tobytes()).hexdigest(), 16)
    return _hash % _CONSTANT / _CONSTANT


class AddFingerprintFeaturesStep(FeaturePreprocessingTransformerStep):
    """Adds a fingerprint feature to the features based on hash of each row.

    If `is_test = True`, it keeps the first hash even if there are collisions.
    If `is_test = False`, it handles hash collisions by counting up and rehashing
    until a unique hash is found.
    The idea is basically to add a random feature to help the model distinguish between
    identical rows. We use hashing to make sure the result does not depend on the order
    of the rows.
    """

    def __init__(self, random_state: int | np.random.Generator | None = None):
        super().__init__()
        self.random_state = random_state

    @override
    def _fit(
        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
    ) -> list[int]:
        _, rng = infer_random_state(self.random_state)
        self.rnd_salt_ = int(rng.integers(0, 2**16))
        return [*categorical_features]

    @override
    def _transform(  # type: ignore
        self,
        X: np.ndarray | torch.Tensor,
        *,
        is_test: bool = False,
    ) -> np.ndarray | torch.Tensor:
        X_det = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        # no detach necessary for numpy
        X_h = np.zeros(X.shape[0], dtype=X_det.dtype)
        if is_test:
            # Keep the first hash even if there are collisions
            salted_X = X_det + self.rnd_salt_
            for i, row in enumerate(salted_X):
                h = _float_hash_arr(row + self.rnd_salt_)
                X_h[i] = h
        else:
            # Handle hash collisions by counting up and rehashing
            seen_hashes = set()
            salted_X = X_det + self.rnd_salt_
            for i, row in enumerate(salted_X):
                h = _float_hash_arr(row)
                add_to_hash = 0
                while h in seen_hashes and not np.isnan(row).all():
                    add_to_hash += 1
                    h = _float_hash_arr(row + add_to_hash)
                X_h[i] = h
                seen_hashes.add(h)

        if isinstance(X, torch.Tensor):
            return torch.cat(
                [X, torch.from_numpy(X_h).float().reshape(-1, 1).to(X.device)], dim=1
            )
        else:  # noqa: RET505
            return np.concatenate([X, X_h.reshape(-1, 1)], axis=1)


__all__ = [
    "AddFingerprintFeaturesStep",
]
