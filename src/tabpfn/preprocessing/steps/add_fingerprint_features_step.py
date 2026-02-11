"""Add Fingerprint Features Step."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing_extensions import override

import numpy as np
import torch

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
)

_CONSTANT = 2**64 - 1  # Use this to efficiently compute modulo 2**64
_MAX_COLLISION_RETRIES = 100

# Round to 12 decimal places before hashing to absorb floating-point
# noise (~1e-16) that prior preprocessing steps may introduce between
# batch and single-sample transforms.
_HASH_ROUND_DECIMALS = 12


def _float_hash_arr(arr: np.ndarray, offset: int = 0) -> float:
    data = np.around(arr, decimals=_HASH_ROUND_DECIMALS).tobytes()
    if offset != 0:
        # Append offset as raw bytes (not numeric addition) to avoid precision issues
        data += offset.to_bytes(8, "little", signed=False)
    _hash = int(hashlib.sha256(data).hexdigest(), 16)
    return (_hash & _CONSTANT) / _CONSTANT


class AddFingerprintFeaturesStep(PreprocessingStep):
    """Adds a fingerprint feature to the features based on hash of each row.

    If `is_test = True`, it keeps the first hash even if there are collisions.
    If `is_test = False`, it handles hash collisions by counting up and rehashing
    until a unique hash is found.

    The idea is basically to add a random feature to help the model distinguish between
    identical rows. We use hashing to make sure the result does not depend on the order
    of the rows.

    The fingerprint column is returned via `added_columns` in the result, and the
    pipeline handles concatenation. The step does NOT modify the input array.
    """

    def __init__(self):
        super().__init__()
        self.added_fingerprint: np.ndarray | torch.Tensor | None = None

    @override
    def _fit(
        self,
        X: np.ndarray | torch.Tensor,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        # Store n_cells as a deterministic salt appended to every hash input.
        # This prevents the fingerprint for a given row from being the same
        # across different datasets, reducing the chance the model learns to
        # overfit on this feature.
        self.n_cells_ = X.shape[0] * X.shape[1]
        # Return input schema unchanged - pipeline handles adding fingerprint column
        return feature_schema

    @override
    def _transform(  # type: ignore
        self,
        X: np.ndarray | torch.Tensor,
        *,
        is_test: bool = False,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, FeatureModality]:
        """Transform the input and compute fingerprint.

        Args:
            X: Input array of shape (n_samples, n_features).
            is_test: If True, duplicate rows share the same fingerprint.

        Returns:
            The input X unchanged. Fingerprint is available via _get_added_columns().
        """
        X_det = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        # Compute fingerprint hash for each row
        X_h = np.zeros(X.shape[0], dtype=X_det.dtype)
        salt = self.n_cells_
        if is_test:
            # Keep the first hash even if there are collisions
            for i, row in enumerate(X_det):
                h = _float_hash_arr(row, salt)
                X_h[i] = h
        else:
            # Handle hash collisions by counting up and rehashing
            seen_hashes = set()
            # Map initial hash -> next candidate offset to avoid O(N^2) on duplicates
            hash_counter = defaultdict(int)

            for i, row in enumerate(X_det):
                # Calculate the base hash to identify the row content
                h_base = _float_hash_arr(row, salt)

                # Start checking from the last known count for this row content
                add_to_hash = hash_counter[h_base]

                h = _float_hash_arr(row, salt + add_to_hash)

                # Resolve remaining collisions (if row+k accidentally collides with
                # another row)
                retries = 0
                while h in seen_hashes and not np.isnan(row).all():
                    add_to_hash += 1
                    retries += 1
                    if retries > _MAX_COLLISION_RETRIES:
                        raise RuntimeError(
                            f"Fingerprint hash collision not resolved after "
                            f"{_MAX_COLLISION_RETRIES} retries for row {i}."
                        )
                    h = _float_hash_arr(row, salt + add_to_hash)

                X_h[i] = h
                seen_hashes.add(h)

                # Update counter so next identical row starts checking from new offset
                hash_counter[h_base] = add_to_hash + 1

        if isinstance(X, torch.Tensor):
            added_fingerprint = (
                torch.from_numpy(X_h).float().reshape(-1, 1).to(X.device)
            )
        else:
            added_fingerprint = X_h.reshape(-1, 1)

        return X, added_fingerprint, FeatureModality.NUMERICAL


__all__ = [
    "AddFingerprintFeaturesStep",
]
