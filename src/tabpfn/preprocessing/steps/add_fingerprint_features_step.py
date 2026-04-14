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


def _hash_row_bytes(row_data: bytes, salt_bytes: bytes) -> float:
    """Hash pre-rounded row bytes with salt. Avoids repeated rounding."""
    _hash = int(hashlib.sha256(row_data + salt_bytes).hexdigest(), 16)
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

        # Round once for all rows instead of per-row
        X_rounded = np.ascontiguousarray(
            np.around(X_det, decimals=_HASH_ROUND_DECIMALS)
        )

        salt = self.n_cells_
        salt_bytes = salt.to_bytes(8, "little", signed=False)

        X_h = np.zeros(X.shape[0], dtype=X_det.dtype)
        if is_test:
            for i in range(X_rounded.shape[0]):
                row_data = X_rounded[i].tobytes()
                X_h[i] = _hash_row_bytes(row_data, salt_bytes)
        else:
            # Handle hash collisions by counting up and rehashing
            seen_hashes = set()
            hash_counter: dict[float, int] = defaultdict(int)

            def _hash_with_offset(row_bytes: bytes, offset: int) -> float:
                ob = (salt + offset).to_bytes(8, "little", signed=False)
                return _hash_row_bytes(row_bytes, ob)

            for i in range(X_rounded.shape[0]):
                row_data = X_rounded[i].tobytes()

                # Calculate the base hash to identify the row content
                h_base = _hash_row_bytes(row_data, salt_bytes)

                # Start checking from the last known count for this row content
                add_to_hash = hash_counter[h_base]

                h = (
                    h_base
                    if add_to_hash == 0
                    else _hash_with_offset(row_data, add_to_hash)
                )

                # Resolve remaining collisions
                retries = 0
                while h in seen_hashes and not np.isnan(X_det[i]).all():
                    add_to_hash += 1
                    retries += 1
                    if retries > _MAX_COLLISION_RETRIES:
                        raise RuntimeError(
                            f"Fingerprint hash collision not resolved after "
                            f"{_MAX_COLLISION_RETRIES} retries for row {i}."
                        )
                    h = _hash_with_offset(row_data, add_to_hash)

                X_h[i] = h
                seen_hashes.add(h)
                hash_counter[h_base] = add_to_hash + 1

        if isinstance(X, torch.Tensor):
            added_fingerprint = (
                torch.from_numpy(X_h).float().reshape(-1, 1).to(X.device)
            )
        else:
            added_fingerprint = X_h.reshape(-1, 1)

        return X, added_fingerprint, FeatureModality.NUMERICAL

    @override
    def num_added_features(self, n_samples: int, feature_schema: FeatureSchema) -> int:
        """Return the number of added features."""
        del n_samples, feature_schema
        return 1


__all__ = [
    "AddFingerprintFeaturesStep",
]
