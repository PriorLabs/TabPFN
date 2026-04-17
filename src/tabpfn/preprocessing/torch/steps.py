#  Copyright (c) Prior Labs GmbH 2025.

"""Pipeline step wrappers for torch preprocessing operations."""

from __future__ import annotations

from typing import Literal
from typing_extensions import override

import numpy as np
import torch

from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.steps.add_fingerprint_features_step import (
    AddFingerprintFeaturesStep,
)
from tabpfn.preprocessing.steps.add_svd_features_step import get_svd_n_components
from tabpfn.preprocessing.torch.pipeline_interface import (
    TorchPreprocessingStep,
    TorchPreprocessingStepResult,
)
from tabpfn.preprocessing.torch.torch_quantile_transformer import (
    TorchQuantileTransformer,
)
from tabpfn.preprocessing.torch.torch_soft_clip_outliers import TorchSoftClipOutliers
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler
from tabpfn.preprocessing.torch.torch_svd import (
    TorchSafeStandardScaler,
    TorchTruncatedSVD,
)
from tabpfn.utils import infer_random_state


class TorchQuantileTransformerStep(TorchPreprocessingStep):
    """Pipeline step wrapper for TorchQuantileTransformer."""

    def __init__(self, n_quantiles: int = 1_000) -> None:
        """Initialize the quantile transformer step."""
        super().__init__()
        self._quantile_transformer = TorchQuantileTransformer(n_quantiles=n_quantiles)

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit the quantile transformer on the selected columns."""
        return self._quantile_transformer.fit(x)

    @override
    def _transform(
        self, x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Transform columns using the fitted quantile transformer."""
        return (
            self._quantile_transformer.transform(x, fitted_cache=fitted_cache),
            None,
            None,
        )


class TorchStandardScalerStep(TorchPreprocessingStep):
    """Pipeline step wrapper for TorchStandardScaler."""

    def __init__(self) -> None:
        """Initialize the standard scaler step."""
        super().__init__()
        self._scaler = TorchStandardScaler()

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit the scaler on the selected columns."""
        return self._scaler.fit(x)

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Transform columns using the fitted scaler."""
        return self._scaler.transform(x, fitted_cache=fitted_cache), None, None


class TorchSoftClipOutliersStep(TorchPreprocessingStep):
    """Pipeline step wrapper for TorchSoftClipOutliers."""

    def __init__(self, n_sigma: float = 4.0) -> None:
        """Initialize the outlier removal step.

        Args:
            n_sigma: Number of standard deviations to use for outlier threshold.
        """
        super().__init__()
        self._outlier_remover = TorchSoftClipOutliers(n_sigma=n_sigma)

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit the outlier remover on the selected columns."""
        return self._outlier_remover.fit(x)

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Transform columns using the fitted outlier remover."""
        return self._outlier_remover.transform(x, fitted_cache=fitted_cache), None, None


class TorchSelectiveQuantileTransformerStep(TorchPreprocessingStep):
    """Quantile transformer that only transforms specified column indices.

    This step needs to be registered with ``modalities=None`` in the GPU pipeline so it
    receives the full tensor. It selects only preconfigured ``target_column_indices``
    and applies the quantile transform, and writes the results back. All other columns
    are left unchanged.
    """

    def __init__(
        self,
        n_quantiles: int,
        target_column_indices: list[int],
    ) -> None:
        super().__init__()
        self._qt = TorchQuantileTransformer(n_quantiles=n_quantiles)
        self._target_column_indices = target_column_indices

    @override
    def fit_transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
        fitted_cache: dict[str, torch.Tensor] | None = None,
    ) -> TorchPreprocessingStepResult:
        """Override to select only target columns for quantile transform."""
        del column_indices
        orig_device = x.device
        target_x = x[:, :, self._target_column_indices]

        # The quantile transform uses many small ops (searchsorted, gather,
        # where) that are faster on CPU than on MPS due to per-kernel launch
        # overhead.  On CUDA the GPU is preferred.
        run_on_cpu = orig_device.type == "mps"
        if run_on_cpu:
            target_x = target_x.cpu()

        if fitted_cache is None:
            fitted_cache = self._qt.fit(target_x[:num_train_rows])

        transformed = self._qt.transform(target_x, fitted_cache=fitted_cache)

        if run_on_cpu:
            transformed = transformed.to(orig_device)

        x = x.clone()
        x[:, :, self._target_column_indices] = transformed

        return TorchPreprocessingStepResult(x=x, fitted_cache=fitted_cache)

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._qt.fit(x)

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        return self._qt.transform(x, fitted_cache=fitted_cache), None, None


class TorchShuffleFeaturesStep(TorchPreprocessingStep):
    """Shuffle features on GPU, consistent with CPU ShuffleFeaturesStep.

    This step needs to be registered with ``modalities=None`` in the GPU pipeline so it
    receives ALL columns.  It permutes the columns and returns the updated
    ``schema_permutation`` so the pipeline can update the feature schema.

    The permutation is computed using the same logic and random state as the CPU
    ``ShuffleFeaturesStep`` to guarantee identical results.
    """

    def __init__(
        self,
        shuffle_method: Literal["shuffle", "rotate"] | None,
        shuffle_index: int,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.shuffle_method = shuffle_method
        self.shuffle_index = shuffle_index
        self.random_state = random_state

    @override
    def fit_transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
        fitted_cache: dict[str, torch.Tensor] | None = None,
    ) -> TorchPreprocessingStepResult:
        """Override to apply permutation to the full tensor."""
        del column_indices
        if fitted_cache is None:
            fitted_cache = self._fit(x[:num_train_rows])

        perm = fitted_cache["permutation"]
        x_shuffled = x[:, :, perm.to(x.device)]

        return TorchPreprocessingStepResult(
            x=x_shuffled,
            fitted_cache=fitted_cache,
            schema_permutation=perm.tolist(),
        )

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the permutation identically to CPU ShuffleFeaturesStep."""
        n_cols = x.shape[-1]
        _, rng = infer_random_state(self.random_state)

        if self.shuffle_method == "rotate":
            perm = np.roll(np.arange(n_cols), self.shuffle_index).tolist()
        elif self.shuffle_method == "shuffle":
            perm = rng.permutation(n_cols).tolist()
        elif self.shuffle_method is None:
            perm = list(range(n_cols))
        else:
            raise ValueError(f"Unknown shuffle method {self.shuffle_method}")

        return {"permutation": torch.tensor(perm, dtype=torch.long)}

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        perm = fitted_cache["permutation"].to(x.device)
        return x[:, :, perm], None, None


class TorchAddSVDFeaturesStep(TorchPreprocessingStep):
    """Pipeline step that adds SVD features to the input columns.

    This step applies safe standard scaling (without mean centering) followed
    by truncated SVD, and appends the SVD components as additional numerical
    features. The original columns are left unchanged.

    This matches the behavior of the sklearn reshape_feature_distribution_step.
    """

    def __init__(
        self,
        global_transformer_name: Literal["svd", "svd_quarter_components"] = "svd",
    ) -> None:
        """Initialize the SVD features step.

        Args:
            global_transformer_name: Name of the SVD variant. The number of
                components is computed inside ``_fit`` via
                :func:`get_svd_n_components`, matching the CPU pipeline.
        """
        super().__init__()
        self.global_transformer_name = global_transformer_name
        self._scaler = TorchSafeStandardScaler()
        self._svd: TorchTruncatedSVD | None = None

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fit the scaler and SVD on the selected columns.

        Args:
            x: Tensor of selected columns [num_train_rows, batch_size, num_cols].

        Returns:
            Cache dictionary with scaler and SVD parameters.
        """
        num_train_rows = x.shape[0]
        num_features = x.shape[-1]

        effective_n_components = get_svd_n_components(
            self.global_transformer_name,
            n_samples=num_train_rows,
            n_features=num_features,
        )

        self._svd = TorchTruncatedSVD(n_components=effective_n_components)

        # Fit scaler on training data (flatten batch dimension for fitting)
        # Shape: [num_train_rows, batch_size, num_cols] -> flattened
        batch_size = x.shape[1]
        x_flat = x.reshape(-1, num_features)
        scaler_cache = self._scaler.fit(x_flat)

        x_scaled = self._scaler.transform(x_flat, scaler_cache)
        svd_cache = self._svd.fit(x_scaled)

        return {
            "scaler_std": scaler_cache["std"],
            "scaler_mean": scaler_cache["mean"],
            "svd_components": svd_cache["components"],
            "svd_singular_values": svd_cache["singular_values"],
            "batch_size": torch.tensor(batch_size),
            "n_components": torch.tensor(effective_n_components),
        }

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        """Transform columns and return original plus SVD features.

        Args:
            x: Tensor of selected columns [num_rows, batch_size, num_cols].
            fitted_cache: Cache returned by _fit.

        Returns:
            Tuple of (original_columns, svd_features, NUMERICAL modality).
        """
        num_rows, batch_size, num_features = x.shape

        # Extract caches
        scaler_cache = {
            "std": fitted_cache["scaler_std"],
            "mean": fitted_cache["scaler_mean"],
        }
        svd_cache = {
            "components": fitted_cache["svd_components"],
            "singular_values": fitted_cache["svd_singular_values"],
        }

        # Initialize SVD if needed (for transform-only calls)
        n_components = int(fitted_cache["n_components"].item())
        if self._svd is None:
            self._svd = TorchTruncatedSVD(n_components=n_components)

        # Flatten for processing
        x_flat = x.reshape(-1, num_features)

        x_scaled = self._scaler.transform(x_flat, scaler_cache)
        svd_features_flat = self._svd.transform(x_scaled, svd_cache)
        svd_features = svd_features_flat.reshape(num_rows, batch_size, -1)

        # Return original columns unchanged, with SVD features as added columns
        return x, svd_features, FeatureModality.NUMERICAL


class TorchAddFingerprintFeaturesStep(TorchPreprocessingStep):
    """Add fingerprint features inside the GPU preprocessing pipeline.

    This wraps the CPU-based :class:`AddFingerprintFeaturesStep` so the
    fingerprint is computed at the *same* position in the pipeline regardless
    of whether ``enable_gpu_preprocessing`` is on or off.  The hashing itself
    runs on CPU (SHA-256 is not GPU-acceleratable), but the data movement
    overhead is negligible for the single added column.

    The step is registered with ``modalities=None`` so it receives the full
    tensor.  Training rows (``x[:num_train_rows]``) get collision-resolved
    fingerprints; remaining rows get simple hashes.

    TODO: Implement this on GPU natively.
    """

    @override
    def fit_transform(
        self,
        x: torch.Tensor,
        column_indices: list[int],
        num_train_rows: int,
        fitted_cache: dict[str, torch.Tensor] | None = None,
    ) -> TorchPreprocessingStepResult:
        """Compute fingerprints and return them as added columns."""
        del column_indices  # operates on all columns

        n_rows, batch_size, n_cols = x.shape

        if fitted_cache is None:
            fitted_cache = {"n_cells": torch.tensor(num_train_rows * n_cols)}

        n_cells = int(fitted_cache["n_cells"].item())

        all_fingerprints = []
        for b in range(batch_size):
            x_2d = x[:, b, :]  # [rows, cols]

            step = AddFingerprintFeaturesStep()
            step.n_cells_ = n_cells

            # Train portion: collision-resolved fingerprints
            _, fp_train, _ = step._transform(x_2d[:num_train_rows], is_test=False)
            # Test portion: simple hashes
            if num_train_rows < n_rows:
                _, fp_test, _ = step._transform(x_2d[num_train_rows:], is_test=True)
                fp = torch.cat([fp_train, fp_test], dim=0)  # [rows, 1]
            else:
                fp = fp_train  # [rows, 1]

            all_fingerprints.append(fp)

        # Stack across batch: [rows, batch, 1].
        # Fingerprints are float32 from AddFingerprintFeaturesStep.  Cast to
        # input dtype for concatenation.  Note that for float16 this loses precision
        # (~1024 unique values in [0,1)) and won't provide unique identifiers for
        # training sets larger than 1k rows.
        added = torch.stack(all_fingerprints, dim=1).to(dtype=x.dtype, device=x.device)

        return TorchPreprocessingStepResult(
            x=x,
            added_columns=added,
            added_modality=FeatureModality.NUMERICAL,
            fitted_cache=fitted_cache,
        )

    @override
    def _fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n_rows = x.shape[0]
        n_cols = x.shape[-1]
        return {"n_cells": torch.tensor(n_rows * n_cols)}

    @override
    def _transform(
        self,
        x: torch.Tensor,
        fitted_cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, FeatureModality | None]:
        # Not called directly - fit_transform is overridden
        raise NotImplementedError("Use fit_transform instead")
