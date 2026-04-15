"""Out-of-fold target encoding for categorical features.

Adapted from Autogluon's OOFTargetEncodingFeatureGenerator:
https://github.com/autogluon/autogluon/blob/master/features/src/autogluon/features/generators/oof_target_encoder.py

Encodes each category as a smoothed conditional target statistic using K-fold
cross-validation on training data to prevent target leakage.

Based on the OOF target encoding approach from Autogluon, adapted to the TabPFN
preprocessing pipeline interface.
"""

from __future__ import annotations

from typing import Literal
from typing_extensions import override

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
    PreprocessingStepResult,
)
from tabpfn.utils import infer_random_state


class TargetEncodingStep(PreprocessingStep):
    """Target-encode categorical features using out-of-fold predictions.

    For binary classification and regression, each category is encoded as a single
    smoothed target statistic (E[y | feature=category]).

    For multiclass classification the encoding strategy is controlled by
    ``multiclass_strategy``:

    * ``"ordinal"`` (default) — treat class labels as ordinal integers and
      compute E[y | feature=category].  Produces **1 column per categorical**,
      same as binary/regression.
    * ``"per_class"`` — compute per-class conditional probabilities
      P(y=c | feature=category) for every class.  Produces **n_classes columns
      per categorical**.

    Training data is encoded using K-fold OOF predictions to prevent target leakage.
    Test data is encoded using full training-set statistics.

    This is a bare step (no modality registration) because when
    ``multiclass_strategy="per_class"`` and ``duplicate_features=False``, the
    output column count differs from the input.

    Parameters
    ----------
    task_type : {"classification", "regression"}
        Whether the target is a classification or regression target. Binary vs
        multiclass is inferred from ``y`` during fit.
    n_folds : int, default=5
        Number of cross-validation folds for OOF encoding.
    smoothing : float, default=10.0
        Bayesian smoothing parameter.  Higher values shrink category encodings
        toward the global prior more aggressively.
    multiclass_strategy : {"ordinal", "per_class"}, default="ordinal"
        How to encode multiclass targets.  ``"ordinal"`` produces a single
        column per categorical (treats labels as ordered integers).
        ``"per_class"`` produces one column per class per categorical.
        Ignored for binary classification and regression.
    duplicate_features : bool, default=True
        If True, keep original categorical columns unchanged and append
        target-encoded columns as new numerical features.  If False, replace
        categorical columns with target-encoded columns.
    random_state : int or np.random.Generator or None, default=None
        Random state for reproducible fold splits.
    """

    def __init__(
        self,
        *,
        task_type: Literal["classification", "regression"],
        n_folds: int = 5,
        smoothing: float = 10.0,
        multiclass_strategy: Literal["ordinal", "per_class"] = "ordinal",
        duplicate_features: bool = True,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.multiclass_strategy = multiclass_strategy
        self.duplicate_features = duplicate_features
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    @override
    def _fit(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
        *,
        y: np.ndarray | None = None,
    ) -> FeatureSchema:
        if y is None:
            raise ValueError(
                "TargetEncodingStep requires y during fit. "
                "Pass y to pipeline.fit_transform()."
            )

        self.cat_indices_ = feature_schema.indices_for(FeatureModality.CATEGORICAL)
        self.n_original_features_ = X.shape[1]

        if not self.cat_indices_:
            self.target_type_ = "none"
            self.encodings_: dict[int, dict] = {}
            self.train_encoded_: np.ndarray | None = None
            return feature_schema

        # Infer target type --------------------------------------------------
        self.target_type_, self.classes_, self.n_targets_ = self._infer_target_type(y)

        # Build target matrix Y: shape (n_samples, n_targets) ----------------
        Y = self._build_target_matrix(y)

        # Compute OOF + full-data encodings per column -----------------------
        n_samples = X.shape[0]
        _, rng = infer_random_state(self.random_state)

        if self.target_type_ in ("binary", "multiclass"):
            kf: KFold | StratifiedKFold = StratifiedKFold(
                n_splits=min(self.n_folds, n_samples),
                shuffle=True,
                random_state=int(rng.integers(0, 2**31)),
            )
        else:
            kf = KFold(
                n_splits=min(self.n_folds, n_samples),
                shuffle=True,
                random_state=int(rng.integers(0, 2**31)),
            )

        kf_splits = list(kf.split(np.zeros(n_samples), y))

        self.encodings_ = {}
        oof_blocks: list[np.ndarray] = []

        for cat_col in self.cat_indices_:
            col_values = X[:, cat_col]
            enc_info, oof_col = self._fit_column(col_values, Y, n_samples, kf_splits)
            self.encodings_[cat_col] = enc_info
            oof_blocks.append(oof_col)

        # Stack all OOF blocks: (n_samples, total_te_cols)
        self.train_encoded_ = np.concatenate(oof_blocks, axis=1)

        return self._build_output_schema(feature_schema)

    # ------------------------------------------------------------------
    # Fit-transform (bare-step override)
    # ------------------------------------------------------------------
    @override
    def fit_transform(
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
        *,
        y: np.ndarray | None = None,
    ) -> PreprocessingStepResult:
        # Reset cached validation state
        if hasattr(self, "n_added_columns_"):
            del self.n_added_columns_
        if hasattr(self, "modality_added_"):
            del self.modality_added_

        self.feature_schema_updated_ = self._fit(X, feature_schema, y=y)

        X_out = self._assemble_output(X, self.train_encoded_)
        self.train_encoded_ = None  # free memory

        return PreprocessingStepResult(
            X=X_out,
            feature_schema=self.feature_schema_updated_,
        )

    # ------------------------------------------------------------------
    # Transform (test data)
    # ------------------------------------------------------------------
    @override
    def _transform(
        self,
        X: np.ndarray,
        *,
        is_test: bool = False,
    ) -> tuple[np.ndarray, None, None]:
        if not self.cat_indices_ or self.target_type_ == "none":
            return X, None, None

        te_blocks: list[np.ndarray] = []
        for cat_col in self.cat_indices_:
            te_blocks.append(
                self._encode_column(X[:, cat_col], self.encodings_[cat_col])
            )

        te_features = np.concatenate(te_blocks, axis=1)
        return self._assemble_output(X, te_features), None, None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _infer_target_type(self, y: np.ndarray) -> tuple[str, np.ndarray | None, int]:
        """Return (target_type, classes, n_targets).

        For multiclass with ``multiclass_strategy="ordinal"``, n_targets is 1
        (labels treated as ordinal integers).  With ``"per_class"``, n_targets
        equals the number of classes.
        """
        if self.task_type == "regression":
            return "regression", None, 1

        classes = (
            np.unique(y[~np.isnan(y)])
            if np.issubdtype(y.dtype, np.floating)
            else np.unique(y)
        )
        n_classes = len(classes)
        if n_classes == 2:
            return "binary", classes, 1
        if self.multiclass_strategy == "ordinal":
            return "multiclass", classes, 1
        return "multiclass", classes, n_classes

    def _build_target_matrix(self, y: np.ndarray) -> np.ndarray:
        """Convert y to a target matrix of shape (n_samples, n_targets)."""
        if self.target_type_ == "regression":
            return y.astype(float).reshape(-1, 1)
        elif self.target_type_ == "binary":
            return (y == self.classes_[-1]).astype(float).reshape(-1, 1)
        elif self.multiclass_strategy == "ordinal":
            # Treat class labels as ordinal integers → single column
            return y.astype(float).reshape(-1, 1)
        else:  # multiclass per_class
            return (y[:, None] == self.classes_[None, :]).astype(float)

    def _fit_column(
        self,
        col_values: np.ndarray,
        Y: np.ndarray,
        n_samples: int,
        kf_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[dict, np.ndarray]:
        """Fit encoding for a single categorical column.

        Returns (encoding_info_dict, oof_array_of_shape_(n_samples, n_targets)).
        """
        # Map values to integer codes; NaN → -1
        categories, codes = self._factorize(col_values)
        n_cat = len(categories)
        mask_valid = codes >= 0
        codes_valid = codes[mask_valid]
        Y_valid = Y[mask_valid]

        # ---- Full-data statistics (for test-time encoding) ----
        count_all = np.bincount(codes_valid, minlength=n_cat).astype(float)
        sum_all = np.vstack(
            [
                np.bincount(codes_valid, weights=Y_valid[:, j], minlength=n_cat)
                for j in range(self.n_targets_)
            ]
        ).T  # (n_cat, n_targets)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_all = sum_all / count_all[:, None]

        # global_mean: mean of per-category means (Autogluon convention)
        global_mean = np.nanmean(mean_all, axis=0)  # (n_targets,)

        denom_all = count_all[:, None] + self.smoothing
        num_all = mean_all * count_all[:, None] + self.smoothing * global_mean[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            enc_all = num_all / denom_all  # (n_cat, n_targets)

        enc_info = {
            "categories": categories,
            "enc_matrix": enc_all.astype(float, copy=False),
            "global_mean": global_mean.astype(float, copy=False),
        }

        # ---- OOF encodings ----
        oof = np.zeros((n_samples, self.n_targets_), dtype=float)

        for tr_idx, val_idx in kf_splits:
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[val_idx] = True
            tr_mask_valid = mask_valid & ~val_mask

            codes_tr = codes[tr_mask_valid]
            Y_tr = Y[tr_mask_valid]

            if codes_tr.size == 0:
                oof[val_idx, :] = global_mean[None, :]
                continue

            count_tr = np.bincount(codes_tr, minlength=n_cat).astype(float)
            sum_tr = np.vstack(
                [
                    np.bincount(codes_tr, weights=Y_tr[:, j], minlength=n_cat)
                    for j in range(self.n_targets_)
                ]
            ).T

            with np.errstate(divide="ignore", invalid="ignore"):
                mean_tr = sum_tr / count_tr[:, None]

            valid_cats = count_tr > 0
            m_mean = np.where(valid_cats[:, None], mean_tr, np.nan)
            m_mean = np.nanmean(m_mean, axis=0)  # (n_targets,)

            denom = count_tr[:, None] + self.smoothing
            num = mean_tr * count_tr[:, None] + self.smoothing * m_mean[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                enc_tr = num / denom

            enc_tr[~valid_cats, :] = m_mean

            # Assign encodings to OOF for this fold's validation set
            enc_val = np.full((len(val_idx), self.n_targets_), 0.0, dtype=float)
            enc_val[:] = m_mean[None, :]

            val_codes = codes[val_idx]
            non_nan_mask = val_codes >= 0
            if np.any(non_nan_mask):
                enc_val[non_nan_mask, :] = enc_tr[val_codes[non_nan_mask]]

            oof[val_idx, :] = enc_val

        return enc_info, oof

    def _factorize(self, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map column values to integer codes. NaN values get code -1.

        Returns (unique_categories, codes_array).
        """
        nan_mask = (
            np.isnan(col)
            if np.issubdtype(col.dtype, np.floating)
            else np.zeros(len(col), dtype=bool)
        )
        valid_vals = col[~nan_mask]

        categories = np.unique(valid_vals)
        # Build a mapping: category_value → index
        cat_to_code = {v: i for i, v in enumerate(categories)}

        codes = np.full(len(col), -1, dtype=np.intp)
        for i, val in enumerate(col):
            if not nan_mask[i]:
                codes[i] = cat_to_code.get(val, -1)

        return categories, codes

    def _encode_column(self, col_values: np.ndarray, enc_info: dict) -> np.ndarray:
        """Encode a column using stored full-data statistics."""
        categories = enc_info["categories"]
        enc_matrix = enc_info["enc_matrix"]  # (n_cat, n_targets)
        global_mean = enc_info["global_mean"]  # (n_targets,)

        n_rows = len(col_values)
        n_targets = enc_matrix.shape[1]

        result = np.full((n_rows, n_targets), 0.0, dtype=float)
        result[:] = global_mean[None, :]

        # Map values to category codes using searchsorted
        nan_mask = (
            np.isnan(col_values)
            if np.issubdtype(col_values.dtype, np.floating)
            else np.zeros(n_rows, dtype=bool)
        )
        valid_mask = ~nan_mask

        if np.any(valid_mask):
            valid_vals = col_values[valid_mask]
            # searchsorted to find positions in sorted categories array
            positions = np.searchsorted(categories, valid_vals)
            # Check for exact matches (unseen categories won't match)
            in_bounds = positions < len(categories)
            exact_match = np.zeros(len(valid_vals), dtype=bool)
            exact_match[in_bounds] = (
                categories[positions[in_bounds]] == valid_vals[in_bounds]
            )

            # Apply encodings for known categories
            known_idx = np.where(valid_mask)[0][exact_match]
            known_positions = positions[exact_match]
            result[known_idx] = enc_matrix[known_positions]

        return result

    def _assemble_output(
        self, X: np.ndarray, te_features: np.ndarray | None
    ) -> np.ndarray:
        """Combine original features with target-encoded features."""
        if te_features is None or not self.cat_indices_:
            return X

        if self.duplicate_features:
            return np.concatenate([X, te_features], axis=1)
        else:
            non_cat_indices = [
                i for i in range(X.shape[1]) if i not in self.cat_indices_
            ]
            X_non_cat = (
                X[:, non_cat_indices]
                if non_cat_indices
                else np.empty((X.shape[0], 0), dtype=X.dtype)
            )
            return np.concatenate([X_non_cat, te_features], axis=1)

    def _build_output_schema(self, feature_schema: FeatureSchema) -> FeatureSchema:
        """Build the output feature schema."""
        total_te_cols = len(self.cat_indices_) * self.n_targets_

        if self.duplicate_features:
            return feature_schema.append_columns(
                FeatureModality.NUMERICAL, total_te_cols
            )
        else:
            new_schema = feature_schema.remove_columns(self.cat_indices_)
            return new_schema.append_columns(FeatureModality.NUMERICAL, total_te_cols)

    @override
    def num_added_features(self, n_samples: int, feature_schema: FeatureSchema) -> int:
        cat_count = len(feature_schema.indices_for(FeatureModality.CATEGORICAL))
        if cat_count == 0:
            return 0
        # Conservative estimate: 1 TE column per categorical (binary/regression).
        # Multiclass creates more, but we don't know n_classes yet.
        if self.duplicate_features:
            return cat_count
        return 0

    @override
    def has_data_dependent_feature_expansion(self) -> bool:
        # Only per_class multiclass has data-dependent column count.
        return (
            self.task_type == "classification"
            and self.multiclass_strategy == "per_class"
        )


__all__ = [
    "TargetEncodingStep",
]
