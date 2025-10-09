"""Remove Highly Correlated Features Step."""

from __future__ import annotations

from typing_extensions import override

import numpy as np
import torch
import faiss
import torch.nn.functional as F

from tabpfn.preprocessors.preprocessing_helpers import (
    FeaturePreprocessingTransformerStep,
)


class RemoveHighlyCorrelatedFeaturesStep(FeaturePreprocessingTransformerStep):
    """Remove features that are highly correlated in the training data.
    Further improvements:
    - faiss-gpu can be used if installed dynamically based on environment
    - use approximate nearest neighbor search instead of exact ( see https://github.com/facebookresearch/faiss/wiki/Faiss-indexes )
    - introduce support for categorical features ( eg. Chi-square test based removal )
    - add parameter to choose which feature to keep among correlated ones (currently first one is kept) 
    """

    def __init__(
        self,
        *,
        threshold: float = 0.98
        ) -> None:
        super().__init__()
        self.threshold = threshold
        self.sel_: list[bool] | None = None

    @override
    def _fit(  # type: ignore
        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
    ) -> list[int]:
        x_norm = None
        
        if isinstance(X, torch.Tensor):
            x_norm = F.normalize(X, p=2, dim=0).numpy() 
        else:
            def normalize_np(X, axis=0, p=2, eps=1e-12):
                norm = np.linalg.norm(X, ord=p, axis=axis, keepdims=True)
                norm = np.maximum(norm, eps)  # prevent division by zero
                return X / norm
            x_norm = normalize_np(X, axis=0)
        
        # Transpose to treat columns as vectors: shape (n_features, n_samples)
        vectors = x_norm.T

        d = vectors.shape[1]
        
        sel_ = [True] * vectors.shape[0]

        # Build FAISS index on columns as vectors
        index = faiss.IndexFlatIP(d)
        index.add(vectors)

        indices_to_skip = set()

        for idx, vector in enumerate(vectors) :
            if idx not in indices_to_skip:
                query = vector.reshape(1, -1)
                
                negated_query = -query
                queries = np.vstack((query, negated_query))
                # Perform range search to find all columns with similarity >= threshold
                _, _, labels = index.range_search(query, thresh = self.threshold)
                _, _, labels_neg = index.range_search(negated_query, thresh = self.threshold)
                all_labels = np.concatenate((labels, labels_neg))
                similar_indices = all_labels[all_labels != idx]
                index.remove_ids(similar_indices)
                indices_to_skip.update(similar_indices.tolist())

        for idx in indices_to_skip:
            sel_[idx] = False

        self.sel_ = sel_

        # ACT: I would need some guidance here to understand how and what do we return from transformer steps.
        return [
            new_idx
            for new_idx, idx in enumerate(np.where(sel_)[0])
            if idx in categorical_features
        ]

    @override
    def _transform(
        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
    ) -> np.ndarray:
        assert self.sel_ is not None, "You must call fit first"
        X_transformed = X[:, self.sel_]
        return X_transformed
