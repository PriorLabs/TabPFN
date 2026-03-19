"""Function for loading the pre-generated column embeddings."""

from __future__ import annotations

from pathlib import Path

import torch


def load_column_embeddings() -> torch.Tensor:
    """Load the embeddings that are added to the columns to prevent feature collapse.

    These embeddings were originally sampled from a standard normal distribution. As
    they need to be equal between training and evaluation, we persist them to disk to
    ensure that they are consistent across platforms.

    Returns:
        The embeddings for the first 2000 columns.
        These have shape (2000 columns, 48 embedding dimensions).
    """
    col_embedding_path = Path(__file__).parent / "tabpfn_col_embedding.pt"
    return torch.load(col_embedding_path, weights_only=True)
