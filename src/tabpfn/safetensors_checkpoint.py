"""Utilities for loading TabPFN checkpoints stored as SafeTensors plus metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file


def _metadata_path_for_safetensors(path: Path) -> Path:
    """Return the expected sidecar metadata path for a SafeTensors checkpoint."""
    return path.with_suffix(".non_tensor_metadata.json")


def load_safetensors_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a TabPFN checkpoint from SafeTensors plus sidecar JSON metadata.

    The SafeTensors file stores tensor values. The sidecar JSON file stores
    non-tensor checkpoint metadata such as architecture name, model config,
    and inference config.

    Args:
        path: Path to the ``.safetensors`` file.

    Returns:
        A checkpoint-like dictionary compatible with TabPFN model loading.
    """
    safetensors_path = Path(path)
    metadata_path = _metadata_path_for_safetensors(safetensors_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            "SafeTensors checkpoint metadata file not found. "
            f"Expected sidecar file: {metadata_path}"
        )

    tensors = load_file(str(safetensors_path), device="cpu")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    checkpoint: dict[str, Any] = dict(metadata)
    checkpoint["state_dict"] = tensors

    return checkpoint