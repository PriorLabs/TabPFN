#  Copyright (c) Prior Labs GmbH 2026.

"""Checkpoint loading for TabPFN.

Handles both ``.ckpt`` (pickled torch) and ``.safetensors`` checkpoints
behind a single :class:`Checkpoint` object that owns format detection, cache
fingerprinting, and the load itself.
"""

from __future__ import annotations

import json
import warnings
from enum import Enum
from pathlib import Path
from typing import Any
from typing_extensions import override

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class Checkpoint:
    """A TabPFN model checkpoint on disk.

    Wraps a checkpoint path and knows which format it is, how to fingerprint
    it for cache invalidation, and how to materialize it into a dict.

    Example:
        ckpt = Checkpoint("model.safetensors")
        data = ckpt.load()
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)

    @property
    def is_safetensors(self) -> bool:
        """Whether this checkpoint is stored in the safetensors format."""
        return self.path.suffix == ".safetensors"

    def identity(self) -> tuple[int, int]:
        """Cheap (mtime_ns, size) fingerprint; used as an lru_cache key."""
        st = self.path.stat()
        return (st.st_mtime_ns, st.st_size)

    def load(self) -> dict[str, Any]:
        """Load the checkpoint into a dict with ``state_dict`` and metadata."""
        if self.is_safetensors:
            return self._load_safetensors()
        return self._load_torch()

    def _load_safetensors(self) -> dict[str, Any]:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(self.path), framework="pt", device="cpu") as f:
            # Header metadata is dict[str, str]; each value is a JSON string.
            header = f.metadata() or {}
            for key in f.keys():  # noqa: SIM118 -- safe_open is not iterable
                tensors[key] = f.get_tensor(key)

        checkpoint: dict[str, Any] = {k: json.loads(v) for k, v in header.items()}
        checkpoint["state_dict"] = tensors
        return checkpoint

    def _load_torch(self) -> dict[str, Any]:
        # `torch.load` raises a FutureWarning because the default for
        # `weights_only` will flip to True and disallow arbitrary objects.
        # TabPFN checkpoints currently rely on the legacy behavior.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return torch.load(str(self.path), map_location="cpu", weights_only=None)


def save_as_safetensors(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Save a TabPFN checkpoint dict as a SafeTensors file with header metadata.

    Non-tensor fields are JSON-encoded into the safetensors header so the
    resulting file is a self-contained replacement for the legacy ``.ckpt``.
    """
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a dict-valued 'state_dict'.")

    tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Expected all state_dict values to be tensors. "
                f"Key {key!r} has type {type(value).__name__}."
            )
        tensors[key] = value.detach().cpu().contiguous()

    # safetensors header metadata is dict[str, str]; JSON-encode each value.
    metadata = {
        key: json.dumps(value, cls=_CheckpointJSONEncoder, sort_keys=True)
        for key, value in checkpoint.items()
        if key != "state_dict"
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata=metadata)


class _CheckpointJSONEncoder(json.JSONEncoder):
    """JSON encoder for TabPFN checkpoint metadata values.

    Handles types commonly found in checkpoint configs: Path, torch dtype/device,
    Enum, set. Raises ``TypeError`` on anything else rather than silently dropping data.
    """

    @override
    def default(self, o: Any) -> Any:
        """Encode types not natively handled by ``json.JSONEncoder``."""
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (torch.dtype, torch.device)):
            return str(o)
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, set):
            # key=str so heterogeneous sets (e.g. {1, "a"}) don't raise; the
            # sort keeps output deterministic across saves.
            return sorted(o, key=str)
        raise TypeError(
            f"Cannot encode value of type {type(o).__name__!r} for the checkpoint "
            f"header. Add a branch to _CheckpointJSONEncoder.default() if this "
            f"type should be supported. Value: {o!r}"
        )
