"""Convert a TabPFN PyTorch checkpoint to SafeTensors plus sidecar metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _json_safe(value: Any) -> Any:
    """Convert common checkpoint values into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    try:
        json.dumps(value)
        return value
    except TypeError:
        return {
            "__unsupported_type__": type(value).__name__,
            "__repr__": repr(value),
        }


def convert_checkpoint(
    input_checkpoint: Path,
    output_safetensors: Path,
    output_metadata: Path,
) -> None:
    """Convert a TabPFN checkpoint into SafeTensors plus JSON metadata."""
    checkpoint = torch.load(input_checkpoint, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint to be a dict, got {type(checkpoint).__name__}."
        )

    state_dict = checkpoint.get("state_dict")

    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a dict-valued 'state_dict'.")

    tensors = {}

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Expected all state_dict values to be tensors. "
                f"Key {key!r} has type {type(value).__name__}."
            )
        tensors[key] = value.detach().cpu().contiguous()

    metadata = {
        key: _json_safe(value)
        for key, value in checkpoint.items()
        if key != "state_dict"
    }

    output_safetensors.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    save_file(tensors, str(output_safetensors))

    with output_metadata.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Saved SafeTensors file: {output_safetensors}")
    print(f"Saved metadata file: {output_metadata}")
    print(f"Tensor count: {len(tensors)}")
    print(f"Metadata keys: {sorted(metadata)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a TabPFN .ckpt file to .safetensors plus metadata JSON."
    )
    parser.add_argument("--input-checkpoint", required=True, type=Path)
    parser.add_argument("--output-safetensors", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(
        input_checkpoint=args.input_checkpoint,
        output_safetensors=args.output_safetensors,
        output_metadata=args.output_metadata,
    )


if __name__ == "__main__":
    main()