# ruff: noqa: T201
#  Copyright (c) Prior Labs GmbH 2026.

"""Convert a TabPFN PyTorch checkpoint to SafeTensors.

Non-tensor checkpoint fields (architecture name, config, inference config, …)
are JSON-encoded and stored in the safetensors header.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabpfn.checkpoint import Checkpoint, save_as_safetensors


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert a TabPFN .ckpt file to .safetensors with header metadata."
    )
    parser.add_argument("--input-checkpoint", required=True, type=Path)
    parser.add_argument("--output-safetensors", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = Checkpoint(args.input_checkpoint).load()
    save_as_safetensors(checkpoint, args.output_safetensors)
    print(f"Saved SafeTensors file: {args.output_safetensors}")


if __name__ == "__main__":
    main()
