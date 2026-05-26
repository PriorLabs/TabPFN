#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tabpfn.checkpoint import Checkpoint, save_as_safetensors


def _sample_checkpoint() -> dict:
    return {
        "architecture_name": "fake_arch",
        "config": {"max_num_classes": 10, "num_buckets": 100, "name": "test"},
        "inference_config": {"PREPROCESS_TRANSFORMS": []},
        "state_dict": {
            "encoder.weight": torch.randn(4, 3),
            "head.bias": torch.zeros(5),
        },
    }


def test__checkpoint__detects_safetensors_by_suffix() -> None:
    assert Checkpoint("model.safetensors").is_safetensors
    assert not Checkpoint("model.ckpt").is_safetensors


def test__checkpoint__torch_load_returns_full_dict(tmp_path: Path) -> None:
    original = _sample_checkpoint()
    path = tmp_path / "x.ckpt"
    torch.save(original, path)

    loaded = Checkpoint(path).load()

    assert loaded["architecture_name"] == original["architecture_name"]
    assert loaded["config"] == original["config"]
    assert torch.equal(
        loaded["state_dict"]["encoder.weight"],
        original["state_dict"]["encoder.weight"],
    )


def test__checkpoint__safetensors_round_trip_preserves_data(tmp_path: Path) -> None:
    original = _sample_checkpoint()
    safetensors_path = tmp_path / "x.safetensors"

    save_as_safetensors(original, safetensors_path)

    loaded = Checkpoint(safetensors_path).load()

    assert loaded["architecture_name"] == original["architecture_name"]
    assert loaded["config"] == original["config"]
    assert loaded["inference_config"] == original["inference_config"]
    assert set(loaded["state_dict"]) == set(original["state_dict"])
    for key, tensor in original["state_dict"].items():
        assert torch.equal(loaded["state_dict"][key], tensor)


def test__checkpoint__safetensors_load_without_metadata_returns_only_state_dict(
    tmp_path: Path,
) -> None:
    """A safetensors file written without header metadata still loads;
    the returned dict just has no non-tensor fields beyond state_dict.
    """
    path = tmp_path / "weights_only.safetensors"
    tensors = {"w": torch.randn(3, 3)}
    save_file(tensors, str(path))

    loaded = Checkpoint(path).load()

    assert set(loaded) == {"state_dict"}
    assert torch.equal(loaded["state_dict"]["w"], tensors["w"])


def test__checkpoint__identity_changes_when_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "x.ckpt"
    torch.save({"state_dict": {}}, path)
    ckpt = Checkpoint(path)
    identity_before = ckpt.identity()

    torch.save({"state_dict": {"a": torch.randn(100)}}, path)
    identity_after = ckpt.identity()

    assert identity_before != identity_after


def test__checkpoint__identity_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Checkpoint(tmp_path / "does_not_exist.ckpt").identity()


def test__save_as_safetensors__encodes_path_dtype_and_enum(tmp_path: Path) -> None:
    class Color(Enum):
        RED = "red"

    save_dir = tmp_path / "model"
    original = {
        "state_dict": {"w": torch.zeros(2)},
        "config": {
            "save_dir": save_dir,
            "dtype": torch.float32,
            "color": Color.RED,
            "tags": {"a", "b"},
        },
    }
    path = tmp_path / "x.safetensors"

    save_as_safetensors(original, path)
    loaded = Checkpoint(path).load()

    assert loaded["config"]["save_dir"] == str(save_dir)
    assert loaded["config"]["dtype"] == "torch.float32"
    assert loaded["config"]["color"] == "red"
    assert loaded["config"]["tags"] == ["a", "b"]


def test__save_as_safetensors__unknown_type_raises(tmp_path: Path) -> None:
    class Unserializable:
        pass

    bad = {
        "state_dict": {"w": torch.zeros(2)},
        "config": {"thing": Unserializable()},
    }
    with pytest.raises(TypeError, match="Unserializable"):
        save_as_safetensors(bad, tmp_path / "x.safetensors")
