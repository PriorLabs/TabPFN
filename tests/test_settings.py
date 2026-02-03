from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tabpfn.settings import TabPFNSettings, TestingSettings
from tests.utils import get_pytest_devices


def test__load_settings__env_file_contains_variables_for_other_apps__does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    env_file = tmp_path / ".env"
    with env_file.open(mode="w") as f:
        f.write("OTHER_APP_VAR=1\n")
        f.write("TABPFN_MODEL_CACHE_DIR=test_cache_dir\n")
    tabpfn_settings = TabPFNSettings(_env_file=env_file)
    assert str(tabpfn_settings.model_cache_dir) == "test_cache_dir"


def test__ci_env_non_boolean_sets_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "azure")
    testing_settings = TestingSettings()
    assert testing_settings.ci is True


def test__ci_env_false_sets_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "false")
    testing_settings = TestingSettings()
    assert testing_settings.ci is False


@pytest.mark.skipif("mps" not in get_pytest_devices(), reason="MPS not available")
def test__mps_memory_limit__low_limit__causes_oom() -> None:
    """Test that a very low MPS memory limit causes OOM on large allocations."""
    TabPFNSettings(mps_memory_fraction=0.001)  # 0.1% - triggers model_post_init
    torch.mps.empty_cache()

    with pytest.raises(RuntimeError, match="out of memory"):
        _ = [torch.randn(5000, 5000, device="mps") for _ in range(10)]
