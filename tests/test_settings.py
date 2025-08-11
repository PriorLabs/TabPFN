from __future__ import annotations

from pathlib import Path

from tabpfn.settings import TabPFNSettings


def test__load_settings__env_file_contains_variables_for_other_apps__does_not_crash(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    with env_file.open(mode="w") as f:
        f.write("OTHER_APP_VAR=1\n")
        f.write("TABPFN_MODEL_CACHE_DIR=test_cache_dir\n")
    tabpfn_settings = TabPFNSettings(_env_file=env_file)
    assert str(tabpfn_settings.model_cache_dir) == "test_cache_dir"
