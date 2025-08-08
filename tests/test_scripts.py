# tests/test_scripts.py

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 1. Import the main function from the NEW, merged script
from scripts.generate_dependencies import main as generate_deps_main

# --- Shared Test Data (This part is unchanged) ---

PYPROJECT_CONTENT = """
[project]
name = "my-package"
dependencies = [
  "torch>=1.0,<3",
  "scipy<2.0",
  "pandas>=1.4.0",
  "einops",
  "huggingface-hub>=0.0.1,<1",
]
"""
EXPECTED_MAX_REQS = sorted(
    ["torch<3", "scipy<2.0", "pandas", "einops", "huggingface-hub<1"]
)
EXPECTED_MIN_REQS = sorted(["torch==1.0", "pandas==1.4.0", "huggingface-hub==0.0.1"])


# 2. Update the parametrize decorator to pass the 'mode' string instead of a function
@pytest.mark.parametrize(
    ("mode", "expected_requirements"),
    [
        ("maximum", EXPECTED_MAX_REQS),
        ("minimum", EXPECTED_MIN_REQS),
    ],
)
def test_dependency_script_generation(
    mode: str,
    expected_requirements: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests the unified generate_dependencies.py script for both 'min' and 'max' modes.

    This test simulates the command-line arguments required by the new script.
    """
    # ARRANGE
    # This part is the same: create the pyproject.toml and change directory
    p = tmp_path / "pyproject.toml"
    p.write_text(PYPROJECT_CONTENT)
    monkeypatch.chdir(tmp_path)

    # 3. Simulate command-line arguments using monkeypatch.
    #    We are setting sys.argv to what it would be if called from the terminal,
    #    e.g., ['generate_dependencies.py', 'min']
    monkeypatch.setattr(sys, "argv", ["generate_dependencies.py", mode])

    # ACT
    # Call the single main function. It will now parse the mocked sys.argv.
    generate_deps_main()

    # ASSERT
    # The assertion logic remains exactly the same.
    output_file = tmp_path / "requirements.txt"
    assert (
        output_file.is_file()
    ), f"Script did not create requirements.txt in '{mode}' mode."

    with output_file.open() as f:
        actual_requirements = sorted(line.strip() for line in f if line.strip())

    assert (
        actual_requirements == expected_requirements
    ), f"Output in '{mode}' mode did not match expectations."
