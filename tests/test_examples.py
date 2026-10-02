#  Copyright (c) Prior Labs GmbH 2026.
"""Validate the executable examples and example notebooks."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tabpfn

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
PYTHON_EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))
NOTEBOOKS = sorted((EXAMPLES_DIR / "notebooks").glob("*.ipynb"))
SMOKE_EXAMPLES = (
    "tabpfn_for_binary_classification.py",
    "tabpfn_for_regression.py",
)
_SHELL_PREFIXES = ("!", "%")


def _notebook_source(cell: dict[str, Any]) -> str:
    source = cell.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    raise AssertionError("Notebook code cell has no textual source")


def _without_shell_commands(source: str) -> str:
    return "\n".join(
        "" if line.lstrip().startswith(_SHELL_PREFIXES) else line
        for line in source.splitlines()
    )


def _tabpfn_imports(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "tabpfn":
            names.update(alias.name for alias in node.names if alias.name != "*")
    return names


def test_example_files_are_present() -> None:
    assert PYTHON_EXAMPLES
    assert NOTEBOOKS


@pytest.mark.parametrize("path", PYTHON_EXAMPLES, ids=lambda path: path.name)
def test_python_examples_compile_and_use_public_tabpfn_api(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    for name in _tabpfn_imports(source):
        assert hasattr(tabpfn, name), f"{path.name} imports unavailable {name}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_have_compilable_code_cells(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    cells = notebook.get("cells")
    assert isinstance(cells, list)
    assert all(isinstance(cell, dict) for cell in cells)

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    assert code_cells

    imported_names: set[str] = set()
    install_lines: list[str] = []
    for index, cell in enumerate(code_cells, start=1):
        source = _notebook_source(cell)
        for line in source.splitlines():
            if line.lstrip().startswith(_SHELL_PREFIXES):
                install_lines.append(line)
        python_source = _without_shell_commands(source)
        compile(python_source, f"{path}:cell-{index}", "exec")
        imported_names.update(_tabpfn_imports(python_source))

    assert any("tabpfn" in line.lower() for line in install_lines)
    for name in imported_names:
        assert hasattr(tabpfn, name), f"{path.name} imports unavailable {name}"


@pytest.mark.skipif(
    os.environ.get("TABPFN_RUN_EXAMPLE_SMOKE_TESTS") != "1",
    reason="set TABPFN_RUN_EXAMPLE_SMOKE_TESTS=1 to run example smoke tests",
)
@pytest.mark.parametrize("filename", SMOKE_EXAMPLES)
def test_core_examples_run(filename: str, tmp_path: Path) -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLES_DIR / filename)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
