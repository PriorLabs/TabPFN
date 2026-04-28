"""Verify each example notebook resolves `tabpfn` to the latest PyPI release.

Reproduces the install sequence from each `examples/notebooks/*.ipynb` in a
fresh venv and asserts that `tabpfn` resolves to the version currently
published as latest on PyPI. Catches transitive caps (e.g. a stale
`tabpfn-extensions[all]` pinning `tabpfn<7`) before users hit them.

Skipped by default. Set `RUN_NOTEBOOK_INSTALL_CHECK=1` to enable.
Intended to run from `.github/workflows/nightly-notebook-check.yml`, not
from the regular PR test matrix.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

import pytest

if not os.environ.get("RUN_NOTEBOOK_INSTALL_CHECK"):
    pytest.skip(
        "set RUN_NOTEBOOK_INSTALL_CHECK=1 to enable (intended for nightly CI)",
        allow_module_level=True,
    )

NOTEBOOK_DIR = Path(__file__).parents[1] / "examples" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
PIP_RE = re.compile(r"^\s*[!%](?:uv\s+)?pip\s+install\s+(.+?)\s*$")
NETWORKY = (
    "connection",
    "timeout",
    "503",
    "504",
    "failed to fetch",
    "network",
    "temporary failure",
)


def _extract_install_lines(notebook_path: Path) -> list[str]:
    """Return pip-install argument strings from a notebook's code cells."""
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    lines = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for line in source.splitlines():
            m = PIP_RE.match(line)
            if m:
                lines.append(m.group(1))
    return lines


def _latest_tabpfn_version(retries: int = 3, backoff: float = 5.0) -> str:
    """Fetch latest tabpfn version from PyPI, retrying on transient failures."""
    for attempt in range(retries):
        try:
            req = Request(
                "https://pypi.org/pypi/tabpfn/json",
                headers={"User-Agent": "tabpfn-notebook-check/1.0"},
            )
            with urlopen(req, timeout=15) as r:  # noqa: S310
                return cast("str", json.load(r)["info"]["version"])
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                pytest.skip(f"PyPI metadata fetch failed (network): {e}")
            time.sleep(backoff)
    raise RuntimeError("unreachable")


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_resolves_latest_tabpfn(notebook: Path, tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")

    install_lines = _extract_install_lines(notebook)
    if not any("tabpfn" in line for line in install_lines):
        pytest.skip(f"{notebook.name} does not install tabpfn")

    # Don't inherit [tool.uv] settings from any pyproject.toml above us
    # (e.g. TabPFN's `exclude-newer`). We're simulating a fresh Colab user
    # with no project context, so any repo-local resolver constraints would
    # produce false negatives.
    base_env = {**os.environ, "UV_NO_CONFIG": "1"}

    venv = tmp_path / "venv"
    subprocess.run(  # noqa: S603
        ["uv", "venv", "--python", "3.12", str(venv)],  # noqa: S607
        check=True,
        capture_output=True,
        env=base_env,
    )
    env = {
        **base_env,
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv}/bin:{os.environ.get('PATH', '')}",
    }

    for line in install_lines:
        result = subprocess.run(  # noqa: S603
            ["uv", "pip", "install", *shlex.split(line)],  # noqa: S607
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.lower()
            tail = result.stderr.strip()[-500:]
            if any(s in stderr for s in NETWORKY):
                pytest.skip(f"network error during '{line}': {tail[-200:]}")
            pytest.fail(f"install failed: {line}\n--- stderr ---\n{tail}")

    show = subprocess.run(
        ["uv", "pip", "show", "tabpfn"],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    installed = next(
        line.split(":", 1)[1].strip()
        for line in show.stdout.splitlines()
        if line.startswith("Version:")
    )

    latest = _latest_tabpfn_version()
    assert installed == latest, (
        f"{notebook.name}: sequential install resolves tabpfn=={installed}, "
        f"but PyPI latest is {latest}. Likely cause: a transitive cap (e.g. "
        f"tabpfn-extensions or another notebook dep) is holding tabpfn back."
    )
