#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE_PATH = (
    Path(__file__).parent.parent / "examples" / "plot_regression_distribution.py"
)


def _load_example_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "plot_regression_distribution_example", EXAMPLE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regression_distribution_example_saves_plot(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    module = _load_example_module()
    output_path = tmp_path / "regression_distribution.png"
    module.main(output_path=str(output_path), show=False)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
