#  Copyright (c) Prior Labs GmbH 2026.

"""Execute the demo notebook end to end against one of the TabPFN backends.

Runs every cell of `examples/notebooks/TabPFN_Demo_Local.ipynb` in a real
Jupyter kernel with `TABPFN_DEMO_BACKEND` set, so a cell that raises fails the
test, then checks that the headline metrics the notebook prints still clear
their floors.

Skipped by default. Set `RUN_NOTEBOOK_EXECUTION_CHECK=1` to enable and
`TABPFN_DEMO_BACKEND` to `local` or `client` to pick the backend. Intended to
run from `.github/workflows/notebook-execution.yml`, not from the regular PR
test matrix: a full pass installs several GB of packages, downloads public
datasets and takes tens of minutes on a GPU.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

if not os.environ.get("RUN_NOTEBOOK_EXECUTION_CHECK"):
    pytest.skip(
        "set RUN_NOTEBOOK_EXECUTION_CHECK=1 to enable (intended for nightly CI)",
        allow_module_level=True,
    )

NOTEBOOK = (
    Path(__file__).parents[1] / "examples" / "notebooks" / "TabPFN_Demo_Local.ipynb"
)

BACKEND = os.environ.get("TABPFN_DEMO_BACKEND", "")

# Per-cell wall clock. The slowest cells fit a model per cross-validation fold
# for four estimators, so this is generous by design; the workflow's job
# timeout is what bounds a hung run.
CELL_TIMEOUT_S = 60 * 30

# Floors on the metrics the notebook prints, as
# `label -> (regex over cell output, lowest acceptable value)`. They sit far
# enough below the observed values to absorb the run-to-run spread of an
# unseeded server-side ensemble, and still catch a backend that has stopped
# learning. Metrics where lower is better are negated before comparison.
METRIC_FLOORS: dict[str, tuple[str, float]] = {
    "parkinsons roc auc": (r"TabPFN ROC AUC Score: ([0-9.]+)", 0.85),
    "boston neg rmse": (r"TabPFN RMSE: (-?[0-9.]+)", -4.5),
}
NEGATED_METRICS = {"boston neg rmse"}


def _cell_text(notebook: dict) -> str:
    """Return every stream and text/plain output in the notebook, concatenated."""
    chunks = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append("".join(output.get("text", [])))
            data = output.get("data", {})
            if "text/plain" in data:
                chunks.append("".join(data["text/plain"]))
    return "\n".join(chunks)


@pytest.fixture(scope="module")
def executed_notebook(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the notebook in a fresh working directory and return it."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    if BACKEND not in {"local", "client"}:
        pytest.fail(f"TABPFN_DEMO_BACKEND must be 'local' or 'client', got {BACKEND!r}")

    workdir = tmp_path_factory.mktemp("notebook-run")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT_S,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
    ).execute()

    # Keep the executed copy so a CI failure can be inspected from the run's
    # artifacts rather than only from the traceback.
    out = Path(os.environ.get("NOTEBOOK_EXECUTION_OUT", workdir / "executed.ipynb"))
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, out)
    return notebook


def test_notebook_runs_end_to_end(executed_notebook: dict) -> None:
    """Every code cell produced output; none raised."""
    errors = [
        output
        for cell in executed_notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, json.dumps(errors, indent=2)[:4000]


@pytest.mark.parametrize("metric", sorted(METRIC_FLOORS))
def test_headline_metric_did_not_regress(executed_notebook: dict, metric: str) -> None:
    pattern, floor = METRIC_FLOORS[metric]
    text = _cell_text(executed_notebook)
    match = re.search(pattern, text)
    assert match, f"{metric}: no cell output matched {pattern!r}"
    value = float(match.group(1))
    if metric in NEGATED_METRICS:
        value = -value
    assert value >= floor, f"{metric}: {value} fell below {floor}"
