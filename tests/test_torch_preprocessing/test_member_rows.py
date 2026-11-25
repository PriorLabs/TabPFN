#  Copyright (c) Prior Labs GmbH 2026.

"""`MemberRows` reads a member's model input as its GPU preprocessing builds it."""

from __future__ import annotations

import sys
import threading

import numpy as np
import pandas as pd
import pytest
import torch

from tabpfn import TabPFNClassifier
from tabpfn.base import clean_predict_input
from tabpfn.inference import _maybe_run_gpu_preprocessing, _prepare_model_inputs
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.preprocessing.torch import pipeline_interface
from tabpfn.preprocessing.torch.member_rows import (
    MemberRows,
    MemberRowSource,
    _cells,
)

N_TRAIN = 120
N_TEST = 30

# The preprocessors of TabPFN-4, each with sign flips, and its outlier clipping and
# feature shuffle.
_V4_LIKE = {
    "ENABLE_GPU_PREPROCESSING": True,
    "FINGERPRINT_FEATURE": False,
    "PASSTHROUGH_INF": True,
    "OUTLIER_REMOVAL_STD": 12.0,
    "FEATURE_SHIFT_METHOD": "shuffle",
    "PREPROCESS_TRANSFORMS": [
        PreprocessorConfig(
            name, categorical_name=categorical, random_sign_flip_numericals=True
        )
        for name, categorical in [
            ("none", "ordinal"),
            ("safepower", "ordinal_shuffled"),
            ("squashing_scaler_default", "ordinal_shuffled"),
            ("quantile_norm", "ordinal"),
        ]
    ],
}


def _members(**overrides: object) -> tuple[list, np.ndarray]:
    """Fitted members of a table with numeric, categorical and infinite cells, and the
    cleaned test rows.
    """
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_TEST
    X = pd.DataFrame(
        {
            **{f"x{i}": rng.standard_normal(n) for i in range(7)},
            "few": rng.integers(0, 3, n),
            "colour": pd.Categorical(rng.choice(["r", "g", "b"], n)),
        }
    )
    X.iloc[5, 2] = np.inf
    X.iloc[N_TRAIN + 3, 4] = -np.inf
    X.iloc[7, 1] = np.nan
    model = TabPFNClassifier(
        n_estimators=4,
        device="cpu",
        random_state=0,
        inference_config={**_V4_LIKE, **overrides},
    )
    model.fit(X.iloc[:N_TRAIN], rng.integers(0, 2, N_TRAIN))
    X_test = clean_predict_input(model, X.iloc[N_TRAIN:])
    return model.executor_.ensemble_members, X_test


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Groups of 5 columns moved to the compute device, cut into in-place blocks of 3.

    Set in the pipeline only: `MemberRows` must take its blocks from there.
    """
    column = (N_TRAIN + N_TEST) * 4
    monkeypatch.setattr(pipeline_interface, "IN_PLACE_BLOCK_BYTES", 3 * column)
    monkeypatch.setattr(pipeline_interface, "DEVICE_BLOCK_BYTES", 5 * column)


_DEVICES = pytest.mark.parametrize(
    ("device", "stage_on_cpu"),
    [("cpu", False), ("cpu", True), ("cuda", False), ("mps", False)],
    ids=["cpu", "cpu-staged", "cuda", "mps"],
)


def _device(
    device: str, stage_on_cpu: bool, monkeypatch: pytest.MonkeyPatch
) -> torch.device:
    """`device`; with `stage_on_cpu`, the pipeline stages a CPU input as for a GPU."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Needs a GPU")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Needs MPS")
    if stage_on_cpu:
        monkeypatch.setattr(pipeline_interface, "_same_device", lambda _a, _b: False)
    return torch.device(device)


def _host_staged(
    member, X_test: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, object]:
    """The member's model input, built in host memory and preprocessed on `device`."""
    X_full, _ = _prepare_model_inputs(
        device,
        None,
        member.X_train,
        X_test,
        member.y_train,
        x_device=torch.device("cpu"),
    )
    X_full, schema = _maybe_run_gpu_preprocessing(
        X_full,
        gpu_preprocessor=member.gpu_preprocessor,
        num_train_rows=member.X_train.shape[0],
        feature_schema=member.feature_schema,
        consume_input=True,
        compute_device=device,
    )
    return X_full[:, 0], schema


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual, expected.to(actual.device), rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.usefixtures("small_blocks")
@_DEVICES
def test__read__equals_the_host_staged_model_input(
    device: str, stage_on_cpu: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = _device(device, stage_on_cpu, monkeypatch)
    members, X_test = _members()
    rng = np.random.default_rng(1)
    for member in members:
        X_test_member = member.transform_X_test(X_test)
        expected, schema = _host_staged(member, X_test_member, device)
        rows = MemberRows.fit(
            member.X_train,
            X_test_member,
            member.gpu_preprocessor,
            member.feature_schema,
            device,
        )
        assert rows is not None
        assert rows.feature_schema == schema
        assert rows.shape == tuple(expected.shape)
        R, C = rows.shape
        _assert_equal(rows.read(slice(0, R), np.arange(C), device), expected)
        across = slice(N_TRAIN - 4, N_TRAIN + 5)
        some = rng.permutation(C)[: C // 2]
        _assert_equal(rows.read(across, some, device), expected[across][:, some])
        picked = np.sort(rng.permutation(R)[:20])
        _assert_equal(rows.read(picked, some, device), expected[picked][:, some])
        backwards = picked[::-1].copy()
        _assert_equal(rows.read(backwards, some, device), expected[backwards][:, some])


@pytest.mark.usefixtures("small_blocks")
@_DEVICES
def test__fit__fits_each_step_on_the_pipelines_blocks(
    device: str, stage_on_cpu: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step's fitted values can round differently on other blocks of columns."""
    device = _device(device, stage_on_cpu, monkeypatch)
    members, X_test = _members()
    for member in members:
        X_test_member = member.transform_X_test(X_test)
        fits = _record_fits(member.gpu_preprocessor, monkeypatch)
        _host_staged(member, X_test_member, device)
        expected = fits.copy()
        fits.clear()

        MemberRows.fit(
            member.X_train,
            X_test_member,
            member.gpu_preprocessor,
            member.feature_schema,
            device,
        )

        assert [step for step, _ in fits] == [step for step, _ in expected]
        assert len(expected) > len({id(step) for step, _ in expected})
        for (_, actual), (_, wanted) in zip(fits, expected, strict=True):
            _assert_equal(actual, wanted)


def _record_fits(
    pipeline, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[object, torch.Tensor]]:
    """Records, from now on, each input a columnwise step of `pipeline` fits on."""
    fits: list[tuple[object, torch.Tensor]] = []
    for step, _ in pipeline.steps:
        if step.columnwise:

            def spy(x: torch.Tensor, step=step, fit=step._fit) -> dict:
                fits.append((step, x.cpu().clone()))
                return fit(x)

            monkeypatch.setattr(step, "_fit", spy)
    return fits


@pytest.mark.parametrize("rows", [slice(1, 7), np.array([0, 3, 5])])
def test__cells__in_inference_mode__gathers_the_cells(rows: slice | np.ndarray) -> None:
    """The model reads its input inside `torch.inference_mode`."""
    table = np.arange(40, dtype=np.float32).reshape(8, 5)
    columns = np.array([4, 0, 2])

    with torch.inference_mode():
        cells = _cells(table, rows, columns)

    np.testing.assert_array_equal(cells.numpy(), table[rows][:, columns])


def test__cells__of_whole_rows__does_not_share_the_tables_memory() -> None:
    """The steps write into the cells a read returns."""
    table = np.zeros((4, 3), dtype=np.float32)

    cells = _cells(table, slice(0, 4), np.arange(3))

    assert not np.shares_memory(cells.numpy(), table)


@pytest.mark.usefixtures("small_blocks")
def test__row_source__reads_each_member_along_the_batch() -> None:
    members, X_test = _members()
    fitted = [
        MemberRows.fit(
            m.X_train,
            m.transform_X_test(X_test),
            m.gpu_preprocessor,
            m.feature_schema,
            torch.device("cpu"),
        )
        for m in members[:2]
    ]
    source = MemberRowSource(fitted)
    R, B, C = source.shape
    rows_BP = torch.stack(
        [torch.arange(0, R, 7), torch.arange(1, R, 7)[: len(range(0, R, 7))]]
    )

    gathered = source.read(rows_BP, slice(2, 6), torch.device("cpu"))
    one = source.read(torch.arange(3, 9), torch.tensor([0, 4]), torch.device("cpu"), 1)
    every = source.read(slice(None), slice(None), torch.device("cpu"), 0)

    for b in range(B):
        torch.testing.assert_close(
            gathered[b],
            fitted[b].read(rows_BP[b].numpy(), np.arange(2, 6), torch.device("cpu")),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    torch.testing.assert_close(
        one,
        fitted[1].read(np.arange(3, 9), np.array([0, 4]), torch.device("cpu")),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        every,
        fitted[0].read(slice(0, R), np.arange(C), torch.device("cpu")),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test__read__of_no_columns__is_empty() -> None:
    members, X_test = _members()
    member = members[0]
    rows = MemberRows.fit(
        member.X_train,
        member.transform_X_test(X_test),
        member.gpu_preprocessor,
        member.feature_schema,
        torch.device("cpu"),
    )

    cells = rows.read(slice(0, 10), np.arange(0), torch.device("cpu"))

    assert tuple(cells.shape) == (10, 0)


def test__read__from_many_threads__equals_the_host_staged_model_input() -> None:
    """A model can read one member from several threads, each its own columns."""
    device = torch.device("cpu")
    members, X_test = _members()
    member = members[1]
    X_test_member = member.transform_X_test(X_test)
    expected, _ = _host_staged(member, X_test_member, device)
    rows = MemberRows.fit(
        member.X_train,
        X_test_member,
        member.gpu_preprocessor,
        member.feature_schema,
        device,
    )
    R, C = rows.shape
    rng = np.random.default_rng(0)
    column_sets = [rng.permutation(C)[:k] for k in (3, 5, 3, 7)]
    failures: list[str] = []

    def read_again_and_again(columns: np.ndarray) -> None:
        try:
            for _ in range(100):
                _assert_equal(
                    rows.read(slice(0, R), columns, device), expected[:, columns]
                )
        except Exception as error:  # noqa: BLE001
            failures.append(repr(error))

    # Switch threads often, so that reads interleave within a step.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [
            threading.Thread(target=read_again_and_again, args=(columns,))
            for columns in column_sets
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(interval)

    assert not failures, failures[:2]


def test__fit__with_a_step_that_reads_whole_rows__is_none() -> None:
    """The fingerprint hashes whole rows, so the input must be built whole."""
    members, X_test = _members(FINGERPRINT_FEATURE=True)
    member = members[0]

    rows = MemberRows.fit(
        member.X_train,
        member.transform_X_test(X_test),
        member.gpu_preprocessor,
        member.feature_schema,
        torch.device("cpu"),
    )

    assert rows is None


def test__fit__on_cuda__leaves_nothing_on_the_device() -> None:
    """The model sizes its chunks from free device memory, which must be what the
    input built whole leaves free.
    """
    if not torch.cuda.is_available():
        pytest.skip("Needs a GPU")
    device = torch.device("cuda")
    members, X_test = _members()
    for member in members:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
        rows = MemberRows.fit(
            member.X_train,
            member.transform_X_test(X_test),
            member.gpu_preprocessor,
            member.feature_schema,
            device,
        )
        torch.cuda.synchronize()

        assert rows is not None
        assert torch.cuda.memory_allocated(device) == before
