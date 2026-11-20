#  Copyright (c) Prior Labs GmbH 2026.

"""`MemberRows` reads a member's model input as its GPU preprocessing builds it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tabpfn import TabPFNClassifier
from tabpfn.base import clean_predict_input
from tabpfn.inference import _maybe_run_gpu_preprocessing, _prepare_model_inputs
from tabpfn.preprocessing import PreprocessorConfig
from tabpfn.preprocessing.torch import member_rows, pipeline_interface
from tabpfn.preprocessing.torch.member_rows import MemberRows, MemberRowSource

N_TRAIN = 120
N_TEST = 30
COLUMNS_PER_BLOCK = 3

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
    """Fit blocks of `COLUMNS_PER_BLOCK` columns in the eager pipeline and in
    `MemberRows`. Unstaged on the CPU, the eager pipeline fits in in-place blocks
    only, so `MemberRows`' groups are made no wider.
    """
    block = (N_TRAIN + N_TEST) * 4 * COLUMNS_PER_BLOCK
    monkeypatch.setattr(pipeline_interface, "IN_PLACE_BLOCK_BYTES", block)
    monkeypatch.setattr(member_rows, "IN_PLACE_BLOCK_BYTES", block)
    monkeypatch.setattr(member_rows, "DEVICE_BLOCK_BYTES", block)


def _eager(
    member, X_test: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, object]:
    X_full, _ = _prepare_model_inputs(
        device, None, member.X_train, X_test, member.y_train
    )
    X_full, schema = _maybe_run_gpu_preprocessing(
        X_full,
        gpu_preprocessor=member.gpu_preprocessor,
        num_train_rows=member.X_train.shape[0],
        feature_schema=member.feature_schema,
        consume_input=True,
    )
    return X_full[:, 0], schema


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Equal on CUDA. On the CPU, a vectorized elementwise kernel can round an element
    in its last bit by where it falls in a tensor of another shape.
    """
    exact = {"rtol": 0, "atol": 0} if actual.device.type == "cuda" else {}
    torch.testing.assert_close(
        actual, expected.to(actual.device), equal_nan=True, **exact
    )


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test__read__equals_the_eager_model_input(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Needs a GPU")
    device = torch.device(device)
    members, X_test = _members()
    rng = np.random.default_rng(1)
    for member in members:
        X_test_member = member.transform_X_test(X_test)
        expected, schema = _eager(member, X_test_member, device)
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
        picked = rng.permutation(R)[:20]
        _assert_equal(rows.read(picked, some, device), expected[picked][:, some])


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
    R, B, _C = source.shape
    rows_BP = torch.stack(
        [torch.arange(0, R, 7), torch.arange(1, R, 7)[: len(range(0, R, 7))]]
    )

    gathered = source.read(rows_BP, slice(2, 6), torch.device("cpu"))
    one = source.read(torch.arange(3, 9), torch.tensor([0, 4]), torch.device("cpu"), 1)

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


def test__fit__with_a_step_that_reads_whole_rows__is_none() -> None:
    """The fingerprint hashes whole rows, so the input has to be built whole."""
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
