#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.torch_version import TorchVersion

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.utils import (
    _cdf,
    _cpu_supports_fast_bf16,
    _repair_borders,
    _translate_probs_across_borders_unchunked,
    balance_probas_by_class_counts,
    infer_autocast_inference_mode,
    infer_devices,
    translate_probs_across_borders,
)


def _spiky_source(
    num_buckets: int, *, floor: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """A spiky source distribution whose smallest bucket still holds `floor`.

    Narrow spikes are what a retrieval-style head produces. The floor puts every
    bucket's mass far above float64's resolution near a CDF of 1 (~1e-16) and far
    below float32's (~6e-8), so a bucket coming out as exactly zero can only be
    the differencing losing it, never the source genuinely having none.
    """
    borders = torch.linspace(-4.0, 4.0, num_buckets + 1, dtype=torch.float64)
    mids = (borders[1:] + borders[:-1]) / 2
    generator = torch.Generator().manual_seed(0)
    centers = torch.rand(40, generator=generator, dtype=torch.float64) * 6 - 3
    density = sum(torch.exp(-0.5 * ((mids - c) / 0.02) ** 2) for c in centers)
    probs = density / density.sum()
    probs = probs * (1 - num_buckets * floor) + floor
    return borders, (probs / probs.sum()).log()[None, :]


def test__infer_devices__auto__cuda_and_mps_not_available__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
    mocker.patch("torch.cuda").is_available.return_value = False
    mocker.patch("torch.backends.mps").is_available.return_value = False
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__auto__single_cuda_gpu_available__selects_it(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cuda:0"),)


def test__infer_devices__auto__multiple_cuda_gpus_available__selects_all(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 3
    mocker.patch("torch.backends.mps").is_available.return_value = True

    assert infer_devices(devices="auto") == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    )


def test__infer_devices__auto__cuda_and_mps_available_but_excluded__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "mps,cuda")
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__auto__mps_available_but_torch_too_old__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", TorchVersion("2.4.0"))
    mocker.patch("torch.cuda").is_available.return_value = False
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__device_specified__selects_it(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2
    mocker.patch("torch.backends.mps").is_available.return_value = True

    assert infer_devices(devices="cuda:0") == (torch.device("cuda:0"),)


def test__infer_devices__multiple_devices_specified___selects_them(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 3
    mocker.patch("torch.backends.mps").is_available.return_value = False

    inferred = set(infer_devices(devices=["cuda:0", "cuda:1", "cuda:4"]))
    expected = {torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:4")}
    assert inferred == expected


def test__infer_devices__device_specified_twice__raises() -> None:
    with pytest.raises(
        ValueError,
        match="The list of devices for inference cannot contain the same device more ",
    ):
        infer_devices(devices=["cpu", "cpu"])


def test__infer_devices__mps_specified_but_torch_too_old__raises(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", TorchVersion("2.4.0"))
    mocker.patch("torch.backends.mps").is_available.return_value = True
    with pytest.raises(ValueError, match="The MPS device was selected"):
        infer_devices(devices="mps")


# --- Test Data for the "test_process_text_na_dataframe" test ---
test_cases = [
    {
        # Mixed dtypes & None / pd.Na
        "df": pd.DataFrame(
            {
                "ratio": [0.4, 0.5, 0.6],
                "risk": ["High", None, "Low"],
                "height": ["Low", "Low", "Low"],
                "amount": [10.2, 20.4, 20.5],
                "type": ["guest", "member", pd.NA],
            }
        ),
        "categorical_indices": [1, 2, 4],
        "ground_truth": np.array(
            [
                [0.4, 0, 0, 10.2, 0],
                [0.5, np.nan, 0, 20.4, 1],
                [0.6, 1, 0, 20.5, np.nan],
            ]
        ),
    },
    {
        # Mixed dtypes & no missing values
        "df": pd.DataFrame(
            {
                "ratio": [0.4, 0.5, 0.6],
                "risk": ["High", "Medium", "Low"],
                "height": ["Low", "Low", "High"],
                "amount": [10.2, 20.4, np.nan],
                "type": ["guest", "member", "vip"],
            }
        ),
        "categorical_indices": [1, 2, 4],
        "ground_truth": np.array(
            [
                [0.4, 0, 1, 10.2, 0],
                [0.5, 2, 1, 20.4, 1],
                [0.6, 1, 0, np.nan, 2],
            ]
        ),
    },
    {
        # All numerical no nan
        "df": pd.DataFrame(
            {
                "ratio": [0.1, 0.2, 0.3],
                "amount": [5.0, 15.5, 25.0],
                "score": [1.0, 2.5, 3.5],
            }
        ),
        "categorical_indices": [],
        "ground_truth": np.array(
            [
                [0.1, 5.0, 1.0],
                [0.2, 15.5, 2.5],
                [0.3, 25.0, 3.5],
            ]
        ),
    },
    {
        # all categorical no nan
        "df": pd.DataFrame(
            {
                "risk": ["High", "High", "High"],
                "height": ["Low", "Low", "Low"],
                "type": ["guest", "guest", "guest"],
            }
        ),
        "categorical_indices": [0, 1, 2],
        "ground_truth": np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    },
]


def test_balance_probas_by_class_counts():
    """Test balancing probabilities by class counts."""
    probas = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
    class_counts = np.array([1, 2])

    balanced = balance_probas_by_class_counts(probas, class_counts)

    # Check that each row sums to one
    sums = balanced.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), rtol=1e-5, atol=1e-5)

    expected_balanced = torch.tensor([[1 / 3, 2 / 3], [0.75, 0.25], [2 / 3, 1 / 3]])
    assert torch.allclose(balanced, expected_balanced, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch", [1, 32, 4097])
def test__translate_probs_across_borders__matches_unchunked(batch: int) -> None:
    """Chunked path must produce identical output to the unchunked reference."""
    torch.manual_seed(0)
    num_buckets = 5000
    logits = torch.randn(batch, num_buckets)
    frm = torch.linspace(-3.0, 3.0, num_buckets + 1)
    to = torch.linspace(-3.0, 3.0, num_buckets + 1)

    out_unchunked = _translate_probs_across_borders_unchunked(logits, frm=frm, to=to)
    out_public = translate_probs_across_borders(logits, frm=frm, to=to)

    assert out_public.shape == out_unchunked.shape
    # Each row is processed independently, so chunking must be bit-exact.
    assert torch.equal(out_public, out_unchunked)


@pytest.mark.parametrize("shape", [(128, 200), (3, 128, 200), (2, 3, 128, 200)])
def test__translate_probs_across_borders__forces_chunking(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    """Force the chunked path and verify it runs across arbitrary batch shapes.

    Passes a tiny ``chunk_budget_elements`` (= ``num_buckets``) so every batch
    row triggers its own chunked call, and spies on the unchunked helper to
    confirm the chunked dispatch is actually used.
    """
    torch.manual_seed(1)
    num_buckets = shape[-1]
    logits = torch.randn(*shape)
    frm = torch.linspace(-3.0, 3.0, num_buckets + 1)
    to = torch.linspace(-3.0, 3.0, num_buckets + 1)

    out_unchunked = _translate_probs_across_borders_unchunked(logits, frm=frm, to=to)

    call_counter = {"n": 0}
    orig = _translate_probs_across_borders_unchunked

    def counting_unchunked(*args, **kwargs) -> torch.Tensor:
        call_counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(
        "tabpfn.utils._translate_probs_across_borders_unchunked",
        counting_unchunked,
    )
    out_chunked = translate_probs_across_borders(
        logits, frm=frm, to=to, chunk_budget_elements=num_buckets
    )

    total_rows = 1
    for d in shape[:-1]:
        total_rows *= d
    # Expect one unchunked call per chunk: more than one proves we chunked.
    assert call_counter["n"] > 1
    assert call_counter["n"] == total_rows  # chunk_size == 1 row here
    assert out_chunked.shape == out_unchunked.shape
    assert torch.equal(out_chunked, out_unchunked)


def test__translate_probs_across_borders__upcast_prevents_lost_bucket_mass() -> None:
    """Small destination buckets must not round to zero (PRI-361, mechanism 1).

    A bucket's mass is the difference of two cumulative sums, so its resolution
    is that of the CDF near 1 -- ~6e-8 in float32. Anything below that lands on
    exactly zero, `regressor.py` takes `.log()` of it, and the bar
    distribution's NLL for a target in that bucket is `inf`. The destination
    grid here is nested inside the source, so no bucket is out of range and
    every one of them genuinely holds at least ~1e-12.
    """
    num_buckets = 5000
    frm, logits = _spiky_source(num_buckets)
    to = torch.linspace(-3.9, 3.9, num_buckets + 1, dtype=torch.float64)

    out = translate_probs_across_borders(logits, frm=frm, to=to)

    assert int((out == 0).sum()) == 0
    assert out.sum(-1).item() == pytest.approx(1.0, abs=1e-12)


def test__translate_probs_across_borders__float32_differencing_is_the_culprit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the upcast as the thing doing the work in the test above.

    Without it the very same inputs lose hundreds of buckets, so this guards
    against the upcast being quietly dropped and the test above still passing
    for some unrelated reason.
    """
    num_buckets = 5000
    frm, logits = _spiky_source(num_buckets)
    to = torch.linspace(-3.9, 3.9, num_buckets + 1, dtype=torch.float64)

    monkeypatch.setattr("tabpfn.utils._TRANSLATE_COMPUTE_DTYPE", torch.float32)
    out = translate_probs_across_borders(logits, frm=frm, to=to)

    assert int((out == 0).sum()) > 100


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test__translate_probs_across_borders__identity_remap_is_the_identity(
    dtype: torch.dtype,
) -> None:
    """`to == frm` must return the input probabilities in the caller's dtype.

    The outer buckets are half-normal tails whose mass extends past the
    outermost border, and folding that mass back into the outermost destination
    buckets is what keeps this identity (and the sum to 1) intact.
    """
    torch.manual_seed(0)
    logits = torch.randn(64, 300, dtype=dtype)
    borders = torch.linspace(-3.0, 3.0, 301, dtype=dtype)

    out = translate_probs_across_borders(logits, frm=borders, to=borders)

    assert out.dtype == dtype
    torch.testing.assert_close(out, logits.softmax(-1))


def test__cdf__outer_buckets_are_half_normal_tails() -> None:
    """`_cdf` must read the outer buckets the way `forward` does (PRI-361, mech 2).

    `FullSupportBarDistribution` puts a half-normal on each outer bucket, so
    half of that bucket's mass lies beyond the outermost border. Pinning the CDF
    to 0 and 1 at the outermost borders instead drops those two half-tails at
    any precision.
    """
    num_buckets = 500
    borders = torch.linspace(-4.0, 4.0, num_buckets + 1, dtype=torch.float64)
    p_outer = 0.05
    probs = torch.full(
        (num_buckets,), (1.0 - 2 * p_outer) / (num_buckets - 2), dtype=torch.float64
    )
    probs[0] = probs[-1] = p_outer
    logits = probs.log()[None, :]

    at_borders = _cdf(logits, borders=borders, ys=borders[[0, -1]].clone())
    # Exactly half of each outer bucket's mass sits outside the grid.
    assert at_borders[0, 0].item() == pytest.approx(p_outer / 2, rel=1e-12)
    assert (1.0 - at_borders[0, 1]).item() == pytest.approx(p_outer / 2, rel=1e-12)

    # Well outside the grid the CDF is small but strictly positive, not zero.
    far = torch.tensor([-4.05, 4.05], dtype=torch.float64)
    outside = _cdf(logits, borders=borders, ys=far)
    assert outside[0, 0].item() > 0.0
    assert outside[0, 1].item() < 1.0

    # And it agrees with the half-normal those tails are defined by.
    sigma = FullSupportBarDistribution.halfnormal_with_p_weight_before(
        borders[1] - borders[0],
    ).scale
    expected = p_outer * torch.erfc(
        (borders[1] - far[0]) / (sigma * math.sqrt(2.0)),
    )
    assert outside[0, 0].item() == pytest.approx(expected.item(), rel=1e-12)


def test__cdf__tails_leave_interior_borders_untouched() -> None:
    """Reading the outer buckets as tails must not move any interior border.

    A tail redistributes mass *within* its outer bucket as well as past the
    border, but `CDF(borders[1]) == probs[0]` either way, so every interior
    bucket keeps exactly the mass it had.
    """
    num_buckets = 500
    borders, logits = _spiky_source(num_buckets)
    ks = [1, 2, 50, 250, num_buckets - 2, num_buckets - 1]

    got = _cdf(logits, borders=borders, ys=borders[ks].clone())
    expected = torch.cumsum(logits.softmax(-1)[0], 0)[[k - 1 for k in ks]]

    # The tolerance is set by the reference, not by `_cdf`: `cumsum` accumulates
    # one rounding per bucket, so `num_buckets * eps` is the floor. A tail
    # actually leaking into the interior would move a border by O(p_outer).
    torch.testing.assert_close(got[0], expected, rtol=0.0, atol=1e-13)


def test__translate_probs_across_borders__fills_buckets_outside_source_range() -> None:
    """Destination buckets past the source grid must receive the tail mass.

    This is the case where a member's y-transform gives a narrower grid than the
    pooled `znorm_space_bardist_` grid, so `to` extends past `frm`. Before the
    tails those buckets were exactly zero at any precision, and the mass was
    misplaced into the single destination bucket straddling the border.
    """
    num_buckets = 500
    frm = torch.linspace(-4.0, 4.0, num_buckets + 1, dtype=torch.float64)
    to = torch.linspace(-4.5, 4.5, num_buckets + 1, dtype=torch.float64)
    p_outer = 0.05
    probs = torch.full(
        (num_buckets,), (1.0 - 2 * p_outer) / (num_buckets - 2), dtype=torch.float64
    )
    probs[0] = probs[-1] = p_outer
    logits = probs.log()[None, :]

    out = translate_probs_across_borders(logits, frm=frm, to=to)[0]

    # The buckets immediately below `frm[0]` carry the lower half-tail. The
    # far ones are not asserted on: a half-normal scaled to the bucket width
    # decays fast enough that their true mass underflows float64 legitimately.
    below = (to[1:] <= frm[0]).nonzero().flatten()
    assert len(below) > 0
    assert (out[below[-3:]] > 0).all()

    # Mass placed strictly below `frm[0]` must match the analytic tail, which
    # is everything the source's half-normal puts below the destination border
    # closest to `frm[0]` from the outside.
    sigma = FullSupportBarDistribution.halfnormal_with_p_weight_before(
        frm[1] - frm[0],
    ).scale
    edge = to[int(below[-1]) + 1]
    expected = p_outer * torch.erfc((frm[1] - edge) / (sigma * math.sqrt(2.0)))
    assert out[below].sum().item() == pytest.approx(expected.item(), rel=1e-9)

    # The remap still normalises: nothing is created or destroyed overall.
    assert out.sum().item() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("chunked", [False, True])
def test__translate_probs_across_borders__mps_matches_cpu(chunked: bool) -> None:
    """The float64 differencing must survive a device that has no float64.

    MPS has none, so that path computes on the CPU. The conversions have to be
    staged (move, then cast): a combined `.to(device=..., dtype=...)` off MPS
    converts on the source device and silently returns all zeros rather than
    raising, which produces a plausible-looking but completely wrong
    distribution.
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    torch.manual_seed(0)
    num_buckets = 2000
    logits = torch.randn(512, num_buckets)
    frm = torch.linspace(-4.0, 4.0, num_buckets + 1)
    to = torch.linspace(-4.2, 4.2, num_buckets + 1)
    budget = {"chunk_budget_elements": num_buckets + 1} if chunked else {}

    on_cpu = translate_probs_across_borders(logits, frm=frm, to=to, **budget)
    on_mps = translate_probs_across_borders(
        logits.to("mps"), frm=frm.to("mps"), to=to.to("mps"), **budget
    )

    assert on_mps.device.type == "mps"
    assert on_mps.dtype == logits.dtype
    # The whole computation happens on the CPU either way, so it is exact.
    assert torch.equal(on_mps.cpu(), on_cpu)


def test__translate_probs_across_borders__degenerate_outer_bucket_is_finite() -> None:
    """A zero-width outer bucket must not turn the tail into NaN.

    `_repair_borders` widens a collapsed outer bucket by a fraction of its own
    value, which is a no-op when that border sits exactly at 0.0, so a
    zero-width outer bucket can still reach here.
    """
    frm = torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 3.0], dtype=torch.float64)
    to = torch.linspace(-1.0, 4.0, 6, dtype=torch.float64)
    logits = torch.zeros(4, 5, dtype=torch.float64)

    out = translate_probs_across_borders(logits, frm=frm, to=to)

    assert bool(torch.isfinite(out).all())
    torch.testing.assert_close(out.sum(-1), torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize(
    ("mkldnn", "avx512_bf16", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
    ids=["bf16_hardware", "no_bf16_hardware", "no_onednn_build"],
)
def test__cpu_supports_fast_bf16(
    mocker: MagicMock,
    mkldnn: bool,
    avx512_bf16: bool,
    expected: bool,
) -> None:
    mocker.patch.object(torch.backends.mkldnn, "is_available", return_value=mkldnn)
    mocker.patch.object(
        torch.cpu, "_is_avx512_bf16_supported", return_value=avx512_bf16
    )
    assert _cpu_supports_fast_bf16() is expected


def test__cpu_supports_fast_bf16__torch_helper_missing__warns_and_returns_false(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocker.patch.object(torch.backends.mkldnn, "is_available", return_value=True)
    monkeypatch.delattr(torch.cpu, "_is_avx512_bf16_supported", raising=False)
    with pytest.warns(UserWarning, match="cannot detect CPU bf16 support"):
        assert _cpu_supports_fast_bf16() is False


@pytest.mark.parametrize(
    ("supports_bf16", "enable", "expected"),
    [(True, None, True), (False, None, False), (True, False, False)],
    ids=["auto_with_bf16", "auto_without_bf16", "explicitly_disabled"],
)
def test__infer_autocast_inference_mode__cpu(
    mocker: MagicMock,
    supports_bf16: bool,
    enable: bool | None,
    expected: bool,
) -> None:
    mocker.patch("tabpfn.utils._cpu_supports_fast_bf16", return_value=supports_bf16)
    mocker.patch("tabpfn.utils.is_autocast_available", return_value=True)
    assert (
        infer_autocast_inference_mode([torch.device("cpu")], enable=enable) is expected
    )


def test__infer_autocast_inference_mode__cpu_without_fast_bf16_and_enabled__raises(
    mocker: MagicMock,
) -> None:
    mocker.patch("tabpfn.utils._cpu_supports_fast_bf16", return_value=False)
    mocker.patch("tabpfn.utils.is_autocast_available", return_value=True)
    with pytest.raises(ValueError, match="does not support it"):
        infer_autocast_inference_mode([torch.device("cpu")], enable=True)


@pytest.mark.parametrize(
    ("borders", "expected_last"),
    [
        # Widening must move the top border up regardless of sign. Multiplying by
        # 1.1 moves a negative border further down, past borders[-2].
        ([-10.0, -5.0, -4.9999999], -4.5),
        ([1.0, 5.0, 5.0000001], 5.5),
        ([-2.0, 3.0, 3.0000001], 3.3),
    ],
)
def test__repair_borders__collapsed_top_gap__widens_upwards(borders, expected_last):
    repaired = np.array(borders)
    _repair_borders(repaired, inplace=True)

    assert repaired[-1] == pytest.approx(expected_last)
    assert np.all(np.diff(repaired) > 0)


@pytest.mark.parametrize(
    ("borders", "expected_last"),
    [
        ([-10.0, -5.0, np.nan], 0.0),
        ([1.0, 5.0, np.nan], 10.0),
    ],
)
def test__repair_borders__nan_top__widens_upwards(borders, expected_last):
    repaired = np.array(borders)
    _repair_borders(repaired, inplace=True)

    assert repaired[-1] == pytest.approx(expected_last)
    assert np.all(np.diff(repaired) > 0)
