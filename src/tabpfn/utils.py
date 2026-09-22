"""A collection of random utilities for the TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import contextlib
import functools
import math
import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.constants import (
    REGRESSION_NAN_BORDER_LIMIT_LOWER,
    REGRESSION_NAN_BORDER_LIMIT_UPPER,
)
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)


def get_autocast_context(
    device: torch.device, *, enabled: bool
) -> contextlib.AbstractContextManager:
    """Returns a torch.autocast context manager.

    Args:
        device: The torch device being used.
        enabled: Whether to enable autocast.

    Returns:
        A context manager for autocasting.
    """
    return torch.autocast(device.type, enabled=enabled)


def _repair_borders(borders: np.ndarray, *, inplace: Literal[True]) -> None:
    # Try to repair a broken transformation of the borders:
    #   This is needed when a transformation of the ys leads to very extreme values
    #   in the transformed borders, since the borders spanned a very large range in
    #   the original space.
    #   Borders that were transformed to extreme values are all set to the same
    #   value, the maximum of the transformed borders. Thus probabilities predicted
    #   in these buckets have no effects. The outermost border is set to the
    #   maximum of the transformed borders times 2, so still allow for some weight
    #   in the long tailed distribution and avoid infinite loss.
    if inplace is not True:
        raise NotImplementedError("Only inplace is supported")

    if np.isnan(borders[-1]):
        nans = np.isnan(borders)
        largest = borders[~nans].max()
        borders[nans] = largest
        borders[-1] += np.abs(borders[-1])

    if borders[-1] - borders[-2] < 1e-6:
        borders[-1] += np.abs(borders[-1] * 0.1)

    if borders[0] == borders[1]:
        borders[0] -= np.abs(borders[0] * 0.1)


def _cancel_nan_borders(
    *,
    borders: np.ndarray,
    broken_mask: npt.NDArray[np.bool_],
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # OPTIM: You could do one check at a time
    # assert it is consecutive areas starting from both ends
    borders = borders.copy()
    num_right_borders = (broken_mask[:-1] > broken_mask[1:]).sum()
    num_left_borders = (broken_mask[1:] > broken_mask[:-1]).sum()
    assert num_left_borders <= 1
    assert num_right_borders <= 1

    if num_right_borders:
        assert bool(broken_mask[0]) is True
        rightmost_nan_of_left = np.where(broken_mask[:-1] > broken_mask[1:])[0][0] + 1
        borders[:rightmost_nan_of_left] = borders[rightmost_nan_of_left]
        borders[0] = borders[1] - 1.0

    if num_left_borders:
        assert bool(broken_mask[-1]) is True
        leftmost_nan_of_right = np.where(broken_mask[1:] > broken_mask[:-1])[0][0]
        borders[leftmost_nan_of_right + 1 :] = borders[leftmost_nan_of_right]
        borders[-1] = borders[-2] + 1.0

    # logit mask, mask out the nan positions, the borders are 1 more than logits
    logit_cancel_mask = broken_mask[1:] | broken_mask[:-1]
    return borders, logit_cancel_mask


DevicesSpecification = (
    torch.device | str | Sequence[torch.device | str] | Literal["auto"]
)


def infer_devices(devices: DevicesSpecification) -> tuple[torch.device, ...]:
    """Selects the appropriate PyTorch devices for inference.

    If `device` is "auto" then the devices are selected as follows:
    1. If CUDA is available and not excluded, returns all available "cuda" devices
    2. Otherwise, if MPS is available and not excluded, returns the "mps" device
    3. Otherwise, returns the "cpu" device

    CUDA and MPS can be excluded from the "auto" selection by specifying the
    TABPFN_EXCLUDE_DEVICES environment variable. This can be either "cuda", "mps", or
    "cuda,mps". This is useful for excluding device classes in CI.

    Args:
        devices: The device specification. One of:
            - "auto": the device will be selected as described above
            - a PyTorch device string like "cuda", "mps", or "cpu": this single device
                will be selected by parsing the string to a torch.device
            - a torch.device: this single device will be selected
            - a list of PyTorch device strings or torch.device: each item will be
                converted to a torch.device, and all of the devices selected

    Returns:
        A tuple of at least one device.
    """
    exclude_devices = {
        d.strip()
        for d in os.getenv("TABPFN_EXCLUDE_DEVICES", "").split(",")
        if d.strip()
    }

    if devices == "auto":
        if "cuda" not in exclude_devices and torch.cuda.is_available():
            return tuple(
                torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
            )

        if "mps" not in exclude_devices and torch.backends.mps.is_available():
            if _is_torch_mps_supported():
                return (torch.device("mps"),)
            warnings.warn(
                "An MPS device is available, but TabPFN disables MPS for "
                "PyTorch < 2.6 (earlier versions can give poor accuracy and "
                "lack bfloat16 autocast support on the MPS backend). Falling "
                "back to CPU. Install torch>=2.6 to use MPS.",
                stacklevel=2,
            )

        return (torch.device("cpu"),)

    if isinstance(devices, (str, torch.device)):
        devices = (devices,)

    devices = tuple(_parse_device(device) for device in devices)

    if len(set(devices)) != len(devices):
        raise ValueError(
            "The list of devices for inference cannot contain the same device more "
            f"than once. It contained: {devices}"
        )

    if any(d.type == "mps" for d in devices):
        if not torch.backends.mps.is_available():
            raise ValueError(
                "The MPS device was selected, but MPS is not available on this system."
            )
        if not _is_torch_mps_supported():
            raise ValueError(
                "The MPS device was selected, "
                "but TabPFN requires PyTorch >= 2.6 for MPS. "
                "Upgrade PyTorch, or set device='cpu' instead."
            )

    return devices


def _parse_device(device: str | torch.device) -> torch.device:
    # This is safe because torch.device(torch.device(...)) is a noop.
    # torch.device(device) returns a fairly informative error message if `device` is not
    # a valid device, thus do no extra error reporting.
    device = torch.device(device)

    # In older versions of PyTorch, some torch.cuda functions fail if the device has no
    # index. 0 is implicit if no index is specified, so add it.
    if device == torch.device("cuda"):
        device = torch.device("cuda:0")

    return device


def _is_torch_mps_supported() -> bool:
    """Return True if the MPS device is supported, otherwise False.

    We require PyTorch >= 2.6 for MPS to support all used operations.
    """
    return torch.__version__ >= "2.6"


def is_autocast_available(device_type: str) -> bool:
    """Infer whether autocast is available for the given device type.

    Args:
        device_type: The device type to check for autocast availability.

    Returns:
        Whether autocast is available for the given device type.
    """
    # Try to use PyTorch's built-in function first
    try:
        # Check if the function is available in torch
        if hasattr(torch.amp.autocast_mode, "is_autocast_available"):
            # Use function directly
            torch_is_autocast_available = torch.amp.autocast_mode.is_autocast_available
            return bool(torch_is_autocast_available(device_type))
        # Fall back to custom implementation
        raise AttributeError("is_autocast_available not found")
    except (ImportError, AttributeError):
        # Fall back to custom implementation if the function isn't available
        return bool(
            hasattr(torch.cuda, "amp")
            and hasattr(torch.cuda.amp, "autocast")
            and (
                device_type == torch.device("cuda").type
                or (
                    device_type == torch.device("cpu").type
                    and hasattr(torch.cpu, "amp")
                )
            ),
        )


def _cpu_supports_fast_bf16() -> bool:
    """Whether the CPU accelerates bfloat16 (Intel AMX / AVX512-BF16, AMD Zen 4+).

    Requires a torch build with oneDNN, which provides the fast bf16 kernels
    (absent e.g. on macOS wheels, where CPU bf16 falls back to slow reference
    kernels). AMX CPUs also enumerate AVX512-BF16, so this one check covers
    both instruction sets.
    """
    # bf16 without oneDNN's fast kernels is far slower than float32. Official
    # wheels always ship oneDNN; this guards distro/self-built torch without it.
    if not torch.backends.mkldnn.is_available():
        return False
    # Private torch API with no public equivalent; if a torch release removes
    # it, warn and stay on float32 rather than risk slow emulated bf16.
    avx512_bf16 = getattr(torch.cpu, "_is_avx512_bf16_supported", None)
    if avx512_bf16 is None:
        warnings.warn(
            "torch.cpu._is_avx512_bf16_supported() does not exist in this torch"
            " version, so TabPFN cannot detect CPU bf16 support and disables"
            " CPU bf16 autocast. Please report this at"
            " https://github.com/PriorLabs/TabPFN/issues so detection can be"
            " updated.",
            stacklevel=2,
        )
        return False
    return bool(avx512_bf16())


def infer_autocast_inference_mode(
    devices: Sequence[torch.device], *, enable: bool | None
) -> bool:
    """Infer whether reduced-precision (autocast) inference should be enabled.

    On GPU this is fp16 autocast; on CPU ``torch.autocast`` runs in bfloat16,
    which is enabled only on CPUs with native bf16 support (see
    :func:`_cpu_supports_fast_bf16`).

    Args:
        devices: The devices to validate against.
        enable:
            Whether it should be enabled, `True` or `False`, otherwise if `None`,
            detect if it's possible and use it if so.

    Returns:
        Whether to use autocast inference or not.

    Raises:
        ValueError: If autocast was enabled and any of the selected devices do
            not support it.
    """
    is_cpu = any(device.type.lower() == "cpu" for device in devices)
    if is_cpu:
        # CPU autocast runs in bfloat16, which is only faster than float32 on CPUs
        # with native bf16 support.
        autocast_available = (
            all(device.type.lower() == "cpu" for device in devices)
            and is_autocast_available("cpu")
            and _cpu_supports_fast_bf16()
        )
    else:
        autocast_available = any(
            is_autocast_available(device.type) for device in devices
        )

    if enable is None:
        return autocast_available

    if enable is True:
        if not autocast_available:
            raise ValueError(
                'You specified `inference_precision="autocast"`, however one or'
                f" more of the selected devices ({devices=}) does not support it."
                " On CPU, autocast requires hardware-accelerated bfloat16"
                " (Intel AMX / AVX512-BF16, AMD Zen 4+)."
                '\nSet `inference_precision="auto"` to fall back to full'
                " precision automatically.",
            )
        return True

    if enable is False:
        return False

    raise ValueError(f"Unrecognized argument '{enable}'")


def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer the random state from the given input.

    Args:
        random_state: The random state to infer.

    Returns:
        A static integer seed and a random number generator.
    """
    if isinstance(random_state, (int, np.integer)):
        np_rng = np.random.default_rng(random_state)
        static_seed = int(random_state)
    elif isinstance(random_state, np.random.RandomState):
        static_seed = int(random_state.randint(0, MAXINT_RANDOM_SEED))
        np_rng = np.random.default_rng(static_seed)
    elif isinstance(random_state, np.random.Generator):
        np_rng = random_state
        static_seed = int(np_rng.integers(0, MAXINT_RANDOM_SEED))
    elif random_state is None:
        np_rng = np.random.default_rng()
        static_seed = int(np_rng.integers(0, MAXINT_RANDOM_SEED))
    else:
        raise ValueError(f"Invalid random_state {random_state}")

    return static_seed, np_rng


def _halfnormal_tail_survival(
    distance_from_inner_border: torch.Tensor,
    outer_bucket_width: torch.Tensor,
) -> torch.Tensor:
    """Fraction of an outer bucket's half-normal tail past `distance_from_inner_border`.

    1.0 at the inner border, 0.5 at one bucket width out, as in
    `FullSupportBarDistribution`. Uses the complementary error function rather
    than `1 - cdf`, which cancels to exactly 0 a few sigma out.
    """
    # Repaired borders can leave a degenerate outer bucket, whose zero scale
    # would give 0/0. Flooring the width makes it a point mass instead.
    width = outer_bucket_width.clamp_min(torch.finfo(outer_bucket_width.dtype).tiny)
    sigma = FullSupportBarDistribution.halfnormal_with_p_weight_before(width).scale
    z = distance_from_inner_border / (sigma * math.sqrt(2.0))
    # `erfc(z)` via the scaled `erfcx`, which agrees to 5e-14 relative: ROCm
    # has no float64 `erfc` kernel and raises HIP error 209 on an MI250X.
    return torch.special.erfcx(z) * torch.exp(-z * z)


class _RemapWeights(NamedTuple):
    """The weights of one grid pair, see `_remap_weights`."""

    source: torch.Tensor
    destination: torch.Tensor
    pairs_per_destination: torch.Tensor
    weight: torch.Tensor
    lower_tail: torch.Tensor
    upper_tail: torch.Tensor


def _remap_weights(
    frm: torch.Tensor, to: torch.Tensor, *, dtype: torch.dtype
) -> _RemapWeights:
    """Weights that translate bucket masses from the `frm` grid to the `to` grid.

    Destination bucket `j` gets `weight[k] * probs[source[k]]` for each pair `k`
    with `destination[k] == j`. It also gets `lower_tail[j] * probs[0]` and
    `upper_tail[j] * probs[-1]`. The pairs are sorted by destination, and
    `pairs_per_destination[j]` is the number of pairs of bucket `j`.

    An interior source bucket is a uniform bar. Its weight for a destination
    bucket is the share of the source bucket that the two buckets overlap. The
    two outer source buckets are half-normal tails, as in
    `FullSupportBarDistribution`. A tail covers its bucket and the space beyond
    the grid. The first and last destination buckets take all the mass beyond
    `to`.

    Every weight is positive. The weights depend on the grids alone, so they
    are computed once, in float64 on the CPU.
    """
    num_buckets_frm = frm.shape[0] - 1
    if num_buckets_frm < 3:
        raise ValueError("`frm` needs an interior bucket between its two tails.")
    frm = frm.detach().to("cpu").to(torch.float64)
    to = to.detach().to("cpu").to(torch.float64)
    num_buckets_to = to.shape[0] - 1

    # Cut the axis at every border of both grids. Each piece lies in one source
    # bucket and one destination bucket. The tails below handle the pieces in
    # the two outer source buckets.
    edges = torch.unique(torch.cat([frm[1:-1], to]))
    mids = (edges[:-1] + edges[1:]) / 2
    source = torch.searchsorted(frm, mids) - 1
    destination = (torch.searchsorted(to, mids) - 1).clamp(0, num_buckets_to - 1)
    interior = (source >= 1) & (source <= num_buckets_frm - 2)
    source, destination = source[interior], destination[interior]
    weight = (edges[1:] - edges[:-1])[interior] / (frm[source + 1] - frm[source])

    lower = to[:-1].clone()
    upper = to[1:].clone()
    lower[0] = -math.inf
    upper[-1] = math.inf
    inner_low, inner_high = frm[1], frm[-2]
    width_low, width_high = frm[1] - frm[0], frm[-1] - frm[-2]

    def lower_survival(y: torch.Tensor) -> torch.Tensor:
        return _halfnormal_tail_survival(
            inner_low - torch.minimum(y, inner_low), width_low
        )

    def upper_survival(y: torch.Tensor) -> torch.Tensor:
        return _halfnormal_tail_survival(
            torch.maximum(y, inner_high) - inner_high, width_high
        )

    lower_tail = lower_survival(upper) - lower_survival(lower)
    upper_tail = upper_survival(lower) - upper_survival(upper)
    return _RemapWeights(
        source,
        destination,
        torch.bincount(destination, minlength=num_buckets_to),
        weight.to(dtype),
        lower_tail.clamp_min(0.0).to(dtype),
        upper_tail.clamp_min(0.0).to(dtype),
    )


def _grid_key(borders: torch.Tensor) -> bytes:
    # Move to the CPU first, then cast. A combined `.to(device, dtype)` casts on
    # MPS, which has no float64 and returns zeros instead of an error.
    return borders.detach().to("cpu").to(torch.float64).numpy().tobytes()


# A fitted regressor uses the same grids at every `predict`. Building the
# weights takes about 1 ms on the CPU, and the GPU waits for it. 32 entries
# take about 7 MB.
@functools.lru_cache(maxsize=32)
def _cached_remap_weights(
    frm: bytes, to: bytes, dtype: torch.dtype, device: torch.device
) -> _RemapWeights:
    weights = _remap_weights(
        torch.tensor(np.frombuffer(frm, dtype=np.float64)),
        torch.tensor(np.frombuffer(to, dtype=np.float64)),
        dtype=dtype,
    )
    return _RemapWeights(*(t.to(device) for t in weights))


def _apply_remap_weights(logits: torch.Tensor, weights: _RemapWeights) -> torch.Tensor:
    """Translate `(rows, num_buckets_frm)` logits, see `_remap_weights`."""
    probs = torch.softmax(logits, dim=-1)
    out = probs[:, :1] * weights.lower_tail + probs[:, -1:] * weights.upper_tail
    if probs.device.type == "cuda":
        # On CUDA, `index_add_` sums with atomics, so the order changes between
        # runs. A segmented sum gives the same bits every run at the same cost.
        contributions = probs.T.index_select(0, weights.source).mul_(
            weights.weight[:, None]
        )
        out += torch.segment_reduce(
            contributions, "sum", lengths=weights.pairs_per_destination, axis=0
        ).T
    else:
        out.index_add_(
            1,
            weights.destination,
            probs.index_select(1, weights.source).mul_(weights.weight),
        )
    return out


# `_apply_remap_weights` allocates one `(rows, pairs)` tensor in the caller's
# dtype, and `pairs` is at most `len(frm) + len(to)`. Chunks of
# `chunk_size * pairs <= _TRANSLATE_CHUNK_BUDGET_ELEMENTS` keep it near 40 MB in
# float32, whatever `n_test` is.
_TRANSLATE_CHUNK_BUDGET_ELEMENTS = 10_000_000


def translate_probs_across_borders(
    logits: torch.Tensor,
    *,
    frm: torch.Tensor,
    to: torch.Tensor,
    chunk_budget_elements: int = _TRANSLATE_CHUNK_BUDGET_ELEMENTS,
) -> torch.Tensor:
    """Translate the probabilities from the `frm` grid to the `to` grid.

    A destination bucket's mass is a sum of positive terms: its overlap with
    each source bucket, times that bucket's probability, plus a slice of each
    half-normal tail. Positive terms do not cancel, so no bucket is lost,
    however small it is. The tails fill the destination buckets outside `frm`,
    as `FullSupportBarDistribution.forward` reads them. The result has the
    dtype of `logits`.

    Large batches run in chunks, so the `(rows, pairs)` tensor inside
    `_apply_remap_weights` stays small. All batch dimensions are flattened
    first, so the cap holds whichever dimension is large. Chunking does not
    change the result.

    Args:
        logits: The logits of the distributions to translate. The last
            dimension indexes the buckets of `frm`. Every other dimension is a
            batch dimension. `TabPFNRegressor.predict` passes
            `(num_rows, num_buckets)`.
        frm: The borders to translate from.
        to: The borders to translate to.
        chunk_budget_elements: The largest `(rows, pairs)` tensor one chunk may
            allocate. Lower it to trade time for memory, mostly in tests.

    Returns:
        The translated probabilities.
    """
    if frm.shape == to.shape and frm.dtype == to.dtype and torch.equal(frm, to):
        # Every destination bucket is a source bucket, and reading the outer
        # ones as tails only moves mass around inside them, so the softmax is
        # already the answer. Half of the regressor's default ensemble lands
        # here, because every other member leaves its target untransformed.
        return torch.softmax(logits, dim=-1)

    weights = _cached_remap_weights(
        _grid_key(frm), _grid_key(to), logits.dtype, logits.device
    )
    batch_shape = logits.shape[:-1]
    num_buckets_to = to.shape[0] - 1

    # Flatten batch dims so chunking is independent of which dim is large.
    logits_flat = logits.reshape(-1, logits.shape[-1])
    num_rows = logits_flat.shape[0]
    chunk_size = max(1, chunk_budget_elements // max(weights.source.shape[0], 1))
    if num_rows <= chunk_size:
        out_flat = _apply_remap_weights(logits_flat, weights)
    else:
        # Preallocate output and write chunks in-place to avoid the transient
        # `torch.cat` would create (which would double peak memory).
        out_flat = torch.empty(
            num_rows,
            num_buckets_to,
            dtype=logits.dtype,
            device=logits.device,
        )
        for i in range(0, num_rows, chunk_size):
            out_flat[i : i + chunk_size] = _apply_remap_weights(
                logits_flat[i : i + chunk_size], weights
            )
    return out_flat.reshape(*batch_shape, num_buckets_to)


def transform_borders_one(
    borders: np.ndarray,
    target_transform: TransformerMixin | Pipeline,
    *,
    repair_nan_borders_after_transform: bool,
) -> tuple[npt.NDArray[np.bool_] | None, bool, np.ndarray]:
    """Transforms the borders used for the bar distribution for regression.

    Args:
        borders: The borders to transform.
        target_transform: The target transformer to use.
        repair_nan_borders_after_transform:
            Whether to repair any borders that are NaN after the transformation.

    Returns:
        logit_cancel_mask:
            The mask of the logit values to ignore,
            those that mapped to NaN borders.
        descending_borders: Whether the borders are descending after transformation
        borders_t: The transformed borders themselves.
    """
    borders_t = target_transform.inverse_transform(borders.reshape(-1, 1)).squeeze()  # type: ignore

    logit_cancel_mask: npt.NDArray[np.bool_] | None = None
    if repair_nan_borders_after_transform:
        broken_mask = (
            ~np.isfinite(borders_t)
            | (borders_t > REGRESSION_NAN_BORDER_LIMIT_UPPER)
            | (borders_t < REGRESSION_NAN_BORDER_LIMIT_LOWER)
        )
        if broken_mask.any():
            borders_t, logit_cancel_mask = _cancel_nan_borders(
                borders=borders_t,
                broken_mask=broken_mask,
            )

    _repair_borders(borders_t, inplace=True)

    reversed_order = np.arange(len(borders_t) - 1, -1, -1)
    descending_borders = (np.argsort(borders_t) == reversed_order).all()
    if descending_borders:
        borders_t = borders_t[::-1]
        logit_cancel_mask = (
            logit_cancel_mask[::-1] if logit_cancel_mask is not None else None
        )

    return logit_cancel_mask, descending_borders, borders_t


def pad_tensors(
    tensor_list: list[torch.Tensor],
    padding_val: float | None = 0,
    *,
    labels: bool = False,
) -> list[torch.Tensor]:
    """Pad tensors to maximum dims at the last dimensions.
    if labels=False, 2d tensors are expected, if labels=True, one 1d
    vectors are expected as inputs.

    Args:
        tensor_list: List of tensors to be padded.
        padding_val: what value to use for padding.
        labels: If true, the tensor list should contain 1D
            tensors that are padded only along this dimension.
            If false, rows and feature dimensions are padded.
    """
    max_size_clms = max(item.size(-1) for item in tensor_list)
    if not labels:
        max_size_rows = max(item.size(-2) for item in tensor_list)
    ret_list = []
    for item in tensor_list:
        pad_seqence = [0, max_size_clms - item.size(-1)]
        if not labels:
            pad_seqence.extend([0, max_size_rows - item.size(-2)])
        padded_item = torch.nn.functional.pad(
            item, pad_seqence, mode="constant", value=padding_val
        )
        ret_list.append(padded_item)
    return ret_list


def balance_probas_by_class_counts(
    probas: torch.Tensor,
    class_counts: np.ndarray,
) -> torch.Tensor:
    """Balance probabilities by class counts.

    Args:
        probas: The probabilities to balance.
        class_counts: The class counts to use for balancing.

    Returns:
        The balanced probabilities.
    """
    class_prob_in_train = class_counts / class_counts.sum()
    balanced_probas = probas / torch.from_numpy(class_prob_in_train).float().to(
        probas.device
    )
    return balanced_probas / balanced_probas.sum(dim=-1, keepdim=True)


def convert_batch_of_cat_ix_to_schema(
    batch_of_cat_indices: list[list[list[int]]],
    num_features: int,
) -> list[list[FeatureSchema]]:
    """Convert a batch of categorical indices to a schema."""
    feature_schema = []
    for ibatch in batch_of_cat_indices:
        feature_schema.append([])
        for cat_indices in ibatch:
            features = [
                Feature(
                    name=f"c{i}",
                    modality=FeatureModality.CATEGORICAL
                    if i in cat_indices
                    else FeatureModality.NUMERICAL,
                )
                for i in range(num_features)
            ]
            feature_schema[-1].append(FeatureSchema(features=features))

    return feature_schema
