"""A collection of random utilities for the TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import numpy.typing as npt
import torch
from sklearn.base import (
    TransformerMixin,
)

from tabpfn.architectures.encoders import (
    MulticlassClassificationTargetEncoderStep,
    TorchPreprocessingPipeline,
)
from tabpfn.constants import (
    REGRESSION_NAN_BORDER_LIMIT_LOWER,
    REGRESSION_NAN_BORDER_LIMIT_UPPER,
)
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline

    from tabpfn.architectures.interface import Architecture

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)


def get_autocast_context(
    device: torch.device, *, enabled: bool
) -> contextlib.AbstractContextManager:
    """Returns a torch.autocast context manager, disabling it for MPS devices.

    Args:
        device: The torch device being used.
        enabled: Whether to enable autocast.

    Returns:
        A context manager for autocasting.
    """
    if device.type == "mps":
        return contextlib.nullcontext()
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
        borders[-1] = borders[-1] * 2

    if borders[-1] - borders[-2] < 1e-6:
        borders[-1] = borders[-1] * 1.1

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


DevicesSpecification = Union[
    torch.device, str, Sequence[Union[torch.device, str]], Literal["auto"]
]


def infer_devices(devices: DevicesSpecification) -> tuple[torch.device, ...]:
    """Selects the appropriate PyTorch devices for inference.

    If `device` is "auto" then the devices are selected as follows:
    1. If CUDA is available and not excluded, returns the first "cuda" device
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
            return (torch.device("cuda:0"),)

        if _is_mps_supported() and "mps" not in exclude_devices:
            return (torch.device("mps"),)

        return (torch.device("cpu"),)

    if isinstance(devices, (str, torch.device)):
        devices = (devices,)

    devices = tuple(_parse_device(device) for device in devices)

    if len(set(devices)) != len(devices):
        raise ValueError(
            "The list of devices for inference cannot contain the same device more "
            f"than once. It contained: {devices}"
        )

    if not _is_mps_supported() and any(d.type == "mps" for d in devices):
        raise ValueError(
            "The MPS device was selected, "
            "but this is not supported by TabPFN before PyTorch 2.5. "
            'Set `device="cpu"` instead.'
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


def _is_mps_supported() -> bool:
    """Return True if the MPS device is supported, otherwise False.

    We have found that using MPS can lead to poor accuracy on PyTorch <2.5. See
    https://github.com/PriorLabs/TabPFN/pull/619
    """
    return torch.__version__ >= "2.5" and torch.backends.mps.is_available()


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


def infer_fp16_inference_mode(
    devices: Sequence[torch.device], *, enable: bool | None
) -> bool:
    """Infer whether fp16 inference should be enabled.

    Args:
        devices: The devices to validate against.
        enable:
            Whether it should be enabled, `True` or `False`, otherwise if `None`,
            detect if it's possible and use it if so.

    Returns:
        Whether to use fp16 inference or not.

    Raises:
        ValueError: If fp16 inference was enabled and any of the selected devices do
            not support it.
    """
    is_cpu = any(device.type.lower() == "cpu" for device in devices)
    fp16_available = (
        not is_cpu  # CPU can show enabled, yet it kills inference speed
        and any(is_autocast_available(device.type) for device in devices)
    )

    if enable is None:
        return fp16_available

    if enable is True:
        if not fp16_available:
            raise ValueError(
                "You specified `fp16_inference=True`, however"
                "`torch.amp.autocast_mode.is_autocast_available()`"
                f" reported that one or more of the selected devices ({devices=})"
                " does not support it."
                "\nPlease ensure your version of torch and device type"
                " are compatible with torch.autocast()`"
                " or set `fp16_inference=False`.",
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


def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix


# TODO (eddiebergman): Can probably put this back to the Bar distribution.
# However we don't really need the full BarDistribution class and this was
# put here to make that a bit more obvious in terms of what was going on.
def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    ys = ys.repeat((*logits.shape[:-1], 1))
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)

    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)

    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = (ys - borders[y_buckets]) / bucket_widths[y_buckets]
    share_of_bucket_left = share_of_bucket_left.clamp(0.0, 1.0)

    prob_in_bucket = probs.gather(index=y_buckets, dim=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return prob_left_of_ys.clip(0.0, 1.0)


def translate_probs_across_borders(
    logits: torch.Tensor,
    *,
    frm: torch.Tensor,
    to: torch.Tensor,
) -> torch.Tensor:
    """Translate the probabilities across the borders.

    Args:
        logits: The logits defining the distribution to translate.
        frm: The borders to translate from.
        to: The borders to translate to.

    Returns:
        The translated probabilities.
    """
    prob_left = _cdf(logits, borders=frm, ys=to)
    prob_left[..., 0] = 0.0
    prob_left[..., -1] = 1.0

    return (prob_left[..., 1:] - prob_left[..., :-1]).clamp_min(0.0)


def remove_non_differentiable_preprocessing_from_models(
    models: list[Architecture],
) -> None:
    """Remove non-differentiable encoder steps from the model.

    Args:
        models: The models to update.
    """
    for model in models:
        diffable_steps = []  # only differentiable encoder steps.
        for module in model.y_encoder:
            if isinstance(module, MulticlassClassificationTargetEncoderStep):
                pass
            else:
                diffable_steps.append(module)

        model.y_encoder = TorchPreprocessingPipeline(
            steps=diffable_steps, output_key="output"
        )


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
    max_size_clms = max([item.size(-1) for item in tensor_list])
    if not labels:
        max_size_rows = max([item.size(-2) for item in tensor_list])
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
